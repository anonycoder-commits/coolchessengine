package engine.search;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorSpecies;

import engine.board.Piece;
import engine.board.Position;

/**
 * NNUE evaluation (roadmap Phase 1: correctness-first, NON-incremental).
 *
 * Architecture: {@code (768 -> N)x2 -> 1} perspective net, single hidden layer, SCReLU,
 * integer-quantized end-to-end. Every {@link #evaluate} does a FULL accumulator refresh from
 * the {@link Position} (no incremental accumulator, no Position coupling) -- slow but trivially
 * correct. Phase 3 adds the efficiently-updatable accumulator.
 *
 * <p>This mirrors bullet's stock {@code examples/simple.rs} exactly -- same feature set
 * ({@code Chess768}), activation (SCReLU), quantization (QA/QB/SCALE) and inference arithmetic
 * -- so a net trained by that example loads and evaluates identically. The engine reads bullet's
 * raw saved format directly (no conversion step).
 *
 * <h2>Feature indexing (bullet {@code Chess768})</h2>
 * 768 = 12 piece-indices x 64 squares. A piece with engine index {@code p} (0..11:
 * {@code W_PAWN..B_KING}, i.e. {@code color*6 + type}) on square {@code sq} (a1=0):
 * <ul>
 *   <li>white-perspective feature = {@code p*64 + sq};</li>
 *   <li>black-perspective feature = {@code ((color^1)*6 + type)*64 + (sq ^ 56)} -- colours
 *       swapped so black's pieces are "friendly", board vertically flipped ({@code ^56}, the
 *       same mirror the handcrafted PSTs use).</li>
 * </ul>
 * The output layer reads the concatenation <b>[side-to-move half, other-side half]</b>: the
 * first {@code hidden} output weights multiply the STM accumulator, the next {@code hidden}
 * the non-STM accumulator. This STM-first ordering is what encodes tempo.
 *
 * <h2>Quantization / inference (matches bullet {@code Network::evaluate})</h2>
 * ft weights/bias scaled x{@code QA} (=255); output weights x{@code QB} (=64); output bias
 * x{@code QA*QB}. Activation SCReLU {@code s = clamp(acc,0,QA)^2}. Then, in this exact staged
 * order (integer, truncating toward zero at each division):
 * <pre>
 *   out  = sum over both halves of SCReLU(acc) * outWeight   // scale QA*QA*QB
 *   out /= QA                                                // -> QA*QB
 *   out += outBias                                           // QA*QB
 *   out *= SCALE
 *   out /= QA*QB                                             // centipawns
 * </pre>
 * The result is clamped to stay well inside the {@link Evaluator} mate-safety band.
 *
 * <h2>File format (bullet raw {@code SavedFormat} dump, little-endian, no header)</h2>
 * {@code ftWeights[768*hidden]} (feature-major), {@code ftBias[hidden]},
 * {@code outWeights[buckets*2*hidden]} (bucket-major; within a bucket STM half first),
 * {@code outBias[buckets]}. All {@code i16}. {@code hidden} and the material-output-bucket count
 * (1 = original single-output net, 8 = bullet {@code MaterialCount<8>}) are inferred from the
 * file length; QA/QB/SCALE are fixed to bullet's defaults. Requires
 * {@code hidden} a multiple of 32 so bullet's {@code align(64)} accumulator layout is
 * padding-free (256 satisfies this). Bullet also pads the WHOLE file to a 64-byte boundary
 * with filler bytes after {@code outBias} (observed: the literal ASCII "bullet" repeated) --
 * up to 31 trailing shorts are tolerated as padding, not rejected as a parse error.
 *
 * <p><b>Correctness note.</b> The {@code ^56} flip, colour-swap and STM-first concat are no
 * longer just the assumed canonical convention -- they are verified line-by-line against
 * bullet's actual {@code Chess768::map_features} (in
 * {@code crates/bullet_lib/src/game/inputs/chess768.rs}: {@code stm = [0,384][c] + 64*type + sq},
 * {@code ntm = [384,0][c] + 64*type + (sq^56)}, where {@code c} is 1 for the non-mover after
 * {@code bulletformat}'s own mover-relative canonicalization) and against
 * {@code stm_hidden.concat(ntm_hidden)} in {@code examples/simple.rs}. Both reduce to exactly
 * this class's formulas for every (mover, piece-colour) combination. {@code NnueSymmetryTest}
 * additionally pins self-consistency (mirror invariance) for any weights, and
 * {@code NnueCrossCheckTest} pins Java against the independent Python reference.
 */
public final class NnueEvaluator implements Evaluator {

    static final int INPUTS = 768;
    // bullet simple.rs defaults; the raw file carries no header, so these are fixed here.
    static final int QA = 255;
    static final int QB = 64;
    static final int SCALE = 400;

    // SIMD width for the feature-transformer accumulation (hidden is a multiple of 32, and every
    // preferred short-species length -- 8/16/32 -- divides 32, so the loop bound is exact).
    private static final VectorSpecies<Short> SP = ShortVector.SPECIES_PREFERRED;

    // Output-bucket counts this loader understands. B=1 is the original single-output net;
    // B=8 is the material-bucketed net (bullet MaterialCount<8>). The file is header-less, so
    // the count is detected from the file size like `hidden` is.
    private static final int[] SUPPORTED_BUCKETS = {1, 8};

    /** Immutable, thread-shareable parsed network. One instance is shared across workers. */
    public static final class Network {
        final int hidden, buckets, qa, qb, scale;
        final short[] ftWeights; // [feature*hidden + i], feature-major (0..INPUTS-1)
        final short[] ftBias;    // [hidden]
        final short[] outWeights;// [buckets * 2*hidden], bucket-major; within a bucket STM half first
        final int[] outBias;     // [buckets], scaled QA*QB

        Network(int hidden, int buckets, int qa, int qb, int scale,
                short[] ftWeights, short[] ftBias, short[] outWeights, int[] outBias) {
            this.hidden = hidden;
            this.buckets = buckets;
            this.qa = qa;
            this.qb = qb;
            this.scale = scale;
            this.ftWeights = ftWeights;
            this.ftBias = ftBias;
            this.outWeights = outWeights;
            this.outBias = outBias;
        }
    }

    private final Network net;
    // Per-instance scratch (one NnueEvaluator per worker thread -> no sharing races). short[]
    // (matching bullet's own i16 accumulator; values provably stay in range) so the hot refresh
    // is a clean short[] += short[] loop the JIT auto-vectorizes, with no widening in the loop.
    private final short[] accWhite;
    private final short[] accBlack;

    public NnueEvaluator(Network net) {
        this.net = net;
        this.accWhite = new short[net.hidden];
        this.accBlack = new short[net.hidden];
    }

    /**
     * Reads bullet's raw saved format (header-less, little-endian i16). {@code hidden} and the
     * output-bucket count {@code B} are inferred from the byte count: total shorts =
     * {@code 768*h + h + B*2*h + B = h*(769+2B) + B} for some supported {@code B}, PLUS bullet
     * pads the whole file to a 64-byte boundary with filler bytes (observed: the literal ASCII
     * "bullet" repeated) -- up to 31 trailing shorts (62 bytes) that are neither a header nor
     * real weights and must be tolerated, not rejected. For B&gt;1 the output weights are saved
     * TRANSPOSED (bullet {@code SavedFormat.transpose()}), i.e. bucket-major with each bucket's
     * {@code 2*h} weights contiguous -- which is exactly the layout inference indexes into.
     */
    public static Network load(InputStream in) throws IOException {
        byte[] all = in.readAllBytes();
        if (all.length < 2 || (all.length & 1) != 0) {
            throw new IOException("NNUE: odd/empty file (" + all.length + " bytes)");
        }
        long shorts = all.length / 2L;
        int hidden = 0, buckets = 0;
        for (int b : SUPPORTED_BUCKETS) {
            long body = shorts - b;              // shorts minus the B outBias values
            long perHidden = INPUTS + 1 + 2L * b; // ftW + ftBias + outW shorts per hidden unit
            if (body <= 0) continue;
            long h = body / perHidden;
            long trailing = body % perHidden;    // tolerating pad-to-64-byte filler
            if (trailing < 32 && h > 0 && (h % 32) == 0) {
                if (hidden != 0) {
                    throw new IOException("NNUE: " + all.length + " bytes is ambiguous between "
                            + buckets + " and " + b + " output buckets -- refusing to guess");
                }
                hidden = (int) h;
                buckets = b;
            }
        }
        if (hidden == 0) {
            throw new IOException("NNUE: " + all.length + " bytes is not a valid (768->N)x2->B net"
                    + " for any supported bucket count (hidden must be a positive multiple of 32)");
        }
        ByteBuffer buf = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN);
        short[] ftWeights = readShorts(buf, INPUTS * hidden);
        short[] ftBias = readShorts(buf, hidden);
        short[] outWeights = readShorts(buf, buckets * 2 * hidden);
        int[] outBias = new int[buckets];
        for (int i = 0; i < buckets; i++) outBias[i] = buf.getShort();
        return new Network(hidden, buckets, QA, QB, SCALE, ftWeights, ftBias, outWeights, outBias);
    }

    private static short[] readShorts(ByteBuffer buf, int n) {
        short[] out = new short[n];
        for (int i = 0; i < n; i++) out[i] = buf.getShort();
        return out;
    }

    @Override
    public int evaluate(Position pos) {
        final int hidden = net.hidden;
        final short[] ftW = net.ftWeights;
        final short[] w = accWhite;
        final short[] b = accBlack;

        // Refresh both perspective accumulators from the bias, then add every piece's column.
        System.arraycopy(net.ftBias, 0, w, 0, hidden);
        System.arraycopy(net.ftBias, 0, b, 0, hidden);

        for (int p = 0; p < 12; p++) {
            long bb = pos.pieces(p);
            if (bb == 0) continue;
            int color = p / 6;
            int type = p % 6;
            int whitePieceIdx = p;                      // friendly=white when viewed from white
            int blackPieceIdx = (color ^ 1) * 6 + type; // colours swapped for the black view
            int wBase = whitePieceIdx * 64;
            int bBase = blackPieceIdx * 64;
            while (bb != 0) {
                int sq = Long.numberOfTrailingZeros(bb);
                bb &= bb - 1;
                int wOff = (wBase + sq) * hidden;
                int bOff = (bBase + (sq ^ 56)) * hidden;
                int bound = SP.loopBound(hidden);
                int i = 0;
                for (; i < bound; i += SP.length()) {
                    ShortVector.fromArray(SP, w, i)
                            .add(ShortVector.fromArray(SP, ftW, wOff + i)).intoArray(w, i);
                    ShortVector.fromArray(SP, b, i)
                            .add(ShortVector.fromArray(SP, ftW, bOff + i)).intoArray(b, i);
                }
                for (; i < hidden; i++) { // scalar tail (none when hidden % SP.length() == 0)
                    w[i] += ftW[wOff + i];
                    b[i] += ftW[bOff + i];
                }
            }
        }

        // Material output bucket (bullet MaterialCount<B>): (pieces-2) / ceil(32/B), so with B=8
        // each bucket spans 4 piece-counts. Clamped defensively for kingless test positions; any
        // legal position (2..32 pieces) lands in range on its own. B=1 always selects bucket 0.
        final int bucket;
        if (net.buckets == 1) {
            bucket = 0;
        } else {
            int divisor = (32 + net.buckets - 1) / net.buckets;
            int raw = (Long.bitCount(pos.occupied()) - 2) / divisor;
            bucket = Math.max(0, Math.min(net.buckets - 1, raw));
        }
        final int outOff = bucket * 2 * hidden;

        // Concatenate STM half first, SCReLU, dot with the bucket's output weights -- bullet's
        // exact staging.
        short[] stm = pos.sideToMove() == Piece.WHITE ? w : b;
        short[] nstm = pos.sideToMove() == Piece.WHITE ? b : w;
        final short[] outW = net.outWeights;
        final int qa = net.qa;
        long out = 0;
        for (int i = 0; i < hidden; i++) {
            out += (long) screlu(stm[i], qa) * outW[outOff + i];
        }
        for (int i = 0; i < hidden; i++) {
            out += (long) screlu(nstm[i], qa) * outW[outOff + hidden + i];
        }
        out /= qa;                       // QA*QA*QB -> QA*QB
        out += net.outBias[bucket];      // QA*QB
        out *= net.scale;
        out /= (long) qa * net.qb;       // -> centipawns (truncates toward zero)

        // Honour the Evaluator contract: never collide with mate scores.
        int bound = Search.MATE_IN_MAX - 1;
        if (out > bound) out = bound;
        else if (out < -bound) out = -bound;
        return (int) out;
    }

    /** Square Clipped ReLU: clamp to [0, QA] then square (0 .. QA*QA). */
    private static int screlu(int x, int qa) {
        int y = x < 0 ? 0 : (x > qa ? qa : x);
        return y * y;
    }
}
