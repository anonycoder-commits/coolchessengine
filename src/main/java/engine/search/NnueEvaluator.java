package engine.search;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

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
 * {@code outWeights[2*hidden]} (STM half first), {@code outBias}. All {@code i16}. {@code hidden}
 * is inferred from the file length; QA/QB/SCALE are fixed to bullet's defaults. Requires
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

    /** Immutable, thread-shareable parsed network. One instance is shared across workers. */
    public static final class Network {
        final int hidden, qa, qb, scale;
        final short[] ftWeights; // [feature*hidden + i], feature-major (0..INPUTS-1)
        final short[] ftBias;    // [hidden]
        final short[] outWeights;// [2*hidden], STM half first
        final int outBias;       // scaled QA*QB

        Network(int hidden, int qa, int qb, int scale,
                short[] ftWeights, short[] ftBias, short[] outWeights, int outBias) {
            this.hidden = hidden;
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
    // Per-instance scratch (one NnueEvaluator per worker thread -> no sharing races).
    private final int[] accWhite;
    private final int[] accBlack;

    public NnueEvaluator(Network net) {
        this.net = net;
        this.accWhite = new int[net.hidden];
        this.accBlack = new int[net.hidden];
    }

    /**
     * Reads bullet's raw saved format (header-less, little-endian i16). {@code hidden} is
     * inferred from the byte count: total shorts = {@code 768*h + h + 2*h + 1 = 771*h + 1},
     * PLUS bullet pads the whole file to a 64-byte boundary with filler bytes (observed:
     * the literal ASCII "bullet" repeated) -- up to 31 trailing shorts (62 bytes) that are
     * neither a header nor real weights and must be tolerated, not rejected.
     */
    public static Network load(InputStream in) throws IOException {
        byte[] all = in.readAllBytes();
        if (all.length < 2 || (all.length & 1) != 0) {
            throw new IOException("NNUE: odd/empty file (" + all.length + " bytes)");
        }
        long shorts = all.length / 2L;
        long trailing = (shorts - 1) % (INPUTS + 3); // real data remainder, tolerating pad-to-64-byte filler
        if (trailing >= 32) {
            throw new IOException("NNUE: " + all.length + " bytes is not a valid (768->N)x2->1 net"
                    + " (unexplained remainder " + trailing + " shorts)");
        }
        int hidden = (int) ((shorts - 1) / (INPUTS + 3)); // 768 + 1 (ftBias) + 2 (outW) per hidden
        if (hidden <= 0 || (hidden % 32) != 0) {
            throw new IOException("NNUE: hidden=" + hidden + " must be a positive multiple of 32");
        }
        ByteBuffer buf = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN);
        short[] ftWeights = readShorts(buf, INPUTS * hidden);
        short[] ftBias = readShorts(buf, hidden);
        short[] outWeights = readShorts(buf, 2 * hidden);
        int outBias = buf.getShort();
        return new Network(hidden, QA, QB, SCALE, ftWeights, ftBias, outWeights, outBias);
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
        final int[] w = accWhite;
        final int[] b = accBlack;

        // Refresh both perspective accumulators from the bias, then add every piece's column.
        final short[] ftBias = net.ftBias;
        for (int i = 0; i < hidden; i++) {
            w[i] = ftBias[i];
            b[i] = ftBias[i];
        }

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
                for (int i = 0; i < hidden; i++) {
                    w[i] += ftW[wOff + i];
                    b[i] += ftW[bOff + i];
                }
            }
        }

        // Concatenate STM half first, SCReLU, dot with output weights -- bullet's exact staging.
        int[] stm = pos.sideToMove() == Piece.WHITE ? w : b;
        int[] nstm = pos.sideToMove() == Piece.WHITE ? b : w;
        final short[] outW = net.outWeights;
        final int qa = net.qa;
        long out = 0;
        for (int i = 0; i < hidden; i++) {
            out += (long) screlu(stm[i], qa) * outW[i];
        }
        for (int i = 0; i < hidden; i++) {
            out += (long) screlu(nstm[i], qa) * outW[hidden + i];
        }
        out /= qa;                       // QA*QA*QB -> QA*QB
        out += net.outBias;              // QA*QB
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
