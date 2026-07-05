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
 * Architecture: {@code (768 -> N)x2 -> 1} perspective net, single hidden layer, clipped
 * ReLU, integer-quantized end-to-end. Every {@link #evaluate} does a FULL accumulator
 * refresh from the {@link Position} (no incremental accumulator, no Position coupling) --
 * slow but trivially correct. Phase 3 adds the efficiently-updatable accumulator.
 *
 * <h2>Feature indexing (must match bullet's {@code Chess768})</h2>
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
 * <h2>Quantization</h2>
 * ft weights/bias scaled x{@code QA} (=255); output weights x{@code QB} (=64); output bias
 * x{@code QA*QB}. Activation is clipped ReLU {@code clamp(acc, 0, QA)}. Final centipawns =
 * {@code (sum + outBias) * SCALE / (QA*QB)}, truncated toward zero, clamped to stay well
 * inside the {@link Evaluator} mate-safety band.
 *
 * <p><b>Correctness note.</b> The {@code ^56} flip, colour-swap and concat order above are the
 * canonical bullet convention but are only <i>proven</i> against a real trained net by the
 * Java-vs-reference-Python cross-check (see {@code tools/nnue/nnue_ref.py}). Internally,
 * {@code NnueSymmetryTest} pins that this class is self-consistent (mirror invariance) for any
 * weights, which catches an index/flip/order bug without a trained net.
 */
public final class NnueEvaluator implements Evaluator {

    /** Magic 'C','N','U','E' + version, guarding against a truncated/foreign weights file. */
    static final int MAGIC = ('C' << 24) | ('N' << 16) | ('U' << 8) | 'E';
    static final int VERSION = 1;
    static final int INPUTS = 768;

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
     * Reads the little-endian weights format written by {@code tools/nnue/nnue_ref.py}
     * (and, later, the bullet export). Layout after the 28-byte header:
     * ftWeights[INPUTS*hidden], ftBias[hidden], outWeights[2*hidden], outBias -- all i16
     * except the i32 header fields.
     */
    public static Network load(InputStream in) throws IOException {
        byte[] all = in.readAllBytes();
        ByteBuffer buf = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN);
        int magic = buf.getInt();
        if (magic != MAGIC) {
            throw new IOException("NNUE: bad magic 0x" + Integer.toHexString(magic));
        }
        int version = buf.getInt();
        if (version != VERSION) {
            throw new IOException("NNUE: unsupported version " + version);
        }
        int inputs = buf.getInt();
        if (inputs != INPUTS) {
            throw new IOException("NNUE: expected " + INPUTS + " inputs, got " + inputs);
        }
        int hidden = buf.getInt();
        int qa = buf.getInt();
        int qb = buf.getInt();
        int scale = buf.getInt();
        if (hidden <= 0 || qa <= 0 || qb <= 0 || scale <= 0) {
            throw new IOException("NNUE: nonsensical header " + hidden + "/" + qa + "/" + qb + "/" + scale);
        }

        short[] ftWeights = readShorts(buf, inputs * hidden);
        short[] ftBias = readShorts(buf, hidden);
        short[] outWeights = readShorts(buf, 2 * hidden);
        int outBias = buf.getShort();
        return new Network(hidden, qa, qb, scale, ftWeights, ftBias, outWeights, outBias);
    }

    private static short[] readShorts(ByteBuffer buf, int n) throws IOException {
        if (buf.remaining() < n * 2) {
            throw new IOException("NNUE: file truncated, wanted " + n + " shorts, "
                    + buf.remaining() + " bytes left");
        }
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
            int whitePieceIdx = p;                    // friendly=white when viewed from white
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

        // Concatenate STM half first, apply clipped ReLU, dot with the output weights.
        int[] stm = pos.sideToMove() == Piece.WHITE ? w : b;
        int[] nstm = pos.sideToMove() == Piece.WHITE ? b : w;
        final short[] outW = net.outWeights;
        final int qa = net.qa;
        long sum = 0;
        for (int i = 0; i < hidden; i++) {
            sum += (long) crelu(stm[i], qa) * outW[i];
        }
        for (int i = 0; i < hidden; i++) {
            sum += (long) crelu(nstm[i], qa) * outW[hidden + i];
        }
        sum += net.outBias;

        long cp = sum * net.scale / ((long) qa * net.qb); // integer, truncates toward zero
        // Honour the Evaluator contract: never collide with mate scores.
        int bound = Search.MATE_IN_MAX - 1;
        if (cp > bound) cp = bound;
        else if (cp < -bound) cp = -bound;
        return (int) cp;
    }

    private static int crelu(int x, int qa) {
        if (x < 0) return 0;
        if (x > qa) return qa;
        return x;
    }
}
