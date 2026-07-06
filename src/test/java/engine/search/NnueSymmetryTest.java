package engine.search;

import java.util.Random;

import org.junit.jupiter.api.Test;

import engine.board.Position;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Reference-free correctness pin for the perspective net. Unlike the handcrafted eval (which is
 * <i>antisymmetric</i> under a colour-swap that keeps the side-to-move letter), an NNUE
 * perspective net is <b>invariant</b> under the FULL mirror -- vertical flip + colour swap +
 * side-to-move swap -- because that maps a position to the identical game state seen from the
 * mover's side, leaving both accumulator halves unchanged. So {@code eval(pos) == eval(mirror)}
 * exactly, for ANY weights.
 *
 * <p>This holds only if the feature indexing, the {@code ^56} flip of the non-STM perspective,
 * and the STM-first output concatenation are all correct -- so it catches those bugs with no
 * trained net or external reference. It must pass for a randomly weighted net.
 */
class NnueSymmetryTest {

    private static final String[] FENS = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4",
        "2r3k1/pp3ppp/2n1b3/q7/3P4/2P1B3/P1Q2PPP/3R2K1 w - - 0 1",
        "8/5pk1/6p1/8/8/1P6/P4PPP/6K1 b - - 0 1",
        "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1",
        "7k/Q7/7K/8/8/8/8/8 w - - 0 1",
    };

    @Test
    void mirrorInvariant() {
        NnueEvaluator eval = new NnueEvaluator(randomNet(20260705, 32, 1));
        for (String fen : FENS) {
            int e = eval.evaluate(Position.fromFen(fen));
            int em = eval.evaluate(Position.fromFen(mirror(fen)));
            assertEquals(e, em, "NNUE eval not mirror-invariant for: " + fen);
        }
    }

    /** Same invariance for a material-bucketed net: the mirror preserves the piece count, so
     *  both sides of the comparison select the same bucket -- catching a bucket-offset bug on
     *  top of the indexing bugs the single-output test pins. */
    @Test
    void mirrorInvariantWithOutputBuckets() {
        NnueEvaluator eval = new NnueEvaluator(randomNet(20260706, 32, 8));
        for (String fen : FENS) {
            int e = eval.evaluate(Position.fromFen(fen));
            int em = eval.evaluate(Position.fromFen(mirror(fen)));
            assertEquals(e, em, "bucketed NNUE eval not mirror-invariant for: " + fen);
        }
    }

    /** Full mirror: vertical flip, colour swap, AND side-to-move swap. */
    private static String mirror(String fen) {
        String[] parts = fen.trim().split("\\s+");
        String[] ranks = parts[0].split("/");
        StringBuilder board = new StringBuilder();
        for (int i = ranks.length - 1; i >= 0; i--) {
            if (i != ranks.length - 1) board.append('/');
            for (char c : ranks[i].toCharArray()) {
                board.append(Character.isLetter(c) ? swap(c) : c);
            }
        }
        String stm = parts[1].equals("w") ? "b" : "w"; // <- swapped, unlike the HCE test
        // Only placement + stm affect the 768-feature eval; castling/ep just keep the FEN valid.
        return board + " " + stm + " - - 0 1";
    }

    private static char swap(char c) {
        return Character.isUpperCase(c) ? Character.toLowerCase(c) : Character.toUpperCase(c);
    }

    /** Seeded net with values wide enough that clipped ReLU both clips and passes. */
    private static NnueEvaluator.Network randomNet(long seed, int hidden, int buckets) {
        Random rng = new Random(seed);
        short[] ftw = new short[NnueEvaluator.INPUTS * hidden];
        for (int i = 0; i < ftw.length; i++) ftw[i] = (short) (rng.nextInt(129) - 64);
        short[] ftb = new short[hidden];
        for (int i = 0; i < ftb.length; i++) ftb[i] = (short) (rng.nextInt(257) - 128);
        short[] outw = new short[buckets * 2 * hidden];
        for (int i = 0; i < outw.length; i++) outw[i] = (short) (rng.nextInt(129) - 64);
        int[] outb = new int[buckets];
        for (int i = 0; i < outb.length; i++) outb[i] = rng.nextInt(4001) - 2000;
        return new NnueEvaluator.Network(hidden, buckets, 255, 64, 400, ftw, ftb, outw, outb);
    }
}
