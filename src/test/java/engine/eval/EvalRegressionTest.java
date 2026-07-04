package engine.eval;

import engine.board.Position;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Pins the evaluator's exact output on a fixed set of positions. The term helpers were
 * refactored from allocating {@code int[2]} returns to packed-long (mg,eg) pairs, which must
 * be score-neutral: these expected values were captured from the pre-refactor evaluator, so
 * any packing/unpacking sign or shift mistake (or a future accidental behavior change hiding
 * inside a "mechanical" refactor) fails here immediately with the offending FEN.
 */
class EvalRegressionTest {

    private static final String[] FENS = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4",
        "2r3k1/pp3ppp/2n1b3/q7/3P4/2P1B3/P1Q2PPP/3R2K1 w - - 0 1",
        "8/5pk1/6p1/8/8/1P6/P4PPP/6K1 b - - 0 1",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1",
        "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1",
        "7k/Q7/7K/8/8/8/8/8 w - - 0 1",
    };

    // Same order as FENS. Re-pinned after the threats + safe-check terms were added, then again
    // after the mating-technique drive term (only the last FEN, KQ-vs-lone-K, changed: 1077 ->
    // 1167, the +90 bonus for driving the cornered king). Prior baselines:
    //   packed-long refactor era: {15, 108, -13, 151, 100, 16, -239, -310, 15, -964, 1077}
    //   threats + safe-checks era: {15, 88, -13, 83, 130, 16, -218, -310, 15, -911, 1077}
    //   mop-up era:                {15, 88, -13, 83, 130, 16, -218, -310, 15, -911, 1167}
    //   with rook-behind-passer ON (useRookBehindPasser, default off after its referee gate
    //   came back neutral-negative): positions 4/7 become 88 and -208.
    //   useOutsidePasser went default ON (2026-07-04, gated at 51.2%/600 games, kept for fixing
    //   a targeted loss -- see Evaluator.OUTSIDE_PASSER_EG): positions 7/8 (indices 6/7) change
    //   from -218/-310 to -214/-320 -- both have a passer >= OUTSIDE_KING_DIST files from the
    //   enemy king. All other positions are unaffected by this flip.
    //   usePawnMajority ALSO went default ON (2026-07-04, gated at 51.7%/600 games vs the
    //   outside-passer-on baseline): positions 7/8 (indices 6/7) shift further to -210/-344.
    //   Re-verify both changed FENs fresh (not from memory) after touching either toggle --
    //   an earlier stale-Gradle-cache false pass on this exact test taught that lesson.
    private static final int[] EXPECTED = {
        15, 96, -7, 83, 130, 16, -210, -344, 15, -911, 1167,
    };

    @Test
    void evaluationMatchesThePinnedBaseline() {
        for (int i = 0; i < FENS.length; i++) {
            assertEquals(EXPECTED[i], Evaluator.evaluate(Position.fromFen(FENS[i])),
                    "score changed for: " + FENS[i]);
        }
    }

    @Test
    void pawnCacheIsScoreNeutral() {
        // The per-thread pawn cache must be a pure speed win: a cached evaluation (second
        // call = guaranteed hit) must be bit-identical to the uncached static path.
        engine.search.HandcraftedEvaluator cached = new engine.search.HandcraftedEvaluator();
        for (String fen : FENS) {
            Position pos = Position.fromFen(fen);
            int uncached = Evaluator.evaluate(pos);
            assertEquals(uncached, cached.evaluate(pos), "cold-cache eval differs for: " + fen);
            assertEquals(uncached, cached.evaluate(pos), "warm-cache eval differs for: " + fen);
        }
    }
}
