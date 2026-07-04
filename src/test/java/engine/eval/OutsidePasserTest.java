package engine.eval;

import org.junit.jupiter.api.Test;

import engine.board.Position;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Targeted correctness checks for the outside-passer / flank-majority terms (see
 * Evaluator.OUTSIDE_PASSER_EG, PAWN_MAJORITY_EG) added after a loss where the engine misjudged
 * a queenside majority that became a winning outside a-pawn (HCE showed +0.35 for the side that
 * was actually lost). Each test restores the static toggles afterward since they are mutable
 * shared state across the test JVM.
 */
class OutsidePasserTest {

    // White has a lone outside a-pawn (enemy king stuck on the kingside); otherwise symmetric
    // pawn structure. The term should meaningfully favor White here.
    private static final String OUTSIDE_PASSER_FEN = "6k1/6pp/8/8/8/P7/6PP/6K1 w - - 0 1";
    // Mirror per EvalSymmetryTest's convention: swap piece colors, flip ranks vertically, KEEP
    // the same side-to-move letter (that's what makes the side-to-move-relative scores negate).
    private static final String OUTSIDE_PASSER_MIRROR = "6k1/6pp/p7/8/8/8/6PP/6K1 w - - 0 1";

    // White has a 2-vs-1 queenside pawn majority while the black king sits on the kingside.
    private static final String MAJORITY_FEN = "r5k1/1p3ppp/8/8/8/1P6/P4PPP/R5K1 w - - 0 1";
    // Mirror of MAJORITY_FEN, same convention as OUTSIDE_PASSER_MIRROR above.
    private static final String MAJORITY_MIRROR = "r5k1/p4ppp/1p6/8/8/8/1P3PPP/R5K1 w - - 0 1";

    @Test
    void outsidePasserFavorsTheSideWithTheDistantPasser() {
        assertTrue(Evaluator.useOutsidePasser, "useOutsidePasser should be the shipped default");
        int score = Evaluator.evaluate(Position.fromFen(OUTSIDE_PASSER_FEN));
        assertTrue(score > 100, "outside a-pawn position should score clearly in White's favor: " + score);
    }

    @Test
    void outsidePasserTermIsColorSymmetric() {
        boolean saved = Evaluator.useOutsidePasser;
        try {
            Evaluator.useOutsidePasser = true;
            int e = Evaluator.evaluate(Position.fromFen(OUTSIDE_PASSER_FEN)) - Evaluator.TEMPO;
            int em = Evaluator.evaluate(Position.fromFen(OUTSIDE_PASSER_MIRROR)) - Evaluator.TEMPO;
            assertEquals(e, -em, "outside-passer term must be exactly antisymmetric under color swap");
        } finally {
            Evaluator.useOutsidePasser = saved;
        }
    }

    @Test
    void outsidePasserTermIsZeroWhenDisabled() {
        boolean saved = Evaluator.useOutsidePasser;
        try {
            Evaluator.useOutsidePasser = false;
            int off = Evaluator.evaluate(Position.fromFen(OUTSIDE_PASSER_FEN));
            Evaluator.useOutsidePasser = true;
            int on = Evaluator.evaluate(Position.fromFen(OUTSIDE_PASSER_FEN));
            assertTrue(on > off, "enabling the term must raise the score for the side with the distant passer");
        } finally {
            Evaluator.useOutsidePasser = saved;
        }
    }

    @Test
    void pawnMajorityTermFavorsTheOutsideMajoritySide() {
        boolean saved = Evaluator.usePawnMajority;
        try {
            Evaluator.usePawnMajority = false;
            int off = Evaluator.evaluate(Position.fromFen(MAJORITY_FEN));
            Evaluator.usePawnMajority = true;
            int on = Evaluator.evaluate(Position.fromFen(MAJORITY_FEN));
            assertTrue(on > off, "enabling the majority term must raise the score for the side with the outside majority");
        } finally {
            Evaluator.usePawnMajority = saved;
        }
    }

    @Test
    void pawnMajorityTermIsColorSymmetric() {
        boolean saved = Evaluator.usePawnMajority;
        try {
            Evaluator.usePawnMajority = true;
            int e = Evaluator.evaluate(Position.fromFen(MAJORITY_FEN)) - Evaluator.TEMPO;
            int em = Evaluator.evaluate(Position.fromFen(MAJORITY_MIRROR)) - Evaluator.TEMPO;
            assertEquals(e, -em, "pawn-majority term must be exactly antisymmetric under color swap");
        } finally {
            Evaluator.usePawnMajority = saved;
        }
    }
}
