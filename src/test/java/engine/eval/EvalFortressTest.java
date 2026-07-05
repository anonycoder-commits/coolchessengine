package engine.eval;

import engine.board.Position;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Pins the scaleFactor() fortress case: pawnless R + one minor vs R (KRBKR / KRNKR) is a
 *  theoretical draw the HCE would otherwise score as ~+a whole minor, which led it to simplify
 *  a real advantage straight into the drawn ending. The eg is scaled hard toward zero. */
class EvalFortressTest {

    // Pawnless KRB vs KR: White up the bishop, but a textbook fortress draw.
    private static final String KRB_v_KR = "4k3/8/8/8/8/3B4/4KR2/6r1 w - - 0 1";
    // Same, vertical + colour mirror (Black up the bishop) -- must be the exact negation.
    private static final String KRB_v_KR_MIRROR = "6R1/4kr2/3b4/8/8/8/8/4K3 b - - 0 1";
    // Pawnless KRN vs KR: the same fortress with a knight instead of a bishop.
    private static final String KRN_v_KR = "4k3/8/8/8/8/3N4/4KR2/6r1 w - - 0 1";
    // KRB vs K (no black rook): a trivial win -- must NOT be scaled.
    private static final String KRB_v_K = "4k3/8/8/8/8/3B4/4KR2/8 w - - 0 1";
    // KRB + a pawn vs KR: with a pawn it is winning, not the pawnless fortress -- must NOT scale.
    private static final String KRBP_v_KR = "4k3/8/8/8/4P3/3B4/4KR2/6r1 w - - 0 1";

    private static int eval(String fen) {
        return Evaluator.evaluate(Position.fromFen(fen));
    }

    @Test
    void fortressIsScaledTowardDraw() {
        // Unscaled this would read ~+a whole minor (~300+); the fortress scaling pulls it well
        // below that so the engine won't prefer simplifying INTO it over a real advantage.
        assertTrue(eval(KRB_v_KR) < 200, "KRBvKR should be scaled drawish, got " + eval(KRB_v_KR));
        assertTrue(eval(KRN_v_KR) < 200, "KRNvKR should be scaled drawish, got " + eval(KRN_v_KR));
    }

    @Test
    void fortressScalingIsSymmetric() {
        // TEMPO is side-to-move-relative, so strip it before comparing the mirror (see EvalSymmetryTest).
        int a = eval(KRB_v_KR) - Evaluator.TEMPO;
        int b = eval(KRB_v_KR_MIRROR) - Evaluator.TEMPO;
        assertEquals(a, b, "fortress scaling must be color-symmetric");
    }

    @Test
    void realWinsAreNotScaled() {
        // KRB vs a lone king is trivially winning (the rook alone mates) -- full material value.
        assertTrue(eval(KRB_v_K) > 800, "KRBvK is a real win, must not be scaled: " + eval(KRB_v_K));
        // Adding a pawn takes it out of the pawnless fortress case; it is winning, not drawn.
        assertTrue(eval(KRBP_v_KR) > 300, "KRB+P vs KR is winning, must not scale: " + eval(KRBP_v_KR));
    }
}
