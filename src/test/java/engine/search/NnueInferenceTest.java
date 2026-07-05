package engine.search;

import org.junit.jupiter.api.Test;

import engine.board.Position;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Pins the exact quantized arithmetic (feature index -> accumulator -> clipped ReLU -> output
 * scale -> truncating division) against a hand-computed net, independent of the Python
 * reference. If the cross-check fixture ever regenerates wrong, this hand math still guards the
 * forward pass.
 */
class NnueInferenceTest {

    @Test
    void handComputedForwardPass() {
        // hidden = 1. Position: white king a1 (sq0), black king h8 (sq63), white to move.
        //   white king  (idx 5):  white-persp feat 5*64+0 = 320 ; black-persp ((0^1)*6+5)*64+56 = 760
        //   black king  (idx 11): white-persp feat 11*64+63 = 767; black-persp ((1^1)*6+5)*64+7  = 327
        // ftBias[0]=0.  accW = ftW[320]+ftW[767] = 10+3 = 13 ; accB = ftW[760]+ftW[327] = 5+7 = 12
        // stm=white -> stm half = accW = 13, nstm half = accB = 12.
        //   sum = crelu(13)*outW[0] + crelu(12)*outW[1] + outBias = 13*2 + 12*4 + 100 = 174
        //   cp  = trunc(174 * 400 / (255*64)) = trunc(69600 / 16320) = trunc(4.264..) = 4
        short[] ftw = new short[NnueEvaluator.INPUTS * 1];
        ftw[320] = 10;
        ftw[767] = 3;
        ftw[760] = 5;
        ftw[327] = 7;
        short[] ftb = {0};
        short[] outw = {2, 4}; // [STM half, NSTM half]
        NnueEvaluator.Network net = new NnueEvaluator.Network(1, 255, 64, 400, ftw, ftb, outw, 100);

        int cp = new NnueEvaluator(net).evaluate(
                Position.fromFen("7k/8/8/8/8/8/8/K7 w - - 0 1"));
        assertEquals(4, cp, "hand-computed forward pass mismatch");
    }

    @Test
    void negativeScoreTruncatesTowardZero() {
        // Same skeleton but a negative sum: force sum = -174 -> cp = trunc(-69600/16320)
        //   = trunc(-4.264..) = -4 (toward zero, NOT floor -5).
        short[] ftw = new short[NnueEvaluator.INPUTS * 1];
        ftw[320] = 10;
        ftw[767] = 3;
        ftw[760] = 5;
        ftw[327] = 7;
        short[] ftb = {0};
        short[] outw = {-2, -4};
        NnueEvaluator.Network net = new NnueEvaluator.Network(1, 255, 64, 400, ftw, ftb, outw, -100);

        int cp = new NnueEvaluator(net).evaluate(
                Position.fromFen("7k/8/8/8/8/8/8/K7 w - - 0 1"));
        assertEquals(-4, cp, "negative division must truncate toward zero");
    }
}
