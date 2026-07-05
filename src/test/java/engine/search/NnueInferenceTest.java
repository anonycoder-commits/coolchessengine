package engine.search;

import org.junit.jupiter.api.Test;

import engine.board.Position;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Pins the exact quantized arithmetic (feature index -> accumulator -> SCReLU -> staged output
 * scale with truncating divisions) against a hand-computed net, independent of the Python
 * reference. Matches bullet's {@code Network::evaluate} staging.
 */
class NnueInferenceTest {

    @Test
    void handComputedForwardPass() {
        // hidden = 1. Position: white king a1 (sq0), black king h8 (sq63), white to move.
        //   white king  (idx 5):  white-persp feat 5*64+0 = 320 ; black-persp ((0^1)*6+5)*64+56 = 760
        //   black king  (idx 11): white-persp feat 11*64+63 = 767; black-persp ((1^1)*6+5)*64+7  = 327
        // ftBias[0]=0. accW = ftW[320]+ftW[767] = 100+50 = 150 ; accB = ftW[760]+ftW[327] = 40+20 = 60
        // stm=white -> stm half = accW = 150, nstm half = accB = 60.
        //   out  = screlu(150)*outW[0] + screlu(60)*outW[1] = 150^2*3 + 60^2*5 = 67500 + 18000 = 85500
        //   out /= QA(255)          -> 335   (335*255=85425 <= 85500)
        //   out += outBias(1000)    -> 1335
        //   out *= SCALE(400)       -> 534000
        //   out /= QA*QB(16320)     -> 32    (32*16320=522240 <= 534000)
        short[] ftw = new short[NnueEvaluator.INPUTS * 1];
        ftw[320] = 100;
        ftw[767] = 50;
        ftw[760] = 40;
        ftw[327] = 20;
        short[] ftb = {0};
        short[] outw = {3, 5}; // [STM half, NSTM half]
        NnueEvaluator.Network net = new NnueEvaluator.Network(1, 255, 64, 400, ftw, ftb, outw, 1000);

        int cp = new NnueEvaluator(net).evaluate(
                Position.fromFen("7k/8/8/8/8/8/8/K7 w - - 0 1"));
        assertEquals(32, cp, "hand-computed SCReLU forward pass mismatch");
    }

    @Test
    void negativeScoreTruncatesTowardZero() {
        // Same accumulators, negated output weights/bias: out = -85500 -> ... -> trunc(-534000/16320)
        //   = trunc(-32.7) = -32 (toward zero, NOT floor -33). The /QA step: trunc(-85500/255) = -335.
        short[] ftw = new short[NnueEvaluator.INPUTS * 1];
        ftw[320] = 100;
        ftw[767] = 50;
        ftw[760] = 40;
        ftw[327] = 20;
        short[] ftb = {0};
        short[] outw = {-3, -5};
        NnueEvaluator.Network net = new NnueEvaluator.Network(1, 255, 64, 400, ftw, ftb, outw, -1000);

        int cp = new NnueEvaluator(net).evaluate(
                Position.fromFen("7k/8/8/8/8/8/8/K7 w - - 0 1"));
        assertEquals(-32, cp, "negative division must truncate toward zero at each stage");
    }
}
