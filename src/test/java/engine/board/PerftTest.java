package engine.board;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Standard 5-position perft suite validated against published node counts.
 * This is the project's move-generation correctness gate.
 */
class PerftTest {

    private static void check(String fen, int depth, long expected) {
        Position pos = Position.fromFen(fen);
        long actual = Perft.perft(pos, depth);
        assertEquals(expected, actual, "perft(" + depth + ") for FEN: " + fen);
    }

    @Test
    void startposShallow() {
        String fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        check(fen, 1, 20L);
        check(fen, 2, 400L);
        check(fen, 3, 8902L);
        check(fen, 4, 197281L);
        check(fen, 5, 4865609L);
    }

    @Test
    void startposDepth6() {
        check("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324L);
    }

    @Test
    void kiwipete() {
        String fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        check(fen, 1, 48L);
        check(fen, 2, 2039L);
        check(fen, 3, 97862L);
        check(fen, 4, 4085603L);
        check(fen, 5, 193690690L);
    }

    @Test
    void position3() {
        String fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
        check(fen, 1, 14L);
        check(fen, 2, 191L);
        check(fen, 3, 2812L);
        check(fen, 4, 43238L);
        check(fen, 5, 674624L);
        check(fen, 6, 11030083L);
    }

    @Test
    void position4() {
        String fen = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
        check(fen, 1, 6L);
        check(fen, 2, 264L);
        check(fen, 3, 9467L);
        check(fen, 4, 422333L);
        check(fen, 5, 15833292L);
    }

    @Test
    void position5() {
        String fen = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1";
        check(fen, 1, 44L);
        check(fen, 2, 1486L);
        check(fen, 3, 62379L);
        check(fen, 4, 2103487L);
        check(fen, 5, 89941194L);
    }
}
