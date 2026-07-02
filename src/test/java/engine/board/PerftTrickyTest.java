package engine.board;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The TalkChess/Martin Sedlak "tricky" perft suite -- positions chosen to exercise the
 * classic legal-move-generation traps (en-passant pins and discovered checks, castling
 * interactions, promotion-that-gives-check, self-stalemate, double check) that the
 * standard 5-position suite in {@link PerftTest} doesn't cover as directly. Added as
 * extra hardening alongside the Attacks magic/PEXT migration (Phase 1): these depths are
 * deep enough that any subtle sliding-attack table bug affecting check/pin detection
 * would surface here even though it might not move the standard suite's totals.
 */
class PerftTrickyTest {

    private static void check(String fen, int depth, long expected) {
        Position pos = Position.fromFen(fen);
        long actual = Perft.perft(pos, depth);
        assertEquals(expected, actual, "perft(" + depth + ") for FEN: " + fen);
    }

    @Test
    void epCapturePinnedAgainstHorizontalRookCheckIsIllegal() {
        // Capturing en passant would remove both the capturing pawn and its victim from
        // the same rank, exposing the king to Ra5-style horizontal check.
        check("3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 6, 1134888L);
    }

    @Test
    void epCaptureResolvesADiagonalPin() {
        check("8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", 6, 1015133L);
    }

    @Test
    void epCaptureGivesADiscoveredCheck() {
        check("8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 6, 1440467L);
    }

    @Test
    void castlingWhileLeavingTheRookGivingCheck() {
        check("5k2/8/8/8/8/8/8/4K2R w K - 0 1", 6, 661072L);
    }

    @Test
    void castlingQueensideWithOpenFiles() {
        check("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 6, 803711L);
    }

    @Test
    void castlingRightsLostViaRookCapture() {
        check("r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 4, 1274206L);
    }

    @Test
    void castlingPreventedByAttackOnTransitSquare() {
        check("r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1", 4, 1720476L);
    }

    @Test
    void promotionDeliversCheck() {
        check("2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 6, 3821001L);
    }

    @Test
    void underpromotionVariety() {
        check("8/P1k5/K7/8/8/8/8/8 w - - 0 1", 6, 92683L);
    }

    @Test
    void selfStalemateTrap() {
        check("K1k5/8/P7/8/8/8/8/8 w - - 0 1", 6, 2217L);
    }

    @Test
    void stalemateAndCheckmateMix() {
        check("8/k1P5/8/1K6/8/8/8/8 w - - 0 1", 7, 567584L);
    }

    @Test
    void doubleCheckOnlyKingMoveIsLegal() {
        check("8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", 4, 23527L);
    }
}
