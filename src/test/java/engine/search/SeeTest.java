package engine.search;

import org.junit.jupiter.api.Test;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Position;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Static Exchange Evaluation correctness checks, independent of full search/alpha-beta. */
class SeeTest {

    private static int findMove(Position pos, String uci) {
        MoveList moves = new MoveList();
        MoveGenerator.generateLegal(pos, moves);
        for (int i = 0; i < moves.size; i++) {
            if (Move.toUci(moves.moves[i]).equals(uci)) return moves.moves[i];
        }
        throw new IllegalArgumentException("no legal move " + uci);
    }

    @Test
    void undefendedCaptureWinsFullVictimValue() {
        // White rook takes an undefended black rook: nothing recaptures on a8.
        Position pos = Position.fromFen("r3k3/8/8/8/8/8/8/R3K3 w - - 0 1");
        Search s = new Search();
        assertEquals(500, s.see(pos, findMove(pos, "a1a8")), "undefended rook capture should net a full rook");
    }

    @Test
    void capturingIntoAPawnDefenderLosesMaterial() {
        // Rxa6 captures a pawn, but b7xa6 recaptures: net loss of rook-for-pawn (100 - 500).
        Position pos = Position.fromFen("4k3/1p6/p7/8/8/8/8/R3K3 w - - 0 1");
        Search s = new Search();
        assertEquals(-400, s.see(pos, findMove(pos, "a1a6")), "rook capturing a pawn defended by a pawn should lose material");
    }

    @Test
    void winningExchangeStopsAtTheRightMaterialCount() {
        // dxe6 captures a knight; Rxe6 recaptures the pawn but nothing recaptures the rook.
        // Net for white: +knight (320) - pawn (100) = +220.
        Position pos = Position.fromFen("4rk2/8/4n3/3P4/8/8/8/4K3 w - - 0 1");
        Search s = new Search();
        assertEquals(220, s.see(pos, findMove(pos, "d5e6")), "winning a knight for a pawn should net +220");
    }

    @Test
    void quietMoveHasNoExchangeValue() {
        Position pos = Position.startpos();
        Search s = new Search();
        assertEquals(0, s.see(pos, findMove(pos, "e2e4")), "a non-capture has no captured piece to gain");
    }

    @Test
    void promotionCaptureIntoADefendedSquareLosesThePromotedPieceValue() {
        // bxa8=Q captures a rook (+500), but the bishop on c6 recaptures the new queen
        // (-900) once b7 is vacated: net for white = 500 - 900 = -400. A SEE that forgets
        // to upgrade the attacker from PAWN to QUEEN after promotion would instead see the
        // "recapture" as only threatening a pawn's worth of loss, stop simulating early,
        // and wrongly return +500.
        Position pos = Position.fromFen("r7/1P6/2b5/8/8/8/8/4K2k w - - 0 1");
        Search s = new Search();
        assertEquals(-400, s.see(pos, findMove(pos, "b7a8q")),
                "promotion-capture recaptured by a defender should account for the promoted piece's value");
    }
}
