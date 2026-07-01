package engine.search;

import org.junit.jupiter.api.Test;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Position;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Walking-skeleton search integration checks (board + move-gen + negamax). */
class SearchTest {

    private static boolean isLegal(Position pos, int move) {
        MoveList moves = new MoveList();
        MoveGenerator.generateLegal(pos, moves);
        for (int i = 0; i < moves.size; i++) {
            if (moves.moves[i] == move) return true;
        }
        return false;
    }

    private static Search quietSearch() {
        Search s = new Search();
        s.printInfo = false;
        return s;
    }

    @Test
    void startposReturnsLegalMove() {
        Position pos = Position.startpos();
        Search s = quietSearch();
        s.search(pos, 4);
        assertTrue(s.bestMove != 0, "expected a best move");
        assertTrue(isLegal(pos, s.bestMove), "best move must be legal");
    }

    @Test
    void grabsHangingQueen() {
        Position pos = Position.fromFen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1");
        Search s = quietSearch();
        s.search(pos, 3);
        assertEquals("e4d5", Move.toUci(s.bestMove), "pawn should capture the queen");
    }

    @Test
    void findsMateInOne() {
        Position pos = Position.fromFen("7k/Q7/7K/8/8/8/8/8 w - - 0 1");
        Search s = quietSearch();
        s.search(pos, 2);
        assertEquals("a7g7", Move.toUci(s.bestMove), "should find the mating move");
        assertTrue(s.bestScore >= Search.MATE - 10, "mate score expected, got " + s.bestScore);
    }

    @Test
    void checkmatedPositionHasNoMove() {
        // White is checkmated (fool's mate).
        Position pos = Position.fromFen(
                "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
        Search s = quietSearch();
        s.search(pos, 3);
        assertEquals(0, s.bestMove, "no legal move when checkmated");
        assertEquals(-Search.MATE, s.bestScore, "checkmate should score -MATE at root");
    }
}
