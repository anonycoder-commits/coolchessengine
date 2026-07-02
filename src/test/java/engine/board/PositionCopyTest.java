package engine.board;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;

/** Verifies the deep-copy constructor used to give each Lazy SMP worker its own board. */
class PositionCopyTest {

    private static final String KIWIPETE =
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    @Test
    void copyMatchesOriginalState() {
        Position original = Position.fromFen(KIWIPETE);
        Position copy = new Position(original);

        assertEquals(original.zobristKey(), copy.zobristKey());
        assertEquals(original.sideToMove(), copy.sideToMove());
        assertEquals(original.castlingRights(), copy.castlingRights());
        assertEquals(original.epSquare(), copy.epSquare());
        assertEquals(original.halfmoveClock(), copy.halfmoveClock());
        assertEquals(original.fullmoveNumber(), copy.fullmoveNumber());
        for (int sq = 0; sq < 64; sq++) {
            assertEquals(original.pieceAt(sq), copy.pieceAt(sq), "square " + sq);
        }
        assertEquals(original.occupied(), copy.occupied());
    }

    @Test
    void copyIsIndependentOfOriginal() {
        Position original = Position.fromFen(KIWIPETE);
        Position copy = new Position(original);
        assertNotSame(original, copy);

        MoveList moves = new MoveList();
        MoveGenerator.generateLegal(copy, moves);
        int move = moves.moves[0];
        long originalKeyBefore = original.zobristKey();

        // Mutating the copy (make + unmake, as a search thread would) must never be
        // visible on the original -- this is the whole point of giving each SMP worker
        // its own board instead of sharing one.
        copy.makeMove(move);
        assertEquals(originalKeyBefore, original.zobristKey(), "original must be untouched by copy's makeMove");
        assertNotEquals(originalKeyBefore, copy.zobristKey(), "copy's key should change after makeMove");

        copy.unmakeMove(move);
        assertEquals(originalKeyBefore, copy.zobristKey(), "copy should restore after unmakeMove");
        assertEquals(originalKeyBefore, original.zobristKey(), "original must still be untouched");
    }
}
