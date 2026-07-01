package engine.board;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** Verifies make/unmake reversibility and Zobrist-hash consistency across a move tree. */
class MakeUnmakeTest {

    private static final String[] FENS = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
    };

    private void walk(Position pos, int depth) {
        long keyBefore = pos.zobristKey();
        int castlingBefore = pos.castlingRights();
        int epBefore = pos.epSquare();

        // The incremental hash must always equal a from-scratch recomputation.
        assertEquals(pos.computeKey(), pos.zobristKey(), "incremental key != recomputed key");

        if (depth == 0) return;

        MoveList moves = new MoveList();
        MoveGenerator.generateLegal(pos, moves);
        for (int i = 0; i < moves.size; i++) {
            int move = moves.moves[i];
            pos.makeMove(move);
            walk(pos, depth - 1);
            pos.unmakeMove(move);
            assertEquals(keyBefore, pos.zobristKey(), "key not restored after unmake: " + Move.toUci(move));
            assertEquals(castlingBefore, pos.castlingRights(), "castling not restored: " + Move.toUci(move));
            assertEquals(epBefore, pos.epSquare(), "ep not restored: " + Move.toUci(move));
        }
    }

    @Test
    void reversibleAndHashConsistent() {
        for (String fen : FENS) {
            Position pos = Position.fromFen(fen);
            walk(pos, 3);
        }
    }
}
