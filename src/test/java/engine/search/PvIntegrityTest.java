package engine.search;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Position;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.fail;

/**
 * Every move in a reported PV must be legal when the line is replayed from the root.
 *
 * Regression coverage for a live illegal-PV bug: negamax's stopped/draw/max-ply early
 * returns used to exit BEFORE resetting pvLength[ply], so a repetition-draw child left its
 * slot holding a stale line from an unrelated subtree, and a PV-node parent accepting the
 * draw score spliced those stale moves into the reported PV (observed on lichess as
 * "pv ... f4c1 b2a3" with b2a3 impossible, crashing the wrapper's PV parser).
 */
class PvIntegrityTest {

    private static final String[] FENS = {
        // The position from the live incident (perpetual-check / repetition-heavy).
        "1qr1r1k1/4bpp1/1p2p2p/pQp1P3/3P4/P3P2P/1PR2PB1/2R3K1 b - - 1 27",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    };

    @Test
    void reportedPvIsAlwaysReplayableFromTheRoot() {
        for (String fen : FENS) {
            // One Search per FEN, reused across depths: the warm TT and history tables make
            // deeper iterations traverse repetition/draw paths a cold search may not reach.
            Search s = new Search();
            s.printInfo = false;
            for (int depth = 8; depth <= 16; depth += 4) {
                s.search(Position.fromFen(fen), depth);
                assertPvLegal(fen, s.pvString());
            }
        }
    }

    private static void assertPvLegal(String fen, String pv) {
        if (pv.equals("0000")) return; // no PV recorded: nothing to validate
        Position pos = Position.fromFen(fen);
        String[] moves = pv.split("\\s+");
        for (int i = 0; i < moves.length; i++) {
            MoveList legal = new MoveList();
            MoveGenerator.generateLegal(pos, legal);
            int found = 0;
            for (int j = 0; j < legal.size; j++) {
                if (Move.toUci(legal.moves[j]).equals(moves[i])) { found = legal.moves[j]; break; }
            }
            if (found == 0) {
                fail("illegal PV move '" + moves[i] + "' at index " + i + " in pv \"" + pv
                        + "\" from " + fen);
            }
            pos.makeMove(found);
        }
    }
}
