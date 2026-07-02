package engine.search;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.jupiter.api.Test;

import engine.board.Move;
import engine.board.Position;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Correctness and thread-isolation checks for the Lazy SMP search manager. */
class LazySmpSearchTest {

    @Test
    void singleWorkerFindsALegalMove() {
        LazySmpSearch smp = new LazySmpSearch(1, 4);
        int move = smp.think(Position.startpos(), SearchLimits.depth(4));
        assertTrue(move != 0, "expected a best move");
    }

    @Test
    void multipleWorkersAgreeOnAnObviousTacticalCapture() {
        // Same position as SearchTest.grabsHangingQueen: helper threads must not corrupt
        // the master's result via the shared transposition table.
        Position pos = Position.fromFen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1");
        LazySmpSearch smp = new LazySmpSearch(4, 4);
        int move = smp.think(pos, SearchLimits.depth(5));
        assertEquals("e4d5", Move.toUci(move), "pawn should capture the hanging queen even with helper threads racing the TT");
    }

    @Test
    void onlyTheMasterThreadPrintsSearchInfo() throws Exception {
        LazySmpSearch smp = new LazySmpSearch(4, 4);

        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.out;
        System.setOut(new PrintStream(captured));
        try {
            smp.think(Position.startpos(), SearchLimits.depth(6));
        } finally {
            System.out.flush();
            System.setOut(original);
        }

        // think() blocks until every worker (master + helpers) has returned, so nothing
        // more can be printed after this point -- it's safe to inspect the full capture.
        String[] lines = captured.toString().split("\\R");
        int lastDepth = 0;
        int infoLines = 0;
        for (String line : lines) {
            if (line.isBlank()) continue;
            assertTrue(line.startsWith("info depth "), "unexpected output line, a helper thread may have printed: " + line);
            infoLines++;
            int depth = Integer.parseInt(line.substring("info depth ".length()).split(" ")[0]);
            // If a helper thread's own iterations were interleaved into stdout, depth
            // would repeat or go backwards instead of forming one strictly increasing run.
            assertTrue(depth > lastDepth, "depth did not strictly increase, suggesting interleaved worker output: " + line);
            lastDepth = depth;
        }
        assertTrue(infoLines > 0, "expected at least one 'info depth' line from the master thread");
    }
}
