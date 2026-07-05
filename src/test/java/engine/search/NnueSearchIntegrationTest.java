package engine.search;

import java.io.InputStream;

import org.junit.jupiter.api.Test;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Position;
import engine.search.SearchLimits;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Proves the Phase-2 wiring: {@link NnueEvaluator} is a drop-in {@link Evaluator} that a real
 * search (single-threaded and Lazy SMP via the per-worker factory) drives to a LEGAL move. The
 * fixture net's scores are meaningless, so this checks plumbing/legality, not move quality --
 * move quality is what the self-play gate measures once a real net exists.
 */
class NnueSearchIntegrationTest {

    private static NnueEvaluator.Network fixtureNet() throws Exception {
        try (InputStream in = NnueSearchIntegrationTest.class.getResourceAsStream("/nnue/fixture_net.bin")) {
            assertNotNull(in, "fixture_net.bin missing");
            return NnueEvaluator.load(in);
        }
    }

    private static void assertLegal(int move, Position pos, String where) {
        assertTrue(move != 0, where + ": search returned no move");
        MoveList legal = new MoveList();
        MoveGenerator.generateLegal(pos, legal);
        boolean found = false;
        for (int i = 0; i < legal.size; i++) {
            if (legal.moves[i] == move) { found = true; break; }
        }
        assertTrue(found, where + ": " + Move.toUci(move) + " is not legal");
    }

    @Test
    void singleThreadedSearchWithNnuePlaysLegalMove() throws Exception {
        NnueEvaluator.Network net = fixtureNet();
        Position pos = Position.startpos();
        Search search = new Search(new NnueEvaluator(net), 16);
        SearchLimits limits = new SearchLimits();
        limits.depth = 5;
        int best = search.think(new Position(pos), limits);
        assertLegal(best, pos, "single-threaded NNUE");
    }

    @Test
    void smpSearchWithNnueFactoryPlaysLegalMove() throws Exception {
        NnueEvaluator.Network net = fixtureNet();
        Position pos = Position.startpos();
        LazySmpSearch smp = new LazySmpSearch(2, 16);
        smp.setEvaluatorFactory(() -> new NnueEvaluator(net));
        SearchLimits limits = new SearchLimits();
        limits.depth = 5;
        int best = smp.think(new Position(pos), limits);
        assertLegal(best, pos, "SMP NNUE");
    }
}
