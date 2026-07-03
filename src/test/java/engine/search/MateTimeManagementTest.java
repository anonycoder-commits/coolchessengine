package engine.search;

import engine.board.Position;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression coverage for the "instant resignation" bug: iterative deepening used to stop
 * on ANY mate score (|score| >= MATE_IN_MAX), including a mate AGAINST the engine. Because
 * the previous move's search leaves the transposition table full of high-depth mate-distance
 * entries, the next move's depth-1/2 iterations re-derive the losing score after a handful
 * of nodes (observed on lichess: 8-16 nodes at depth 2 on consecutive moves) and the engine
 * moved instantly instead of hunting for the longest resistance. The exit is now asymmetric:
 * only a mate FOR the engine short-circuits the clock.
 */
class MateTimeManagementTest {

    private static Search quietSearch() {
        Search s = new Search();
        s.printInfo = false;
        s.moveOverheadMs = 0;
        return s;
    }

    @Test
    void losingSideKeepsSearchingWithAWarmTt() {
        // Black to move with two legal king moves (no forced-single-reply bypass), and every
        // reply loses to Rh8#: the root score is a mate AGAINST the side to move from depth 2 on.
        Position pos = Position.fromFen("3k4/8/3K4/8/8/8/8/7R b - - 0 1");
        Search s = quietSearch();

        SearchLimits limits = new SearchLimits();
        limits.btime = 10_000;

        // First search fills the TT with mate-distance entries -- the exact state the engine
        // is in when the opponent's reply arrives and the next "go" lands on move N+1.
        s.think(pos, limits);
        assertTrue(s.bestScore <= -(Search.MATE - 100),
                "position must score as a mate against the side to move, got " + s.bestScore);

        // Second search: before the fix, the warm TT let depth 1-2 re-derive the mate score
        // in a few dozen nodes and the symmetric mate break committed a move instantly. The
        // losing side must instead keep deepening (longest resistance / mate refutation).
        s.think(pos, limits);
        assertTrue(s.bestScore <= -(Search.MATE - 100),
                "second search must still see the mate, got " + s.bestScore);
        assertTrue(s.nodes() > 500,
                "a losing search with a warm TT must not stop after a trivial iteration; "
                        + "searched only " + s.nodes() + " nodes");
        assertTrue(s.bestMove != 0, "must still commit a real move");
    }

    @Test
    void winningMateStillBanksTheClockImmediately() {
        // White mates in one; the clock allows ~2s soft. Finding the mate must still end the
        // search almost immediately -- the asymmetry only removes the LOSING-side exit.
        Position pos = Position.fromFen("7k/Q7/7K/8/8/8/8/8 w - - 0 1");
        Search s = quietSearch();

        SearchLimits limits = new SearchLimits();
        limits.wtime = 60_000;

        long start = System.nanoTime();
        s.think(pos, limits);
        long elapsedMs = (System.nanoTime() - start) / 1_000_000L;

        assertTrue(s.bestScore >= Search.MATE_IN_MAX,
                "expected a winning mate score, got " + s.bestScore);
        assertTrue(elapsedMs < 1_000,
                "a found winning mate must stop the search well before the soft budget "
                        + "(~2000ms here); took " + elapsedMs + "ms");
    }

    @Test
    void ttMateDistanceStaysExactAcrossWarmResearches() {
        // Mate in two for white (1.Kb6 Kb8 2.Rg8#), no mate in one: root score is exactly
        // MATE-3 (mate delivered at ply 3). Re-searching the same position on the same Search
        // instance reuses TT entries stored at various plies in earlier calls; if the
        // scoreToTt/scoreFromTt ply-relativization ever corrupted, the reported mate distance
        // would drift between these re-searches instead of staying pinned at MATE-3.
        Position pos = Position.fromFen("k7/6R1/8/2K5/8/8/8/8 w - - 0 1");
        Search s = quietSearch();
        for (int depth = 4; depth <= 10; depth++) {
            s.search(pos, depth);
            assertEquals(Search.MATE - 3, s.bestScore,
                    "mate distance drifted at depth " + depth
                            + " (TT mate-score ply relativization broken?)");
        }
    }

    @Test
    void shortestMateFloorKeepsSearchingPastTheFirstMateReport() {
        // With a warm TT holding a forced mate, the winning-mate break used to fire the instant
        // depth 1 re-derived the mate score (~tens of nodes), committing whatever mating move
        // the shallow read picked -- observed live as the reported mate distance LENGTHENING
        // (#3 -> #4 -> #5). The shortest-mate floor makes it keep deepening until d >= the mate
        // distance, so it resolves and plays the FASTEST mate -- markedly more search. This is
        // deterministic (not wall-clock): the mate resolves in far fewer nodes than any time
        // limit, so the node counts reflect the floor's search DEPTH, not the clock.
        String fen = "k7/6R1/8/2K5/8/8/8/8 w - - 0 1"; // legal KR vs K, MATE-3 (mate in 2)

        long withFloor = warmThenClockNodes(fen, true);
        long withoutFloor = warmThenClockNodes(fen, false);

        assertTrue(withFloor > withoutFloor,
                "the shortest-mate floor must search deeper than the old depth-1 winning-mate "
                        + "break: withFloor=" + withFloor + " nodes vs withoutFloor="
                        + withoutFloor + " nodes");
    }

    /** Warms the TT with a deep fixed search, then runs a clock search and returns its node
     *  count; asserts the search still finds the exact shortest mate either way. */
    private static long warmThenClockNodes(String fen, boolean shortestMate) {
        Search s = quietSearch();
        s.useShortestMate = shortestMate;
        Position pos = Position.fromFen(fen);
        s.search(pos, 12); // warm the TT so a depth-1 clock search would otherwise break instantly
        SearchLimits limits = new SearchLimits();
        limits.wtime = 60_000;
        s.think(pos, limits);
        assertEquals(Search.MATE - 3, s.bestScore, "must still report the exact shortest mate");
        return s.nodes();
    }

    @Test
    void winningBigWithoutAMateSpendsFullBudgetHuntingForOne() {
        // White is up ~+14 but the forced mate is far too deep to see in the ~2s budget here,
        // so the position sits in the mate-hunt band (score >= MATE_HUNT_SCORE_CP, not yet a
        // mate) for the whole search. With useMateHunt the decisive-score acceleration (bank
        // the clock after 2 stable iterations) is suppressed, so the engine keeps searching to
        // the soft limit instead of exiting at the optimum-time target -- meaning distinctly
        // more nodes than with the acceleration left on. This is a WALL-CLOCK comparison, but a
        // robust one: the node RATIO is STRUCTURAL (bank stops at optimum = 0.6*soft, hunt at
        // ~1.0*soft, so ~1.6x), independent of machine speed because both scale with the same
        // soft budget and nps. The one load-sensitive edge is the bank config failing to reach
        // its 2-stable-iteration banking depth before the optimum target when the machine is
        // heavily loaded (Gradle runs test classes in parallel) -- a generous clock removes it
        // by giving ample absolute time to reach that depth well before optimum.
        String fen = "6k1/5p1p/6p1/8/8/1Q6/5PPP/6K1 w - - 0 1";

        long huntNodes = nodesUnderClock(fen, true);
        long bankNodes = nodesUnderClock(fen, false);

        assertTrue(huntNodes > bankNodes * 6 / 5,
                "mate-hunt should keep searching a decisively-won-but-not-mate position rather "
                        + "than banking the clock after 2 stable iterations: hunt=" + huntNodes
                        + " nodes vs bank=" + bankNodes + " nodes (expected hunt > 1.2x bank)");
    }

    /** Runs {@code fen} twice on one Search (warm TT, mimicking consecutive game moves) under a
     *  real clock, returning the node count of the second (measured) search. */
    private static long nodesUnderClock(String fen, boolean mateHunt) {
        Search s = quietSearch();
        s.useMateHunt = mateHunt;
        // Isolate the mate-hunt gate: since the mop-up eval term landed, a decisively-won
        // position's score CLIMBS across iterations (deeper king-driving lines keep raising
        // it), which trips the volatility gate's score-swing signal and suppresses the
        // stability exit in BOTH configurations -- masking the very mechanism under test.
        s.useVolatilityGate = false;
        SearchLimits limits = new SearchLimits();
        limits.wtime = 120_000; // soft ~4000ms, optimum ~2400ms -> ample time to reach the
                                // banking depth before optimum even under parallel-suite load
        Position pos = Position.fromFen(fen);
        s.think(pos, limits); // warm the TT
        s.think(pos, limits); // measured
        // Confirm the band was actually active (score decisive but not a mate); otherwise the
        // comparison would be meaningless (e.g. if a mate were found, both would break early).
        assertTrue(s.bestScore >= 800 && s.bestScore < Search.MATE_IN_MAX,
                "test position must stay in the winning-but-not-mate band, was " + s.bestScore);
        return s.nodes();
    }
}
