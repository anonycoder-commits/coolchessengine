package engine.search;

import engine.board.Piece;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression coverage for the blitz/bullet time-forfeit bug: a single exploding ply could
 * legally run for tens of seconds because hardLimitMs (the only bound checkTime() enforces
 * inside the recursive search) was derived purely from playableTime/4, with no relationship
 * to optimumTimeMs (the much smaller per-move target, only consulted between completed
 * iterations). Early in a game, with a nearly-full clock, that let hardLimitMs balloon to
 * 10x+ optimumTimeMs -- observed in practice as a move taking 64.2s against a 6.0s target.
 */
class TimeManagementTest {

    @Test
    void hardLimitStaysWithinABoundedMultipleOfOptimumEvenWithANearlyFullClock() {
        Search search = new Search();
        SearchLimits limits = new SearchLimits();
        limits.wtime = 300_000; // 5 minutes -- start-of-game blitz clock
        limits.winc = 2_000;

        search.computeTimeLimits(limits, Piece.WHITE);

        long allowedHard = (long) (search.optimumTimeMs * Search.HARD_LIMIT_OPTIMUM_MULTIPLIER);
        assertTrue(search.hardLimitMs <= allowedHard,
                "hardLimitMs=" + search.hardLimitMs + " must stay within "
                        + Search.HARD_LIMIT_OPTIMUM_MULTIPLIER + "x optimumTimeMs=" + search.optimumTimeMs
                        + " (allowed=" + allowedHard + "); otherwise a single exploding ply can run "
                        + "unchecked for the entire, much larger hard limit before the soft-bound "
                        + "stability check -- which only runs between completed iterations -- ever "
                        + "gets a chance to react");
    }

    @Test
    void hardNeverFallsBelowSoft() {
        Search search = new Search();
        SearchLimits limits = new SearchLimits();
        limits.wtime = 300_000;
        limits.winc = 2_000;

        search.computeTimeLimits(limits, Piece.WHITE);

        assertTrue(search.hardLimitMs >= search.softLimitMs,
                "hard=" + search.hardLimitMs + " must never be tighter than soft=" + search.softLimitMs);
    }

    @Test
    void panicModeIsUnaffectedByTheOptimumBasedHardCap() {
        Search search = new Search();
        search.moveOverheadMs = 500;
        SearchLimits limits = new SearchLimits();
        limits.wtime = 200; // below moveOverheadMs -> playableTime goes negative -> panic mode

        search.computeTimeLimits(limits, Piece.WHITE);

        assertTrue(search.panicMode);
        assertEquals(5L, search.hardLimitMs);
    }

    @Test
    void shortTimeControlStillProducesSaneBounds() {
        Search search = new Search();
        SearchLimits limits = new SearchLimits();
        limits.wtime = 10_000; // 10s bullet, no increment
        limits.winc = 0;

        search.computeTimeLimits(limits, Piece.WHITE);

        assertTrue(search.softLimitMs > 0);
        assertTrue(search.optimumTimeMs > 0);
        assertTrue(search.hardLimitMs >= search.softLimitMs);
    }

    // --- node-effort tally (adaptive time's third factor; see ADAPTIVE_EFFORT_*) ---

    @Test
    void effortFactorMapsBestFractionOntoTheClampedRange() {
        // Spec's own reference points: full concentration clamps low (stop sooner), near-zero
        // concentration clamps high (spend longer), an even split lands in between.
        assertEquals(0.60, Search.effortFactor(1.0), 1e-9);  // 1.60 - 1.10 -> 0.50, clamped up
        assertEquals(1.50, Search.effortFactor(0.0), 1e-9);  // 1.60 raw, clamped down
        assertEquals(1.05, Search.effortFactor(0.5), 1e-9);  // 1.60 - 0.55, inside the clamp
    }

    @Test
    void rootEffortTallyTracksNodesOnlyWhenAdaptiveTimeIsOn() {
        engine.board.Position pos = engine.board.Position.fromFen(
                "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1"); // WAC.001 (Bench set)

        // OFF (shipped default): the bookkeeping must never run.
        Search off = new Search();
        off.printInfo = false;
        off.think(pos, SearchLimits.depth(6));
        assertEquals(0L, off.rootEffortTotal(),
                "with useAdaptiveTime off, no effort tally may ever be recorded");

        // ON: the best move's tally is populated and the totals stay within the node count.
        Search on = new Search();
        on.printInfo = false;
        on.useAdaptiveTime = true;
        int best = on.think(engine.board.Position.fromFen(
                "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1"), SearchLimits.depth(6));
        assertTrue(best != 0, "search must return a move");
        assertTrue(on.rootEffortFor(best) > 0,
                "the returned best move must have accumulated root effort");
        assertTrue(on.rootEffortFor(best) <= on.nodes(),
                "one move's effort cannot exceed the whole search's node count");
        assertTrue(on.rootEffortTotal() <= on.nodes(),
                "summed root effort cannot exceed the whole search's node count");
    }
}
