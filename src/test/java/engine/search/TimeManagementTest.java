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
}
