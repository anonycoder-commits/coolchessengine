package engine.search;

import engine.board.Position;

/**
 * Adapts the static Phase 3 handcrafted evaluation ({@link engine.eval.Evaluator}) to the
 * instance-based {@link Evaluator} seam that {@link Search} depends on.
 *
 * Each instance owns a private pawn-structure cache (see the layout notes in
 * {@link engine.eval.Evaluator}), so instances must NOT be shared across search threads --
 * {@link LazySmpSearch} constructs one per worker. The cache is exact-verifying, so a cached
 * evaluation is always bit-identical to an uncached one; it only changes speed, never scores.
 */
public final class HandcraftedEvaluator implements Evaluator {

    private final long[] pawnCache = new long[
            engine.eval.Evaluator.PAWN_CACHE_ENTRIES * engine.eval.Evaluator.PAWN_CACHE_STRIDE];

    @Override
    public int evaluate(Position pos) {
        return engine.eval.Evaluator.evaluate(pos, pawnCache);
    }
}
