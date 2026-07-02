package engine.search;

import engine.board.Position;

/**
 * Adapts the static Phase 3 handcrafted evaluation ({@link engine.eval.Evaluator}) to the
 * instance-based {@link Evaluator} seam that {@link Search} depends on.
 */
public final class HandcraftedEvaluator implements Evaluator {

    @Override
    public int evaluate(Position pos) {
        return engine.eval.Evaluator.evaluate(pos);
    }
}
