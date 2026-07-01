package engine.search;

import engine.board.Position;

/**
 * Static evaluation seam between search (Phase 2) and the handcrafted evaluation (Phase 3).
 *
 * Contract: {@link #evaluate(Position)} returns a centipawn score from the side-to-move's
 * perspective (positive = good for the side to move), suitable for direct use in negamax.
 * Scores must stay well within +/-{@link Search#MATE_IN_MAX} so they never collide with
 * mate scores. Phase 3's handcrafted evaluation should implement this interface.
 */
public interface Evaluator {
    int evaluate(Position pos);
}
