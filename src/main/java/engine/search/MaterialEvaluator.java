package engine.search;

import engine.board.Piece;
import engine.board.Position;

/**
 * Material-only evaluation stub used by the Phase 2 search until Phase 3's handcrafted
 * evaluation is wired in. Returns a centipawn score from the side-to-move perspective.
 */
public final class MaterialEvaluator implements Evaluator {

    private static final int[] VALUE = {100, 320, 330, 500, 900, 0};

    @Override
    public int evaluate(Position pos) {
        int score = 0;
        for (int type = Piece.PAWN; type <= Piece.QUEEN; type++) {
            int white = Long.bitCount(pos.pieces(Piece.index(Piece.WHITE, type)));
            int black = Long.bitCount(pos.pieces(Piece.index(Piece.BLACK, type)));
            score += VALUE[type] * (white - black);
        }
        return pos.sideToMove() == Piece.WHITE ? score : -score;
    }
}
