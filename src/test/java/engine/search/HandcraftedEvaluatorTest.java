package engine.search;

import org.junit.jupiter.api.Test;

import engine.board.Position;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Confirms Search's default evaluator is the real handcrafted eval, not material-only. */
class HandcraftedEvaluatorTest {

    // Same material (one knight each) but white's knight is buried in the corner while
    // black's is centralized -- material counting alone can't tell these positions apart.
    private static final String CORNER_VS_CENTER_KNIGHT =
            "4k3/8/3n4/8/8/8/8/N3K3 w - - 0 1";

    @Test
    void handcraftedEvalPenalizesCorneredKnightBeyondMaterial() {
        Position pos = Position.fromFen(CORNER_VS_CENTER_KNIGHT);

        int material = new MaterialEvaluator().evaluate(pos);
        int handcrafted = engine.eval.Evaluator.evaluate(pos);

        assertEquals(0, material, "material count is equal, so the material-only eval sees no difference");
        assertTrue(Math.abs(handcrafted) > 40,
                "handcrafted eval should penalize the cornered knight well beyond material, got " + handcrafted);
    }

    @Test
    void defaultSearchScoreDiffersFromMaterialOnlySearch() {
        Search handcrafted = new Search();
        handcrafted.printInfo = false;
        handcrafted.useQuiescence = false;
        handcrafted.search(Position.fromFen(CORNER_VS_CENTER_KNIGHT), 1);

        Search material = new Search(new MaterialEvaluator(), 16);
        material.printInfo = false;
        material.useQuiescence = false;
        material.search(Position.fromFen(CORNER_VS_CENTER_KNIGHT), 1);

        assertTrue(Math.abs(handcrafted.bestScore - material.bestScore) > 20,
                "default Search (handcrafted eval, score " + handcrafted.bestScore
                        + ") should disagree with a material-only Search (score "
                        + material.bestScore + ") on a structurally lopsided, materially equal position");
    }
}
