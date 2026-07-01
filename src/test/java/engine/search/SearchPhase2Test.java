package engine.search;

import org.junit.jupiter.api.Test;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Position;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Phase 2 correctness checks: alpha-beta soundness, TT soundness, mate scoring, draws. */
class SearchPhase2Test {

    private static final int REF_INF = 32000;
    private static final MaterialEvaluator EVAL = new MaterialEvaluator();

    private static final String KIWIPETE =
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    private static final String ITALIAN =
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3";
    private static final String STARTPOS =
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    /** Plain full-width negamax reference (no alpha-beta, no TT, no quiescence). */
    private static int refNegamax(Position pos, int depth, int ply) {
        if (ply > 0 && pos.isDrawByRuleOrRepetition()) return 0;
        if (depth == 0) return EVAL.evaluate(pos);
        MoveList moves = new MoveList();
        MoveGenerator.generateLegal(pos, moves);
        if (moves.size == 0) {
            return pos.inCheck(pos.sideToMove()) ? -Search.MATE + ply : 0;
        }
        int best = -REF_INF;
        for (int i = 0; i < moves.size; i++) {
            int move = moves.moves[i];
            pos.makeMove(move);
            int v = -refNegamax(pos, depth - 1, ply + 1);
            pos.unmakeMove(move);
            if (v > best) best = v;
        }
        return best;
    }

    private static Search plainSearch() {
        Search s = new Search();
        s.printInfo = false;
        s.useTT = false;
        s.useQuiescence = false;
        s.useCheckExtension = false;
        return s;
    }

    private void assertAbEqualsMinimax(String fen, int depth) {
        Position pos = Position.fromFen(fen);
        int expected = refNegamax(pos, depth, 0);
        Search s = plainSearch();
        s.search(pos, depth);
        assertEquals(expected, s.bestScore,
                "alpha-beta score must equal full-width minimax for " + fen + " d" + depth);
    }

    @Test
    void alphaBetaEqualsMinimaxStartpos() {
        assertAbEqualsMinimax(STARTPOS, 4);
    }

    @Test
    void alphaBetaEqualsMinimaxItalian() {
        assertAbEqualsMinimax(ITALIAN, 4);
    }

    @Test
    void alphaBetaEqualsMinimaxKiwipete() {
        assertAbEqualsMinimax(KIWIPETE, 3);
    }

    /** TT must not change the search result (classic silent-corruption guard). */
    private void assertTtConsistent(String fen, int depth) {
        Position p1 = Position.fromFen(fen);
        Search withTt = new Search();
        withTt.printInfo = false;
        withTt.useTT = true;
        withTt.search(p1, depth);

        Position p2 = Position.fromFen(fen);
        Search noTt = new Search();
        noTt.printInfo = false;
        noTt.useTT = false;
        noTt.search(p2, depth);

        assertEquals(noTt.bestScore, withTt.bestScore,
                "TT-on and TT-off scores must match for " + fen + " d" + depth);
    }

    @Test
    void ttDoesNotChangeResultStartpos() {
        assertTtConsistent(STARTPOS, 6);
    }

    @Test
    void ttDoesNotChangeResultKiwipete() {
        assertTtConsistent(KIWIPETE, 5);
    }

    @Test
    void findsMateInTwoWithCorrectScore() {
        // White mates in 2: 1.Qd7+ Kb8 2.Qd8# (engine-verified).
        Position pos = Position.fromFen("2k5/8/2K5/8/8/8/8/3Q4 w - - 0 1");
        Search s = new Search();
        s.printInfo = false;
        s.search(pos, 6);
        // Mate in 2 = 3 plies from root.
        assertEquals(Search.MATE - 3, s.bestScore, "expected mate-in-2 score");
        assertTrue(s.bestScore >= Search.MATE_IN_MAX, "score must be a mate score");
    }

    @Test
    void repetitionIsDetectedAsDraw() {
        Position pos = Position.startpos();
        String[] shuffle = {"g1f3", "g8f6", "f3g1", "f6g8"};
        for (int i = 0; i < shuffle.length; i++) {
            int move = parse(pos, shuffle[i]);
            pos.makeMove(move);
            if (i < shuffle.length - 1) {
                assertFalse(pos.isDrawByRuleOrRepetition(),
                        "no repetition yet after " + (i + 1) + " plies");
            }
        }
        assertTrue(pos.isDrawByRuleOrRepetition(),
                "position repeats the start after the knight shuffle");
    }

    @Test
    void fiftyMoveRuleIsDraw() {
        assertTrue(Position.fromFen("4k3/8/8/8/8/8/8/4K2R w K - 100 80")
                .isDrawByRuleOrRepetition(), "halfmove clock 100 is a draw");
        assertFalse(Position.fromFen("4k3/8/8/8/8/8/8/4K2R w K - 99 80")
                .isDrawByRuleOrRepetition(), "halfmove clock 99 is not yet a draw");
    }

    @Test
    void transpositionTableStoreProbeRoundTrip() {
        TranspositionTable tt = new TranspositionTable(1);
        long key = 0x123456789abcdefL;
        tt.store(key, 7, 123, TranspositionTable.FLAG_EXACT, Move.make(12, 28, Move.QUIET));
        assertTrue(tt.probe(key), "stored key should be found");
        assertEquals(7, tt.ttDepth);
        assertEquals(123, tt.ttScore);
        assertEquals(TranspositionTable.FLAG_EXACT, tt.ttFlag);
        assertEquals(Move.make(12, 28, Move.QUIET), tt.ttMove);
        assertFalse(tt.probe(key ^ 0xFFFFL), "different key should miss");
    }

    private static int parse(Position pos, String uci) {
        MoveList moves = new MoveList();
        MoveGenerator.generateLegal(pos, moves);
        for (int i = 0; i < moves.size; i++) {
            if (Move.toUci(moves.moves[i]).equals(uci)) return moves.moves[i];
        }
        throw new IllegalArgumentException("no legal move " + uci);
    }
}
