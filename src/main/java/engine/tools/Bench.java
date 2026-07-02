package engine.tools;

import engine.board.Move;
import engine.board.Position;
import engine.search.HandcraftedEvaluator;
import engine.search.Search;
import engine.search.SearchLimits;

/**
 * Deterministic fixed-depth benchmark over a fixed position set -- the project's
 * before/after gate for every search change (see the Search Health Check plan).
 *
 * Each position gets a fresh {@link Search} (fresh TT, fresh heuristics), runs
 * single-threaded to a fixed depth, and reports nodes + best move; the summary line's
 * total node count is the "bench signature": identical engine behavior must produce a
 * byte-identical signature across runs, and any search-behavior change shows up as a
 * signature change. Fixed-depth search is immune to time management by construction
 * (obvious-move root-gap pruning and all soft-bound exits only fire under useTime).
 *
 * Position selection biases toward FENs already verified elsewhere in the repo (the
 * PerftTest suite) plus a handful of classics: a closed Spanish and a QGD middlegame
 * (quiet maneuvering), WAC.001 and a Stockfish-bench attack position (sharp tactics),
 * Fine 70 and a pawn race (pawn endgames -- zugzwang/NMP stress), and Lucena (rook
 * endgame).
 */
public final class Bench {
    private Bench() {}

    static final int BENCH_DEPTH = 10;
    static final int BENCH_HASH_MB = 16;

    private static final String[] FENS = {
        // Verified by PerftTest:
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",              // startpos
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // Kiwipete
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                            // perft pos3 (R+P)
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",     // perft pos4 (promos)
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",            // perft pos5 (pins)
        // Quiet middlegames:
        "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 0 9", // closed Spanish
        "rnbq1rk1/ppp1bppp/4pn2/3p2B1/2PP4/2N1PN2/PP3PPP/R2QKB1R b KQ - 3 6",    // QGD main line
        // Sharp tactics:
        "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",          // WAC.001
        "r1b2rk1/2q1b1pp/p2ppn2/1p6/3QP3/1BN1B3/PPP3PP/R4RK1 w - - 0 1",        // SF-bench attack
        // Pawn endgames (zugzwang / NMP stress):
        "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1",                              // Fine 70
        "8/6pk/8/8/8/8/KP6/8 w - - 0 1",                                        // pawn race
        // Rook endgame:
        "1k6/1P6/2K5/8/8/8/8/4R3 w - - 0 1",                                    // Lucena
    };

    /** Runs the bench and prints per-position lines plus the summary signature line. */
    public static void run() {
        long totalNodes = 0;
        long startNanos = System.nanoTime();
        for (int i = 0; i < FENS.length; i++) {
            Position pos = Position.fromFen(FENS[i]);
            Search search = new Search(new HandcraftedEvaluator(), BENCH_HASH_MB);
            search.printInfo = false;
            search.think(pos, SearchLimits.depth(BENCH_DEPTH));
            long nodes = search.nodes();
            totalNodes += nodes;
            System.out.println("position " + (i + 1) + "/" + FENS.length
                    + " nodes " + nodes
                    + " best " + (search.bestMove != 0 ? Move.toUci(search.bestMove) : "0000")
                    + " fen " + FENS[i]);
        }
        long ms = (System.nanoTime() - startNanos) / 1_000_000L;
        long nps = ms > 0 ? totalNodes * 1000L / ms : totalNodes;
        System.out.println("bench: " + totalNodes + " nodes " + ms + " ms " + nps + " nps");
    }

    public static void main(String[] args) {
        run();
    }
}
