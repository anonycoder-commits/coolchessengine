package engine.search;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Piece;
import engine.board.Position;

/**
 * Classical single-threaded alpha-beta search with iterative deepening.
 *
 * Features (Phase 2): negamax + alpha-beta, iterative deepening with soft/hard time
 * management, a Zobrist-keyed transposition table with bound flags and ply-relative
 * mate scoring, move ordering (TT move, MVV-LVA, killers, history), quiescence search
 * with bounded check evasions, and threefold-repetition / fifty-move draw detection.
 *
 * Evaluation is pluggable via {@link Evaluator}; Phase 3 supplies the real handcrafted
 * evaluation, this phase ships a {@link MaterialEvaluator} stub.
 */
public final class Search {

    public static final int MATE = 30000;
    public static final int MAX_PLY = 128;
    public static final int MATE_IN_MAX = MATE - MAX_PLY; // |score| >= this means a mate score
    private static final int INF = 31000;
    private static final int DRAW = 0;

    private static final int DEFAULT_HASH_MB = 16;
    private static final int TIME_OVERHEAD_MS = 30;
    private static final int QS_CHECK_BUDGET = 2;

    // Move-ordering score bands.
    private static final int TT_SCORE = 2_000_000;
    private static final int CAPTURE_BASE = 1_000_000;
    private static final int PROMO_BASE = 950_000;
    private static final int KILLER0 = 900_000;
    private static final int KILLER1 = 800_000;

    private Evaluator evaluator;
    private final TranspositionTable tt;

    private final int[][] killers = new int[MAX_PLY + 1][2];
    private final int[][][] history = new int[2][64][64];

    // Search state
    private long nodes;
    private long startNanos;
    private long softLimitMs;
    private long hardLimitMs;
    private boolean useTime;
    private int currentDepth;
    private volatile boolean stop;

    private int rootBestMove;
    private int rootBestScore;

    // Public results / toggles
    public int bestMove;
    public int bestScore;
    public boolean printInfo = true;
    public boolean useTT = true;
    public boolean useQuiescence = true;
    public boolean useCheckExtension = true;

    public Search() {
        this(new MaterialEvaluator(), DEFAULT_HASH_MB);
    }

    public Search(Evaluator evaluator, int hashMb) {
        this.evaluator = evaluator;
        this.tt = new TranspositionTable(hashMb);
    }

    public void setEvaluator(Evaluator evaluator) { this.evaluator = evaluator; }
    public void setHashSize(int mb) { tt.resize(mb); }
    public void newGame() { tt.clear(); resetHeuristics(); }
    public void requestStop() { stop = true; }
    public long nodes() { return nodes; }

    /** Convenience fixed-depth entry point (keeps the Phase 1b call site working). */
    public int search(Position pos, int depth) {
        return think(pos, SearchLimits.depth(depth));
    }

    /** Iterative-deepening search under the given limits. Returns the best move. */
    public int think(Position pos, SearchLimits limits) {
        nodes = 0;
        stop = false;
        rootBestMove = 0;
        rootBestScore = 0;
        resetHeuristics();
        startNanos = System.nanoTime();
        computeTimeLimits(limits, pos.sideToMove());

        int maxDepth = limits.depth > 0 ? Math.min(limits.depth, MAX_PLY - 1) : MAX_PLY - 1;
        int committedMove = 0;
        int committedScore = 0;

        for (int d = 1; d <= maxDepth; d++) {
            currentDepth = d;
            int score = negamax(pos, d, -INF, INF, 0);
            if (stop && d > 1) break; // abandon the partial iteration, keep last completed result
            committedMove = rootBestMove;
            committedScore = score;
            if (printInfo) printInfo(d, score);
            if (useTime && elapsedMs() >= softLimitMs) break;
            if (Math.abs(score) >= MATE_IN_MAX) break; // forced mate found
        }

        if (committedMove == 0) {
            MoveList legal = new MoveList();
            MoveGenerator.generateLegal(pos, legal);
            if (legal.size > 0) committedMove = legal.moves[0];
        }

        bestMove = committedMove;
        bestScore = committedScore;
        return committedMove;
    }

    private int negamax(Position pos, int depth, int alpha, int beta, int ply) {
        if (stop) return 0;
        if (ply > 0 && pos.isDrawByRuleOrRepetition()) return DRAW;

        boolean inCheck = pos.inCheck(pos.sideToMove());
        if (useCheckExtension && inCheck) depth++;

        if (depth <= 0) {
            return useQuiescence ? quiescence(pos, alpha, beta, ply, QS_CHECK_BUDGET)
                                 : evaluator.evaluate(pos);
        }

        if ((++nodes & 2047) == 0) checkTime();

        long key = pos.zobristKey();
        int ttMove = 0;
        if (useTT && tt.probe(key)) {
            ttMove = tt.ttMove;
            if (ply > 0 && tt.ttDepth >= depth) {
                int s = scoreFromTt(tt.ttScore, ply);
                int f = tt.ttFlag;
                if (f == TranspositionTable.FLAG_EXACT) return s;
                if (f == TranspositionTable.FLAG_LOWER && s >= beta) return s;
                if (f == TranspositionTable.FLAG_UPPER && s <= alpha) return s;
            }
        }

        MoveList moves = new MoveList();
        MoveGenerator.generateLegal(pos, moves);
        if (moves.size == 0) {
            return inCheck ? -MATE + ply : DRAW; // checkmate or stalemate
        }

        int us = pos.sideToMove();
        int[] scores = scoreMoves(pos, moves, ttMove, ply, us);

        int bestScore = -INF;
        int localBest = 0;
        int origAlpha = alpha;

        for (int i = 0; i < moves.size; i++) {
            selectNext(moves, scores, i);
            int move = moves.moves[i];
            pos.makeMove(move);
            int score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1);
            pos.unmakeMove(move);
            if (stop) return 0;

            if (score > bestScore) {
                bestScore = score;
                localBest = move;
                if (ply == 0) {
                    rootBestMove = move;
                    rootBestScore = score;
                }
                if (score > alpha) {
                    alpha = score;
                    if (alpha >= beta) {
                        if (!Move.isCapture(move) && !Move.isPromotion(move)) {
                            updateKillers(ply, move);
                            history[us][Move.from(move)][Move.to(move)] += depth * depth;
                        }
                        break;
                    }
                }
            }
        }

        if (useTT) {
            int flag = bestScore <= origAlpha ? TranspositionTable.FLAG_UPPER
                     : bestScore >= beta ? TranspositionTable.FLAG_LOWER
                     : TranspositionTable.FLAG_EXACT;
            tt.store(key, depth, scoreToTt(bestScore, ply), flag, localBest);
        }
        return bestScore;
    }

    /** Quiescence search over captures/promotions, with bounded check evasions. */
    private int quiescence(Position pos, int alpha, int beta, int ply, int checkBudget) {
        if (stop) return 0;
        if ((++nodes & 2047) == 0) checkTime();
        if (pos.isDrawByRuleOrRepetition()) return DRAW;

        boolean inCheck = pos.inCheck(pos.sideToMove());
        if (ply >= MAX_PLY) return evaluator.evaluate(pos);

        int bestScore;
        if (inCheck) {
            if (checkBudget <= 0) return evaluator.evaluate(pos);
            bestScore = -INF;
        } else {
            int standPat = evaluator.evaluate(pos);
            bestScore = standPat;
            if (standPat >= beta) return standPat;
            if (standPat > alpha) alpha = standPat;
        }

        MoveList moves = new MoveList();
        int nextBudget = checkBudget;
        if (inCheck) {
            MoveGenerator.generateLegal(pos, moves);
            nextBudget = checkBudget - 1;
            if (moves.size == 0) return -MATE + ply; // checkmate
        } else {
            MoveGenerator.generateLegalCaptures(pos, moves);
        }

        int[] scores = scoreMovesQ(pos, moves);
        for (int i = 0; i < moves.size; i++) {
            selectNext(moves, scores, i);
            int move = moves.moves[i];
            pos.makeMove(move);
            int score = -quiescence(pos, -beta, -alpha, ply + 1, nextBudget);
            pos.unmakeMove(move);
            if (stop) return 0;
            if (score > bestScore) {
                bestScore = score;
                if (score > alpha) {
                    alpha = score;
                    if (alpha >= beta) break;
                }
            }
        }
        return bestScore;
    }

    // --- move ordering ---

    private int[] scoreMoves(Position pos, MoveList moves, int ttMove, int ply, int us) {
        int[] s = new int[moves.size];
        for (int i = 0; i < moves.size; i++) {
            s[i] = scoreMove(pos, moves.moves[i], ttMove, ply, us);
        }
        return s;
    }

    private int scoreMove(Position pos, int move, int ttMove, int ply, int us) {
        if (move == ttMove) return TT_SCORE;
        int flag = Move.flag(move);
        boolean capture = (flag & 4) != 0;
        boolean promo = (flag & 8) != 0;
        if (capture) {
            int victim = flag == Move.EP_CAPTURE ? Piece.PAWN : Piece.type(pos.pieceAt(Move.to(move)));
            int attacker = Piece.type(pos.pieceAt(Move.from(move)));
            int score = CAPTURE_BASE + victim * 16 - attacker;
            if (promo) score += 64;
            return score;
        }
        if (promo) {
            return PROMO_BASE + Move.promoType(move);
        }
        if (move == killers[ply][0]) return KILLER0;
        if (move == killers[ply][1]) return KILLER1;
        return history[us][Move.from(move)][Move.to(move)];
    }

    private int[] scoreMovesQ(Position pos, MoveList moves) {
        int[] s = new int[moves.size];
        for (int i = 0; i < moves.size; i++) {
            int move = moves.moves[i];
            int flag = Move.flag(move);
            int score = 0;
            if ((flag & 4) != 0) {
                int victim = flag == Move.EP_CAPTURE ? Piece.PAWN : Piece.type(pos.pieceAt(Move.to(move)));
                int attacker = Piece.type(pos.pieceAt(Move.from(move)));
                score = victim * 16 - attacker;
            }
            if ((flag & 8) != 0) score += 100;
            s[i] = score;
        }
        return s;
    }

    private void selectNext(MoveList moves, int[] scores, int i) {
        int best = i;
        for (int j = i + 1; j < moves.size; j++) {
            if (scores[j] > scores[best]) best = j;
        }
        if (best != i) {
            int tm = moves.moves[i];
            moves.moves[i] = moves.moves[best];
            moves.moves[best] = tm;
            int ts = scores[i];
            scores[i] = scores[best];
            scores[best] = ts;
        }
    }

    private void updateKillers(int ply, int move) {
        if (killers[ply][0] != move) {
            killers[ply][1] = killers[ply][0];
            killers[ply][0] = move;
        }
    }

    private void resetHeuristics() {
        for (int[] k : killers) {
            k[0] = 0;
            k[1] = 0;
        }
        for (int c = 0; c < 2; c++) {
            for (int f = 0; f < 64; f++) {
                java.util.Arrays.fill(history[c][f], 0);
            }
        }
    }

    // --- mate-score ply relativization for the TT ---

    private int scoreToTt(int score, int ply) {
        if (score >= MATE_IN_MAX) return score + ply;
        if (score <= -MATE_IN_MAX) return score - ply;
        return score;
    }

    private int scoreFromTt(int score, int ply) {
        if (score >= MATE_IN_MAX) return score - ply;
        if (score <= -MATE_IN_MAX) return score + ply;
        return score;
    }

    // --- time management ---

    private void computeTimeLimits(SearchLimits limits, int side) {
        useTime = false;
        softLimitMs = Long.MAX_VALUE;
        hardLimitMs = Long.MAX_VALUE;
        if (limits.infinite) return;
        if (limits.movetime > 0) {
            long t = Math.max(1, limits.movetime - TIME_OVERHEAD_MS);
            softLimitMs = t;
            hardLimitMs = t;
            useTime = true;
            return;
        }
        if (limits.depth > 0 && !limits.hasClock()) return; // pure depth-limited

        int time = side == Piece.WHITE ? limits.wtime : limits.btime;
        int inc = side == Piece.WHITE ? limits.winc : limits.binc;
        if (time <= 0) return; // no usable clock -> depth/infinite

        int mtg = limits.movestogo > 0 ? limits.movestogo : 30;
        long soft = (long) time / mtg + (long) inc * 3 / 4;
        long hard = Math.min((long) time / 4, soft * 4);
        long cap = Math.max(1, time - TIME_OVERHEAD_MS);
        soft = Math.min(soft, cap);
        hard = Math.min(hard, cap);
        if (soft < 1) soft = 1;
        if (hard < soft) hard = soft;
        softLimitMs = soft;
        hardLimitMs = hard;
        useTime = true;
    }

    private void checkTime() {
        if (useTime && currentDepth >= 2 && elapsedMs() >= hardLimitMs) {
            stop = true;
        }
    }

    private long elapsedMs() {
        return (System.nanoTime() - startNanos) / 1_000_000L;
    }

    private void printInfo(int depth, int score) {
        long ms = elapsedMs();
        long nps = ms > 0 ? nodes * 1000L / ms : nodes;
        System.out.println("info depth " + depth
                + " score " + formatScore(score)
                + " nodes " + nodes
                + " time " + ms
                + " nps " + nps
                + " pv " + (rootBestMove != 0 ? Move.toUci(rootBestMove) : "0000"));
    }

    private String formatScore(int score) {
        if (score >= MATE_IN_MAX) {
            int plies = MATE - score;
            return "mate " + ((plies + 1) / 2);
        }
        if (score <= -MATE_IN_MAX) {
            int plies = MATE + score;
            return "mate " + (-((plies + 1) / 2));
        }
        return "cp " + score;
    }
}
