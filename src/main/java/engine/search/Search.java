package engine.search;

import java.util.concurrent.atomic.AtomicBoolean;

import engine.board.Attacks;
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
 * Evaluation is pluggable via {@link Evaluator}; the default constructor uses
 * {@link HandcraftedEvaluator} (Phase 3's real handcrafted evaluation). {@link MaterialEvaluator}
 * remains available as a material-only stub for tests and reference comparisons.
 */
public final class Search {

    public static final int MATE = 30000;
    public static final int MAX_PLY = 128;
    public static final int MATE_IN_MAX = MATE - MAX_PLY; // |score| >= this means a mate score
    private static final int INF = 31000;
    private static final int DRAW = 0;

    private static final int DEFAULT_HASH_MB = 16;
    private static final int TIME_OVERHEAD_MS = 30;
    // Check evasions in qsearch are only ever fully generated once per call chain -- the
    // very first in-check node encountered consumes the budget's single unit, so any check
    // met one ply deeper (a check chain) falls through to the checkBudget<=0 branch and is
    // resolved with a static eval instead of another full legal-move generation. Searching
    // check evasions to unbounded depth is what turns a single sharp tactical line into a
    // multi-million-node qsearch blowup, since every evasion (not just captures) has to be
    // tried at each such node.
    private static final int QS_CHECK_BUDGET = 1;

    // Aspiration windows: initial half-width (centipawns) of the window built around the
    // previous iteration's score; applied from this depth onward.
    private static final int ASPIRATION_DELTA = 20;
    private static final int ASPIRATION_MIN_DEPTH = 4;
    // Cap on how many times a single side of the window may double before giving up and
    // opening that side fully to (-INF)/(+INF). Left uncapped, repeated fail-high/fail-low
    // cycles (common once RFP/null-move/futility pruning's approximate margins cause a
    // bigger swing than the initial +-20cp window expects) double the delta all the way from
    // 20 toward 31000 -- up to ~11 successive full-depth re-searches of the same iteration
    // before the window is finally wide enough, each one costing nearly as much as the
    // original search. Capping retries bounds any one iteration to at most
    // ASPIRATION_MAX_FAILS + 1 negamax invocations before it's forced fully open.
    private static final int ASPIRATION_MAX_FAILS = 2;

    // Late Move Reduction: move-loop thresholds below which no reduction is applied. The
    // first LMR_MIN_MOVE_INDEX + 1 moves (PV/TT move, killers) are always searched at full
    // depth, since move ordering makes them the most likely to be best.
    private static final int LMR_MIN_DEPTH = 2;
    private static final int LMR_MIN_MOVE_INDEX = 2;
    // Deeper reduction (R=2 instead of R=1) once both the remaining depth and the move index
    // are large enough that a wrong reduction is cheap to recover from.
    private static final int LMR_DEEP_MIN_DEPTH = 6;
    private static final int LMR_DEEP_MIN_MOVE_INDEX = 8;

    // Reverse futility pruning (a.k.a. static null-move pruning): near the leaves, if the
    // static eval already beats beta by more than a depth-scaled margin, the position is so
    // good that even a pessimistic view of it should still hold -- return the eval without
    // searching any moves. Only tried at low depth, since the margin's reliability shrinks
    // fast as more plies (and thus more opponent tries) remain.
    private static final int RFP_MAX_DEPTH = 6;
    private static final int RFP_MARGIN_PER_DEPTH = 90;

    // Null-move pruning: let the side to move "pass" and search the opponent's best reply at
    // reduced depth; if even a free move for the opponent doesn't save them (their best score
    // is still <= our beta), our position is winning enough to prune without exploring the
    // rest of this node's real moves. NULL_MOVE_MIN_DEPTH keeps the reduced recursive search
    // meaningful; the position must also have at least one non-pawn, non-king piece for the
    // side to move (see hasNonPawnMaterial), since null-move reasoning is unsound in
    // zugzwang-prone king/pawn endgames where passing is actually the losing option.
    private static final int NULL_MOVE_MIN_DEPTH = 3;
    private static final int NULL_MOVE_REDUCTION = 3;

    // Futility pruning: near the leaves, a quiet move that can't plausibly close the gap to
    // alpha (staticEval + a depth-scaled margin still falls short) is skipped without being
    // searched -- the same reasoning as quiescence's delta pruning, one ply higher up. The
    // very first move tried at a node is never skipped (see the move loop), so a node can
    // never return with zero moves actually searched.
    private static final int FUTILITY_MAX_DEPTH = 6;
    private static final int FUTILITY_BASE_MARGIN = 100;
    private static final int FUTILITY_MARGIN_PER_DEPTH = 80;

    // Quiescence delta pruning: a capture that can't plausibly close the gap to alpha even
    // in the best case (standPat + captured piece's value + this margin) is skipped without
    // being searched. The margin absorbs positional gains the static stand-pat eval misses.
    private static final int DELTA_MARGIN = 200;

    // Soft-bound (optimum time) early exit: once the root move has stopped changing
    // between iterations, further depth rarely changes the decision, so it's safe to bank
    // the unused clock time instead of confirming a result we already trust.
    //
    // optimumTimeMs is a fraction of the REAL per-move budget (softLimitMs). It was previously
    // derived from playableTime/45 -- a per-move slice that assumed ~45 more moves at the
    // current clock, wildly disconnected from the actual budget: on a 5-minute+increment game
    // it produced ~2s, so the stability exit banked EVERY quiet move down to ~2s and the engine
    // finished games with 100+ seconds unused (observed: mated with 142s in hand), never
    // searching deep enough to notice a slow positional collapse. Anchoring it to softLimitMs
    // makes the banking target scale with what the engine actually intends to spend.
    private static final double OPTIMUM_TIME_FRACTION = 0.6; // optimumTime = softLimitMs * 0.6
    private static final int SOFT_BOUND_MIN_DEPTH = 5; // don't even consider exiting before this depth
    // Consecutive same-root-move iterations required before the engine trusts a "quiet"
    // position enough to bank the unused clock time. Kept deliberately high (14, not the
    // textbook 3-4) -- a shallow stability streak is exactly what a slow-building positional
    // advantage looks like right up until the point the shallow depths finally see it, so a
    // low threshold was handing back time on moves that hadn't actually been verified deeply.
    private static final int SOFT_BOUND_STABLE_ITERATIONS = 14;
    private static final int SOFT_BOUND_STABLE_ITERATIONS_DECISIVE = 2; // relaxed threshold once clearly winning
    private static final int SOFT_BOUND_DECISIVE_SCORE_CP = 200; // +2.00 pawns

    // Volatility gate: even a stable root move must not trigger the time-banking soft exit when
    // the position is sharp -- a forcing pawn breakthrough (e.g. ...d3+) can sit at a mild
    // static eval for many stable iterations while the real damage lurks a few plies deeper.
    // When the committed move is itself a breakthrough (promotion, advanced pawn push, or a
    // check) OR the recent iteration scores are swinging, the optimum-time soft exit is
    // suppressed so the search keeps going (still bounded by softLimitMs and the hard ceiling).
    private static final int VOLATILITY_SWING_CP = 30;
    private static final int VOLATILITY_WINDOW = 3;
    private static final int BREAKTHROUGH_MIN_REL_RANK = 4; // push to the relative 5th rank or beyond

    // Evaluation-based time lock: a near-dead-equal root score (|score| <= this many
    // centipawns) during the opening/early middlegame (game ply < EVAL_LOCK_MAX_GAME_PLY) is
    // precisely the situation where quiet, creeping positional threats matter most and a
    // shallow best-move stability streak is least trustworthy -- so the stable-iteration
    // soft-bound exit above is disabled outright while locked, and the search is additionally
    // held to at least EVAL_LOCK_MIN_TIME_FRACTION of optimumTimeMs before any other soft
    // (non-hard-limit) time-based exit is allowed to fire.
    private static final int EVAL_LOCK_MAX_ABS_SCORE_CP = 50; // +-0.50 pawns
    private static final int EVAL_LOCK_MAX_GAME_PLY = 15;
    private static final double EVAL_LOCK_MIN_TIME_FRACTION = 0.8;

    // Hard-limit-vs-optimum-time ratio cap: the absolute ceiling (hardLimitMs) is the only
    // time bound checkTime() enforces *inside* the recursive search -- optimumTimeMs is only
    // consulted between completed iterations, in think()'s soft-bound stability check. Left
    // unbounded, hardLimitMs (derived from playableTime/4) can run to many multiples of
    // optimumTimeMs early in a game when the clock is still nearly full, so a single
    // expensive ply (deep tactics, aspiration-window re-search cascade, LMR re-search
    // cascade) can legally run tens of seconds past the intended per-move budget before the
    // soft-bound logic -- which only fires once that ply has already finished -- ever gets a
    // chance to react. Capping hard to a bounded multiple of optimum keeps the worst case for
    // any single ply within a sane multiple of what the engine actually intended to spend.
    static final double HARD_LIMIT_OPTIMUM_MULTIPLIER = 4.0;

    // Move Overhead panic guardrail: if network/process lag has eaten the clock down to
    // (almost) nothing, don't trust normal allocation math -- force a near-instant emergency
    // stop instead of risking a time forfeit.
    private static final int PANIC_THRESHOLD_MS = 50; // playableTime at/below this triggers panic mode
    private static final long PANIC_HARD_LIMIT_MS = 5;
    private static final long PANIC_OPTIMUM_TIME_MS = 1;
    private static final int PANIC_MAX_DEPTH = 2; // iterative deepening never goes past this in panic mode

    // Move-ordering score bands, strictly descending and non-overlapping:
    //   TT move > winning captures > promotions > killers > quiet history > losing captures.
    // Losing captures sit in their OWN tier below every quiet-history score, not above it:
    // SEE has already proven such a capture loses material, so a quiet move backed by any
    // real cutoff history is a better bet than blindly trying a capture already known to be
    // bad. (An earlier version placed BAD_CAPTURE_BASE just above the history range, which
    // meant nearly every losing capture -- regardless of how bad -- outranked every
    // history-backed quiet move; whether an irrelevant bad capture happened to be available
    // in a given position was enough on its own to reorder the quiet moves behind it,
    // producing exactly the position-to-position branching-factor instability this fixes.)
    private static final int TT_SCORE = 2_000_000;
    private static final int CAPTURE_BASE = 1_000_000;
    private static final int PROMO_BASE = 950_000;
    private static final int KILLER0 = 900_000;
    private static final int KILLER1 = 800_000;
    // Quiet-history ceiling/floor: comfortably below KILLER1 (so even a maxed-out history
    // score can never outrank a killer) and symmetric around 0 so the gravity-based update
    // in updateHistory() self-bounds cleanly in both directions.
    private static final int HISTORY_MAX = 100_000;
    private static final int HISTORY_MIN = -HISTORY_MAX;
    // Deliberately far below HISTORY_MIN: a losing capture must never be able to outscore
    // even the most heavily-malused quiet move.
    private static final int BAD_CAPTURE_BASE = -1_000_000;

    // Centipawn piece values for SEE, indexed by Piece.PAWN..Piece.KING.
    private static final int[] SEE_VALUE = {100, 320, 330, 500, 900, 20000};

    private Evaluator evaluator;
    private final TranspositionTable tt;

    private final int[][] killers = new int[MAX_PLY + 1][2];
    private final int[][][] history = new int[2][64][64];

    // Per-ply scratch buffers, reused across nodes instead of allocating a MoveList +
    // score array at every call. Safe single-threaded: negamax/quiescence recursion
    // strictly increases ply down the call stack, so a given ply's slot is never
    // "live" in two frames at once (make/unmake backtracking guarantees this).
    private final MoveList[] moveListPool = new MoveList[MAX_PLY + 2];
    private final int[][] scorePool = new int[MAX_PLY + 2][256];
    // Records every quiet move actually searched (not skipped by futility pruning) at the
    // current node, in order, so that if one of them causes a beta cutoff, every quiet tried
    // and rejected before it can receive a matching history malus -- see the move loop in
    // negamax(). Ply-indexed like the pools above for the same reentrancy-safety reason.
    private final int[][] quietsTriedPool = new int[MAX_PLY + 2][256];
    {
        for (int i = 0; i < moveListPool.length; i++) moveListPool[i] = new MoveList();
    }

    // Scratch buffers for see(): scoreMove/scoreMovesQ call it in a flat loop over one
    // ply's move list, never re-entrantly, so a single shared pair is safe to reuse.
    private final long[] seeByType = new long[12];
    private final int[] seeGain = new int[32];

    // Search state
    private long nodes;
    private long startNanos;
    // Package-private (rather than private) so a test can drive computeTimeLimits() directly
    // and assert on the resulting bounds without running a full search -- see SeeTest's
    // similar rationale for see() being package-private instead of private.
    long softLimitMs;
    long hardLimitMs;
    long optimumTimeMs;
    private boolean useTime;
    boolean panicMode;
    private int currentDepth;
    private volatile boolean stop;

    // Mid-iteration soft-bound stop: armed at the end of a completed iteration once the
    // stability requirement (see the soft-bound block in think()) is met, independent of
    // whether elapsed time has crossed optimumTimeMs yet. Letting checkTime() (called
    // periodically deep inside the recursion) also consult this -- rather than only
    // think()'s per-iteration loop body -- means a runaway iteration (e.g. an aspiration
    // re-search cascade) can be aborted the moment elapsed time crosses optimumTimeMs,
    // instead of being allowed to run all the way to hardLimitMs before anything reacts.
    private boolean softStopArmed;
    // Set by checkTime() when it stops the search via softStopArmed (as opposed to
    // hardLimitMs), purely so think() can print an equivalent "soft-bound early exit"
    // diagnostic line for a stop that happened mid-iteration instead of between iterations.
    private boolean softStopFiredMidIteration;

    // Move Overhead: milliseconds deducted from the raw remaining clock before any time
    // allocation math runs, to compensate for network transit and wrapper/process dispatch
    // lag (e.g. a Python UCI wrapper like lichess-bot) that eats into the clock between the
    // opponent's move landing and this engine's own thinking time actually starting. Public
    // and mutable so a UCI "Move Overhead"-style option can adjust it per deployment.
    public long moveOverheadMs = 500;

    // Lazy SMP: set by LazySmpSearch on every worker (master and helpers alike) so a
    // stop decided anywhere -- the master's own timer, or an external UCI 'stop' -- is
    // visible to all threads. Null for standalone single-threaded use (unaffected).
    private AtomicBoolean sharedStop;

    private int rootBestMove;
    private int rootBestScore;
    private int rootSecondBestScore;

    // Public results / toggles
    public int bestMove;
    public int bestScore;
    public boolean printInfo = true;
    public boolean useTT = true;
    public boolean useQuiescence = true;
    public boolean useCheckExtension = true;
    public boolean useLmr = true;
    public boolean useRfp = true;
    public boolean useNullMove = true;
    public boolean useFutility = true;
    public boolean useVolatilityGate = true; // suppress the soft early exit in sharp positions

    // Obvious-move pruning: skip search entirely on a forced single reply, and cut
    // iterative deepening short once a shallow iteration shows a lopsided root gap.
    public boolean useObviousMovePruning = true;
    public int obviousMoveCheckDepth = 3;
    public int obviousMoveMargin = 300;
    public boolean bypassPrinted;

    public Search() {
        this(new HandcraftedEvaluator(), DEFAULT_HASH_MB);
    }

    public Search(Evaluator evaluator, int hashMb) {
        this.evaluator = evaluator;
        this.tt = new TranspositionTable(hashMb);
    }

    /** Builds a worker that shares an existing table instead of owning one (Lazy SMP). */
    public Search(Evaluator evaluator, TranspositionTable sharedTt) {
        this.evaluator = evaluator;
        this.tt = sharedTt;
    }

    public void setEvaluator(Evaluator evaluator) { this.evaluator = evaluator; }
    public void setHashSize(int mb) { tt.resize(mb); }
    public void newGame() { tt.clear(); resetHeuristics(); }
    public void requestStop() { stop = true; }
    public long nodes() { return nodes; }

    /** Wires this worker to an externally-owned stop flag (see {@link LazySmpSearch}). */
    void setSharedStop(AtomicBoolean sharedStop) { this.sharedStop = sharedStop; }

    private boolean stopped() {
        return stop || (sharedStop != null && sharedStop.get());
    }

    /** Convenience fixed-depth entry point (keeps the Phase 1b call site working). */
    public int search(Position pos, int depth) {
        return think(pos, SearchLimits.depth(depth));
    }

    /** Iterative-deepening search under the given limits. Returns the best move. */
    public int think(Position pos, SearchLimits limits) {
        bypassPrinted = false;
        nodes = 0;
        stop = false;
        softStopArmed = false;
        softStopFiredMidIteration = false;

        MoveList rootMoves = moveListPool[0];
        rootMoves.clear();
        MoveGenerator.generateLegal(pos, rootMoves);
        if (rootMoves.size == 1) {
            return bypassForcedMove(pos, rootMoves.moves[0]);
        }

        rootBestMove = 0;
        rootBestScore = 0;
        rootSecondBestScore = 0;
        resetHeuristics();
        startNanos = System.nanoTime();
        computeTimeLimits(limits, pos.sideToMove());

        int maxDepth = limits.depth > 0 ? Math.min(limits.depth, MAX_PLY - 1) : MAX_PLY - 1;
        if (panicMode) {
            // Emergency stop configuration: the clock (after deducting moveOverheadMs) is all
            // but gone, so don't let iterative deepening attempt normal depths at all -- cap it
            // structurally at PANIC_MAX_DEPTH so the loop returns the first stable move found
            // at depth 1 or 2, regardless of how the (already razor-thin) time bounds resolve.
            maxDepth = Math.min(maxDepth, PANIC_MAX_DEPTH);
        }
        int committedMove = 0;
        int committedScore = 0;

        // Game ply (half-moves since the game/FEN start, independent of this search's own
        // recursion ply) drives the evaluation-based time lock below -- fullmoveNumber/
        // sideToMove survive a "position fen ... moves ..." reload the same way an actual
        // game history would, unlike Position's internal recursion-ply counter.
        int gamePly = (pos.fullmoveNumber() - 1) * 2 + (pos.sideToMove() == Piece.BLACK ? 1 : 0);

        int previousScore = 0;
        // Best-move stability tracking: local to this think() call (not instance state), so
        // it starts fresh on every search and needs no explicit reset/flush between calls.
        int previousBestMove = 0;
        int stableIterationsCount = 0;
        // Ring buffer of the last few committed scores, for the volatility swing signal.
        int[] recentScores = new int[VOLATILITY_WINDOW];
        int recentScoreCount = 0;
        for (int d = 1; d <= maxDepth; d++) {
            currentDepth = d;
            int score;
            if (d >= ASPIRATION_MIN_DEPTH) {
                // Aspiration windows: rather than searching the full (-INF, INF) range, start
                // with a narrow window centered on the previous iteration's score. Most
                // positions are stable from one depth to the next, so the true score usually
                // falls inside this window and we save the cost of exploring branches that
                // only matter for scores far outside the expected range.
                int delta = ASPIRATION_DELTA;
                int alpha = Math.max(-INF, previousScore - delta);
                int beta = Math.min(INF, previousScore + delta);
                int lowFails = 0;
                int highFails = 0;
                while (true) {
                    rootBestScore = -INF;
                    rootSecondBestScore = -INF;
                    score = negamax(pos, d, alpha, beta, 0, 0);
                    if (stopped()) break;
                    if (score <= alpha) {
                        // Fail-low: the true score is at or below the window floor, so the
                        // move we expected to be fine actually isn't. Double the miss delta
                        // (exponential widening: 20 -> 40 -> 80 -> ...) and drop the floor by
                        // that much, but only for a bounded number of attempts
                        // (ASPIRATION_MAX_FAILS): once that many re-searches have still failed
                        // low, stop guessing at a bound and open the floor fully so this side
                        // can never fail low again, capping the total cost of this iteration's
                        // re-search cascade instead of doubling all the way to +-INF.
                        lowFails++;
                        if (lowFails >= ASPIRATION_MAX_FAILS) {
                            alpha = -INF;
                        } else {
                            delta *= 2;
                            alpha = Math.max(-INF, previousScore - delta);
                        }
                        continue;
                    }
                    if (score >= beta) {
                        // Fail-high: same bounded exponential widening as the fail-low branch,
                        // but raising the ceiling instead of lowering the floor.
                        highFails++;
                        if (highFails >= ASPIRATION_MAX_FAILS) {
                            beta = INF;
                        } else {
                            delta *= 2;
                            beta = Math.min(INF, previousScore + delta);
                        }
                        continue;
                    }
                    break; // score landed strictly inside the window: iteration is settled
                }
            } else {
                rootBestScore = -INF;
                rootSecondBestScore = -INF;
                score = negamax(pos, d, -INF, INF, 0, 0);
            }
            if (stopped() && d > 1) {
                // Abandon the partial iteration, keep last completed result. If checkTime()
                // stopped us via softStopArmed rather than hardLimitMs, log it the same way
                // as the between-iterations soft-bound exit below -- this is the mid-iteration
                // counterpart, firing when a runaway iteration (e.g. an aspiration re-search
                // cascade) is cut short instead of being allowed to run to hardLimitMs.
                if (softStopFiredMidIteration && printInfo) {
                    System.err.println("info string soft-bound early exit (mid-iteration): depth=" + d
                            + " move=" + Move.toUci(committedMove)
                            + " stableIterations=" + stableIterationsCount
                            + " elapsedMs=" + elapsedMs()
                            + " optimumTimeMs=" + optimumTimeMs);
                }
                break;
            }
            committedMove = rootBestMove;
            committedScore = score;
            previousScore = score;
            if (printInfo) printInfo(d, score);

            // Track how many consecutive iterations the root best move has stayed the same.
            // A fresh move resets the streak to 1 (this iteration is its first sighting);
            // a repeat extends it.
            if (committedMove != 0 && committedMove == previousBestMove) {
                stableIterationsCount++;
            } else {
                stableIterationsCount = 1;
            }
            previousBestMove = committedMove;

            // Record this iteration's score and compute the recent swing (volatility signal B).
            recentScores[d % VOLATILITY_WINDOW] = committedScore;
            recentScoreCount++;
            int scoreSwing = 0;
            if (recentScoreCount >= VOLATILITY_WINDOW) {
                int lo = Integer.MAX_VALUE, hi = Integer.MIN_VALUE;
                for (int s : recentScores) { if (s < lo) lo = s; if (s > hi) hi = s; }
                scoreSwing = hi - lo;
            }
            // Volatility gate: a sharp committed move (promotion / advanced pawn push / check)
            // or an unsettled recent score means the shallow-but-stable read can't be trusted,
            // so the optimum-time soft exit is suppressed (search continues, still bounded by
            // softLimitMs and the hard ceiling).
            boolean volatilePos = useTime && useVolatilityGate
                    && (isBreakthroughMove(pos, committedMove) || scoreSwing > VOLATILITY_SWING_CP);

            // Evaluation-based time lock (see EVAL_LOCK_* constants): active only while the
            // position is still opening/early-middlegame AND dead-equal on the board, i.e.
            // exactly the profile of a quiet position with a not-yet-visible creeping
            // advantage. panicMode is excluded defensively even though SOFT_BOUND_MIN_DEPTH
            // (5) already structurally outranks PANIC_MAX_DEPTH (2) and can never let this
            // block run under panic -- hardLimitMs/moveOverheadMs stay the sole, unconditional
            // ceiling checkTime() enforces regardless of this flag either way.
            boolean evalLockActive = useTime && !panicMode
                    && gamePly < EVAL_LOCK_MAX_GAME_PLY
                    && Math.abs(committedScore) <= EVAL_LOCK_MAX_ABS_SCORE_CP;

            // Soft-bound (optimum time) stability early exit: only considered once the search
            // is deep enough (>= SOFT_BOUND_MIN_DEPTH) for the root move to be trustworthy, and
            // never while the evaluation-based time lock is active -- disabling this exit
            // outright there is the whole point of the lock (see EVAL_LOCK_* above).
            // Requires BOTH: (a) elapsed time already past the optimum-time target, and
            // (b) the root move has stopped changing for several iterations in a row -- a
            // depth confirming the same move again is unlikely to change the final decision,
            // so the remaining clock time is better saved than spent re-confirming it.
            if (evalLockActive || volatilePos) {
                // Explicitly clear rather than merely skip-setting: softStopArmed is a field
                // that persists across iterations, so a prior iteration (before the lock/
                // volatility engaged, or before committedScore dropped into the dead-equal
                // band) could otherwise leave a stale `true` behind that checkTime() would
                // still honor mid-iteration even though this iteration's block never re-armed
                // it. A volatile position must never take the optimum-time soft exit.
                softStopArmed = false;
            } else if (useTime && d >= SOFT_BOUND_MIN_DEPTH) {
                // Acceleration rule: a decisively winning score (> +2.00 pawns) needs only 2
                // stable iterations instead of 3 -- the position is unlikely to be spoiled by
                // one fewer confirmation, so we can bank the extra time even faster.
                int requiredStableIterations = committedScore > SOFT_BOUND_DECISIVE_SCORE_CP
                        ? SOFT_BOUND_STABLE_ITERATIONS_DECISIVE
                        : SOFT_BOUND_STABLE_ITERATIONS;
                // Arm the mid-iteration check (see checkTime()) as soon as stability is met,
                // regardless of whether elapsed time has crossed optimumTimeMs yet -- once
                // armed, the *next* iteration no longer needs to run to completion (or to
                // hardLimitMs) before the engine can react to the clock catching up.
                softStopArmed = stableIterationsCount >= requiredStableIterations;
                if (softStopArmed && elapsedMs() >= optimumTimeMs) {
                    stop = true;
                    // Propagate immediately instead of waiting for the caller to notice this
                    // worker finished -- e.g. Lazy SMP helper threads should stop racing the
                    // shared TT the moment the master commits to a move, not after it returns.
                    if (sharedStop != null) sharedStop.set(true);
                    if (printInfo) {
                        System.err.println("info string soft-bound early exit: depth=" + d
                                + " move=" + Move.toUci(committedMove)
                                + " stableIterations=" + stableIterationsCount
                                + " elapsedMs=" + elapsedMs()
                                + " optimumTimeMs=" + optimumTimeMs);
                    }
                    break;
                }
            }

            // Minimum-consumption floor: while the evaluation-based time lock is active, no
            // soft (non-mate, non-hard-limit) exit below is allowed to fire before at least
            // EVAL_LOCK_MIN_TIME_FRACTION of optimumTimeMs has actually elapsed, so a quiet
            // dead-equal opening position always gets a genuine chance to reveal a slow
            // positional edge instead of returning the instant *any* soft signal fires.
            boolean evalLockTimeSatisfied = !evalLockActive
                    || elapsedMs() >= (long) (optimumTimeMs * EVAL_LOCK_MIN_TIME_FRACTION);

            if (useTime && elapsedMs() >= softLimitMs && evalLockTimeSatisfied) break;
            if (Math.abs(score) >= MATE_IN_MAX) break; // forced mate found: never worth locking against
            // Only cut fixed-depth/analysis searches short when they're actually racing a
            // clock; otherwise "go depth N" would silently return a shallower result than asked.
            if (useTime && evalLockTimeSatisfied && hasResolvedRootGap(d)) break;
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

    /** Instant bypass: with exactly one legal reply, skip the search tree and play it. */
    private int bypassForcedMove(Position pos, int onlyMove) {
        bestMove = onlyMove;
        bestScore = evaluator.evaluate(pos);
        if (printInfo) {
            System.out.println("bestmove " + Move.toUci(onlyMove));
            System.out.flush();
            bypassPrinted = true;
        }
        return onlyMove;
    }

    /**
     * Soft-bound early exit: once {@code obviousMoveCheckDepth} completes, a root-move
     * score gap beyond {@code obviousMoveMargin} centipawns is treated as a resolved
     * position, and iterative deepening stops instead of spending time confirming it.
     */
    private boolean hasResolvedRootGap(int depth) {
        if (!useObviousMovePruning || depth < obviousMoveCheckDepth) return false;
        if (rootSecondBestScore <= -INF) return false; // fewer than two scored root moves
        return (rootBestScore - rootSecondBestScore) > obviousMoveMargin;
    }

    /**
     * True if {@code move} is a "breakthrough" whose consequences a shallow-but-stable search
     * can't be trusted to have seen: a promotion, an advanced pawn push (to the relative 5th
     * rank or beyond), or a checking move. Used by the volatility gate to suppress the
     * soft-bound early exit. Costs one make/unmake for the check test, run at most once per
     * completed root iteration.
     */
    private boolean isBreakthroughMove(Position pos, int move) {
        if (move == 0) return false;
        if (Move.isPromotion(move)) return true;
        if (Piece.type(pos.pieceAt(Move.from(move))) == Piece.PAWN) {
            int toRank = Move.to(move) >> 3;
            int rel = pos.sideToMove() == Piece.WHITE ? toRank : 7 - toRank;
            if (rel >= BREAKTHROUGH_MIN_REL_RANK) return true;
        }
        pos.makeMove(move);
        boolean givesCheck = pos.inCheck(pos.sideToMove());
        pos.unmakeMove(move);
        return givesCheck;
    }

    private int negamax(Position pos, int depth, int alpha, int beta, int ply, int extensions) {
        if (stopped()) return 0;
        if (ply > 0 && pos.isDrawByRuleOrRepetition()) return DRAW;
        // Hard leaf: without this, a check-extension chain (each check bumping depth back up
        // by 1, see below) can keep depth > 0 indefinitely and push ply past the MAX_PLY-sized
        // scratch arrays (killers, moveListPool, scorePool, quietsTriedPool) before this
        // function ever falls through to quiescence()'s own ply >= MAX_PLY guard.
        if (ply >= MAX_PLY) return evaluator.evaluate(pos);

        // A node is a PV node iff it was reached through a non-null (wide) window -- only the
        // very first move tried at any node recurses with the full window (see the i == 0
        // branch below), so a PV node's whole ancestor chain back to the root is itself PV
        // nodes. Non-PV ("scout") nodes are searched with a 1-wide null window purely to prove
        // "not better than alpha" as cheaply as possible; LMR is only safe to trust there,
        // since a PV node's true score still needs to be pinned down exactly.
        boolean isPvNode = beta - alpha > 1;

        boolean inCheck = pos.inCheck(pos.sideToMove());
        // Check extension: a forcing check is extended by a full ply so tactical lines don't
        // vanish past the iteration's nominal horizon just because the nominal depth ran out
        // mid-sequence. Capped at half of this iteration's root depth (currentDepth / 2) so a
        // long forcing/perpetual-check chain can't inflate a shallow iteration into an
        // effectively unbounded-depth search -- MAX_PLY (the ply guard above) is only the
        // last-resort backstop; without this budget the engine could burn its entire time
        // allotment on a single check-heavy line long before ever reaching it.
        if (useCheckExtension && inCheck && extensions < currentDepth / 2) {
            depth++;
            extensions++;
        }

        if (depth <= 0) {
            return useQuiescence ? quiescence(pos, alpha, beta, ply, QS_CHECK_BUDGET)
                                 : evaluator.evaluate(pos);
        }

        if ((++nodes & 2047) == 0) checkTime();

        long key = pos.zobristKey();
        int ttMove = 0;
        if (useTT) {
            long entry = tt.probe(key);
            int f = TranspositionTable.flagOf(entry);
            if (f != TranspositionTable.FLAG_NONE) {
                ttMove = TranspositionTable.moveOf(entry);
                if (ply > 0 && TranspositionTable.depthOf(entry) >= depth) {
                    int s = scoreFromTt(TranspositionTable.scoreOf(entry), ply);
                    if (f == TranspositionTable.FLAG_EXACT) return s;
                    // Fail-hard on TT-sourced bounds: a stored lower bound only proves the true
                    // score is >= s, so once s already clears beta, the safe cutoff value is
                    // beta itself (not the unverified s, which this shallower-or-equal-depth
                    // entry never re-confirmed against the *current* window). Symmetric for the
                    // upper-bound/alpha case.
                    if (f == TranspositionTable.FLAG_LOWER && s >= beta) return beta;
                    if (f == TranspositionTable.FLAG_UPPER && s <= alpha) return alpha;
                }
            }
        }

        // Reverse futility pruning: a node-level shortcut tried before move generation, since
        // it can skip generating/scoring moves entirely on a cutoff. Excluded near mate scores
        // (an eval-based margin can't be trusted to reason about forced mates) and at the root
        // (ply 0 must always return a real move).
        if (useRfp && !inCheck && ply > 0 && depth <= RFP_MAX_DEPTH && beta < MATE_IN_MAX) {
            int staticEval = evaluator.evaluate(pos);
            if (staticEval - RFP_MARGIN_PER_DEPTH * depth >= beta) {
                return staticEval;
            }
        }

        // Null-move pruning: also tried before move generation/scoring. Skipped while in
        // check (passing would leave our own king in check -- meaningless), at the root, at
        // low depth (too shallow a reduced re-search to trust), near mate scores, and in
        // king/pawn-only endgames (see hasNonPawnMaterial) where zugzwang makes "a free move
        // can't help the opponent" an unsound assumption. Also gated on the static eval
        // already meeting beta: if the position doesn't even look good enough on its face,
        // a reduced-depth verification search is unlikely to confirm a cutoff, so the eval
        // call is spent up front to skip the (much pricier) reduced search entirely.
        if (useNullMove && !inCheck && ply > 0 && depth >= NULL_MOVE_MIN_DEPTH
                && beta < MATE_IN_MAX && hasNonPawnMaterial(pos, pos.sideToMove())
                && evaluator.evaluate(pos) >= beta) {
            pos.makeNullMove();
            int nullScore = -negamax(pos, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, ply + 1, extensions);
            pos.unmakeNullMove();
            if (stopped()) return 0;
            if (nullScore >= beta) {
                return beta;
            }
        }

        MoveList moves = moveListPool[ply];
        moves.clear();
        MoveGenerator.generateLegal(pos, moves);
        if (moves.size == 0) {
            return inCheck ? -MATE + ply : DRAW; // checkmate or stalemate
        }

        int us = pos.sideToMove();
        int[] scores = scoreArray(ply, moves.size);
        scoreMoves(pos, moves, ttMove, ply, us, scores);

        // Futility pruning gate: computed once per node (not per move) so the move loop below
        // only pays for a static eval call when a low enough depth makes it useful. Excluded
        // at the root (ply > 0, matching RFP/NMP above): the root must always fully search
        // every candidate move, since skipping a late-ordered quiet move here on the strength
        // of a surface-level static eval can silently discard the true best defense in a
        // losing position -- exactly the move whose slower-mate score would otherwise have won
        // the root's alpha/bestScore comparisons below.
        boolean futilityGateOpen = useFutility && !inCheck && ply > 0 && depth <= FUTILITY_MAX_DEPTH
                && alpha > -MATE_IN_MAX;
        int futilityEval = futilityGateOpen ? evaluator.evaluate(pos) : 0;

        int bestScore = -INF;
        int localBest = 0;
        int origAlpha = alpha;
        // Tracks whether *this call* committed a root move via the alpha-gated path below --
        // deliberately local rather than checking rootBestMove == 0, since rootBestMove is an
        // instance field that can still hold a legitimately-set value from a *previous*
        // iteration/call; a zero-check there would wrongly treat this call as having already
        // committed and skip the fail-low fallback after the loop.
        boolean rootMoveCommitted = false;
        int[] quietsTried = quietsTriedPool[ply];
        int quietsTriedCount = 0;

        for (int i = 0; i < moves.size; i++) {
            selectNext(moves, scores, i);
            int move = moves.moves[i];
            boolean isCapture = Move.isCapture(move);
            boolean isPromotion = Move.isPromotion(move);
            boolean isQuiet = !isCapture && !isPromotion;
            pos.makeMove(move);

            // `givesCheck` is computed once and shared by both futility pruning and LMR below,
            // but only when at least one of them could actually use it (isQuiet, plus either
            // node-level futility being active or the move-level LMR thresholds already being
            // met) -- it costs a square-attack scan, so it's still skipped entirely for capture/
            // promotion moves and for quiet moves neither technique would touch anyway.
            // PV nodes (isPvNode, see above) are excluded from LMR entirely: reducing there
            // risks under-searching a move that could turn out to be the new principal
            // variation, whereas a reduced-then-verified score at a scout node only has to
            // clear a 1-wide window before the cheap re-search safety net (below) catches it.
            boolean lmrCandidate = useLmr && !isPvNode && !inCheck && depth > LMR_MIN_DEPTH
                    && i > LMR_MIN_MOVE_INDEX && isQuiet;
            boolean needsGivesCheck = isQuiet && (futilityGateOpen || lmrCandidate);
            boolean givesCheck = needsGivesCheck && pos.inCheck(pos.sideToMove());

            // Futility pruning: skip a quiet, non-checking move ordered after the first when
            // even the best-case static eval can't close the gap to alpha. Never applied to the
            // first move tried at a node (i == 0), so a node always searches at least one move
            // and can never wrongly return as though it had no legal moves.
            if (futilityGateOpen && i > 0 && isQuiet && !givesCheck
                    && futilityEval + FUTILITY_BASE_MARGIN + FUTILITY_MARGIN_PER_DEPTH * depth <= alpha) {
                pos.unmakeMove(move);
                continue;
            }

            // Record every quiet actually searched (not pruned away above) so that, if a
            // later move in this loop causes a cutoff, every quiet tried and rejected before
            // it can be identified for a history malus below.
            if (isQuiet) quietsTried[quietsTriedCount++] = move;

            int score;
            // Late Move Reduction qualification: only quiet moves (no capture, no promotion, no
            // check given), never when the side to move at this node is itself in check (every
            // legal move there is a forced evasion), and only ordered late (index > 4) once
            // remaining depth (> 2) leaves room for a meaningful reduction.
            boolean lmrEligible = lmrCandidate && !givesCheck;

            if (i == 0) {
                // Principal Variation Search: move ordering (TT move, then SEE-ranked
                // captures, then killers/history) makes the first move at every node the one
                // most likely to be best, so it alone gets the full alpha-beta window -- its
                // score becomes the alpha every later move in this loop only has to beat.
                score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, extensions);
            } else {
                // Every later move is assumed to be worse: probe it first with a minimal
                // (null/zero) window, (-alpha-1, -alpha). Proving "not better than alpha"
                // costs a fraction of a full-window search, so a correctly-ordered move list
                // turns most of these probes into cheap cutoffs instead of full re-explorations.
                int searchDepth = depth - 1;
                int reduction = 0;
                if (lmrEligible) {
                    // Reduction grows with how late/deep we are, but always leaves at least 1
                    // ply less than the normal (depth - 1) continuation. The Math.max(1, ...)
                    // floor only applies here -- unlike the un-reduced case, where depth - 1
                    // must be allowed to reach 0 so the recursive call falls through to
                    // quiescence instead of looping at depth 1 forever.
                    reduction = (depth >= LMR_DEEP_MIN_DEPTH && i >= LMR_DEEP_MIN_MOVE_INDEX) ? 2 : 1;
                    searchDepth = Math.max(1, depth - 1 - reduction);
                }
                score = -negamax(pos, searchDepth, -alpha - 1, -alpha, ply + 1, extensions);
                if (score > alpha && reduction > 0) {
                    // LMR re-search guardrail: a reduced-depth probe is not trustworthy enough
                    // to accept a fail-high -- the shallower search may simply have missed a
                    // refutation the reduction skipped over. Re-verify at the original,
                    // unreduced depth, still under the null window, before deciding whether
                    // this move earns the full PVS re-search below.
                    score = -negamax(pos, depth - 1, -alpha - 1, -alpha, ply + 1, extensions);
                }
                if (score > alpha && score < beta) {
                    // PVS re-search trigger: the null window failed high without reaching
                    // beta, so this move might be the new best move at this node. The value
                    // just returned is only a lower-bound proof (a null-window fail-high
                    // typically comes from an internal cutoff on the first move tried at that
                    // child, so the child's true value -- and thus this move's true score --
                    // could be higher still); the exact value can only be pinned down by
                    // re-opening the *full* original window, (-beta, -alpha), not a window
                    // narrowed to the unverified probe score.
                    score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, extensions);
                }
            }
            pos.unmakeMove(move);
            if (stopped()) return 0;

            if (ply == 0) updateRootMargin(score);

            if (score > bestScore) {
                bestScore = score;
                localBest = move;
                // Root best-move commitment is gated strictly on score > alpha (a real
                // alpha-raise), not merely score > bestScore -- otherwise the very first root
                // move tried would unconditionally claim rootBestMove (since bestScore starts
                // at -INF) even when its score never actually clears alpha, letting a move that
                // never beat anything overwrite whatever the *previous* iteration had legitimately
                // committed. localBest/bestScore keep tracking the running max regardless (for TT
                // storage and as the fail-low fallback below); only the field that think() actually
                // plays is held to the stricter bar.
                if (score > alpha) {
                    alpha = score;
                    if (ply == 0) {
                        rootBestMove = move;
                        rootMoveCommitted = true;
                    }
                    if (alpha >= beta) {
                        if (isQuiet) {
                            updateKillers(ply, move);
                            int bonus = Math.min(depth * depth, HISTORY_MAX);
                            updateHistory(us, move, bonus);
                            // Malus: every other quiet tried (and searched) at this node
                            // before the one that just caused the cutoff evidently didn't
                            // refute the position as well -- nudge it down by the same
                            // magnitude. Without this, a move that got a lucky cutoff at a
                            // shallow, unreliable depth keeps outranking moves later, deeper
                            // iterations show are actually better, which is exactly the kind
                            // of ordering instability that makes node counts swing wildly
                            // between similar positions once the TT move stops cutting off.
                            for (int qi = 0; qi < quietsTriedCount - 1; qi++) {
                                updateHistory(us, quietsTried[qi], -bonus);
                            }
                        }
                        break;
                    }
                }
            }
        }

        // Losing-fallback safety net: if every root move failed low against origAlpha (e.g. a
        // fully forced loss, where even the least-bad defense doesn't clear the aspiration
        // window), the strict score > alpha gate above never fires during this call and
        // rootMoveCommitted stays false -- rootBestMove would otherwise be left holding
        // whatever a *previous*, unrelated iteration last committed there. Fall back explicitly
        // to localBest -- the move that produced the highest score seen this call (the slowest
        // mate / smallest material loss), which is always non-null once at least one move was
        // searched (the first move at any node is never futility-pruned). In practice the
        // aspiration-window retry in think() re-searches with a widened alpha until this case
        // stops happening, but this fallback keeps the guarantee local to negamax itself rather
        // than relying entirely on that outer retry loop.
        if (ply == 0 && !rootMoveCommitted && localBest != 0) {
            rootBestMove = localBest;
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
        if (stopped()) return 0;
        if ((++nodes & 2047) == 0) checkTime();
        if (pos.isDrawByRuleOrRepetition()) return DRAW;

        boolean inCheck = pos.inCheck(pos.sideToMove());
        if (ply >= MAX_PLY) return evaluator.evaluate(pos);

        int bestScore;
        int standPat = 0;
        if (inCheck) {
            if (checkBudget <= 0) return evaluator.evaluate(pos);
            bestScore = -INF;
        } else {
            standPat = evaluator.evaluate(pos);
            bestScore = standPat;
            if (standPat >= beta) return standPat;
            if (standPat > alpha) alpha = standPat;
        }

        MoveList moves = moveListPool[ply];
        moves.clear();
        int nextBudget = checkBudget;
        if (inCheck) {
            MoveGenerator.generateLegal(pos, moves);
            nextBudget = checkBudget - 1;
            if (moves.size == 0) return -MATE + ply; // checkmate
        } else {
            MoveGenerator.generateLegalCaptures(pos, moves);
        }

        int[] scores = scoreArray(ply, moves.size);
        scoreMovesQ(pos, moves, scores);
        for (int i = 0; i < moves.size; i++) {
            selectNext(moves, scores, i);
            int move = moves.moves[i];
            if (!inCheck && !Move.isPromotion(move)) {
                int flag = Move.flag(move);
                int capturedType = flag == Move.EP_CAPTURE ? Piece.PAWN : Piece.type(pos.pieceAt(Move.to(move)));
                // Delta pruning: even the best case (winning the full captured piece, plus a
                // margin for positional gains the stand-pat eval misses) can't reach alpha.
                if (standPat + SEE_VALUE[capturedType] + DELTA_MARGIN <= alpha) continue;
                // SEE pruning: scores[i] is this capture's full static-exchange result (see
                // scoreMovesQ) -- a capture that provably loses material after the whole
                // recapture sequence (e.g. a bishop taking a pawn defended by a pawn) can
                // never be a real tactical threat, so it's discarded outright instead of
                // being searched just to prove what SEE already established.
                if (scores[i] < 0) continue;
            }
            pos.makeMove(move);
            int score = -quiescence(pos, -beta, -alpha, ply + 1, nextBudget);
            pos.unmakeMove(move);
            if (stopped()) return 0;
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

    /** Returns the pooled score buffer for {@code ply}, growing it only in the (never-hit
     *  in legal chess) case where a move list exceeds the pool's default capacity. */
    private int[] scoreArray(int ply, int minSize) {
        int[] s = scorePool[ply];
        if (s.length < minSize) {
            s = new int[minSize];
            scorePool[ply] = s;
        }
        return s;
    }

    private void scoreMoves(Position pos, MoveList moves, int ttMove, int ply, int us, int[] scores) {
        for (int i = 0; i < moves.size; i++) {
            scores[i] = scoreMove(pos, moves.moves[i], ttMove, ply, us);
        }
    }

    private int scoreMove(Position pos, int move, int ttMove, int ply, int us) {
        if (move == ttMove) return TT_SCORE;
        int flag = Move.flag(move);
        boolean capture = (flag & 4) != 0;
        boolean promo = (flag & 8) != 0;
        if (capture) {
            // SEE replaces raw victim/attacker MVV-LVA: it plays out the full recapture
            // sequence on the target square, so a capture that just loses material to a
            // recapture sorts below quiet/killer moves instead of above them.
            int seeValue = see(pos, move);
            int score = seeValue >= 0 ? CAPTURE_BASE + seeValue : BAD_CAPTURE_BASE + seeValue;
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

    private void scoreMovesQ(Position pos, MoveList moves, int[] scores) {
        for (int i = 0; i < moves.size; i++) {
            int move = moves.moves[i];
            int flag = Move.flag(move);
            int score = 0;
            if ((flag & 4) != 0) {
                score = see(pos, move);
            }
            if ((flag & 8) != 0) score += 100;
            scores[i] = score;
        }
    }

    /**
     * Static Exchange Evaluation: the net centipawn result of playing out the full
     * capture sequence on {@code move}'s destination square, assuming both sides always
     * recapture with their least valuable attacker, and each side stops recapturing the
     * moment doing so would lose material. Unlike MVV-LVA (which only looks at the first
     * capture), this tells apart a capture that wins material from one that hangs a piece
     * to a defended square, so losing captures can be sorted below quiet/killer moves
     * instead of above them.
     */
    int see(Position pos, int move) {
        int flag = Move.flag(move);
        if ((flag & 4) == 0) return 0; // not a capture: nothing to exchange

        int to = Move.to(move);
        int from = Move.from(move);
        int us = pos.sideToMove();
        int them = 1 - us;

        long[] byType = seeByType;
        for (int i = 0; i < 12; i++) byType[i] = pos.pieces(i);
        long occ = pos.occupied();

        int capturedType = flag == Move.EP_CAPTURE ? Piece.PAWN : Piece.type(pos.pieceAt(to));
        int originalType = Piece.type(pos.pieceAt(from));
        // The attacker lands on `to` as the promoted piece, not a pawn -- if the exchange
        // continues, that's the value a recapture actually wins. The board bitboards still
        // hold it as a pawn until makeMove() actually runs, though, so the bitboard removed
        // below must use originalType (PAWN), not the promoted attackerType.
        int attackerType = Move.isPromotion(move) ? Move.promoType(move) : originalType;

        // Take the initial attacker off the board; it's "in flight" onto `to`, and a
        // square's occupant never blocks attacks landing on that same square.
        long fromBit = 1L << from;
        occ &= ~fromBit;
        byType[Piece.index(us, originalType)] &= ~fromBit;
        if (flag == Move.EP_CAPTURE) {
            int capSq = us == Piece.WHITE ? to - 8 : to + 8;
            long capBit = 1L << capSq;
            occ &= ~capBit;
            byType[Piece.index(them, Piece.PAWN)] &= ~capBit;
        }

        int[] gain = seeGain;
        int depth = 0;
        gain[0] = SEE_VALUE[capturedType];
        int side = them;

        while (true) {
            long sideAttackers = attackersTo(to, side, occ, byType);
            if (sideAttackers == 0) break;

            int type = Piece.PAWN;
            long bit = 0L;
            for (; type <= Piece.KING; type++) {
                bit = sideAttackers & byType[Piece.index(side, type)];
                if (bit != 0) break;
            }
            long thisBit = Long.lowestOneBit(bit);

            depth++;
            gain[depth] = SEE_VALUE[attackerType] - gain[depth - 1];
            // Both sides play optimally: if continuing the exchange can't improve on
            // stopping now, cut the simulation short instead of running it to the end.
            if (Math.max(-gain[depth - 1], gain[depth]) < 0) break;

            occ &= ~thisBit;
            byType[Piece.index(side, type)] &= ~thisBit;
            attackerType = type;
            side = 1 - side;
        }

        for (int d = depth; d > 0; d--) {
            gain[d - 1] = -Math.max(-gain[d - 1], gain[d]);
        }
        return gain[0];
    }

    /** Bitboard of every {@code side} piece attacking {@code sq}, given occupancy {@code occ}
     *  and per-piece-type bitboards {@code byType} (both possibly mid-simulation, see {@link #see}). */
    private long attackersTo(int sq, int side, long occ, long[] byType) {
        long att = Attacks.PAWN[1 - side][sq] & byType[Piece.index(side, Piece.PAWN)];
        att |= Attacks.KNIGHT[sq] & byType[Piece.index(side, Piece.KNIGHT)];
        att |= Attacks.KING[sq] & byType[Piece.index(side, Piece.KING)];
        long bishopsQueens = byType[Piece.index(side, Piece.BISHOP)] | byType[Piece.index(side, Piece.QUEEN)];
        att |= Attacks.bishop(sq, occ) & bishopsQueens;
        long rooksQueens = byType[Piece.index(side, Piece.ROOK)] | byType[Piece.index(side, Piece.QUEEN)];
        att |= Attacks.rook(sq, occ) & rooksQueens;
        return att;
    }

    /** True if {@code side} has any piece besides pawns and king -- null-move pruning's
     *  "a free move can't help the opponent" reasoning is unsound without this, since
     *  king/pawn-only endgames are exactly where zugzwang (passing would be losing, if the
     *  rules allowed it) is common. */
    private boolean hasNonPawnMaterial(Position pos, int side) {
        return (pos.pieces(Piece.index(side, Piece.KNIGHT))
                | pos.pieces(Piece.index(side, Piece.BISHOP))
                | pos.pieces(Piece.index(side, Piece.ROOK))
                | pos.pieces(Piece.index(side, Piece.QUEEN))) != 0;
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

    /** Tracks the best and second-best root-move scores seen this iteration (ply 0 only). */
    private void updateRootMargin(int score) {
        if (score > rootBestScore) {
            rootSecondBestScore = rootBestScore;
            rootBestScore = score;
        } else if (score > rootSecondBestScore) {
            rootSecondBestScore = score;
        }
    }

    private void updateKillers(int ply, int move) {
        if (killers[ply][0] != move) {
            killers[ply][1] = killers[ply][0];
            killers[ply][0] = move;
        }
    }

    /**
     * Applies a signed {@code bonus} to a quiet move's history cell using the standard
     * "history gravity" formula: the update shrinks as the current value approaches
     * +-HISTORY_MAX (in the direction of the bonus), self-limiting the table without a hard
     * clamp, and an opposite-signed bonus pulls an extreme value back toward zero faster than
     * a same-signed one would push it further. This is also the table's decay/aging
     * mechanism -- a move whose score was inflated by stale, shallow-iteration cutoffs gets
     * pulled back down as soon as a fresh negative signal (a malus, see the move loop in
     * negamax()) arrives, instead of requiring a separate periodic reset or halving pass.
     */
    private void updateHistory(int us, int move, int bonus) {
        int from = Move.from(move), to = Move.to(move);
        int current = history[us][from][to];
        history[us][from][to] = current + bonus - current * Math.abs(bonus) / HISTORY_MAX;
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

    void computeTimeLimits(SearchLimits limits, int side) {
        useTime = false;
        panicMode = false;
        softLimitMs = Long.MAX_VALUE;
        hardLimitMs = Long.MAX_VALUE;
        optimumTimeMs = Long.MAX_VALUE;
        if (limits.infinite) return;
        if (limits.movetime > 0) {
            long t = Math.max(1, limits.movetime - TIME_OVERHEAD_MS);
            softLimitMs = t;
            hardLimitMs = t;
            useTime = true;
            return; // fixed "go movetime" has no remaining-clock concept to derive an optimum bound from
        }
        if (limits.depth > 0 && !limits.hasClock()) return; // pure depth-limited

        int time = side == Piece.WHITE ? limits.wtime : limits.btime;
        int inc = side == Piece.WHITE ? limits.winc : limits.binc;
        if (time <= 0) return; // no usable clock -> depth/infinite

        // Move Overhead deduction: subtract the configured lag buffer from the raw clock
        // *before* any division/allocation math runs, so every bound below is computed
        // against time we can actually spend thinking, not time already effectively spent
        // on network transit or wrapper dispatch (e.g. lichess-bot's Python process).
        long playableTime = (long) time - moveOverheadMs;

        if (playableTime <= PANIC_THRESHOLD_MS) {
            // Panic mode: lag has already consumed (almost) the entire clock. The normal
            // soft/hard allocation math below could still compute a limit long enough to
            // flag the game on time, so skip it entirely and force a structural emergency
            // stop: a handful of milliseconds of search and a hard-capped depth (see the
            // panicMode check against PANIC_MAX_DEPTH in think()), so the engine always has
            // time to commit whatever move it already found.
            hardLimitMs = PANIC_HARD_LIMIT_MS;
            softLimitMs = PANIC_HARD_LIMIT_MS; // keep soft <= hard so the existing ceiling checks stay coherent
            optimumTimeMs = PANIC_OPTIMUM_TIME_MS;
            panicMode = true;
            useTime = true;
            return;
        }

        // TIME_OVERHEAD_MS is a separate, smaller safety margin for this engine's own UCI
        // command turnaround; it stacks on top of (rather than replaces) the moveOverheadMs
        // deduction already folded into playableTime above.
        long cap = Math.max(1, playableTime - TIME_OVERHEAD_MS);

        int mtg = limits.movestogo > 0 ? limits.movestogo : 30;
        long soft = playableTime / mtg + (long) inc * 3 / 4;
        soft = Math.min(soft, cap);
        if (soft < 1) soft = 1;

        // Optimum time (the best-move-stability early-exit target in think()): a fraction of
        // the REAL per-move budget `soft`, so banking scales with what the engine actually
        // intends to spend. Also bounds the absolute hard ceiling (see
        // HARD_LIMIT_OPTIMUM_MULTIPLIER below). Computed before `hard` so the latter can be
        // bounded relative to it.
        long optimum = (long) (soft * OPTIMUM_TIME_FRACTION);
        optimum = Math.min(optimum, cap);
        if (optimum < 1) optimum = 1;

        long hard = Math.min(playableTime / 4, soft * 4);
        // Absolute ceiling: checkTime() enforces hardLimitMs unconditionally inside the
        // recursive search (every 2048 nodes), but never checks optimumTimeMs mid-search --
        // that's only consulted between completed iterations. Without this cap, hard (derived
        // solely from playableTime/4) can run to 10x+ optimum early in a game when the clock
        // is nearly full, letting one exploding ply legally run tens of seconds past the
        // intended per-move budget before the soft-bound logic ever gets a chance to react.
        hard = Math.min(hard, (long) (optimum * HARD_LIMIT_OPTIMUM_MULTIPLIER));
        hard = Math.min(hard, cap);
        if (hard < soft) hard = soft; // hard must never be tighter than soft (see checkTime()/softLimitMs roles)

        softLimitMs = soft;
        hardLimitMs = hard;
        optimum = Math.min(optimum, hardLimitMs); // never a "stop early" target beyond the real ceiling
        optimumTimeMs = optimum;
        useTime = true;
    }

    private void checkTime() {
        if (!useTime || currentDepth < 2) return;
        long elapsed = elapsedMs();
        if (elapsed >= hardLimitMs) {
            stop = true;
            if (sharedStop != null) sharedStop.set(true);
            return;
        }
        // Mid-iteration soft-bound: softStopArmed means a prior *completed* iteration already
        // met the stability requirement (see think()'s soft-bound block). Without this, only
        // the hardLimitMs check above can end an overrunning iteration -- e.g. an aspiration
        // re-search cascade -- letting it burn all the way up to hardLimitMs (up to
        // HARD_LIMIT_OPTIMUM_MULTIPLIER x optimumTimeMs) even though the position was already
        // settled before that iteration even started. Checking it here too means the search
        // reacts within one checkTime() interval (every 2048 nodes) of crossing optimumTimeMs,
        // not only between fully-completed iterations.
        if (softStopArmed && elapsed >= optimumTimeMs) {
            stop = true;
            softStopFiredMidIteration = true;
            if (sharedStop != null) sharedStop.set(true);
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
