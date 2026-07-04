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
    // Log-based reduction table: LMR_TABLE[depth][moveIndex] = floor(0.75 + ln(d)*ln(m)/2.25),
    // the standard smoothly-growing reduction that replaces the old binary 1/2 rule -- more
    // reduction the later and deeper a quiet move sits, with fine-grained steps in between.
    private static final int[][] LMR_TABLE = new int[64][64];
    static {
        for (int d = 1; d < 64; d++) {
            for (int m = 1; m < 64; m++) {
                LMR_TABLE[d][m] = (int) Math.floor(0.75 + Math.log(d) * Math.log(m) / 2.25);
            }
        }
    }

    // Late move pruning: at shallow non-PV nodes, once this many quiet moves have been tried
    // the rest are skipped outright (quadratic-in-depth budget, larger when "improving").
    private static final int LMP_MAX_DEPTH = 8;
    // Floor on the LMP quiet-move budget so very shallow nodes (depth 1-3, where the quadratic
    // term is tiny) don't prune after only 2-3 quiets and risk discarding a late-ordered but
    // genuinely good positional move.
    private static final int LMP_MIN_QUIETS = 4;
    // SEE pruning in the main search: at shallow non-PV nodes, a capture that loses more than
    // SEE_PRUNE_MARGIN * depth centipawns by static exchange is skipped without being searched.
    private static final int SEE_PRUNE_MAX_DEPTH = 8;
    private static final int SEE_PRUNE_MARGIN = 100;
    // Internal iterative reductions: with no TT move to guide ordering at a node deep enough to
    // matter, search one ply shallower rather than exploring an unordered subtree at full depth.
    private static final int IIR_MIN_DEPTH = 4;
    // Countermove ordering tier: below KILLER1 (800k) and above the quiet-history range.
    private static final int COUNTERMOVE = 700_000;

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
    // Depth from which a null-move fail-high is re-verified by a reduced, null-move-disabled
    // search before it is trusted -- catches the zugzwang-style positions where passing looks
    // fine but every real move is bad, which the static-eval and non-pawn-material gates alone
    // don't fully exclude at high depth (where a wrong cutoff is most expensive).
    private static final int NMP_VERIFY_DEPTH = 10;

    // Singular extensions: when a TT move from a trustworthy entry (depth within
    // SINGULAR_TT_DEPTH_MARGIN of the current one, lower-bound or exact) is re-verified by a
    // reduced-depth search of every OTHER move against a margin below the TT score and nothing
    // comes close, the TT move is the position's only real try -- extend it a ply so forced
    // sequences aren't cut off at the horizon. The same exclusion search doubles as multicut:
    // if even without the TT move the reduced search beats beta, two independent moves fail
    // high and the whole node can be cut immediately.
    private static final int SINGULAR_MIN_DEPTH = 8;
    private static final int SINGULAR_TT_DEPTH_MARGIN = 3; // ttDepth >= depth - this
    private static final int SINGULAR_MARGIN_PER_DEPTH = 2; // singularBeta = ttScore - 2*depth
    // Bundle D search additions (all behind toggles, default OFF, gated together at -movetime 100).
    private static final int RAZOR_MAX_DEPTH = 3;
    private static final int RAZOR_MARGIN_BASE = 300;
    private static final int RAZOR_MARGIN_PER_DEPTH = 200; // margin = BASE + PER_DEPTH*(depth-1)
    private static final int PROBCUT_MIN_DEPTH = 5;
    private static final int PROBCUT_MARGIN = 190;   // probCutBeta = beta + this
    private static final int PROBCUT_REDUCTION = 4;  // verified capture re-searched at depth - this
    private static final int SINGULAR_DOUBLE_MARGIN = 25; // double-extend when value < singularBeta - this

    // Futility pruning: near the leaves, a quiet move that can't plausibly close the gap to
    // alpha (staticEval + a depth-scaled margin still falls short) is skipped without being
    // searched -- the same reasoning as quiescence's delta pruning, one ply higher up. The
    // very first move tried at a node is never skipped (see the move loop), so a node can
    // never return with zero moves actually searched.
    private static final int FUTILITY_MAX_DEPTH = 6;
    private static final int FUTILITY_BASE_MARGIN = 100;
    private static final int FUTILITY_MARGIN_PER_DEPTH = 80;

    // History pruning: at shallow non-PV nodes, a quiet move whose combined ordering score
    // (butterfly history + both continuation-history tables, see scoreMove) is deeply
    // negative has repeatedly failed to refute anything in this exact context -- skip it
    // without a search. The margin scales with depth so only ever-worse offenders are pruned
    // as more search would otherwise be spent on them. The per-depth margin is expressed as
    // a fraction of HISTORY_MAX because the combined score spans +-3*HISTORY_MAX: an early
    // version used a flat -2048 (borrowed from engines whose history tables span ~+-16k),
    // which on this scale pruned any quiet with even a mild recent malus and measurably
    // lost self-play games (0/2 vs baseline; 50% with the pruning disabled).
    private static final int HISTORY_PRUNE_MAX_DEPTH = 4;
    private static final int HISTORY_PRUNE_MARGIN_PER_DEPTH = -12_500; // = -(HISTORY_MAX / 8)

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

    // Adaptive time management (useAdaptiveTime; default OFF pending a self-play gate). Replaces
    // the binary stability threshold with a *continuous* stop-time scale and adds a game-ply
    // moves-to-go estimate plus an eval-trend factor. All constants only take effect when the
    // toggle is on, so the shipped default path is byte-identical to the tuned legacy behaviour.
    private static final int ADAPTIVE_MTG_MIN = 20;   // sudden-death moves-to-go floor
    private static final int ADAPTIVE_MTG_BASE = 40;  // mtg = clamp(BASE - gamePly/2, MIN, BASE)
    private static final int ADAPTIVE_MIN_STABLE = 4; // arm the continuous exit past this streak
    private static final int ADAPTIVE_STABILITY_CAP = 8;
    private static final double ADAPTIVE_STABILITY_BASE = 1.40;  // stopTarget = optimum * (BASE - SLOPE*stab)
    private static final double ADAPTIVE_STABILITY_SLOPE = 0.09; // 1.40 (fresh) -> 0.68 (very stable)
    private static final double ADAPTIVE_TREND_SLOPE = 0.004;    // per cp the eval dropped since last iter
    private static final double ADAPTIVE_TREND_MIN = 0.80;       // rising eval banks time
    private static final double ADAPTIVE_TREND_MAX = 1.30;       // falling eval buys time
    private static final double ADAPTIVE_FACTOR_MIN = 0.50;      // clamp the stability*trend product
    private static final double ADAPTIVE_FACTOR_MAX = 2.00;
    // Node-effort factor (the third adaptive signal, standard in modern engines): the fraction
    // of ALL search nodes this think() spent under the current best root move. Near 1.0 the
    // move dominates the tree (settled -- stop sooner); spread-out effort means the root is
    // still genuinely contested (spend longer). Folded into the stability*trend product above,
    // inside the same [FACTOR_MIN, FACTOR_MAX] safety clamp.
    private static final double ADAPTIVE_EFFORT_BASE = 1.60;  // effort = clamp(BASE - SLOPE*bestFrac, ...)
    private static final double ADAPTIVE_EFFORT_SLOPE = 1.10; // bestFrac 1.0 -> 0.50 raw -> clamped 0.60
    private static final double ADAPTIVE_EFFORT_MIN = 0.60;
    private static final double ADAPTIVE_EFFORT_MAX = 1.50;
    // Mate-hunt band: a crushing but not-yet-mate score (>= this) means a forced mate is very
    // likely just past the horizon, so the decisive-score acceleration above (which would bank
    // the clock after only 2 stable iterations) is suppressed and the full soft budget is spent
    // hunting for the actual mate. Set well above SOFT_BOUND_DECISIVE_SCORE_CP so ordinary
    // winning positions still bank time; only genuinely won ones trigger the hunt. Observed
    // motivation: a game where the eval ground up to +21 across several moves (exiting ~depth
    // 15 each time) before a forced mate was ever reported, because the +2.00 acceleration kept
    // banking time instead of searching deep enough to see the mate.
    private static final int MATE_HUNT_SCORE_CP = 800; // +8.00 pawns

    // Volatility gate: the soft-bound early exit banks time when the root move has been stable,
    // but stability alone is blind to *sharp* positions -- a forcing pawn breakthrough (e.g.
    // ...d3+) can sit at a mild static eval for many stable iterations while the real damage
    // lurks a few plies deeper. When the committed move is itself a breakthrough (promotion,
    // advanced pawn push, or a check) OR the recent scores are swinging, the optimum-time soft
    // exit is suppressed so the search keeps going (still bounded by softLimitMs and the
    // lost-position brake). BREAKTHROUGH_MIN_REL_RANK 4 = a push to the relative 5th rank+.
    private static final int VOLATILITY_SWING_CP = 30;
    private static final int VOLATILITY_WINDOW = 3;
    private static final int BREAKTHROUGH_MIN_REL_RANK = 4;

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
    static final double HARD_LIMIT_OPTIMUM_MULTIPLIER = 3.0;

    // Lost-position time brake: the missing symmetric counterpart to the EVAL_LOCK "gas pedal"
    // that spends extra time on dead-equal positions. Once the score has been clearly losing
    // for a couple of completed iterations, no realistic amount of extra search reverses it,
    // so the engine stops pouring clock into it and banks the time for positions where a
    // defense actually exists -- capping the move near optimum instead of the multiple-of-
    // optimum hard ceiling. A found forced mate (|score| >= MATE_IN_MAX) is excluded: those
    // have their own "stop as soon as found" handling.
    private static final int LOST_SCORE_THRESHOLD_CP = -400; // ~ -4.00 pawns
    private static final int LOST_MIN_ITERATIONS = 2;        // consecutive losing iterations required
    private static final double LOST_TIME_FRACTION = 0.5;    // effective soft limit while lost

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
    // Countermove table: counterMoves[prevFrom][prevTo] = the quiet move that most recently
    // produced a beta cutoff as the reply to that previous move. Cleared only in newGame().
    private final int[][] counterMoves = new int[64][64];
    // Continuation history: how well a quiet move (keyed by its piece-and-destination,
    // pieceIndex*64+to) has refuted positions reached by a specific PREVIOUS piece-and-
    // destination -- contHist1 keys on the move one ply back (countermove history), contHist2
    // on the move two plies back (follow-up history: "after my Nf3 plan, ...Re8 works").
    // Where the butterfly `history` table sees only from/to squares averaged over every
    // context, these tables make quiet-move ordering context-sensitive, which is the largest
    // single ordering signal a classical engine adds after killers/history. Same gravity
    // update/decay as `history` (see gravity()), same lifetime (cleared only in newGame()).
    // ~2.4MB per table per Search instance -- allocated once here, never inside the search.
    private final int[][] contHist1 = new int[12 * 64][12 * 64];
    private final int[][] contHist2 = new int[12 * 64][12 * 64];
    // Per-ply piece-and-destination (pieceIndex*64+to) of the move made at that ply, the
    // continuation-history key for descendants; -1 after a null move (no continuation).
    private final int[] pieceToStack = new int[MAX_PLY + 2];
    // Correction history: the evaluator is systematically wrong in specific pawn structures
    // (observed live: a near-flat eval for ten moves of an advancing pawn storm that was in
    // fact losing), and every eval-trusting pruning decision -- RFP, futility, the improving
    // signal, the NMP eval gate -- inherits that bias. This table accumulates, per side to
    // move and pawn-structure hash, a moving average of (search result - static eval) from
    // completed non-mate nodes whose bound actually constrains the true score, and future
    // static evals are nudged by it BEFORE the pruning stack consumes them. Fixed-point
    // (CORR_GRAIN per centipawn), clamped to +-32cp of correction; per-instance (one per
    // search thread, isolation like contHist), cleared only in newGame(). Zero allocation:
    // sized once here, indexed by a 2-multiply mix of the pawn bitboards.
    private static final int CORR_SIZE = 16384;      // power-of-two entries per side
    private static final int CORR_GRAIN = 256;       // fixed-point units per centipawn
    private static final int CORR_LIMIT = 32 * CORR_GRAIN; // max correction: +-32cp
    private static final int CORR_WEIGHT_MAX = 16;   // deeper results move the average harder
    private static final int CORR_WEIGHT_SCALE = 64;
    private final int[][] corrHist = new int[2][CORR_SIZE];
    // Per-ply static-eval stack: records each node's static eval (or a sentinel when in
    // check), so a node can cheaply ask whether the side to move is "improving" relative to
    // the same side's eval two plies earlier -- consumed by depth-scaled pruning heuristics.
    private final int[] evalStack = new int[MAX_PLY + 2];
    // Triangular principal-variation table: pvTable[ply] holds the best line found from that
    // ply onward (moves at indices [ply, pvLength[ply])), assembled bottom-up by copying each
    // child's PV onto the move that raised alpha. pvTable[0] is the full mainline the engine
    // actually intends to play -- what "info ... pv" reports, instead of only the root move.
    private final int[][] pvTable = new int[MAX_PLY + 1][MAX_PLY + 1];
    private final int[] pvLength = new int[MAX_PLY + 1];
    // Per-ply record of the move played to reach the child at ply+1 (0 == a null move), so a
    // node can consult the move that led to it (moveStack[ply-1]) -- used to forbid two null
    // moves in a row, and (later) to key the countermove heuristic.
    private final int[] moveStack = new int[MAX_PLY + 2];
    // TT generation (aging): bumped once per search so stores from prior searches can be
    // recognized as stale and preferentially overwritten even in the depth-preferred slot.
    private int ttGeneration;

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
    // Captures actually searched (past SEE pruning) at the current node, mirroring
    // quietsTriedPool, so a capture beta cutoff can malus the captures tried before it.
    private final int[][] capturesTriedPool = new int[MAX_PLY + 2][256];
    // Capture history: [movingPiece12][toSquare][capturedType] gravity table refining the
    // ordering AMONG captures (SEE still decides the winning/losing split and the coarse
    // band; this breaks ties, which SEE produces constantly). Same gravity update/decay and
    // lifetime as the quiet histories; per-instance, ~18KB, allocated once. Contribution is
    // scaled down (CAPHIST_ORDER_DIV) so it stays SUBORDINATE to the SEE value inside the
    // band: SEE differences within the winning-capture band are on the order of one piece
    // value (~100-900 centipawns), so the history contribution must stay well under ~100 or
    // it stops being a tie-break and starts overriding material order outright. An earlier
    // divisor of 8 gave +-12,500 -- history could sort "capture the queen" below "capture a
    // pawn" on a modest malus, corrupting capture ordering wholesale (the same scale-mismatch
    // class of bug as the -2048 history-pruning margin incident, from the other direction).
    private final int[][][] capHist = new int[12][64][6];
    private static final int CAPHIST_ORDER_DIV = 2048; // max ordering effect +-HISTORY_MAX/2048 ~= +-48
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
    // Per-root-move node-effort tally (adaptive time only; see ADAPTIVE_EFFORT_*). Keyed by the
    // move integer -- NOT by move-list index, because selectNext() swap-reorders the root move
    // array in place during iteration. Accumulated across the whole think() call (every depth
    // iteration and aspiration re-search); linear scan is fine at root branching factors.
    // MoveList grows past 256 in theory, but 256 exceeds the legal-move maximum (218).
    private final int[] rootEffortMoves = new int[256];
    private final long[] rootEffortNodes = new long[256];
    private int rootEffortCount;
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
    // Default kept small (100ms): the lichess-bot wrapper already applies its own
    // move_overhead on top of this, and stacking a large value here starved bullet searches.
    public long moveOverheadMs = 100;

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
    // Ponder-move hint: the 2nd PV move captured from the LAST COMPLETED iteration (pvTable is
    // reset on entry to each iteration and left empty if that iteration is aborted by time, so
    // it can't be read reliably after think() returns -- it's captured in the loop instead).
    private int bestPonderMove;

    /** The move we expect the opponent to reply with, for the UCI "bestmove &lt;best&gt;
     *  ponder &lt;this&gt;" hint. 0 if no reliable continuation was found. */
    public int ponderMove() { return bestPonderMove; }

    public boolean printInfo = true;
    public boolean useTT = true;
    public boolean useQuiescence = true;
    public boolean useCheckExtension = true;
    public boolean useLmr = true;
    public boolean useRfp = true;
    public boolean useNullMove = true;
    public boolean useFutility = true;
    // When true, quiescence generates full check evasions at every in-check node (bounded only
    // by MAX_PLY) instead of static-eval'ing a meaningless in-check position once a small check
    // budget is spent. Fixes tactical blindness in forcing lines; the old budget path remains
    // as the A/B fallback (useQsEvasions = false).
    public boolean useQsEvasions = true;
    // Stage D node-reduction levers, each independently switchable for A/B testing.
    public boolean useLmp = true;          // late move (move-count) pruning
    public boolean useSeePruning = true;   // skip provably-losing captures at shallow depth
    public boolean useIir = true;          // internal iterative reductions (no TT move -> depth-1)
    public boolean useCounterMove = true;  // countermove ordering heuristic
    public boolean useContHist = true;     // 1-/2-ply continuation-history quiet ordering
    // Default OFF (data-driven): correctly implemented and equivalence-tested, but the
    // unbiased referee gate measured the batch at 47.8% over 200 games at 100ms/move even
    // after the propagation/scale bug fixes -- no evidence of gain at fast TC. Candidates to
    // re-enable if a longer-TC retest or a post-Texel-tuning run shows a positive score.
    public boolean useCaptureHistory = false; // capture-history ordering refinement
    public boolean useCutnodeLmr = false;   // reduce one extra ply at expected cut-nodes
    public boolean useSingular = true;     // singular extensions + multicut (needs the TT)
    // Bundle D: gated ON (2026-07-04) after a 600-game referee gate at -movetime 100 scored
    // 51.7% (+168 -147 =285, Elo +12 [-16,40]) -- positive lean, standard techniques, bounded
    // downside. useCutnodeLmr stays OFF (prior batch-2 negative evidence; separate retest).
    public boolean useRazoring = true;        // shallow non-PV: drop to qsearch when eval << alpha
    public boolean useProbcut = true;         // depth>=5: a good capture that survives a raised-beta qsearch cuts
    public boolean useSingularExtDouble = true; // double / negative singular extensions
    public boolean useLmrTtCapture = true;    // reduce one extra ply when the TT move is a capture
    public boolean useHistoryPruning = true; // skip quiets with deeply negative history at shallow depth
    public boolean useQsTt = true;         // quiescence transposition-table probe/store
    public boolean useVolatilityGate = true; // suppress the soft early exit in sharp positions
    public boolean useMateHunt = true;     // keep searching (don't bank time) when winning big but no mate yet
    public boolean useShortestMate = true; // don't commit a winning mate until deep enough to play the FASTEST one

    // Contempt (centipawns): how much the ROOT side dislikes a draw. A positive value shades
    // in-search draw returns so the engine avoids repetitions / 50-move draws when it isn't
    // worse -- exactly the "+1.0 shuffled into a threefold" leak seen vs weaker bots. 0 = pure
    // objective play. Set via the UCI "Contempt" option; applied by drawScore().
    public int contempt = 10;

    // Pondering: while true, the search ignores its time budget entirely (it was launched on
    // the opponent's clock via UCI "go ponder"), so no time-based or found-mate exit fires --
    // only an external stop ends it. ponderHit() flips it off AND rebases the clock to now, so
    // the normal soft/hard/mate-hunt logic then counts the move budget from the moment the
    // opponent actually played the predicted move. Volatile: set on the UCI input thread,
    // read on the search thread.
    private volatile boolean pondering;

    /** UCI "ponderhit": the pondered move was played. Rebase the clock to now and let the
     *  normal time management take over the still-running search. */
    public void ponderHit() {
        startNanos = System.nanoTime();
        pondering = false;
    }

    /** Arms ponder mode for the next {@link #think}; cleared by {@link #ponderHit} or a stop. */
    public void setPondering(boolean p) { pondering = p; }
    // Default OFF (data-driven): shipped enabled earlier on a biased-harness 56.9%, but a clean
    // unbiased referee re-gate scored corrHist-on at 48.2%/300 games (Elo -13, CI [-52,27]) --
    // no evidence of gain at 100ms/move. Theoretically sound (standard in strong engines) and
    // may pay off at slower TC or after eval tuning, so the code stays as a retest candidate.
    public boolean useCorrectionHistory = false; // pawn-structure-keyed static-eval correction

    // Adaptive time management: game-ply moves-to-go estimate + continuous stability/eval-trend
    // stop-time scaling (see ADAPTIVE_* constants). Default OFF: it reshapes the delicate, tuned
    // soft-bound logic, so it ships dormant until an unbiased referee gate (e.g. -tc 10+0.1 and
    // 60+0.6) confirms a gain. Flip to true in a branch build and referee it against main.
    public boolean useAdaptiveTime = false;

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
        rootEffortCount = 0; // per-think() node-effort tally (see rootEffortMoves)
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
        // History now PERSISTS across moves within a game (its gravity update self-decays,
        // so stale signal fades rather than needing a hard flush) -- only killers, which are
        // ply-specific and meaningless once the tree shifts by a move, are cleared per search.
        // A full flush (history + killers) still happens in newGame(). Advancing the TT
        // generation ages prior searches' entries for replacement.
        clearKillers();
        ttGeneration = (ttGeneration + 1) & 0x3F;
        startNanos = System.nanoTime();
        // Game ply (half-moves since the game/FEN start, independent of this search's own
        // recursion ply): drives both the adaptive moves-to-go estimate in computeTimeLimits and
        // the evaluation-based time lock below. fullmoveNumber/sideToMove survive a
        // "position fen ... moves ..." reload the same way an actual game history would.
        int gamePly = (pos.fullmoveNumber() - 1) * 2 + (pos.sideToMove() == Piece.BLACK ? 1 : 0);
        computeTimeLimits(limits, pos.sideToMove(), gamePly);

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
        int committedPonderMove = 0;

        int previousScore = 0;
        // Best-move stability tracking: local to this think() call (not instance state), so
        // it starts fresh on every search and needs no explicit reset/flush between calls.
        int previousBestMove = 0;
        int stableIterationsCount = 0;
        int lostIterationsCount = 0;
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
                    score = negamax(pos, d, alpha, beta, 0, 0, 0, false);
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
                score = negamax(pos, d, -INF, INF, 0, 0, 0, false);
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
            // Capture the ponder move from THIS just-completed iteration's PV (pvTable[0] is
            // valid here; it may be wiped by a later aborted iteration). Only keep it when the
            // PV's first move actually is the committed best move: otherwise the pair we emit
            // would be (bestmove from this iteration, ponder from a stale/mismatched PV), which
            // can be illegal in the position after bestmove. Clear rather than keep a stale one.
            if (pvLength[0] >= 2 && pvTable[0][0] == committedMove) {
                committedPonderMove = pvTable[0][1];
            } else {
                committedPonderMove = 0;
            }
            int prevIterScore = previousScore; // last iteration's score, for the adaptive eval-trend factor
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

            // Track how many consecutive iterations the position has looked clearly lost (but
            // not an already-found forced mate, which stops on its own). Drives the lost-
            // position time brake below.
            if (committedScore < LOST_SCORE_THRESHOLD_CP && committedScore > -MATE_IN_MAX) {
                lostIterationsCount++;
            } else {
                lostIterationsCount = 0;
            }

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
            // softLimitMs and the lost-position brake).
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

            // Getting mated: a forced mate AGAINST the engine. Every soft/early time exit is
            // disabled in this state -- the previous search's TT entries make shallow
            // iterations re-derive the mate score after a handful of nodes (observed: 8-16
            // nodes at depth 2 on consecutive moves), and any exit that trusts such an
            // iteration commits a move with zero resistance search. Deeper iterations are the
            // only thing that can find the longest defense or a hole in the "forced" mate, so
            // the full soft budget is always spent here. A mate FOR the engine is the
            // opposite case and keeps its stop-as-soon-as-found handling (see the winning-mate
            // break at the bottom of this loop).
            boolean gettingMated = committedScore <= -MATE_IN_MAX;

            // Hunting mate: winning decisively (>= MATE_HUNT_SCORE_CP) but no forced mate found
            // yet. Suppress the stability early exit so the full soft budget is spent looking
            // for the actual mate instead of banking the clock after 2 stable iterations. Still
            // bounded by softLimitMs/hardLimitMs below, and superseded the instant a mate is
            // found (score >= MATE_IN_MAX breaks the loop). The symmetric counterpart to
            // gettingMated: one side searches full time for the longest defense, the other for
            // the fastest kill. Excluded once the score IS a mate -- that path banks time by
            // design (the winning-mate break).
            boolean huntingMate = useTime && useMateHunt
                    && committedScore >= MATE_HUNT_SCORE_CP && committedScore < MATE_IN_MAX;

            // Winning mate found but not yet resolved to the fastest one: keep deepening (don't
            // let the stability exit bank time on a possibly-slower mating move) until depth
            // reaches the mate distance, at which point the winning-mate break below fires with
            // the shortest mate. The counterpart to the break's own floor; without it the
            // decisive-score stability acceleration (2 iterations) could commit an unresolved
            // deep mate at SOFT_BOUND_MIN_DEPTH before the floor is ever reached.
            boolean winningMateUnresolved = useTime && useShortestMate
                    && committedScore >= MATE_IN_MAX && d < MATE - committedScore;

            // Soft-bound (optimum time) stability early exit: only considered once the search
            // is deep enough (>= SOFT_BOUND_MIN_DEPTH) for the root move to be trustworthy, and
            // never while the evaluation-based time lock is active -- disabling this exit
            // outright there is the whole point of the lock (see EVAL_LOCK_* above).
            // Requires BOTH: (a) elapsed time already past the optimum-time target, and
            // (b) the root move has stopped changing for several iterations in a row -- a
            // depth confirming the same move again is unlikely to change the final decision,
            // so the remaining clock time is better saved than spent re-confirming it.
            if (evalLockActive || volatilePos || gettingMated || huntingMate || winningMateUnresolved) {
                // Explicitly clear rather than merely skip-setting: softStopArmed is a field
                // that persists across iterations, so a prior iteration (before the lock/
                // volatility engaged, or before committedScore dropped into the dead-equal
                // band) could otherwise leave a stale `true` behind that checkTime() would
                // still honor mid-iteration even though this iteration's block never re-armed
                // it. A volatile position must never take the optimum-time soft exit, and
                // neither must a search that is being mated (gettingMated) or closing in on a
                // mate it hasn't yet found (huntingMate).
                softStopArmed = false;
            } else if (useAdaptiveTime && !pondering && useTime && d >= SOFT_BOUND_MIN_DEPTH
                    && optimumTimeMs != Long.MAX_VALUE) {
                // Only with a real clock-derived optimum (fixed "go movetime" leaves it at
                // MAX_VALUE, which the factor multiply below would overflow -- and movetime has
                // no stability-banking concept anyway, matching the legacy path's never-fires).
                // Adaptive continuous exit: instead of a single hard stability threshold, scale
                // the optimum-time target down as the root move gets more stable and up as the
                // eval falls (a dropping score means the position is turning against us and wants
                // more thought). stopTarget = optimum * stability * trend * effort, clamped.
                int stab = Math.min(stableIterationsCount, ADAPTIVE_STABILITY_CAP);
                double stabilityFactor = ADAPTIVE_STABILITY_BASE - ADAPTIVE_STABILITY_SLOPE * stab;
                double trend = 1.0 + (prevIterScore - committedScore) * ADAPTIVE_TREND_SLOPE;
                trend = Math.max(ADAPTIVE_TREND_MIN, Math.min(ADAPTIVE_TREND_MAX, trend));
                // Node-effort factor: the share of the whole search spent under the committed
                // best move (see addRootEffort). Neutral (1.0) until a tally exists.
                double effort = 1.0;
                if (committedMove != 0 && nodes > 0) {
                    long bestNodes = rootEffortFor(committedMove);
                    if (bestNodes > 0) effort = effortFactor(bestNodes / (double) nodes);
                }
                double factor = Math.max(ADAPTIVE_FACTOR_MIN,
                        Math.min(ADAPTIVE_FACTOR_MAX, stabilityFactor * trend * effort));
                long stopTarget = (long) (optimumTimeMs * factor);
                int minStable = committedScore > SOFT_BOUND_DECISIVE_SCORE_CP
                        ? SOFT_BOUND_STABLE_ITERATIONS_DECISIVE
                        : ADAPTIVE_MIN_STABLE;
                softStopArmed = stableIterationsCount >= minStable;
                if (softStopArmed && elapsedMs() >= stopTarget) {
                    stop = true;
                    if (sharedStop != null) sharedStop.set(true);
                    if (printInfo) {
                        System.err.println("info string adaptive soft exit: depth=" + d
                                + " move=" + Move.toUci(committedMove)
                                + " stable=" + stableIterationsCount
                                + " factor=" + String.format("%.2f", factor)
                                + " effort=" + String.format("%.2f", effort)
                                + " elapsedMs=" + elapsedMs()
                                + " stopTargetMs=" + stopTarget);
                    }
                    break;
                }
            } else if (!pondering && useTime && d >= SOFT_BOUND_MIN_DEPTH) {
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

            // Obvious-move handling as a SOFT time-scale, not a hard depth cap. When one root
            // move stands out by a wide margin, bank time by treating the soft limit as halved
            // -- iterative deepening then stops on a *time* basis (so a given clock still yields
            // a consistent depth), rather than hard-breaking at whatever shallow depth the gap
            // first appeared, which was the dominant cause of the reported inter-move depth/node
            // collapse. Only ever tightens the budget (fixed-depth/analysis, useTime == false,
            // is untouched and still searches to the requested depth).
            long effectiveSoftLimit = softLimitMs;
            // Never while being mated OR hunting a mate: a lopsided root gap there is expected
            // (one defense delays the mate far longer than the rest, or one move is the kill),
            // exactly the position that needs the full budget, not half of it -- halving here
            // would undo the huntingMate/gettingMated suppression above.
            if (useTime && useObviousMovePruning && !gettingMated && !huntingMate
                    && hasResolvedRootGap(d)) {
                effectiveSoftLimit = softLimitMs / 2;
            }

            // Lost-position time brake: once the score has been clearly losing for a couple of
            // deep-enough iterations, stop pouring clock into an un-savable position. Arming
            // softStopArmed lets checkTime() end even a mid-flight runaway iteration at
            // optimumTimeMs (instead of the multiple-of-optimum hard ceiling), and tightening
            // the between-iteration soft limit ends it promptly here too. Requires
            // SOFT_BOUND_MIN_DEPTH so a shallow eval blip can't trigger it.
            if (useTime && d >= SOFT_BOUND_MIN_DEPTH && lostIterationsCount >= LOST_MIN_ITERATIONS) {
                softStopArmed = true;
                effectiveSoftLimit = Math.min(effectiveSoftLimit,
                        (long) (softLimitMs * LOST_TIME_FRACTION));
            }

            if (!pondering && useTime && elapsedMs() >= effectiveSoftLimit && evalLockTimeSatisfied) break;
            // Winning-mate short-circuit -- deliberately asymmetric. A mate FOR the engine
            // banks the clock, but ONLY once the search is deep enough to have resolved the
            // SHORTEST mate. Breaking the instant any iteration first reports a mate (typically
            // a depth-1 TT hit) commits whatever mating move the shallow read happened to pick,
            // which need not be the fastest -- observed live as the reported mate distance
            // LENGTHENING across consecutive moves (#3 -> #4 -> #5) during a depth-1 playout.
            // For a winning mate, score == MATE - matePlies, so matePlies == MATE - score;
            // searching to d >= matePlies guarantees the whole mate is within the horizon and
            // the fastest one is chosen. This SELF-CORRECTS: as depth grows and a shorter mate
            // is found, score rises and the required depth shrinks, so it breaks the moment the
            // true shortest mate is confirmed. Mate trees are tiny, so this costs negligible
            // time. A mate AGAINST the engine (score <= -MATE_IN_MAX) still never stops
            // deepening (see gettingMated). useShortestMate=false restores the old plain break.
            // While pondering, never stop early -- not even on a found mate: the pondered move
            // may not be played, so we must keep searching until stop/ponderhit.
            if (!pondering && score >= MATE_IN_MAX && (!useShortestMate || d >= MATE - score)) break;
        }

        if (committedMove == 0) {
            MoveList legal = new MoveList();
            MoveGenerator.generateLegal(pos, legal);
            if (legal.size > 0) committedMove = legal.moves[0];
        }

        bestMove = committedMove;
        bestScore = committedScore;
        bestPonderMove = committedPonderMove;
        return committedMove;
    }

    /** Instant bypass: with exactly one legal reply, skip the search tree and play it. */
    private int bypassForcedMove(Position pos, int onlyMove) {
        bestMove = onlyMove;
        bestScore = evaluator.evaluate(pos);
        bestPonderMove = 0;
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

    /** {@code excludedMove}: non-zero only inside a singular-verification search, where that
     *  one move is skipped and TT cutoffs/stores are suppressed (the table's entry describes
     *  the position WITH the move; the exclusion search is asking about the position without
     *  it, a different question whose answer must not overwrite the real one). */
    private int negamax(Position pos, int depth, int alpha, int beta, int ply, int extensions,
                        int excludedMove, boolean cutNode) {
        // Reset this node's PV length BEFORE any early return below. pvTable/pvLength slots
        // are reused across the whole tree, so a node that exits through the stopped/draw/
        // max-ply shortcuts without resetting its slot leaves pvLength[ply] holding a line
        // from an EARLIER visit to this ply in an unrelated subtree -- and a PV-node parent
        // that then accepts this child's score (the repetition-draw 0 raising alpha is the
        // classic case) splices those stale, now-illegal moves into the reported PV.
        // Observed live: "info ... pv ... f4c1 b2a3" where b2a3 was impossible at the root,
        // crashing the lichess-bot wrapper's PV parser.
        pvLength[ply] = ply;
        if (stopped()) return 0;
        if (ply > 0 && pos.isDrawByRuleOrRepetition()) return drawScore(ply);
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
        int ttEval = TranspositionTable.NO_EVAL;
        int ttFlag = TranspositionTable.FLAG_NONE;
        int ttDepth = 0;
        int ttScore = 0; // ply-adjusted; meaningful only when ttFlag != FLAG_NONE
        if (useTT) {
            long entry = tt.probe(key);
            int f = TranspositionTable.flagOf(entry);
            if (f != TranspositionTable.FLAG_NONE) {
                ttMove = TranspositionTable.moveOf(entry);
                ttEval = TranspositionTable.evalOf(entry);
                ttFlag = f;
                ttDepth = TranspositionTable.depthOf(entry);
                ttScore = scoreFromTt(TranspositionTable.scoreOf(entry), ply);
                if (ply > 0 && excludedMove == 0 && ttDepth >= depth) {
                    int s = ttScore;
                    if (f == TranspositionTable.FLAG_EXACT) return s;
                    // Fail-SOFT on TT-sourced bounds: return the stored score itself, not the
                    // window edge. A lower bound proved the true value is >= s (>= beta here),
                    // so s is a legal, strictly more informative return than beta -- it tells
                    // the parent HOW MUCH the cutoff exceeded beta instead of flattening every
                    // winning line to exactly the window edge, which is what previously erased
                    // the engine's sense of how large its advantage was. Symmetric for the
                    // upper-bound/alpha case.
                    if (f == TranspositionTable.FLAG_LOWER && s >= beta) return s;
                    if (f == TranspositionTable.FLAG_UPPER && s <= alpha) return s;
                }
            }
        }

        // Internal iterative reductions: no TT move means this node has never been searched to
        // a useful depth, so its move ordering is unguided; rather than pay full depth to
        // explore an unordered subtree, drop a ply (the shallower search populates a TT move
        // for the eventual re-search). Gated on useTT: with the TT off, ttMove is always 0, so
        // without this gate IIR would fire at every node and break the minimax-equivalence test.
        if (useIir && useTT && ttMove == 0 && !inCheck && depth >= IIR_MIN_DEPTH) {
            depth--;
        }

        // Single static evaluation per node (was formerly computed up to three separate times
        // -- once each in RFP, the null-move gate, and futility). Reuse the TT's stored eval
        // when this position was seen before, else evaluate once. Meaningless while in check,
        // so left as a sentinel there. Node counts are unchanged by this (nodes tick on entry,
        // not per eval); it is purely an nps win plus the source of the `improving` signal.
        int rawEval; // uncorrected: what the TT stores, so corrections never compound
        int staticEval;
        if (inCheck) {
            rawEval = TranspositionTable.NO_EVAL;
            staticEval = TranspositionTable.NO_EVAL;
            evalStack[ply] = -INF; // sentinel: an in-check node never counts as "improving"
        } else {
            rawEval = ttEval != TranspositionTable.NO_EVAL ? ttEval : evaluator.evaluate(pos);
            staticEval = useCorrectionHistory ? correctedEval(pos, rawEval) : rawEval;
            evalStack[ply] = staticEval;
        }
        // "Improving": the side to move's static eval rose versus two plies ago -- when it
        // did, we are likely on an upswing and can afford to prune slightly harder (RFP
        // margin below, LMP budget and LMR depth in the move loop).
        boolean improving = !inCheck && ply >= 2 && evalStack[ply] > evalStack[ply - 2];

        // Razoring: at shallow non-PV nodes whose static eval sits far below alpha, a full-width
        // search rarely rescues the position. Verify with a quiescence search; if even that can't
        // reach alpha, return its (fail-low) value instead of searching the full node. Cheaper
        // than RFP's mirror because it collapses the node to qsearch rather than continuing.
        if (useRazoring && !isPvNode && !inCheck && ply > 0 && excludedMove == 0
                && depth <= RAZOR_MAX_DEPTH && beta < MATE_IN_MAX
                && staticEval + RAZOR_MARGIN_BASE + RAZOR_MARGIN_PER_DEPTH * (depth - 1) < alpha) {
            int razorScore = quiescence(pos, alpha, beta, ply, QS_CHECK_BUDGET);
            if (stopped()) return 0;
            if (razorScore < alpha) return razorScore;
        }

        // Reverse futility pruning: a node-level shortcut tried before move generation, since
        // it can skip generating/scoring moves entirely on a cutoff. Excluded near mate scores
        // (an eval-based margin can't be trusted to reason about forced mates) and at the root
        // (ply 0 must always return a real move).
        if (useRfp && !inCheck && ply > 0 && excludedMove == 0
                && depth <= RFP_MAX_DEPTH && beta < MATE_IN_MAX) {
            // An improving node effectively prunes one depth-step harder: the eval trend is
            // already in our favor, so the margin doesn't need as much slack against it.
            if (staticEval - RFP_MARGIN_PER_DEPTH * (depth - (improving ? 1 : 0)) >= beta) {
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
        if (useNullMove && !inCheck && ply > 0 && excludedMove == 0
                && depth >= NULL_MOVE_MIN_DEPTH
                && beta < MATE_IN_MAX && hasNonPawnMaterial(pos, pos.sideToMove())
                && staticEval >= beta
                && moveStack[ply - 1] != 0) { // never two null moves in a row
            // Dynamic reduction: deeper searches can afford (and benefit from) a larger null
            // reduction, since the confirming re-search still has ample depth left.
            int r = NULL_MOVE_REDUCTION + depth / 6;
            pos.makeNullMove();
            moveStack[ply] = 0;
            pieceToStack[ply] = -1; // a null move is nobody's continuation
            int nullScore = -negamax(pos, depth - 1 - r, -beta, -beta + 1, ply + 1, extensions, 0, !cutNode);
            pos.unmakeNullMove();
            if (stopped()) return 0;
            if (nullScore >= beta) {
                if (depth < NMP_VERIFY_DEPTH) {
                    return beta;
                }
                // High-depth verification: re-search this node shallow with null-move pruning
                // disabled; only trust the cutoff if the real search also fails high. Guards
                // against zugzwang, where the "free" null move hides that every real move loses.
                boolean savedNull = useNullMove;
                useNullMove = false;
                int verify = negamax(pos, depth - 1 - r, beta - 1, beta, ply, extensions, 0, cutNode);
                useNullMove = savedNull;
                if (stopped()) return 0;
                if (verify >= beta) {
                    return beta;
                }
            }
        }

        // ProbCut: tried before the singular test. At higher depths, a capture that beats a
        // raised beta (beta + margin) under a shallow qsearch AND a reduced verification search
        // almost always means the node fails high, so cut early with a reduced-depth lower bound.
        // Skipped when a trustworthy TT entry already says the raised-beta test would fail. Uses
        // moveListPool[ply] for its capture list -- safe because it runs to completion before the
        // singular block or the main move loop touch that pool (same pattern as null-move above).
        if (useProbcut && !isPvNode && !inCheck && ply > 0 && excludedMove == 0
                && depth >= PROBCUT_MIN_DEPTH && Math.abs(beta) < MATE_IN_MAX
                && !(ttFlag != TranspositionTable.FLAG_NONE
                        && ttDepth >= depth - 3 && ttScore < beta + PROBCUT_MARGIN)) {
            int probCutBeta = beta + PROBCUT_MARGIN;
            MoveList caps = moveListPool[ply];
            caps.clear();
            MoveGenerator.generateLegalCaptures(pos, caps);
            for (int i = 0; i < caps.size; i++) {
                int move = caps.moves[i];
                // Only captures whose static exchange covers the gap from staticEval to probCutBeta.
                if (see(pos, move) < probCutBeta - staticEval) continue;
                int movedPieceTo = pos.pieceAt(Move.from(move)) * 64 + Move.to(move);
                pos.makeMove(move);
                moveStack[ply] = move;
                pieceToStack[ply] = movedPieceTo;
                // Shallow verification at the raised beta, then confirm with a reduced search.
                int score = -quiescence(pos, -probCutBeta, -probCutBeta + 1, ply + 1, QS_CHECK_BUDGET);
                if (score >= probCutBeta) {
                    score = -negamax(pos, depth - PROBCUT_REDUCTION, -probCutBeta, -probCutBeta + 1,
                            ply + 1, extensions, 0, !cutNode);
                }
                pos.unmakeMove(move);
                if (stopped()) return 0;
                if (score >= probCutBeta) {
                    tt.store(key, depth - 3, scoreToTt(score, ply), TranspositionTable.FLAG_LOWER,
                            move, staticEval, ttGeneration);
                    return score;
                }
            }
        }

        // Singular-extension verification (and multicut). Same-ply recursion is safe for the
        // per-ply pools for the same reason the null-move verification search above is: it
        // runs to completion BEFORE this node touches moveListPool[ply]/scorePool[ply].
        // A TT move whose entry is deep enough (within SINGULAR_TT_DEPTH_MARGIN) and at least
        // a lower bound is re-tested by searching every OTHER move at half depth against
        // (ttScore - margin): if nothing comes close, the TT move is the only real try and
        // gets a one-ply extension (under the same budget as check extensions, so forcing
        // lines can't explode depth unboundedly). If the excluded search instead beats beta,
        // two separate moves fail high here and the node is cut outright (multicut).
        int singularExtension = 0;
        if (useSingular && useTT && excludedMove == 0 && ply > 0
                && depth >= SINGULAR_MIN_DEPTH
                && ttMove != 0
                && (ttFlag == TranspositionTable.FLAG_LOWER || ttFlag == TranspositionTable.FLAG_EXACT)
                && ttDepth >= depth - SINGULAR_TT_DEPTH_MARGIN
                && Math.abs(ttScore) < MATE_IN_MAX
                && extensions < currentDepth / 2) {
            int singularBeta = ttScore - SINGULAR_MARGIN_PER_DEPTH * depth;
            int value = negamax(pos, (depth - 1) / 2, singularBeta - 1, singularBeta,
                    ply, extensions, ttMove, cutNode);
            if (stopped()) return 0;
            if (value < singularBeta) {
                singularExtension = 1;
                // Double extension: the TT move is not merely singular but singular by a wide
                // margin (no other move even came close), so it is worth two plies. Non-PV only,
                // and bounded by the same extensions budget the +1 case already respects.
                if (useSingularExtDouble && !isPvNode && value < singularBeta - SINGULAR_DOUBLE_MARGIN) {
                    singularExtension = 2;
                }
            } else if (value >= beta) {
                // Multicut: the fail-soft value is a valid lower bound for this node even
                // with the TT move removed, so with it back the bound can only be stronger.
                return value;
            } else if (useSingularExtDouble && ttScore >= beta) {
                // Negative extension: the TT move is NOT singular (another move reached
                // singularBeta) yet the TT score says this node likely fails high anyway, so the
                // TT move can be searched a ply shallower without much risk.
                singularExtension = -1;
            }
        }

        MoveList moves = moveListPool[ply];
        moves.clear();
        MoveGenerator.generateLegal(pos, moves);
        if (moves.size == 0) {
            return inCheck ? -MATE + ply : drawScore(ply); // checkmate or (contempt-shaded) stalemate
        }

        int us = pos.sideToMove();
        // The move that led to this node (0 after a null move); keys the countermove heuristic.
        int prevMove = ply > 0 ? moveStack[ply - 1] : 0;
        int counterMove = (useCounterMove && prevMove != 0)
                ? counterMoves[Move.from(prevMove)][Move.to(prevMove)] : 0;
        // Continuation-history keys of the ancestor moves one and two plies back (-1 when
        // unavailable: root shallow plies, a null move, or the toggle being off).
        int pt1 = (useContHist && ply >= 1) ? pieceToStack[ply - 1] : -1;
        int pt2 = (useContHist && ply >= 2) ? pieceToStack[ply - 2] : -1;
        int[] scores = scoreArray(ply, moves.size);
        scoreMoves(pos, moves, ttMove, ply, us, scores, counterMove, pt1, pt2);

        // Futility pruning gate: computed once per node (not per move) so the move loop below
        // only pays for a static eval call when a low enough depth makes it useful. Excluded
        // at the root (ply > 0, matching RFP/NMP above): the root must always fully search
        // every candidate move, since skipping a late-ordered quiet move here on the strength
        // of a surface-level static eval can silently discard the true best defense in a
        // losing position -- exactly the move whose slower-mate score would otherwise have won
        // the root's alpha/bestScore comparisons below.
        boolean futilityGateOpen = useFutility && !inCheck && ply > 0 && depth <= FUTILITY_MAX_DEPTH
                && alpha > -MATE_IN_MAX;
        int futilityEval = futilityGateOpen ? staticEval : 0;

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
        int[] capturesTried = capturesTriedPool[ply];
        int capturesTriedCount = 0;

        for (int i = 0; i < moves.size; i++) {
            selectNext(moves, scores, i);
            int move = moves.moves[i];
            if (move == excludedMove) continue; // singular verification asks "and without this?"
            boolean isCapture = Move.isCapture(move);
            boolean isPromotion = Move.isPromotion(move);
            boolean isQuiet = !isCapture && !isPromotion;

            // Advanced pawn push: a quiet pawn move reaching the relative 5th rank or beyond.
            // These are exactly the slow, long-term threats (passed-pawn advances, space gains)
            // that late-move pruning must NOT silently discard, so they are exempted below.
            // Computed before makeMove, while the mover still sits on `from`.
            boolean isAdvancedPawnPush = false;
            if (isQuiet && Piece.type(pos.pieceAt(Move.from(move))) == Piece.PAWN) {
                int toRank = Move.to(move) >> 3;
                int relRank = us == Piece.WHITE ? toRank : 7 - toRank;
                isAdvancedPawnPush = relRank >= 4;
            }

            // SEE pruning (before making the move): at a shallow non-PV node, a capture that
            // loses more than a depth-scaled margin by static exchange can't be a real threat,
            // so skip it outright. Never the first move; never near mate scores.
            if (useSeePruning && !isPvNode && i > 0 && isCapture
                    && depth <= SEE_PRUNE_MAX_DEPTH && bestScore > -MATE_IN_MAX
                    && see(pos, move) < -SEE_PRUNE_MARGIN * depth) {
                continue;
            }

            // History pruning (also before making the move): scores[i] for a plain quiet is
            // its combined butterfly + continuation history (killers/countermove sit in
            // positive bands far above the threshold, so they are never caught). A quiet this
            // context has repeatedly punished isn't worth a search at shallow depth.
            if (useHistoryPruning && !isPvNode && !inCheck && isQuiet && i > 0
                    && depth <= HISTORY_PRUNE_MAX_DEPTH && bestScore > -MATE_IN_MAX
                    && scores[i] < HISTORY_PRUNE_MARGIN_PER_DEPTH * depth) {
                continue;
            }

            // Continuation-history key of this move (mover is still on `from` pre-make).
            int movedPieceTo = pos.pieceAt(Move.from(move)) * 64 + Move.to(move);
            // Node-effort tally (adaptive time only, root only): snapshot the node counter so
            // the delta after unmakeMove credits this move's whole subtree. Root moves are
            // never pruned (all pruning above needs ply > 0 or !isPvNode; the root is PV), so
            // the tally is complete. Guarded so the shipped OFF default does zero bookkeeping.
            long effortNodesBefore = (useAdaptiveTime && ply == 0) ? nodes : -1;
            pos.makeMove(move);
            moveStack[ply] = move; // the move that leads to the child node at ply+1
            pieceToStack[ply] = movedPieceTo;

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

            // Late move pruning: at a shallow non-PV node, once enough quiet moves have already
            // been searched, the remaining (latest-ordered, least promising) quiets are skipped
            // outright. Budget is quadratic in depth (with a floor so very shallow nodes don't
            // prune after only a couple of quiets) and larger when improving. Never the first
            // move, never a checking move, never an advanced pawn push (those carry the slow
            // positional threats LMP must not discard), never near mate scores.
            int lmpThreshold = Math.max(LMP_MIN_QUIETS,
                    improving ? 3 + depth * depth : (3 + depth * depth) / 2);
            if (useLmp && !isPvNode && !inCheck && isQuiet && !givesCheck && !isAdvancedPawnPush
                    && i > 0 && depth <= LMP_MAX_DEPTH && bestScore > -MATE_IN_MAX
                    && quietsTriedCount >= lmpThreshold) {
                pos.unmakeMove(move);
                continue;
            }

            // Record every quiet actually searched (not pruned away above) so that, if a
            // later move in this loop causes a cutoff, every quiet tried and rejected before
            // it can be identified for a history malus below.
            if (isQuiet) quietsTried[quietsTriedCount++] = move;
            else if (isCapture) capturesTried[capturesTriedCount++] = move;

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
                // A singular TT move (see the verification block above) is extended a ply
                // here; the TT move always sorts first, so i == 0 is the only site needed.
                // The singular result carries the amount: +1 (singular), +2 (clearly singular,
                // double-extended), or -1 (not singular and TT says fail-high, so shorten). Only
                // the TT move is affected. With the double/negative toggle off, singularExtension
                // is only ever 0 or 1, so this reduces exactly to the legacy "+1 if singular".
                int ext = (move == ttMove) ? singularExtension : 0;
                // Child node-type label: the first child of a PV node is the next PV node
                // (false); at a NON-PV node this "full-window" call is a zero-width window
                // anyway, and the standard alternation applies -- a cut-node's first child is
                // an all-node and vice versa (!cutNode). The extension budget only counts
                // positive extensions (a negative one shortens depth without freeing budget).
                score = -negamax(pos, depth - 1 + ext, -beta, -alpha, ply + 1,
                        extensions + Math.max(0, ext), 0, isPvNode ? false : !cutNode);
            } else {
                // Every later move is assumed to be worse: probe it first with a minimal
                // (null/zero) window, (-alpha-1, -alpha). Proving "not better than alpha"
                // costs a fraction of a full-window search, so a correctly-ordered move list
                // turns most of these probes into cheap cutoffs instead of full re-explorations.
                int searchDepth = depth - 1;
                int reduction = 0;
                if (lmrEligible) {
                    // Log-based reduction (LMR_TABLE), softened by one ply for killers and
                    // strong-history quiets (moves ordering already trusts), then clamped so at
                    // least one ply of real search remains (searchDepth >= 1) -- unlike the
                    // un-reduced case, depth-1 must still be free to reach 0 and fall into
                    // quiescence, but a reduced search must never collapse the node to a leaf.
                    reduction = LMR_TABLE[Math.min(depth, 63)][Math.min(i, 63)];
                    // Reduce a step harder when the node isn't improving (eval trending down
                    // makes late quiets even less likely to matter)...
                    if (!improving) reduction++;
                    // ...and a step softer for moves ordering already trusts: killers, or a
                    // strong combined history/continuation-history score (scores[i] is that
                    // combined signal for plain quiets; killer band values also clear it).
                    if (move == killers[ply][0] || move == killers[ply][1]
                            || scores[i] >= HISTORY_MAX / 2) {
                        reduction--;
                    }
                    // Same carve-out LMP already applies: an advanced pawn push carries slow,
                    // long-term value that a quiet move's low history badly underrates, so it is
                    // never reduced by more than a single ply -- reducing it 4-5 plies (as the
                    // raw log table would at a high index) pushes exactly those strategic
                    // resources past the horizon before the search can appreciate them.
                    if (isAdvancedPawnPush && reduction > 1) reduction = 1;
                    // Cut-node reduction: a node expected to fail high spends its late moves
                    // only to confirm the cutoff, so they can afford one more ply of reduction.
                    if (useCutnodeLmr && cutNode) reduction++;
                    // TT-move-is-capture reduction: when the best known move here is a capture,
                    // late quiets are even less likely to be the answer, so reduce one more ply.
                    if (useLmrTtCapture && ttMove != 0 && Move.isCapture(ttMove)) reduction++;
                    if (reduction < 0) reduction = 0;
                    if (reduction > depth - 2) reduction = depth - 2;
                    searchDepth = depth - 1 - reduction;
                }
                // Child expectation flips at each null-window level: a cut-node's children
                // are all-nodes and vice versa (at a PV node, cutNode == false, so later
                // scout children are expected cut-nodes -- the standard PVS pattern). An
                // earlier version passed `true` unconditionally here, which labelled nearly
                // every non-PV node in the tree an expected cut-node and made the cutnode-LMR
                // extra reduction fire everywhere: measured +23% fixed-depth nodes from
                // re-search cascades and a self-play regression.
                score = -negamax(pos, searchDepth, -alpha - 1, -alpha, ply + 1, extensions, 0, !cutNode);
                if (score > alpha && reduction > 0) {
                    // LMR re-search guardrail: a reduced-depth probe is not trustworthy enough
                    // to accept a fail-high -- the shallower search may simply have missed a
                    // refutation the reduction skipped over. Re-verify at the original,
                    // unreduced depth, still under the null window, before deciding whether
                    // this move earns the full PVS re-search below.
                    score = -negamax(pos, depth - 1, -alpha - 1, -alpha, ply + 1, extensions, 0, !cutNode);
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
                    score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, extensions, 0, false);
                }
            }
            pos.unmakeMove(move);
            // Credit this move's subtree nodes even on an aborted search (before the stopped()
            // return below) -- the nodes were genuinely spent under it either way.
            if (effortNodesBefore >= 0) addRootEffort(move, nodes - effortNodesBefore);
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
                    // Record the PV only at PV (exact-window) nodes, and splice the child's line
                    // only on an exact score (score < beta). A scout/non-PV node never has a
                    // real PV, and a fail-high cutoff (score >= beta) never ran the full-window
                    // re-search that would make pvTable[ply+1] this move's genuine continuation
                    // -- splicing it there would graft a stale, unrelated line into the reported
                    // PV. In that case we record just this move. This keeps `info pv` a true
                    // principal variation; move selection (rootBestMove below) is independent of
                    // it either way.
                    if (isPvNode) {
                        pvTable[ply][ply] = move;
                        if (score < beta) {
                            int childLen = pvLength[ply + 1];
                            for (int j = ply + 1; j < childLen; j++) {
                                pvTable[ply][j] = pvTable[ply + 1][j];
                            }
                            pvLength[ply] = childLen;
                        } else {
                            pvLength[ply] = ply + 1;
                        }
                    }
                    if (ply == 0) {
                        rootBestMove = move;
                        rootMoveCommitted = true;
                    }
                    if (alpha >= beta) {
                        if (isQuiet) {
                            updateKillers(ply, move);
                            if (useCounterMove && prevMove != 0) {
                                counterMoves[Move.from(prevMove)][Move.to(prevMove)] = move;
                            }
                            int bonus = Math.min(depth * depth, HISTORY_MAX);
                            updateHistory(us, move, bonus);
                            updateContHist(pos, ply, move, bonus);
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
                                updateContHist(pos, ply, quietsTried[qi], -bonus);
                            }
                        } else if (useCaptureHistory && isCapture) {
                            // A capture caused the cutoff: reward it, and malus every capture
                            // tried and rejected before it (the last recorded capture is this
                            // one, so skip it). Board is already unmade, so pieceAt() reads the
                            // parent position -- captured piece is back on `to`.
                            int bonus = Math.min(depth * depth, HISTORY_MAX);
                            updateCaptureHistory(pos, move, bonus);
                            for (int ci = 0; ci < capturesTriedCount - 1; ci++) {
                                updateCaptureHistory(pos, capturesTried[ci], -bonus);
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

        // Never store an exclusion-search result: it scored the position WITHOUT one of its
        // moves, which is not this zobrist key's real value (see the negamax javadoc).
        // Correction-history update: fold in this node's (result - static eval) difference,
        // but only when the result can legitimately disagree with the eval -- never in check
        // (no eval exists), never on mate scores (not an eval-scale quantity), never from an
        // exclusion search, and only when the bound direction actually constrains the true
        // score (a fail-low BELOW the eval or fail-high ABOVE it says nothing about where the
        // exact value sits relative to the eval).
        if (useCorrectionHistory && !inCheck && excludedMove == 0
                && Math.abs(bestScore) < MATE_IN_MAX) {
            boolean failHigh = bestScore >= beta;
            boolean failLow = bestScore <= origAlpha;
            if ((!failHigh || bestScore > staticEval) && (!failLow || bestScore < staticEval)) {
                updateCorrection(pos, depth, bestScore - staticEval);
            }
        }

        if (useTT && excludedMove == 0) {
            int flag = bestScore <= origAlpha ? TranspositionTable.FLAG_UPPER
                     : bestScore >= beta ? TranspositionTable.FLAG_LOWER
                     : TranspositionTable.FLAG_EXACT;
            int storedEval = inCheck ? TranspositionTable.NO_EVAL : rawEval;
            tt.store(key, depth, scoreToTt(bestScore, ply), flag, localBest, storedEval, ttGeneration);
        }
        return bestScore;
    }

    /** Quiescence search over captures/promotions, with bounded check evasions. */
    private int quiescence(Position pos, int alpha, int beta, int ply, int checkBudget) {
        if (stopped()) return 0;
        if ((++nodes & 2047) == 0) checkTime();
        if (pos.isDrawByRuleOrRepetition()) return drawScore(ply);

        boolean inCheck = pos.inCheck(pos.sideToMove());
        if (ply >= MAX_PLY) return evaluator.evaluate(pos);

        // Quiescence TT probe: qsearch nodes are stored at depth 0, so any stored entry is at
        // least as deep and its bound is usable for a cutoff under the standard flag rules.
        // The stored eval doubles as a cached stand-pat below, and the stored move seeds
        // ordering (it will be a capture/promotion at a non-check node).
        long key = pos.zobristKey();
        int qttMove = 0;
        int ttStoredEval = TranspositionTable.NO_EVAL;
        if (useTT && useQsTt) {
            long entry = tt.probe(key);
            int f = TranspositionTable.flagOf(entry);
            if (f != TranspositionTable.FLAG_NONE) {
                qttMove = TranspositionTable.moveOf(entry);
                ttStoredEval = TranspositionTable.evalOf(entry);
                int s = scoreFromTt(TranspositionTable.scoreOf(entry), ply);
                if (f == TranspositionTable.FLAG_EXACT) return s;
                if (f == TranspositionTable.FLAG_LOWER && s >= beta) return beta;
                if (f == TranspositionTable.FLAG_UPPER && s <= alpha) return alpha;
            }
        }

        int origAlpha = alpha;
        int qBestMove = 0;
        int bestScore;
        int standPat = 0;
        int rawStandPat = 0; // uncorrected twin of standPat; what the QS TT stores
        if (inCheck) {
            // Static eval of an in-check position is meaningless (it ignores the check
            // entirely), so returning it here is a prime source of tactical blindness. With
            // useQsEvasions we never do that: evasions are always generated, bounded only by
            // the MAX_PLY guard above. The old budget-exhaustion static-eval return is kept as
            // the A/B fallback.
            if (!useQsEvasions && checkBudget <= 0) return evaluator.evaluate(pos);
            bestScore = -INF;
        } else {
            // rawStandPat is what the TT stores (see negamax's rawEval: corrections must
            // never compound through the table); the corrected value drives the cutoffs.
            rawStandPat = (useTT && useQsTt && ttStoredEval != TranspositionTable.NO_EVAL)
                    ? ttStoredEval : evaluator.evaluate(pos);
            standPat = useCorrectionHistory ? correctedEval(pos, rawStandPat) : rawStandPat;
            bestScore = standPat;
            if (standPat >= beta) {
                if (useTT && useQsTt) {
                    tt.store(key, 0, scoreToTt(standPat, ply), TranspositionTable.FLAG_LOWER,
                            0, rawStandPat, ttGeneration);
                }
                return standPat;
            }
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
        scoreMovesQ(pos, moves, scores, qttMove);
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
                qBestMove = move;
                if (score > alpha) {
                    alpha = score;
                    if (alpha >= beta) break;
                }
            }
        }

        if (useTT && useQsTt) {
            int flag = bestScore <= origAlpha ? TranspositionTable.FLAG_UPPER
                     : bestScore >= beta ? TranspositionTable.FLAG_LOWER
                     : TranspositionTable.FLAG_EXACT;
            int se = inCheck ? TranspositionTable.NO_EVAL : rawStandPat;
            tt.store(key, 0, scoreToTt(bestScore, ply), flag, qBestMove, se, ttGeneration);
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

    private void scoreMoves(Position pos, MoveList moves, int ttMove, int ply, int us, int[] scores,
                            int counterMove, int pt1, int pt2) {
        for (int i = 0; i < moves.size; i++) {
            scores[i] = scoreMove(pos, moves.moves[i], ttMove, ply, us, counterMove, pt1, pt2);
        }
    }

    private int scoreMove(Position pos, int move, int ttMove, int ply, int us, int counterMove,
                          int pt1, int pt2) {
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
            // Capture-history tie-break within the band (scaled to stay inside it).
            if (useCaptureHistory) {
                int mover = pos.pieceAt(Move.from(move));
                score += capHist[mover][Move.to(move)][capturedType(pos, move)] / CAPHIST_ORDER_DIV;
            }
            return score;
        }
        if (promo) {
            return PROMO_BASE + Move.promoType(move);
        }
        if (move == killers[ply][0]) return KILLER0;
        if (move == killers[ply][1]) return KILLER1;
        // Countermove: a quiet that refuted this exact predecessor before, ranked just under
        // the killers and above generic history.
        if (move == counterMove) return COUNTERMOVE;
        // Quiet ordering: butterfly history plus the two continuation-history tables. Max
        // combined magnitude is 3*HISTORY_MAX = 300k, still strictly below COUNTERMOVE (700k),
        // so the ordering bands stay layered.
        int score = history[us][Move.from(move)][Move.to(move)];
        if (pt1 >= 0 || pt2 >= 0) {
            int cur = pos.pieceAt(Move.from(move)) * 64 + Move.to(move);
            if (pt1 >= 0) score += contHist1[pt1][cur];
            if (pt2 >= 0) score += contHist2[pt2][cur];
        }
        return score;
    }

    private void scoreMovesQ(Position pos, MoveList moves, int[] scores, int ttMove) {
        for (int i = 0; i < moves.size; i++) {
            int move = moves.moves[i];
            if (move == ttMove) {
                scores[i] = TT_SCORE; // search the TT move first in qsearch too
                continue;
            }
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

    /** Adds {@code delta} nodes to {@code move}'s whole-search effort tally (ply 0 only). */
    private void addRootEffort(int move, long delta) {
        for (int i = 0; i < rootEffortCount; i++) {
            if (rootEffortMoves[i] == move) {
                rootEffortNodes[i] += delta;
                return;
            }
        }
        if (rootEffortCount < rootEffortMoves.length) {
            rootEffortMoves[rootEffortCount] = move;
            rootEffortNodes[rootEffortCount] = delta;
            rootEffortCount++;
        }
    }

    /** Nodes spent under {@code move} at the root this think() call, or 0 if never searched.
     *  Package-private for tests (same rationale as computeTimeLimits/see). */
    long rootEffortFor(int move) {
        for (int i = 0; i < rootEffortCount; i++) {
            if (rootEffortMoves[i] == move) return rootEffortNodes[i];
        }
        return 0;
    }

    /** Sum of all root-move effort tallies this think() call. Package-private for tests. */
    long rootEffortTotal() {
        long sum = 0;
        for (int i = 0; i < rootEffortCount; i++) sum += rootEffortNodes[i];
        return sum;
    }

    /** The node-effort time factor for a best-move node fraction in [0,1]: high concentration
     *  on one move means the root is settled (factor < 1, stop sooner); spread-out effort means
     *  it is contested (factor > 1, spend longer). Package-private static for direct unit tests. */
    static double effortFactor(double bestFrac) {
        double f = ADAPTIVE_EFFORT_BASE - ADAPTIVE_EFFORT_SLOPE * bestFrac;
        return Math.max(ADAPTIVE_EFFORT_MIN, Math.min(ADAPTIVE_EFFORT_MAX, f));
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
        history[us][from][to] = gravity(history[us][from][to], bonus);
    }

    /** The gravity update itself, shared by every history-family table (see updateHistory). */
    private static int gravity(int current, int bonus) {
        return current + bonus - current * Math.abs(bonus) / HISTORY_MAX;
    }

    // --- correction history (see the corrHist field notes) ---

    /** Pawn-structure hash slot: same 2-multiply mix the evaluator's pawn cache uses. */
    private int pawnCorrIndex(Position pos) {
        long wp = pos.pieces(Piece.index(Piece.WHITE, Piece.PAWN));
        long bp = pos.pieces(Piece.index(Piece.BLACK, Piece.PAWN));
        long h = wp * 0x9E3779B97F4A7C15L ^ Long.rotateLeft(bp * 0xC2B2AE3D27D4EB4FL, 32);
        h ^= h >>> 32;
        return (int) h & (CORR_SIZE - 1);
    }

    /** The raw evaluator output nudged by the learned pawn-structure correction, kept
     *  strictly clear of the mate band so a corrected eval can never masquerade as mate. */
    private int correctedEval(Position pos, int rawEval) {
        int v = rawEval + corrHist[pos.sideToMove()][pawnCorrIndex(pos)] / CORR_GRAIN;
        if (v >= MATE_IN_MAX) v = MATE_IN_MAX - 1;
        if (v <= -MATE_IN_MAX) v = -MATE_IN_MAX + 1;
        return v;
    }

    /** Folds a completed node's (search result - static eval) difference into the moving
     *  average, weighted by depth (deeper results are more trustworthy) and clamped. */
    private void updateCorrection(Position pos, int depth, int diff) {
        int stm = pos.sideToMove();
        int idx = pawnCorrIndex(pos);
        int target = diff * CORR_GRAIN;
        if (target > CORR_LIMIT) target = CORR_LIMIT;
        if (target < -CORR_LIMIT) target = -CORR_LIMIT;
        int weight = Math.min(depth + 1, CORR_WEIGHT_MAX);
        int next = (corrHist[stm][idx] * (CORR_WEIGHT_SCALE - weight) + target * weight)
                / CORR_WEIGHT_SCALE;
        if (next > CORR_LIMIT) next = CORR_LIMIT;
        if (next < -CORR_LIMIT) next = -CORR_LIMIT;
        corrHist[stm][idx] = next;
    }

    /**
     * Applies a gravity {@code bonus} to both continuation-history tables for {@code move}
     * at {@code ply}, keyed by the ancestor moves one and two plies back. Called at the same
     * points as {@link #updateHistory} (cutoff bonus + tried-quiet malus); the move is
     * already unmade here, so its mover is back on its from-square.
     */
    /** Captured piece TYPE for {@code move} (PAWN for en passant), for capture-history keys. */
    private int capturedType(Position pos, int move) {
        return Move.flag(move) == Move.EP_CAPTURE ? Piece.PAWN : Piece.type(pos.pieceAt(Move.to(move)));
    }

    /** Applies a gravity {@code bonus} to the capture-history cell of {@code move} (board must
     *  be in the pre-move / unmade state so the mover sits on `from` and the victim on `to`). */
    private void updateCaptureHistory(Position pos, int move, int bonus) {
        int mover = pos.pieceAt(Move.from(move));
        int to = Move.to(move);
        int cap = capturedType(pos, move);
        capHist[mover][to][cap] = gravity(capHist[mover][to][cap], bonus);
    }

    private void updateContHist(Position pos, int ply, int move, int bonus) {
        if (!useContHist) return;
        int cur = pos.pieceAt(Move.from(move)) * 64 + Move.to(move);
        if (ply >= 1) {
            int pt1 = pieceToStack[ply - 1];
            if (pt1 >= 0) contHist1[pt1][cur] = gravity(contHist1[pt1][cur], bonus);
        }
        if (ply >= 2) {
            int pt2 = pieceToStack[ply - 2];
            if (pt2 >= 0) contHist2[pt2][cur] = gravity(contHist2[pt2][cur], bonus);
        }
    }

    private void clearKillers() {
        for (int[] k : killers) {
            k[0] = 0;
            k[1] = 0;
        }
    }

    // Package-private so LazySmpSearch can flush its persistent workers' tables on
    // ucinewgame without re-clearing the shared TT once per worker via newGame().
    void resetHeuristics() {
        clearKillers();
        for (int c = 0; c < 2; c++) {
            for (int f = 0; f < 64; f++) {
                java.util.Arrays.fill(history[c][f], 0);
            }
        }
        for (int f = 0; f < 64; f++) {
            java.util.Arrays.fill(counterMoves[f], 0);
        }
        for (int i = 0; i < contHist1.length; i++) {
            java.util.Arrays.fill(contHist1[i], 0);
            java.util.Arrays.fill(contHist2[i], 0);
        }
        java.util.Arrays.fill(corrHist[0], 0);
        java.util.Arrays.fill(corrHist[1], 0);
        for (int p = 0; p < 12; p++) {
            for (int s = 0; s < 64; s++) java.util.Arrays.fill(capHist[p][s], 0);
        }
    }

    // --- mate-score ply relativization for the TT ---

    /** Contempt-shaded draw value, from the perspective of the side to move at {@code ply}.
     *  Even ply = the root side is to move (dislikes the draw => -contempt); odd ply = the
     *  opponent is to move (so the same draw is +contempt from their view). contempt==0 gives a
     *  plain 0, identical to the old behaviour. */
    private int drawScore(int ply) {
        return (ply & 1) == 0 ? -contempt : contempt;
    }

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

    /** Legacy 2-arg entry (used by tests): game ply is only consulted by the adaptive path, which
     *  is off by default, so a neutral 0 is harmless here. */
    void computeTimeLimits(SearchLimits limits, int side) {
        computeTimeLimits(limits, side, 0);
    }

    void computeTimeLimits(SearchLimits limits, int side, int gamePly) {
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

        // Moves-to-go estimate. Legacy: a flat 30 in sudden death. Adaptive: a game-ply taper
        // (fewer assumed moves as the game shortens => more time per move in the middlegame where
        // decisions are made), with a floor so the endgame never over-commits the clock.
        int mtg;
        if (limits.movestogo > 0) {
            mtg = useAdaptiveTime ? Math.min(limits.movestogo, 40) : limits.movestogo;
        } else {
            mtg = useAdaptiveTime ? Math.max(ADAPTIVE_MTG_MIN, ADAPTIVE_MTG_BASE - gamePly / 2) : 30;
        }
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
        if (pondering || !useTime || currentDepth < 2) return;
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
                + " pv " + pvString());
    }

    /** The full principal variation for the just-completed root iteration (falls back to the
     *  committed root move if, e.g., the PV table wasn't populated). Package-private so
     *  PvIntegrityTest can validate every reported PV move for legality. */
    String pvString() {
        if (pvLength[0] <= 0 || pvTable[0][0] == 0) {
            return rootBestMove != 0 ? Move.toUci(rootBestMove) : "0000";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < pvLength[0]; i++) {
            int m = pvTable[0][i];
            if (m == 0) break;
            if (i > 0) sb.append(' ');
            sb.append(Move.toUci(m));
        }
        return sb.length() == 0 ? "0000" : sb.toString();
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
