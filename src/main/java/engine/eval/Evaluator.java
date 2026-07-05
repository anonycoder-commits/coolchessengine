package engine.eval;

import static engine.board.Piece.*;

import engine.board.Attacks;
import engine.board.Position;

/**
 * Handcrafted, tapered evaluation: material + piece-square tables interpolated by
 * game phase, plus pawn structure, king safety, and mobility terms.
 *
 * Score is returned from the side-to-move's perspective (positive = better for the
 * side to move), matching the convention used by the search. All terms are strictly
 * color-symmetric, so {@code evaluate(p) == -evaluate(mirror(p))} for a color-swapped,
 * vertically flipped position (see EvalSymmetryTest).
 *
 * The material values and piece-square tables are the well-known PeSTO tables
 * (midgame/endgame pairs); the remaining terms are conventional handcrafted heuristics.
 */
public final class Evaluator {
    private Evaluator() {}

    // Midgame/endgame material values indexed by piece TYPE (PAWN..KING).
    // Non-final so engine.tools.Tune can discover and fit them (the rofchade result on the
    // zurichess corpus came specifically from tuning material + PST, not the auxiliary terms).
    // King entries are a flat direction (one king each side cancels in the color difference).
    public static int[] MG_VALUE = {81, 339, 366, 478, 1025, 0};
    public static int[] EG_VALUE = {90, 282, 299, 517, 937, 0};

    // Tempo: a small flat bonus for the side to move, reflecting the standing initiative of
    // having the move. Beyond a genuine (if small) positional truth, it damps the odd/even
    // evaluation oscillation between plies that otherwise churns the aspiration window and
    // destabilizes node counts between iterations. Added to the side-to-move-relative score,
    // so it is deliberately NOT color-symmetric the way the board terms are (see
    // EvalSymmetryTest, which subtracts it out before checking antisymmetry).
    public static int TEMPO = 23;

    // --- pawn structure / king safety / mobility weights ---
    // Every term below is a genuine MG/EG *pair* (not a single flat value split by eye) --
    // doubled pawns bite harder once there are no piece-play compensations left to offset
    // them, isolated pawns matter less once a king can help defend them directly, and passed
    // pawns are dramatically more dangerous once there's no piece firepower left to blockade
    // them at leisure. Feeding these into the same mg/eg accumulators as material+PST (see
    // evaluate()) is what actually taper these terms -- adding a single flat value after the
    // phase blend, as this evaluator previously did, meant every one of them applied at full
    // strength in a bare king-and-pawn endgame exactly as if the queens were still on the
    // board, which is the classic "endgame bleeding" failure mode this refactor fixes.
    public static int DOUBLED_MG = 9;
    public static int DOUBLED_EG = 19;
    public static int ISOLATED_MG = 9;
    public static int ISOLATED_EG = 6;
    // Passed-pawn bonus indexed by the pawn's rank relative to its own side (0..7). EG ramps
    // far more steeply toward the promotion rank: with no pieces left to blockade or round up
    // a passer, a pawn on the 6th/7th is close to a second queen, not just a small edge.
    public static int[] PASSED_MG = {0, 5, 8, 12, 21, 31, 49, 0};
    public static int[] PASSED_EG = {0, 9, 18, 33, 54, 88, 146, 0};

    // King-safety shield/file terms are an almost purely midgame concept -- they price in the
    // danger of enemy queen/rook fire down an open line at the king, which stops mattering
    // once those attackers are gone. EG values are small, not zero, since a king still prefers
    // some cover against a lone rook/queen endgame, but nowhere near midgame weight; EG king
    // placement is instead driven by EG_PST[KING] rewarding centralization (see initPst()).
    public static int SHIELD_MG = 8;
    public static int SHIELD_EG = -4;
    public static int OPEN_FILE_MG = 21;
    public static int OPEN_FILE_EG = 6;
    public static int SEMI_OPEN_FILE_MG = 10;
    public static int SEMI_OPEN_FILE_EG = -1;

    // King-attack ("attack units"): each enemy piece bearing on the squares around a king
    // contributes units weighted by piece type and by how many king-zone squares it hits; the
    // total is squared (danger grows super-linearly as attackers pile up) into a penalty. This
    // is what lets the engine value a building attack BEFORE tactics force it into qsearch,
    // rather than reading sharp middlegames as flat. Requires >= 2 attackers (a lone piece near
    // the king is not a real threat) and is almost purely a middlegame concern (small eg).
    public static int[] KING_ATTACK_WEIGHT = {0, 2, 2, 2, 6, 0}; // by piece TYPE
    private static final int KING_ATTACK_DIVISOR = 6;   // penalty = min(units*units/DIVISOR, CAP)
    private static final int KING_ATTACK_CAP = 400;     // ceiling so one lopsided node can't dominate
    private static final int KING_ATTACK_EG_SHIFT = 4;  // eg penalty = mg penalty >> this (small)

    // Space: safe central squares (files c-f) in one's own half, not attacked by enemy pawns,
    // scored super-linearly in the number of pieces still on the board -- the dominant factor
    // in closed middlegames, which the engine was previously blind to. Squares sheltered behind
    // an own pawn count extra. Purely a middlegame concept (no eg component): once pieces trade
    // off, space no longer translates into attacking potential.
    private static final long SPACE_MASK_WHITE = (0x3CL << 8) | (0x3CL << 16) | (0x3CL << 24);
    private static final long SPACE_MASK_BLACK = (0x3CL << 32) | (0x3CL << 40) | (0x3CL << 48);
    private static final int SPACE_DIVISOR = 16; // score = (safe + behind) * pieces^2 / DIVISOR

    // Bad bishop: a bishop hemmed in by its own pawns fixed on its square color is a lasting
    // structural liability, worse in the endgame where it can't be traded off into activity.
    private static final long LIGHT_SQUARES = 0x55AA55AA55AA55AAL;
    public static int BAD_BISHOP_MG = 1; // per own pawn on the bishop's square color
    public static int BAD_BISHOP_EG = 5;

    // Mobility by piece TYPE: rooks/queens gain relative value in the endgame (open lines
    // decide races once material thins out), while knights/bishops taper down slightly.
    public static int[] MOBILITY_MG = {0, 4, 5, 2, 1, 0};
    public static int[] MOBILITY_EG = {0, 4, 4, 4, 6, 0};
    // Mobility floor: squares a piece reaches that are contested only by an enemy PAWN still
    // count, at reduced weight (1/this), instead of being masked out entirely. Without it, a
    // closed position where enemy pawns blanket the board collapses both sides' mobility to
    // near zero, erasing positional discrimination and nudging the engine to "escape" by
    // trading its cramped pieces into a structurally worse open line. Full-value squares (safe
    // from enemy pawns) still drive the center/open-file bonuses.
    private static final int MOBILITY_MASK_DIVISOR = 4;

    // Pawn levers / breaks: a lever is an own pawn poised to open the structure -- it already
    // attacks an enemy pawn, or is a single push from attacking one. The side with more levers
    // holds the initiative to open the position on its own terms, the factor flat mobility is
    // blind to in closed structures. Tapered mg-heavy (opening matters most with pieces on).
    public static int LEVER_MG = 6;
    public static int LEVER_EG = 2;

    // Passed-pawn king proximity (endgame only): a passer's value in the endgame depends
    // heavily on which king can reach its promotion square first, which pure rank scaling
    // ignores. Rewards the friendly king being close and the enemy king being far, weighted
    // by how advanced the passer already is.
    public static int PASSER_KING_EG = 18;

    // Outside passed pawn (endgame only): a passer whose file is far from the enemy king is one
    // the king struggles to stop -- it decoys the king off the other wing and often decides the
    // game. This is the motif the engine was blind to in a loss where a queenside majority
    // became a winning outside a-pawn. Eg-only (like PASSER_KING_EG), scaled by how advanced the
    // passer already is.
    // Default ON (2026-07-04): unbiased referee gate scored 51.2% over 600 games (Elo +8,
    // CI [-20,36]) -- grey-zone/no-regression on aggregate self-play, which balanced fast-TC
    // openings rarely exercise anyway (same reasoning as the storm/rule50 terms). Kept on
    // because a dedicated probe confirms it fixes the targeted motif: a lone outside a-pawn
    // position scores +19cp with this term on vs off, and remains exactly color-symmetric.
    public static int OUTSIDE_PASSER_EG = 23;
    public static int OUTSIDE_KING_DIST = 3; // enemy-king file distance at/above which a passer is "outside"
    public static boolean useOutsidePasser = true;

    // Outside flank pawn majority (endgame only): more pawns than the enemy on the flank AWAY
    // from the enemy king can manufacture a passer the king is too far to catch -- the classic
    // "queenside majority" trump. Gated SEPARATELY from the outside-passer term (a fuzzier
    // signal shouldn't be able to sink the higher-confidence one).
    // Default ON (2026-07-04): unbiased referee gate (vs the outside-passer-on baseline) scored
    // 51.7% over 600 games (Elo +12, CI [-16,40]) -- same grey-zone-positive read as the outside-
    // passer term itself; kept for the same reason (narrow anti-specific-loss term, verified to
    // fix the targeted motif and exactly color-symmetric via OutsidePasserTest).
    public static int PAWN_MAJORITY_EG = 7;
    public static boolean usePawnMajority = true;
    private static final long QUEENSIDE_FILES = 0x0F0F0F0F0F0F0F0FL; // files a-d
    private static final long KINGSIDE_FILES = 0xF0F0F0F0F0F0F0F0L;  // files e-h

    // Connected/phalanx pawn bonus and backward-pawn penalty (small, standard mg/eg pairs).
    public static int CONNECTED_MG = 6;
    public static int CONNECTED_EG = 3;
    public static int BACKWARD_MG = 8;
    public static int BACKWARD_EG = 10;

    // Endgame scale factor (out of SCALE_MAX): the eg score is multiplied by this before the
    // phase blend, to reflect that some material configurations are far more drawish than raw
    // material suggests -- opposite-colored bishops halve, dead-drawn minor endings zero out.
    private static final int SCALE_MAX = 16;

    // King-ring mobility (relative king-ring control): for each side, the net attack density
    // over its OWN 3x3 king ring -- total bit-attacks its pieces exert on the ring (defensive
    // coverage) minus total bit-attacks the enemy exerts on it (assault pressure). Differenced
    // between the two sides, this measures who owns the spatial contest around the kings BEFORE
    // any sacrifice lands, steering the engine away from suffocating shells. Heavily middlegame-
    // weighted: king shelter matters while heavy pieces are on the board; in the endgame an
    // active, centralized king is preferred over a sheltered one, so the eg weight is small.
    public static int KING_RING_MG = 2;
    public static int KING_RING_EG = 0;
    // A/B toggle (static: the evaluator is a stateless utility). Flip to false to disable the
    // term for match comparison. Shared across threads -- set once between searches, not mid-search.
    public static boolean useKingRingMobility = true;

    // Weighted-mobility "Ambition" top-ups: on top of the flat per-square MOBILITY_MG/EG
    // rate above, a mobility square gets an extra bonus when it represents genuine central/
    // open-line pressure rather than just square count -- a rook eyeing an open file or a
    // knight/bishop hitting a center square is worth more than an equally-mobile piece
    // shuffling along the back rank. Added on top of (not instead of) the base rate, so total
    // weight for a "premium" square is MOBILITY_*[type] + the matching bonus below.
    public static int ROOK_OPEN_FILE_MOBILITY_MG = 4;
    public static int ROOK_OPEN_FILE_MOBILITY_EG = -2;
    public static int QUEEN_OPEN_FILE_MOBILITY_MG = 0;
    public static int QUEEN_OPEN_FILE_MOBILITY_EG = 2;
    public static int KNIGHT_CENTER_MOBILITY_MG = 6;
    public static int KNIGHT_CENTER_MOBILITY_EG = 3;
    public static int BISHOP_CENTER_MOBILITY_MG = 5;
    public static int BISHOP_CENTER_MOBILITY_EG = 2;

    // Central outposts: a knight or bishop parked on d4/d5/e4/e5 that no enemy pawn can ever
    // structurally challenge (see outpostScore()) is a long-term thorn the opponent can't
    // easily remove, so it earns a standing "tension bonus" independent of mobility/PST --
    // this is what keeps the engine from retreating such a piece to a "safer" but passive
    // square purely to reduce short-term tactical exposure. Knights get the larger bonus
    // since, unlike bishops, they have no long-diagonal alternative outlet once parked.
    private static final long CENTER_SQUARES = (1L << 27) | (1L << 28) | (1L << 35) | (1L << 36); // d4,e4,d5,e5
    public static int OUTPOST_KNIGHT_MG = 18;
    public static int OUTPOST_KNIGHT_EG = 11;
    public static int OUTPOST_BISHOP_MG = 12;
    public static int OUTPOST_BISHOP_EG = 7;

    // Threats: concrete tactical pressure the mobility/space terms are blind to. Three
    // signals, each color-differenced: an enemy piece under attack by a pawn (the cheapest
    // attacker -- the victim must usually move, conceding tempo or material), a hanging piece
    // (attacked by anything, defended by nothing), and a safe pawn push that would attack a
    // piece next move (initiative the opponent must spend a tempo answering).
    // THREAT_BY_PAWN is indexed by the attacked piece's TYPE.
    public static int[] THREAT_BY_PAWN_MG = {0, 32, 32, 40, 45, 0};
    public static int[] THREAT_BY_PAWN_EG = {0, 22, 23, 27, 32, 0};
    public static int HANGING_MG = 24;
    public static int HANGING_EG = 20;
    public static int PAWN_PUSH_THREAT_MG = 12;
    public static int PAWN_PUSH_THREAT_EG = 6;
    // A/B toggle (see useKingRingMobility for the sharing convention).
    public static boolean useThreats = true;

    // Pawn storm: enemy pawns advancing down the king's file or an adjacent file are the
    // slowest-burning and most underrated king threat -- each individual push is quiet and
    // barely moves a shield/attack-unit-based danger score, right up until the storm pawn
    // levers the shield open and every attacking piece pours in at once (observed live: a
    // g5/h5-h4 storm advanced for ten moves at a near-flat eval, then the score collapsed
    // -2 -> -4 -> -6.7 in three moves). Indexed by the pawn's rank distance ahead of the
    // king from the king's own perspective (1 = adjacent rank); mg-heavy since storms need
    // pieces behind them to matter.
    public static int[] STORM_MG = {0, 27, 16, 7};
    public static int[] STORM_EG = {0, 4, 3, 3};
    private static final int STORM_MAX_DIST = 3;
    public static boolean useKingPawnStorm = true;

    // Fifty-move-rule fade: scale the final blended score toward zero as the halfmove clock
    // grows. A "+0.8 forever" line that never moves a pawn or trades is NOT worth +0.8 --
    // it is drifting toward a rule-50 draw (observed live: fifty reversible shuffle moves
    // holding a flat +0.5..0.8 that ended drawn). Because a pawn push or capture resets the
    // clock and therefore RESTORES the faded score, progress-making moves become
    // score-maximizing exactly when the advantage is real, with no explicit "progress"
    // heuristic needed. Denominator 212 fades ~47% of the score at clock 100 -- strong
    // enough to force breaks, gentle enough not to distort normal play (clock is 0-10 in
    // virtually all middlegame positions).
    private static final int RULE50_FADE_DIVISOR = 212;
    public static boolean useRule50Scaling = true;

    // Safe checks: a square from which an enemy piece could deliver check next move, that is
    // empty, not covered by a defending pawn, and not adjacent to the king, is a standing
    // tactical loan against the king -- counted into the attack-unit total (per checking piece
    // TYPE) before the super-linear danger formula, so imminent check threats raise danger
    // even before any capture sequence exists for qsearch to see.
    public static int[] SAFE_CHECK_WEIGHT = {0, 5, 3, 4, 6, 0};
    public static boolean useSafeChecks = true;

    // Mating-technique drive ("mop-up"): with overwhelming material against a near-bare king,
    // plain material/PST give no guidance on HOW to force mate, so the engine can shuffle for
    // many moves before the mate net appears (observed: a won position took ~20 moves to reach
    // a forced mate). This standard term rewards driving the losing king toward the edge/corner
    // (where it has the fewest escape squares) and marching the winning king in to help. Applied
    // only when the losing side is down to at most a lone minor with no pawns and the winning
    // side has a rook or queen -- unambiguous mating material where ANY corner works (unlike the
    // bishop+knight mate, which needs a specific corner, so it is deliberately excluded).
    // Endgame-only (added to eg): when it applies, phase is already near zero so eg dominates.
    private static final int MATE_DRIVE_CENTER = 10;    // per unit of loser-king distance from center (0..6)
    private static final int MATE_DRIVE_PROXIMITY = 4;  // per unit the winning king is closer (0..12)
    public static boolean useMateDrive = true;

    // --- bishop pair / rook activity weights (tapered: separate mg/eg, unlike the flat
    // pawn/mobility/king-safety terms above) ---
    public static int BISHOP_PAIR_MG = 17;
    public static int BISHOP_PAIR_EG = 32;

    public static int ROOK_OPEN_FILE_MG = 21;
    public static int ROOK_OPEN_FILE_EG = 13;
    public static int ROOK_SEMI_OPEN_FILE_MG = 12;
    public static int ROOK_SEMI_OPEN_FILE_EG = 14;
    public static int ROOK_7TH_MG = 20;
    public static int ROOK_7TH_EG = 29;
    // Rook behind a passed pawn (Tarrasch's rule): a rook on the same file as, and behind,
    // an own passed pawn supports its advance the whole way to promotion. Endgame-heavy --
    // the motif is decisive in rook endings, minor in the middlegame. Toggleable for A/B.
    public static int ROOK_BEHIND_PASSER_MG = 5;
    public static int ROOK_BEHIND_PASSER_EG = 20;
    // Default OFF (data-driven, see Search.useCaptureHistory note): chess-theoretically sound
    // and cheap after the rooks!=0 guard, but no referee-gate evidence of gain at fast TC.
    public static boolean useRookBehindPasser = false;

    // Game-phase weight per piece TYPE; full opening phase sums to 24.
    private static final int[] PHASE_INC = {0, 1, 1, 2, 4, 0};
    private static final int PHASE_MAX = 24;

    // Piece-square tables, [type][square], written a8-first (see static init for orientation).
    // Non-final so engine.tools.Tune can discover and fit them (see MG_VALUE note above).
    // Both colors read the same table (black mirrored via sq^56), so per-square tuning
    // preserves color symmetry structurally.
    public static int[][] MG_PST = new int[6][];
    public static int[][] EG_PST = new int[6][];

    // Precomputed file / adjacent-file / passed-pawn masks.
    private static final long[] FILES = new long[8];
    private static final long[] ADJ = new long[8];
    private static final long[] WHITE_PASSED = new long[64];
    private static final long[] BLACK_PASSED = new long[64];

    static {
        for (int f = 0; f < 8; f++) FILES[f] = Attacks.FILE_A << f;
        for (int f = 0; f < 8; f++) {
            long m = 0L;
            if (f > 0) m |= FILES[f - 1];
            if (f < 7) m |= FILES[f + 1];
            ADJ[f] = m;
        }
        for (int sq = 0; sq < 64; sq++) {
            int f = sq & 7, r = sq >> 3;
            long span = FILES[f] | ADJ[f];
            long above = r < 7 ? (-1L << ((r + 1) * 8)) : 0L;
            long below = (1L << (r * 8)) - 1L;
            WHITE_PASSED[sq] = span & above;
            BLACK_PASSED[sq] = span & below;
        }
        initPst();
    }

    // --- packed mg/eg pair ---
    //
    // Every positional term returns its (mg, eg) contribution packed into a single long --
    // mg in the high 32 bits, eg in the low 32 -- instead of a freshly allocated int[2].
    // evaluate() sits on the hottest path in the engine (called at every quiescence node on
    // every search thread), so its term helpers must be allocation-free; packing also keeps
    // the evaluator fully stateless (no shared scratch fields), which is what makes it safe
    // to share across Lazy SMP workers. Terms are always unpacked before being combined:
    // packed values must never be added/subtracted directly, since a borrow/carry in the low
    // (eg) half would silently corrupt the high (mg) half.
    private static long taper(int mg, int eg) {
        return ((long) mg << 32) | (eg & 0xFFFFFFFFL);
    }

    private static int mgOf(long packed) { return (int) (packed >> 32); }

    private static int egOf(long packed) { return (int) packed; }

    // --- pawn-structure cache layout ---
    //
    // The pawn terms (doubled/isolated/passed-rank/connected/backward) depend only on the two
    // pawn bitboards, which change on a small fraction of moves, yet were recomputed at every
    // node -- the single most expensive evaluation term. Callers that own a cache (one per
    // search thread, see engine.search.HandcraftedEvaluator) pass a long[] of
    // PAWN_CACHE_ENTRIES * PAWN_CACHE_STRIDE; each entry stores BOTH full pawn bitboards for
    // exact verification (a hit is bit-for-bit what a recompute would produce -- a hash
    // collision just misses and recomputes, it can never return a wrong score), the packed
    // pawn-only mg/eg pair, and each side's passed-pawn set so the king-proximity passer term
    // (which also depends on the kings) can be recomputed cheaply outside the cache.
    public static final int PAWN_CACHE_ENTRIES = 8192; // power of two
    public static final int PAWN_CACHE_STRIDE = 5;     // wp, bp, packed mg/eg, wPassers, bPassers

    private static int pawnCacheIndex(long wp, long bp) {
        long h = wp * 0x9E3779B97F4A7C15L ^ Long.rotateLeft(bp * 0xC2B2AE3D27D4EB4FL, 32);
        h ^= h >>> 32;
        return ((int) h & (PAWN_CACHE_ENTRIES - 1)) * PAWN_CACHE_STRIDE;
    }

    /** Tapered evaluation from the side-to-move's perspective (uncached pawn terms). */
    public static int evaluate(Position pos) {
        return evaluate(pos, null);
    }

    /**
     * Tapered evaluation from the side-to-move's perspective. {@code pawnCache} may be null
     * (recompute pawn terms) or a per-thread cache array (see the layout notes above); the
     * score is identical either way.
     */
    public static int evaluate(Position pos, long[] pawnCache) {
        int mg = 0, eg = 0, phase = 0;

        for (int type = PAWN; type <= KING; type++) {
            long w = pos.pieces(index(WHITE, type));
            while (w != 0) {
                int sq = Long.numberOfTrailingZeros(w);
                w &= w - 1;
                mg += MG_VALUE[type] + MG_PST[type][sq ^ 56];
                eg += EG_VALUE[type] + EG_PST[type][sq ^ 56];
                phase += PHASE_INC[type];
            }
            long b = pos.pieces(index(BLACK, type));
            while (b != 0) {
                int sq = Long.numberOfTrailingZeros(b);
                b &= b - 1;
                mg -= MG_VALUE[type] + MG_PST[type][sq];
                eg -= EG_VALUE[type] + EG_PST[type][sq];
                phase += PHASE_INC[type];
            }
        }

        long bishopPair = evalBishopPair(pos);
        mg += mgOf(bishopPair);
        eg += egOf(bishopPair);

        long rookActivity = evalRooks(pos);
        mg += mgOf(rookActivity);
        eg += egOf(rookActivity);

        long pawns = evalPawns(pos, pawnCache);
        mg += mgOf(pawns);
        eg += egOf(pawns);

        long mobility = evalMobility(pos);
        mg += mgOf(mobility);
        eg += egOf(mobility);

        long kingSafety = evalKingSafety(pos);
        mg += mgOf(kingSafety);
        eg += egOf(kingSafety);

        long outposts = evalOutposts(pos);
        mg += mgOf(outposts);
        eg += egOf(outposts);

        long kingAttack = evalKingAttack(pos);
        mg += mgOf(kingAttack);
        eg += egOf(kingAttack);

        long space = evalSpace(pos);
        mg += mgOf(space);
        eg += egOf(space);

        long badBishop = evalBadBishop(pos);
        mg += mgOf(badBishop);
        eg += egOf(badBishop);

        long levers = evalLevers(pos);
        mg += mgOf(levers);
        eg += egOf(levers);

        if (useKingRingMobility) {
            long kingRing = evalKingRingMobility(pos);
            mg += mgOf(kingRing);
            eg += egOf(kingRing);
        }

        if (useThreats) {
            long threats = evalThreats(pos);
            mg += mgOf(threats);
            eg += egOf(threats);
        }

        if (useMateDrive) {
            long mateDrive = evalMateDrive(pos);
            mg += mgOf(mateDrive);
            eg += egOf(mateDrive);
        }

        // Linear interpolation core: every term above (material, PST, bishop pair, rook
        // activity, pawn structure, mobility, king safety, outposts) has now fed both an mg
        // and an eg contribution into the same two accumulators, so this single blend is the
        // only place game phase is applied -- nothing bypasses it and reaches the final score
        // unscaled.
        // Drawishness scaling: damp the endgame component for material configurations that are
        // far more drawish than raw material implies (opposite-colored bishops, dead-drawn
        // minor endings). Applied to eg before the blend, so it fades out with the endgame
        // weight itself and never touches middlegame scoring. Symmetric: the scale depends only
        // on (color-symmetric) material, so a mirrored position scales identically.
        int scaledEg = eg * scaleFactor(pos) / SCALE_MAX;

        int mgPhase = Math.min(phase, PHASE_MAX);
        int egPhase = PHASE_MAX - mgPhase;
        int score = (mg * mgPhase + scaledEg * egPhase) / PHASE_MAX;

        // Fifty-move-rule fade (see RULE50_FADE_DIVISOR above). Applied to the blended
        // board score only, never to TEMPO. Symmetric: Java's toward-zero integer division
        // gives (-x)/d == -(x/d), so a mirrored position fades to the exact negation.
        if (useRule50Scaling) {
            score -= score * Math.min(pos.halfmoveClock(), 100) / RULE50_FADE_DIVISOR;
        }

        int stmScore = pos.sideToMove() == WHITE ? score : -score;
        return stmScore + TEMPO;
    }

    /**
     * Endgame scale factor out of {@link #SCALE_MAX}. Conservative by design -- only clearly
     * drawish configurations are damped, so a genuinely winning endgame is never scaled down:
     * opposite-colored bishops (one bishop each on opposite square colors, no other pieces)
     * halve the eg score, and a pawnless ending with at most a single minor per side (which
     * cannot force mate) zeroes it out.
     */
    private static int scaleFactor(Position pos) {
        long wp = pos.pieces(index(WHITE, PAWN));
        long bp = pos.pieces(index(BLACK, PAWN));
        long wn = pos.pieces(index(WHITE, KNIGHT)), bn = pos.pieces(index(BLACK, KNIGHT));
        long wb = pos.pieces(index(WHITE, BISHOP)), bb = pos.pieces(index(BLACK, BISHOP));
        long wr = pos.pieces(index(WHITE, ROOK)), br = pos.pieces(index(BLACK, ROOK));
        long wq = pos.pieces(index(WHITE, QUEEN)), bq = pos.pieces(index(BLACK, QUEEN));

        // Opposite-colored bishops with no other pieces: notoriously drawish even a pawn or two up.
        if ((wn | bn | wr | br | wq | bq) == 0
                && Long.bitCount(wb) == 1 && Long.bitCount(bb) == 1) {
            boolean wLight = isLightSquare(Long.numberOfTrailingZeros(wb));
            boolean bLight = isLightSquare(Long.numberOfTrailingZeros(bb));
            if (wLight != bLight) return SCALE_MAX / 2;
        }

        // Pawnless with at most a single minor on the board TOTAL (KvK, KNvK, KBvK): no mate is
        // possible, a dead draw. Note this is one minor across both sides, not one per side --
        // KNvKN and the like remain fully scored so the engine still prefers better placement
        // and can play on for the opponent's error rather than treating it as a settled draw.
        if ((wp | bp) == 0 && (wr | br | wq | bq) == 0) {
            int minors = Long.bitCount(wn) + Long.bitCount(wb)
                    + Long.bitCount(bn) + Long.bitCount(bb);
            if (minors <= 1) return 0;
        }
        return SCALE_MAX;
    }

    private static boolean isLightSquare(int sq) {
        return (((sq >> 3) + (sq & 7)) & 1) == 1;
    }

    // --- pawn structure (tapered, white-relative) ---

    private static long evalPawns(Position pos, long[] cache) {
        long wp = pos.pieces(index(WHITE, PAWN));
        long bp = pos.pieces(index(BLACK, PAWN));

        long packed;
        long wPassers, bPassers;
        int idx = cache != null ? pawnCacheIndex(wp, bp) : 0;
        if (cache != null && cache[idx] == wp && cache[idx + 1] == bp) {
            // Exact hit (both bitboards match): reuse the pawn-only score and passer sets.
            // A zero-initialized entry can only "match" a genuinely pawnless position, whose
            // correct pawn-only score and passer sets are themselves all zero.
            packed = cache[idx + 2];
            wPassers = cache[idx + 3];
            bPassers = cache[idx + 4];
        } else {
            long w = pawnOnlyScore(wp, bp, WHITE);
            long b = pawnOnlyScore(bp, wp, BLACK);
            packed = taper(mgOf(w) - mgOf(b), egOf(w) - egOf(b));
            wPassers = passerSet(wp, bp, WHITE);
            bPassers = passerSet(bp, wp, BLACK);
            if (cache != null) {
                cache[idx] = wp;
                cache[idx + 1] = bp;
                cache[idx + 2] = packed;
                cache[idx + 3] = wPassers;
                cache[idx + 4] = bPassers;
            }
        }

        // King-proximity passer term: depends on the kings as well as the pawns, so it lives
        // outside the cache, recomputed per call from the (cached) passer sets.
        int wk = pos.kingSquare(WHITE);
        int bk = pos.kingSquare(BLACK);
        int eg = passerKingEg(wPassers, WHITE, wk, bk) - passerKingEg(bPassers, BLACK, bk, wk);
        if (useOutsidePasser) {
            eg += outsidePasserEg(wPassers, WHITE, bk) - outsidePasserEg(bPassers, BLACK, wk);
        }
        if (usePawnMajority) {
            eg += flankMajorityEg(wp, bp, bk) - flankMajorityEg(bp, wp, wk);
        }
        return taper(mgOf(packed), egOf(packed) + eg);
    }

    /** Bitboard of {@code color}'s passed pawns. */
    private static long passerSet(long own, long enemy, int color) {
        long passers = 0L;
        long b = own;
        while (b != 0) {
            int sq = Long.numberOfTrailingZeros(b);
            b &= b - 1;
            long mask = color == WHITE ? WHITE_PASSED[sq] : BLACK_PASSED[sq];
            if ((enemy & mask) == 0) passers |= 1L << sq;
        }
        return passers;
    }

    /** Endgame king-proximity bonus over {@code color}'s passers (see PASSER_KING_EG). */
    private static int passerKingEg(long passers, int color, int ownKing, int enemyKing) {
        int eg = 0;
        while (passers != 0) {
            int sq = Long.numberOfTrailingZeros(passers);
            passers &= passers - 1;
            int f = sq & 7;
            int rel = color == WHITE ? (sq >> 3) : 7 - (sq >> 3);
            int promoSq = color == WHITE ? (56 + f) : f;
            int prox = chebyshev(enemyKing, promoSq) - chebyshev(ownKing, promoSq);
            eg += PASSER_KING_EG * prox * rel / 6;
        }
        return eg;
    }

    /** Endgame bonus for {@code color}'s passers that sit far (by file) from the enemy king --
     *  an "outside" passer the king can't easily stop. Scaled by the passer's advancement,
     *  mirroring {@link #passerKingEg}. See OUTSIDE_PASSER_EG / OUTSIDE_KING_DIST. */
    private static int outsidePasserEg(long passers, int color, int enemyKing) {
        int eg = 0;
        int ekFile = enemyKing & 7;
        while (passers != 0) {
            int sq = Long.numberOfTrailingZeros(passers);
            passers &= passers - 1;
            int f = sq & 7;
            int rel = color == WHITE ? (sq >> 3) : 7 - (sq >> 3);
            if (Math.abs(f - ekFile) >= OUTSIDE_KING_DIST) {
                eg += OUTSIDE_PASSER_EG * rel / 6;
            }
        }
        return eg;
    }

    /** Endgame bonus for a pawn majority on the flank away from the enemy king -- an outside
     *  majority that can produce a passer the king is too distant to catch. See PAWN_MAJORITY_EG. */
    private static int flankMajorityEg(long ownPawns, long enemyPawns, int enemyKing) {
        long farFlank = (enemyKing & 7) >= 4 ? QUEENSIDE_FILES : KINGSIDE_FILES;
        int own = Long.bitCount(ownPawns & farFlank);
        int enemy = Long.bitCount(enemyPawns & farFlank);
        return own > enemy ? PAWN_MAJORITY_EG * (own - enemy) : 0;
    }

    /** Pawn-structure terms that depend ONLY on the two pawn bitboards (cacheable): doubled,
     *  isolated, passed-rank, connected, backward. The king-proximity passer term lives in
     *  {@link #passerKingEg}. */
    private static long pawnOnlyScore(long own, long enemy, int color) {
        int mg = 0, eg = 0;
        for (int f = 0; f < 8; f++) {
            int cnt = Long.bitCount(own & FILES[f]);
            if (cnt > 1) {
                mg -= DOUBLED_MG * (cnt - 1);
                eg -= DOUBLED_EG * (cnt - 1);
            }
            if (cnt > 0 && (own & ADJ[f]) == 0) {
                mg -= ISOLATED_MG * cnt;
                eg -= ISOLATED_EG * cnt;
            }
        }
        long enemyPawnAtt = pawnAttacks(enemy, 1 - color);
        long b = own;
        while (b != 0) {
            int sq = Long.numberOfTrailingZeros(b);
            b &= b - 1;
            int f = sq & 7;
            int rel = color == WHITE ? (sq >> 3) : 7 - (sq >> 3);

            long mask = color == WHITE ? WHITE_PASSED[sq] : BLACK_PASSED[sq];
            if ((enemy & mask) == 0) {
                mg += PASSED_MG[rel];
                eg += PASSED_EG[rel];
            }

            // Connected/phalanx: a friendly pawn beside this one (same rank) or defending it
            // (one rank back on an adjacent file) makes a mutually-supporting pawn chain.
            long adjFiles = ADJ[f];
            long sameRank = adjFiles & rankMask(sq >> 3);
            long behindRank = color == WHITE ? rankMask((sq >> 3) - 1) : rankMask((sq >> 3) + 1);
            if ((own & sameRank) != 0 || (own & adjFiles & behindRank) != 0) {
                mg += CONNECTED_MG;
                eg += CONNECTED_EG;
            }

            // Backward: no friendly pawn on adjacent files at or behind this pawn's rank (so it
            // can't be supported from behind), and the square in front is controlled by an
            // enemy pawn (so it can't safely advance) -- a lasting weakness.
            long supportZone = color == WHITE ? (adjFiles & atOrBelow(sq >> 3))
                                              : (adjFiles & atOrAbove(sq >> 3));
            int stopSq = color == WHITE ? sq + 8 : sq - 8;
            boolean stopAttacked = stopSq >= 0 && stopSq < 64
                    && (enemyPawnAtt & (1L << stopSq)) != 0;
            if ((own & supportZone) == 0 && stopAttacked) {
                mg -= BACKWARD_MG;
                eg -= BACKWARD_EG;
            }
        }
        return taper(mg, eg);
    }

    private static long rankMask(int rank) {
        return rank >= 0 && rank < 8 ? (0xFFL << (rank * 8)) : 0L;
    }

    /** All ranks at or below (numerically <=) {@code rank}, i.e. rank and everything toward rank 1. */
    private static long atOrBelow(int rank) {
        long m = 0L;
        for (int r = 0; r <= rank && r < 8; r++) m |= 0xFFL << (r * 8);
        return m;
    }

    /** All ranks at or above (numerically >=) {@code rank}, i.e. rank and everything toward rank 8. */
    private static long atOrAbove(int rank) {
        long m = 0L;
        for (int r = rank; r < 8; r++) if (r >= 0) m |= 0xFFL << (r * 8);
        return m;
    }

    /** Chebyshev (king-move) distance between two squares. */
    private static int chebyshev(int a, int b) {
        int df = Math.abs((a & 7) - (b & 7));
        int dr = Math.abs((a >> 3) - (b >> 3));
        return Math.max(df, dr);
    }

    // --- king attack / danger (tapered, white-relative) ---

    private static long evalKingAttack(Position pos) {
        // A king in danger is a penalty to ITS side, so the enemy's danger is the friendly
        // side's gain: white-relative term = (danger to black king) - (danger to white king).
        long wDanger = kingDanger(pos, WHITE);
        long bDanger = kingDanger(pos, BLACK);
        return taper(mgOf(bDanger) - mgOf(wDanger), egOf(bDanger) - egOf(wDanger));
    }

    /** Attack-unit danger (>= 0) to {@code kingColor}'s king from the opposing pieces. */
    private static long kingDanger(Position pos, int kingColor) {
        int ksq = pos.kingSquare(kingColor);
        long zone = Attacks.KING[ksq] | (1L << ksq);
        int enemy = 1 - kingColor;
        long occ = pos.occupied();
        int units = 0, attackers = 0;

        // Safe-check target squares by checking-piece movement type: empty squares a piece
        // could move to and give check from, not covered by a defending pawn and not adjacent
        // to the king itself (see SAFE_CHECK_WEIGHT above). Queens use both slider sets.
        long knightChecks = 0L, bishopChecks = 0L, rookChecks = 0L;
        if (useSafeChecks) {
            long safeMask = ~pawnAttacks(pos.pieces(index(kingColor, PAWN)), kingColor)
                    & ~Attacks.KING[ksq] & ~occ;
            knightChecks = Attacks.KNIGHT[ksq] & safeMask;
            bishopChecks = Attacks.bishop(ksq, occ) & safeMask;
            rookChecks = Attacks.rook(ksq, occ) & safeMask;
        }

        long n = pos.pieces(index(enemy, KNIGHT));
        while (n != 0) {
            int sq = Long.numberOfTrailingZeros(n);
            n &= n - 1;
            long att = Attacks.KNIGHT[sq];
            int c = Long.bitCount(att & zone);
            if (c > 0) { units += KING_ATTACK_WEIGHT[KNIGHT] * c; attackers++; }
            units += SAFE_CHECK_WEIGHT[KNIGHT] * Long.bitCount(att & knightChecks);
        }
        long b = pos.pieces(index(enemy, BISHOP));
        while (b != 0) {
            int sq = Long.numberOfTrailingZeros(b);
            b &= b - 1;
            long att = Attacks.bishop(sq, occ);
            int c = Long.bitCount(att & zone);
            if (c > 0) { units += KING_ATTACK_WEIGHT[BISHOP] * c; attackers++; }
            units += SAFE_CHECK_WEIGHT[BISHOP] * Long.bitCount(att & bishopChecks);
        }
        long r = pos.pieces(index(enemy, ROOK));
        while (r != 0) {
            int sq = Long.numberOfTrailingZeros(r);
            r &= r - 1;
            long att = Attacks.rook(sq, occ);
            int c = Long.bitCount(att & zone);
            if (c > 0) { units += KING_ATTACK_WEIGHT[ROOK] * c; attackers++; }
            units += SAFE_CHECK_WEIGHT[ROOK] * Long.bitCount(att & rookChecks);
        }
        long q = pos.pieces(index(enemy, QUEEN));
        while (q != 0) {
            int sq = Long.numberOfTrailingZeros(q);
            q &= q - 1;
            long att = Attacks.queen(sq, occ);
            int c = Long.bitCount(att & zone);
            if (c > 0) { units += KING_ATTACK_WEIGHT[QUEEN] * c; attackers++; }
            units += SAFE_CHECK_WEIGHT[QUEEN] * Long.bitCount(att & (bishopChecks | rookChecks));
        }

        // A lone attacker near the king isn't a real threat; danger is only credited once at
        // least two enemy pieces bear on the zone, and grows super-linearly from there.
        if (attackers < 2) return taper(0, 0);
        int penalty = Math.min(units * units / KING_ATTACK_DIVISOR, KING_ATTACK_CAP);
        return taper(penalty, penalty >> KING_ATTACK_EG_SHIFT);
    }

    // --- space (mg only, white-relative) ---

    private static long evalSpace(Position pos) {
        int w = spaceScore(pos, WHITE);
        int b = spaceScore(pos, BLACK);
        return taper(w - b, 0); // mg only
    }

    private static int spaceScore(Position pos, int color) {
        long ownPawns = pos.pieces(index(color, PAWN));
        long enemyPawns = pos.pieces(index(1 - color, PAWN));
        long enemyPawnAtt = pawnAttacks(enemyPawns, 1 - color);
        long areaMask = color == WHITE ? SPACE_MASK_WHITE : SPACE_MASK_BLACK;

        // Safe = central own-half squares that aren't own pawns and aren't hit by enemy pawns.
        long safe = areaMask & ~ownPawns & ~enemyPawnAtt;
        // "Behind a pawn": one or two ranks in front (from this side's view) sits an own pawn --
        // such squares are extra-sheltered space. Shift own pawns back toward our side and mask.
        long behind = color == WHITE
                ? ((ownPawns >>> 8) | (ownPawns >>> 16))
                : ((ownPawns << 8) | (ownPawns << 16));
        int bonus = Long.bitCount(safe) + Long.bitCount(safe & behind);

        int pieces = Long.bitCount(pos.pieces(index(color, KNIGHT)))
                + Long.bitCount(pos.pieces(index(color, BISHOP)))
                + Long.bitCount(pos.pieces(index(color, ROOK)))
                + Long.bitCount(pos.pieces(index(color, QUEEN)));
        // Super-linear in piece count: space matters far more with a full board than a thin one.
        return bonus * pieces * pieces / SPACE_DIVISOR;
    }

    // --- bad bishop (tapered, white-relative) ---

    private static long evalBadBishop(Position pos) {
        long w = badBishopPenalty(pos, WHITE);
        long b = badBishopPenalty(pos, BLACK);
        // A bad bishop hurts its owner, so the enemy's bad bishop is our gain.
        return taper(mgOf(b) - mgOf(w), egOf(b) - egOf(w));
    }

    private static long badBishopPenalty(Position pos, int color) {
        long bishops = pos.pieces(index(color, BISHOP));
        if (bishops == 0) return taper(0, 0);
        long ownPawns = pos.pieces(index(color, PAWN));
        int lightPawns = Long.bitCount(ownPawns & LIGHT_SQUARES);
        int darkPawns = Long.bitCount(ownPawns & ~LIGHT_SQUARES);
        int lightBishops = Long.bitCount(bishops & LIGHT_SQUARES);
        int darkBishops = Long.bitCount(bishops & ~LIGHT_SQUARES);
        // Each bishop is dragged down by the own pawns fixed on its own square color.
        int units = lightBishops * lightPawns + darkBishops * darkPawns;
        return taper(units * BAD_BISHOP_MG, units * BAD_BISHOP_EG);
    }

    // --- weighted mobility (tapered, white-relative) ---
    //
    // "Ambition" weighting: every reachable square still earns the flat MOBILITY_MG/EG rate,
    // but a square that represents genuine pressure -- a rook/queen move landing on a
    // pawnless (open) file, or a knight/bishop move landing on a center square -- earns an
    // additional top-up on top of that base rate (see the *_MOBILITY_* constants above).
    // openFiles is computed once per evaluate() call and shared by both colors' rook/queen
    // loops rather than recomputed per piece.

    private static long evalMobility(Position pos) {
        long occ = pos.occupied();
        long wp = pos.pieces(index(WHITE, PAWN));
        long bp = pos.pieces(index(BLACK, PAWN));
        long allPawns = wp | bp;
        long openFiles = 0L;
        for (int f = 0; f < 8; f++) {
            if ((allPawns & FILES[f]) == 0) openFiles |= FILES[f];
        }
        long whitePawnAtt = pawnAttacks(wp, WHITE);
        long blackPawnAtt = pawnAttacks(bp, BLACK);
        // A square attacked by an enemy pawn is not real mobility -- a piece there is just
        // hanging to the pawn -- so it is excluded from each side's mobility area.
        long w = mobilityScore(pos, WHITE, occ, openFiles, blackPawnAtt);
        long b = mobilityScore(pos, BLACK, occ, openFiles, whitePawnAtt);
        return taper(mgOf(w) - mgOf(b), egOf(w) - egOf(b));
    }

    // --- pawn levers / breaks (tapered, white-relative) ---

    private static long evalLevers(Position pos) {
        int diff = leverCount(pos, WHITE) - leverCount(pos, BLACK);
        return taper(diff * LEVER_MG, diff * LEVER_EG);
    }

    /** Count of {@code color}'s pawn levers: own pawns that already attack an enemy pawn, plus
     *  own pawns a single (unobstructed) push away from attacking one. */
    private static int leverCount(Position pos, int color) {
        long own = pos.pieces(index(color, PAWN));
        long enemy = pos.pieces(index(1 - color, PAWN));
        long empty = ~pos.occupied();
        // Pawns whose current attacks land on an enemy pawn: pull the enemy-pawn set back onto
        // the attacking pawns (an own pawn attacks an enemy iff the enemy is in own's attacks).
        long attackNow = pawnAttacks(own, color) & enemy;
        int direct = Long.bitCount(attackNow);
        // Pawns one push from attacking: push each pawn one square if empty, then see whose
        // resulting attacks hit an enemy pawn.
        long pushed = color == WHITE ? (own << 8) & empty : (own >>> 8) & empty;
        long attackAfterPush = pawnAttacks(pushed, color) & enemy;
        int levers = direct + Long.bitCount(attackAfterPush);
        return levers;
    }

    // --- threats (tapered, white-relative) ---

    private static long evalThreats(Position pos) {
        long occ = pos.occupied();
        // Full attack unions, computed once and shared by both sides' hanging-piece checks.
        long wAtt = allAttacks(pos, WHITE, occ);
        long bAtt = allAttacks(pos, BLACK, occ);
        long w = threatScore(pos, WHITE, wAtt, bAtt);
        long b = threatScore(pos, BLACK, bAtt, wAtt);
        return taper(mgOf(w) - mgOf(b), egOf(w) - egOf(b));
    }

    /** Threats {@code color} exerts on the enemy: pawn attacks on pieces, hanging enemy
     *  pieces, and safe pawn pushes that would attack a piece next move. */
    private static long threatScore(Position pos, int color, long ownAtt, long enemyAtt) {
        int them = 1 - color;
        int mg = 0, eg = 0;
        long ownPawns = pos.pieces(index(color, PAWN));
        long enemyPawns = pos.pieces(index(them, PAWN));
        long ownPawnAtt = pawnAttacks(ownPawns, color);

        // (a) Enemy pieces (never the king: attacks on it are the king-danger terms' job)
        // currently attacked by one of our pawns, weighted by victim type.
        for (int type = KNIGHT; type <= QUEEN; type++) {
            int cnt = Long.bitCount(ownPawnAtt & pos.pieces(index(them, type)));
            if (cnt > 0) {
                mg += THREAT_BY_PAWN_MG[type] * cnt;
                eg += THREAT_BY_PAWN_EG[type] * cnt;
            }
        }

        // (b) Hanging: enemy non-pawn, non-king pieces attacked by anything of ours and
        // defended by nothing of theirs.
        long enemyValuable = pos.pieces(index(them, KNIGHT)) | pos.pieces(index(them, BISHOP))
                | pos.pieces(index(them, ROOK)) | pos.pieces(index(them, QUEEN));
        int hanging = Long.bitCount(enemyValuable & ownAtt & ~enemyAtt);
        mg += HANGING_MG * hanging;
        eg += HANGING_EG * hanging;

        // (c) Safe pawn pushes that would attack an enemy piece: single pushes to an empty
        // square not covered by an enemy pawn, whose resulting attacks hit a piece.
        long empty = ~pos.occupied();
        long pushed = color == WHITE ? (ownPawns << 8) & empty : (ownPawns >>> 8) & empty;
        pushed &= ~pawnAttacks(enemyPawns, them);
        int pushThreats = Long.bitCount(pawnAttacks(pushed, color) & enemyValuable);
        mg += PAWN_PUSH_THREAT_MG * pushThreats;
        eg += PAWN_PUSH_THREAT_EG * pushThreats;

        return taper(mg, eg);
    }

    /** Union of every square {@code color} attacks (occupancy-aware sliders, no allocation). */
    private static long allAttacks(Position pos, int color, long occ) {
        long att = pawnAttacks(pos.pieces(index(color, PAWN)), color);
        long n = pos.pieces(index(color, KNIGHT));
        while (n != 0) {
            int sq = Long.numberOfTrailingZeros(n);
            n &= n - 1;
            att |= Attacks.KNIGHT[sq];
        }
        long b = pos.pieces(index(color, BISHOP));
        while (b != 0) {
            int sq = Long.numberOfTrailingZeros(b);
            b &= b - 1;
            att |= Attacks.bishop(sq, occ);
        }
        long r = pos.pieces(index(color, ROOK));
        while (r != 0) {
            int sq = Long.numberOfTrailingZeros(r);
            r &= r - 1;
            att |= Attacks.rook(sq, occ);
        }
        long q = pos.pieces(index(color, QUEEN));
        while (q != 0) {
            int sq = Long.numberOfTrailingZeros(q);
            q &= q - 1;
            att |= Attacks.queen(sq, occ);
        }
        att |= Attacks.KING[pos.kingSquare(color)];
        return att;
    }

    // --- mating-technique drive (endgame-only, white-relative) ---

    private static long evalMateDrive(Position pos) {
        int winner = overwhelmingSide(pos);
        if (winner == -1) return 0L;
        int loser = 1 - winner;
        int loserKing = pos.kingSquare(loser);
        int winnerKing = pos.kingSquare(winner);
        // Drive the losing king off-center (fewer escape squares near the edge/corner) and the
        // winning king toward it (needed to support the mate). centerDistance is 0..6 (corner),
        // king manhattan distance is 2..14 so (14 - dist) is 0..12.
        int score = MATE_DRIVE_CENTER * centerDistance(loserKing)
                + MATE_DRIVE_PROXIMITY * (14 - manhattan(winnerKing, loserKing));
        int wRel = winner == WHITE ? score : -score;
        return taper(0, wRel); // endgame-only: no mg component
    }

    /** WHITE or BLACK if that side has overwhelming material against a near-bare enemy king
     *  (unambiguous rook/queen mate), else -1. */
    private static int overwhelmingSide(Position pos) {
        if (isNearBareKing(pos, BLACK) && hasRookOrQueen(pos, WHITE)) return WHITE;
        if (isNearBareKing(pos, WHITE) && hasRookOrQueen(pos, BLACK)) return BLACK;
        return -1;
    }

    /** True if {@code side} is down to its king plus at most a single minor, with no pawns --
     *  i.e. it cannot meaningfully defend against a rook/queen mate. */
    private static boolean isNearBareKing(Position pos, int side) {
        if (pos.pieces(index(side, PAWN)) != 0) return false;
        if (pos.pieces(index(side, ROOK)) != 0) return false;
        if (pos.pieces(index(side, QUEEN)) != 0) return false;
        int minors = Long.bitCount(pos.pieces(index(side, KNIGHT)))
                + Long.bitCount(pos.pieces(index(side, BISHOP)));
        return minors <= 1;
    }

    private static boolean hasRookOrQueen(Position pos, int side) {
        return (pos.pieces(index(side, ROOK)) | pos.pieces(index(side, QUEEN))) != 0;
    }

    /** Manhattan distance of {@code sq} from the central 2x2 (d4/d5/e4/e5): 0 at the center,
     *  3 on an edge-center square, 6 in a corner. Invariant under the vertical flip used by the
     *  color-symmetry mirror, so the term stays perfectly antisymmetric. */
    private static int centerDistance(int sq) {
        int f = sq & 7, r = sq >> 3;
        int fd = f <= 3 ? 3 - f : f - 4;
        int rd = r <= 3 ? 3 - r : r - 4;
        return fd + rd;
    }

    private static int manhattan(int a, int b) {
        return Math.abs((a & 7) - (b & 7)) + Math.abs((a >> 3) - (b >> 3));
    }

    /** All squares attacked by {@code color}'s pawns (file-wrap masked). */
    private static long pawnAttacks(long pawns, int color) {
        if (color == WHITE) {
            return ((pawns & ~Attacks.FILE_A) << 7) | ((pawns & ~Attacks.FILE_H) << 9);
        }
        return ((pawns & ~Attacks.FILE_A) >>> 9) | ((pawns & ~Attacks.FILE_H) >>> 7);
    }

    private static long mobilityScore(Position pos, int color, long occ, long openFiles,
                                      long enemyPawnAtt) {
        long notOwn = ~pos.occupied(color); // any square not blocked by an own piece
        int mg = 0, eg = 0;

        long n = pos.pieces(index(color, KNIGHT));
        while (n != 0) {
            int sq = Long.numberOfTrailingZeros(n);
            n &= n - 1;
            long reach = Attacks.KNIGHT[sq] & notOwn;
            long safe = reach & ~enemyPawnAtt;
            int cnt = flooredMobility(reach, safe);
            int centerCnt = Long.bitCount(safe & CENTER_SQUARES);
            mg += MOBILITY_MG[KNIGHT] * cnt + KNIGHT_CENTER_MOBILITY_MG * centerCnt;
            eg += MOBILITY_EG[KNIGHT] * cnt + KNIGHT_CENTER_MOBILITY_EG * centerCnt;
        }
        long bsh = pos.pieces(index(color, BISHOP));
        while (bsh != 0) {
            int sq = Long.numberOfTrailingZeros(bsh);
            bsh &= bsh - 1;
            long reach = Attacks.bishop(sq, occ) & notOwn;
            long safe = reach & ~enemyPawnAtt;
            int cnt = flooredMobility(reach, safe);
            int centerCnt = Long.bitCount(safe & CENTER_SQUARES);
            mg += MOBILITY_MG[BISHOP] * cnt + BISHOP_CENTER_MOBILITY_MG * centerCnt;
            eg += MOBILITY_EG[BISHOP] * cnt + BISHOP_CENTER_MOBILITY_EG * centerCnt;
        }
        long r = pos.pieces(index(color, ROOK));
        while (r != 0) {
            int sq = Long.numberOfTrailingZeros(r);
            r &= r - 1;
            long reach = Attacks.rook(sq, occ) & notOwn;
            long safe = reach & ~enemyPawnAtt;
            int cnt = flooredMobility(reach, safe);
            int openCnt = Long.bitCount(safe & openFiles);
            mg += MOBILITY_MG[ROOK] * cnt + ROOK_OPEN_FILE_MOBILITY_MG * openCnt;
            eg += MOBILITY_EG[ROOK] * cnt + ROOK_OPEN_FILE_MOBILITY_EG * openCnt;
        }
        long q = pos.pieces(index(color, QUEEN));
        while (q != 0) {
            int sq = Long.numberOfTrailingZeros(q);
            q &= q - 1;
            long reach = Attacks.queen(sq, occ) & notOwn;
            long safe = reach & ~enemyPawnAtt;
            int cnt = flooredMobility(reach, safe);
            int openCnt = Long.bitCount(safe & openFiles);
            mg += MOBILITY_MG[QUEEN] * cnt + QUEEN_OPEN_FILE_MOBILITY_MG * openCnt;
            eg += MOBILITY_EG[QUEEN] * cnt + QUEEN_OPEN_FILE_MOBILITY_EG * openCnt;
        }
        return taper(mg, eg);
    }

    /** Mobility count with the floor: squares safe from enemy pawns count fully; squares only
     *  an enemy pawn contests still count, at reduced (1/MOBILITY_MASK_DIVISOR) weight, so a
     *  closed structure can't zero the term out. {@code safe} is a subset of {@code reach}. */
    private static int flooredMobility(long reach, long safe) {
        int full = Long.bitCount(safe);
        int contested = Long.bitCount(reach) - full;
        return full + contested / MOBILITY_MASK_DIVISOR;
    }

    // --- central outposts (tapered, white-relative) ---

    private static long evalOutposts(Position pos) {
        long w = outpostScore(pos, WHITE);
        long b = outpostScore(pos, BLACK);
        return taper(mgOf(w) - mgOf(b), egOf(w) - egOf(b));
    }

    private static long outpostScore(Position pos, int color) {
        int mg = 0, eg = 0;
        long enemyPawns = pos.pieces(index(1 - color, PAWN));

        long knights = pos.pieces(index(color, KNIGHT)) & CENTER_SQUARES;
        while (knights != 0) {
            int sq = Long.numberOfTrailingZeros(knights);
            knights &= knights - 1;
            if (isSafeOutpost(sq, color, enemyPawns)) {
                mg += OUTPOST_KNIGHT_MG;
                eg += OUTPOST_KNIGHT_EG;
            }
        }
        long bishops = pos.pieces(index(color, BISHOP)) & CENTER_SQUARES;
        while (bishops != 0) {
            int sq = Long.numberOfTrailingZeros(bishops);
            bishops &= bishops - 1;
            if (isSafeOutpost(sq, color, enemyPawns)) {
                mg += OUTPOST_BISHOP_MG;
                eg += OUTPOST_BISHOP_EG;
            }
        }
        return taper(mg, eg);
    }

    /** True if no enemy pawn -- now or ever, structurally -- can capture onto {@code sq}: no
     *  enemy pawn sits on an adjacent file at a rank still "ahead" of {@code sq} from the
     *  piece's own side. Reuses WHITE_PASSED/BLACK_PASSED (already "adjacent-files-plus-own-
     *  file, ranks ahead") intersected with ADJ[f] to drop the irrelevant own-file component,
     *  rather than computing a dedicated mask. */
    private static boolean isSafeOutpost(int sq, int color, long enemyPawns) {
        int f = sq & 7;
        long aheadOnAdjFiles = ADJ[f] & (color == WHITE ? WHITE_PASSED[sq] : BLACK_PASSED[sq]);
        return (enemyPawns & aheadOnAdjFiles) == 0;
    }

    // --- bishop pair (tapered, white-relative) ---

    private static long evalBishopPair(Position pos) {
        int mg = 0, eg = 0;
        if (Long.bitCount(pos.pieces(index(WHITE, BISHOP))) >= 2) {
            mg += BISHOP_PAIR_MG;
            eg += BISHOP_PAIR_EG;
        }
        if (Long.bitCount(pos.pieces(index(BLACK, BISHOP))) >= 2) {
            mg -= BISHOP_PAIR_MG;
            eg -= BISHOP_PAIR_EG;
        }
        return taper(mg, eg);
    }

    // --- rook file activity + 7th/8th-rank infiltration (tapered, white-relative) ---

    private static long evalRooks(Position pos) {
        long w = rookScore(pos, WHITE);
        long b = rookScore(pos, BLACK);
        return taper(mgOf(w) - mgOf(b), egOf(w) - egOf(b));
    }

    private static long rookScore(Position pos, int color) {
        int mg = 0, eg = 0;
        long ownPawns = pos.pieces(index(color, PAWN));
        long enemyPawns = pos.pieces(index(1 - color, PAWN));
        long rooks = pos.pieces(index(color, ROOK));
        // Own passed pawns (no enemy pawn on the file or adjacent files ahead), reused for the
        // rook-behind-passer test below. Computed only when the term is active AND this side
        // actually has a rook -- otherwise the scan is pure waste on every eval call.
        long ownPassers = 0L;
        if (useRookBehindPasser && rooks != 0) {
            long p = ownPawns;
            while (p != 0) {
                int psq = Long.numberOfTrailingZeros(p);
                p &= p - 1;
                long mask = color == WHITE ? WHITE_PASSED[psq] : BLACK_PASSED[psq];
                if ((enemyPawns & mask) == 0) ownPassers |= 1L << psq;
            }
        }
        while (rooks != 0) {
            int sq = Long.numberOfTrailingZeros(rooks);
            rooks &= rooks - 1;
            int f = sq & 7;
            // Rook behind an own passer: a passer on this file, on a square ahead of the rook
            // (higher rank for white, lower for black).
            if (useRookBehindPasser && (ownPassers & FILES[f]) != 0) {
                int r = sq >> 3;
                long aheadOnFile = FILES[f] & (color == WHITE
                        ? (r < 7 ? (-1L << ((r + 1) * 8)) : 0L)
                        : ((1L << (r * 8)) - 1L));
                if ((ownPassers & aheadOnFile) != 0) {
                    mg += ROOK_BEHIND_PASSER_MG;
                    eg += ROOK_BEHIND_PASSER_EG;
                }
            }
            boolean ownOnFile = (ownPawns & FILES[f]) != 0;
            boolean enemyOnFile = (enemyPawns & FILES[f]) != 0;
            if (!ownOnFile && !enemyOnFile) {
                mg += ROOK_OPEN_FILE_MG;
                eg += ROOK_OPEN_FILE_EG;
            } else if (!ownOnFile) {
                mg += ROOK_SEMI_OPEN_FILE_MG;
                eg += ROOK_SEMI_OPEN_FILE_EG;
            }
            // Same relative-rank convention as PASSED above: rel 6/7 is the 7th/8th rank
            // from this rook's own side, i.e. deep in (or on) the enemy's back ranks.
            int rel = color == WHITE ? (sq >> 3) : 7 - (sq >> 3);
            if (rel >= 6) {
                mg += ROOK_7TH_MG;
                eg += ROOK_7TH_EG;
            }
        }
        return taper(mg, eg);
    }

    // --- king safety (tapered, white-relative) ---

    private static long evalKingSafety(Position pos) {
        long w = kingSafetyScore(pos, WHITE);
        long b = kingSafetyScore(pos, BLACK);
        return taper(mgOf(w) - mgOf(b), egOf(w) - egOf(b));
    }

    private static long kingSafetyScore(Position pos, int color) {
        int mg = 0, eg = 0;
        int ksq = pos.kingSquare(color);
        int kf = ksq & 7;
        long ownPawns = pos.pieces(index(color, PAWN));
        long enemyPawns = pos.pieces(index(1 - color, PAWN));

        // Pawn shield: own pawns on the king's file and adjacent files, ahead of the king.
        long zone = FILES[kf] | ADJ[kf];
        long ahead = color == WHITE ? WHITE_PASSED[ksq] : BLACK_PASSED[ksq];
        long shield = ownPawns & zone & ahead;
        int shieldCount = Long.bitCount(shield);
        mg += SHIELD_MG * shieldCount;
        eg += SHIELD_EG * shieldCount;

        // Pawn storm: enemy pawns on the king's file or adjacent files, ahead of the king
        // and within STORM_MAX_DIST ranks, penalized more the closer they loom (see the
        // STORM_* constants above). Uses the same `zone` mask as the shield.
        if (useKingPawnStorm) {
            long stormers = enemyPawns & zone;
            int kRank = ksq >> 3;
            while (stormers != 0) {
                int sq = Long.numberOfTrailingZeros(stormers);
                stormers &= stormers - 1;
                int dist = color == WHITE ? (sq >> 3) - kRank : kRank - (sq >> 3);
                if (dist >= 1 && dist <= STORM_MAX_DIST) {
                    mg -= STORM_MG[dist];
                    eg -= STORM_EG[dist];
                }
            }
        }

        // Open / semi-open files next to the king are dangerous mainly while enemy
        // heavy pieces are still on the board to exploit them (mg-heavy, see the
        // SHIELD/OPEN_FILE/SEMI_OPEN_FILE constants above).
        int lo = Math.max(0, kf - 1), hi = Math.min(7, kf + 1);
        for (int f = lo; f <= hi; f++) {
            boolean ownOnFile = (ownPawns & FILES[f]) != 0;
            boolean enemyOnFile = (enemyPawns & FILES[f]) != 0;
            if (!ownOnFile && !enemyOnFile) {
                mg -= OPEN_FILE_MG;
                eg -= OPEN_FILE_EG;
            } else if (!ownOnFile) {
                mg -= SEMI_OPEN_FILE_MG;
                eg -= SEMI_OPEN_FILE_EG;
            }
        }
        return taper(mg, eg);
    }

    // --- king-ring mobility (tapered, white-relative) ---

    /**
     * White-relative king-ring control: {@code ringControl(WHITE) - ringControl(BLACK)},
     * scaled into a tapered mg/eg pair. Perfectly color-symmetric -- both sides are measured
     * by the identical procedure and differenced -- so it preserves {@code evaluate(p) ==
     * -evaluate(mirror(p))} and stays 0 in a mirror-symmetric position such as the start.
     */
    private static long evalKingRingMobility(Position pos) {
        int diff = ringControl(pos, WHITE) - ringControl(pos, BLACK);
        return taper(diff * KING_RING_MG, diff * KING_RING_EG);
    }

    /**
     * Net attack density over {@code kingColor}'s own 3x3 king ring: total bit-attacks that
     * {@code kingColor}'s pieces land inside the ring (friendly coverage) minus total
     * bit-attacks the enemy lands inside it (assault pressure). Positive = the king's own side
     * dominates the squares around it.
     */
    private static int ringControl(Position pos, int kingColor) {
        int ksq = pos.kingSquare(kingColor);
        long ring = Attacks.KING[ksq] | (1L << ksq); // 3x3 bounding box, clipped at board edges
        return attacksInto(pos, kingColor, ring) - attacksInto(pos, 1 - kingColor, ring);
    }

    /**
     * Total bit-attacks from every {@code color} piece that land inside {@code ring}. Attackers
     * are summed independently, so two pieces hitting the same ring square count twice -- this
     * is deliberately attack *density*, not unique-square coverage. Uses only the precomputed
     * leaper tables and the occupancy-aware slider attacks; no allocation in the loops.
     */
    private static int attacksInto(Position pos, int color, long ring) {
        long occ = pos.occupied();
        int count = 0;

        long p = pos.pieces(index(color, PAWN));
        while (p != 0) {
            int sq = Long.numberOfTrailingZeros(p);
            p &= p - 1;
            count += Long.bitCount(Attacks.PAWN[color][sq] & ring);
        }
        long n = pos.pieces(index(color, KNIGHT));
        while (n != 0) {
            int sq = Long.numberOfTrailingZeros(n);
            n &= n - 1;
            count += Long.bitCount(Attacks.KNIGHT[sq] & ring);
        }
        long b = pos.pieces(index(color, BISHOP));
        while (b != 0) {
            int sq = Long.numberOfTrailingZeros(b);
            b &= b - 1;
            count += Long.bitCount(Attacks.bishop(sq, occ) & ring);
        }
        long r = pos.pieces(index(color, ROOK));
        while (r != 0) {
            int sq = Long.numberOfTrailingZeros(r);
            r &= r - 1;
            count += Long.bitCount(Attacks.rook(sq, occ) & ring);
        }
        long q = pos.pieces(index(color, QUEEN));
        while (q != 0) {
            int sq = Long.numberOfTrailingZeros(q);
            q &= q - 1;
            count += Long.bitCount(Attacks.queen(sq, occ) & ring);
        }
        // The king defends its own ring; the enemy king is virtually never adjacent to it.
        count += Long.bitCount(Attacks.KING[pos.kingSquare(color)] & ring);

        return count;
    }

    // --- piece-square table data ---

    /**
     * Tables are written human-readably with rank 8 first (index 0 = a8, index 63 = h1),
     * i.e. the same orientation used in the Chess Programming Wiki / PeSTO. A white piece
     * on board square {@code sq} (a1=0) therefore reads {@code table[sq ^ 56]}, and a black
     * piece reads {@code table[sq]} (vertical mirror), which makes the eval color-symmetric.
     */
    private static void initPst() {
        MG_PST[PAWN] = new int[] {
               0,    0,    0,    0,    0,    0,    0,    0,
              98,  134,   61,   95,   68,  126,   34,  -11,
              -6,    7,   26,   31,   65,   56,   25,  -20,
             -14,   13,    6,   20,   23,   12,   17,  -23,
             -26,   -3,   -5,   10,   16,    6,    9,  -24,
             -25,   -4,   -3,  -10,    4,    3,   33,  -11,
             -35,   -3,  -19,  -21,  -14,   25,   36,  -21,
               0,    0,    0,    0,    0,    0,    0,    0,
        };
        EG_PST[PAWN] = new int[] {
               0,    0,    0,    0,    0,    0,    0,    0,
             177,  172,  157,  133,  147,  132,  165,  187,
              93,   99,   84,   66,   55,   53,   82,   84,
              32,   23,   13,    5,   -1,    4,   17,   18,
              14,    9,   -3,   -7,   -6,   -7,    3,    1,
               4,    6,   -6,    1,    1,   -4,   -3,   -7,
              14,    7,    8,   10,   13,    0,    0,   -6,
               0,    0,    0,    0,    0,    0,    0,    0,
        };
        MG_PST[KNIGHT] = new int[] {
            -167,  -89,  -34,  -49,   61,  -97,  -15, -107,
             -73,  -41,   72,   36,   23,   62,    7,  -17,
             -47,   60,   37,   65,   84,  129,   73,   44,
              -9,   18,   19,   53,   37,   69,   18,   22,
             -13,    4,   16,   13,   28,   19,   21,   -8,
             -23,   -9,   11,   10,   19,   16,   25,  -16,
             -29,  -53,  -12,   -2,    0,   18,  -14,  -19,
            -105,  -21,  -58,  -33,  -17,  -28,  -19,  -23,
        };
        EG_PST[KNIGHT] = new int[] {
             -58,  -38,  -13,  -28,  -31,  -27,  -63,  -99,
             -25,   -8,  -25,   -2,   -9,  -25,  -24,  -52,
             -24,  -20,   10,    9,   -1,   -9,  -19,  -41,
             -17,    3,   22,   22,   22,   11,    8,  -18,
             -18,   -6,   16,   25,   16,   17,    4,  -18,
             -23,   -3,   -1,   15,   10,   -3,  -20,  -22,
             -42,  -20,  -10,   -5,   -2,  -20,  -23,  -44,
             -29,  -51,  -23,  -15,  -22,  -18,  -50,  -64,
        };
        MG_PST[BISHOP] = new int[] {
             -29,    4,  -82,  -37,  -25,  -42,    7,   -8,
             -26,   16,  -18,  -13,   30,   59,   18,  -47,
             -16,   37,   43,   40,   35,   50,   37,   -2,
              -4,    5,   19,   50,   37,   37,    7,   -2,
              -6,   13,   13,   26,   34,   11,   10,    4,
               0,   15,   15,   14,   13,   27,   18,   10,
               4,   16,   16,    0,    8,   21,   34,    1,
             -33,   -3,  -12,  -21,  -13,  -12,  -39,  -21,
        };
        EG_PST[BISHOP] = new int[] {
             -14,  -21,  -11,   -8,   -7,   -9,  -17,  -24,
              -8,   -4,    7,  -12,   -3,  -13,   -4,  -14,
               2,   -8,    0,   -1,   -2,    6,    0,    4,
              -3,    9,   12,    9,   14,   10,    3,    2,
              -6,    3,   13,   19,    7,   10,   -3,   -9,
             -12,   -3,    8,   10,   13,    3,   -7,  -15,
             -14,  -18,   -7,   -1,    4,   -9,  -15,  -27,
             -23,   -9,  -22,   -5,   -9,  -15,   -5,  -17,
        };
        MG_PST[ROOK] = new int[] {
              32,   42,   32,   51,   63,    9,   31,   43,
              26,   31,   58,   62,   80,   67,   26,   44,
              -5,   19,   26,   36,   17,   45,   61,   16,
             -24,  -11,    7,   26,   24,   35,   -8,  -20,
             -36,  -26,  -12,   -1,    9,   -7,    6,  -23,
             -45,  -25,  -16,  -17,    3,    0,   -5,  -33,
             -44,  -16,  -20,   -9,   -1,   11,   -6,  -71,
             -17,  -13,    1,   16,   16,    9,  -37,  -26,
        };
        EG_PST[ROOK] = new int[] {
              12,    9,   17,   14,   11,   12,    8,    5,
              10,   12,   12,   10,   -4,    2,    8,    3,
               7,    7,    7,    5,    4,   -3,   -5,   -3,
               5,    3,   13,    1,    2,    1,   -1,    3,
               4,    6,    8,    4,   -5,   -6,   -8,  -11,
              -3,    0,   -5,   -1,   -7,  -12,   -8,  -16,
              -6,   -6,    0,    2,   -9,   -9,  -11,   -3,
              -8,    3,    3,   -1,   -5,  -13,    4,  -19,
        };
        MG_PST[QUEEN] = new int[] {
             -28,    0,   29,   12,   59,   44,   43,   45,
             -24,  -39,   -5,    1,  -16,   57,   28,   54,
             -13,  -17,    7,    8,   29,   56,   47,   57,
             -27,  -27,  -16,  -16,   -1,   17,   -2,    1,
              -9,  -26,   -9,  -10,   -2,   -4,    3,   -3,
             -14,    2,  -11,   -2,   -5,    2,   14,    5,
             -35,   -8,   11,    2,    8,   15,   -3,    1,
              -1,  -18,   -9,   10,  -15,  -25,  -31,  -50,
        };
        EG_PST[QUEEN] = new int[] {
              -9,   22,   22,   27,   27,   19,   10,   20,
             -17,   20,   32,   41,   58,   25,   30,    0,
             -20,    6,    9,   49,   47,   35,   19,    9,
               3,   22,   24,   45,   57,   40,   57,   36,
             -18,   28,   19,   47,   31,   34,   39,   23,
             -16,  -27,   15,    6,    9,   17,   10,    5,
             -22,  -23,  -30,  -16,  -16,  -23,  -36,  -32,
             -33,  -28,  -22,  -43,   -5,  -32,  -20,  -41,
        };
        MG_PST[KING] = new int[] {
             -65,   23,   16,  -15,  -56,  -34,    2,   13,
              29,   -1,  -20,   -7,   -8,   -4,  -38,  -29,
              -9,   24,    2,  -16,  -20,    6,   22,  -22,
             -17,  -20,  -12,  -27,  -30,  -25,  -14,  -36,
             -49,   -1,  -27,  -39,  -46,  -44,  -33,  -51,
             -14,  -14,  -22,  -46,  -44,  -30,  -15,  -27,
               1,    7,   -8,  -64,  -43,  -16,    9,    8,
             -15,   36,   12,  -54,    7,  -28,   24,   14,
        };
        EG_PST[KING] = new int[] {
             -74,  -35,  -18,  -18,  -11,   15,    4,  -17,
             -12,   17,   14,   17,   17,   38,   23,   11,
              10,   17,   23,   15,   20,   45,   44,   13,
              -8,   22,   24,   27,   26,   33,   26,    3,
             -18,   -4,   21,   24,   27,   23,    9,  -11,
             -19,   -3,   11,   21,   23,   16,    8,   -9,
             -27,  -11,    4,   13,   14,    4,   -5,  -18,
             -53,  -34,  -21,  -11,  -28,  -14,  -26,  -44,
        };
    }
}
