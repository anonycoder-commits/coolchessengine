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
    private static final int[] MG_VALUE = {82, 337, 365, 477, 1025, 0};
    private static final int[] EG_VALUE = {94, 281, 297, 512, 936, 0};

    // Tempo: a small flat bonus for the side to move, reflecting the standing initiative of
    // having the move. Beyond a genuine (if small) positional truth, it damps the odd/even
    // evaluation oscillation between plies that otherwise churns the aspiration window and
    // destabilizes node counts between iterations. Added to the side-to-move-relative score,
    // so it is deliberately NOT color-symmetric the way the board terms are (see
    // EvalSymmetryTest, which subtracts it out before checking antisymmetry).
    public static final int TEMPO = 15;

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
    private static final int DOUBLED_MG = 12;
    private static final int DOUBLED_EG = 20;
    private static final int ISOLATED_MG = 14;
    private static final int ISOLATED_EG = 10;
    // Passed-pawn bonus indexed by the pawn's rank relative to its own side (0..7). EG ramps
    // far more steeply toward the promotion rank: with no pieces left to blockade or round up
    // a passer, a pawn on the 6th/7th is close to a second queen, not just a small edge.
    private static final int[] PASSED_MG = {0, 5, 8, 12, 20, 32, 50, 0};
    private static final int[] PASSED_EG = {0, 10, 18, 30, 50, 90, 150, 0};

    // King-safety shield/file terms are an almost purely midgame concept -- they price in the
    // danger of enemy queen/rook fire down an open line at the king, which stops mattering
    // once those attackers are gone. EG values are small, not zero, since a king still prefers
    // some cover against a lone rook/queen endgame, but nowhere near midgame weight; EG king
    // placement is instead driven by EG_PST[KING] rewarding centralization (see initPst()).
    private static final int SHIELD_MG = 8;
    private static final int SHIELD_EG = 2;
    private static final int OPEN_FILE_MG = 20;
    private static final int OPEN_FILE_EG = 4;
    private static final int SEMI_OPEN_FILE_MG = 10;
    private static final int SEMI_OPEN_FILE_EG = 3;

    // King-attack ("attack units"): each enemy piece bearing on the squares around a king
    // contributes units weighted by piece type and by how many king-zone squares it hits; the
    // total is squared (danger grows super-linearly as attackers pile up) into a penalty. This
    // is what lets the engine value a building attack BEFORE tactics force it into qsearch,
    // rather than reading sharp middlegames as flat. Requires >= 2 attackers (a lone piece near
    // the king is not a real threat) and is almost purely a middlegame concern (small eg).
    private static final int[] KING_ATTACK_WEIGHT = {0, 2, 2, 3, 5, 0}; // by piece TYPE
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
    private static final int BAD_BISHOP_MG = 3; // per own pawn on the bishop's square color
    private static final int BAD_BISHOP_EG = 5;

    // Mobility by piece TYPE: rooks/queens gain relative value in the endgame (open lines
    // decide races once material thins out), while knights/bishops taper down slightly.
    private static final int[] MOBILITY_MG = {0, 4, 4, 3, 1, 0};
    private static final int[] MOBILITY_EG = {0, 3, 3, 4, 2, 0};

    // Passed-pawn king proximity (endgame only): a passer's value in the endgame depends
    // heavily on which king can reach its promotion square first, which pure rank scaling
    // ignores. Rewards the friendly king being close and the enemy king being far, weighted
    // by how advanced the passer already is.
    private static final int PASSER_KING_EG = 5;

    // Connected/phalanx pawn bonus and backward-pawn penalty (small, standard mg/eg pairs).
    private static final int CONNECTED_MG = 8;
    private static final int CONNECTED_EG = 6;
    private static final int BACKWARD_MG = 10;
    private static final int BACKWARD_EG = 14;

    // Endgame scale factor (out of SCALE_MAX): the eg score is multiplied by this before the
    // phase blend, to reflect that some material configurations are far more drawish than raw
    // material suggests -- opposite-colored bishops halve, dead-drawn minor endings zero out.
    private static final int SCALE_MAX = 16;

    // Weighted-mobility "Ambition" top-ups: on top of the flat per-square MOBILITY_MG/EG
    // rate above, a mobility square gets an extra bonus when it represents genuine central/
    // open-line pressure rather than just square count -- a rook eyeing an open file or a
    // knight/bishop hitting a center square is worth more than an equally-mobile piece
    // shuffling along the back rank. Added on top of (not instead of) the base rate, so total
    // weight for a "premium" square is MOBILITY_*[type] + the matching bonus below.
    private static final int ROOK_OPEN_FILE_MOBILITY_MG = 3;
    private static final int ROOK_OPEN_FILE_MOBILITY_EG = 2;
    private static final int QUEEN_OPEN_FILE_MOBILITY_MG = 2;
    private static final int QUEEN_OPEN_FILE_MOBILITY_EG = 2;
    private static final int KNIGHT_CENTER_MOBILITY_MG = 6;
    private static final int KNIGHT_CENTER_MOBILITY_EG = 4;
    private static final int BISHOP_CENTER_MOBILITY_MG = 5;
    private static final int BISHOP_CENTER_MOBILITY_EG = 3;

    // Central outposts: a knight or bishop parked on d4/d5/e4/e5 that no enemy pawn can ever
    // structurally challenge (see outpostScore()) is a long-term thorn the opponent can't
    // easily remove, so it earns a standing "tension bonus" independent of mobility/PST --
    // this is what keeps the engine from retreating such a piece to a "safer" but passive
    // square purely to reduce short-term tactical exposure. Knights get the larger bonus
    // since, unlike bishops, they have no long-diagonal alternative outlet once parked.
    private static final long CENTER_SQUARES = (1L << 27) | (1L << 28) | (1L << 35) | (1L << 36); // d4,e4,d5,e5
    private static final int OUTPOST_KNIGHT_MG = 18;
    private static final int OUTPOST_KNIGHT_EG = 12;
    private static final int OUTPOST_BISHOP_MG = 12;
    private static final int OUTPOST_BISHOP_EG = 8;

    // --- bishop pair / rook activity weights (tapered: separate mg/eg, unlike the flat
    // pawn/mobility/king-safety terms above) ---
    private static final int BISHOP_PAIR_MG = 15;
    private static final int BISHOP_PAIR_EG = 30;

    private static final int ROOK_OPEN_FILE_MG = 20;
    private static final int ROOK_OPEN_FILE_EG = 15;
    private static final int ROOK_SEMI_OPEN_FILE_MG = 10;
    private static final int ROOK_SEMI_OPEN_FILE_EG = 10;
    private static final int ROOK_7TH_MG = 25;
    private static final int ROOK_7TH_EG = 40;

    // Game-phase weight per piece TYPE; full opening phase sums to 24.
    private static final int[] PHASE_INC = {0, 1, 1, 2, 4, 0};
    private static final int PHASE_MAX = 24;

    // Piece-square tables, [type][square], written a8-first (see static init for orientation).
    private static final int[][] MG_PST = new int[6][];
    private static final int[][] EG_PST = new int[6][];

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

    /**
     * Tapered evaluation from the side-to-move's perspective.
     */
    public static int evaluate(Position pos) {
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

        int[] bishopPair = evalBishopPair(pos);
        mg += bishopPair[0];
        eg += bishopPair[1];

        int[] rookActivity = evalRooks(pos);
        mg += rookActivity[0];
        eg += rookActivity[1];

        int[] pawns = evalPawns(pos);
        mg += pawns[0];
        eg += pawns[1];

        int[] mobility = evalMobility(pos);
        mg += mobility[0];
        eg += mobility[1];

        int[] kingSafety = evalKingSafety(pos);
        mg += kingSafety[0];
        eg += kingSafety[1];

        int[] outposts = evalOutposts(pos);
        mg += outposts[0];
        eg += outposts[1];

        int[] kingAttack = evalKingAttack(pos);
        mg += kingAttack[0];
        eg += kingAttack[1];

        int[] space = evalSpace(pos);
        mg += space[0];
        eg += space[1];

        int[] badBishop = evalBadBishop(pos);
        mg += badBishop[0];
        eg += badBishop[1];

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

    private static int[] evalPawns(Position pos) {
        long wp = pos.pieces(index(WHITE, PAWN));
        long bp = pos.pieces(index(BLACK, PAWN));
        int wk = pos.kingSquare(WHITE);
        int bk = pos.kingSquare(BLACK);
        // Own king / enemy king passed to each side for the passer-proximity term.
        int[] w = pawnScore(wp, bp, WHITE, wk, bk);
        int[] b = pawnScore(bp, wp, BLACK, bk, wk);
        return new int[] {w[0] - b[0], w[1] - b[1]};
    }

    private static int[] pawnScore(long own, long enemy, int color, int ownKing, int enemyKing) {
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
                // Endgame king proximity to the promotion square, weighted by advancement.
                int promoSq = color == WHITE ? (56 + f) : f;
                int prox = chebyshev(enemyKing, promoSq) - chebyshev(ownKing, promoSq);
                eg += PASSER_KING_EG * prox * rel / 6;
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
        return new int[] {mg, eg};
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

    private static int[] evalKingAttack(Position pos) {
        // A king in danger is a penalty to ITS side, so the enemy's danger is the friendly
        // side's gain: white-relative term = (danger to black king) - (danger to white king).
        int[] wDanger = kingDanger(pos, WHITE);
        int[] bDanger = kingDanger(pos, BLACK);
        return new int[] {bDanger[0] - wDanger[0], bDanger[1] - wDanger[1]};
    }

    /** Attack-unit danger (>= 0) to {@code kingColor}'s king from the opposing pieces. */
    private static int[] kingDanger(Position pos, int kingColor) {
        int ksq = pos.kingSquare(kingColor);
        long zone = Attacks.KING[ksq] | (1L << ksq);
        int enemy = 1 - kingColor;
        long occ = pos.occupied();
        int units = 0, attackers = 0;

        long n = pos.pieces(index(enemy, KNIGHT));
        while (n != 0) {
            int sq = Long.numberOfTrailingZeros(n);
            n &= n - 1;
            int c = Long.bitCount(Attacks.KNIGHT[sq] & zone);
            if (c > 0) { units += KING_ATTACK_WEIGHT[KNIGHT] * c; attackers++; }
        }
        long b = pos.pieces(index(enemy, BISHOP));
        while (b != 0) {
            int sq = Long.numberOfTrailingZeros(b);
            b &= b - 1;
            int c = Long.bitCount(Attacks.bishop(sq, occ) & zone);
            if (c > 0) { units += KING_ATTACK_WEIGHT[BISHOP] * c; attackers++; }
        }
        long r = pos.pieces(index(enemy, ROOK));
        while (r != 0) {
            int sq = Long.numberOfTrailingZeros(r);
            r &= r - 1;
            int c = Long.bitCount(Attacks.rook(sq, occ) & zone);
            if (c > 0) { units += KING_ATTACK_WEIGHT[ROOK] * c; attackers++; }
        }
        long q = pos.pieces(index(enemy, QUEEN));
        while (q != 0) {
            int sq = Long.numberOfTrailingZeros(q);
            q &= q - 1;
            int c = Long.bitCount(Attacks.queen(sq, occ) & zone);
            if (c > 0) { units += KING_ATTACK_WEIGHT[QUEEN] * c; attackers++; }
        }

        // A lone attacker near the king isn't a real threat; danger is only credited once at
        // least two enemy pieces bear on the zone, and grows super-linearly from there.
        if (attackers < 2) return new int[] {0, 0};
        int penalty = Math.min(units * units / KING_ATTACK_DIVISOR, KING_ATTACK_CAP);
        return new int[] {penalty, penalty >> KING_ATTACK_EG_SHIFT};
    }

    // --- space (mg only, white-relative) ---

    private static int[] evalSpace(Position pos) {
        int w = spaceScore(pos, WHITE);
        int b = spaceScore(pos, BLACK);
        return new int[] {w - b, 0}; // mg only
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

    private static int[] evalBadBishop(Position pos) {
        int[] w = badBishopPenalty(pos, WHITE);
        int[] b = badBishopPenalty(pos, BLACK);
        // A bad bishop hurts its owner, so the enemy's bad bishop is our gain.
        return new int[] {b[0] - w[0], b[1] - w[1]};
    }

    private static int[] badBishopPenalty(Position pos, int color) {
        long bishops = pos.pieces(index(color, BISHOP));
        if (bishops == 0) return new int[] {0, 0};
        long ownPawns = pos.pieces(index(color, PAWN));
        int lightPawns = Long.bitCount(ownPawns & LIGHT_SQUARES);
        int darkPawns = Long.bitCount(ownPawns & ~LIGHT_SQUARES);
        int lightBishops = Long.bitCount(bishops & LIGHT_SQUARES);
        int darkBishops = Long.bitCount(bishops & ~LIGHT_SQUARES);
        // Each bishop is dragged down by the own pawns fixed on its own square color.
        int units = lightBishops * lightPawns + darkBishops * darkPawns;
        return new int[] {units * BAD_BISHOP_MG, units * BAD_BISHOP_EG};
    }

    // --- weighted mobility (tapered, white-relative) ---
    //
    // "Ambition" weighting: every reachable square still earns the flat MOBILITY_MG/EG rate,
    // but a square that represents genuine pressure -- a rook/queen move landing on a
    // pawnless (open) file, or a knight/bishop move landing on a center square -- earns an
    // additional top-up on top of that base rate (see the *_MOBILITY_* constants above).
    // openFiles is computed once per evaluate() call and shared by both colors' rook/queen
    // loops rather than recomputed per piece.

    private static int[] evalMobility(Position pos) {
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
        int[] w = mobilityScore(pos, WHITE, occ, openFiles, blackPawnAtt);
        int[] b = mobilityScore(pos, BLACK, occ, openFiles, whitePawnAtt);
        return new int[] {w[0] - b[0], w[1] - b[1]};
    }

    /** All squares attacked by {@code color}'s pawns (file-wrap masked). */
    private static long pawnAttacks(long pawns, int color) {
        if (color == WHITE) {
            return ((pawns & ~Attacks.FILE_A) << 7) | ((pawns & ~Attacks.FILE_H) << 9);
        }
        return ((pawns & ~Attacks.FILE_A) >>> 9) | ((pawns & ~Attacks.FILE_H) >>> 7);
    }

    private static int[] mobilityScore(Position pos, int color, long occ, long openFiles,
                                       long enemyPawnAtt) {
        long own = pos.occupied(color);
        long area = ~own & ~enemyPawnAtt; // exclude own pieces and enemy-pawn-controlled squares
        int mg = 0, eg = 0;

        long n = pos.pieces(index(color, KNIGHT));
        while (n != 0) {
            int sq = Long.numberOfTrailingZeros(n);
            n &= n - 1;
            long dest = Attacks.KNIGHT[sq] & area;
            int cnt = Long.bitCount(dest);
            int centerCnt = Long.bitCount(dest & CENTER_SQUARES);
            mg += MOBILITY_MG[KNIGHT] * cnt + KNIGHT_CENTER_MOBILITY_MG * centerCnt;
            eg += MOBILITY_EG[KNIGHT] * cnt + KNIGHT_CENTER_MOBILITY_EG * centerCnt;
        }
        long bsh = pos.pieces(index(color, BISHOP));
        while (bsh != 0) {
            int sq = Long.numberOfTrailingZeros(bsh);
            bsh &= bsh - 1;
            long dest = Attacks.bishop(sq, occ) & area;
            int cnt = Long.bitCount(dest);
            int centerCnt = Long.bitCount(dest & CENTER_SQUARES);
            mg += MOBILITY_MG[BISHOP] * cnt + BISHOP_CENTER_MOBILITY_MG * centerCnt;
            eg += MOBILITY_EG[BISHOP] * cnt + BISHOP_CENTER_MOBILITY_EG * centerCnt;
        }
        long r = pos.pieces(index(color, ROOK));
        while (r != 0) {
            int sq = Long.numberOfTrailingZeros(r);
            r &= r - 1;
            long dest = Attacks.rook(sq, occ) & area;
            int cnt = Long.bitCount(dest);
            int openCnt = Long.bitCount(dest & openFiles);
            mg += MOBILITY_MG[ROOK] * cnt + ROOK_OPEN_FILE_MOBILITY_MG * openCnt;
            eg += MOBILITY_EG[ROOK] * cnt + ROOK_OPEN_FILE_MOBILITY_EG * openCnt;
        }
        long q = pos.pieces(index(color, QUEEN));
        while (q != 0) {
            int sq = Long.numberOfTrailingZeros(q);
            q &= q - 1;
            long dest = Attacks.queen(sq, occ) & area;
            int cnt = Long.bitCount(dest);
            int openCnt = Long.bitCount(dest & openFiles);
            mg += MOBILITY_MG[QUEEN] * cnt + QUEEN_OPEN_FILE_MOBILITY_MG * openCnt;
            eg += MOBILITY_EG[QUEEN] * cnt + QUEEN_OPEN_FILE_MOBILITY_EG * openCnt;
        }
        return new int[] {mg, eg};
    }

    // --- central outposts (tapered, white-relative) ---

    private static int[] evalOutposts(Position pos) {
        int[] w = outpostScore(pos, WHITE);
        int[] b = outpostScore(pos, BLACK);
        return new int[] {w[0] - b[0], w[1] - b[1]};
    }

    private static int[] outpostScore(Position pos, int color) {
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
        return new int[] {mg, eg};
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

    private static int[] evalBishopPair(Position pos) {
        int mg = 0, eg = 0;
        if (Long.bitCount(pos.pieces(index(WHITE, BISHOP))) >= 2) {
            mg += BISHOP_PAIR_MG;
            eg += BISHOP_PAIR_EG;
        }
        if (Long.bitCount(pos.pieces(index(BLACK, BISHOP))) >= 2) {
            mg -= BISHOP_PAIR_MG;
            eg -= BISHOP_PAIR_EG;
        }
        return new int[] {mg, eg};
    }

    // --- rook file activity + 7th/8th-rank infiltration (tapered, white-relative) ---

    private static int[] evalRooks(Position pos) {
        int[] w = rookScore(pos, WHITE);
        int[] b = rookScore(pos, BLACK);
        return new int[] {w[0] - b[0], w[1] - b[1]};
    }

    private static int[] rookScore(Position pos, int color) {
        int mg = 0, eg = 0;
        long ownPawns = pos.pieces(index(color, PAWN));
        long enemyPawns = pos.pieces(index(1 - color, PAWN));
        long rooks = pos.pieces(index(color, ROOK));
        while (rooks != 0) {
            int sq = Long.numberOfTrailingZeros(rooks);
            rooks &= rooks - 1;
            int f = sq & 7;
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
        return new int[] {mg, eg};
    }

    // --- king safety (tapered, white-relative) ---

    private static int[] evalKingSafety(Position pos) {
        int[] w = kingSafetyScore(pos, WHITE);
        int[] b = kingSafetyScore(pos, BLACK);
        return new int[] {w[0] - b[0], w[1] - b[1]};
    }

    private static int[] kingSafetyScore(Position pos, int color) {
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
        return new int[] {mg, eg};
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
              0,   0,   0,   0,   0,   0,   0,   0,
             98, 134,  61,  95,  68, 126,  34, -11,
             -6,   7,  26,  31,  65,  56,  25, -20,
            -14,  13,   6,  21,  23,  12,  17, -23,
            -27,  -2,  -5,  12,  17,   6,  10, -25,
            -26,  -4,  -4, -10,   3,   3,  33, -12,
            -35,  -1, -20, -23, -15,  24,  38, -22,
              0,   0,   0,   0,   0,   0,   0,   0,
        };
        EG_PST[PAWN] = new int[] {
              0,   0,   0,   0,   0,   0,   0,   0,
            178, 173, 158, 134, 147, 132, 165, 187,
             94, 100,  85,  67,  56,  53,  82,  84,
             32,  24,  13,   5,  -2,   4,  17,  17,
             13,   9,  -3,  -7,  -7,  -8,   3,  -1,
              4,   7,  -6,   1,   0,  -5,  -1,  -8,
             13,   8,   8,  10,  13,   0,   2,  -7,
              0,   0,   0,   0,   0,   0,   0,   0,
        };
        MG_PST[KNIGHT] = new int[] {
            -167, -89, -34, -49,  61, -97, -15, -107,
             -73, -41,  72,  36,  23,  62,   7,  -17,
             -47,  60,  37,  65,  84, 129,  73,   44,
              -9,  17,  19,  53,  37,  69,  18,   22,
             -13,   4,  16,  13,  28,  19,  21,   -8,
             -23,  -9,  12,  10,  19,  17,  25,  -16,
             -29, -53, -12,  -3,  -1,  18, -14,  -19,
            -105, -21, -58, -33, -17, -28, -19,  -23,
        };
        EG_PST[KNIGHT] = new int[] {
            -58, -38, -13, -28, -31, -27, -63, -99,
            -25,  -8, -25,  -2,  -9, -25, -24, -52,
            -24, -20,  10,   9,  -1,  -9, -19, -41,
            -17,   3,  22,  22,  22,  11,   8, -18,
            -18,  -6,  16,  25,  16,  17,   4, -18,
            -23,  -3,  -1,  15,  10,  -3, -20, -22,
            -42, -20, -10,  -5,  -2, -20, -23, -44,
            -29, -51, -23, -15, -22, -18, -50, -64,
        };
        MG_PST[BISHOP] = new int[] {
            -29,   4, -82, -37, -25, -42,   7,  -8,
            -26,  16, -18, -13,  30,  59,  18, -47,
            -16,  37,  43,  40,  35,  50,  37,  -2,
             -4,   5,  19,  50,  37,  37,   7,  -2,
             -6,  13,  13,  26,  34,  12,  10,   4,
              0,  15,  15,  15,  14,  27,  18,  10,
              4,  15,  16,   0,   7,  21,  33,   1,
            -33,  -3, -14, -21, -13, -12, -39, -21,
        };
        EG_PST[BISHOP] = new int[] {
            -14, -21, -11,  -8,  -7,  -9, -17, -24,
             -8,  -4,   7, -12,  -3, -13,  -4, -14,
              2,  -8,   0,  -1,  -2,   6,   0,   4,
             -3,   9,  12,   9,  14,  10,   3,   2,
             -6,   3,  13,  19,   7,  10,  -3,  -9,
            -12,  -3,   8,  10,  13,   3,  -7, -15,
            -14, -18,  -7,  -1,   4,  -9, -15, -27,
            -23,  -9, -23,  -5,  -9, -16,  -5, -17,
        };
        MG_PST[ROOK] = new int[] {
             32,  42,  32,  51,  63,   9,  31,  43,
             27,  32,  58,  62,  80,  67,  26,  44,
             -5,  19,  26,  36,  17,  45,  61,  16,
            -24, -11,   7,  26,  24,  35,  -8, -20,
            -36, -26, -12,  -1,   9,  -7,   6, -23,
            -45, -25, -16, -17,   3,   0,  -5, -33,
            -44, -16, -20,  -9,  -1,  11,  -6, -71,
            -19, -13,   1,  17,  16,   7, -37, -26,
        };
        EG_PST[ROOK] = new int[] {
            13, 10, 18, 15, 12,  12,   8,   5,
            11, 13, 13, 11, -3,   3,   8,   3,
             7,  7,  7,  5,  4,  -3,  -5,  -3,
             4,  3, 13,  1,  2,   1,  -1,   2,
             3,  5,  8,  4, -5,  -6,  -8, -11,
            -4,  0, -5, -1, -7, -12,  -8, -16,
            -6, -6,  0,  2, -9,  -9, -11,  -3,
            -9,  2,  3, -1, -5, -13,   4, -20,
        };
        MG_PST[QUEEN] = new int[] {
            -28,   0,  29,  12,  59,  44,  43,  45,
            -24, -39,  -5,   1, -16,  57,  28,  54,
            -13, -17,   7,   8,  29,  56,  47,  57,
            -27, -27, -16, -16,  -1,  17,  -2,   1,
             -9, -26,  -9, -10,  -2,  -4,   3,  -3,
            -14,   2, -11,  -2,  -5,   2,  14,   5,
            -35,  -8,  11,   2,   8,  15,  -3,   1,
             -1, -18,  -9,  10, -15, -25, -31, -50,
        };
        EG_PST[QUEEN] = new int[] {
             -9,  22,  22,  27,  27,  19,  10,  20,
            -17,  20,  32,  41,  58,  25,  30,   0,
            -20,   6,   9,  49,  47,  35,  19,   9,
              3,  22,  24,  45,  57,  40,  57,  36,
            -18,  28,  19,  47,  31,  34,  39,  23,
            -16, -27,  15,   6,   9,  17,  10,   5,
            -22, -23, -30, -16, -16, -23, -36, -32,
            -33, -28, -22, -43,  -5, -32, -20, -41,
        };
        MG_PST[KING] = new int[] {
            -65,  23,  16, -15, -56, -34,   2,  13,
             29,  -1, -20,  -7,  -8,  -4, -38, -29,
             -9,  24,   2, -16, -20,   6,  22, -22,
            -17, -20, -12, -27, -30, -25, -14, -36,
            -49,  -1, -27, -39, -46, -44, -33, -51,
            -14, -14, -22, -46, -44, -30, -15, -27,
              1,   7,  -8, -64, -43, -16,   9,   8,
            -15,  36,  12, -54,   8, -28,  24,  14,
        };
        EG_PST[KING] = new int[] {
            -74, -35, -18, -18, -11,  15,   4, -17,
            -12,  17,  14,  17,  17,  38,  23,  11,
             10,  17,  23,  15,  20,  45,  44,  13,
             -8,  22,  24,  27,  26,  33,  26,   3,
            -18,  -4,  21,  24,  27,  23,   9, -11,
            -19,  -3,  11,  21,  23,  16,   7,  -9,
            -27, -11,   4,  13,  14,   4,  -5, -17,
            -53, -34, -21, -11, -28, -14, -24, -43,
        };
    }
}
