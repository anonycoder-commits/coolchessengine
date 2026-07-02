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

    // Mobility by piece TYPE: rooks/queens gain relative value in the endgame (open lines
    // decide races once material thins out), while knights/bishops taper down slightly.
    private static final int[] MOBILITY_MG = {0, 4, 4, 3, 1, 0};
    private static final int[] MOBILITY_EG = {0, 3, 3, 4, 2, 0};

    // King-ring mobility (relative king-ring control): for each side, the net attack density
    // over its OWN 3x3 king ring -- total bit-attacks its pieces exert on the ring (defensive
    // coverage) minus total bit-attacks the enemy exerts on it (assault pressure). Differenced
    // between the two sides, this measures who owns the spatial contest around the kings BEFORE
    // any sacrifice lands, steering the engine away from suffocating shells. Heavily middlegame-
    // weighted: king shelter matters while heavy pieces are on the board; in the endgame an
    // active, centralized king is preferred over a sheltered one, so the eg weight is small.
    private static final int KING_RING_MG = 4;
    private static final int KING_RING_EG = 1;
    // A/B toggle (static: the evaluator is a stateless utility). Flip to false to disable the
    // term for match comparison. Shared across threads -- set once between searches, not mid-search.
    public static boolean useKingRingMobility = true;

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

        if (useKingRingMobility) {
            int[] kingRing = evalKingRingMobility(pos);
            mg += kingRing[0];
            eg += kingRing[1];
        }

        // Linear interpolation core: every term above (material, PST, bishop pair, rook
        // activity, pawn structure, mobility, king safety, outposts) has now fed both an mg
        // and an eg contribution into the same two accumulators, so this single blend is the
        // only place game phase is applied -- nothing bypasses it and reaches the final score
        // unscaled.
        int mgPhase = Math.min(phase, PHASE_MAX);
        int egPhase = PHASE_MAX - mgPhase;
        int score = (mg * mgPhase + eg * egPhase) / PHASE_MAX;

        return pos.sideToMove() == WHITE ? score : -score;
    }

    // --- pawn structure (tapered, white-relative) ---

    private static int[] evalPawns(Position pos) {
        long wp = pos.pieces(index(WHITE, PAWN));
        long bp = pos.pieces(index(BLACK, PAWN));
        int[] w = pawnScore(wp, bp, WHITE);
        int[] b = pawnScore(bp, wp, BLACK);
        return new int[] {w[0] - b[0], w[1] - b[1]};
    }

    private static int[] pawnScore(long own, long enemy, int color) {
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
        long b = own;
        while (b != 0) {
            int sq = Long.numberOfTrailingZeros(b);
            b &= b - 1;
            long mask = color == WHITE ? WHITE_PASSED[sq] : BLACK_PASSED[sq];
            if ((enemy & mask) == 0) {
                int rel = color == WHITE ? (sq >> 3) : 7 - (sq >> 3);
                mg += PASSED_MG[rel];
                eg += PASSED_EG[rel];
            }
        }
        return new int[] {mg, eg};
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
        long allPawns = pos.pieces(index(WHITE, PAWN)) | pos.pieces(index(BLACK, PAWN));
        long openFiles = 0L;
        for (int f = 0; f < 8; f++) {
            if ((allPawns & FILES[f]) == 0) openFiles |= FILES[f];
        }
        int[] w = mobilityScore(pos, WHITE, occ, openFiles);
        int[] b = mobilityScore(pos, BLACK, occ, openFiles);
        return new int[] {w[0] - b[0], w[1] - b[1]};
    }

    private static int[] mobilityScore(Position pos, int color, long occ, long openFiles) {
        long own = pos.occupied(color);
        int mg = 0, eg = 0;

        long n = pos.pieces(index(color, KNIGHT));
        while (n != 0) {
            int sq = Long.numberOfTrailingZeros(n);
            n &= n - 1;
            long dest = Attacks.KNIGHT[sq] & ~own;
            int cnt = Long.bitCount(dest);
            int centerCnt = Long.bitCount(dest & CENTER_SQUARES);
            mg += MOBILITY_MG[KNIGHT] * cnt + KNIGHT_CENTER_MOBILITY_MG * centerCnt;
            eg += MOBILITY_EG[KNIGHT] * cnt + KNIGHT_CENTER_MOBILITY_EG * centerCnt;
        }
        long bsh = pos.pieces(index(color, BISHOP));
        while (bsh != 0) {
            int sq = Long.numberOfTrailingZeros(bsh);
            bsh &= bsh - 1;
            long dest = Attacks.bishop(sq, occ) & ~own;
            int cnt = Long.bitCount(dest);
            int centerCnt = Long.bitCount(dest & CENTER_SQUARES);
            mg += MOBILITY_MG[BISHOP] * cnt + BISHOP_CENTER_MOBILITY_MG * centerCnt;
            eg += MOBILITY_EG[BISHOP] * cnt + BISHOP_CENTER_MOBILITY_EG * centerCnt;
        }
        long r = pos.pieces(index(color, ROOK));
        while (r != 0) {
            int sq = Long.numberOfTrailingZeros(r);
            r &= r - 1;
            long dest = Attacks.rook(sq, occ) & ~own;
            int cnt = Long.bitCount(dest);
            int openCnt = Long.bitCount(dest & openFiles);
            mg += MOBILITY_MG[ROOK] * cnt + ROOK_OPEN_FILE_MOBILITY_MG * openCnt;
            eg += MOBILITY_EG[ROOK] * cnt + ROOK_OPEN_FILE_MOBILITY_EG * openCnt;
        }
        long q = pos.pieces(index(color, QUEEN));
        while (q != 0) {
            int sq = Long.numberOfTrailingZeros(q);
            q &= q - 1;
            long dest = Attacks.queen(sq, occ) & ~own;
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

    // --- king-ring mobility (tapered, white-relative) ---

    /**
     * White-relative king-ring control: {@code ringControl(WHITE) - ringControl(BLACK)},
     * scaled into a tapered mg/eg pair. Perfectly color-symmetric -- both sides are measured
     * by the identical procedure and differenced -- so it preserves {@code evaluate(p) ==
     * -evaluate(mirror(p))} and stays 0 in a mirror-symmetric position such as the start.
     */
    private static int[] evalKingRingMobility(Position pos) {
        int diff = ringControl(pos, WHITE) - ringControl(pos, BLACK);
        return new int[] {diff * KING_RING_MG, diff * KING_RING_EG};
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
