package engine.board;

/**
 * Precomputed attack tables for leapers, plus three interchangeable implementations of
 * sliding-piece (bishop/rook/queen) attacks, selectable at class-init via the
 * {@code engine.attacks} system property ({@code ray}, {@code magic}, or {@code pext};
 * default {@code pext}, benchmarked fastest -- see migration_plan.md Phase 2):
 *
 * <ul>
 *   <li>{@code ray} -- the original classical ray-attack method: for each of 8 directions
 *       a ray bitboard is precomputed per square; the first blocker along the ray (found
 *       via bitscan) truncates the ray, keeping the blocker square so captures are
 *       included. Retained permanently as the correctness oracle both the other two modes
 *       are built from and verified against (see AttackEquivalenceTest) -- it is never
 *       deleted, and {@code -Dengine.attacks=ray} is the standing rollback path.</li>
 *   <li>{@code magic} -- fancy magic bitboards: {@code (occ & mask) * magic >>> shift}
 *       indexes a per-square slice of a flat lookup table.</li>
 *   <li>{@code pext} -- the same table layout indexed via {@code Long.compress} (JDK 19+,
 *       intrinsified to hardware PEXT on x86-64 BMI2) instead of a magic multiply.</li>
 * </ul>
 *
 * MODE is {@code static final}, so the dispatch in {@link #bishop}/{@link #rook} below is
 * constant-folded and dead-branch-eliminated by C2 -- the compiled hot path is a single
 * inlinable table lookup, not a virtual/lambda dispatch (which would go megamorphic and
 * block inlining in what is the single hottest function in the engine).
 */
public final class Attacks {
    private Attacks() {}

    public static final long FILE_A = 0x0101010101010101L;
    public static final long FILE_H = 0x8080808080808080L;
    public static final long RANK_1 = 0x00000000000000FFL;
    public static final long RANK_2 = 0x000000000000FF00L;
    public static final long RANK_3 = 0x0000000000FF0000L;
    public static final long RANK_4 = 0x00000000FF000000L;
    public static final long RANK_5 = 0x000000FF00000000L;
    public static final long RANK_6 = 0x0000FF0000000000L;
    public static final long RANK_7 = 0x00FF000000000000L;
    public static final long RANK_8 = 0xFF00000000000000L;

    public static final long[] KNIGHT = new long[64];
    public static final long[] KING = new long[64];
    // PAWN[color][square] = squares attacked by a pawn of that color on that square
    public static final long[][] PAWN = new long[2][64];

    // Ray directions
    private static final int NORTH = 0, SOUTH = 1, EAST = 2, WEST = 3;
    private static final int NE = 4, NW = 5, SE = 6, SW = 7;
    private static final long[][] RAY = new long[8][64];

    // dr, df for each direction index
    private static final int[] DR = {1, -1, 0, 0, 1, 1, -1, -1};
    private static final int[] DF = {0, 0, 1, -1, 1, -1, 1, -1};

    // --- sliding-attack mode selection ---
    private static final int MODE_RAY = 0, MODE_MAGIC = 1, MODE_PEXT = 2;
    private static final int MODE = resolveMode();

    private static int resolveMode() {
        // Default is "pext": benchmarked consistently fastest on this machine (~18-20%
        // over ray, ~4-6% over magic; see migration_plan.md Phase 2 results). The property
        // override is the permanent rollback path to either of the other two modes.
        String prop = System.getProperty("engine.attacks", "pext");
        switch (prop) {
            case "ray": return MODE_RAY;
            case "magic": return MODE_MAGIC;
            case "pext":
            default: return MODE_PEXT;
        }
    }

    // --- fancy magic / PEXT table metadata, shared by both indexing schemes since table
    // size per square (2^popcount(mask)) is identical regardless of how it's indexed ---
    private static final long[] ROOK_MASK = new long[64];
    private static final long[] BISHOP_MASK = new long[64];
    private static final int[] ROOK_SHIFT = new int[64];
    private static final int[] BISHOP_SHIFT = new int[64];
    private static final int[] ROOK_OFFSET = new int[64];
    private static final int[] BISHOP_OFFSET = new int[64];

    private static long[] ROOK_MAGIC_TBL;
    private static long[] BISHOP_MAGIC_TBL;
    private static long[] ROOK_PEXT_TBL;
    private static long[] BISHOP_PEXT_TBL;

    static {
        initLeapers();
        initRays();
        initSliderMasks();
        initSliderTables();
    }

    private static void initLeapers() {
        int[][] knightOff = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
        int[][] kingOff = {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
        for (int sq = 0; sq < 64; sq++) {
            int f = sq & 7, r = sq >> 3;
            long n = 0L, k = 0L;
            for (int[] o : knightOff) {
                int nf = f + o[0], nr = r + o[1];
                if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) n |= 1L << (nr * 8 + nf);
            }
            for (int[] o : kingOff) {
                int nf = f + o[0], nr = r + o[1];
                if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) k |= 1L << (nr * 8 + nf);
            }
            KNIGHT[sq] = n;
            KING[sq] = k;

            long wp = 0L, bp = 0L;
            if (f > 0 && r < 7) wp |= 1L << (sq + 7);
            if (f < 7 && r < 7) wp |= 1L << (sq + 9);
            if (f > 0 && r > 0) bp |= 1L << (sq - 9);
            if (f < 7 && r > 0) bp |= 1L << (sq - 7);
            PAWN[Piece.WHITE][sq] = wp;
            PAWN[Piece.BLACK][sq] = bp;
        }
    }

    private static void initRays() {
        for (int dir = 0; dir < 8; dir++) {
            for (int sq = 0; sq < 64; sq++) {
                long ray = 0L;
                int f = sq & 7, r = sq >> 3;
                while (true) {
                    f += DF[dir];
                    r += DR[dir];
                    if (f < 0 || f > 7 || r < 0 || r > 7) break;
                    ray |= 1L << (r * 8 + f);
                }
                RAY[dir][sq] = ray;
            }
        }
    }

    private static long positiveRay(int dir, int sq, long occ) {
        long attacks = RAY[dir][sq];
        long blockers = attacks & occ;
        if (blockers != 0) {
            int b = Long.numberOfTrailingZeros(blockers);
            attacks ^= RAY[dir][b];
        }
        return attacks;
    }

    private static long negativeRay(int dir, int sq, long occ) {
        long attacks = RAY[dir][sq];
        long blockers = attacks & occ;
        if (blockers != 0) {
            int b = 63 - Long.numberOfLeadingZeros(blockers);
            attacks ^= RAY[dir][b];
        }
        return attacks;
    }

    /** Classical ray-scan bishop attacks -- the correctness oracle; see class Javadoc. */
    static long bishopRay(int sq, long occ) {
        return positiveRay(NE, sq, occ) | positiveRay(NW, sq, occ)
                | negativeRay(SE, sq, occ) | negativeRay(SW, sq, occ);
    }

    /** Classical ray-scan rook attacks -- the correctness oracle; see class Javadoc. */
    static long rookRay(int sq, long occ) {
        return positiveRay(NORTH, sq, occ) | positiveRay(EAST, sq, occ)
                | negativeRay(SOUTH, sq, occ) | negativeRay(WEST, sq, occ);
    }

    // --- fancy magic / PEXT table construction ---

    /** Relevant-occupancy mask: the full ray minus the board-edge square in each
     *  direction, since a piece on that edge square is always attacked/reachable
     *  regardless of whether it's occupied -- it never changes the attack set, so
     *  excluding it shrinks the table without losing any information. */
    private static long computeRookMask(int sq) {
        return (RAY[NORTH][sq] & ~RANK_8)
                | (RAY[SOUTH][sq] & ~RANK_1)
                | (RAY[EAST][sq] & ~FILE_H)
                | (RAY[WEST][sq] & ~FILE_A);
    }

    private static long computeBishopMask(int sq) {
        long edges = FILE_A | FILE_H | RANK_1 | RANK_8;
        return (RAY[NE][sq] | RAY[NW][sq] | RAY[SE][sq] | RAY[SW][sq]) & ~edges;
    }

    private static void initSliderMasks() {
        int rookOff = 0, bishopOff = 0;
        for (int sq = 0; sq < 64; sq++) {
            long rMask = computeRookMask(sq);
            long bMask = computeBishopMask(sq);
            ROOK_MASK[sq] = rMask;
            BISHOP_MASK[sq] = bMask;
            int rBits = Long.bitCount(rMask);
            int bBits = Long.bitCount(bMask);
            ROOK_SHIFT[sq] = 64 - rBits;
            BISHOP_SHIFT[sq] = 64 - bBits;
            ROOK_OFFSET[sq] = rookOff;
            BISHOP_OFFSET[sq] = bishopOff;
            rookOff += 1 << rBits;
            bishopOff += 1 << bBits;
        }
        // rook: 4*4096 + 24*2048 + 36*1024 = 102,400 entries (800 KB); bishop: 5,248 (41 KB)
        ROOK_MAGIC_TBL = new long[rookOff];
        ROOK_PEXT_TBL = new long[rookOff];
        BISHOP_MAGIC_TBL = new long[bishopOff];
        BISHOP_PEXT_TBL = new long[bishopOff];
    }

    private static void initSliderTables() {
        for (int sq = 0; sq < 64; sq++) {
            fillSquare(sq, ROOK_MASK[sq], ROOK_SHIFT[sq], ROOK_OFFSET[sq],
                    MagicConstants.ROOK[sq], true, ROOK_MAGIC_TBL, ROOK_PEXT_TBL);
            fillSquare(sq, BISHOP_MASK[sq], BISHOP_SHIFT[sq], BISHOP_OFFSET[sq],
                    MagicConstants.BISHOP[sq], false, BISHOP_MAGIC_TBL, BISHOP_PEXT_TBL);
        }
    }

    /**
     * Enumerates every subset of {@code mask} via the Carry-Rippler trick
     * ({@code subset = (subset - mask) & mask}), computes the true attack set for each
     * subset from the ray oracle (a subset containing only non-edge bits is a valid
     * occupancy to feed the ray scanner -- see computeRookMask/computeBishopMask), and
     * fills both the magic-indexed and PEXT-indexed table slices for this square. Throws
     * on any destructive magic collision (two subsets landing on the same slot with
     * different attack sets), which both verifies MagicConstants at startup and would
     * catch a masks/shifts/offsets bug immediately rather than silently corrupting attacks.
     */
    private static void fillSquare(int sq, long mask, int shift, int offset, long magic,
                                    boolean rook, long[] magicTbl, long[] pextTbl) {
        long subset = 0L;
        do {
            long attacks = rook ? rookRay(sq, subset) : bishopRay(sq, subset);

            int pextIdx = offset + (int) Long.compress(subset, mask);
            pextTbl[pextIdx] = attacks;

            int magicIdx = offset + (int) ((subset * magic) >>> shift);
            long existing = magicTbl[magicIdx];
            if (existing != 0L && existing != attacks) {
                throw new IllegalStateException(
                        "Destructive magic collision for " + (rook ? "rook" : "bishop")
                                + " at square " + sq + " -- MagicConstants entry is invalid");
            }
            magicTbl[magicIdx] = attacks;

            subset = (subset - mask) & mask;
        } while (subset != 0L);
    }

    // --- direct per-mode accessors, exposed package-private for AttackEquivalenceTest ---

    static long rookMagic(int sq, long occ) {
        long masked = occ & ROOK_MASK[sq];
        int idx = ROOK_OFFSET[sq] + (int) ((masked * MagicConstants.ROOK[sq]) >>> ROOK_SHIFT[sq]);
        return ROOK_MAGIC_TBL[idx];
    }

    static long bishopMagic(int sq, long occ) {
        long masked = occ & BISHOP_MASK[sq];
        int idx = BISHOP_OFFSET[sq] + (int) ((masked * MagicConstants.BISHOP[sq]) >>> BISHOP_SHIFT[sq]);
        return BISHOP_MAGIC_TBL[idx];
    }

    static long rookPext(int sq, long occ) {
        int idx = ROOK_OFFSET[sq] + (int) Long.compress(occ, ROOK_MASK[sq]);
        return ROOK_PEXT_TBL[idx];
    }

    static long bishopPext(int sq, long occ) {
        int idx = BISHOP_OFFSET[sq] + (int) Long.compress(occ, BISHOP_MASK[sq]);
        return BISHOP_PEXT_TBL[idx];
    }

    // --- public interface: signatures frozen, unchanged by this migration ---

    public static long bishop(int sq, long occ) {
        if (MODE == MODE_PEXT) return bishopPext(sq, occ);
        if (MODE == MODE_MAGIC) return bishopMagic(sq, occ);
        return bishopRay(sq, occ);
    }

    public static long rook(int sq, long occ) {
        if (MODE == MODE_PEXT) return rookPext(sq, occ);
        if (MODE == MODE_MAGIC) return rookMagic(sq, occ);
        return rookRay(sq, occ);
    }

    public static long queen(int sq, long occ) {
        return bishop(sq, occ) | rook(sq, occ);
    }
}
