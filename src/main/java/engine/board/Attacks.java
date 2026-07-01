package engine.board;

/**
 * Precomputed attack tables and magic-free sliding-piece attacks.
 *
 * Sliding attacks use the classical ray-attack method: for each of 8 directions a
 * ray bitboard is precomputed per square; the first blocker along the ray (found via
 * bitscan) truncates the ray, keeping the blocker square so captures are included.
 * This is intentionally NOT magic bitboards (deferred as a later verified optimization).
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

    static {
        initLeapers();
        initRays();
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

    public static long bishop(int sq, long occ) {
        return positiveRay(NE, sq, occ) | positiveRay(NW, sq, occ)
                | negativeRay(SE, sq, occ) | negativeRay(SW, sq, occ);
    }

    public static long rook(int sq, long occ) {
        return positiveRay(NORTH, sq, occ) | positiveRay(EAST, sq, occ)
                | negativeRay(SOUTH, sq, occ) | negativeRay(WEST, sq, occ);
    }

    public static long queen(int sq, long occ) {
        return bishop(sq, occ) | rook(sq, occ);
    }
}
