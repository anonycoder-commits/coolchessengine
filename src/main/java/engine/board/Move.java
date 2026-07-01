package engine.board;

/**
 * Moves are encoded into a single int:
 *   bits 0-5   : from square (0..63)
 *   bits 6-11  : to square (0..63)
 *   bits 12-15 : flag
 */
public final class Move {
    private Move() {}

    // Flags
    public static final int QUIET = 0;
    public static final int DOUBLE_PUSH = 1;
    public static final int KING_CASTLE = 2;
    public static final int QUEEN_CASTLE = 3;
    public static final int CAPTURE = 4;
    public static final int EP_CAPTURE = 5;
    public static final int PROMO_N = 8;
    public static final int PROMO_B = 9;
    public static final int PROMO_R = 10;
    public static final int PROMO_Q = 11;
    public static final int PROMO_N_CAP = 12;
    public static final int PROMO_B_CAP = 13;
    public static final int PROMO_R_CAP = 14;
    public static final int PROMO_Q_CAP = 15;

    public static int make(int from, int to, int flag) {
        return from | (to << 6) | (flag << 12);
    }

    public static int from(int move) { return move & 0x3F; }
    public static int to(int move) { return (move >>> 6) & 0x3F; }
    public static int flag(int move) { return (move >>> 12) & 0xF; }

    public static boolean isCapture(int move) { return (flag(move) & 4) != 0; }
    public static boolean isPromotion(int move) { return (flag(move) & 8) != 0; }

    /** Promoted piece TYPE (KNIGHT..QUEEN). Only valid when isPromotion. */
    public static int promoType(int move) { return Piece.KNIGHT + (flag(move) & 3); }

    /** UCI long-algebraic string, e.g. "e2e4", "e7e8q". */
    public static String toUci(int move) {
        StringBuilder sb = new StringBuilder(5);
        sb.append(squareName(from(move)));
        sb.append(squareName(to(move)));
        if (isPromotion(move)) {
            int t = promoType(move);
            char c;
            switch (t) {
                case Piece.KNIGHT: c = 'n'; break;
                case Piece.BISHOP: c = 'b'; break;
                case Piece.ROOK: c = 'r'; break;
                default: c = 'q'; break;
            }
            sb.append(c);
        }
        return sb.toString();
    }

    public static String squareName(int sq) {
        int file = sq & 7;
        int rank = sq >>> 3;
        return "" + (char) ('a' + file) + (char) ('1' + rank);
    }
}
