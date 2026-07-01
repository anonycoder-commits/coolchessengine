package engine.board;

/** Color, piece-type, and piece-index constants. */
public final class Piece {
    private Piece() {}

    // Colors
    public static final int WHITE = 0;
    public static final int BLACK = 1;

    // Piece types
    public static final int PAWN = 0;
    public static final int KNIGHT = 1;
    public static final int BISHOP = 2;
    public static final int ROOK = 3;
    public static final int QUEEN = 4;
    public static final int KING = 5;

    // Piece indices (color * 6 + type), 0..11
    public static final int W_PAWN = 0;
    public static final int W_KNIGHT = 1;
    public static final int W_BISHOP = 2;
    public static final int W_ROOK = 3;
    public static final int W_QUEEN = 4;
    public static final int W_KING = 5;
    public static final int B_PAWN = 6;
    public static final int B_KNIGHT = 7;
    public static final int B_BISHOP = 8;
    public static final int B_ROOK = 9;
    public static final int B_QUEEN = 10;
    public static final int B_KING = 11;

    public static final int EMPTY = 12;

    // Castling-rights bitmask
    public static final int CASTLE_WK = 1;
    public static final int CASTLE_WQ = 2;
    public static final int CASTLE_BK = 4;
    public static final int CASTLE_BQ = 8;

    public static int color(int piece) { return piece / 6; }
    public static int type(int piece) { return piece % 6; }
    public static int index(int color, int type) { return color * 6 + type; }

    private static final char[] CHARS = {'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'};

    public static char toChar(int piece) { return CHARS[piece]; }

    public static int fromChar(char c) {
        switch (c) {
            case 'P': return W_PAWN;
            case 'N': return W_KNIGHT;
            case 'B': return W_BISHOP;
            case 'R': return W_ROOK;
            case 'Q': return W_QUEEN;
            case 'K': return W_KING;
            case 'p': return B_PAWN;
            case 'n': return B_KNIGHT;
            case 'b': return B_BISHOP;
            case 'r': return B_ROOK;
            case 'q': return B_QUEEN;
            case 'k': return B_KING;
            default: return EMPTY;
        }
    }
}
