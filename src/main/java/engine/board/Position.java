package engine.board;

import static engine.board.Piece.*;

/** Bitboard chess position with make/unmake and incremental Zobrist hashing. */
public final class Position {

    private static final int MAX_PLY = 2048;

    // Per-piece bitboards (index 0..11)
    final long[] bb = new long[12];
    // Occupancy by color and total
    final long[] occByColor = new long[2];
    long occupied;
    // Mailbox: square -> piece index, or EMPTY
    final int[] board = new int[64];

    int sideToMove;
    int castling;     // bitmask of CASTLE_* rights
    int epSquare;     // en-passant target square, or -1
    int halfmoveClock;
    int fullmoveNumber;
    long key;

    // Undo stack
    private final int[] undoCaptured = new int[MAX_PLY];
    private final int[] undoCastling = new int[MAX_PLY];
    private final int[] undoEp = new int[MAX_PLY];
    private final int[] undoHalfmove = new int[MAX_PLY];
    private final long[] undoKey = new long[MAX_PLY];
    private int ply = 0;

    // Per-square castling-right masks: AND a square into rights when a piece
    // leaves or is captured on it.
    private static final int[] CASTLE_MASK = new int[64];
    static {
        for (int i = 0; i < 64; i++) CASTLE_MASK[i] = 0xF;
        CASTLE_MASK[0] &= ~CASTLE_WQ;   // a1
        CASTLE_MASK[7] &= ~CASTLE_WK;   // h1
        CASTLE_MASK[4] &= ~(CASTLE_WK | CASTLE_WQ); // e1
        CASTLE_MASK[56] &= ~CASTLE_BQ;  // a8
        CASTLE_MASK[63] &= ~CASTLE_BK;  // h8
        CASTLE_MASK[60] &= ~(CASTLE_BK | CASTLE_BQ); // e8
    }

    public Position() {}

    /**
     * Deep copy: an independent board with identical state, safe to make/unmake moves
     * on from another thread without racing the original. Used to give each Lazy SMP
     * search worker its own board instead of sharing one mutable Position across threads.
     */
    public Position(Position other) {
        System.arraycopy(other.bb, 0, this.bb, 0, this.bb.length);
        System.arraycopy(other.occByColor, 0, this.occByColor, 0, this.occByColor.length);
        this.occupied = other.occupied;
        System.arraycopy(other.board, 0, this.board, 0, this.board.length);
        this.sideToMove = other.sideToMove;
        this.castling = other.castling;
        this.epSquare = other.epSquare;
        this.halfmoveClock = other.halfmoveClock;
        this.fullmoveNumber = other.fullmoveNumber;
        this.key = other.key;
        System.arraycopy(other.undoCaptured, 0, this.undoCaptured, 0, this.undoCaptured.length);
        System.arraycopy(other.undoCastling, 0, this.undoCastling, 0, this.undoCastling.length);
        System.arraycopy(other.undoEp, 0, this.undoEp, 0, this.undoEp.length);
        System.arraycopy(other.undoHalfmove, 0, this.undoHalfmove, 0, this.undoHalfmove.length);
        System.arraycopy(other.undoKey, 0, this.undoKey, 0, this.undoKey.length);
        this.ply = other.ply;
    }

    public static Position startpos() {
        return fromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    public static Position fromFen(String fen) {
        Position p = new Position();
        p.setFen(fen);
        return p;
    }

    public void setFen(String fen) {
        for (int i = 0; i < 12; i++) bb[i] = 0L;
        for (int i = 0; i < 64; i++) board[i] = EMPTY;
        occByColor[0] = occByColor[1] = 0L;
        occupied = 0L;
        ply = 0;

        String[] parts = fen.trim().split("\\s+");
        String[] rows = parts[0].split("/");
        for (int r = 0; r < 8; r++) {
            int rank = 7 - r;
            int file = 0;
            for (int i = 0; i < rows[r].length(); i++) {
                char c = rows[r].charAt(i);
                if (Character.isDigit(c)) {
                    file += c - '0';
                } else {
                    int sq = rank * 8 + file;
                    int piece = Piece.fromChar(c);
                    bb[piece] |= 1L << sq;
                    board[sq] = piece;
                    file++;
                }
            }
        }

        sideToMove = parts[1].equals("w") ? WHITE : BLACK;

        castling = 0;
        if (parts.length > 2 && !parts[2].equals("-")) {
            if (parts[2].indexOf('K') >= 0) castling |= CASTLE_WK;
            if (parts[2].indexOf('Q') >= 0) castling |= CASTLE_WQ;
            if (parts[2].indexOf('k') >= 0) castling |= CASTLE_BK;
            if (parts[2].indexOf('q') >= 0) castling |= CASTLE_BQ;
        }

        epSquare = -1;
        if (parts.length > 3 && !parts[3].equals("-")) {
            int f = parts[3].charAt(0) - 'a';
            int rnk = parts[3].charAt(1) - '1';
            epSquare = rnk * 8 + f;
        }

        halfmoveClock = parts.length > 4 ? Integer.parseInt(parts[4]) : 0;
        fullmoveNumber = parts.length > 5 ? Integer.parseInt(parts[5]) : 1;

        recomputeOccupancy();
        key = computeKey();
    }

    private void recomputeOccupancy() {
        long white = 0L, black = 0L;
        for (int i = 0; i < 6; i++) white |= bb[i];
        for (int i = 6; i < 12; i++) black |= bb[i];
        occByColor[WHITE] = white;
        occByColor[BLACK] = black;
        occupied = white | black;
    }

    /** Full Zobrist key recomputation from scratch. */
    long computeKey() {
        long k = 0L;
        for (int piece = 0; piece < 12; piece++) {
            long b = bb[piece];
            while (b != 0) {
                int sq = Long.numberOfTrailingZeros(b);
                b &= b - 1;
                k ^= Zobrist.PIECE[piece][sq];
            }
        }
        if (sideToMove == BLACK) k ^= Zobrist.SIDE;
        k ^= Zobrist.CASTLING[castling];
        if (epSquare >= 0 && epRelevant(epSquare, sideToMove)) {
            k ^= Zobrist.EP_FILE[epSquare & 7];
        }
        return k;
    }

    /**
     * True if a pawn of {@code capturingSide} is positioned to capture en passant
     * onto {@code epSq}. The en-passant file is only hashed into the key when this holds.
     */
    boolean epRelevant(int epSq, int capturingSide) {
        long ourPawns = bb[index(capturingSide, PAWN)];
        return (Attacks.PAWN[1 - capturingSide][epSq] & ourPawns) != 0;
    }

    // --- piece manipulation (keeps bb, occupancy, mailbox, and key in sync) ---

    private void addPiece(int piece, int sq) {
        long bit = 1L << sq;
        bb[piece] |= bit;
        occByColor[piece / 6] |= bit;
        occupied |= bit;
        board[sq] = piece;
        key ^= Zobrist.PIECE[piece][sq];
    }

    private void removePiece(int piece, int sq) {
        long bit = 1L << sq;
        bb[piece] &= ~bit;
        occByColor[piece / 6] &= ~bit;
        occupied &= ~bit;
        board[sq] = EMPTY;
        key ^= Zobrist.PIECE[piece][sq];
    }

    private void movePiece(int piece, int from, int to) {
        removePiece(piece, from);
        addPiece(piece, to);
    }

    public void makeMove(int move) {
        int from = Move.from(move);
        int to = Move.to(move);
        int flag = Move.flag(move);
        int us = sideToMove;
        int them = 1 - us;
        int piece = board[from];

        undoCastling[ply] = castling;
        undoEp[ply] = epSquare;
        undoHalfmove[ply] = halfmoveClock;
        undoKey[ply] = key;

        // Clear previous ep hash contribution
        if (epSquare >= 0 && epRelevant(epSquare, us)) {
            key ^= Zobrist.EP_FILE[epSquare & 7];
        }
        epSquare = -1;

        // Captures
        int captured = EMPTY;
        if (flag == Move.EP_CAPTURE) {
            int capSq = us == WHITE ? to - 8 : to + 8;
            captured = board[capSq];
            removePiece(captured, capSq);
        } else if ((flag & 4) != 0) {
            captured = board[to];
            removePiece(captured, to);
        }
        undoCaptured[ply] = captured;

        // Move / promote the piece
        if (Move.isPromotion(move)) {
            removePiece(piece, from);
            addPiece(index(us, Move.promoType(move)), to);
        } else {
            movePiece(piece, from, to);
        }

        // Special moves
        if (flag == Move.DOUBLE_PUSH) {
            int ep = us == WHITE ? from + 8 : from - 8;
            epSquare = ep;
            if (epRelevant(ep, them)) key ^= Zobrist.EP_FILE[ep & 7];
        } else if (flag == Move.KING_CASTLE) {
            if (us == WHITE) movePiece(W_ROOK, 7, 5);
            else movePiece(B_ROOK, 63, 61);
        } else if (flag == Move.QUEEN_CASTLE) {
            if (us == WHITE) movePiece(W_ROOK, 0, 3);
            else movePiece(B_ROOK, 56, 59);
        }

        // Castling rights
        key ^= Zobrist.CASTLING[castling];
        castling &= CASTLE_MASK[from] & CASTLE_MASK[to];
        key ^= Zobrist.CASTLING[castling];

        // Halfmove clock
        if ((flag & 4) != 0 || type(piece) == PAWN) halfmoveClock = 0;
        else halfmoveClock++;

        if (us == BLACK) fullmoveNumber++;
        sideToMove = them;
        key ^= Zobrist.SIDE;

        ply++;
    }

    public void unmakeMove(int move) {
        ply--;
        int from = Move.from(move);
        int to = Move.to(move);
        int flag = Move.flag(move);
        sideToMove = 1 - sideToMove;
        int us = sideToMove;
        if (us == BLACK) fullmoveNumber--;

        // Restore moved piece
        if (Move.isPromotion(move)) {
            removePiece(index(us, Move.promoType(move)), to);
            addPiece(index(us, PAWN), from);
        } else {
            int piece = board[to];
            movePiece(piece, to, from);
        }

        // Undo castling rook
        if (flag == Move.KING_CASTLE) {
            if (us == WHITE) movePiece(W_ROOK, 5, 7);
            else movePiece(B_ROOK, 61, 63);
        } else if (flag == Move.QUEEN_CASTLE) {
            if (us == WHITE) movePiece(W_ROOK, 3, 0);
            else movePiece(B_ROOK, 59, 56);
        }

        // Restore captured piece
        int captured = undoCaptured[ply];
        if (flag == Move.EP_CAPTURE) {
            int capSq = us == WHITE ? to - 8 : to + 8;
            addPiece(captured, capSq);
        } else if ((flag & 4) != 0) {
            addPiece(captured, to);
        }

        castling = undoCastling[ply];
        epSquare = undoEp[ply];
        halfmoveClock = undoHalfmove[ply];
        key = undoKey[ply];
    }

    /**
     * Passes the move without moving any piece (for null-move pruning): flips the side to
     * move and clears the en-passant square, exactly the side-to-move/ep bookkeeping a real
     * move does, with no piece movement, capture, or castling-rights change. Shares the
     * regular undo stack ({@code undoEp}/{@code undoKey}/{@code ply}) since it is sized far
     * beyond any search's recursion depth and a null move advances/retreats {@code ply}
     * exactly like {@link #makeMove}/{@link #unmakeMove} do.
     */
    public void makeNullMove() {
        undoEp[ply] = epSquare;
        undoKey[ply] = key;
        if (epSquare >= 0 && epRelevant(epSquare, sideToMove)) {
            key ^= Zobrist.EP_FILE[epSquare & 7];
        }
        epSquare = -1;
        sideToMove = 1 - sideToMove;
        key ^= Zobrist.SIDE;
        ply++;
    }

    public void unmakeNullMove() {
        ply--;
        sideToMove = 1 - sideToMove;
        epSquare = undoEp[ply];
        key = undoKey[ply];
    }

    // --- attack queries ---

    /** True if {@code sq} is attacked by any piece of {@code bySide}. */
    public boolean isSquareAttacked(int sq, int bySide) {
        if ((Attacks.PAWN[1 - bySide][sq] & bb[index(bySide, PAWN)]) != 0) return true;
        if ((Attacks.KNIGHT[sq] & bb[index(bySide, KNIGHT)]) != 0) return true;
        if ((Attacks.KING[sq] & bb[index(bySide, KING)]) != 0) return true;
        long bishopsQueens = bb[index(bySide, BISHOP)] | bb[index(bySide, QUEEN)];
        if ((Attacks.bishop(sq, occupied) & bishopsQueens) != 0) return true;
        long rooksQueens = bb[index(bySide, ROOK)] | bb[index(bySide, QUEEN)];
        if ((Attacks.rook(sq, occupied) & rooksQueens) != 0) return true;
        return false;
    }

    public int kingSquare(int color) {
        return Long.numberOfTrailingZeros(bb[index(color, KING)]);
    }

    public boolean inCheck(int color) {
        return isSquareAttacked(kingSquare(color), 1 - color);
    }

    /**
     * True if the current position is a draw by the fifty-move rule or by repetition
     * within the played line. Uses the per-ply key history on the undo stack: positions
     * with the same side to move sit two plies apart, and nothing earlier than the last
     * irreversible move (bounded by the halfmove clock) can repeat the current key.
     * A single earlier occurrence is treated as a draw inside the search to discourage
     * the engine from shuffling into repetitions.
     */
    public boolean isDrawByRuleOrRepetition() {
        if (halfmoveClock >= 100) return true;
        int end = ply - halfmoveClock;
        if (end < 0) end = 0;
        for (int i = ply - 2; i >= end; i -= 2) {
            if (undoKey[i] == key) return true;
        }
        return false;
    }

    // --- accessors ---

    public int sideToMove() { return sideToMove; }
    public long zobristKey() { return key; }
    public int epSquare() { return epSquare; }
    public int castlingRights() { return castling; }
    public int halfmoveClock() { return halfmoveClock; }
    public int fullmoveNumber() { return fullmoveNumber; }
    public int pieceAt(int sq) { return board[sq]; }
    public long pieces(int pieceIndex) { return bb[pieceIndex]; }
    public long occupied() { return occupied; }
    public long occupied(int color) { return occByColor[color]; }
}
