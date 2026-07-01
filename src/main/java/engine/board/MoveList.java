package engine.board;

/** Lightweight growable container of encoded int moves (no boxing). */
public final class MoveList {
    public int[] moves = new int[256];
    public int size = 0;

    public void clear() { size = 0; }

    public void add(int move) {
        if (size == moves.length) {
            int[] grown = new int[moves.length * 2];
            System.arraycopy(moves, 0, grown, 0, moves.length);
            moves = grown;
        }
        moves[size++] = move;
    }

    public int get(int i) { return moves[i]; }
}
