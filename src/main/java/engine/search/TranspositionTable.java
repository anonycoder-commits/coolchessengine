package engine.search;

/**
 * Zobrist-keyed transposition table.
 *
 * Each bucket holds two slots: a depth-preferred slot and an always-replace slot
 * (two-tier scheme). Entries store a bound flag (EXACT/LOWER/UPPER) so the search can
 * use them soundly. Mate-score ply-relativization is the caller's responsibility
 * (see {@link Search}): the table stores whatever score it is given.
 */
public final class TranspositionTable {

    public static final int FLAG_NONE = 0;
    public static final int FLAG_EXACT = 1;
    public static final int FLAG_LOWER = 2; // fail-high: stored score is a lower bound
    public static final int FLAG_UPPER = 3; // fail-low: stored score is an upper bound

    // key(8) + move(4) + score(2) + depth(2) + flag(1), padded; cluster of 2.
    private static final int ENTRY_BYTES = 16;

    private long[] keys;
    private int[] moves;
    private short[] scores;
    private short[] depths;
    private byte[] flags;
    private int bucketMask;

    // Probe results (single-threaded, no allocation on the hot path).
    public int ttMove;
    public int ttScore;
    public int ttDepth;
    public int ttFlag;

    public TranspositionTable(int sizeMb) {
        resize(sizeMb);
    }

    /** (Re)allocates the table to approximately {@code sizeMb} megabytes (power-of-two buckets). */
    public void resize(int sizeMb) {
        if (sizeMb < 1) sizeMb = 1;
        long bytes = (long) sizeMb * 1024L * 1024L;
        long totalEntries = bytes / ENTRY_BYTES;
        long buckets = Math.max(1, totalEntries / 2);
        int pow2 = Integer.highestOneBit((int) Math.min(buckets, 1 << 26));
        if (pow2 < 1) pow2 = 1;
        int slots = pow2 * 2;
        keys = new long[slots];
        moves = new int[slots];
        scores = new short[slots];
        depths = new short[slots];
        flags = new byte[slots];
        bucketMask = pow2 - 1;
    }

    public void clear() {
        java.util.Arrays.fill(keys, 0L);
        java.util.Arrays.fill(moves, 0);
        java.util.Arrays.fill(flags, (byte) FLAG_NONE);
    }

    private int base(long key) {
        return ((int) (key & bucketMask)) << 1;
    }

    /** Returns true and fills the {@code tt*} fields if an entry for {@code key} exists. */
    public boolean probe(long key) {
        int b = base(key);
        for (int i = 0; i < 2; i++) {
            int idx = b + i;
            if (flags[idx] != FLAG_NONE && keys[idx] == key) {
                ttMove = moves[idx];
                ttScore = scores[idx];
                ttDepth = depths[idx];
                ttFlag = flags[idx];
                return true;
            }
        }
        return false;
    }

    public void store(long key, int depth, int score, int flag, int move) {
        int b = base(key);
        int deep = b;       // depth-preferred slot
        int always = b + 1; // always-replace slot

        int target;
        if (flags[deep] == FLAG_NONE || keys[deep] == key || depth >= depths[deep]) {
            target = deep;
        } else {
            target = always;
        }

        // Preserve a known best move if this store carries none for the same position.
        if (move == 0 && keys[target] == key && moves[target] != 0) {
            move = moves[target];
        }

        keys[target] = key;
        moves[target] = move;
        scores[target] = (short) score;
        depths[target] = (short) depth;
        flags[target] = (byte) flag;
    }
}
