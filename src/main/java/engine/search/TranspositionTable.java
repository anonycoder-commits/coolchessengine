package engine.search;

import java.util.concurrent.atomic.AtomicLongArray;

/**
 * Zobrist-keyed transposition table, safe for concurrent {@link #probe} / {@link #store}
 * calls from multiple search threads (Lazy SMP) with no locks.
 *
 * Each bucket holds two slots (depth-preferred + always-replace, as before). Every slot
 * is exactly two atomic 64-bit words: a packed "data" word (move + score + depth + flag)
 * and a "lockless key" word holding {@code zobristKey ^ data} instead of the raw key.
 * This is the standard lockless-hashing technique (see the Chess Programming Wiki,
 * "Shared Hash Table" / "Lockless Hashing"): {@link AtomicLongArray} guarantees each
 * individual word is never torn, but the *pair* of words for one slot is still written
 * as two separate operations, so a concurrent writer can interleave between them.
 * Reconstructing the candidate key as {@code storedXorKey ^ storedData} and comparing it
 * against the probed key catches that interleaving: a torn read pairs a data word from
 * one write with a key word from a different write, and the XOR identity essentially
 * never coincidentally holds for unrelated writes, so the entry is treated as a miss
 * instead of silently returning a corrupted move/score/depth/flag combination.
 *
 * This is a deliberate, well-known "benign race," not a correctness gap: at worst a
 * probe misses an entry it could have used, or a store loses a race to another thread's
 * store into the same slot. Neither corrupts search results, and both are exactly as
 * likely (and as harmless) as an ordinary TT collision/overwrite already is in a
 * single-threaded engine.
 *
 * {@link #resize} and {@link #clear} are NOT safe to call while any search thread may be
 * probing/storing concurrently (they replace/zero the backing arrays outright) — callers
 * (UCI {@code setoption}/{@code ucinewgame}) must only invoke them between searches.
 */
public final class TranspositionTable {

    public static final int FLAG_NONE = 0;
    public static final int FLAG_EXACT = 1;
    public static final int FLAG_LOWER = 2; // fail-high: stored score is a lower bound
    public static final int FLAG_UPPER = 3; // fail-low: stored score is an upper bound

    /** Sentinel returned by {@link #probe} on a miss: {@code flagOf(MISS) == FLAG_NONE}. */
    public static final long MISS = 0L;

    private static final int ENTRY_BYTES = 16; // key word + data word, cluster of 2

    private AtomicLongArray xorKeys;
    private AtomicLongArray data;
    private int bucketMask;

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
        xorKeys = new AtomicLongArray(slots);
        data = new AtomicLongArray(slots);
        bucketMask = pow2 - 1;
    }

    public void clear() {
        int n = data.length();
        for (int i = 0; i < n; i++) {
            data.set(i, 0L);
            xorKeys.set(i, 0L);
        }
    }

    private int base(long key) {
        return ((int) (key & bucketMask)) << 1;
    }

    /**
     * Probes for {@code key}. Returns the packed data word on a hit (unpack with
     * {@link #moveOf}, {@link #scoreOf}, {@link #depthOf}, {@link #flagOf}), or
     * {@link #MISS} otherwise. Thread-safe: results are returned directly rather than
     * stashed in shared instance fields, so concurrent probes from different threads
     * never clobber each other's results.
     */
    public long probe(long key) {
        int b = base(key);
        for (int i = 0; i < 2; i++) {
            int idx = b + i;
            long d = data.get(idx);
            long xk = xorKeys.get(idx);
            if (flagOf(d) != FLAG_NONE && (xk ^ d) == key) {
                return d;
            }
        }
        return MISS;
    }

    public void store(long key, int depth, int score, int flag, int move) {
        int b = base(key);
        int deep = b;       // depth-preferred slot
        int always = b + 1; // always-replace slot

        long deepData = data.get(deep);
        long deepXor = xorKeys.get(deep);
        boolean deepEmpty = flagOf(deepData) == FLAG_NONE;
        boolean deepSameKey = !deepEmpty && (deepXor ^ deepData) == key;

        int target = (deepEmpty || deepSameKey || depth >= depthOf(deepData)) ? deep : always;

        // Preserve a known best move if this store carries none for the same position.
        // Benign race: another thread may replace `target` between this read and our
        // write below; worst case we fail to preserve an old move, we never corrupt one.
        if (move == 0) {
            long curData = target == deep ? deepData : data.get(target);
            long curXor = target == deep ? deepXor : xorKeys.get(target);
            if (flagOf(curData) != FLAG_NONE && (curXor ^ curData) == key) {
                int oldMove = moveOf(curData);
                if (oldMove != 0) move = oldMove;
            }
        }

        long entry = pack(move, score, depth, flag);
        data.set(target, entry);
        xorKeys.set(target, key ^ entry);
    }

    private static long pack(int move, int score, int depth, int flag) {
        return (move & 0xFFFFL)
                | ((score & 0xFFFFL) << 16)
                | ((depth & 0xFFFFL) << 32)
                | ((long) (flag & 0xFFL) << 48);
    }

    public static int moveOf(long entry) { return (int) (entry & 0xFFFFL); }
    public static int scoreOf(long entry) { return (short) ((entry >>> 16) & 0xFFFFL); }
    public static int depthOf(long entry) { return (short) ((entry >>> 32) & 0xFFFFL); }
    public static int flagOf(long entry) { return (int) ((entry >>> 48) & 0xFFL); }
}
