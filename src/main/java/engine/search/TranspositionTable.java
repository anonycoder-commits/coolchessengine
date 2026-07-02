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

    /** Sentinel stored in the eval field when a store carries no static eval. */
    public static final int NO_EVAL = 32767;

    public void store(long key, int depth, int score, int flag, int move, int eval, int generation) {
        int b = base(key);
        int deep = b;       // depth-preferred slot
        int always = b + 1; // always-replace slot

        long deepData = data.get(deep);
        long deepXor = xorKeys.get(deep);
        boolean deepEmpty = flagOf(deepData) == FLAG_NONE;
        boolean deepSameKey = !deepEmpty && (deepXor ^ deepData) == key;

        // Depth-preferred slot is eligible when empty, same key, from an older generation
        // (stale entries from previous searches yield to fresh ones regardless of depth), or
        // this store is at least as deep as what's there.
        boolean deepFromOldGen = !deepEmpty && generationOf(deepData) != generation;
        int target = (deepEmpty || deepSameKey || deepFromOldGen || depth >= depthOf(deepData))
                ? deep : always;

        // Preserve a known best move / static eval if this store carries none for the same
        // position. Benign race: another thread may replace `target` between this read and our
        // write below; worst case we fail to preserve an old value, we never corrupt one.
        long curData = target == deep ? deepData : data.get(target);
        long curXor = target == deep ? deepXor : xorKeys.get(target);
        boolean curSameKey = flagOf(curData) != FLAG_NONE && (curXor ^ curData) == key;
        if (move == 0 && curSameKey) {
            int oldMove = moveOf(curData);
            if (oldMove != 0) move = oldMove;
        }
        if (eval == NO_EVAL && curSameKey) {
            eval = evalOf(curData); // may still be NO_EVAL; harmless
        }

        long entry = pack(move, score, depth, flag, eval, generation);
        data.set(target, entry);
        xorKeys.set(target, key ^ entry);
    }

    // Layout (bits): move[0..15] | score[16..31] | eval[32..47] | depth[48..55] |
    //                flag[56..57] | generation[58..63].
    private static long pack(int move, int score, int depth, int flag, int eval, int generation) {
        int d = Math.max(0, Math.min(127, depth)); // 8-bit unsigned depth; clamp defensively
        return (move & 0xFFFFL)
                | ((score & 0xFFFFL) << 16)
                | ((eval & 0xFFFFL) << 32)
                | ((long) (d & 0xFFL) << 48)
                | ((long) (flag & 0x3L) << 56)
                | ((long) (generation & 0x3FL) << 58);
    }

    public static int moveOf(long entry) { return (int) (entry & 0xFFFFL); }
    public static int scoreOf(long entry) { return (short) ((entry >>> 16) & 0xFFFFL); }
    public static int evalOf(long entry) { return (short) ((entry >>> 32) & 0xFFFFL); }
    public static int depthOf(long entry) { return (int) ((entry >>> 48) & 0xFFL); }
    public static int flagOf(long entry) { return (int) ((entry >>> 56) & 0x3L); }
    public static int generationOf(long entry) { return (int) ((entry >>> 58) & 0x3FL); }
}
