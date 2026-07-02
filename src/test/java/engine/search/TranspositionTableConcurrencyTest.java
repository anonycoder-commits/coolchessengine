package engine.search;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * Hammers {@link TranspositionTable} from many threads at once (the Lazy SMP usage
 * pattern) and verifies the lockless-hashing verification never lets a torn write slip
 * through as a false hit.
 *
 * All logical keys are deliberately forced into the same two physical slots (they share
 * the table's low bucket bits, see {@link #BUCKET_BITS}), so every store from every
 * thread constantly evicts and overwrites the other keys' entries -- the adversarial
 * case for the {@code data}/{@code xorKey} word pair. Crucially the keys otherwise keep
 * full 64-bit random entropy in their upper bits, like real Zobrist keys: the lockless
 * scheme's false-hit probability comes down to how likely {@code data ^ data'} is to
 * collide with {@code key ^ key'} for two racing writers, and that's only meaningfully
 * small when the keys involved actually differ with real entropy. An earlier version of
 * this test used tiny, closely-spaced keys (bad key entropy) and could flake with an
 * occasional false-hit as a result -- not a bug in the table, but an unrealistic test.
 */
class TranspositionTableConcurrencyTest {

    private static final int KEY_COUNT = 8;
    private static final int THREAD_COUNT = 8;
    private static final int ITERATIONS_PER_THREAD = 50_000;

    // bucketMask for a 1 MB table today (32768 buckets -> 15 mask bits; see
    // TranspositionTable.resize). Clearing these low bits forces every key into the same
    // bucket while leaving the other 49 bits fully random, unlike real Zobrist keys only
    // in that they're guaranteed, not merely likely, to collide on bucket index.
    private static final long BUCKET_BITS = 0x7FFFL;

    @Test
    void concurrentStoreAndProbeNeverReturnsATornEntry() throws InterruptedException {
        TranspositionTable tt = new TranspositionTable(1);

        Random keyRnd = new Random(42);
        long[] keys = new long[KEY_COUNT];
        for (int i = 0; i < KEY_COUNT; i++) keys[i] = keyRnd.nextLong() & ~BUCKET_BITS;

        AtomicBoolean corrupted = new AtomicBoolean(false);
        AtomicReference<String> failureDetail = new AtomicReference<>();

        ExecutorService pool = Executors.newFixedThreadPool(THREAD_COUNT);
        List<Future<?>> futures = new ArrayList<>(THREAD_COUNT);
        try {
            for (int t = 0; t < THREAD_COUNT; t++) {
                futures.add(pool.submit(() -> {
                    Random rnd = new Random();
                    for (int iter = 0; iter < ITERATIONS_PER_THREAD && !corrupted.get(); iter++) {
                        int ki = rnd.nextInt(KEY_COUNT);
                        long key = keys[ki];
                        int depth = rnd.nextInt(8);
                        int flag = 1 + rnd.nextInt(3); // EXACT/LOWER/UPPER, never FLAG_NONE
                        int score = rnd.nextInt(2001) - 1000;
                        // Tag the move field with the key index so a returned hit can be
                        // checked against the key it was actually probed for.
                        int move = ki | (ki << 6);
                        tt.store(key, depth, score, flag, move);

                        long entry = tt.probe(key);
                        if (TranspositionTable.flagOf(entry) == TranspositionTable.FLAG_NONE) continue;

                        int gotMove = TranspositionTable.moveOf(entry);
                        int gotFrom = gotMove & 0x3F;
                        int gotTo = (gotMove >>> 6) & 0x3F;
                        if (gotFrom != ki || gotTo != ki) {
                            corrupted.set(true);
                            failureDetail.compareAndSet(null,
                                    "probe(key for index " + ki + ") returned an entry tagged for index "
                                            + gotFrom + "/" + gotTo);
                        }
                    }
                }));
            }
            for (Future<?> f : futures) f.get();
        } catch (java.util.concurrent.ExecutionException e) {
            throw new AssertionError("worker thread threw", e.getCause());
        } finally {
            pool.shutdown();
        }

        assertNull(failureDetail.get(), failureDetail.get());
    }
}
