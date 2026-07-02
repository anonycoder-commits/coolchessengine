package engine.search;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import engine.board.Position;

/**
 * Lazy SMP search manager: runs {@code threadCount} independent {@link Search} workers
 * against the same root position, all sharing one {@link TranspositionTable}.
 *
 * Thread isolation needs no special wrapper beyond "don't share the mutable objects":
 * each worker gets its own {@link Search} instance (which already owns its killer table,
 * history table, and move-ordering scratch buffers as private instance fields -- nothing
 * about those is static) and its own deep-copied {@link Position} (see
 * {@link Position#Position(Position)}), so two workers making/unmaking moves concurrently
 * never touch the same board. Only the {@link TranspositionTable} and the stateless
 * {@link Evaluator} are shared, and both are safe for concurrent use.
 *
 * Only the thread-0 ("master") worker is timed and allowed to print UCI output; the
 * other ("helper") workers search with no time limit of their own -- purely to widen the
 * shared transposition table along different move orderings -- and stop only when the
 * master does (propagated through a shared {@link AtomicBoolean}) or when they reach the
 * maximum ply.
 */
public final class LazySmpSearch {

    private static final ThreadFactory WORKER_THREADS = new ThreadFactory() {
        private final AtomicInteger n = new AtomicInteger();
        @Override public Thread newThread(Runnable r) {
            Thread t = new Thread(r, "smp-worker-" + n.getAndIncrement());
            t.setDaemon(true);
            return t;
        }
    };

    private final int threadCount;
    private final TranspositionTable tt;
    private final Evaluator evaluator;
    private final AtomicBoolean sharedStop = new AtomicBoolean(false);

    public int bestMove;
    public int bestScore;
    public boolean bypassPrinted;

    public LazySmpSearch(int threadCount, int hashMb) {
        this.threadCount = Math.max(1, threadCount);
        this.tt = new TranspositionTable(hashMb);
        this.evaluator = new HandcraftedEvaluator();
    }

    /** Builds an SMP manager that shares an externally-owned table instead of allocating its own. */
    public LazySmpSearch(int threadCount, TranspositionTable sharedTt) {
        this.threadCount = Math.max(1, threadCount);
        this.tt = sharedTt;
        this.evaluator = new HandcraftedEvaluator();
    }

    public int threadCount() { return threadCount; }

    /** Not safe to call while {@link #think} is running on another thread (see {@link TranspositionTable}). */
    public void setHashSize(int mb) { tt.resize(mb); }

    /** Not safe to call while {@link #think} is running on another thread (see {@link TranspositionTable}). */
    public void newGame() { tt.clear(); }

    /** Signals every worker to stop immediately (e.g. a UCI 'stop' command). */
    public void requestStop() { sharedStop.set(true); }

    /** Runs the Lazy SMP search and returns the master worker's chosen move. */
    public int think(Position rootPosition, SearchLimits limits) {
        sharedStop.set(false);

        ExecutorService pool = Executors.newFixedThreadPool(threadCount, WORKER_THREADS);
        Search[] workers = new Search[threadCount];
        List<Future<Void>> futures = new ArrayList<>(threadCount);
        try {
            for (int id = 0; id < threadCount; id++) {
                boolean isMaster = id == 0;
                Search worker = new Search(evaluator, tt);
                worker.printInfo = isMaster; // only the master prints UCI 'info'/'bestmove' text
                worker.setSharedStop(sharedStop);
                workers[id] = worker;

                Position workerPosition = new Position(rootPosition); // per-thread deep-copied board
                SearchLimits workerLimits = isMaster ? limits : helperLimits(limits);
                futures.add(pool.submit(searchTask(worker, workerPosition, workerLimits)));
            }

            // Block only on the master; whatever reason it stops for (time, forced mate,
            // an obvious-move cutoff, a single legal reply), tell every helper to stop too.
            try {
                awaitMaster(futures.get(0));
            } finally {
                // Always signal helpers to stop and wait for them, even if the master threw --
                // otherwise they'd keep racing the shared TT indefinitely as daemon threads
                // outliving this think() call.
                sharedStop.set(true);
                // Helpers are advisory (they only widen the shared TT); a bug in one must
                // never cost the master's already-computed, valid result.
                for (int id = 1; id < threadCount; id++) awaitHelper(futures.get(id));
            }

            bestMove = workers[0].bestMove;
            bestScore = workers[0].bestScore;
            bypassPrinted = workers[0].bypassPrinted;
            return bestMove;
        } finally {
            pool.shutdownNow();
        }
    }

    private static Callable<Void> searchTask(Search worker, Position position, SearchLimits limits) {
        return () -> {
            worker.think(position, limits);
            return null;
        };
    }

    /** Helper workers ignore the clock entirely and rely solely on {@link #sharedStop},
     *  set once the master finishes; they still respect a fixed-depth ceiling if one was given. */
    private static SearchLimits helperLimits(SearchLimits master) {
        SearchLimits helper = new SearchLimits();
        helper.depth = master.depth;
        helper.infinite = true;
        return helper;
    }

    private static void awaitMaster(Future<Void> future) {
        try {
            future.get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while waiting for the master SMP worker", e);
        } catch (ExecutionException e) {
            throw new RuntimeException("Master SMP worker failed", e.getCause());
        }
    }

    /** A helper's job is purely to widen the shared TT; if it fails, that's a lost
     *  optimization, not a search failure, so it's logged rather than propagated. */
    private static void awaitHelper(Future<Void> future) {
        try {
            future.get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (ExecutionException e) {
            System.err.println("SMP helper worker failed (ignored): " + e.getCause());
        }
    }
}
