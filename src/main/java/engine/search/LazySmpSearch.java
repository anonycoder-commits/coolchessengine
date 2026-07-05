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
 * never touch the same board, and its own {@link HandcraftedEvaluator} (whose pawn cache is
 * per-instance mutable state). Only the {@link TranspositionTable} is shared, and it is safe
 * for concurrent use.
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
    private final AtomicBoolean sharedStop = new AtomicBoolean(false);
    // Workers persist across think() calls: each owns several MB of history/continuation
    // tables (see Search), so rebuilding them per move would both churn allocation and throw
    // away the within-game ordering signal those tables accumulate -- persistence makes SMP
    // match the single-threaded Search, whose tables also survive from move to move.
    private Search[] workers;
    private volatile boolean pendingPonder; // armed via setPondering, applied to the master
    private volatile long moveOverheadMs = 100; // applied to the master worker each think()
    private volatile int contempt = 10;         // applied to EVERY worker each think()

    public int bestMove;
    public int bestScore;
    public boolean bypassPrinted;

    public LazySmpSearch(int threadCount, int hashMb) {
        this.threadCount = Math.max(1, threadCount);
        this.tt = new TranspositionTable(hashMb);
    }

    /** Builds an SMP manager that shares an externally-owned table instead of allocating its own. */
    public LazySmpSearch(int threadCount, TranspositionTable sharedTt) {
        this.threadCount = Math.max(1, threadCount);
        this.tt = sharedTt;
    }

    public int threadCount() { return threadCount; }

    /** Not safe to call while {@link #think} is running on another thread (see {@link TranspositionTable}). */
    public void setHashSize(int mb) { tt.resize(mb); }

    /** Not safe to call while {@link #think} is running on another thread (see {@link TranspositionTable}). */
    public void newGame() {
        tt.clear();
        if (workers != null) {
            for (Search worker : workers) worker.resetHeuristics();
        }
    }

    /** Signals every worker to stop immediately (e.g. a UCI 'stop' command). */
    public void requestStop() { sharedStop.set(true); }

    /** Arms ponder mode for the next {@link #think} (only the master worker is timed). */
    public void setPondering(boolean p) { pendingPonder = p; }

    /** Sets the move-overhead budget applied to the (timed) master worker on the next {@link #think}. */
    public void setMoveOverhead(long ms) { moveOverheadMs = ms; }

    /** Sets the contempt (draw aversion, cp) applied to every worker on the next {@link #think}. */
    public void setContempt(int cp) { contempt = cp; }

    /** UCI "ponderhit": rebase the master worker's clock and let normal timing take over. */
    public void ponderHit() { if (workers != null) workers[0].ponderHit(); }

    /** The master worker's ponder-move hint (2nd PV move), or 0. */
    public int ponderMove() { return workers != null ? workers[0].ponderMove() : 0; }

    /** Runs the Lazy SMP search and returns the master worker's chosen move. */
    public int think(Position rootPosition, SearchLimits limits) {
        sharedStop.set(false);

        ExecutorService pool = Executors.newFixedThreadPool(threadCount, WORKER_THREADS);
        if (workers == null) {
            workers = new Search[threadCount];
            for (int id = 0; id < threadCount; id++) {
                // One evaluator per worker: HandcraftedEvaluator owns a pawn cache and is
                // therefore NOT shareable across threads (see its class javadoc).
                Search worker = new Search(new HandcraftedEvaluator(), tt);
                worker.printInfo = id == 0; // only the master prints UCI 'info'/'bestmove' text
                worker.setSharedStop(sharedStop);
                workers[id] = worker;
            }
            // The master's info lines report the aggregate SMP node count/nps: its own nodes
            // plus every helper's published snapshot (refreshed each checkTime() interval).
            // Without this, 'info nodes/nps' under-reports by ~threadCount x.
            final Search[] all = workers;
            workers[0].extraNodes = () -> {
                long sum = 0;
                for (int i = 1; i < all.length; i++) sum += all[i].publishedNodes();
                return sum;
            };
        }
        // Zero helper snapshots from this (manager) thread before the new search starts:
        // workers persist across think() calls, and a helper resets its own counter only
        // once its task is running -- the master's first info line would otherwise include
        // the helpers' counts from the PREVIOUS move.
        for (int id = 1; id < threadCount; id++) workers[id].resetPublishedNodes();
        workers[0].setPondering(pendingPonder); // only the master is timed, so only it ponders
        workers[0].moveOverheadMs = moveOverheadMs; // the master owns the clock; keep it in sync
        for (Search worker : workers) worker.contempt = contempt; // every worker scores draws alike
        List<Future<Void>> futures = new ArrayList<>(threadCount);
        try {
            for (int id = 0; id < threadCount; id++) {
                Position workerPosition = new Position(rootPosition); // per-thread deep-copied board
                SearchLimits workerLimits = id == 0 ? limits : helperLimits(limits);
                futures.add(pool.submit(searchTask(workers[id], workerPosition, workerLimits)));
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
