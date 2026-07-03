package engine.tools;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Piece;
import engine.board.Position;
import engine.search.Search;
import engine.search.SearchLimits;

/**
 * Self-play / gauntlet harness: referees games between a tested engine and an opponent,
 * reporting a W/L/D tally, score, and a naive Elo estimate with a 95% confidence interval.
 *
 * <p><b>Referee mode ({@code -engine <path> -opponent <path>}):</b> BOTH players are external
 * UCI subprocesses, so the match is perfectly symmetric. Use this for any A/B gate. An
 * earlier version ran the tested engine in-process (direct {@link Search}) against a
 * subprocess opponent; a 200-game identical-build self-test measured that setup at 59.8%
 * (+69 Elo, 95% CI [20,118]) -- a large structural bias favoring the in-process side
 * (subprocess scheduling / IPC overhead under concurrency). Referee mode removes it.
 *
 * <p><b>Legacy in-process mode ({@code -opponent <path>} only):</b> the tested engine runs
 * in-process for speed. Fast, but carries the bias above -- use only for rough smoke checks,
 * never for shipping decisions.
 *
 * <p>Games use a rotating set of balanced openings (see {@link #OPENINGS}), PAIRED by color
 * (games 2k and 2k+1 play opening k with the tested engine as white then black) so first-move
 * and opening imbalance cancel. Games run {@code -concurrency} at a time; this is an offline
 * tool, so it allocates freely (the zero-alloc rule governs the engine's search loop, not this).
 */
public final class MatchRunner {

    private static final int DEFAULT_GAMES = 20;
    private static final int DEFAULT_MOVETIME_MS = 100;
    private static final int DEFAULT_CONCURRENCY = 2;
    private static final int MAX_PLIES = 300;

    /** Balanced, well-known opening lines (6-10 plies) as UCI move strings. Each is played by
     *  both colors (see class doc). Validated at startup -- a typo aborts before any game. */
    private static final String[] OPENINGS = {
        "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6",              // Ruy Lopez
        "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6",              // Italian
        "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6",              // Scotch
        "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4",              // Petroff
        "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",    // Najdorf Sicilian
        "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6",              // Sicilian ...Nc6
        "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4",                        // French Winawer
        "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 b8d7",             // Caro-Kann
        "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5",                        // Scandinavian
        "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 f2f4 f8g7",             // Pirc
        "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7",             // QGD
        "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4",             // Slav
        "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6 e2e3 e7e6",             // QGA
        "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",             // King's Indian
        "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8",             // Nimzo-Indian
        "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5",             // Grunfeld
        "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7",             // Catalan
        "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6 b1c3 e6d5 c4d5 d7d6",   // Benoni
        "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6 g1f3 f8e7",             // Dutch
        "d2d4 d7d5 g1f3 g8f6 c1f4 e7e6 e2e3 f8d6",             // London
        "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 f8b4",             // English
        "g1f3 d7d5 c2c4 e7e6 g2g3 g8f6 f1g2 f8e7",             // Reti
        "e2e4 e7e5 b1c3 g8f6 g2g3 d7d5 e4d5 f6d5",             // Vienna
        "f2f4 d7d5 g1f3 g8f6 e2e3 g7g6 f1e2 f8g7",             // Bird
    };

    public static void main(String[] args) {
        String enginePath = null;   // tested engine; null => run it in-process (legacy, biased)
        String opponentPath = null;
        int games = DEFAULT_GAMES;
        int movetimeMs = DEFAULT_MOVETIME_MS;
        int concurrency = DEFAULT_CONCURRENCY;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-engine": enginePath = args[++i]; break;
                case "-opponent": opponentPath = args[++i]; break;
                case "-games": games = Integer.parseInt(args[++i]); break;
                case "-movetime": movetimeMs = Integer.parseInt(args[++i]); break;
                case "-concurrency": concurrency = Integer.parseInt(args[++i]); break;
                default:
                    System.err.println("Unknown argument: " + args[i]);
                    printUsage();
                    System.exit(1);
                    return;
            }
        }

        if (opponentPath == null || !checkSpec(opponentPath) || (enginePath != null && !checkSpec(enginePath))) {
            if (opponentPath == null) printUsage();
            System.exit(1);
            return;
        }

        validateOpenings(); // fail fast on a book typo, before spawning any subprocess
        new MatchRunner().run(enginePath, opponentPath, games, movetimeMs, Math.max(1, concurrency));
    }

    /** Player spec is either {@code cp:<classes-dir>} (our own build: launched by invoking
     *  {@code java} directly, no shell/.bat/cmd.exe hop -- see SubprocessEngine) or a path to
     *  an external UCI executable (unchanged). */
    private static boolean checkSpec(String spec) {
        if (spec.startsWith("cp:")) {
            Path dir = Path.of(spec.substring(3));
            if (!Files.isDirectory(dir)) { System.err.println("Not a classes directory: " + spec); return false; }
            return true;
        }
        Path path = Path.of(spec);
        if (!Files.exists(path)) { System.err.println("Not found: " + spec); return false; }
        if (!Files.isExecutable(path)) { System.err.println("Not executable: " + spec); return false; }
        return true;
    }

    private static void printUsage() {
        System.out.println("Usage: MatchRunner [-engine <spec>] -opponent <spec> "
                + "[-games N] [-movetime ms] [-concurrency N]");
        System.out.println("  <spec> is either cp:<classes-dir> for our own build (direct java exec, no");
        System.out.println("  shell hop -- fast) or a path to an external UCI engine executable.");
        System.out.println("  -engine given  -> referee mode (both subprocess, unbiased; use for gates)");
        System.out.println("  -engine absent -> tested engine runs in-process (fast, biased; smoke only)");
    }

    private static void validateOpenings() {
        for (String opening : OPENINGS) {
            Position pos = Position.startpos();
            for (String uci : opening.split("\\s+")) {
                MoveList legal = new MoveList();
                MoveGenerator.generateLegal(pos, legal);
                int move = findMove(legal, uci);
                if (move == 0) {
                    throw new IllegalStateException("Illegal opening move '" + uci + "' in: " + opening);
                }
                pos.makeMove(move);
            }
        }
    }

    // Synchronized tally (games complete on multiple threads).
    private int wins, losses, draws, failed;

    private synchronized void tally(GameResult result, boolean engineIsWhite) {
        if (result == null) { failed++; return; }
        switch (result) {
            case DRAW: draws++; break;
            case WHITE_WINS: if (engineIsWhite) wins++; else losses++; break;
            case BLACK_WINS: if (engineIsWhite) losses++; else wins++; break;
        }
    }

    private void run(String enginePath, String opponentPath, int games, int movetimeMs, int concurrency) {
        System.out.println(enginePath == null
                ? "Mode: in-process tested engine (BIASED -- smoke checks only)"
                : "Mode: referee (both subprocess -- unbiased)");
        ExecutorService pool = Executors.newFixedThreadPool(concurrency);
        List<Future<GameResult>> futures = new ArrayList<>(games);
        boolean[] engineWhiteByGame = new boolean[games];
        try {
            for (int g = 0; g < games; g++) {
                final boolean engineIsWhite = (g % 2 == 0);
                final String opening = OPENINGS[(g / 2) % OPENINGS.length];
                engineWhiteByGame[g] = engineIsWhite;
                futures.add(pool.submit(gameTask(enginePath, opponentPath, engineIsWhite, opening, movetimeMs)));
            }
            for (int g = 0; g < games; g++) {
                GameResult result;
                try {
                    result = futures.get(g).get();
                } catch (ExecutionException e) {
                    System.out.println("Game " + (g + 1) + "/" + games + ": FAILED (" + e.getCause() + ")");
                    tally(null, engineWhiteByGame[g]);
                    continue;
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
                tally(result, engineWhiteByGame[g]);
                System.out.println("Game " + (g + 1) + "/" + games + ": " + result
                        + " (engine played " + (engineWhiteByGame[g] ? "white" : "black") + ")");
            }
        } finally {
            pool.shutdownNow();
        }
        report(games);
    }

    private Callable<GameResult> gameTask(String enginePath, String opponentPath,
                                          boolean engineIsWhite, String opening, int movetimeMs) {
        return () -> {
            Player engine = (enginePath == null) ? new InProcessEngine() : new SubprocessEngine(enginePath);
            Player opponent = new SubprocessEngine(opponentPath);
            try {
                Player white = engineIsWhite ? engine : opponent;
                Player black = engineIsWhite ? opponent : engine;
                return playGame(white, black, opening, movetimeMs);
            } finally {
                engine.close();
                opponent.close();
            }
        };
    }

    private void report(int games) {
        int n = wins + losses + draws;
        double score = n == 0 ? 0.0 : (wins + 0.5 * draws) / n;
        StringBuilder sb = new StringBuilder(String.format(
                "Result: +%d -%d =%d, score %.1f%% over %d games", wins, losses, draws, score * 100.0, n));
        if (failed > 0) sb.append(" (").append(failed).append(" failed)");
        if (n > 0 && score > 0.0 && score < 1.0) {
            double[] ci = wilson(score, n);
            sb.append(String.format(" | Elo %+.0f [%.0f, %.0f] (95%%)", elo(score), elo(ci[0]), elo(ci[1])));
        }
        System.out.println(sb);
    }

    private static double elo(double score) {
        double s = Math.max(1e-4, Math.min(1 - 1e-4, score));
        return -400.0 * Math.log10(1.0 / s - 1.0);
    }

    private static double[] wilson(double p, int n) {
        double z = 1.96, z2 = z * z;
        double denom = 1 + z2 / n;
        double centre = (p + z2 / (2 * n)) / denom;
        double half = z * Math.sqrt(p * (1 - p) / n + z2 / (4.0 * n * n)) / denom;
        return new double[] {Math.max(0, centre - half), Math.min(1, centre + half)};
    }

    private enum GameResult { WHITE_WINS, BLACK_WINS, DRAW }

    /** Referees one game between two players, adjudicating termination itself. Players only
     *  ever receive the move list (symmetric) and return a UCI move. */
    private GameResult playGame(Player white, Player black, String opening, int movetimeMs) {
        Position pos = Position.startpos();
        white.newGame();
        black.newGame();

        List<String> uciMoves = new ArrayList<>();
        for (String uci : opening.split("\\s+")) {
            MoveList legal = new MoveList();
            MoveGenerator.generateLegal(pos, legal);
            uciMoves.add(uci);
            pos.makeMove(findMove(legal, uci));
        }

        for (int ply = 0; ply < MAX_PLIES; ply++) {
            MoveList legal = new MoveList();
            MoveGenerator.generateLegal(pos, legal);
            if (legal.size == 0) {
                boolean whiteToMove = pos.sideToMove() == Piece.WHITE;
                if (pos.inCheck(pos.sideToMove())) {
                    return whiteToMove ? GameResult.BLACK_WINS : GameResult.WHITE_WINS;
                }
                return GameResult.DRAW;
            }
            if (pos.isDrawByRuleOrRepetition()) return GameResult.DRAW;

            Player toMove = pos.sideToMove() == Piece.WHITE ? white : black;
            String uci = toMove.bestMove(uciMoves, movetimeMs);
            int move = findMove(legal, uci);
            if (move == 0) {
                // Unparseable/illegal move: the side to move forfeits.
                boolean whiteToMove = pos.sideToMove() == Piece.WHITE;
                return whiteToMove ? GameResult.BLACK_WINS : GameResult.WHITE_WINS;
            }
            uciMoves.add(Move.toUci(move));
            pos.makeMove(move);
        }
        return GameResult.DRAW; // move-count safety cap
    }

    /** Absolute path to the currently-running JVM's java executable, so a {@code cp:} spec
     *  launches with the exact same JDK this tool is running under. */
    private static String javaBin() {
        return Path.of(System.getProperty("java.home"), "bin", "java").toString();
    }

    private static int findMove(MoveList legal, String uci) {
        if (uci == null) return 0;
        for (int i = 0; i < legal.size; i++) {
            if (Move.toUci(legal.moves[i]).equals(uci)) return legal.moves[i];
        }
        return 0;
    }

    // --- players ---

    private interface Player {
        void newGame();
        String bestMove(List<String> movesSoFar, int movetimeMs);
        void close();
    }

    /** In-process tested engine (legacy, biased -- see class doc). Rebuilds the board from the
     *  move list each turn to stay behaviourally identical to how a UCI driver feeds it. */
    private static final class InProcessEngine implements Player {
        private final Search search = new Search();
        InProcessEngine() { search.printInfo = false; }

        @Override public void newGame() { search.newGame(); }

        @Override public String bestMove(List<String> movesSoFar, int movetimeMs) {
            Position pos = Position.startpos();
            for (String u : movesSoFar) {
                MoveList legal = new MoveList();
                MoveGenerator.generateLegal(pos, legal);
                pos.makeMove(findMove(legal, u));
            }
            int move = search.think(pos, SearchLimits.moveTime(movetimeMs));
            if (move == 0) {
                MoveList legal = new MoveList();
                MoveGenerator.generateLegal(pos, legal);
                if (legal.size > 0) move = legal.moves[0];
            }
            return move == 0 ? null : Move.toUci(move);
        }

        @Override public void close() {}
    }

    /** Drives an external UCI engine subprocess through the standard handshake.
     *
     * <p>Every wait for a response is BOUNDED: a background reader thread continuously drains
     * the process's stdout into a queue, and every {@code awaitLine} call polls that queue
     * with a timeout instead of blocking on {@link BufferedReader#readLine} directly (which
     * cannot be interrupted once blocked in native I/O). Without this, a single stuck opponent
     * process -- a bug, a crash that leaves the pipe open, anything -- silently wedges the
     * entire match forever with 0% CPU on every thread (observed live: a run stalled for 15+
     * minutes at 0% CPU growth with no error, no forfeit, nothing). A timeout now surfaces as
     * a null response, which callers already treat as a forfeit for that side.
     */
    private static final class SubprocessEngine implements Player {
        private static final long HANDSHAKE_TIMEOUT_MS = 10_000;

        private final Process process;
        private final PrintWriter in;
        private final java.util.concurrent.BlockingQueue<String> lines = new java.util.concurrent.LinkedBlockingQueue<>();
        // Sentinel enqueued by the reader thread on stream EOF/error, so a poller blocked past
        // that point unblocks immediately instead of waiting out its full timeout for nothing.
        private static final String EOF = new Object().toString();

        SubprocessEngine(String spec) throws IOException {
            ProcessBuilder pb = spec.startsWith("cp:")
                    // Direct java exec: no cmd.exe/.bat hop (that indirection roughly doubled
                    // process-spawn latency and, under concurrency, caused severe stalls --
                    // see the class doc's referee-mode note).
                    ? new ProcessBuilder(javaBin(), "-cp", spec.substring(3), "engine.uci.Uci")
                    : new ProcessBuilder(spec);
            process = pb.redirectErrorStream(true).start();
            in = new PrintWriter(new OutputStreamWriter(process.getOutputStream()), true);
            Thread reader = new Thread(this::pump, "matchrunner-reader");
            reader.setDaemon(true);
            reader.start();
            send("uci");
            if (awaitLine("uciok", HANDSHAKE_TIMEOUT_MS) == null) {
                throw new IOException("no uciok from " + spec + " within " + HANDSHAKE_TIMEOUT_MS + "ms");
            }
            send("isready");
            if (awaitLine("readyok", HANDSHAKE_TIMEOUT_MS) == null) {
                throw new IOException("no readyok from " + spec + " within " + HANDSHAKE_TIMEOUT_MS + "ms");
            }
        }

        /** Runs on the dedicated reader thread: reads until EOF/error, queuing every line. */
        private void pump() {
            try (BufferedReader r = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = r.readLine()) != null) lines.offer(line);
            } catch (IOException ignored) {
                // process died/pipe broke -- fall through to the EOF sentinel below
            } finally {
                lines.offer(EOF);
            }
        }

        @Override public void newGame() {
            send("ucinewgame");
            send("isready");
            awaitLine("readyok", HANDSHAKE_TIMEOUT_MS);
        }

        @Override public String bestMove(List<String> movesSoFar, int movetimeMs) {
            StringBuilder cmd = new StringBuilder("position startpos");
            if (!movesSoFar.isEmpty()) {
                cmd.append(" moves");
                for (String m : movesSoFar) cmd.append(' ').append(m);
            }
            send(cmd.toString());
            send("go movetime " + movetimeMs);
            // Generous margin over the requested think time: real engines occasionally overrun
            // movetime by a bit under load, but a search that's still running at many multiples
            // of what it was asked for is a hang, not slowness, and must not block the match.
            long timeoutMs = Math.max(3000, movetimeMs * 10L) + 5000;
            String line = awaitLine("bestmove", timeoutMs);
            if (line == null) return null;
            String[] parts = line.split("\\s+");
            return parts.length >= 2 ? parts[1] : null;
        }

        private void send(String command) {
            in.println(command);
            in.flush();
        }

        /** Waits up to {@code timeoutMs} total for a line starting with {@code prefix},
         *  discarding non-matching lines (e.g. "info ...") along the way. */
        private String awaitLine(String prefix, long timeoutMs) {
            long deadline = System.nanoTime() + timeoutMs * 1_000_000L;
            while (true) {
                long remaining = (deadline - System.nanoTime()) / 1_000_000L;
                if (remaining <= 0) return null;
                String line;
                try {
                    line = lines.poll(remaining, java.util.concurrent.TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return null;
                }
                if (line == null || line == EOF) return null; // timeout or stream ended
                if (line.startsWith(prefix)) return line;
            }
        }

        @Override public void close() {
            try {
                send("quit");
            } catch (Exception ignored) {
                // process may already be gone
            }
            process.destroy();
            try {
                process.waitFor(2, TimeUnit.SECONDS);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
            if (process.isAlive()) process.destroyForcibly();
        }
    }
}
