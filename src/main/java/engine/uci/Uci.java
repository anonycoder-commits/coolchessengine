package engine.uci;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Position;
import engine.search.HandcraftedEvaluator;
import engine.search.LazySmpSearch;
import engine.search.NnueEvaluator;
import engine.search.Search;
import engine.search.SearchLimits;
import engine.search.TranspositionTable;

/**
 * Minimal UCI handler (Phase 1b walking skeleton).
 *
 * Supports: uci, isready, ucinewgame, setoption (Hash, Threads), position (startpos|fen
 * ... [moves ...]), go [depth N | movetime | wtime/btime/winc/binc/movestogo | infinite],
 * stop, quit. 'go' runs on a background thread so the input loop can keep reading
 * 'stop'/'quit' while a search is in flight.
 */
public final class Uci {

    private static final String NAME = "ChessEngine v0.1";
    private static final String AUTHOR = "chess-engine team";
    private static final int DEFAULT_DEPTH = 6;
    private static final int DEFAULT_HASH_MB = 16;
    private static final int MAX_THREADS = 128;

    private Position position = Position.startpos();
    private final TranspositionTable tt = new TranspositionTable(DEFAULT_HASH_MB);
    private final Search search = new Search(new HandcraftedEvaluator(), tt);
    private int threads = 1;
    private long moveOverheadMs = 100; // UCI "Move Overhead" option; applied to the search per 'go'
    private int contempt = 10;         // UCI "Contempt" option; applied to the search per 'go'
    private LazySmpSearch smp; // built lazily once Threads > 1 is requested

    // NNUE eval selection (off by default -- handcrafted eval remains the shipping default and
    // the bench signature is unchanged unless UseNNUE is explicitly turned on with a valid net).
    private boolean useNnue = false;
    private String evalFile = "";
    private NnueEvaluator.Network nnueNet; // parsed once, shared read-only across workers

    // A single persistent daemon thread (rather than a fresh Thread per 'go') so the input
    // loop can keep reading 'stop'/'quit' while a search is in flight. Persistent also
    // matters for correctness under piped stdout (e.g. UciProtocolTest): a short-lived
    // per-call Thread that dies right after printing "bestmove" trips PipedInputStream's
    // "Write end dead" check the next time *any* thread writes to the pipe, since it
    // tracks the last writer thread's liveness rather than the stream's as a whole.
    private final ExecutorService searchExecutor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "uci-search");
        t.setDaemon(true);
        return t;
    });
    private volatile boolean searching;

    public static void main(String[] args) throws IOException {
        if (args.length > 0 && args[0].equals("bench")) {
            engine.tools.Bench.run();
            return;
        }
        new Uci().run();
    }

    public void run() throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line = in.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] tokens = line.split("\\s+");
            switch (tokens[0]) {
                case "uci":
                    System.out.println("id name " + NAME);
                    System.out.println("id author " + AUTHOR);
                    System.out.println("option name Hash type spin default 16 min 1 max 1024");
                    System.out.println("option name Threads type spin default 1 min 1 max " + MAX_THREADS);
                    System.out.println("option name Ponder type check default false");
                    System.out.println("option name Move Overhead type spin default 100 min 0 max 5000");
                    System.out.println("option name Contempt type spin default 10 min -50 max 100");
                    System.out.println("option name UseNNUE type check default false");
                    System.out.println("option name EvalFile type string default <empty>");
                    System.out.println("uciok");
                    break;
                case "isready":
                    System.out.println("readyok");
                    break;
                case "ucinewgame":
                    position = Position.startpos();
                    search.newGame();
                    break;
                case "setoption":
                    handleSetOption(tokens);
                    break;
                case "position":
                    handlePosition(tokens);
                    break;
                case "go":
                    // Guard against overlapping 'go' calls, since a compliant GUI won't send
                    // one anyway.
                    if (!searching) {
                        searching = true;
                        searchExecutor.submit(() -> {
                            try {
                                handleGo(tokens);
                            } finally {
                                searching = false;
                            }
                        });
                    }
                    break;
                case "ponderhit":
                    // Opponent played the predicted move: hand the still-running ponder search
                    // over to normal time management (clock rebased to now inside ponderHit).
                    search.ponderHit();
                    if (smp != null) smp.ponderHit();
                    break;
                case "stop":
                    search.requestStop();
                    if (smp != null) smp.requestStop();
                    break;
                case "bench":
                    engine.tools.Bench.run();
                    break;
                case "quit":
                    shutdownSearch();
                    return;
                default:
                    // Unknown command: ignore, per UCI convention.
                    break;
            }
            System.out.flush();
        }
    }

    /**
     * Stops any in-flight search and waits for the search thread to finish before we let the
     * process exit. Without this, {@code quit} returns immediately and the JVM tears down the
     * daemon search thread mid-{@code println}, emitting a truncated {@code info ...} line
     * (e.g. "info depth 5 score cp 0" with no newline) that a UCI wrapper's info parser then
     * chokes on. Bounded so a wedged search can never hang the shutdown.
     */
    private void shutdownSearch() {
        search.requestStop();
        if (smp != null) smp.requestStop();
        searchExecutor.shutdown();
        try {
            searchExecutor.awaitTermination(2, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        System.out.flush();
    }

    /** Handles {@code setoption name <id> [value <x>]}; unknown options are ignored per UCI convention. */
    private void handleSetOption(String[] tokens) {
        int nameIdx = -1, valueIdx = -1;
        for (int i = 1; i < tokens.length; i++) {
            if (tokens[i].equalsIgnoreCase("name")) nameIdx = i;
            else if (tokens[i].equalsIgnoreCase("value")) valueIdx = i;
        }
        if (nameIdx < 0 || nameIdx + 1 >= tokens.length) return;
        // Option names may contain spaces (e.g. "Move Overhead"), so join every token between
        // 'name' and 'value' rather than taking a single token.
        int nameEnd = valueIdx >= 0 ? valueIdx : tokens.length;
        StringBuilder nameBuf = new StringBuilder();
        for (int i = nameIdx + 1; i < nameEnd; i++) {
            if (nameBuf.length() > 0) nameBuf.append(' ');
            nameBuf.append(tokens[i]);
        }
        String name = nameBuf.toString();
        if (name.equalsIgnoreCase("Hash") && valueIdx >= 0) {
            int mb = parseInt(tokens, valueIdx + 1);
            if (mb > 0) {
                tt.resize(mb);
            }
        } else if (name.equalsIgnoreCase("Threads") && valueIdx >= 0) {
            int n = parseInt(tokens, valueIdx + 1);
            if (n > 0) threads = Math.min(n, MAX_THREADS);
        } else if (name.equalsIgnoreCase("Move Overhead") && valueIdx >= 0) {
            int ms = parseInt(tokens, valueIdx + 1);
            if (ms >= 0) moveOverheadMs = Math.min(ms, 5000);
        } else if (name.equalsIgnoreCase("Contempt") && valueIdx >= 0) {
            contempt = Math.max(-50, Math.min(100, parseInt(tokens, valueIdx + 1)));
        } else if (name.equalsIgnoreCase("UseNNUE") && valueIdx >= 0) {
            useNnue = Boolean.parseBoolean(tokens[valueIdx + 1]);
            applyEvaluator();
        } else if (name.equalsIgnoreCase("EvalFile") && valueIdx >= 0) {
            // A path may contain spaces, so join every token after 'value'.
            StringBuilder v = new StringBuilder();
            for (int i = valueIdx + 1; i < tokens.length; i++) {
                if (v.length() > 0) v.append(' ');
                v.append(tokens[i]);
            }
            evalFile = v.toString();
            nnueNet = null; // force a reload from the new path
            if (useNnue) applyEvaluator();
        }
    }

    /**
     * Points the single-threaded search (and the SMP manager, if built) at the currently
     * selected evaluator. NNUE only if it is enabled AND a net loads; otherwise -- including
     * every failure path -- it falls back to the handcrafted eval so the engine always plays.
     */
    private void applyEvaluator() {
        if (useNnue) {
            if (nnueNet == null) loadNnueNet();
            if (nnueNet != null) {
                final NnueEvaluator.Network net = nnueNet;
                search.setEvaluator(new NnueEvaluator(net));
                if (smp != null) smp.setEvaluatorFactory(() -> new NnueEvaluator(net));
                return;
            }
            System.out.println("info string NNUE requested but no net loaded; using handcrafted eval");
        }
        search.setEvaluator(new HandcraftedEvaluator());
        if (smp != null) smp.setEvaluatorFactory(HandcraftedEvaluator::new);
    }

    /** Loads the net from EvalFile, else a bundled {@code /nnue/default.nnue} resource if present. */
    private void loadNnueNet() {
        try {
            if (evalFile != null && !evalFile.isBlank() && !evalFile.equals("<empty>")) {
                try (java.io.InputStream in =
                             java.nio.file.Files.newInputStream(java.nio.file.Path.of(evalFile))) {
                    nnueNet = NnueEvaluator.load(in);
                }
                System.out.println("info string NNUE net loaded from " + evalFile);
                return;
            }
            try (java.io.InputStream in = Uci.class.getResourceAsStream("/nnue/default.nnue")) {
                if (in != null) {
                    nnueNet = NnueEvaluator.load(in);
                    System.out.println("info string NNUE net loaded from bundled default");
                    return;
                }
            }
            System.out.println("info string NNUE: no EvalFile set and no bundled net");
        } catch (Exception e) {
            nnueNet = null;
            System.out.println("info string NNUE load failed: " + e.getMessage());
        }
    }

    private void handlePosition(String[] tokens) {
        int idx;
        if (tokens.length >= 2 && tokens[1].equals("startpos")) {
            position = Position.startpos();
            idx = 2;
        } else if (tokens.length >= 2 && tokens[1].equals("fen")) {
            // FEN is the next 6 tokens.
            StringBuilder fen = new StringBuilder();
            int end = Math.min(tokens.length, 8);
            for (int i = 2; i < end; i++) {
                if (i > 2) fen.append(' ');
                fen.append(tokens[i]);
            }
            position = Position.fromFen(fen.toString());
            idx = 8;
        } else {
            return;
        }

        if (idx < tokens.length && tokens[idx].equals("moves")) {
            for (int i = idx + 1; i < tokens.length; i++) {
                int move = parseMove(tokens[i]);
                if (move == 0) break;
                position.makeMove(move);
            }
        }
    }

    private void handleGo(String[] tokens) {
        SearchLimits limits = new SearchLimits();
        boolean anyLimit = false;
        boolean ponder = false;
        for (int i = 1; i < tokens.length; i++) {
            switch (tokens[i]) {
                case "ponder": ponder = true; break;
                case "depth": limits.depth = parseInt(tokens, ++i); anyLimit = true; break;
                case "movetime": limits.movetime = parseInt(tokens, ++i); anyLimit = true; break;
                case "wtime": limits.wtime = parseInt(tokens, ++i); anyLimit = true; break;
                case "btime": limits.btime = parseInt(tokens, ++i); anyLimit = true; break;
                case "winc": limits.winc = parseInt(tokens, ++i); break;
                case "binc": limits.binc = parseInt(tokens, ++i); break;
                case "movestogo": limits.movestogo = parseInt(tokens, ++i); break;
                case "infinite": limits.infinite = true; anyLimit = true; break;
                default: break;
            }
        }
        if (!anyLimit) limits.depth = DEFAULT_DEPTH;

        int best;
        boolean bypassPrinted;
        int ponderMove = 0;
        if (threads <= 1) {
            // Ponder mode is wired for the single-threaded path (the common bot configuration);
            // the flag makes think() ignore the clock until "ponderhit" rebases it. See Search.
            search.setPondering(ponder);
            search.moveOverheadMs = moveOverheadMs;
            search.contempt = contempt;
            best = search.think(position, limits);
            bypassPrinted = search.bypassPrinted;
            ponderMove = search.ponderMove();
        } else {
            if (smp == null || smp.threadCount() != threads) {
                smp = new LazySmpSearch(threads, tt);
                if (useNnue && nnueNet != null) {
                    final NnueEvaluator.Network net = nnueNet;
                    smp.setEvaluatorFactory(() -> new NnueEvaluator(net));
                }
            }
            smp.setPondering(ponder);
            smp.setMoveOverhead(moveOverheadMs);
            smp.setContempt(contempt);
            best = smp.think(position, limits);
            bypassPrinted = smp.bypassPrinted;
            ponderMove = smp.ponderMove();
        }
        if (bypassPrinted) return; // the search already emitted bestmove for the forced-move case
        String pv = best != 0 ? Move.toUci(best) : "0000";
        // Offer a ponder move so a pondering GUI can search the opponent's expected reply, but
        // only if it is actually legal in the position that arises after 'best' is played. A
        // stale PV or TT artifact can otherwise yield an illegal ponder move, which GUIs such as
        // lichess-bot reject outright ("Engine sent invalid ponder move"). A bare 'bestmove X'
        // with no ponder token is always valid UCI, so we simply drop the token when in doubt.
        if (ponderMove != 0 && best != 0 && !isLegalReply(best, ponderMove)) ponderMove = 0;
        System.out.println(ponderMove != 0
                ? "bestmove " + pv + " ponder " + Move.toUci(ponderMove)
                : "bestmove " + pv);
    }

    /** True if {@code reply} is a legal move in the position reached after playing {@code best}. */
    private boolean isLegalReply(int best, int reply) {
        position.makeMove(best);
        try {
            MoveList legal = new MoveList();
            MoveGenerator.generateLegal(position, legal);
            for (int i = 0; i < legal.size; i++) {
                if (legal.moves[i] == reply) return true;
            }
            return false;
        } finally {
            position.unmakeMove(best);
        }
    }

    private static int parseInt(String[] tokens, int idx) {
        if (idx >= tokens.length) return 0;
        try {
            return Integer.parseInt(tokens[idx]);
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    /** Resolves a UCI long-algebraic move string against the legal moves of the current position. */
    private int parseMove(String uci) {
        MoveList moves = new MoveList();
        MoveGenerator.generateLegal(position, moves);
        for (int i = 0; i < moves.size; i++) {
            if (Move.toUci(moves.moves[i]).equals(uci)) {
                return moves.moves[i];
            }
        }
        return 0;
    }
}
