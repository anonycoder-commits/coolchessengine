package engine.uci;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Position;
import engine.search.HandcraftedEvaluator;
import engine.search.LazySmpSearch;
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
    private LazySmpSearch smp; // built lazily once Threads > 1 is requested

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
                case "stop":
                    search.requestStop();
                    if (smp != null) smp.requestStop();
                    break;
                case "quit":
                    return;
                default:
                    // Unknown command: ignore, per UCI convention.
                    break;
            }
            System.out.flush();
        }
    }

    /** Handles {@code setoption name <id> [value <x>]}; unknown options are ignored per UCI convention. */
    private void handleSetOption(String[] tokens) {
        int nameIdx = -1, valueIdx = -1;
        for (int i = 1; i < tokens.length; i++) {
            if (tokens[i].equalsIgnoreCase("name")) nameIdx = i;
            else if (tokens[i].equalsIgnoreCase("value")) valueIdx = i;
        }
        if (nameIdx < 0 || nameIdx + 1 >= tokens.length) return;
        String name = tokens[nameIdx + 1];
        if (name.equalsIgnoreCase("Hash") && valueIdx >= 0) {
            int mb = parseInt(tokens, valueIdx + 1);
            if (mb > 0) {
                tt.resize(mb);
            }
        } else if (name.equalsIgnoreCase("Threads") && valueIdx >= 0) {
            int n = parseInt(tokens, valueIdx + 1);
            if (n > 0) threads = Math.min(n, MAX_THREADS);
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
        for (int i = 1; i < tokens.length; i++) {
            switch (tokens[i]) {
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
        if (threads <= 1) {
            best = search.think(position, limits);
            bypassPrinted = search.bypassPrinted;
        } else {
            if (smp == null || smp.threadCount() != threads) smp = new LazySmpSearch(threads, tt);
            best = smp.think(position, limits);
            bypassPrinted = smp.bypassPrinted;
        }
        if (bypassPrinted) return; // the search already emitted bestmove for the forced-move case
        String pv = best != 0 ? Move.toUci(best) : "0000";
        System.out.println("bestmove " + pv);
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
