package engine.uci;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Position;
import engine.search.Search;
import engine.search.SearchLimits;

/**
 * Minimal UCI handler (Phase 1b walking skeleton).
 *
 * Supports: uci, isready, ucinewgame, position (startpos|fen ... [moves ...]),
 * go [depth N], stop, quit. Drives a fixed-depth negamax with material-only eval.
 * The full basic command set (time controls, Hash option, etc.) arrives in Phase 4.
 */
public final class Uci {

    private static final String NAME = "ChessEngine v0.1";
    private static final String AUTHOR = "chess-engine team";
    private static final int DEFAULT_DEPTH = 6;

    private Position position = Position.startpos();
    private final Search search = new Search();

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
                    System.out.println("uciok");
                    break;
                case "isready":
                    System.out.println("readyok");
                    break;
                case "ucinewgame":
                    position = Position.startpos();
                    search.newGame();
                    break;
                case "position":
                    handlePosition(tokens);
                    break;
                case "go":
                    handleGo(tokens);
                    break;
                case "stop":
                    // Single-threaded skeleton: nothing in flight to stop.
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

        int best = search.think(position, limits);
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
