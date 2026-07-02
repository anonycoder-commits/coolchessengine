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
import java.util.concurrent.TimeUnit;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Piece;
import engine.board.Position;
import engine.search.Search;
import engine.search.SearchLimits;

/**
 * Phase-5 scaffold: plays our engine against an external UCI opponent process over
 * N games, alternating colors each game, and reports a W/L/D tally and score percentage.
 * Not yet run against a real reference engine binary.
 */
public final class MatchRunner {

    private static final int DEFAULT_GAMES = 20;
    private static final int DEFAULT_MOVETIME_MS = 100;
    private static final int MAX_PLIES = 300;

    public static void main(String[] args) {
        String opponentPath = null;
        int games = DEFAULT_GAMES;
        int movetimeMs = DEFAULT_MOVETIME_MS;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-opponent":
                    opponentPath = args[++i];
                    break;
                case "-games":
                    games = Integer.parseInt(args[++i]);
                    break;
                case "-movetime":
                    movetimeMs = Integer.parseInt(args[++i]);
                    break;
                default:
                    System.err.println("Unknown argument: " + args[i]);
                    printUsage();
                    System.exit(1);
                    return;
            }
        }

        if (opponentPath == null) {
            printUsage();
            System.exit(1);
            return;
        }

        Path path = Path.of(opponentPath);
        if (!Files.exists(path)) {
            System.err.println("Opponent engine not found: " + opponentPath);
            System.exit(1);
            return;
        }
        if (!Files.isExecutable(path)) {
            System.err.println("Opponent engine is not executable: " + opponentPath);
            System.exit(1);
            return;
        }

        new MatchRunner().run(opponentPath, games, movetimeMs);
    }

    private static void printUsage() {
        System.out.println("Usage: MatchRunner -opponent <path-to-uci-engine> [-games N] [-movetime ms]");
    }

    private int wins, losses, draws;

    private void run(String opponentPath, int games, int movetimeMs) {
        for (int g = 0; g < games; g++) {
            boolean engineIsWhite = (g % 2 == 0);
            OpponentProcess opponent;
            try {
                opponent = new OpponentProcess(opponentPath);
            } catch (IOException e) {
                System.err.println("Failed to start opponent process: " + e.getMessage());
                System.exit(1);
                return;
            }
            try {
                GameResult result = playGame(opponent, engineIsWhite, movetimeMs);
                tally(result, engineIsWhite);
                System.out.println("Game " + (g + 1) + "/" + games + ": " + result
                        + " (engine played " + (engineIsWhite ? "white" : "black") + ")");
            } finally {
                opponent.close();
            }
        }

        int totalDecided = wins + losses + draws;
        double score = totalDecided == 0 ? 0.0 : (wins + 0.5 * draws) / totalDecided;
        System.out.println(String.format(
                "Result: +%d -%d =%d, score %.1f%% over %d games",
                wins, losses, draws, score * 100.0, totalDecided));
    }

    private void tally(GameResult result, boolean engineIsWhite) {
        switch (result) {
            case DRAW: draws++; break;
            case WHITE_WINS: if (engineIsWhite) wins++; else losses++; break;
            case BLACK_WINS: if (engineIsWhite) losses++; else wins++; break;
        }
    }

    private enum GameResult { WHITE_WINS, BLACK_WINS, DRAW }

    private GameResult playGame(OpponentProcess opponent, boolean engineIsWhite, int movetimeMs) {
        Position pos = Position.startpos();
        Search search = new Search();
        search.printInfo = false;
        opponent.newGame();

        List<String> uciMoves = new ArrayList<>();

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
            if (pos.isDrawByRuleOrRepetition()) {
                return GameResult.DRAW;
            }

            boolean engineToMove = (pos.sideToMove() == Piece.WHITE) == engineIsWhite;
            int move;
            if (engineToMove) {
                move = search.think(pos, SearchLimits.moveTime(movetimeMs));
                if (move == 0) move = legal.moves[0];
            } else {
                String uci = opponent.bestMove(uciMoves, movetimeMs);
                move = findMove(legal, uci);
                if (move == 0) {
                    // Opponent returned an unparseable/illegal move; score it as a loss for that side.
                    boolean whiteToMove = pos.sideToMove() == Piece.WHITE;
                    return whiteToMove ? GameResult.BLACK_WINS : GameResult.WHITE_WINS;
                }
            }

            uciMoves.add(Move.toUci(move));
            pos.makeMove(move);
        }
        return GameResult.DRAW; // move-count safety cap
    }

    private static int findMove(MoveList legal, String uci) {
        if (uci == null) return 0;
        for (int i = 0; i < legal.size; i++) {
            if (Move.toUci(legal.moves[i]).equals(uci)) return legal.moves[i];
        }
        return 0;
    }

    /** Drives an external UCI engine subprocess through the standard handshake. */
    private static final class OpponentProcess {
        private final Process process;
        private final BufferedReader out;
        private final PrintWriter in;

        OpponentProcess(String path) throws IOException {
            process = new ProcessBuilder(path).redirectErrorStream(true).start();
            out = new BufferedReader(new InputStreamReader(process.getInputStream()));
            in = new PrintWriter(new OutputStreamWriter(process.getOutputStream()), true);
            send("uci");
            awaitLine("uciok");
            send("isready");
            awaitLine("readyok");
        }

        void newGame() {
            send("ucinewgame");
            send("isready");
            awaitLine("readyok");
        }

        String bestMove(List<String> movesSoFar, int movetimeMs) {
            StringBuilder cmd = new StringBuilder("position startpos");
            if (!movesSoFar.isEmpty()) {
                cmd.append(" moves");
                for (String m : movesSoFar) cmd.append(' ').append(m);
            }
            send(cmd.toString());
            send("go movetime " + movetimeMs);
            String line = awaitLine("bestmove");
            if (line == null) return null;
            String[] parts = line.split("\\s+");
            return parts.length >= 2 ? parts[1] : null;
        }

        private void send(String command) {
            in.println(command);
            in.flush();
        }

        private String awaitLine(String prefix) {
            try {
                String line;
                while ((line = out.readLine()) != null) {
                    if (line.startsWith(prefix)) return line;
                }
            } catch (IOException e) {
                return null;
            }
            return null;
        }

        void close() {
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
