package engine.tools;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Piece;
import engine.board.Position;
import engine.search.Search;
import engine.search.SearchLimits;

/**
 * NNUE self-play data generator (roadmap Phase 4 flywheel).
 *
 * Plays engine-vs-itself games from randomized openings and writes, for each quiet position
 * visited, one bullet-text line {@code <FEN> | <eval-cp> | <wdl>}:
 * <ul>
 *   <li><b>eval</b> = the engine's own shallow-search score, WHITE-relative centipawns -- a
 *       dense, low-noise label (unlike the game-outcome-only bootstrap corpus). Trained blended
 *       with WDL, this is the real strength lever.</li>
 *   <li><b>wdl</b> = the eventual game result, WHITE-relative (1.0 / 0.5 / 0.0).</li>
 * </ul>
 *
 * Uses the handcrafted eval (the strongest current evaluator) as the teacher for this first
 * generation; later generations point {@code -eval} at the best NNUE net (the flywheel).
 * Positions are filtered to QUIET ones -- side-to-move not in check, best move not a capture,
 * and non-mate scores -- so eval labels are stable. Openings are randomized ({@code -random}
 * plies of uniform-random legal moves) so games diversify instead of repeating.
 *
 * <p>Shallow by design: the roadmap notes deep (d20) labels train WORSE from scratch; d6-8 /
 * a few-ms movetime is the sweet spot. Output is appended incrementally and is usable at any
 * point -- stop with Ctrl+C once you have enough positions.
 *
 * <pre>
 *   java -cp build/classes/java/main engine.tools.DataGen \
 *       -out tools/nnue/data/selfplay.txt -games 20000 -depth 8 -random 8 -concurrency 8
 * </pre>
 */
public final class DataGen {

    private static final int MAX_PLIES = 400; // safety cap per game

    private final BufferedWriter out;
    private final int baseRandomPlies;
    private final AtomicInteger gamesDone = new AtomicInteger();
    private final AtomicLong positionsWritten = new AtomicLong();
    private final long startMs = System.currentTimeMillis();

    private DataGen(BufferedWriter out, int baseRandomPlies) {
        this.out = out;
        this.baseRandomPlies = baseRandomPlies;
    }

    public static void main(String[] args) throws Exception {
        String outPath = "tools/nnue/data/selfplay.txt";
        int games = 20000;
        int depth = 8;
        int movetime = 0;      // if >0, overrides depth
        int randomPlies = 8;
        int concurrency = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        long seed = 1;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-out": outPath = args[++i]; break;
                case "-games": games = Integer.parseInt(args[++i]); break;
                case "-depth": depth = Integer.parseInt(args[++i]); break;
                case "-movetime": movetime = Integer.parseInt(args[++i]); break;
                case "-random": randomPlies = Integer.parseInt(args[++i]); break;
                case "-concurrency": concurrency = Integer.parseInt(args[++i]); break;
                case "-seed": seed = Long.parseLong(args[++i]); break;
                default:
                    System.err.println("Unknown arg: " + args[i]);
                    System.err.println("Usage: DataGen [-out FILE] [-games N] [-depth D | -movetime MS]"
                            + " [-random PLIES] [-concurrency N] [-seed S]");
                    System.exit(2);
                    return;
            }
        }

        Path path = Path.of(outPath);
        if (path.getParent() != null) Files.createDirectories(path.getParent());
        final int fGames = games, fDepth = depth, fMovetime = movetime, fRandom = randomPlies;
        final long fSeed = seed;

        System.out.printf("DataGen: %d games, %s, random=%d plies, concurrency=%d -> %s%n",
                games, movetime > 0 ? movetime + "ms/move" : "depth " + depth, randomPlies,
                concurrency, outPath);

        try (BufferedWriter w = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            DataGen gen = new DataGen(w, fRandom);
            ExecutorService pool = Executors.newFixedThreadPool(concurrency);
            AtomicInteger nextGame = new AtomicInteger();
            List<Runnable> workers = new ArrayList<>();
            for (int t = 0; t < concurrency; t++) {
                workers.add(() -> {
                    Search search = new Search();
                    search.printInfo = false;
                    search.moveOverheadMs = 0; // don't burn the tiny per-move budget on overhead
                    SearchLimits limits = fMovetime > 0
                            ? SearchLimits.moveTime(fMovetime) : SearchLimits.depth(fDepth);
                    int g;
                    while ((g = nextGame.getAndIncrement()) < fGames) {
                        gen.playAndRecord(search, limits, fSeed + g);
                    }
                });
            }
            for (Runnable r : workers) pool.submit(r);
            pool.shutdown();
            // Progress ticker until all games finish.
            while (!pool.awaitTermination(5, TimeUnit.SECONDS)) {
                gen.reportProgress(fGames);
            }
            gen.reportProgress(fGames);
        }
        System.out.println("Done.");
    }

    /** Plays one randomized-opening self-play game and appends its quiet positions' labels. */
    private void playAndRecord(Search search, SearchLimits limits, long seed) {
        Random rng = new Random(seed);
        Position pos = Position.startpos();
        search.newGame();

        // Randomized opening: uniform-random legal moves for diversity.
        for (int r = 0; r < randomPliesFrom(seed); r++) {
            MoveList legal = new MoveList();
            MoveGenerator.generateLegal(pos, legal);
            if (legal.size == 0) return; // opening ended the game; discard
            pos.makeMove(legal.moves[rng.nextInt(legal.size)]);
        }

        List<String> fenEval = new ArrayList<>(); // "<fen> | <white-eval>" pending the game result
        int result; // 1=white win, 0=draw, -1=black win
        int ply = 0;
        while (true) {
            MoveList legal = new MoveList();
            MoveGenerator.generateLegal(pos, legal);
            boolean whiteToMove = pos.sideToMove() == Piece.WHITE;
            if (legal.size == 0) {
                result = pos.inCheck(pos.sideToMove()) ? (whiteToMove ? -1 : 1) : 0;
                break;
            }
            if (pos.isDrawByRuleOrRepetition() || ply >= MAX_PLIES) { result = 0; break; }

            int move = search.think(new Position(pos), limits);
            int score = search.bestScore; // side-to-move relative cp
            if (move == 0) { result = 0; break; } // defensive

            boolean quiet = !pos.inCheck(pos.sideToMove())
                    && !Move.isCapture(move)
                    && Math.abs(score) < Search.MATE_IN_MAX;
            if (quiet) {
                int whiteEval = whiteToMove ? score : -score;
                fenEval.add(fen(pos) + " | " + whiteEval);
            }
            pos.makeMove(move);
            ply++;
        }

        String wdl = result > 0 ? "1.0" : result < 0 ? "0.0" : "0.5";
        writeBlock(fenEval, wdl);
        gamesDone.incrementAndGet();
    }

    /** Small per-game jitter in opening length (+/-2) so games don't all branch at the same ply. */
    private int randomPliesFrom(long seed) {
        return Math.max(0, baseRandomPlies + (int) (Math.floorMod(seed * 2654435761L, 5) - 2));
    }

    private synchronized void writeBlock(List<String> fenEval, String wdl) {
        try {
            for (String line : fenEval) {
                out.write(line);
                out.write(" | ");
                out.write(wdl);
                out.write('\n');
            }
            positionsWritten.addAndGet(fenEval.size());
        } catch (IOException e) {
            throw new RuntimeException("write failed", e);
        }
    }

    private void reportProgress(int totalGames) {
        int g = gamesDone.get();
        long p = positionsWritten.get();
        double secs = (System.currentTimeMillis() - startMs) / 1000.0;
        double pps = secs > 0 ? p / secs : 0;
        System.out.printf("  %d/%d games, %d positions (%.0f pos/s)%n", g, totalGames, p, pps);
    }

    /** Minimal FEN (placement + side-to-move; bullet's text parser ignores the rest). */
    private static String fen(Position pos) {
        StringBuilder sb = new StringBuilder(72);
        for (int rank = 7; rank >= 0; rank--) {
            int empty = 0;
            for (int file = 0; file < 8; file++) {
                int pc = pos.pieceAt(rank * 8 + file);
                if (pc == Piece.EMPTY) {
                    empty++;
                } else {
                    if (empty > 0) { sb.append(empty); empty = 0; }
                    sb.append(Piece.toChar(pc));
                }
            }
            if (empty > 0) sb.append(empty);
            if (rank > 0) sb.append('/');
        }
        sb.append(pos.sideToMove() == Piece.WHITE ? " w - - 0 1" : " b - - 0 1");
        return sb.toString();
    }
}
