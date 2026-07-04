package engine.tools;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import engine.board.Piece;
import engine.board.Position;
import engine.eval.Evaluator;

/**
 * Texel tuner for the handcrafted evaluation's term weights.
 *
 * <p>Method: for a corpus of (position, game-result) pairs, the eval's centipawn score is mapped
 * to a predicted win probability via a logistic sigmoid, and the mean squared error against the
 * actual results is minimised by integer coordinate descent over the tunable weights.
 *
 * <p><b>Overfitting controls (added after a first naive run overfit hard -- it minimised MSE on
 * 100% of the data with no validation and produced sign-flipped, blown-out weights that lost a
 * self-play gate at -90 Elo):</b>
 * <ul>
 *   <li><b>Train/validation split by GAME</b> ({@code -valFrac}, default 0.1): whole games (not
 *       individual positions -- positions from one game share an outcome and would leak) are
 *       held out. Coordinate descent optimises the TRAIN objective; VALIDATION MSE is watched
 *       every epoch and tuning early-stops once it stops improving, keeping the best-validation
 *       weights. This detects overfitting during tuning rather than after, via self-play.
 *   <li><b>Per-game position cap</b> ({@code -maxPerGame}, default 10): a single long game can't
 *       dominate; reduces the heavy within-game correlation of per-ply corpora.
 *   <li><b>Optional L2 regularisation toward the current hand-set values</b> ({@code -lambda},
 *       default 0): a weight only moves far from its prior if the data strongly supports it,
 *       countering the correlated-feature compensation that flipped signs on solid terms.
 * </ul>
 *
 * <p>Scope: every {@code public static} non-final {@code int}/{@code int[]}/{@code int[][]}
 * field of {@link Evaluator}, discovered by reflection -- the additive term weights plus
 * material values and piece-square tables (made non-final for exactly this purpose; tuning
 * material+PST on the zurichess corpus is the configuration with a published Elo track record).
 *
 * <p>Corpus: the project's {@code positions.csv} schema (columns include {@code fen},
 * {@code white_result}, {@code game_id}); RFC4180-quoted fields handled. Uses the raw static
 * eval and skips in-check positions ({@code -skipInCheck}); the corpus is otherwise assumed
 * roughly quiet (canonical Texel uses a quiescence eval, a heavier future improvement).
 *
 * <p>Usage: {@code java -Xmx6g -cp ... engine.tools.Tune -corpus positions.csv -out tuned.txt
 * [-lambda L] [-valFrac F] [-maxPerGame N] [-patience N] [-threads N] [-max N] [-maxEpochs N]}.
 */
public final class Tune {

    private static final double LN10_OVER_400 = Math.log(10.0) / 400.0;
    private static final byte TRAIN = 0, VAL = 1;

    // --- flat compact corpus (no per-position Position objects) ---
    private long[] bb;       // N*12 piece bitboards, index(color,type) order
    private byte[] stm;      // 0 = white to move, 1 = black
    private short[] hmc;     // halfmove clock (clamped)
    private double[] target; // game result from White's perspective: 1.0 / 0.5 / 0.0
    private byte[] set;      // TRAIN or VAL, split by game so positions of a game share a set
    private int n;

    private int threads = Math.max(1, Runtime.getRuntime().availableProcessors());
    private double valFrac = 0.1;
    private int maxPerGame = 10;
    private int patience = 3;
    private double lambda = 0.0;
    private boolean skipInCheck = true;

    private ExecutorService pool;
    private Position[] posPool;

    // --- tunable parameter handles (a scalar field, or one element of an int[] / int[][] field) ---
    private static final class Param {
        final Field field;
        final int row;   // -1 => not a 2D field; else first index of an int[][] field
        final int index; // -1 => scalar field; else element index within the (row's) int[]
        final String name;
        Param(Field f, int row, int idx, String n) { field = f; this.row = row; index = idx; name = n; }
        int get() throws IllegalAccessException {
            if (row >= 0) return ((int[][]) field.get(null))[row][index];
            return index < 0 ? field.getInt(null) : ((int[]) field.get(null))[index];
        }
        void set(int v) throws IllegalAccessException {
            if (row >= 0) ((int[][]) field.get(null))[row][index] = v;
            else if (index < 0) field.setInt(null, v);
            else ((int[]) field.get(null))[index] = v;
        }
    }

    private final List<Param> params = new ArrayList<>();
    private int[] w0; // original (hand-set) parameter values, the L2 regularisation prior

    // --- CSV columns (positions.csv schema) ---
    private static final int COL_FEN = 0;
    private static final int COL_GAME_ID = 5;
    private static final int COL_WHITE_RESULT = 10;

    public static void main(String[] args) throws Exception {
        String corpus = null, out = null;
        int max = Integer.MAX_VALUE, maxEpochs = 100, threadArg = -1, maxPerGameArg = 10, patienceArg = 3;
        double valFracArg = 0.1, lambdaArg = 0.0;
        boolean skipInCheckArg = true;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-corpus": corpus = args[++i]; break;
                case "-out": out = args[++i]; break;
                case "-max": max = Integer.parseInt(args[++i]); break;
                case "-maxEpochs": maxEpochs = Integer.parseInt(args[++i]); break;
                case "-threads": threadArg = Integer.parseInt(args[++i]); break;
                case "-valFrac": valFracArg = Double.parseDouble(args[++i]); break;
                case "-maxPerGame": maxPerGameArg = Integer.parseInt(args[++i]); break;
                case "-patience": patienceArg = Integer.parseInt(args[++i]); break;
                case "-lambda": lambdaArg = Double.parseDouble(args[++i]); break;
                case "-skipInCheck": skipInCheckArg = Boolean.parseBoolean(args[++i]); break;
                default: System.err.println("Unknown arg: " + args[i]); System.exit(1); return;
            }
        }
        if (corpus == null) {
            System.out.println("Usage: Tune -corpus <file> [-out <file>] [-lambda L] [-valFrac F]"
                    + " [-maxPerGame N] [-patience N] [-threads N] [-max N] [-maxEpochs N] [-skipInCheck b]");
            System.exit(1);
            return;
        }
        Tune t = new Tune();
        if (threadArg > 0) t.threads = threadArg;
        t.valFrac = valFracArg;
        t.maxPerGame = maxPerGameArg;
        t.patience = patienceArg;
        t.lambda = lambdaArg;
        t.skipInCheck = skipInCheckArg;
        t.run(corpus, max, maxEpochs, out);
    }

    private void run(String corpus, int max, int maxEpochs, String out) throws Exception {
        discoverParams();
        System.out.printf("Tuning %d params on %d threads (valFrac=%.2f, maxPerGame=%d, lambda=%.2e, patience=%d).%n",
                params.size(), threads, valFrac, maxPerGame, lambda, patience);
        loadCorpus(corpus, max);
        int trainN = 0, valN = 0;
        for (int i = 0; i < n; i++) { if (set[i] == VAL) valN++; else trainN++; }
        System.out.println("Split: " + trainN + " train / " + valN + " validation positions.");
        if (trainN == 0 || valN == 0) { System.out.println("Empty train or validation set."); return; }

        pool = Executors.newFixedThreadPool(threads);
        posPool = new Position[threads];
        for (int i = 0; i < threads; i++) posPool[i] = Position.startpos();
        try {
            double c = fitK();
            System.out.printf("Fitted K -> c=%.6f. Initial: train MSE=%.6f, val MSE=%.6f%n",
                    c, mse(c, TRAIN), mse(c, VAL));
            coordinateDescent(c, maxEpochs);
            System.out.printf("Final: train MSE=%.6f, val MSE=%.6f%n", mse(c, TRAIN), mse(c, VAL));
            report(out);
        } finally {
            pool.shutdownNow();
        }
    }

    // --- parameter discovery ---

    private void discoverParams() {
        List<Field> fields = new ArrayList<>();
        for (Field f : Evaluator.class.getDeclaredFields()) {
            int m = f.getModifiers();
            if (!Modifier.isStatic(m) || !Modifier.isPublic(m) || Modifier.isFinal(m)) continue;
            if (f.getType() == int.class || f.getType() == int[].class
                    || f.getType() == int[][].class) fields.add(f);
        }
        fields.sort(Comparator.comparing(Field::getName));
        try {
            for (Field f : fields) {
                f.setAccessible(true);
                if (f.getType() == int.class) {
                    params.add(new Param(f, -1, -1, f.getName()));
                } else if (f.getType() == int[].class) {
                    int len = ((int[]) f.get(null)).length;
                    for (int i = 0; i < len; i++) params.add(new Param(f, -1, i, f.getName() + "[" + i + "]"));
                } else {
                    int[][] rows = (int[][]) f.get(null);
                    for (int r = 0; r < rows.length; r++) {
                        for (int i = 0; i < rows[r].length; i++) {
                            params.add(new Param(f, r, i, f.getName() + "[" + r + "][" + i + "]"));
                        }
                    }
                }
            }
            w0 = new int[params.size()];
            for (int i = 0; i < params.size(); i++) w0[i] = params.get(i).get();
        } catch (IllegalAccessException e) {
            throw new RuntimeException("param discovery failed", e);
        }
    }

    // --- corpus loading (CSV: ...,fen(0),...,game_id(5),...,white_result(10),...) ---

    private void loadCorpus(String path, int max) throws IOException {
        int lines = 0;
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            while (r.readLine() != null && lines < max) lines++;
        }
        int cap = Math.max(0, lines);
        bb = new long[cap * 12];
        stm = new byte[cap];
        hmc = new short[cap];
        target = new double[cap];
        set = new byte[cap];

        // 1 in `valEvery` games goes to validation (whole-game, deterministic by game_id hash).
        int valEvery = valFrac <= 0 ? Integer.MAX_VALUE : Math.max(2, (int) Math.round(1.0 / valFrac));

        int skippedMalformed = 0, skippedInCheck = 0, skippedCap = 0, read = 0;
        long[] scratch = new long[12];
        Position checkPos = Position.startpos();
        String curGame = null;
        int curCount = 0;
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            String line;
            boolean header = true;
            while ((line = r.readLine()) != null && read < max) {
                if (header) { header = false; continue; }
                read++;
                String[] col = splitCsv(line);
                if (col.length <= COL_WHITE_RESULT) { skippedMalformed++; continue; }
                double res = parseResult(col[COL_WHITE_RESULT]);
                if (res < 0) { skippedMalformed++; continue; }

                String gameId = col[COL_GAME_ID];
                if (!gameId.equals(curGame)) { curGame = gameId; curCount = 0; }
                curCount++;
                if (curCount > maxPerGame) { skippedCap++; continue; }

                String[] fen = col[COL_FEN].trim().split("\\s+");
                if (fen.length < 2) { skippedMalformed++; continue; }
                java.util.Arrays.fill(scratch, 0L);
                if (!parsePlacement(fen[0], scratch)) { skippedMalformed++; continue; }
                int side = fen[1].startsWith("b") ? Piece.BLACK : Piece.WHITE;
                int half = 0;
                if (fen.length > 4) { try { half = Integer.parseInt(fen[4]); } catch (NumberFormatException ignore) {} }

                if (skipInCheck) {
                    checkPos.loadForEval(scratch, side, 0);
                    if (checkPos.inCheck(side)) { skippedInCheck++; continue; }
                }

                System.arraycopy(scratch, 0, bb, n * 12, 12);
                stm[n] = (byte) side;
                hmc[n] = (short) Math.max(0, Math.min(1000, half));
                target[n] = res;
                set[n] = (Math.floorMod(gameId.hashCode(), valEvery) == 0) ? VAL : TRAIN;
                n++;
            }
        }
        System.out.println("Loaded " + n + " positions (" + skippedMalformed + " malformed, "
                + skippedInCheck + " in-check, " + skippedCap + " over per-game cap).");
    }

    /** Splits one RFC4180-ish CSV line, honouring double-quoted fields that may contain commas. */
    private static String[] splitCsv(String line) {
        List<String> out = new ArrayList<>(20);
        StringBuilder cur = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char ch = line.charAt(i);
            if (inQuotes) {
                if (ch == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') { cur.append('"'); i++; }
                    else inQuotes = false;
                } else {
                    cur.append(ch);
                }
            } else if (ch == '"') {
                inQuotes = true;
            } else if (ch == ',') {
                out.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(ch);
            }
        }
        out.add(cur.toString());
        return out.toArray(new String[0]);
    }

    /** White-perspective game result from the white_result CSV cell, or -1 if unrecognised. */
    private static double parseResult(String cell) {
        String c = cell.trim();
        if (c.equals("1") || c.equals("1.0") || c.equals("1-0")) return 1.0;
        if (c.equals("0") || c.equals("0.0") || c.equals("0-1")) return 0.0;
        if (c.equals("1/2") || c.equals("0.5") || c.equals("1/2-1/2")) return 0.5;
        return -1;
    }

    /** Fills {@code bb} (pre-zeroed) from a FEN piece-placement field. False on malformed input. */
    private static boolean parsePlacement(String placement, long[] bb) {
        int rank = 7, file = 0;
        for (int i = 0; i < placement.length(); i++) {
            char ch = placement.charAt(i);
            if (ch == '/') { rank--; file = 0; if (rank < 0) return false; continue; }
            if (ch >= '1' && ch <= '8') { file += ch - '0'; continue; }
            int pieceIdx = pieceIndex(ch);
            if (pieceIdx < 0 || file > 7 || rank < 0) return false;
            bb[pieceIdx] |= 1L << (rank * 8 + file);
            file++;
        }
        return true;
    }

    private static int pieceIndex(char ch) {
        boolean white = Character.isUpperCase(ch);
        int type;
        switch (Character.toUpperCase(ch)) {
            case 'P': type = Piece.PAWN; break;
            case 'N': type = Piece.KNIGHT; break;
            case 'B': type = Piece.BISHOP; break;
            case 'R': type = Piece.ROOK; break;
            case 'Q': type = Piece.QUEEN; break;
            case 'K': type = Piece.KING; break;
            default: return -1;
        }
        return (white ? Piece.WHITE : Piece.BLACK) * 6 + type;
    }

    // --- error / MSE (parallel, per set) ---

    /** Mean squared error over the {@code wantSet} positions for logistic constant c=K*ln10/400. */
    private double mse(double c, byte wantSet) {
        int chunk = (n + threads - 1) / threads;
        List<Future<double[]>> futures = new ArrayList<>(threads);
        for (int t = 0; t < threads; t++) {
            final int lo = t * chunk, hi = Math.min(n, (t + 1) * chunk), tid = t;
            if (lo >= hi) break;
            futures.add(pool.submit(() -> partialSq(lo, hi, c, wantSet, posPool[tid])));
        }
        double sum = 0; long cnt = 0;
        try {
            for (Future<double[]> f : futures) { double[] r = f.get(); sum += r[0]; cnt += (long) r[1]; }
        } catch (Exception e) {
            throw new RuntimeException("mse computation failed", e);
        }
        return cnt == 0 ? 0 : sum / cnt;
    }

    /** Returns {sumSquaredError, count} over positions in [lo,hi) that belong to {@code wantSet}. */
    private double[] partialSq(int lo, int hi, double c, byte wantSet, Position pos) {
        long[] scratch = new long[12];
        double sum = 0; int cnt = 0;
        for (int i = lo; i < hi; i++) {
            if (set[i] != wantSet) continue;
            System.arraycopy(bb, i * 12, scratch, 0, 12);
            pos.loadForEval(scratch, stm[i], hmc[i]);
            int e = Evaluator.evaluate(pos);
            double whiteEval = stm[i] == Piece.WHITE ? e : -e;
            double predicted = 1.0 / (1.0 + Math.exp(-c * whiteEval));
            double d = target[i] - predicted;
            sum += d * d;
            cnt++;
        }
        return new double[] {sum, cnt};
    }

    /** L2 penalty (mean squared deviation of current weights from their hand-set priors). */
    private double penalty() {
        if (lambda == 0.0) return 0.0;
        double s = 0;
        try {
            for (int i = 0; i < params.size(); i++) {
                double d = params.get(i).get() - w0[i];
                s += d * d;
            }
        } catch (IllegalAccessException e) { throw new RuntimeException(e); }
        return lambda * s / params.size();
    }

    /** Regularised TRAIN objective that coordinate descent minimises. */
    private double trainObjective(double c) { return mse(c, TRAIN) + penalty(); }

    // --- K fitting (on the training set) ---

    private double fitK() {
        double bestK = 1.0, bestErr = Double.MAX_VALUE;
        for (double k = 0.4; k <= 2.6; k += 0.1) {
            double e = mse(k * LN10_OVER_400, TRAIN);
            if (e < bestErr) { bestErr = e; bestK = k; }
        }
        for (double k = bestK - 0.1; k <= bestK + 0.1; k += 0.01) {
            double e = mse(k * LN10_OVER_400, TRAIN);
            if (e < bestErr) { bestErr = e; bestK = k; }
        }
        return bestK * LN10_OVER_400;
    }

    // --- coordinate descent with validation-based early stopping ---

    private void coordinateDescent(double c, int maxEpochs) throws IllegalAccessException {
        double bestTrain = trainObjective(c);
        double bestVal = mse(c, VAL);
        int[] bestWeights = snapshot();
        int sinceValImproved = 0;

        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            long t0 = System.nanoTime();
            int improvements = 0;
            for (Param p : params) {
                int orig = p.get();
                p.set(orig + 1);
                double up = trainObjective(c);
                if (up < bestTrain) { bestTrain = up; improvements++; continue; }
                p.set(orig - 1);
                double down = trainObjective(c);
                if (down < bestTrain) { bestTrain = down; improvements++; continue; }
                p.set(orig);
            }
            double valMse = mse(c, VAL);
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            System.out.printf("epoch %d: train=%.6f val=%.6f improvements=%d (%d ms)%n",
                    epoch, bestTrain, valMse, improvements, ms);

            if (valMse < bestVal - 1e-9) {
                bestVal = valMse;
                bestWeights = snapshot();
                sinceValImproved = 0;
            } else {
                sinceValImproved++;
            }
            if (improvements == 0) { System.out.println("Train converged."); break; }
            if (sinceValImproved >= patience) {
                System.out.println("Early stop: validation MSE stopped improving.");
                break;
            }
        }
        restore(bestWeights); // keep the best-generalising weights, not the most train-overfit
        System.out.printf("Best validation MSE=%.6f%n", bestVal);
    }

    private int[] snapshot() throws IllegalAccessException {
        int[] w = new int[params.size()];
        for (int i = 0; i < params.size(); i++) w[i] = params.get(i).get();
        return w;
    }

    private void restore(int[] w) throws IllegalAccessException {
        for (int i = 0; i < params.size(); i++) params.get(i).set(w[i]);
    }

    // --- output ---

    private void report(String out) throws Exception {
        StringBuilder sb = new StringBuilder();
        for (Param p : params) sb.append(p.name).append(" = ").append(p.get()).append('\n');
        System.out.println("--- tuned parameters (best validation) ---");
        System.out.print(sb);
        if (out != null) {
            try (PrintWriter w = new PrintWriter(out)) { w.print(sb); }
            System.out.println("Wrote tuned parameters to " + out);
        }
    }
}
