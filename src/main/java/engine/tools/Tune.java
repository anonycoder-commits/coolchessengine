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
 * <p>Method (Texas-Tal / "Texel" tuning): for a corpus of (position, game-result) pairs, the
 * eval's centipawn score is mapped to a predicted win probability via a logistic sigmoid, and
 * the mean squared error against the actual results is minimised by integer coordinate descent
 * over the tunable weights. It fits weights to what actually correlates with winning, which is
 * exactly the calibration the hand-set terms lack.
 *
 * <p>Scope: only the additive term weights are tuned -- every {@code public static} non-final
 * {@code int}/{@code int[]} field of {@link Evaluator}, discovered by reflection. The PeSTO
 * material values and piece-square tables (kept {@code private static final}) are already
 * tuned and left untouched, as are structural constants (divisors, phase, masks).
 *
 * <p>Corpus format: one position per line, {@code <FEN> <result>}. The result is recognised in
 * the common encodings ({@code 1-0}/{@code 0-1}/{@code 1/2-1/2}, {@code [1.0]}/{@code [0.5]}/
 * {@code [0.0]}, or a trailing {@code 1.0}/{@code 0.5}/{@code 0.0}), always from White's view.
 * Only the FEN's piece placement, side-to-move and halfmove clock are used. Positions should be
 * quiet (the corpus is assumed pre-filtered); this uses the raw static eval, not a qsearch.
 *
 * <p>Usage: {@code Tune -corpus <file> [-threads N] [-max N] [-maxEpochs N] [-out <file>]}.
 * Run with a large heap, e.g. {@code java -Xmx6g -cp ... engine.tools.Tune -corpus book.epd}.
 * This is an offline tool: it allocates and uses reflection freely.
 */
public final class Tune {

    private static final double LN10_OVER_400 = Math.log(10.0) / 400.0;

    // --- flat compact corpus (no per-position Position objects; see class doc) ---
    private long[] bb;      // N*12 piece bitboards, index(color,type) order
    private byte[] stm;     // 0 = white to move, 1 = black
    private short[] hmc;    // halfmove clock (clamped)
    private double[] target; // game result from White's perspective: 1.0 / 0.5 / 0.0
    private int n;

    private int threads = Math.max(1, Runtime.getRuntime().availableProcessors());
    private ExecutorService pool;
    private Position[] posPool;

    // --- tunable parameter handles (a scalar field, or one element of an int[] field) ---
    private static final class Param {
        final Field field;
        final int index; // -1 => scalar field; else element of an int[] field
        final String name;
        Param(Field f, int idx, String n) { field = f; index = idx; name = n; }
        int get() throws IllegalAccessException {
            return index < 0 ? field.getInt(null) : ((int[]) field.get(null))[index];
        }
        void set(int v) throws IllegalAccessException {
            if (index < 0) field.setInt(null, v); else ((int[]) field.get(null))[index] = v;
        }
    }

    private final List<Param> params = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        String corpus = null, out = null;
        int max = Integer.MAX_VALUE, maxEpochs = 100, threadArg = -1;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-corpus": corpus = args[++i]; break;
                case "-out": out = args[++i]; break;
                case "-max": max = Integer.parseInt(args[++i]); break;
                case "-maxEpochs": maxEpochs = Integer.parseInt(args[++i]); break;
                case "-threads": threadArg = Integer.parseInt(args[++i]); break;
                default: System.err.println("Unknown arg: " + args[i]); System.exit(1); return;
            }
        }
        if (corpus == null) {
            System.out.println("Usage: Tune -corpus <file> [-threads N] [-max N] [-maxEpochs N] [-out <file>]");
            System.exit(1);
            return;
        }
        Tune t = new Tune();
        if (threadArg > 0) t.threads = threadArg;
        t.run(corpus, max, maxEpochs, out);
    }

    private void run(String corpus, int max, int maxEpochs, String out) throws Exception {
        discoverParams();
        System.out.println("Tuning " + params.size() + " parameters on " + threads + " threads.");
        loadCorpus(corpus, max);
        if (n == 0) { System.out.println("No positions loaded -- check the corpus format."); return; }

        pool = Executors.newFixedThreadPool(threads);
        posPool = new Position[threads];
        for (int i = 0; i < threads; i++) posPool[i] = Position.startpos();
        try {
            double c = fitK();
            System.out.printf("Fitted K -> c=%.6f, initial MSE=%.6f%n", c, mse(c));
            coordinateDescent(c, maxEpochs);
            System.out.printf("Final MSE=%.6f%n", mse(c));
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
            if (f.getType() == int.class || f.getType() == int[].class) fields.add(f);
        }
        fields.sort(Comparator.comparing(Field::getName));
        try {
            for (Field f : fields) {
                f.setAccessible(true);
                if (f.getType() == int.class) {
                    params.add(new Param(f, -1, f.getName()));
                } else {
                    int len = ((int[]) f.get(null)).length;
                    for (int i = 0; i < len; i++) params.add(new Param(f, i, f.getName() + "[" + i + "]"));
                }
            }
        } catch (IllegalAccessException e) {
            throw new RuntimeException("param discovery failed", e);
        }
    }

    // --- corpus loading ---

    private void loadCorpus(String path, int max) throws IOException {
        int lines = 0;
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            while (r.readLine() != null && lines < max) lines++;
        }
        bb = new long[lines * 12];
        stm = new byte[lines];
        hmc = new short[lines];
        target = new double[lines];

        int skipped = 0;
        long[] scratch = new long[12];
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = r.readLine()) != null && n < lines) {
                double res = parseResult(line);
                if (res < 0) { skipped++; continue; }
                String cleaned = line.replace(';', ' ');
                String[] tok = cleaned.trim().split("\\s+");
                if (tok.length < 2) { skipped++; continue; }
                java.util.Arrays.fill(scratch, 0L);
                if (!parsePlacement(tok[0], scratch)) { skipped++; continue; }
                int side = tok[1].startsWith("b") ? Piece.BLACK : Piece.WHITE;
                int half = 0;
                if (tok.length > 4) { try { half = Integer.parseInt(tok[4]); } catch (NumberFormatException ignore) {} }
                System.arraycopy(scratch, 0, bb, n * 12, 12);
                stm[n] = (byte) side;
                hmc[n] = (short) Math.max(0, Math.min(1000, half));
                target[n] = res;
                n++;
            }
        }
        System.out.println("Loaded " + n + " positions (" + skipped + " skipped).");
    }

    /** White-perspective game result, or -1 if the line has no recognisable result marker. */
    private static double parseResult(String line) {
        if (line.contains("1/2-1/2") || line.contains("[0.5]") || line.contains(" 0.5")) return 0.5;
        if (line.contains("1-0") || line.contains("[1.0]") || line.contains(" 1.0")) return 1.0;
        if (line.contains("0-1") || line.contains("[0.0]") || line.contains(" 0.0")) return 0.0;
        return -1;
    }

    /** Fills {@code bb} (must be pre-zeroed) from a FEN piece-placement field. Returns false on
     *  a malformed placement (which the caller skips). */
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

    // --- error / MSE (parallel) ---

    /** Mean squared error over the corpus for logistic constant {@code c} (= K*ln10/400). */
    private double mse(double c) {
        int chunk = (n + threads - 1) / threads;
        List<Future<Double>> futures = new ArrayList<>(threads);
        for (int t = 0; t < threads; t++) {
            final int lo = t * chunk, hi = Math.min(n, (t + 1) * chunk), tid = t;
            if (lo >= hi) break;
            futures.add(pool.submit(() -> partialSq(lo, hi, c, posPool[tid])));
        }
        double sum = 0;
        try {
            for (Future<Double> f : futures) sum += f.get();
        } catch (Exception e) {
            throw new RuntimeException("mse computation failed", e);
        }
        return sum / n;
    }

    private double partialSq(int lo, int hi, double c, Position pos) {
        long[] scratch = new long[12];
        double sum = 0;
        for (int i = lo; i < hi; i++) {
            System.arraycopy(bb, i * 12, scratch, 0, 12);
            pos.loadForEval(scratch, stm[i], hmc[i]);
            int e = Evaluator.evaluate(pos);
            double whiteEval = stm[i] == Piece.WHITE ? e : -e;
            double predicted = 1.0 / (1.0 + Math.exp(-c * whiteEval));
            double d = target[i] - predicted;
            sum += d * d;
        }
        return sum;
    }

    // --- K fitting ---

    /** Finds the logistic constant (returned pre-folded as c = K*ln10/400) minimising MSE with
     *  the current weights, by a coarse grid then a refinement pass. */
    private double fitK() {
        double bestK = 1.0, bestErr = Double.MAX_VALUE;
        for (double k = 0.4; k <= 2.6; k += 0.1) {
            double e = mse(k * LN10_OVER_400);
            if (e < bestErr) { bestErr = e; bestK = k; }
        }
        for (double k = bestK - 0.1; k <= bestK + 0.1; k += 0.01) {
            double e = mse(k * LN10_OVER_400);
            if (e < bestErr) { bestErr = e; bestK = k; }
        }
        return bestK * LN10_OVER_400;
    }

    // --- coordinate descent ---

    private void coordinateDescent(double c, int maxEpochs) throws IllegalAccessException {
        double best = mse(c);
        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            long t0 = System.nanoTime();
            int improvements = 0;
            for (Param p : params) {
                int orig = p.get();
                p.set(orig + 1);
                double up = mse(c);
                if (up < best) { best = up; improvements++; continue; }
                p.set(orig - 1);
                double down = mse(c);
                if (down < best) { best = down; improvements++; continue; }
                p.set(orig); // neither direction helped
            }
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            System.out.printf("epoch %d: MSE=%.6f improvements=%d (%d ms)%n", epoch, best, improvements, ms);
            if (improvements == 0) { System.out.println("Converged."); break; }
        }
    }

    // --- output ---

    private void report(String out) throws Exception {
        StringBuilder sb = new StringBuilder();
        for (Param p : params) sb.append(p.name).append(" = ").append(p.get()).append('\n');
        System.out.println("--- tuned parameters ---");
        System.out.print(sb);
        if (out != null) {
            try (PrintWriter w = new PrintWriter(out)) { w.print(sb); }
            System.out.println("Wrote tuned parameters to " + out);
        }
    }
}
