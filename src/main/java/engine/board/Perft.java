package engine.board;

import java.util.ArrayList;
import java.util.List;

/** Perft node counting and perft-divide for move-generation correctness testing. */
public final class Perft {
    private Perft() {}

    public static long perft(Position pos, int depth) {
        if (depth == 0) return 1L;
        MoveList moves = new MoveList();
        MoveGenerator.generateLegal(pos, moves);
        if (depth == 1) return moves.size;
        long nodes = 0L;
        for (int i = 0; i < moves.size; i++) {
            int move = moves.moves[i];
            pos.makeMove(move);
            nodes += perft(pos, depth - 1);
            pos.unmakeMove(move);
        }
        return nodes;
    }

    public static final class DivideEntry {
        public final String move;
        public final long nodes;
        DivideEntry(String move, long nodes) { this.move = move; this.nodes = nodes; }
    }

    /** Per-root-move subtree counts, to localize a perft mismatch to a single move. */
    public static List<DivideEntry> divide(Position pos, int depth) {
        List<DivideEntry> result = new ArrayList<>();
        MoveList moves = new MoveList();
        MoveGenerator.generateLegal(pos, moves);
        for (int i = 0; i < moves.size; i++) {
            int move = moves.moves[i];
            pos.makeMove(move);
            long n = depth <= 1 ? 1L : perft(pos, depth - 1);
            pos.unmakeMove(move);
            result.add(new DivideEntry(Move.toUci(move), n));
        }
        return result;
    }

    public static void main(String[] args) {
        String fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        int depth = 6;
        boolean divide = false;
        int warmup = 0;
        int runs = 1;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-fen": fen = args[++i]; break;
                case "-depth": depth = Integer.parseInt(args[++i]); break;
                case "-divide": divide = true; break;
                case "-warmup": warmup = Integer.parseInt(args[++i]); break;
                case "-runs": runs = Integer.parseInt(args[++i]); break;
                default: break;
            }
        }
        Position pos = Position.fromFen(fen);

        if (divide) {
            long total = 0;
            for (DivideEntry e : divide(pos, depth)) {
                System.out.println(e.move + ": " + e.nodes);
                total += e.nodes;
            }
            System.out.println();
            System.out.println("Nodes: " + total);
            return;
        }

        // Echo the active sliding-attack mode so benchmark logs are self-describing without
        // needing to separately record which -Dengine.attacks= value produced them.
        String mode = System.getProperty("engine.attacks", "magic");
        long nodes = -1;

        for (int i = 0; i < warmup; i++) {
            nodes = perft(pos, depth);
        }

        double[] npsPerRun = new double[runs];
        for (int i = 0; i < runs; i++) {
            long start = System.nanoTime();
            nodes = perft(pos, depth);
            double secs = (System.nanoTime() - start) / 1e9;
            double nps = nodes / Math.max(secs, 1e-9);
            npsPerRun[i] = nps;
            System.out.println("perft(" + depth + ") [" + mode + "] run " + (i + 1) + "/" + runs
                    + " = " + nodes + "  (" + String.format("%.2f", secs) + "s, "
                    + String.format("%.0f", nps) + " nps)");
        }

        if (runs > 1) {
            double[] sorted = npsPerRun.clone();
            java.util.Arrays.sort(sorted);
            double median = sorted.length % 2 == 1
                    ? sorted[sorted.length / 2]
                    : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2.0;
            System.out.println("[" + mode + "] median nps over " + runs + " runs: "
                    + String.format("%.0f", median));
        }
    }
}
