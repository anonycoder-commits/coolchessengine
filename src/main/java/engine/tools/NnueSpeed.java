package engine.tools;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import engine.board.Position;
import engine.search.NnueEvaluator;
import engine.search.Search;

/** Fixed-depth NNUE search from startpos, reporting nodes/time/nps. Usage: NnueSpeed &lt;net&gt; &lt;depth&gt;. */
public final class NnueSpeed {
    public static void main(String[] args) throws Exception {
        NnueEvaluator.Network net;
        try (InputStream in = Files.newInputStream(Path.of(args[0]))) {
            net = NnueEvaluator.load(in);
        }
        int depth = Integer.parseInt(args[1]);
        Search search = new Search(new NnueEvaluator(net), 64);
        search.printInfo = false;
        // Warm up the JIT so the reported figure reflects compiled code, not the interpreter.
        search.search(new Position(Position.startpos()), Math.min(depth, 8));
        Search fresh = new Search(new NnueEvaluator(net), 64);
        fresh.printInfo = false;
        long t0 = System.nanoTime();
        fresh.search(Position.startpos(), depth);
        long ms = (System.nanoTime() - t0) / 1_000_000L;
        long nodes = fresh.nodes();
        System.out.printf("depth %d: %d nodes, %d ms, %.0f nps%n", depth, nodes, ms,
                ms > 0 ? nodes * 1000.0 / ms : 0);
    }
}
