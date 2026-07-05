package engine.tools;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import engine.board.Position;
import engine.search.NnueEvaluator;

/**
 * Prints {@link NnueEvaluator} scores for positions, one per line, for cross-checking a net
 * against the Python reference and (Phase 2) bullet's own output:
 *
 * <pre>
 *   java -cp build/classes/java/main engine.tools.NnueProbe net.bin &lt; fens.txt
 *   # or: engine.tools.NnueProbe net.bin "&lt;fen1&gt;" "&lt;fen2&gt;"
 * </pre>
 *
 * With no FEN args it reads FENs from stdin (matching {@code nnue_ref.py --eval}), so a diff of
 * the two outputs is the cross-check.
 */
public final class NnueProbe {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: NnueProbe <net-file> [fen ...]   (FENs also read from stdin)");
            System.exit(2);
            return;
        }
        NnueEvaluator.Network net;
        try (InputStream in = Files.newInputStream(Path.of(args[0]))) {
            net = NnueEvaluator.load(in);
        }
        NnueEvaluator eval = new NnueEvaluator(net);

        if (args.length > 1) {
            for (int i = 1; i < args.length; i++) {
                System.out.println(eval.evaluate(Position.fromFen(args[i])));
            }
            return;
        }
        try (BufferedReader r = new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                String fen = line.trim();
                if (!fen.isEmpty()) System.out.println(eval.evaluate(Position.fromFen(fen)));
            }
        }
    }
}
