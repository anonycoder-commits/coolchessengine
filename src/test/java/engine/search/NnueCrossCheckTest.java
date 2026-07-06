package engine.search;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import engine.board.Position;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * The dominant NNUE footgun is an inference mismatch between the Java forward pass and the
 * trainer's: a wrong feature index, quantization step, or perspective flip silently produces a
 * garbage net. This test pins {@link NnueEvaluator} against a golden set of evals produced by
 * the independent Python reference ({@code tools/nnue/nnue_ref.py}) on a committed fixture net.
 * Java == reference is required, exactly (both use integer arithmetic).
 *
 * <p>The fixture is a small seeded random net -- the values are meaningless, but the arithmetic
 * and indexing exercised are identical to a real net, so any transcription bug surfaces here.
 * The final authority (this reference == bullet's own output) is checked on a real net in
 * Phase 2; this test locks Java to the reference.
 */
class NnueCrossCheckTest {

    @Test
    void javaMatchesPythonReference() throws Exception {
        crossCheck("/nnue/fixture_net.bin", "/nnue/fixture_golden.txt", 1);
    }

    /** Same pin for the material-output-bucketed format (bullet MaterialCount&lt;8&gt;): the
     *  fixture FENs span 2..32 pieces, so bucket selection and the bucket-major output-weight
     *  layout are both exercised, not just bucket 0. */
    @Test
    void javaMatchesPythonReferenceWithOutputBuckets() throws Exception {
        crossCheck("/nnue/fixture_net_b8.bin", "/nnue/fixture_golden_b8.txt", 8);
    }

    private void crossCheck(String netResource, String goldenResource, int expectedBuckets)
            throws Exception {
        NnueEvaluator.Network net;
        try (InputStream in = getClass().getResourceAsStream(netResource)) {
            assertNotNull(in, netResource + " missing from test resources");
            net = NnueEvaluator.load(in);
        }
        assertEquals(expectedBuckets, net.buckets, "bucket count mis-detected from file size");
        NnueEvaluator eval = new NnueEvaluator(net);

        List<String> fens = new ArrayList<>();
        List<Integer> expected = new ArrayList<>();
        try (InputStream in = getClass().getResourceAsStream(goldenResource);
             BufferedReader r = new BufferedReader(new InputStreamReader(
                     assertNotNull(in, goldenResource + " missing"), StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.isBlank()) continue;
                int tab = line.indexOf('\t');
                expected.add(Integer.parseInt(line.substring(0, tab).trim()));
                fens.add(line.substring(tab + 1).trim());
            }
        }
        assertEquals(12, fens.size(), "unexpected fixture size");

        for (int i = 0; i < fens.size(); i++) {
            int got = eval.evaluate(Position.fromFen(fens.get(i)));
            assertEquals(expected.get(i), got,
                    "Java NNUE eval != Python reference for: " + fens.get(i));
        }
    }

    /** Tiny helper so the resource null-check reads inline in the try-with-resources. */
    private static InputStream assertNotNull(InputStream in, String msg) {
        org.junit.jupiter.api.Assertions.assertNotNull(in, msg);
        return in;
    }
}
