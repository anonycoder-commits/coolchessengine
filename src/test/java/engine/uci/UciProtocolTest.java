package engine.uci;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Position;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * End-to-end protocol test: drives {@link Uci#run()} over real stdin/stdout pipes,
 * the same boundary a UCI GUI would use, rather than calling its private handlers directly.
 */
class UciProtocolTest {

    private static final long TIMEOUT_SECONDS = 10;

    private InputStream originalIn;
    private PrintStream originalOut;

    private PrintWriter engineIn;
    private BlockingQueue<String> engineOut;
    private Thread uciThread;
    private Thread readerThread;

    @BeforeEach
    void setUp() throws IOException {
        originalIn = System.in;
        originalOut = System.out;

        PipedOutputStream testToEngine = new PipedOutputStream();
        PipedInputStream engineStdin = new PipedInputStream(testToEngine);
        engineIn = new PrintWriter(new OutputStreamWriter(testToEngine), true);

        PipedOutputStream engineStdout = new PipedOutputStream();
        PipedInputStream testFromEngine = new PipedInputStream(engineStdout);

        System.setIn(engineStdin);
        System.setOut(new PrintStream(engineStdout, true));

        engineOut = new LinkedBlockingQueue<>();
        BufferedReader engineOutReader = new BufferedReader(new InputStreamReader(testFromEngine));
        readerThread = new Thread(() -> {
            try {
                String line;
                while ((line = engineOutReader.readLine()) != null) {
                    engineOut.add(line);
                }
            } catch (IOException ignored) {
                // pipe closed on teardown
            }
        }, "uci-test-reader");
        readerThread.setDaemon(true);
        readerThread.start();

        uciThread = new Thread(() -> {
            try {
                new Uci().run();
            } catch (IOException ignored) {
                // pipe closed on teardown
            }
        }, "uci-test-engine");
        uciThread.setDaemon(true);
        uciThread.start();
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        try {
            send("quit");
        } catch (Exception ignored) {
            // engine thread may already have exited
        }
        uciThread.join(2000);
        System.setIn(originalIn);
        System.setOut(originalOut);
    }

    private void send(String command) {
        engineIn.println(command);
        engineIn.flush();
    }

    private String awaitLineStartingWith(String prefix) throws InterruptedException {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(TIMEOUT_SECONDS);
        while (System.nanoTime() < deadlineNanos) {
            String line = engineOut.poll(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            assertNotNull(line, "timed out waiting for a line starting with \"" + prefix + "\"");
            if (line.startsWith(prefix)) return line;
        }
        fail("timed out waiting for a line starting with \"" + prefix + "\"");
        return null;
    }

    @Test
    void fullProtocolSequence() throws Exception {
        send("uci");
        assertTrue(awaitLineStartingWith("id name").startsWith("id name"));
        assertTrue(awaitLineStartingWith("id author").startsWith("id author"));
        awaitLineStartingWith("uciok");

        send("isready");
        awaitLineStartingWith("readyok");

        send("ucinewgame");
        send("position startpos moves e2e4 e7e5");
        send("go depth 4");

        String bestmoveLine = awaitLineStartingWith("bestmove");
        String[] parts = bestmoveLine.split("\\s+");
        // "bestmove <move>" optionally followed by "ponder <move>" (the reply hint for a
        // pondering GUI). Both forms are valid UCI.
        assertTrue(parts.length == 2 || (parts.length == 4 && parts[2].equals("ponder")),
                "expected 'bestmove <move> [ponder <move>]', got: " + bestmoveLine);
        String uciMove = parts[1];
        assertTrue(uciMove.matches("[a-h][1-8][a-h][1-8][nbrq]?"),
                "move should look like UCI long-algebraic notation: " + uciMove);
        if (parts.length == 4) {
            assertTrue(parts[3].matches("[a-h][1-8][a-h][1-8][nbrq]?"),
                    "ponder move should look like UCI long-algebraic notation: " + parts[3]);
        }

        Position pos = Position.startpos();
        pos.makeMove(findMove(pos, "e2e4"));
        pos.makeMove(findMove(pos, "e7e5"));
        assertNotEquals(0, findMove(pos, uciMove), "bestmove must be legal in the resulting position: " + uciMove);

        // Effect on TT sizing depends on a parallel change to Uci; only assert no crash/hang.
        send("setoption name Hash value 32");
        send("isready");
        awaitLineStartingWith("readyok");
    }

    private static int findMove(Position pos, String uci) {
        MoveList moves = new MoveList();
        MoveGenerator.generateLegal(pos, moves);
        for (int i = 0; i < moves.size; i++) {
            if (Move.toUci(moves.moves[i]).equals(uci)) return moves.moves[i];
        }
        return 0;
    }
}
