package engine.search;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import engine.board.Move;
import engine.board.Position;

import static org.junit.jupiter.api.Assertions.fail;

/**
 * Anti-tactical-blindness regression gate. Each entry is a position with a known winning
 * move (WAC-style tactics + a few forced mates); the engine must find one of the accepted
 * moves at a fixed depth. This is deliberately a *regression* guard, not an absolute
 * strength benchmark: the seed set contains only positions the engine already solves at
 * this depth, so any future search/eval change that reintroduces tactical blindness (a
 * too-aggressive pruning rule, a broken qsearch) flips one of these red immediately.
 *
 * Fixed-depth (not node/time budget) keeps it deterministic. All failures are collected
 * into a single report so the whole suite's health is visible in one run.
 */
class TacticalSuiteTest {

    private static final int DEPTH = 10;

    // {FEN, accepted UCI move(s)}. This is a *snapshot* regression seed: every entry is a
    // position the engine currently solves decisively (a winning-material capture or a forced
    // mate) at DEPTH, with the accepted move being what it actually plays. The point is not to
    // certify these as objectively-best by an external oracle, but to lock in tactics the
    // engine already sees -- so any later pruning/eval change that makes it stop seeing one
    // flips red. WAC.001 and the two mate positions are cross-checked against known solutions;
    // the rest are engine-confirmed decisive lines. Grow this set as more tactics are verified.
    private static final String[][] POSITIONS = {
        {"2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", "g3g6"},        // WAC.001 Qg6
        {"3r3k/p4pbp/Bp1qp1p1/8/3P4/P1N2Q2/1PP2PPP/3R2K1 w - - 0 1", "f3f7"},           // Qxf7 wins (+5)
        {"6k1/6p1/p7/3Pn3/5p2/6r1/P5PP/5R1K b - - 0 1", "g3d3"},                        // Rd3 wins (+3)
        {"r1b1kb1r/3q1ppp/pBp1pn2/8/Np3P2/5B2/PPP3PP/R2Q1RK1 w kq - 0 1", "f3c6"},      // Bxc6 wins (+4)
        {"8/8/8/8/8/6k1/4Q3/6K1 w - - 0 1", "e2e4"},                                    // KQ vs K, forced mate
        {"6k1/pp4p1/2p5/2bp4/8/P5Pb/1P3rrP/2BRRN1K b - - 0 1", "g2g1"},                 // Rxg1+ forces mate
        {"2k5/8/2K5/8/8/8/8/3Q4 w - - 0 1", "d1d7"},                                    // mate in 2
        {"6k1/5ppp/8/8/8/8/8/R3K2R w KQ - 0 1", "a1a8"},                                // back-rank mate
        {"6k1/5ppp/8/8/8/8/5PPP/3QK3 w - - 0 1", "d1d8"},                               // Qd8# mate in 1
    };

    @Test
    void enginesolvesKnownTactics() {
        List<String> failures = new ArrayList<>();
        for (String[] entry : POSITIONS) {
            String fen = entry[0];
            Position pos = Position.fromFen(fen);
            // Guard against a mistranscribed (illegal) FEN where the side NOT to move is
            // already in check -- searching it lets the engine "capture" a king and crash.
            if (pos.inCheck(1 - pos.sideToMove())) {
                failures.add("  " + fen + " -> ILLEGAL (opponent already in check)");
                continue;
            }
            Search s = new Search(new HandcraftedEvaluator(), 16);
            s.printInfo = false;
            int best = s.search(pos, DEPTH);
            String bestUci = Move.toUci(best);
            boolean ok = false;
            for (int i = 1; i < entry.length; i++) {
                if (entry[i].equals(bestUci)) { ok = true; break; }
            }
            if (!ok) {
                StringBuilder accepted = new StringBuilder();
                for (int i = 1; i < entry.length; i++) {
                    if (i > 1) accepted.append('/');
                    accepted.append(entry[i]);
                }
                failures.add("  " + fen + " -> got " + bestUci + " score " + s.bestScore
                        + ", expected " + accepted);
            }
        }
        if (!failures.isEmpty()) {
            fail("Tactical suite: " + failures.size() + "/" + POSITIONS.length
                    + " positions failed at depth " + DEPTH + ":\n" + String.join("\n", failures));
        }
    }
}
