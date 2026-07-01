package engine.eval;

import org.junit.jupiter.api.Test;

import engine.board.Position;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The handcrafted evaluation must be perfectly color-symmetric: evaluating a position
 * and its color-swapped, vertically mirrored twin must give exactly opposite scores.
 * This is the standard correctness check for a handcrafted eval — it catches sign and
 * white/black mix-ups in material, piece-square tables, and every positional term.
 *
 * The mirror swaps piece colors and flips the board top-to-bottom but keeps the same
 * side-to-move letter, so the (side-to-move relative) score negates exactly.
 */
class EvalSymmetryTest {

    private static final String[] FENS = {
        // startpos (fully symmetric — must evaluate to 0)
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        // Kiwipete
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
        // a few asymmetric middlegame/endgame positions, both sides to move
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4",
        "2r3k1/pp3ppp/2n1b3/q7/3P4/2P1B3/P1Q2PPP/3R2K1 w - - 0 1",
        "8/5pk1/6p1/8/8/1P6/P4PPP/6K1 b - - 0 1",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1",
    };

    /** Vertical flip + color swap, keeping the side-to-move letter unchanged. */
    private static String mirror(String fen) {
        String[] parts = fen.trim().split("\\s+");
        String[] ranks = parts[0].split("/");

        StringBuilder board = new StringBuilder();
        for (int i = ranks.length - 1; i >= 0; i--) {
            if (i != ranks.length - 1) board.append('/');
            for (char c : ranks[i].toCharArray()) {
                board.append(Character.isLetter(c) ? swapCase(c) : c);
            }
        }

        String stm = parts[1];
        String castling = parts.length > 2 ? swapCaseAll(parts[2]) : "-";
        String ep = "-";
        if (parts.length > 3 && !parts[3].equals("-")) {
            char file = parts[3].charAt(0);
            int rank = parts[3].charAt(1) - '0';
            ep = "" + file + (char) ('0' + (9 - rank));
        }
        return board + " " + stm + " " + castling + " " + ep + " 0 1";
    }

    private static char swapCase(char c) {
        return Character.isUpperCase(c) ? Character.toLowerCase(c) : Character.toUpperCase(c);
    }

    private static String swapCaseAll(String s) {
        if (s.equals("-")) return s;
        StringBuilder sb = new StringBuilder(s.length());
        for (char c : s.toCharArray()) sb.append(swapCase(c));
        return sb.toString();
    }

    @Test
    void evalIsColorSymmetric() {
        for (String fen : FENS) {
            Position pos = Position.fromFen(fen);
            Position mir = Position.fromFen(mirror(fen));
            assertEquals(Evaluator.evaluate(pos), -Evaluator.evaluate(mir),
                    "eval not antisymmetric for: " + fen);
        }
    }

    @Test
    void startposIsBalanced() {
        Position pos = Position.startpos();
        assertEquals(0, Evaluator.evaluate(pos), "startpos must evaluate to exactly 0");
    }
}
