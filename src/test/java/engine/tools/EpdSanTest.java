package engine.tools;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Position;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Verifies Epd.toSan/normalize -- the SAN rendering the EPD runner matches annotations against.
 *  SAN disambiguation and the capture/promotion/castling/en-passant special cases are exactly
 *  where a matcher silently mis-scores, so each is pinned here. */
class EpdSanTest {

    private static final String STARTPOS =
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    /** SAN of the legal move with the given UCI string in the given position. */
    private static String san(String fen, String uci) {
        Position pos = Position.fromFen(fen);
        MoveList ml = new MoveList();
        MoveGenerator.generateLegal(pos, ml);
        for (int i = 0; i < ml.size; i++) {
            if (Move.toUci(ml.moves[i]).equals(uci)) return Epd.toSan(pos, ml.moves[i]);
        }
        throw new AssertionError("no legal move " + uci + " in " + fen);
    }

    @Test
    void pawnAndKnightBasics() {
        assertEquals("e4", san(STARTPOS, "e2e4"));
        assertEquals("Nf3", san(STARTPOS, "g1f3"));
    }

    @Test
    void disambiguationByFile() {
        // Knights on c3 and g1 both reach e2 -> file qualifier.
        String fen = "4k3/8/8/8/8/2N5/8/4K1N1 w - - 0 1";
        assertEquals("Nce2", san(fen, "c3e2"));
        assertEquals("Nge2", san(fen, "g1e2"));
    }

    @Test
    void disambiguationByRank() {
        // Rooks on a1 and a5 (same file) both reach a3 -> rank qualifier.
        String fen = "4k3/8/8/R7/8/8/8/R3K3 w - - 0 1";
        assertEquals("R1a3", san(fen, "a1a3"));
        assertEquals("R5a3", san(fen, "a5a3"));
    }

    @Test
    void captures() {
        assertEquals("exd5", san("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"));
        assertEquals("Nxe5", san("4k3/8/8/4p3/8/5N2/8/4K3 w - - 0 1", "f3e5"));
    }

    @Test
    void promotions() {
        assertEquals("e8=Q", san("6k1/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7e8q"));
        assertEquals("exd8=Q", san("3r2k1/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7d8q"));
    }

    @Test
    void castling() {
        String fen = "4k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        assertEquals("O-O", san(fen, "e1g1"));
        assertEquals("O-O-O", san(fen, "e1c1"));
    }

    @Test
    void enPassant() {
        assertEquals("exd6", san("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"));
    }

    @Test
    void normalizeStripsMarksAndZeros() {
        assertEquals("Qg6", Epd.normalize("Qg6+"));
        assertEquals("Qg6", Epd.normalize("Qg6#"));
        assertEquals("Nf3", Epd.normalize("Nf3!"));
        assertEquals("O-O", Epd.normalize("0-0"));
        assertEquals("O-O-O", Epd.normalize("0-0-0"));
        assertEquals("e8Q", Epd.normalize("e8=Q"));   // '=' dropped so e8Q and e8=Q match
    }
}
