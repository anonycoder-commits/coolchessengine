package engine.tools;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import engine.board.Move;
import engine.board.MoveGenerator;
import engine.board.MoveList;
import engine.board.Piece;
import engine.board.Position;
import engine.search.HandcraftedEvaluator;
import engine.search.Search;
import engine.search.SearchLimits;

/**
 * EPD test-suite runner: scores the engine against a file of positions annotated with a
 * best-move ({@code bm}) and/or avoid-move ({@code am}), e.g. Win At Chess (WAC, tactics) or
 * the Strategic Test Suite (STS, positional judgement). Unlike a self-play gate this plays no
 * games -- each position is one deterministic search with a pass/fail verdict -- so it is a
 * cheap, reproducible "did tactical/positional vision regress" check to run BEFORE spending
 * self-play budget on a change.
 *
 * <p>EPD line = the first four FEN fields (placement, side, castling, ep) followed by
 * {@code opcode operand;} groups; only {@code bm}, {@code am} and {@code id} are read. Matching
 * is by SAN: the engine's chosen move is rendered to SAN (with standard disambiguation) and
 * compared against the annotated move(s), after stripping check/mate/annotation marks and
 * normalising castling zeros -- so {@code Qg6}, {@code Qg6+}, {@code Qg6#} all match.
 *
 * <p>Search is single-threaded (deterministic; LazySMP is not) with a fresh TT/heuristics per
 * position ({@link Search#newGame()}). Usage:
 * {@code java -cp build/classes/java/main engine.tools.Epd -file suite.epd
 * [-movetime 1000] [-depth D] [-hash 64] [-max N] [-v]}.
 */
public final class Epd {
    private Epd() {}

    private static final char[] TYPE_LETTER = {'P', 'N', 'B', 'R', 'Q', 'K'};

    public static void main(String[] args) throws IOException {
        String file = null;
        int movetime = 1000, depth = 0, hash = 64, max = Integer.MAX_VALUE;
        boolean verbose = false;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-file": file = args[++i]; break;
                case "-movetime": movetime = Integer.parseInt(args[++i]); break;
                case "-depth": depth = Integer.parseInt(args[++i]); break;
                case "-hash": hash = Integer.parseInt(args[++i]); break;
                case "-max": max = Integer.parseInt(args[++i]); break;
                case "-v": verbose = true; break;
                default: System.err.println("Unknown arg: " + args[i]); System.exit(1); return;
            }
        }
        if (file == null) {
            System.out.println("Usage: Epd -file <suite.epd> [-movetime ms] [-depth D] "
                    + "[-hash MB] [-max N] [-v]");
            System.exit(1);
            return;
        }

        SearchLimits limits = depth > 0 ? SearchLimits.depth(depth) : SearchLimits.moveTime(movetime);
        Search search = new Search(new HandcraftedEvaluator(), hash);
        search.printInfo = false;

        int solved = 0, total = 0, malformed = 0;
        long startNanos = System.nanoTime();
        try (BufferedReader r = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = r.readLine()) != null && total < max) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                Epd.Entry e = parse(line);
                if (e == null) { malformed++; continue; }
                total++;

                search.newGame();
                search.think(e.pos, limits);
                int best = search.bestMove;
                String san = best == 0 ? "----" : normalize(toSan(e.pos, best));

                boolean pass;
                if (!e.bm.isEmpty()) pass = e.bm.contains(san) && !e.am.contains(san);
                else if (!e.am.isEmpty()) pass = !e.am.contains(san);
                else pass = false; // nothing to score against
                if (pass) solved++;

                if (verbose || !pass) {
                    String expect = !e.bm.isEmpty() ? "bm " + String.join("/", e.bm)
                            : "am " + String.join("/", e.am);
                    System.out.printf("[%s] %-8s played %-7s (%s) %s%n",
                            pass ? "PASS" : "FAIL", e.id,
                            best == 0 ? "----" : toSan(e.pos, best),
                            best == 0 ? "0000" : Move.toUci(best), expect);
                }
            }
        }
        long ms = (System.nanoTime() - startNanos) / 1_000_000L;
        double pct = total > 0 ? 100.0 * solved / total : 0.0;
        String budget = depth > 0 ? ("depth " + depth) : (movetime + "ms/pos");
        System.out.printf("%nSolved %d/%d (%.1f%%) at %s%s in %d ms%n",
                solved, total, pct, budget,
                malformed > 0 ? ", " + malformed + " malformed skipped" : "", ms);
    }

    /** One parsed EPD record: the position plus its acceptable/forbidden SAN move sets. */
    private static final class Entry {
        final Position pos;
        final List<String> bm;
        final List<String> am;
        final String id;
        Entry(Position pos, List<String> bm, List<String> am, String id) {
            this.pos = pos; this.bm = bm; this.am = am; this.id = id;
        }
    }

    /** Parses one EPD line, or null if the board part is malformed. */
    private static Entry parse(String line) {
        // First four whitespace tokens are the board; the remainder is the operation list.
        int p = 0, field = 0;
        int opsStart = -1;
        while (p < line.length()) {
            while (p < line.length() && line.charAt(p) == ' ') p++;
            int s = p;
            while (p < line.length() && line.charAt(p) != ' ') p++;
            if (p > s) {
                field++;
                if (field == 4) { opsStart = p; break; }
            }
        }
        if (field < 4) return null;
        String board = line.substring(0, opsStart).trim();
        String ops = opsStart < line.length() ? line.substring(opsStart) : "";

        Position pos;
        try {
            pos = Position.fromFen(board + " 0 1"); // EPD omits the halfmove/fullmove counters
        } catch (RuntimeException ex) {
            return null;
        }
        // Reject illegal positions where the side NOT to move is in check -- searching one lets
        // the engine "capture" a king and crash (kingSquare of an empty board = 64). A real
        // suite never contains these; a hand-made typo might.
        if (pos.inCheck(1 - pos.sideToMove())) return null;

        List<String> bm = new ArrayList<>(), am = new ArrayList<>();
        String id = "";
        for (String seg : ops.split(";")) {
            seg = seg.trim();
            if (seg.isEmpty()) continue;
            int sp = seg.indexOf(' ');
            String opcode = sp < 0 ? seg : seg.substring(0, sp);
            String operand = sp < 0 ? "" : seg.substring(sp + 1).trim();
            switch (opcode) {
                case "bm": for (String m : operand.split("\\s+")) if (!m.isEmpty()) bm.add(normalize(m)); break;
                case "am": for (String m : operand.split("\\s+")) if (!m.isEmpty()) am.add(normalize(m)); break;
                case "id": id = operand.replace("\"", ""); break;
                default: break;
            }
        }
        return new Entry(pos, bm, am, id);
    }

    /** Canonical SAN for matching: drop check/mate/annotation marks and the promotion '=',
     *  and map castling zeros to O so "0-0"/"O-O" and "e8=Q"/"e8Q" compare equal. */
    static String normalize(String san) {
        StringBuilder sb = new StringBuilder(san.length());
        for (int i = 0; i < san.length(); i++) {
            char c = san.charAt(i);
            if (c == '+' || c == '#' || c == '!' || c == '?' || c == '=') continue;
            sb.append(c == '0' ? 'O' : c);
        }
        return sb.toString();
    }

    /** Renders {@code move} (legal in {@code pos}) to SAN, with standard disambiguation. Does not
     *  append check/mate marks -- {@link #normalize} strips those from the annotations too, so
     *  they never affect matching. Package-private for EpdSanTest. */
    static String toSan(Position pos, int move) {
        int flag = Move.flag(move);
        if (flag == Move.KING_CASTLE) return "O-O";
        if (flag == Move.QUEEN_CASTLE) return "O-O-O";

        int from = Move.from(move), to = Move.to(move);
        int type = Piece.type(pos.pieceAt(from));
        boolean capture = Move.isCapture(move);
        StringBuilder sb = new StringBuilder();

        if (type == Piece.PAWN) {
            if (capture) sb.append((char) ('a' + (from & 7))).append('x');
            sb.append(Move.squareName(to));
            if (Move.isPromotion(move)) sb.append('=').append(TYPE_LETTER[Move.promoType(move)]);
            return sb.toString();
        }

        sb.append(TYPE_LETTER[type]);
        sb.append(disambiguation(pos, move, from, to));
        if (capture) sb.append('x');
        sb.append(Move.squareName(to));
        return sb.toString();
    }

    /** The file/rank/both origin qualifier SAN needs when another same-type piece of the same
     *  side can also legally move to {@code to}; empty when unambiguous. */
    private static String disambiguation(Position pos, int move, int from, int to) {
        int piece = pos.pieceAt(from);
        MoveList legal = new MoveList();
        MoveGenerator.generateLegal(pos, legal);
        boolean ambiguous = false, sameFile = false, sameRank = false;
        for (int i = 0; i < legal.size; i++) {
            int m = legal.moves[i];
            if (m == move) continue;
            if (Move.to(m) != to) continue;
            int f = Move.from(m);
            if (pos.pieceAt(f) != piece) continue; // same color AND type
            ambiguous = true;
            if ((f & 7) == (from & 7)) sameFile = true;
            if ((f >>> 3) == (from >>> 3)) sameRank = true;
        }
        if (!ambiguous) return "";
        if (!sameFile) return String.valueOf((char) ('a' + (from & 7)));   // file is unique
        if (!sameRank) return String.valueOf((char) ('1' + (from >>> 3))); // file clashes, rank unique
        return "" + (char) ('a' + (from & 7)) + (char) ('1' + (from >>> 3)); // need both
    }
}
