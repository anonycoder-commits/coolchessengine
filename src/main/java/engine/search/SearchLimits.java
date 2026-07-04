package engine.search;

/** Search constraints parsed from a UCI {@code go} command. */
public final class SearchLimits {
    public int depth = 0;       // fixed depth (0 = unlimited / use time)
    public int movetime = 0;    // fixed ms per move (0 = none)
    public int wtime = 0;
    public int btime = 0;
    public int winc = 0;
    public int binc = 0;
    public int movestogo = 0;   // 0 = unknown (sudden death / increment)
    public boolean infinite = false;

    public static SearchLimits depth(int d) {
        SearchLimits l = new SearchLimits();
        l.depth = d;
        return l;
    }

    public static SearchLimits moveTime(int ms) {
        SearchLimits l = new SearchLimits();
        l.movetime = ms;
        return l;
    }

    /** Clock-based control: remaining time and increment per side, in milliseconds. */
    public static SearchLimits clock(int wtime, int btime, int winc, int binc) {
        SearchLimits l = new SearchLimits();
        l.wtime = wtime;
        l.btime = btime;
        l.winc = winc;
        l.binc = binc;
        return l;
    }

    public boolean hasClock() {
        return wtime > 0 || btime > 0 || movetime > 0;
    }
}
