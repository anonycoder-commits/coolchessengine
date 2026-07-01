package engine.board;

import java.util.Random;

/**
 * Zobrist key schedule with a fixed seed for reproducibility:
 *   - 12 x 64 piece-square keys
 *   - 1 side-to-move key
 *   - 16 combined castling-rights keys (indexed by the 4-bit rights mask)
 *   - 8 en-passant-file keys
 */
public final class Zobrist {
    private Zobrist() {}

    public static final long[][] PIECE = new long[12][64];
    public static final long SIDE;
    public static final long[] CASTLING = new long[16];
    public static final long[] EP_FILE = new long[8];

    static {
        Random rng = new Random(0x9E3779B97F4A7C15L);
        for (int p = 0; p < 12; p++)
            for (int sq = 0; sq < 64; sq++)
                PIECE[p][sq] = rng.nextLong();
        SIDE = rng.nextLong();
        for (int i = 0; i < 16; i++) CASTLING[i] = rng.nextLong();
        for (int i = 0; i < 8; i++) EP_FILE[i] = rng.nextLong();
    }
}
