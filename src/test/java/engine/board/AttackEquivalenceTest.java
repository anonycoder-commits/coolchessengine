package engine.board;

import java.util.Random;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Verifies the three sliding-attack implementations in {@link Attacks} (classical
 * ray-scan, fancy magic bitboards, PEXT) agree on every occupancy, independent of which
 * mode {@code -Dengine.attacks=} happens to be active for this JVM. The ray-scan
 * implementation ({@code bishopRay}/{@code rookRay}) is the correctness oracle magic/PEXT
 * are built from and are checked against here.
 */
class AttackEquivalenceTest {

    /**
     * Reconstructs Attacks' relevant-occupancy mask for a square independently (standard
     * "interior squares only" magic-bitboard mask definition), without reading Attacks'
     * private mask arrays -- keeps this test's coverage independent of the production code.
     *
     * NOTE: stripping the combined board-edge bitboard (RANK_1|RANK_8|FILE_A|FILE_H) from a
     * single unified ray is only valid for bishops -- a diagonal ray is never parallel to an
     * edge, so it only ever touches each edge at its own terminal square. It is WRONG for
     * rooks: a rook standing on file a has its entire north/south ray lying on file a, so
     * stripping FILE_A from the combined ray would wrongly zero out that whole ray instead of
     * just its rank-1/8 endpoint. Rooks are therefore built per-direction with explicit
     * interior-only loop bounds (files/ranks 1..6), matching Attacks' own computeRookMask.
     */
    private static long relevantMask(int sq, boolean rook) {
        int f = sq & 7, r = sq >> 3;
        long mask = 0L;
        if (rook) {
            for (int nf = f + 1; nf <= 6; nf++) mask |= 1L << (r * 8 + nf);
            for (int nf = f - 1; nf >= 1; nf--) mask |= 1L << (r * 8 + nf);
            for (int nr = r + 1; nr <= 6; nr++) mask |= 1L << (nr * 8 + f);
            for (int nr = r - 1; nr >= 1; nr--) mask |= 1L << (nr * 8 + f);
        } else {
            for (int nf = f + 1, nr = r + 1; nf <= 6 && nr <= 6; nf++, nr++) mask |= 1L << (nr * 8 + nf);
            for (int nf = f - 1, nr = r + 1; nf >= 1 && nr <= 6; nf--, nr++) mask |= 1L << (nr * 8 + nf);
            for (int nf = f + 1, nr = r - 1; nf <= 6 && nr >= 1; nf++, nr--) mask |= 1L << (nr * 8 + nf);
            for (int nf = f - 1, nr = r - 1; nf >= 1 && nr >= 1; nf--, nr--) mask |= 1L << (nr * 8 + nf);
        }
        return mask;
    }

    @Test
    void rookExhaustiveSubsetsAgreeAcrossAllThreeImplementations() {
        int cases = 0;
        for (int sq = 0; sq < 64; sq++) {
            long mask = relevantMask(sq, true);
            long subset = 0L;
            do {
                long ray = Attacks.rookRay(sq, subset);
                long magic = Attacks.rookMagic(sq, subset);
                long pext = Attacks.rookPext(sq, subset);
                assertEquals(ray, magic, "rook magic mismatch sq=" + sq + " occ=" + Long.toHexString(subset));
                assertEquals(ray, pext, "rook pext mismatch sq=" + sq + " occ=" + Long.toHexString(subset));
                cases++;
                subset = (subset - mask) & mask;
            } while (subset != 0L);
        }
        assertEquals(102_400, cases, "expected exactly 102,400 rook relevant-occupancy subsets");
    }

    @Test
    void bishopExhaustiveSubsetsAgreeAcrossAllThreeImplementations() {
        int cases = 0;
        for (int sq = 0; sq < 64; sq++) {
            long mask = relevantMask(sq, false);
            long subset = 0L;
            do {
                long ray = Attacks.bishopRay(sq, subset);
                long magic = Attacks.bishopMagic(sq, subset);
                long pext = Attacks.bishopPext(sq, subset);
                assertEquals(ray, magic, "bishop magic mismatch sq=" + sq + " occ=" + Long.toHexString(subset));
                assertEquals(ray, pext, "bishop pext mismatch sq=" + sq + " occ=" + Long.toHexString(subset));
                cases++;
                subset = (subset - mask) & mask;
            } while (subset != 0L);
        }
        assertEquals(5_248, cases, "expected exactly 5,248 bishop relevant-occupancy subsets");
    }

    @Test
    void randomFullBoardOccupanciesAgreeAcrossAllThreeImplementations() {
        Random rnd = new Random(0xA77ACC5L);
        for (int trial = 0; trial < 10_000; trial++) {
            long occ = rnd.nextLong() & rnd.nextLong(); // sparser than uniform, closer to real positions
            for (int sq = 0; sq < 64; sq++) {
                long rookRay = Attacks.rookRay(sq, occ);
                assertEquals(rookRay, Attacks.rookMagic(sq, occ),
                        "rook magic mismatch sq=" + sq + " occ=" + Long.toHexString(occ));
                assertEquals(rookRay, Attacks.rookPext(sq, occ),
                        "rook pext mismatch sq=" + sq + " occ=" + Long.toHexString(occ));

                long bishopRay = Attacks.bishopRay(sq, occ);
                assertEquals(bishopRay, Attacks.bishopMagic(sq, occ),
                        "bishop magic mismatch sq=" + sq + " occ=" + Long.toHexString(occ));
                assertEquals(bishopRay, Attacks.bishopPext(sq, occ),
                        "bishop pext mismatch sq=" + sq + " occ=" + Long.toHexString(occ));
            }
        }
    }

    @Test
    void publicApiAgreesUnderEveryConfiguredMode() {
        // Whatever -Dengine.attacks= this JVM was launched with, the public bishop/rook/queen
        // entry points must still match the ray oracle -- this is the actual contract Search,
        // Evaluator, and MoveGenerator depend on, not the internal per-mode accessors above.
        Random rnd = new Random(0xB0A2D5L);
        for (int trial = 0; trial < 2_000; trial++) {
            long occ = rnd.nextLong() & rnd.nextLong();
            for (int sq = 0; sq < 64; sq++) {
                assertEquals(Attacks.rookRay(sq, occ), Attacks.rook(sq, occ),
                        "public rook() disagrees with oracle at sq=" + sq);
                assertEquals(Attacks.bishopRay(sq, occ), Attacks.bishop(sq, occ),
                        "public bishop() disagrees with oracle at sq=" + sq);
                assertEquals(Attacks.rookRay(sq, occ) | Attacks.bishopRay(sq, occ), Attacks.queen(sq, occ),
                        "public queen() disagrees with oracle at sq=" + sq);
            }
        }
    }
}
