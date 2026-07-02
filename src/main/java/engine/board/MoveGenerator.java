package engine.board;

import static engine.board.Piece.*;

/** Pseudo-legal move generation plus legal filtering via make/unmake king-safety test. */
public final class MoveGenerator {
    private MoveGenerator() {}

    // Reused scratch buffer for pseudo-legal moves, one per calling thread: each
    // generateLegal* call fully consumes and finishes with its buffer (via filterLegal)
    // before returning, so within one thread no call can be re-entered while a prior
    // call's scratch contents are still in use. Thread-local rather than a single static
    // buffer because Lazy SMP now drives this class from multiple search threads at once,
    // each walking its own Position -- a shared buffer would let one thread's clear()/fill
    // race another's read, handing filterLegal a move that doesn't belong to its position.
    private static final ThreadLocal<MoveList> PSEUDO_SCRATCH = ThreadLocal.withInitial(MoveList::new);

    /** Appends fully legal moves for the side to move into {@code out}. */
    public static void generateLegal(Position pos, MoveList out) {
        MoveList scratch = PSEUDO_SCRATCH.get();
        scratch.clear();
        generatePseudoLegal(pos, scratch);
        filterLegal(pos, scratch, out);
    }

    /**
     * Appends fully legal "tactical" moves (captures, en passant, and promotions) for
     * the side to move. Used by quiescence search.
     */
    public static void generateLegalCaptures(Position pos, MoveList out) {
        MoveList scratch = PSEUDO_SCRATCH.get();
        scratch.clear();
        generateCapturesPseudo(pos, scratch);
        filterLegal(pos, scratch, out);
    }

    private static void filterLegal(Position pos, MoveList pseudo, MoveList out) {
        int us = pos.sideToMove;
        int them = 1 - us;
        for (int i = 0; i < pseudo.size; i++) {
            int move = pseudo.moves[i];
            pos.makeMove(move);
            if (!pos.isSquareAttacked(pos.kingSquare(us), them)) {
                out.add(move);
            }
            pos.unmakeMove(move);
        }
    }

    public static void generatePseudoLegal(Position pos, MoveList out) {
        generate(pos, out, false);
    }

    public static void generateCapturesPseudo(Position pos, MoveList out) {
        generate(pos, out, true);
    }

    private static void generate(Position pos, MoveList out, boolean capturesOnly) {
        int us = pos.sideToMove;
        int them = 1 - us;
        long own = pos.occByColor[us];
        long enemy = pos.occByColor[them];
        long occ = pos.occupied;
        long empty = ~occ;

        generatePawnMoves(pos, out, us, them, enemy, empty, capturesOnly);

        long knights = pos.bb[index(us, KNIGHT)];
        while (knights != 0) {
            int from = Long.numberOfTrailingZeros(knights);
            knights &= knights - 1;
            emit(out, from, Attacks.KNIGHT[from] & ~own, enemy, capturesOnly);
        }

        long bishops = pos.bb[index(us, BISHOP)];
        while (bishops != 0) {
            int from = Long.numberOfTrailingZeros(bishops);
            bishops &= bishops - 1;
            emit(out, from, Attacks.bishop(from, occ) & ~own, enemy, capturesOnly);
        }

        long rooks = pos.bb[index(us, ROOK)];
        while (rooks != 0) {
            int from = Long.numberOfTrailingZeros(rooks);
            rooks &= rooks - 1;
            emit(out, from, Attacks.rook(from, occ) & ~own, enemy, capturesOnly);
        }

        long queens = pos.bb[index(us, QUEEN)];
        while (queens != 0) {
            int from = Long.numberOfTrailingZeros(queens);
            queens &= queens - 1;
            emit(out, from, Attacks.queen(from, occ) & ~own, enemy, capturesOnly);
        }

        int kingSq = pos.kingSquare(us);
        emit(out, kingSq, Attacks.KING[kingSq] & ~own, enemy, capturesOnly);

        if (!capturesOnly) {
            generateCastling(pos, out, us, them, occ);
        }
    }

    private static void emit(MoveList out, int from, long targets, long enemy, boolean capturesOnly) {
        long caps = targets & enemy;
        while (caps != 0) {
            int to = Long.numberOfTrailingZeros(caps);
            caps &= caps - 1;
            out.add(Move.make(from, to, Move.CAPTURE));
        }
        if (capturesOnly) return;
        long quiets = targets & ~enemy;
        while (quiets != 0) {
            int to = Long.numberOfTrailingZeros(quiets);
            quiets &= quiets - 1;
            out.add(Move.make(from, to, Move.QUIET));
        }
    }

    private static void generatePawnMoves(Position pos, MoveList out, int us, int them,
                                          long enemy, long empty, boolean capturesOnly) {
        long pawns = pos.bb[index(us, PAWN)];
        if (us == WHITE) {
            long single = (pawns << 8) & empty;
            long promo = single & Attacks.RANK_8;
            // Promotions are always generated (they swing material); quiet pushes only in full gen.
            while (promo != 0) {
                int to = Long.numberOfTrailingZeros(promo);
                promo &= promo - 1;
                addPromotions(out, to - 8, to, false);
            }
            if (!capturesOnly) {
                long pushes = single & ~Attacks.RANK_8;
                while (pushes != 0) {
                    int to = Long.numberOfTrailingZeros(pushes);
                    pushes &= pushes - 1;
                    out.add(Move.make(to - 8, to, Move.QUIET));
                }
                long dbl = ((single & Attacks.RANK_3) << 8) & empty;
                while (dbl != 0) {
                    int to = Long.numberOfTrailingZeros(dbl);
                    dbl &= dbl - 1;
                    out.add(Move.make(to - 16, to, Move.DOUBLE_PUSH));
                }
            }
            long capL = (pawns << 7) & ~Attacks.FILE_H & enemy;
            long capR = (pawns << 9) & ~Attacks.FILE_A & enemy;
            emitPawnCaptures(out, capL, 7);
            emitPawnCaptures(out, capR, 9);
        } else {
            long single = (pawns >>> 8) & empty;
            long promo = single & Attacks.RANK_1;
            while (promo != 0) {
                int to = Long.numberOfTrailingZeros(promo);
                promo &= promo - 1;
                addPromotions(out, to + 8, to, false);
            }
            if (!capturesOnly) {
                long pushes = single & ~Attacks.RANK_1;
                while (pushes != 0) {
                    int to = Long.numberOfTrailingZeros(pushes);
                    pushes &= pushes - 1;
                    out.add(Move.make(to + 8, to, Move.QUIET));
                }
                long dbl = ((single & Attacks.RANK_6) >>> 8) & empty;
                while (dbl != 0) {
                    int to = Long.numberOfTrailingZeros(dbl);
                    dbl &= dbl - 1;
                    out.add(Move.make(to + 16, to, Move.DOUBLE_PUSH));
                }
            }
            long capL = (pawns >>> 9) & ~Attacks.FILE_H & enemy;
            long capR = (pawns >>> 7) & ~Attacks.FILE_A & enemy;
            emitPawnCaptures(out, capL, -9);
            emitPawnCaptures(out, capR, -7);
        }

        // En passant
        if (pos.epSquare >= 0) {
            long capturers = Attacks.PAWN[them][pos.epSquare] & pawns;
            while (capturers != 0) {
                int from = Long.numberOfTrailingZeros(capturers);
                capturers &= capturers - 1;
                out.add(Move.make(from, pos.epSquare, Move.EP_CAPTURE));
            }
        }
    }

    private static void emitPawnCaptures(MoveList out, long targets, int shift) {
        while (targets != 0) {
            int to = Long.numberOfTrailingZeros(targets);
            targets &= targets - 1;
            int from = to - shift;
            int rank = to >>> 3;
            if (rank == 0 || rank == 7) {
                addPromotions(out, from, to, true);
            } else {
                out.add(Move.make(from, to, Move.CAPTURE));
            }
        }
    }

    private static void addPromotions(MoveList out, int from, int to, boolean capture) {
        if (capture) {
            out.add(Move.make(from, to, Move.PROMO_Q_CAP));
            out.add(Move.make(from, to, Move.PROMO_R_CAP));
            out.add(Move.make(from, to, Move.PROMO_B_CAP));
            out.add(Move.make(from, to, Move.PROMO_N_CAP));
        } else {
            out.add(Move.make(from, to, Move.PROMO_Q));
            out.add(Move.make(from, to, Move.PROMO_R));
            out.add(Move.make(from, to, Move.PROMO_B));
            out.add(Move.make(from, to, Move.PROMO_N));
        }
    }

    private static void generateCastling(Position pos, MoveList out, int us, int them, long occ) {
        if (us == WHITE) {
            if ((pos.castling & CASTLE_WK) != 0
                    && (occ & ((1L << 5) | (1L << 6))) == 0
                    && !pos.isSquareAttacked(4, them)
                    && !pos.isSquareAttacked(5, them)
                    && !pos.isSquareAttacked(6, them)) {
                out.add(Move.make(4, 6, Move.KING_CASTLE));
            }
            if ((pos.castling & CASTLE_WQ) != 0
                    && (occ & ((1L << 1) | (1L << 2) | (1L << 3))) == 0
                    && !pos.isSquareAttacked(4, them)
                    && !pos.isSquareAttacked(3, them)
                    && !pos.isSquareAttacked(2, them)) {
                out.add(Move.make(4, 2, Move.QUEEN_CASTLE));
            }
        } else {
            if ((pos.castling & CASTLE_BK) != 0
                    && (occ & ((1L << 61) | (1L << 62))) == 0
                    && !pos.isSquareAttacked(60, them)
                    && !pos.isSquareAttacked(61, them)
                    && !pos.isSquareAttacked(62, them)) {
                out.add(Move.make(60, 62, Move.KING_CASTLE));
            }
            if ((pos.castling & CASTLE_BQ) != 0
                    && (occ & ((1L << 57) | (1L << 58) | (1L << 59))) == 0
                    && !pos.isSquareAttacked(60, them)
                    && !pos.isSquareAttacked(59, them)
                    && !pos.isSquareAttacked(58, them)) {
                out.add(Move.make(60, 58, Move.QUEEN_CASTLE));
            }
        }
    }
}
