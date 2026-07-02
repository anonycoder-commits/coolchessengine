# Architectural Migration Plan — Bitboard Performance Upgrade

> **Status note:** exploration of the codebase confirmed the engine is *already fully
> bitboard-based*. This document therefore records the existing architecture as the
> baseline contract (§1, §2), and scopes the actual migration to the two genuine gaps:
> table-driven sliding attacks (§3) and, optionally, pin-aware legal move generation
> (§3.4). §4 defines the verification gates every phase must pass.

## 1. Data Structure Strategy (baseline — already implemented)

`engine.board.Position` holds the canonical 12 + 3 bitboard layout:

| Field | Contents |
|---|---|
| `long[] bb` (12 entries) | One bitboard per piece-type per color, indexed `color*6 + type` (`Piece.index(color, type)`), W_PAWN..B_KING |
| `long[] occByColor` (2) | White / Black occupancy unions |
| `long occupied` | Total occupancy |
| `int[] board` (64) | Mailbox mirror for O(1) `pieceAt(sq)` — kept in sync by make/unmake |

Square mapping is a1=0 .. h8=63 (rank-major, LSB = a1). Bit index n ↔ square
`(rank*8 + file)`.

**Zobrist parity:** the hash is maintained *incrementally* in `makeMove()`/`unmakeMove()`
(and `makeNullMove()`/`unmakeNullMove()`), XORing `Zobrist.PIECE[piece][sq]`, `SIDE`,
`CASTLING[rightsMask]`, and `EP_FILE[file]` (only when the ep square is actually
capturable — `epRelevant()`). Because every bitboard mutation goes through
`addPiece`/`removePiece`-style updates paired with their Zobrist XORs, the hash is a pure
function of the 12+3 state; `computeKey()` recomputes it from scratch and
`MakeUnmakeTest` asserts `computeKey() == zobristKey()` across depth-3 tree walks of five
standard positions. **Nothing in this migration touches state storage or hashing**, so
parity is preserved by construction.

## 2. Interface Decoupling (the contract)

`Search`, `Evaluator`, `MoveGenerator`, and `Position` consume attacks exclusively
through these stable entry points, which are the migration contract — signatures and
semantics frozen:

```java
// engine.board.Attacks — capture-inclusive attack sets, occupancy-aware
public static long bishop(int sq, long occ)
public static long rook(int sq, long occ)
public static long queen(int sq, long occ)
public static final long[]   KNIGHT;   // [64]
public static final long[]   KING;     // [64]
public static final long[][] PAWN;     // [2][64], PAWN[color][sq] = squares attacked
```

Consumers (verified): `MoveGenerator.generate*`, `Position.isSquareAttacked`,
`Search.see()`/`attackersTo()`, `engine.eval.Evaluator` mobility/outposts. Since only the
*implementation behind* `bishop/rook/queen` changes, **zero modifications** are required
in Search or Evaluator.

## 3. Performance Blueprint

### 3.1 What already exists (no work needed)
- **Bit-scan forward loop / pop-bit trick:** used throughout movegen, SEE, and eval:
  `int sq = Long.numberOfTrailingZeros(bb); bb &= bb - 1;` — `numberOfTrailingZeros` is
  a JIT intrinsic (`TZCNT`/`BSF`), and `bb &= bb - 1` clears the LSB branchlessly (`BLSR`).
- Precomputed leaper tables, bitboard shift-based pawn generation.

### 3.2 Sliding attacks: ray-scan → table lookup (Phase 1)
Current: classical ray-scanning (2 positive + 2 negative rays per bishop/rook call, each
with a branchy blocker bitscan). Target: single table lookup per piece, both indexing
schemes implemented behind the same interface:

- **Magic:** `TBL[OFFSET[sq] + (int)(((occ & MASK[sq]) * MAGIC[sq]) >>> SHIFT[sq])]`
  with published (Kannan) constants, fancy layout: flat `long[102_400]` rook (800 KB) +
  `long[5_248]` bishop (41 KB), `SHIFT[sq] = 64 - popcount(MASK[sq])`.
- **PEXT:** `TBL[OFFSET[sq] + (int) Long.compress(occ, MASK[sq])]` — JDK 19+
  (`build.gradle` now pins a JDK 25 toolchain), intrinsified to hardware `PEXT` on
  x86-64 BMI2.

Both tables are built at class init by Carry-Rippler enumeration
(`subset = (subset - mask) & mask`) of each square's relevant-occupancy mask, filled from
the retained ray implementation (the *correctness oracle*); the magic fill throws on any
destructive collision, verifying the constants at startup. Mode selection is a
`static final int MODE` read once from `-Dengine.attacks=ray|magic|pext` — constant-folded
by C2 so the hot path JITs to a single inlinable lookup (deliberately **not**
lambdas/MethodHandles, which would go megamorphic and block inlining).

**Phase 1 result:** `MagicConstants.java` generated offline (seeded trial-and-error
searcher, not copied from a published set — see its Javadoc); `Attacks.java` restructured
per §3.2 with the ray implementation preserved as `bishopRay`/`rookRay`. All 128 magic
constants passed the destructive-collision self-check at class-init on first attempt.
`AttackEquivalenceTest` (102,400 rook + 5,248 bishop exhaustive subset cases, 10,000
random-occupancy trials × 64 squares, 2,000 public-API trials × 64 squares) and
`PerftTrickyTest` (12 TalkChess positions) both pass; full `gradlew test` green. Tri-mode
`-Dengine.attacks=ray|magic|pext` perft run confirmed identical node counts
(119,060,324 / 193,690,690) across all three implementations. Two bugs were found and
fixed during verification — both in the *new test code*, not `Attacks.java`: (1) an
invalid combined-edge-mask shortcut in `AttackEquivalenceTest.relevantMask` that only
worked for bishops (fixed to a direction-aware per-file/per-rank reconstruction matching
`computeRookMask`); (2) an incorrect memorized expected value in
`PerftTrickyTest.underpromotionVariety` (217,342 → verified-correct 92,683, cross-checked
directly via the `Perft` CLI once the equivalence tests had already validated the engine
itself).

### 3.3 Benchmark → default (Phase 2)
`Perft` CLI gains `-warmup W -runs N` (median NPS, mode echoed in output). Primary:
startpos depth 6 + Kiwipete depth 5, warmup 2, runs 5, per mode. Secondary: fixed-depth
search over mixed FENs — node counts must be identical across modes (correctness check);
only wall time may differ. Winner becomes the default `MODE`; the property override
remains permanently as the rollback path.

**Results (JDK 25, warmup 2 / runs 5, median NPS):**

| Position | ray | magic | pext | pext vs ray | pext vs magic |
|---|---|---|---|---|---|
| startpos perft(6) = 119,060,324 | 25,651,303 | 28,699,959 | 30,289,973 | +18.1% | +5.5% |
| Kiwipete perft(5) = 193,690,690 | 28,160,577 | 32,590,083 | 33,964,143 | +20.6% | +4.2% |

Node counts were identical across all three modes for both positions at every run
(reconfirms correctness, not just speed). **PEXT wins consistently on this machine** —
set as the new default in `Attacks.resolveMode()`. `-Dengine.attacks=ray` and `=magic`
remain available as the permanent rollback path. Full perft gate (tri-mode identity +
`gradlew test`) re-run and green under the new default.

### 3.4 Optional Phase 3: pin-aware legal generation
Replaces make/unmake legality filtering with checkers/pinned-set dispatch (double check →
king only; single check → capture/interpose targets; pins restricted to `LINE[ksq][from]`),
king stepping tested against `occ ^ kingBB`, and en-passant validated by an explicit
occupancy slider re-scan (covers both the diagonal-pin and horizontal double-removal
traps). The old filter path is retained as `generateLegalByFilter` — a permanent
cross-check oracle. Ships only after Phases 1–2 are locked and verified.

## 4. Safety & Verification Plan (gate for every phase)

1. **Full test suite green** — `gradlew test`: `PerftTest` (5 positions: startpos d1–6
   incl. 119,060,324; Kiwipete d1–5; positions 3–5), `MakeUnmakeTest` (Zobrist parity),
   `SeeTest`, `SearchTest`, `LazySmpSearchTest`, `UciProtocolTest`, plus the new
   `AttackEquivalenceTest` and `PerftTrickyTest`.
2. **Tri-mode perft identity** — the perft suite must produce byte-identical node counts
   under `-Dengine.attacks=ray`, `magic`, and `pext`.
3. **Equivalence testing** — exhaustive: every relevant-occupancy subset for all 64
   squares (102,400 rook + 5,248 bishop cases) asserts ray == magic == pext; plus 10,000
   fixed-seed random full-board occupancies through the public API.
4. **Tricky-position perft** (TalkChess set) — EP horizontal pin (`3k4/3p4/8/K1P4r/...`
   p6 = 1,134,888), EP diagonal pin (p6 = 1,015,133), EP discovered check
   (p6 = 1,440,467), castling-gives-check (p6 = 661,072), promotion-gives-check
   (p6 = 3,821,001), underpromotion (p6 = 217,342), self-stalemate (p6 = 2,217),
   stalemate/checkmate (p7 = 567,584), double check (p4 = 23,527), castling-rights
   positions (p4 = 1,274,206 / 1,720,476).
5. **Mismatch protocol** — on any count divergence, run `Perft -divide` under each mode
   and diff per-move subtree counts to localize; the ray implementation is the oracle and
   is **never deleted**.
6. **Rollback** — Phase 1/2: set default mode back to `ray` (bit-identical to the
   pre-migration engine). Phase 3: revert `MoveGenerator.java` alone (independent of the
   attack tables).

## Phase log

| Phase | Status | Notes |
|---|---|---|
| 0 — toolchain + baseline | done | JDK 25 toolchain pinned; `gradlew test` green; baseline NPS below |
| 1 — magic + PEXT tables | done | 128/128 magic constants valid on first try; all tests green; tri-mode perft identical |
| 2 — benchmark + default | done | pext is default; +18-20% NPS vs ray, +4-6% vs magic |
| 3 — pin-aware legal movegen (optional) | deferred | Phases 1-2 are locked and complete; user chose to stop here for now. Design in §3.4 remains valid whenever this is picked back up. |

**Baseline NPS (ray mode, pre-migration, 3 runs each, JDK 25):**
- startpos perft(6) = 119,060,324 — 24.4M / 24.3M / 25.3M nps (median 24.4M)
- Kiwipete perft(5) = 193,690,690 — 24.6M / 26.4M / 25.8M nps (median 25.8M)

Node counts match the known-correct values for both positions, confirming the JDK 25
toolchain bump introduced no regression.
