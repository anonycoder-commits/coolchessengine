# Plan: Java Chess Engine — Classical Alpha-Beta with Handcrafted Evaluation

Status: pending approval (consensus reached — Architect: conditional approval, blocking items addressed below; Critic: APPROVE WITH IMPROVEMENTS, improvements merged)
Source spec: `.omc/specs/deep-interview-chess-engine.md`
Mode: RALPLAN-DR (short)

## RALPLAN-DR Summary

### Principles
1. **Correctness before strength** — move generation must be provably correct (perft-verified) before any search/eval work is trusted. Optimization work (e.g. magic bitboards) never sits on this correctness gate.
2. **Incremental complexity, with early end-to-end feedback** — single-threaded before multi-threaded, handcrafted eval before NNUE, ladder testing before fixed Elo targets, AND a thin playable engine exists as early as possible rather than only at the very end.
3. **Standard compliance over custom tooling** — UCI protocol adherence (including the `Hash` option) buys compatibility with existing GUIs, bots, and rating tools instead of building bespoke test infrastructure.
4. **Testability at every layer** — build tool + unit test framework declared up front; perft suite (with divide) for move generation; unit tests for search/eval components; automated match testing against reference engines for strength.
5. **Scope discipline** — explicit non-goals (opening book, tablebases, variants, NNUE, multi-threading, pondering, multi-PV) protect the v1 timeline; these are documented future-work, not silently dropped.

### Decision Drivers (top 3)
1. **Practical hobby-engine goal, not SOTA** — favors well-understood, well-documented techniques over maximal performance engineering; the v1 success bar (beat a weak reference-engine tier) is governed by search/eval quality, not move-generator raw speed.
2. **Correctness verifiability** — perft (with divide) against known reference positions is the standard, objective way to catch move-generation bugs; this principle should also govern *how* sliding-piece attacks are first implemented.
3. **Interoperability** — UCI compliance, including the `Hash` option the spec calls out, is required for the spec's success criteria (ladder testing against reference engines via standard GUIs/tools).

### Viable Options

**Option A: Bitboard-based board representation, magic-free attacks first, magics as a later optimization swap-in (Recommended — revised)**
- Approach: 64-bit `long` bitboards per piece type/color. Sliding-piece attacks implemented *first* via a magic-free method (classical ray loop or Kogge-Stone/hyperbola-quintessence) behind an `attacks(square, occupancy)` interface — get perft green this way. Magic bitboards (published, pre-verified magic numbers) added later behind the same interface, purely as a performance swap, validated by a differential equivalence test against the magic-free generator.
- Pros: Bitboard ceiling retained for v2 (mobility/pawn-span eval terms, future Lazy SMP); magic-table risk is fully off the v1 correctness-critical path; the equivalence test makes the later swap low-risk.
- Cons: Slightly more total code than a single sliding-attack implementation (two paths, one of which is transient); requires interface discipline to keep the swap clean.

**Option B: 0x88 / 10x12 mailbox board representation**
- Approach: Single sentinel-padded array; sliding-piece attacks are a simple offset-ray loop that stops at the off-board sentinel.
- Pros: Fastest, easiest-to-hand-verify path to perft-green; simplest make/unmake; natural square-indexed loops for handcrafted PST/king-safety eval terms.
- Cons: Lower raw NPS ceiling (irrelevant to the v1 success bar, which doesn't require bitboard-level speed); less convenient for mobility/pawn-span eval terms and any future multi-threaded search work post-v1.

**Why Option A (revised) is recommended over Option B:** the original recommendation (plain bitboards + magics from the start) was challenged hard during Architect review — magic bitboard generation/indexing is the largest correctness liability in the whole v1 design, and placing it on the Phase-1 correctness gate inverts principle #1. The revised Option A captures the steelman case for mailbox (fast, hand-verifiable path to perft-green) by using a magic-free sliding-attack method first, while still preserving the bitboard ceiling for the "complete competitive engine" identity confirmed in the interview and for any post-v1 work. This is a synthesis, not a compromise: v1 gets mailbox-level correctness confidence and bitboard-level long-term ceiling.

## Acceptance Criteria (inherited + refined from spec + consensus revisions)

- [x] Board representation uses bitboards with magic-free sliding-piece attacks for v1 (magics may be added later as a verified-equivalent optimization); move generator passes **perft to depth 5-6 on the standard test position set (startpos, Kiwipete, position 3, position 4, position 5)**, matching published correct node counts, with **perft divide available to localize any mismatch to a specific root move**
- [x] Make/unmake move is reversible and Zobrist-hash-consistent (hash after make+unmake matches pre-move hash); en passant is only hashed into the Zobrist key when an en-passant capture is actually legal in the position (classic correctness trap)
- [x] Alpha-beta search with iterative deepening, a Zobrist-keyed transposition table (storing a bound flag — EXACT/LOWER/UPPER — per entry, with mate scores stored relative to search ply, not root), basic move ordering (TT move, then MVV-LVA, then killer/history heuristics), and quiescence search is implemented and single-threaded
- [x] Search correctly detects and scores draws by threefold repetition (via a Zobrist position-history stack) and the fifty-move rule, and applies a standard mate/stalemate scoring convention at leaf nodes
- [x] Evaluation function scores material, piece-square tables (tapered by game phase), basic king safety, pawn structure, and mobility; an eval-symmetry unit test passes (`eval(position) == -eval(mirrored position)`)
- [x] Engine implements basic UCI command set — `uci`, `isready`, `ucinewgame`, `position`, `go` (with `wtime`/`btime`/`winc`/`binc`/`movetime`/`depth`), `stop`, `quit`, **and `setoption name Hash value <MB>` wired to actually (re)size the transposition table** — and responds correctly when loaded in a standard UCI GUI (e.g. Arena or CuteChess)
- [ ] A thin end-to-end UCI + depth-limited-search walking skeleton is playable in a GUI immediately after Phase 1's perft suite is green, before full search/eval features are built
- [ ] Engine consistently beats a defined "weak tier" of reference engines/bots — **concretely: ≥65-70% score over ≥100 games against a fixed weak-tier opponent (e.g. Stockfish skill level 1) at a fixed time control (e.g. 10s+0.1s increment)** — and this bar is re-applied as the ladder climbs to stronger tiers
- [ ] Build tool (Gradle or Maven) and unit test framework (JUnit 5) are set up from the start of the project, not introduced ad hoc
- [ ] No opening book, tablebase, Chess960/variant support, pondering, multi-PV, or multi-threading present in v1 (explicit non-goals)

## Implementation Steps

### Phase 0 — Project Tooling
1. Initialize the Java project with Gradle (or Maven) and JUnit 5 wired up; establish package layout (e.g. `engine.board`, `engine.search`, `engine.eval`, `engine.uci`) and a basic CI-less local test-run command (`./gradlew test`). CI itself is out of scope for v1 (noted as future work, not blocking).

### Phase 1 — Core Engine (board + move generation), correctness-gated
2. Define bitboard-based `Position` (piece bitboards per type/color, side to move, castling rights, en passant square, halfmove/fullmove counters).
3. Implement non-sliding attack generation (knight/king lookup tables) and **magic-free sliding-piece attacks** (classical ray loop or Kogge-Stone/hyperbola-quintessence) behind a stable `attacks(square, occupancy)` interface.
4. Implement pseudo-legal move generation, then legal-move filtering (king-safety check via attack detection).
5. Implement make/unmake move with full state restoration (castling rights, en passant, halfmove clock, Zobrist hash incremental update — with the en-passant-only-when-capture-legal rule applied).
6. Define the Zobrist key schedule explicitly: 12×64 piece-square keys, 1 side-to-move key, 4 castling-rights keys (or 16 combined), 8 en-passant-file keys (fixed-seed RNG for reproducibility).
7. Build a perft test harness **including perft divide** (per-root-move subtree counts); validate against the standard 5-position perft suite to depth 5-6. This phase is **blocking**: do not proceed to Phase 1b until perft is fully green.

### Phase 1b — Walking Skeleton (inserted per Architect/Critic consensus)
8. Implement a minimal UCI handler (`uci`, `isready`, `position`, `go depth N`, `bestmove`, `quit`) driving a depth-limited negamax search with a material-only evaluation. Load it in a UCI GUI (Arena/CuteChess) and confirm it plays a full legal game without protocol errors. This surfaces `Position`/`Move`/make-unmake integration issues and basic UCI/time-handling issues while the design is still cheap to change, instead of waiting until the end of the project.

### Phase 2 — Search
9. Implement minimax with alpha-beta pruning over the Phase 1 move generator (extending the Phase 1b skeleton).
10. Add iterative deepening with a concrete time-management model: derive soft/hard time limits from `wtime`/`winc` (or `movetime`/`depth`), check the clock periodically (not every node), and never start an iterative-deepening iteration that can't be allowed to finish before the hard limit; always retain a legal `bestmove` to emit on timeout or `stop`.
11. Add a Zobrist-keyed transposition table with **bound-flag storage (EXACT/LOWER/UPPER) and ply-relative mate-score storage/retrieval**; use a two-tier (depth-preferred + always-replace) scheme; size the table from the UCI `Hash` option (power-of-two sizing, `setoption name Hash` reallocates).
12. Add move ordering: TT move first, then MVV-LVA for captures, then killer moves/history heuristic for quiets.
13. Add quiescence search (captures + check evasions, with check-evasion depth deliberately bounded to avoid node explosion) to avoid horizon effects at leaf nodes.
14. Add threefold-repetition detection (Zobrist position-history stack) and fifty-move-rule draw scoring in the search; add explicit mate/stalemate scoring convention at leaf nodes (e.g. mate = -MATE_VALUE + ply, stalemate = 0).

### Phase 3 — Evaluation
15. Implement material counting with standard piece values.
16. Add piece-square tables (separate midgame/endgame tables, tapered eval by game phase).
17. Add pawn structure terms (doubled/isolated/passed pawns).
18. Add king safety (pawn shield, open files near king) and mobility terms.
19. Add an eval-symmetry unit test (`eval(position) == -eval(mirrored position)`).

### Phase 4 — Interface (full command set)
20. Extend the Phase 1b UCI handler to the full basic command set, including `setoption name Hash value <MB>` wired to transposition-table allocation/resize.
21. Wire the full UCI handler to the Phase 2/3 search+eval with proper time-control translation.
22. Manually validate by loading the engine in a standard GUI (Arena/CuteChess) and playing test games, including games with non-trivial time controls and increments.

### Phase 5 — Validation
23. Automate perft regression testing (Phase 1 suite, including divide) as part of the local test-run process.
24. Add a TT-correctness verification check: fixed-depth search with the transposition table enabled vs. disabled must return the same best move/eval (catches TT bound-flag or mate-relativization bugs).
25. Set up automated match testing against a small ladder of reference engines/bots at increasing strength (e.g. via a CLI match runner), starting with the weak tier and using the falsifiable bar defined in Acceptance Criteria (≥65-70% score over ≥100 games at a fixed time control).
26. Document the current strength tier reached and the next ladder rung as ongoing validation (no fixed final Elo target, per spec).

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Move generation bugs silently corrupt search/eval results | Perft suite (with divide) gates Phase 1 completion before Phase 1b/2 begin; treat perft mismatches as blocking; divide localizes the failing root move instead of a global node-count diff |
| Magic bitboard complexity stalls early progress or masks move-gen bugs | Magics are explicitly **not** on the Phase 1 critical path; magic-free sliding attacks reach perft-green first, magics are a later swap-in behind a stable interface, validated by a differential equivalence test against the magic-free generator before being trusted |
| Transposition table introduces silent search corruption (missing bound flags, non-relative mate scores) | Bound-flag storage and ply-relative mate-score handling are explicit Phase 2 deliverables; Phase 5 adds a dedicated TT-on vs. TT-off verification check |
| Engine misplays/loses drawn or won positions due to missing draw detection | Threefold repetition (Zobrist history stack) and fifty-move-rule detection are explicit Phase 2 deliverables, not left implicit |
| Scope creep toward deferred non-goals (NNUE, opening book, etc.) mid-build | Non-goals are explicit in the spec and this plan; any of these surfacing mid-build should be logged as future work, not implemented |
| No playable/integration signal until the very end of the project | Phase 1b inserts a thin UCI + depth-limited-search walking skeleton immediately after Phase 1 perft-green, surfacing integration and time-handling issues early instead of only in Phase 4 |
| Time-control/UCI edge cases (increment handling, very short time controls) cause time losses in real games | Concrete soft/hard time-limit model (Phase 2, step 10) derived from `wtime`/`winc`; validated manually against a GUI in Phase 1b and again in Phase 4 with non-trivial time controls before automated ladder testing |
| `Hash` UCI option silently unimplemented, failing GUI/match-runner expectations and the spec's interface acceptance criterion | `setoption name Hash` is an explicit acceptance criterion and Phase 4 deliverable, wired directly to TT allocation/resize |
| "Consistently beats weak tier" is too vague to know when v1 is done | Acceptance criteria now specify a falsifiable bar (≥65-70% score over ≥100 games at a fixed time control against a named weak-tier opponent) |

## Verification Steps

1. Run the perft test harness (with divide available) after Phase 1; all 5 standard test positions must match published node counts at depth 5-6 before proceeding to Phase 1b.
2. Run unit tests on make/unmake reversibility and Zobrist hash consistency (including the en-passant-only-when-capture-legal rule).
3. After Phase 1b, manually load the walking-skeleton engine into a UCI GUI and confirm it plays a full legal game (depth-limited, material-only eval) without protocol errors.
4. After Phase 2, run fixed-depth search on tactical test positions (e.g. a small WAC/STS subset) to sanity-check search correctness (not full strength validation yet), and confirm repetition/fifty-move positions are scored as draws.
5. After Phase 3, run the eval-symmetry unit test and confirm it passes.
6. After Phase 4, manually load the full engine into a UCI GUI and confirm it responds to all basic commands (including `setoption name Hash`) and plays full legal games at various time controls without protocol errors.
7. After Phase 5 step 24, confirm the TT-on vs. TT-off fixed-depth search agreement check passes.
8. After Phase 5, run automated matches against the weak-tier reference engine(s); confirm the ≥65-70%-over-≥100-games bar is met before considering the ladder climbed further.

## ADR

**Decision:** Build a Java chess engine using a bitboard-based board representation (magic-free sliding attacks for v1, magics as a later verified-equivalent optimization), classical single-threaded alpha-beta search with iterative deepening, a handcrafted evaluation function, and a UCI protocol implementation including the `Hash` option — with an early thin playable walking skeleton inserted right after Phase 1's correctness gate.

**Drivers:** Practical-but-strong hobby-engine goal (not SOTA) — the v1 success bar is governed by search/eval quality, not move-generator raw speed; objective correctness verification (perft with divide) as the non-negotiable gate before any search/eval work is trusted; interoperability (full UCI compliance, including `Hash`) to satisfy the spec's ladder-based success criteria using existing tools.

**Alternatives considered:** Plain bitboards with magic bitboards from the start (original Planner recommendation — rejected by Architect/Critic consensus: places the largest correctness liability in the whole design on the Phase-1 critical path, inverting the "correctness before strength" principle). 0x88/10x12 mailbox representation (fairly explored — fastest, easiest-to-hand-verify path to perft-green and natural for handcrafted eval terms, but a lower raw-NPS ceiling that's irrelevant to the v1 bar yet would constrain v2/post-v1 work). NNUE evaluation (explicitly deferred per spec — adds training-pipeline complexity not justified by the practical-hobby-engine goal). Multi-threaded search from the start (explicitly deferred per spec — adds concurrency complexity before the single-threaded engine is even correctness-verified). All UCI/interface work deferred entirely to a final phase (original Planner sequencing — revised after Architect/Critic flagged it as contradicting the "each phase ships something testable" principle and as the highest-risk, latest-tested integration point).

**Why chosen:** The revised approach (magic-free bitboards + classical alpha-beta + handcrafted eval + full UCI incl. Hash, with an early walking skeleton) captures the steelman case for mailbox's fast, verifiable path to correctness without sacrificing the bitboard ceiling the "complete competitive engine" identity calls for. It directly resolves the two blocking issues independently raised by Architect and Critic review: magic-bitboard risk on the correctness-critical path, and integration risk from deferring all UCI work to the end. The spec's explicit `Hash` option requirement, draw-detection correctness, and TT-correctness gaps are now first-class acceptance criteria rather than implicit assumptions.

**Consequences:** Phase 1 remains the largest single block of "no visible playing behavior" work, but it is now bounded by a magic-free (simpler, more hand-verifiable) sliding-attack implementation rather than a magic-bitboard subsystem, and is immediately followed by a playable walking skeleton in Phase 1b — so the correctness-before-feedback gap is now measured in one phase, not the whole project. Total scope is slightly larger than the original draft (explicit draw detection, TT bound-flag/mate-relativization handling, Hash wiring, perft divide, build tooling, walking skeleton, eval-symmetry test, TT-correctness check) but each addition closes a specific silent-correctness or spec-compliance gap identified in review, not speculative scope creep.

**Follow-ups (explicitly deferred, not forgotten):** Magic bitboards as a verified-equivalent swap-in optimization over the magic-free sliding-attack implementation; Lazy SMP multi-threaded search; NNUE evaluation; opening book; endgame tablebase probing; Chess960/variant support; pondering; multi-PV output; CI automation (noted but not blocking for a non-high-risk hobby project in short RALPLAN-DR mode).

- A Phase 5 match-runner scaffold (`engine.tools.MatchRunner`) exists — it drives our engine in-process and an opponent as a UCI subprocess over N alternating-color games with a W/L/D tally — but it has not yet been run against a real reference engine binary, so the ≥65-70%-over-≥100-games acceptance bar is still unverified.

## Changelog (Planner pass 2 — improvements applied from Architect + Critic consensus review)

- **[Architect + Critic, blocking]** Removed magic bitboards from the Phase-1 correctness-critical path; Option A revised to magic-free sliding attacks first, magics as a later verified swap-in. Decision rationale in "Viable Options" and ADR rewritten accordingly.
- **[Critic, major]** Restored the spec-mandated `Hash` UCI option to acceptance criteria, step 20 (Phase 4), and the TT-sizing risk row; wired explicitly to TT allocation/resize.
- **[Architect + Critic, major]** Added explicit draw detection (threefold repetition via Zobrist history stack, fifty-move rule) as a Phase 2 deliverable (step 14) and acceptance criterion.
- **[Architect + Critic, major]** Added explicit TT correctness contract: bound-flag storage (EXACT/LOWER/UPPER) and ply-relative mate-score storage/retrieval (step 11), plus a dedicated TT-on-vs-off verification check (step 24, verification step 7).
- **[Architect + Critic, major]** Added `perft divide` as an explicit Phase 1 deliverable (step 7) and acceptance criterion.
- **[Architect + Critic, major]** Added Phase 0 (project tooling: Gradle/Maven + JUnit 5) as an explicit, non-implicit deliverable and acceptance criterion.
- **[Architect, strongly recommended]** Inserted Phase 1b, a thin UCI + depth-limited-search walking skeleton immediately after Phase 1 perft-green, resolving the "each phase ships something testable/playable" principle contradiction the Critic also flagged.
- **[Critic, minor]** Tightened the strength acceptance criterion to a falsifiable bar (≥65-70% score over ≥100 games at a fixed time control vs. a named weak-tier opponent).
- **[Critic, minor]** Added explicit Zobrist key schedule detail (step 6) including the en-passant-only-when-capture-legal correctness trap.
- **[Critic, minor]** Added a concrete time-management model (soft/hard limits, periodic clock checks, always-retain-a-legal-bestmove) to step 10.
- **[Critic, minor]** Added an eval-symmetry unit test as an explicit Phase 3 deliverable and verification step.
- **[Critic, minor]** Stated the mate/stalemate scoring convention explicitly (step 14).
- **[Critic, minor]** Noted CI as explicit future work rather than a blocking gap, consistent with non-high-risk/short RALPLAN-DR mode.
- **[Planner]** Harmonized ADR/Decision Driver wording ("practical-but-strong hobby engine" used consistently).
