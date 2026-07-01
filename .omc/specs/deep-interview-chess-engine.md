# Deep Interview Spec: Sophisticated Chess Engine

## Metadata
- Interview ID: chess-engine-2026-06-30
- Rounds: 13
- Final Ambiguity Score: 20%
- Type: greenfield
- Generated: 2026-07-01
- Threshold: 0.2
- Threshold Source: default
- Initial Context Summarized: no
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.80 | 40% | 0.32 |
| Constraint Clarity | 0.80 | 30% | 0.24 |
| Success Criteria | 0.80 | 30% | 0.24 |
| **Total Clarity** | | | **0.80** |
| **Ambiguity** | | | **0.20** |

## Topology

| Component | Status | Description | Coverage / Deferral Note |
|-----------|--------|-------------|--------------------------|
| Core Engine | active | Board representation, move generation, rules/legality | Standard chess only; correctness verified via perft test suite |
| Search | active | Classical alpha-beta with iterative deepening | Single-threaded v1, Lazy SMP multi-threading as future improvement |
| Evaluation | active | Position scoring | Handcrafted evaluation (material, PST, pawn structure, king safety, mobility) |
| Interface | active | UCI protocol | Basic UCI commands only (position, go w/ time controls, bestmove, Hash option) |

## Goal

Build a complete, focused, competitive chess engine in Java that plays standard chess via a classical alpha-beta search with a handcrafted evaluation function, exposed through the UCI protocol so it can plug into standard chess GUIs and bots. The goal is a "strong but practical hobby engine" — genuinely competitive, not chasing state-of-the-art (no NNUE, no opening book, no tablebases in v1) — built well enough to extend later if desired.

## Constraints

- Language: Java (JVM)
- Search: single-threaded alpha-beta for v1; multi-threaded (Lazy SMP) is an explicit future improvement, not required for v1
- Evaluation: handcrafted (no neural network / NNUE in v1)
- Interface: UCI protocol, basic command set only — no pondering, no multi-PV in v1
- Scope: standard chess rules only (no Chess960 or other variants)

## Non-Goals

- Opening book (deferred — engine calculates openings via search/eval)
- Endgame tablebases (deferred — endgames played out via search)
- Chess960 / other chess variants (deferred — standard chess only)
- Pondering (thinking on opponent's time) — deferred
- Multi-PV (multiple candidate lines) — deferred
- NNUE / neural network evaluation — deferred (handcrafted eval for v1)
- Multi-threaded search — deferred (single-threaded for v1, Lazy SMP later)

## Acceptance Criteria

- [ ] Move generator passes a standard perft test suite (correct leaf-node counts at fixed depths from known test positions) — proves legality/move-generation correctness independent of playing strength
- [ ] Engine implements classical alpha-beta search with iterative deepening (transposition tables and move ordering expected as standard supporting techniques)
- [ ] Evaluation function is handcrafted (material, piece-square tables, pawn structure, king safety, mobility at minimum)
- [ ] Engine is UCI-compliant for the basic command set (position, go incl. time controls, bestmove, at least a Hash size option) and works correctly when loaded into a standard UCI GUI (e.g. Arena, CuteChess)
- [ ] Engine consistently beats known weak reference engines/bots (e.g. low-skill-level Stockfish or similar), with strength validated by progressively testing against stronger engines on a ladder (no fixed final Elo target — success is continuous, measurable improvement)
- [ ] Single-threaded search is correct and stable before any multi-threading work begins

## Assumptions Exposed & Resolved

| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| "Sophisticated" implies maximum strength / SOTA techniques | Asked directly what "sophisticated" means as a goal | Resolved to "strong but practical hobby engine" — not chasing SOTA |
| Top performance requires a systems language (C++/Rust) | Contrarian: challenged whether "practical hobby" goal actually needs that performance ceiling | Resolved to Java — practical goal doesn't require C++/Rust-level performance |
| Strong evaluation requires NNUE | Simplifier: asked for the simplest eval approach that still meets the success bar, given classical search + JVM choice | Resolved to handcrafted evaluation; NNUE explicitly deferred |
| Project identity unclear after many feature decisions made independently | Ontologist: stepped back to ask what the project fundamentally IS | Resolved to "complete, focused competitive engine" rather than an open-ended extensible platform |
| Advanced UCI features (pondering, multi-PV) status unclear after partial non-goal selection | Follow-up confirmation on the one item left unselected in the non-goals round | Resolved to basic UCI only for v1; pondering/multi-PV deferred |

## Technical Context

Greenfield project — no existing code. Technology choices made during interview:
- Language: Java
- Search algorithm family: classical alpha-beta (not MCTS, not pure NN-guided)
- Concurrency: single-threaded first; Lazy SMP-style parallel search is a planned future iteration, not a v1 requirement
- Evaluation: handcrafted/classical (not learned)
- Protocol: UCI (Universal Chess Interface), basic command subset

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| Engine | core domain | language (Java), components (Core Engine, Search, Evaluation, Interface) | Engine has-a Board; Engine has-a SearchAlgorithm; Engine has-a UCIProtocol interface |
| Board | core domain | position state, legality rules | Board has-many Move (legal moves); Board represents Position |
| Move | core domain | from/to squares, special-move flags (castling, en passant, promotion) | Move applied-to Board produces new Position |
| Position | core domain | board state, side to move, castling rights, en passant target | Position validated via perft test suite |
| SearchAlgorithm | core domain | alpha-beta, iterative deepening, transposition tables, move ordering, single-threaded (v1) | SearchAlgorithm explores Move tree from Position; SearchAlgorithm uses HandcraftedEval |
| HandcraftedEval | core domain | material, piece-square tables, pawn structure, king safety, mobility | HandcraftedEval scores Position for SearchAlgorithm |
| UCIProtocol | supporting | basic commands (position, go, bestmove, Hash option) | UCIProtocol exposes Engine to external GUIs/bots |
| Language(Java) | external system / constraint | JVM platform | Constrains implementation of all other entities |
| ReferenceEngineBenchmark | supporting | ladder of opponent engines/bots of increasing strength | Used to validate Engine strength (Success Criteria) |

## Ontology Convergence

| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 1 | 4 | 4 | - | - | N/A |
| 2 | 5 | 1 | 0 | 4 | 80% |
| 3 | 6 | 1 | 0 | 5 | 83% |
| 4 | 7 | 1 | 0 | 6 | 86% |
| 5 | 8 | 1 | 0 | 7 | 88% |
| 6 | 9 | 1 | 0 | 8 | 89% |
| 7 | 9 | 0 | 1 (Language: Go/JVM → Java) | 8 | 100% |
| 8-13 | 9 | 0 | 0 | 9 | 100% |

Entities fully converged by round 7 and remained stable through round 13 — the domain model (Engine, Board, Move, Position, SearchAlgorithm, HandcraftedEval, UCIProtocol, Language, ReferenceEngineBenchmark) is stable.

## Interview Transcript
<details>
<summary>Full Q&A (13 rounds)</summary>

### Round 0 (Topology Gate)
**Q:** Confirm 4-component topology: Core Engine, Search, Evaluation, Interface
**A:** Looks right

### Round 1
**Q:** What's the primary goal driving this project? (Maximize strength / Learning project / Strong practical hobby engine / Other)
**A:** Strong but practical hobby engine
**Ambiguity:** 86%

### Round 2
**Q:** What kind of search approach? (Classical alpha-beta / Neural-guided MCTS / Hybrid NNUE+alpha-beta / Not sure)
**A:** Classical alpha-beta
**Ambiguity:** 86%

### Round 3
**Q:** How do you want to interact with/deploy the engine? (UCI / CLI / Both / Other)
**A:** UCI protocol
**Ambiguity:** 80%

### Round 4 (Contrarian Mode)
**Q:** Challenged the systems-language assumption — what language/platform? (C++/Rust / Python / Go or Java/Kotlin / Other)
**A:** Go or Java/Kotlin
**Ambiguity:** 71%

### Round 5
**Q:** How will you know the engine is succeeding? (Beat reference engines / Target Elo / Correctness + beats me / Not sure)
**A:** Beat reference engines/bots
**Ambiguity:** 59%

### Round 6 (Simplifier Mode)
**Q:** What's the simplest evaluation approach that still meets the bar? (Handcrafted / NNUE / Start handcrafted then NNUE / Not sure)
**A:** Handcrafted evaluation
**Ambiguity:** 55%

### Round 7
**Q:** Final language choice: Go, Kotlin, or Java?
**A:** Java
**Ambiguity:** 49%

### Round 8 (Ontologist Mode)
**Q:** What IS this project fundamentally — complete competitive engine or extensible learning platform?
**A:** Complete, focused competitive engine
**Ambiguity:** 39%

### Round 9
**Q:** What's the concrete success bar/test protocol? (Beat weak engines then climb / Specific Elo / Beat Stockfish at limited level / No fixed protocol)
**A:** Beat known weak engines, then climb
**Ambiguity:** 33%

### Round 10
**Q:** Single-threaded or multi-threaded search? (Single-threaded first, multi later / Multi from start / Single only)
**A:** Single-threaded first, multi-threaded later
**Ambiguity:** 28.5%

### Round 11
**Q:** Which features should be explicit non-goals for v1? (Opening book / Tablebases / Chess960 / Pondering+multi-PV — multiSelect)
**A:** Opening book, Endgame tablebases, Chess960/variants (pondering/multi-PV left unselected)
**Ambiguity:** 25%

### Round 12
**Q:** How do you want to verify move-generation correctness? (Perft suite / Informal play testing / Perft + tactical suites)
**A:** Perft test suite
**Ambiguity:** 22%

### Round 13
**Q:** Should pondering/multi-PV be in v1, given it was left unselected as a non-goal? (Basic UCI only / Include pondering / Include multi-PV)
**A:** Basic UCI only for v1
**Ambiguity:** 20%

</details>
