# Stability fixes from live-game analysis (taunt bug, mojibake, nps under-report)

## Context

Three lichess games were analyzed (two losses vs 2516/2639 bots, one win vs 2090).
The strength-relevant findings (eval collapse in both losses) are already addressed by
the in-flight Texel material+PST sweep on GitHub Actions — no new strength work here.
But the logs exposed three concrete defects, all confirmed by code exploration with
root causes at exact lines. User approved fixing **stability only** this pass;
performance and intelligence work is recorded as backlog at the bottom.

1. **Taunt fires at random moments** (observed twice: taunted at eval +0.5 and +0.7,
   threshold is +2.5). Root cause: `lichess-bot/lib/engine_wrapper.py:272-273` appends
   the upstream sentinel `PovScore(Mate(1))` to `self.scores` whenever a search result
   has no `"score"` in info (happens on ponderhit early exits — both false taunts
   followed searches that logged no Evaluation lines). `conversation.is_winning_big()`
   reads `scores[-1]` → sees ~+100000cp → taunts.
2. **Mojibake in lichess chat**: "coolchess3000 here â€”" sent to real games. Root
   cause: `lichess-bot/lib/config.py:443` — `open(config_file)` without encoding;
   Windows decodes the UTF-8 config.yml as cp1252, mangling the em-dash greeting.
3. **UCI nps under-reports ~4x with Threads=4**: `Search.nodes()` counts only the
   master worker; LazySmpSearch helpers are never aggregated. Game logs show
   "750 Knps" when true aggregate is likely 2-3M. Misleading for any future
   perf/time-management work.

None of the three affects move selection → **no self-play gate needed**; verification
is unit tests + byte-identical Bench signature.

## Fix 1 — Taunt false-positive

Files: `lichess-bot/lib/engine_wrapper.py`, `lichess-bot/lib/conversation.py`,
`lichess-bot/test_bot/test_conversation.py`

- `engine_wrapper.py`:
  - `EngineWrapper.__init__` (~line 97): add
    `self.last_move_search_score: chess.engine.PovScore | None = None`.
  - **Top of `play_move` (~line 158)**: reset the field to `None`. Placement matters —
    this single reset covers all three non-search paths (book line 164, egtb 167,
    online 173) plus any future ones, so a stale score can never leak.
  - `search()` (~line 273): after the existing `self.scores.append(...)`, set
    `self.last_move_search_score = result.info.get("score")` (naturally `None` in the
    missing-score/ponderhit case). **Leave `self.scores` and the Mate(1) sentinel
    exactly as-is** — upstream draw/resign logic (`offer_draw_or_resign`, lines
    232-249) depends on that list's length and sentinel semantics.
- `conversation.py` `is_winning_big()` (lines 30-39): read
  `engine.last_move_search_score` instead of `engine.scores[-1]`; `None` → `False`;
  else `cp = score.relative.score(mate_score=100000)` unchanged (`.relative` POV is
  correct: search root side-to-move = us).
- `test_conversation.py`: `FakeEngine` (line 9) is documented as "only ever reads
  .scores" — it must gain the `last_move_search_score` attribute, and existing tests
  updated to set it. New cases: (a) sentinel Mate(1) in `scores` but field `None` →
  no taunt (the exact observed bug); (b) field `None` after book move → no taunt;
  (c) genuine `Mate(4)` in the field → still taunts.
- Note in a comment: homemade (`MinimalEngine`) engines never set the field → they
  simply never taunt. Acceptable.

## Fix 2 — Config encoding

File: `lichess-bot/lib/config.py` line 443 (the only `open(` in the file):
`with open(config_file) as stream:` → `with open(config_file, encoding="utf-8") as stream:`
Greetings keep the em-dash.

## Fix 3 — SMP node aggregation in UCI info

Files: `src/main/java/engine/search/Search.java`,
`src/main/java/engine/search/LazySmpSearch.java`

Design (verified against actual code by the design agent):
- `Search.java`: `nodes` is a plain `long` (line 388) — not safely readable
  cross-thread (JLS long tearing). Do NOT make it volatile (one volatile write per
  node). Instead add `private volatile long publishedNodes`, published:
  - at the **top of `checkTime()`** (line ~2207) — must be BEFORE the existing early
    return (`pondering || !useTime || currentDepth < 2`), because helpers run with
    infinite limits (`useTime` false) and would otherwise never publish;
  - once at the end of `think()`;
  - reset where `nodes = 0` happens (line 581), plus a package-private
    `resetPublishedNodes()`.
  - Add `public java.util.function.LongSupplier extraNodes` (default null). In
    `printInfo` (lines 2234-2243): `total = nodes + (extraNodes != null ?
    extraNodes.getAsLong() : 0)` for both the `nodes` field and nps.
- `LazySmpSearch.java`: after workers array is filled (~line 114-117), set
  `workers[0].extraNodes` = lambda summing helpers' `publishedNodes()`. In `think()`
  before submitting tasks: call `resetPublishedNodes()` on every helper from the
  manager thread — workers persist across moves (line 49), so without this the first
  info line of a new move includes the previous move's helper counts.
- `Search.nodes()` keeps master-only semantics: its only caller is `Bench.java:61`
  (single-threaded, `extraNodes == null`, `printInfo=false` — three independent
  reasons the bench signature cannot move).

## Verification

1. `cd lichess-bot && python -m pytest test_bot/test_conversation.py -q` (Fix 1),
   then full `python -m pytest` (Fix 2 rides along).
2. `./gradlew test --rerun` from repo root (LazySmpSearchTest, UciProtocolTest,
   EvalRegressionTest, etc.).
3. Bench signature: `java -cp build/classes/java/main engine.tools.Bench` must still
   print `bench: 910430 nodes ...` byte-identical.
4. Manual SMP check: UCI session with `setoption name Threads value 4`,
   `go movetime 2000` → reported nps should read ~3-4x the single-thread figure.
5. No self-play gate required (bot-layer + reporting-only changes).

## Perf profiling results (2026-07-05, JFR on Bench depth 15, 11.3M nodes)

Engine self-time split (898 samples): **eval ~36%** (322), search ~34% (much
irreducible — negamax driver/see/scoreMove/selectNext), Position make/unmake/query
~24% (148 make+unmake), movegen ~12%, attacks (pext) ~0.5%.

**Plan's scoped candidates are REFUTED by the data:**
- Incremental material/PST accumulators: material+PST never appears as a hotspot; the
  piece loop is cheap. Adding accumulators would put cost on the hot make/unmake path
  (~16%) to save from a cold part of eval. Net-negative. DROP.
- Lazy-eval early-out: changes eval outputs → strength-affecting, needs a gate, NOT a
  free speedup. Defer with the strength track.

**UPDATE 2026-07-05 — shared attack context was IMPLEMENTED, MEASURED SLOWER, and
REVERTED.** A `long[64]` per-thread slider-attack scratch (filled once, read by
mobility/king-danger/threats/ring-control) was fully implemented and verified
byte-identical (bench 845451). But an A/B on depth-15 bench showed it ~5% SLOWER
(~900K nps vs ~945K baseline, same node count, consistent). Reason: with pext sliders
(this engine's default), `Attacks.bishop/rook/queen(sq, occ)` is a single PEXT + hot
L1 table load -- already nearly free. The profiler's high sample counts on those terms
came from their bitcount/masking/loop work and call frequency, NOT the slider recompute.
Replacing a near-free op with 512 bytes of per-eval array write+read traffic is a net
loss. Attack-table sharing pays off with expensive magic-multiply/ray-scan slider gen,
not pext. Reverted; kept only the Bench depth-arg + JFR recipe. Lesson recorded in memory.

**Original (now-refuted) hypothesis:** shared attack
computation. Top eval hotspots are all attack-term recomputation:
`evalKingRingMobility→ringControl→attacksInto` (57), `mobilityScore` (37),
`kingDanger` (23), `evalThreats→allAttacks` (21). Each independently loops the same
pieces recomputing `Attacks.bishop/rook/queen(sq, occ)`; `occ` is constant within an
eval, so the same slider set is computed ~5×/piece (mobility, king-danger-as-enemy,
threats, own-ring, enemy-ring). Computing each piece's attack set ONCE per eval into a
shared context and having the four terms read it is provably value-identical → bench
signature must stay 845451 (the verification). This is the classic Stockfish/Ethereal
"attack context" refactor. Moderate size (touches evaluate + 4 terms), strict
correctness constraint, but the only high-value change that is also free (no gate).

Remaining lower-value backlog: staged movegen (captures-first at cut nodes; movegen is
only ~12% and MoveLists are already pooled).

### Old backlog note (superseded by the profiling results above)
- **Intelligence**: wait for the in-flight Texel sweep; if it ships, re-gate
  `useCorrectionHistory` and `useCaptureHistory` (their comments say retest after
  eval changes). Low-priority gate candidates: qsearch quiet checks at ply 1,
  SEE-based quiet pruning — each likely below the gate's ±20 Elo resolution alone.
- Search is otherwise near feature-complete (singular+multicut, probcut, full history
  family, etc. all present and gated); no new hand-crafted eval terms proposed — NNUE
  remains the long-term ceiling-raiser.

---

# ROADMAP: NNUE evaluation (researched 2026-07-05; NOT started)

## Context / why

Both recent losses to 2500+ bots were **positional/planlessness** losses, not tactical
or eval-miscalibration ones: the engine held a smooth, honest, roughly-equal eval and
was slowly outplayed in quiet maneuvering (its own PVs showed queen/knight shuffling
with no plan). This is the hand-crafted eval's structural ceiling — it has no term for
"whose position is easier to play." Neither more search work nor the in-flight Texel
weight-tune fixes it (tuning changes how much existing terms are worth; it can't add
understanding the eval has no term for). A learned NNUE eval encodes exactly that
judgment. This is the real ceiling-raiser and the engine's single largest undertaking —
a months-long, multi-phase project. The HCE stays as the fallback until an NNUE net is
gate-proven to beat it.

**Decisions locked (user, 2026-07-05):** trainer = **Bullet** (jw1912/bullet, the modern
standard; runs on any CUDA GPU) — cloud GPU preferred, local RTX 5060 Ti as zero-cost
fallback. First net = **bootstrap from existing corpora** (no new data infra up front).

## Target architecture (modern hobby-standard, researched)

**`(768 → N)×2 → 1` perspective net**, single hidden layer, no buckets, no king-mirroring
(deliberately the simplest thing that works — the forum/《Neural Networks for Chess》
consensus for a first net):
- **768 inputs** = 6 piece-types × 2 colors × 64 squares (one-hot "piece P of color C on
  square S"). NOT HalfKP/HalfKA (41k features, king-relative) — that's Stockfish-scale
  complexity we explicitly avoid first time.
- **Dual perspective accumulator**: two N-vectors, one from side-to-move's view, one from
  the other side (the non-STM half uses the vertically-flipped board + swapped colors).
  At inference concatenate **[STM half, non-STM half] → 2N → output**. This is what makes
  it understand tempo/whose-move.
- **N**: start **256** (N as low as 16 is already competitive with an HCE per the
  research; 256 is a safe, cheap first target). Scale later as data grows.
- **Activation**: Clipped ReLU, `clamp(x, 0, 1)` (quantized: clamp to [0, 255]). (SCReLU
  is a later refinement, not first-net.)
- **Quantization** (integer end-to-end — floats would drift across make/unmake): int16
  params, int32 accumulator. Accumulator weights scaled ×255, output weights ×64. Final
  cp = `(Σ + bias) × 400 / (255 × 64)`. Must stay within ±29,872 cp (the Evaluator
  contract, so mate scores never collide).
- **Big simplification of this arch vs HalfKP**: a king move is just another feature
  toggle — NO full-accumulator refresh on king moves (HalfKP's main incremental headache
  doesn't exist here).

## Phasing (risk front-loaded; each phase independently verifiable)

### Phase 0 — Trainer + bootstrap data (offline, zero engine code)
- Stand up Bullet on the chosen GPU. Convert the **existing ~1.2M quiet labeled
  positions** — Zurichess `quiet-labeled.epd` (725k) + `tools/tune_data/positions.csv`
  (~450k) — into Bullet's data format (`bulletformat`). Labels: game-result WDL (what we
  already have); blend in eval later. Reuse/adapt the existing Python converters
  (`tools/tune_data/convert_zurichess_epd.py`, `sample_positions.py`).
- Train the first `(768→256)×2→1` net; export quantized int16 weights (~a few hundred KB).
- Deliverable: a weights file. No engine change yet.

### Phase 1 — Engine-side inference, NON-incremental (correctness first)
- New `engine.search.NnueEvaluator implements Evaluator` (the interface is a single
  `int evaluate(Position pos)` — clean drop-in). Load weights from a NEW
  `src/main/resources/nnue/` via `getResourceAsStream` (repo has no resources dir yet;
  add one; bundles into the jar like every other class — the `copyToLichessBot` task then
  ships it automatically).
- **Non-incremental**: `evaluate()` does a FULL accumulator refresh from the Position
  every call (loop the ~32 pieces, sum their feature columns, CReLU, output, scale). No
  Position coupling, no accumulator stack — trivially correct. Slow but a valid MVP.
- Feature indexing MUST match existing conventions: piece = `color*6+type` (0-11), square
  a1=0; the white-perspective flip is the established `sq ^ 56` (mirror Evaluator's PST
  convention); STM-relative sign like `Evaluator` (positive = good for side to move).
- Selectable eval (UCI option or build flag) HCE ↔ NNUE; **HCE stays default** until
  gated. Per-worker instantiation mirrors `LazySmpSearch` line ~114
  (`new HandcraftedEvaluator()` per thread → `new NnueEvaluator()` per thread).
- **#1 footgun = inference-correctness**: the Java forward pass must EXACTLY reproduce the
  trainer's. Cross-check Java `evaluate()` against a reference Python forward-pass on a
  handful of FENs (scores must match within rounding) BEFORE trusting any gate. Add an
  `NnueRegressionTest` (pin outputs on the 11 EvalRegressionTest FENs) + a symmetry test.

### Phase 2 — Gate the MVP (NNUE vs HCE)
- Dispatch `self-play-gate.yml` (candidate = NNUE build, baseline = HCE build), the usual
  600→1500 games. Even a small bootstrap net should be in HCE's ballpark or beat it. If it
  DOESN'T clearly win, suspect an inference bug (feature index / quantization / perspective
  flip) before blaming the net — the cross-check in Phase 1 is what de-risks this.
- Decision: clear win → make NNUE the default eval (HCE retained as fallback/toggle).

### Phase 3 — Incremental accumulator (speed) — only after Phase 2 passes
- Efficiently-updatable: per-ply accumulator stack (mirror `killers[MAX_PLY+2]`,
  MAX_PLY=128), feature deltas hooked at Position's three mutation choke points
  (`addPiece`/`removePiece`/`movePiece`, Position.java ~171-192) covering
  capture/promotion/castling/EP/null-move. King move = plain toggle (no refresh).
- Cleanest coupling: accumulator stack lives in `NnueEvaluator`; Search notifies it
  push/pop + apply-delta at make/unmake. HCE path untouched (no-op when NNUE off).
- Verify **incremental == non-incremental** (bit-identical accumulator) as a test, then
  measure nps recovery. If plain-loop nps is inadequate, add SIMD via the Java Vector API
  (`jdk.incubator.vector`, needs `--add-modules`; JDK 25 already the toolchain).

### Phase 4 — Scale up (self-play data + bigger net) — the flywheel
- Extend `MatchRunner` to dump `(position, search-eval cp, game-result)` tuples at
  **shallow** depth/nodes (~5000 soft-node / d5–8 — research: deep d20 data trains WORSE
  from scratch). Reuse `self-play-gate.yml` sharding to generate data across free CI
  runners (CPU is fine for data-gen). Reuse the CSV pipeline + quiet-position filter.
- Retrain a larger net (N=512+, maybe SCReLU) on the bigger engine-specific dataset with
  an eval+WDL blend (lambda ≈ 0.5–1.0). Re-gate each generation. Iterate: better net →
  better self-play → better data → better net.

## Critical files
- `src/main/java/engine/search/Evaluator.java` (interface — unchanged, just implemented)
- `src/main/java/engine/search/NnueEvaluator.java` (NEW — inference)
- `src/main/resources/nnue/` (NEW — weights, `getResourceAsStream`)
- `src/main/java/engine/search/LazySmpSearch.java` (~114 — per-worker NNUE instance)
- `src/main/java/engine/board/Position.java` (~171-192 add/remove/movePiece — Phase 3 only)
- `src/main/java/engine/tools/MatchRunner.java` (Phase 4 — position/eval/result dump)
- `build.gradle` (resources packaging; `--add-modules jdk.incubator.vector` if SIMD)
- NEW `NnueRegressionTest` / `NnueSymmetryTest` (mirror EvalRegression/EvalSymmetry)
- Reused unchanged: `.github/workflows/self-play-gate.yml` (gates), `texel-tune.yml`
  (sharding pattern to copy for data-gen), `tools/tune_data/*.py` (converter starting pt)

## Honest risks
- Largest project in the engine's history; months, genuinely multi-phase.
- **nps will drop** — NNUE eval costs more per node than HCE (even incremental). Net Elo
  must win despite fewer nodes; historically HCE→NNUE is strongly +Elo (often +hundreds)
  even for hobby nets, so the trade is expected to pay, but it's the thing to watch.
- **Inference correctness** is the dominant footgun (see Phase 1 cross-check).
- SIMD may be required for viable nps (Java Vector API is still an incubator module).
- Training compute lives OUTSIDE the repo/CI (cloud or local GPU) — not gate-able in CI.

## Verification per phase
1. Phase 1: Java-vs-reference-Python forward-pass score match on sample FENs; NNUE
   regression + symmetry tests green; full `gradlew test`.
2. Phase 2: `self-play-gate.yml` NNUE-vs-HCE ≥51.5% to adopt.
3. Phase 3: incremental-vs-non-incremental accumulator bit-identical test; nps measured
   (should recover most of the Phase-1 loss); bench still runs.
4. Phase 4: each net generation re-gated vs the previous best before shipping.
