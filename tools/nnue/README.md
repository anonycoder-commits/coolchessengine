# NNUE tooling (Phase 0 — bootstrap data)

Follows `nnueroadmap.md`. This directory holds the offline, zero-engine-code Phase 0:
turn the existing labeled corpora into a training set and train the first net.

Target first net: **`(768 → 256)×2 → 1`** perspective net, SCReLU, int16 quant — matches
bullet's stock `examples/simple.rs` (see the roadmap for the full architecture rationale).

## 1. Build the training text (done in-repo, CPU, fast)

```
python build_bootstrap_data.py \
    --in ../tune_data/positions.csv \
    --in ../tune_data/positions_zurichess.csv \
    --out data/bootstrap_wdl.txt
```

Produces `data/bootstrap_wdl.txt` (gitignored): 1,125,000 lines in bullet's text format
`<FEN> | <score> | <result>`. WDL-only bootstrap, so `score` is fixed at `0` and `result`
is the white-relative game outcome (`1.0`/`0.5`/`0.0`). WDL split ≈ 41.5% / 19.6% / 38.9%.

## 2. Pack text → binary (on the training box, in a `bullet` checkout)

Bullet ships its own tested converter — we deliberately do **not** hand-emit the 32-byte
`bulletformat::ChessBoard` binary (a wrong byte silently corrupts training). Flags vary by
bullet version, so confirm with `--help` on your checkout:

```
cargo run --release --bin convert -- --input path/to/bootstrap_wdl.txt --output bootstrap_wdl.bin
```

The binary is 32 bytes/position (~36 MB here). Optionally shuffle with bullet's shuffle
utility before training.

## 3. Train (local RTX 5060 Ti, minutes)

Blackwell needs CUDA 12.8+. Start from bullet's stock `examples/simple.rs` — it is already
this exact net (`(768->HIDDEN)x2->1`, `Chess768`, dual perspective, SCReLU, QA=255/QB=64/
SCALE=400). The engine mirrors it, so the only edits needed are:

- `HIDDEN_SIZE = 256` (multiple of 32; the example ships 128)
- `wdl::ConstantWDL { value: 1.0 }` — **pure WDL**, because our text data has score=0 (game
  result is the only label); the stock 0.75 would blend in a meaningless sigmoid(0) target
- data loader → `DirectSequentialDataLoader::new(&["data/bootstrap.data"])`
- leave activation (`screlu`), QA, QB, SCALE, and the save format as-is — the engine matches them

Run with `cargo run --release --example simple`; the quantised `.bin` in `checkpoints/` is the
net. Deliverable of Phase 0 is that file; it drops straight into the engine via `EvalFile`.
(Note: our dataset is small ~1.1M positions, so consider a smaller `end_superbatch` to limit
overfitting — but the defaults will produce a gate-able first net.)

## Phase 1 — engine-side inference (non-incremental), correctness-first

`engine.search.NnueEvaluator` implements the `Evaluator` interface with a full accumulator
refresh every call. Architecture `(768 -> N)x2 -> 1`, **SCReLU**, int16 quant. It mirrors
bullet's stock `examples/simple.rs` exactly (same feature set, activation, quant, and staged
inference arithmetic), so a net trained by that example **drops in with no conversion**. HCE
stays the default eval.

### Weights file format = bullet's raw `SavedFormat` dump (little-endian i16, no header)

| i16 count | field | notes |
|-----------|-------|-------|
| 768·N | ftWeights (`l0w`) | feature-major `[feature*N + i]`, scale QA=255 |
| N | ftBias (`l0b`) | scale QA |
| B·2·N | outWeights (`l1w`) | **bucket-major** (bullet saves `l1w` transposed when B>1); within a bucket STM half first, then non-STM, scale QB=64 |
| B | outBias (`l1b`) | scale QA·QB |

`N` (hidden) and `B` (material output buckets: 1 = original single-output net, 8 = bullet
`MaterialCount<8>`) are both inferred from the file length (`shorts = N·(769+2B) + B`);
QA/QB/SCALE=400 are fixed to bullet's defaults. `N` must be a multiple of 32 (256 works) so
bullet's `align(64)` accumulator layout is padding-free. Bucket selection (matching bullet):
`(piece_count - 2) / ceil(32/B)` — with B=8 each bucket covers 4 piece-counts. Inference
(matching bullet `Network::evaluate`): SCReLU `clamp(acc,0,QA)^2`, then staged
`out /= QA; out += bias[bucket]; out *= SCALE; out /= QA*QB`.

Feature index for a piece (engine index `p` = `color*6+type`, square `sq`, a1=0):
white-perspective `p*64 + sq`; black-perspective `((color^1)*6+type)*64 + (sq^56)`.

### Verification (all green, run `./gradlew test --tests "engine.search.Nnue*"`)

- **`NnueCrossCheckTest`** — Java == the Python reference (`nnue_ref.py`) on committed seeded
  fixture nets: single-output (`src/test/resources/nnue/fixture_net.bin` + `fixture_golden.txt`)
  and 8-bucket (`fixture_net_b8.bin` + `fixture_golden_b8.txt`; the fixture FENs span 2..32
  pieces so all bucket arithmetic is exercised). Catches any index/quant/perspective/bucket
  transcription bug. Regenerate with
  `python nnue_ref.py --gen-fixture <net> <golden> [--buckets 8]`.
- **`NnueSymmetryTest`** — reference-free: a perspective net is invariant under the full mirror
  (flip + colour-swap + stm-swap), so `eval(pos) == eval(mirror)` exactly for any weights.
  This is what pins the `^56` flip and STM-first concat without a trained net.
- **`NnueInferenceTest`** — hand-computed forward pass (incl. truncate-toward-zero division).

`engine.tools.NnueProbe <net> < fens.txt` prints one eval per line, matching
`nnue_ref.py --eval <net> < fens.txt` — the same-spec cross-check for a REAL net, and the
harness to later confirm the reference matches **bullet's own** output (the final authority on
the `^56`/concat convention).

## Phase 2 — engine integration + gate

**Integration wiring (done).** Eval is runtime-selectable and OFF by default, so the shipping
engine and bench signature (`745650`) are unchanged unless explicitly enabled:

- UCI options `UseNNUE` (check, default false) and `EvalFile` (string path).
- `Uci` loads the net from `EvalFile`, else a bundled `/nnue/default.nnue` resource if present;
  any load failure falls back to the handcrafted eval (the engine never refuses to move).
- `LazySmpSearch.setEvaluatorFactory` gives each worker its own `NnueEvaluator` over one shared
  read-only `Network`.
- `NnueSearchIntegrationTest` proves single-threaded and SMP search with NNUE produce legal moves.

Try it: `setoption name EvalFile value path/to/net.bin` then `setoption name UseNNUE value true`.

**Still needs a real trained net (your GPU box):**
- Bundle the chosen net at `src/main/resources/nnue/default.nnue` (auto-ships via
  `copyToLichessBot`) OR pass it by `EvalFile` — the loader handles both.
- Cross-check `NnueProbe <net>` vs `nnue_ref.py --eval <net>` vs **bullet's own** eval on a few
  FENs — the final confirmation of the `^56`/concat convention on real weights.
- Dispatch `self-play-gate.yml`: candidate = `UseNNUE=true`, baseline = default HCE. Adopt at
  ≥51.5%. A weak/failing gate with a sane net points at an inference bug, not the net — the
  cross-check above is what rules that out first.

## Phase 3 (after the gate passes)

Incremental "efficiently-updatable" accumulator (feature deltas at Position's mutation choke
points), then SIMD if nps needs it. See `nnueroadmap.md`.
