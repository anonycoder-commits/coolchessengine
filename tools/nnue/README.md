# NNUE tooling (Phase 0 — bootstrap data)

Follows `nnueroadmap.md`. This directory holds the offline, zero-engine-code Phase 0:
turn the existing labeled corpora into a training set and train the first net.

Target first net: **`(768 → 256)×2 → 1`** perspective net, CReLU, int16 quant — the
simplest thing that works (see the roadmap for the full architecture rationale).

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

Blackwell needs CUDA 12.8+. A bullet training program (Rust, `TrainerBuilder`) configures:

- input features: `Chess768` (768 = 6 piece-types × 2 colors × 64 squares), perspective
- hidden size **256**, activation CReLU
- `wdl` weight (lambda) = **1.0 (pure WDL)** for this net — score column is 0
- quant: accumulator ×255, output ×64 (matches the engine-side contract in the roadmap)
- export quantized weights (a few hundred KB)

Bullet's API changes between versions — copy the closest example under bullet's own
`examples/` / `docs/` rather than pasting a config from here. Deliverable of Phase 0 is the
exported weights file; **no engine change yet** (that's Phase 1: `NnueEvaluator`).

## Phase 1 — engine-side inference (non-incremental), correctness-first

`engine.search.NnueEvaluator` implements the `Evaluator` interface with a full accumulator
refresh every call. Architecture `(768 -> N)x2 -> 1`, clipped ReLU, int16 quant. HCE stays
the default eval — nothing is wired into search/UCI yet (that's Phase 2, once a real net
exists to bundle; a random net has no value in a game).

### Weights file format (little-endian)

| bytes | field | value |
|-------|-------|-------|
| i32   | magic | `0x434E5545` |
| i32   | version | 1 |
| i32   | inputs | 768 |
| i32   | hidden | N |
| i32   | QA | 255 (ft scale) |
| i32   | QB | 64 (output scale) |
| i32   | SCALE | 400 (eval scale) |
| i16 × 768·N | ftWeights | feature-major `[feature*N + i]` |
| i16 × N | ftBias | |
| i16 × 2N | outWeights | **STM half first**, then non-STM |
| i16 | outBias | scaled QA·QB |

Feature index for a piece (engine index `p` = `color*6+type`, square `sq`, a1=0):
white-perspective `p*64 + sq`; black-perspective `((color^1)*6+type)*64 + (sq^56)`. The bullet
export program (Phase 0 step 3) must write this exact layout — adapt its `save`/`quantise`
step, or transpose bullet's native dump into it.

### Verification (all green, run `./gradlew test --tests "engine.search.Nnue*"`)

- **`NnueCrossCheckTest`** — Java == the Python reference (`nnue_ref.py`) on a committed seeded
  fixture net (`src/test/resources/nnue/fixture_net.bin` + `fixture_golden.txt`). Catches any
  index/quant/perspective transcription bug. Regenerate with
  `python nnue_ref.py --gen-fixture <net> <golden>`.
- **`NnueSymmetryTest`** — reference-free: a perspective net is invariant under the full mirror
  (flip + colour-swap + stm-swap), so `eval(pos) == eval(mirror)` exactly for any weights.
  This is what pins the `^56` flip and STM-first concat without a trained net.
- **`NnueInferenceTest`** — hand-computed forward pass (incl. truncate-toward-zero division).

`engine.tools.NnueProbe <net> < fens.txt` prints one eval per line, matching
`nnue_ref.py --eval <net> < fens.txt` — the same-spec cross-check for a REAL net, and the
harness to later confirm the reference matches **bullet's own** output (the final authority on
the `^56`/concat convention).

### Deferred to Phase 2 (needs a real trained net)

Bundling weights in `src/main/resources/nnue/`, the `getResourceAsStream` default load, the
UCI `EvalFile`/`UseNNUE` option, per-worker instantiation in `LazySmpSearch`, and the
NNUE-vs-HCE self-play gate.
