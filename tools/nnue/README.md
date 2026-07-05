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
