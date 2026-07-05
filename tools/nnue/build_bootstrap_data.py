"""Phase 0 (NNUE roadmap): build the bootstrap training set in bullet's TEXT format.

Reads one or more tune-corpus CSVs (the schema Tune.java / convert_zurichess_epd.py
emit: `fen,c1,c2,c3,c4,game_id,c6,c7,c8,c9,white_result`) and writes bullet's
documented text line format, one position per line:

    <FEN> | <score> | <result>

- `score`  is white-relative centipawns. We only have game-outcome (WDL) labels for the
  bootstrap net, so score is fixed at 0 and the first net is trained WDL-only (bullet
  `lambda` picks WDL vs eval; blend eval in on a later generation -- see nnueroadmap.md
  Phase 4). Emitting 0 is a valid, explicit "no eval label", NOT a silent default.
- `result` is white-relative: 1.0 win / 0.5 draw / 0.0 loss. `white_result` in the CSV is
  ALREADY from white's POV (`1-0` / `1/2-1/2` / `0-1`), so the mapping is direct and does
  not depend on side-to-move.

Why text (not the 32-byte binary) is emitted here: bullet ships its own tested converter
that packs this text into `bulletformat`'s ChessBoard binary. Hand-emitting the packed
binary would re-implement bullet's piece-packing / perspective normalization and risk a
silent layout bug (a wrong byte corrupts training with no crash) -- exactly the footgun the
roadmap flags. So we stop at the text boundary and let bullet do the packing:

    # on the training box, in the bullet checkout (verify flags with `--help`, they
    # vary by bullet version):
    cargo run --release --bin convert -- --input bootstrap_wdl.txt --output bootstrap_wdl.bin

FENs are normalized to 6 space-separated fields (zurichess EPDs carry only 4 -- placement,
stm, castling, ep) so any strict FEN parser accepts them; only placement + stm actually
affect the 768-feature transform.

Usage:
    python build_bootstrap_data.py \
        --in ../tune_data/positions.csv \
        --in ../tune_data/positions_zurichess.csv \
        --out data/bootstrap_wdl.txt
"""
import argparse
import csv
import sys
from collections import Counter

RESULT_TO_WDL = {"1-0": "1.0", "1/2-1/2": "0.5", "0-1": "0.0"}


def normalize_fen(fen: str) -> str | None:
    """Ensure a well-formed 6-field FEN; return None if placement/stm are missing."""
    parts = fen.strip().split()
    if len(parts) < 2:
        return None
    # placement, stm, castling, ep, halfmove, fullmove
    while len(parts) < 6:
        parts.append({2: "-", 3: "-", 4: "0", 5: "1"}[len(parts)])
    return " ".join(parts[:6])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", action="append", required=True,
                    help="tune-corpus CSV (repeatable)")
    ap.add_argument("--out", required=True, help="output bullet text file")
    args = ap.parse_args()

    written = 0
    skipped = Counter()
    wdl = Counter()

    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        for path in args.inputs:
            n_before = written
            with open(path, encoding="utf-8", errors="replace", newline="") as inf:
                reader = csv.DictReader(inf)
                for row in reader:
                    result = (row.get("white_result") or "").strip()
                    target = RESULT_TO_WDL.get(result)
                    if target is None:
                        skipped["bad_result"] += 1
                        continue
                    fen = normalize_fen(row.get("fen") or "")
                    if fen is None:
                        skipped["bad_fen"] += 1
                        continue
                    out.write(f"{fen} | 0 | {target}\n")
                    written += 1
                    wdl[target] += 1
            print(f"  {path}: {written - n_before} positions", file=sys.stderr)

    total = written + sum(skipped.values())
    print(f"DONE: {written} written, {sum(skipped.values())} skipped "
          f"(of {total}) -> {args.out}", file=sys.stderr)
    if skipped:
        print(f"  skip reasons: {dict(skipped)}", file=sys.stderr)
    if written:
        print("  WDL split (white POV): "
              + ", ".join(f"{k}:{v} ({100*v/written:.1f}%)"
                          for k, v in sorted(wdl.items(), reverse=True)),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
