"""Merges downloaded `selfplay-combined` artifacts (one per nnue-datagen.yml run -- manual
dispatches and scheduled auto-redispatch runs each produce their own, independent file) into one
growing training corpus.

Each GitHub Actions run's artifact unzips to a file literally named `selfplay-combined.txt`, so
downloading several will clobber each other unless renamed/kept in separate folders -- this
script takes explicit paths or a glob so any naming works.

Validates every line is well-formed bullet text (`<FEN> | <score> | <wdl>`, matching what
build_bootstrap_data.py / DataGen both emit) and reports a WDL summary, same shape as that
script's output, so a bad run stands out immediately rather than silently polluting the corpus.

Also flags exact full-line duplicates as a diagnostic (not removed by default): distinct
self-play games essentially never produce byte-identical FEN+eval+result lines by chance, so a
high duplicate count usually means two runs' seed ranges collided -- worth investigating, not
silently merging over.

Usage:
    python merge_selfplay_data.py --out data/selfplay_all.txt data/run1.txt data/run2.txt
    python merge_selfplay_data.py --out data/selfplay_all.txt "data/runs/*.txt"
"""
import argparse
import glob
import sys
from collections import Counter

VALID_WDL = {"1.0", "0.5", "0.0"}


def validate_line(line: str) -> bool:
    parts = line.split(" | ")
    if len(parts) != 3:
        return False
    fen, _score, wdl = parts
    return len(fen.split()) == 6 and wdl.strip() in VALID_WDL


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="input files or glob patterns")
    ap.add_argument("--out", required=True, help="output merged file")
    ap.add_argument("--dedup", action="store_true",
                    help="drop exact full-line duplicates instead of just reporting them")
    args = ap.parse_args()

    paths = []
    for pattern in args.inputs:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])  # plain path with no glob match: keep as-is

    seen = set()
    wdl = Counter()
    written = malformed = duplicates = 0

    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        for path in paths:
            n_before = written
            try:
                with open(path, encoding="utf-8", errors="replace") as inf:
                    for line in inf:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        if not validate_line(line):
                            malformed += 1
                            continue
                        if args.dedup or line not in seen:
                            if line in seen:
                                duplicates += 1
                                if args.dedup:
                                    continue
                            seen.add(line)
                            out.write(line + "\n")
                            written += 1
                            wdl[line.rsplit(" | ", 1)[1]] += 1
                        else:
                            duplicates += 1
            except OSError as e:
                print(f"  SKIP {path}: {e}", file=sys.stderr)
                continue
            print(f"  {path}: {written - n_before} positions", file=sys.stderr)

    print(f"DONE: {written} written to {args.out}", file=sys.stderr)
    if malformed:
        print(f"  malformed lines skipped: {malformed}", file=sys.stderr)
    if duplicates:
        action = "removed" if args.dedup else "kept (pass --dedup to drop)"
        print(f"  exact-duplicate lines: {duplicates} ({action}) "
              f"-- if this is large, check for a seed-range collision between runs",
              file=sys.stderr)
    if written:
        print("  WDL split (white POV): "
              + ", ".join(f"{k}:{v} ({100*v/written:.1f}%)"
                          for k, v in sorted(wdl.items(), reverse=True)),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
