"""Converts Zurichess's quiet-labeled.epd into Tune.java's CSV corpus schema.

Each EPD line is `<placement> <stm> <castling> <ep> c9 "<result>";` -- already quiet-filtered
(quiescence search found no winning capture) and independently labeled (each position had its
own Stockfish playout, per the archive's README), so there's no real "game" to group by. Every
row gets a synthetic, unique game_id -- an honest match to how the data was generated, not a
weakened stand-in for the per-game split the lichess corpus needed.

Usage:
    python convert_zurichess_epd.py --in quiet-labeled.epd --out positions_zurichess.csv
"""
import argparse
import re
import sys

LINE_RE = re.compile(r'^(?P<fen>.+?)\s+c9\s+"(?P<result>[^"]+)"\s*;?\s*$')
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    written = skipped = 0
    with open(args.inp, encoding="utf-8", errors="replace") as inf, \
            open(args.out, "w", encoding="utf-8", newline="\n") as out:
        out.write("fen,c1,c2,c3,c4,game_id,c6,c7,c8,c9,white_result\n")
        for i, line in enumerate(inf):
            m = LINE_RE.match(line.strip())
            if not m or m.group("result") not in VALID_RESULTS:
                skipped += 1
                continue
            out.write(f'"{m.group("fen")}",,,,,zc{i},,,,,{m.group("result")}\n')
            written += 1

    print(f"DONE: {written} positions written, {skipped} skipped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
