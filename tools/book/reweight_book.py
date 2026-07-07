"""Re-weight a Polyglot opening book by the ENGINE'S OWN evaluation.

Motivation: a big generic book (e.g. Komodo, 578k entries) gives broad, sound coverage, but
`weighted_random` will happily steer the bot into sharp mainlines its small NNUE mis-evaluates
and therefore *plays* badly (the Meran-gambit losses). This tool walks the book from the start
position, has the engine score every candidate move at each reachable position, then rewrites the
weights so lines the engine likes dominate and lines it dislikes are pruned. The result is the
same book format (drop-in `book.bin`) but biased toward positions the engine actually handles
well -- a results patch, self-updating: re-run it when the net improves and the repertoire widens.

Key insight: for the bot's *results* you WANT it to avoid what it evaluates poorly, even if that
eval is objectively wrong -- because it will play those lines as badly as it scores them.

Design (three phases):
  1. DISCOVER (fast, single-thread): BFS the book tree from startpos to --max-ply, dedup by
     Zobrist key (transpositions eval'd once), collect each position's candidate moves. A
     polyglot book is keyed by hash and stores no positions, so we must reconstruct them by
     walking from the root -- which this does for free.
  2. EVAL (parallel): a pool of engine subprocesses scores every candidate move. For a move m at
     position P, we eval the CHILD position (P+m) at fixed depth and NEGATE (the child's score is
     from the opponent's perspective), giving m's value from the mover's side. Node/`go nodes`
     isn't parsed by the engine, so we use `go depth`.
  3. REWEIGHT + EMIT: at each position, prune moves worse than the best by --prune-margin, then
     weight survivors by softmax exp((score-best)/--temp), clamped to polyglot's 16-bit range.
     Emit sorted 16-byte entries (big-endian key/move/weight/learn), preserving each move's
     ORIGINAL raw bytes so castling (king-captures-rook) encoding is never mis-rewritten.

Usage:
    python reweight_book.py --in book.bin --out book_reweighted.bin \
        --engine cp:build/classes/java/main --depth 12 --concurrency 6 \
        --max-ply 16 --min-src-weight 1 --prune-margin 60 --temp 40

The engine defaults to NNUE (the shipped net), so weights reflect the *shipping* engine's taste.
"""
import argparse
import chess
import chess.polyglot
import math
import os
import struct
import subprocess
import sys
import threading
from collections import deque

MATE_CP = 30000  # mate scores mapped to a large centipawn magnitude for ordering


class Engine:
    """Minimal UCI driver for one engine subprocess. `cp:<dir>` launches our own build via java
    (matching MatchRunner's SubprocessEngine, incl. the incubator Vector API module); any other
    spec is treated as a path to a UCI executable."""

    def __init__(self, spec):
        if spec.startswith("cp:"):
            cmd = ["java", "--add-modules", "jdk.incubator.vector", "-cp", spec[3:], "engine.uci.Uci"]
        else:
            cmd = [spec]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self._send("uci")
        self._read_until(lambda l: l == "uciok")
        self._send("isready")
        self._read_until(lambda l: l == "readyok")

    def _send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _read_until(self, pred):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine closed the pipe unexpectedly")
            line = line.strip()
            if pred(line):
                return line

    def eval(self, fen, depth):
        """Score of the side to move at `fen`, in centipawns, from a depth-`depth` search."""
        self._send("position fen " + fen)
        self._send("go depth " + str(depth))
        score = 0
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine closed the pipe mid-search")
            line = line.strip()
            if line.startswith("info ") and " score " in line:
                score = _parse_score(line)
            elif line.startswith("bestmove"):
                return score

    def close(self):
        try:
            self._send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def _parse_score(info):
    t = info.split()
    if "cp" in t:
        try:
            return int(t[t.index("cp") + 1])
        except (ValueError, IndexError):
            return 0
    if "mate" in t:
        try:
            m = int(t[t.index("mate") + 1])
        except (ValueError, IndexError):
            return 0
        return (MATE_CP - abs(m)) * (1 if m >= 0 else -1)
    return 0


def discover(reader, max_ply, min_src_weight):
    """BFS the book from startpos. Returns an ordered list of (key, fen, moves) where moves is a
    list of (raw_move, learn, uci) for that position's book candidates; dedup by Zobrist key."""
    positions = []
    seen = set()
    q = deque([chess.Board()])
    while q:
        board = q.popleft()
        key = chess.polyglot.zobrist_hash(board)
        if key in seen:
            continue
        seen.add(key)
        entries = list(reader.find_all(board, minimum_weight=min_src_weight))
        if not entries:
            continue
        moves = []
        for e in entries:
            moves.append((e.raw_move, e.learn, e.move.uci()))
            if board.ply() + 1 < max_ply:
                child = board.copy(stack=False)
                child.push(e.move)
                q.append(child)
        positions.append((key, board.fen(), moves))
    return positions


def eval_worker(spec, depth, work, results, progress, lock, total):
    """One engine subprocess pulling positions off `work`, scoring each candidate move (eval the
    child, negate for the mover's POV), pushing (key, [(raw_move, learn, score)]) to `results`."""
    engine = Engine(spec)
    try:
        while True:
            try:
                key, fen, moves = work.pop()
            except IndexError:
                return
            board = chess.Board(fen)
            scored = []
            if len(moves) == 1:
                # forced book move: no choice to weigh, skip the eval entirely
                raw, learn, _ = moves[0]
                scored.append((raw, learn, 0))
            else:
                for raw, learn, uci in moves:
                    board.push(chess.Move.from_uci(uci))
                    scored.append((raw, learn, -engine.eval(board.fen(), depth)))
                    board.pop()
            results[key] = scored
            with lock:
                progress[0] += 1
                if progress[0] % 200 == 0 or progress[0] == total:
                    print(f"  eval {progress[0]}/{total} positions", file=sys.stderr)
    finally:
        engine.close()


def reweight(scored, prune_margin, temp, base):
    """Turn per-move engine scores into polyglot weights: drop moves worse than best by
    prune_margin, softmax the survivors, clamp to [1, 65535]."""
    best = max(s for _, _, s in scored)
    out = []
    for raw, learn, s in scored:
        if best - s > prune_margin:
            continue
        w = int(round(base * math.exp((s - best) / temp)))
        out.append((raw, learn, max(1, min(65535, w))))
    # Guard: never emit an empty position (shouldn't happen, best always survives)
    if not out and scored:
        raw, learn, _ = max(scored, key=lambda x: x[2])
        out = [(raw, learn, base)]
    return out


def write_book(path, entries_by_key):
    """entries_by_key: dict key -> list of (raw_move, learn, weight). Writes sorted 16-byte
    big-endian polyglot entries (key ascending, then weight descending)."""
    rows = []
    for key, moves in entries_by_key.items():
        for raw, learn, weight in moves:
            rows.append((key, weight, raw, learn))
    rows.sort(key=lambda r: (r[0], -r[1]))  # key asc, weight desc
    with open(path, "wb") as f:
        for key, weight, raw, learn in rows:
            f.write(struct.pack(">QHHI", key, raw, weight, learn))
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="source polyglot .bin")
    ap.add_argument("--out", required=True, help="output re-weighted polyglot .bin")
    ap.add_argument("--engine", required=True, help="cp:<classes-dir> or path to a UCI engine")
    ap.add_argument("--depth", type=int, default=12, help="eval search depth per move (default 12)")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel engine processes")
    ap.add_argument("--max-ply", type=int, default=16, help="walk the book this many plies deep")
    ap.add_argument("--min-src-weight", type=int, default=1, help="ignore source moves below this weight")
    ap.add_argument("--prune-margin", type=int, default=60,
                    help="drop a move if it is worse than the position's best by more than this (cp)")
    ap.add_argument("--temp", type=float, default=40.0, help="softmax temperature in cp (higher = flatter)")
    ap.add_argument("--base", type=int, default=1000, help="weight assigned to each position's best move")
    args = ap.parse_args()

    print(f"[1/3] discovering book positions (max_ply={args.max_ply}) ...", file=sys.stderr)
    with chess.polyglot.open_reader(args.inp) as reader:
        positions = discover(reader, args.max_ply, args.min_src_weight)
    total = len(positions)
    print(f"      {total} unique positions to score", file=sys.stderr)

    print(f"[2/3] scoring with {args.concurrency} engine(s) at depth {args.depth} ...", file=sys.stderr)
    work = list(positions)  # workers .pop() off the end; list is a thread-safe LIFO for pop()
    results = {}
    progress = [0]
    lock = threading.Lock()
    threads = [threading.Thread(target=eval_worker,
                                args=(args.engine, args.depth, work, results, progress, lock, total))
               for _ in range(max(1, args.concurrency))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("[3/3] re-weighting and writing ...", file=sys.stderr)
    entries_by_key = {}
    for key, _fen, _moves in positions:
        scored = results.get(key)
        if scored:
            entries_by_key[key] = reweight(scored, args.prune_margin, args.temp, args.base)
    n = write_book(args.out, entries_by_key)
    kept = sum(len(v) for v in entries_by_key.values())
    dropped_positions = total - len(entries_by_key)
    print(f"DONE: wrote {n} entries across {len(entries_by_key)} positions to {args.out}",
          file=sys.stderr)
    print(f"      (avg {kept/max(1,len(entries_by_key)):.2f} moves/position kept)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
