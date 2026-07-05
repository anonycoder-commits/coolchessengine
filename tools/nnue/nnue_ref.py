"""Reference NNUE forward pass (roadmap Phase 1 cross-check).

Independent, spec-faithful reimplementation of engine.search.NnueEvaluator, used to prove the
Java inference is correct. Same weights file format, same feature indexing, same integer
arithmetic (truncation toward zero) -- so for a given net + FEN the two MUST return the exact
same centipawn score.

Two roles:
  * `--gen-fixture`  writes a small seeded net + golden (fen, eval) file used by the Java
    `NnueCrossCheckTest` (committed test fixtures, so the cross-check runs in CI).
  * `--eval NET`     reads FENs from stdin, prints one eval per line -- for cross-checking a
    REAL trained net against both this reference AND bullet's own output (Phase 2).

Weights format = bullet's raw SavedFormat dump (little-endian i16, NO header): ftWeights
[768*hidden] (feature-major), ftBias[hidden], outWeights[2*hidden] (STM half first), outBias.
`hidden` is inferred from the file length. Activation is SCReLU and the inference arithmetic is
staged exactly as bullet's `Network::evaluate` (matches engine.search.NnueEvaluator).
"""
import argparse
import struct
import sys

INPUTS = 768
QA, QB, SCALE = 255, 64, 400  # bullet simple.rs defaults (no header carries them)

PIECE_FROM_CHAR = {  # -> engine piece index color*6+type (P N B R Q K)
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}


def trunc_div(a: int, b: int) -> int:
    """Integer division truncating toward zero (matches Java/C `/`); b > 0."""
    q = a // b
    if a < 0 and q * b != a:
        q += 1
    return q


def parse_fen(fen: str):
    """Return (pieces: list of (piece_index, square a1=0), stm: 0|1)."""
    placement, stm = fen.split()[0], fen.split()[1]
    pieces = []
    rank = 7  # first FEN rank is rank 8
    file = 0
    for ch in placement:
        if ch == '/':
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            sq = rank * 8 + file
            pieces.append((PIECE_FROM_CHAR[ch], sq))
            file += 1
    return pieces, (0 if stm == 'w' else 1)


class Net:
    def __init__(self, hidden, qa, qb, scale, ftw, ftb, outw, outb):
        self.hidden, self.qa, self.qb, self.scale = hidden, qa, qb, scale
        self.ftw, self.ftb, self.outw, self.outb = ftw, ftb, outw, outb

    def evaluate(self, fen: str) -> int:
        h = self.hidden
        pieces, stm = parse_fen(fen)
        w = list(self.ftb)
        b = list(self.ftb)
        for p, sq in pieces:
            color, ptype = p // 6, p % 6
            w_feat = p * 64 + sq
            b_feat = ((color ^ 1) * 6 + ptype) * 64 + (sq ^ 56)
            wo, bo = w_feat * h, b_feat * h
            for i in range(h):
                w[i] += self.ftw[wo + i]
                b[i] += self.ftw[bo + i]
        acc_stm, acc_nstm = (w, b) if stm == 0 else (b, w)
        out = 0
        for i in range(h):
            out += _screlu(acc_stm[i], self.qa) * self.outw[i]
        for i in range(h):
            out += _screlu(acc_nstm[i], self.qa) * self.outw[h + i]
        out = trunc_div(out, self.qa)     # QA*QA*QB -> QA*QB
        out += self.outb                  # QA*QB
        out *= self.scale
        return trunc_div(out, self.qa * self.qb)


def _screlu(x, qa):
    y = 0 if x < 0 else (qa if x > qa else x)
    return y * y


def write_net(path, net: Net):
    with open(path, "wb") as f:  # bullet raw format: no header, tight-packed i16
        f.write(struct.pack(f"<{len(net.ftw)}h", *net.ftw))
        f.write(struct.pack(f"<{len(net.ftb)}h", *net.ftb))
        f.write(struct.pack(f"<{len(net.outw)}h", *net.outw))
        f.write(struct.pack("<h", net.outb))


def read_net(path) -> Net:
    """bullet pads the whole saved file to a 64-byte boundary with filler bytes (observed:
    the literal ASCII "bullet" repeated) -- up to 31 trailing shorts that are neither a
    header nor real weights and must be tolerated, not treated as a parse error."""
    with open(path, "rb") as f:
        data = f.read()
    shorts = len(data) // 2
    assert len(data) % 2 == 0, "odd-length file"
    trailing = (shorts - 1) % (INPUTS + 3)
    assert trailing < 32, f"not a valid raw net (unexplained remainder {trailing} shorts)"
    hidden = (shorts - 1) // (INPUTS + 3)
    off = 0

    def take(n):
        nonlocal off
        vals = struct.unpack_from(f"<{n}h", data, off)
        off += 2 * n
        return list(vals)

    ftw = take(INPUTS * hidden)
    ftb = take(hidden)
    outw = take(2 * hidden)
    outb = take(1)[0]
    return Net(hidden, QA, QB, SCALE, ftw, ftb, outw, outb)


# Positions covering both sides to move, castling, en passant, and lopsided material.
FIXTURE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4",
    "2r3k1/pp3ppp/2n1b3/q7/3P4/2P1B3/P1Q2PPP/3R2K1 w - - 0 1",
    "8/5pk1/6p1/8/8/1P6/P4PPP/6K1 b - - 0 1",
    "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1",
    "7k/Q7/7K/8/8/8/8/8 w - - 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
]


def gen_fixture(net_path, golden_path, hidden=32, seed=1234):
    """Deterministic small net (hidden a multiple of 32, like a real net) so the committed
    fixture stays tiny (~49 KB) and reproducible."""
    import random
    rng = random.Random(seed)
    ftw = [rng.randint(-64, 64) for _ in range(INPUTS * hidden)]
    ftb = [rng.randint(-128, 128) for _ in range(hidden)]
    outw = [rng.randint(-64, 64) for _ in range(2 * hidden)]
    outb = rng.randint(-2000, 2000)
    net = Net(hidden, QA, QB, SCALE, ftw, ftb, outw, outb)
    write_net(net_path, net)
    # round-trip through the file so the golden reflects exactly what a loader will parse
    net = read_net(net_path)
    with open(golden_path, "w", encoding="utf-8", newline="\n") as g:
        for fen in FIXTURE_FENS:
            g.write(f"{net.evaluate(fen)}\t{fen}\n")
    print(f"wrote {net_path} (hidden={hidden}) and {golden_path} "
          f"({len(FIXTURE_FENS)} positions)", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-fixture", nargs=2, metavar=("NET", "GOLDEN"),
                    help="write a seeded fixture net + golden evals")
    ap.add_argument("--eval", metavar="NET", help="eval FENs from stdin against NET")
    args = ap.parse_args()

    if args.gen_fixture:
        gen_fixture(args.gen_fixture[0], args.gen_fixture[1])
        return 0
    if args.eval:
        net = read_net(args.eval)
        for line in sys.stdin:
            fen = line.strip()
            if fen:
                print(net.evaluate(fen))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
