# Opening-book tooling

## `reweight_book.py` — bias a book toward what the engine plays well

A big generic polyglot book (e.g. Komodo, 578k entries) gives broad, sound coverage, but
`weighted_random` will steer the bot into sharp mainlines its small NNUE mis-evaluates and
therefore *plays* badly (the Meran-gambit losses). This tool walks the book from the start
position, has the **engine itself** score every candidate move at each reachable position, then
rewrites the weights so lines the engine likes dominate and lines it dislikes are pruned. Output
is the same polyglot format — a drop-in `book.bin`.

The point is deliberately a **results patch**: for the bot's score you *want* it to avoid what it
evaluates poorly, even where that eval is objectively wrong — it will play those lines as badly as
it scores them. Re-run it when the net improves (kb10, bigger nets) and the repertoire widens on
its own.

### How it works
1. **Discover** — BFS the book from startpos to `--max-ply`, dedup by Zobrist key. (A polyglot
   book is keyed by hash and stores no positions, so positions are reconstructed by walking from
   the root.)
2. **Eval** — a pool of engine subprocesses scores each candidate move: eval the *child* position
   at `--depth` and negate (child score is from the opponent's side). Original move bytes are
   preserved, so castling encoding is never mis-rewritten.
3. **Reweight** — at each position, drop moves worse than the best by `--prune-margin`, softmax
   the survivors by `--temp`, clamp to polyglot's 16-bit weight range, write sorted entries.

### Recommended run
Your bot consults the book to `max_depth: 20` plies, so reweighting to `--max-ply 20` covers
everything it uses (~3.5k positions, ~10–15 min):

```
python tools/book/reweight_book.py \
    --in lichess-bot/engines/book.bin \
    --out lichess-bot/engines/book_reweighted.bin \
    --engine cp:build/classes/java/main \
    --max-ply 20 --depth 12 --concurrency 6 \
    --prune-margin 90 --temp 50
```

Then review and install: `cp lichess-bot/engines/book_reweighted.bin lichess-bot/engines/book.bin`
and restart the bot. (Keep a backup of the source book.)

### Tuning knobs (it's fast — experiment)
- `--prune-margin` (cp): the variety-vs-conviction dial. Lower = prune harder = more
  deterministic, fewer lines (60 kept ~1.2 moves/position — very predictable). Higher (90–120)
  keeps 2–3 near-equal moves/position for variety while still dropping clearly-worse lines.
- `--temp` (cp): softmax spread. Higher = flatter weights among survivors.
- `--depth`: eval quality per move (12 is a good speed/quality point; higher is slower, better).
- `--max-ply`: how deep to reweight; match the bot's `max_depth`.
- `--min-src-weight`: skip rare source lines below this weight.

The engine defaults to NNUE (the shipped net), so weights reflect the *shipping* engine's taste.
Re-run after shipping a new net to refresh the repertoire.
