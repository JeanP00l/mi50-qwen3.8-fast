#!/usr/bin/env python3
"""Measure a running llama-server: prompt speed, generation speed, MTP draft acceptance.

    python3 scripts/bench-client.py [--url http://127.0.0.1:8080] [--key KEY] [--runs 3] [--max-tokens 2000]

Uses the server's own `timings`. The task is a coding request (like an agent turn).
Numbers depend on the text: code is predictable, so MTP accepts more drafts than on prose.
"""
import argparse
import json
import statistics
import urllib.request

TASK = """Write a small, dependency-free JavaScript module `snake.js` implementing the logic of the
game Snake (no rendering). Requirements:
- createGame({width, height, rng}) returns an immutable state: snake cells (head first),
  direction, pending direction, food cell, score, over flag.
- changeDirection(state, dir) ignores reversal into the snake's own neck.
- step(state) moves the snake one cell, grows on food, places new food with rng on a free cell,
  sets over=true on wall or self collision (moving into the cell the tail just left is allowed).
- Pure functions only, no globals, JSDoc on every export.
Then write `snake.test.js` using node:test and node:assert covering: initial state, movement,
growth, food placement on a free cell, wall collision, self collision, the tail-leaving edge case,
reversal ignored, and step after game over being a no-op. Explain briefly any design decisions."""


def run(url, key, max_tokens):
    body = {"model": "qwen3.8", "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95,
            "messages": [{"role": "system", "content": "You are a careful senior JavaScript engineer."},
                         {"role": "user", "content": TASK}]}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.load(r)
    t = data.get("timings", {})
    acc = None
    if t.get("draft_n"):
        acc = t.get("draft_n_accepted", 0) / t["draft_n"]
    return t.get("prompt_per_second"), t.get("predicted_per_second"), t.get("predicted_n"), acc


ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--key", default="")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--max-tokens", type=int, default=2000)
a = ap.parse_args()

gens = []
for i in range(1, a.runs + 1):
    pp, tg, n, acc = run(a.url, a.key, a.max_tokens)
    gens.append(tg)
    acc_s = f", MTP accepted {acc:.0%}" if acc is not None else ""
    print(f"run {i}: generation {tg:.1f} tok/s ({n} tokens), prompt {pp:.0f} tok/s{acc_s}", flush=True)
print(f"median generation: {statistics.median(gens):.1f} tok/s")
