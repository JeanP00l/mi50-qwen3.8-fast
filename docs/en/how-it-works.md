# How it works

## Why `-sm tensor` + MTP

Qwen3.8-27B is dense: every token reads all ~15 GB of weights. On one MI50 (HBM2 ~680–770 GB/s
measured) that alone is ~20 ms per token. With `-sm layer` the two GPUs work **in turn**, so the
second card only adds memory, not speed (~22 tok/s). `-sm tensor` splits every matrix between the
cards, both HBMs are read at once (~31 tok/s without speculation), at the cost of an all-reduce
twice per layer (128 per token).

MTP (multi-token prediction) uses the model's own extra head to draft 3 tokens, then verifies
them in one pass of 4 tokens. If the verify pass cost the same as a single-token pass, speed would
scale with the number of accepted tokens (~3 per step on code).

## Why ROCm 7.14

On ROCm 6.3 `-sm tensor` deadlocked about once per 8 long requests: one GPU spinning in the
all-reduce waiting for a flag, the other one stuck; after ~10 s the driver tries a GPU reset and on
this platform the whole machine dies. With the same llama.cpp built on ROCm 7.14 (TheRock build
with gfx906 by [mixa3607](https://github.com/mixa3607/ML-gfx906)) — 0 hangs in 30+ long requests
and a 6 x 900 s load test. Speed is the same.

## The patches

All three are Python scripts that edit llama.cpp source and **check every anchor**: if upstream
changes, the build fails instead of silently producing an unpatched binary. They only affect
wave64 GPUs (GCN/CDNA); NVIDIA and RDNA paths are untouched.

### `gfx906-mmvq-q4_0-multicol.py` — the MTP verify pass (+13 %)

Profile (rocprofv3) of one MTP step showed the 4-token verify pass at 51 ms vs 31 ms for one token.
For 4 columns, `mul_mat_vec_q` re-reads, for every 4 bytes of Q4_0 weights, 8 bytes of activations
**per column** through L1; on Vega that traffic is the limit. The patch loads the weights of 4 rows
and the activations of all columns into registers once and then computes all (row, column) pairs.

Tuning on the real model (4-token pass): stock 51.0 ms, 8 rows x 1 wave 41.9, 4 x 1 41.6,
**4 x 2 43.2 (chosen)**. With 2 waves the split of each dot product between threads is the same
as in the single-column kernel, so every column gives bit-identical results. With 1 wave a token's
value depended on whether it was computed in a 1- or 4-token pass, i.e. on the draft acceptance
history — speculation started to change the answer. 1.6 ms is not worth that.

### `gfx906-gdn-multicolumn.py` — Gated DeltaNet

48 of Qwen3.8's 64 layers are Gated DeltaNet. The stock kernel runs one wavefront per state
column with two dependent reductions: ~35 us per layer, 1.7 ms per token, ~180 GB/s. One
wavefront now handles 4 columns (4 independent reductions in flight), 4x fewer wavefronts, all
in one wave. Same arithmetic in the same order.

### `gfx906-mmvf-few-rows.py` — tiny F32 matrices

`ssm_alpha` / `ssm_beta` are 5120 x 48 F32, two per DeltaNet layer. Upstream launches one
128-thread block per row: 48 blocks on 60 CUs, 13.4 us per matrix, 1.3 ms per token. For very
few long rows the patch uses 512/1024-thread blocks.

## What is left

After the patches one MTP step is ~61 ms for ~3 tokens:

- verify pass: ~44 ms of GPU work; its Q4_0 mat-vec runs at 1.9x the single-column time and is
  now instruction-bound (`dot4` + nibble unpacking per column), not memory-bound;
  single-column mat-vec already reaches ~595 GB/s;
- `flash_attn_tile` for 4 queries at 17K context: ~180 us x 16 layers where ~50 would be possible;
- **10–17 ms of GPU idle per step at pass boundaries** (5 passes per step): logits copied to the
  host, sampling over 248K vocabulary on the CPU, graph inputs written to each GPU with a sync.
  Inside a pass the host is ahead of the GPUs; this is the "gap after all-reduce" that is easy to
  misread in a profile.

Tools used: `rocprofv3 --kernel-trace --hip-runtime-trace`, `perf record` on the server thread,
`llama-bench -p 1,4`, `test-backend-ops` for correctness, `llama-perplexity --kl-divergence`.
