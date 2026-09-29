# Results

Hardware: 2x MI50 16 GB (blower-cooled), Xeon E5-2666 v3, X99, Ubuntu 24.04, kernel 6.8,
power limit 85 W per card, DPM level `high`, large BAR and P2P enabled
([hardware.md](hardware.md)). Model: `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-Q4_0.gguf`
(16.1 GB) + `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1.4 GB).

**Main metric:** generation speed on a recorded real coding-agent request (Claude Code, 17K-token
prompt, 22 tools, thinking on), 6000 output tokens, measured on the client side. Short synthetic
benchmarks hide both the speed of speculative decoding and the hangs.

## Step by step

| Configuration | Generation, tok/s | Prompt, tok/s | Stability |
|---|---|---|---|
| llama.cpp b10621, ROCm 6.3, `-sm layer`, no MTP | 19.4 | 330 | fine |
| b11235, `layer`, MTP | 25–26.5 | 280 | one full machine lockup in ~5 runs |
| b11235, `-sm tensor`, no MTP | 29.3 | 345 | hang ~1 in 4 |
| b11235, `tensor` + MTP draft 3, ROCm 6.3 | 39–45 | 325 | hang ~1 in 8–14 |
| b11235, `tensor` + MTP, **ROCm 7.14** | 40.0 (client), ~44 (server log) | 338 | **0 of 30**, 6 x 900 s load test clean |
| **+ the 3 patches of this repo** | **48.3** (client) | 338 | 6 x 900 s + 6 agent runs clean |

Same text, greedy (temperature 0): 43.1 → 48.7 tok/s with the patches.
Time to first token for the 17K prompt: ~51 s (prompt ~338 tok/s). Within one conversation
only the new tokens of each turn are read: 40–140 tokens, under a second.

`llama-bench` (no MTP, tokens/s, context 0 / 16K): `layer` 22.0 / 19.8; `tensor` 31.3 / 29.5.

Single pass time on 2 GPUs (`llama-bench -p 1,4`): 1 token 31.3 ms; 4 tokens 51.0 ms stock,
43.2 ms patched (the MTP verify pass).

MTP draft length (deterministic agent request): 2 → 42.6, **3 → 46.1**, 4 → 42.5, 5 → 40.5 tok/s.

## Quality

- **Bit-identical output**: greedy answer to the agent request has the same md5 as unpatched b11235.
- Perplexity / KL divergence vs the old reference build on 10 x 2048 tokens of Russian technical
  text, 4-token batches (`-b 4 -ub 4`, so the patched kernel is used):
  PPL 5.1500 vs 5.1528 (stock), KLD 0.00547 vs 0.00534 (±0.00015), same top token 96.7 %.
  For scale: Q4_0 vs Q8_0 is usually KLD 0.02–0.05.
- ROCm 7.14 vs 6.3 build: PPL/KLD identical.

## Tried, did not help

| What | Result |
|---|---|
| vLLM gfx906 fork (`aiinfos/vllm-gfx906-mobydick`), GPTQ-Int4, TP=2, MTP 3 | 34.7 tok/s, first token 71 s vs 51, KV cache only 64K, 11 min start. Its "1720 tok/s prompt" in logs is a 10-second averaging artefact. Hybrid prefix caching works (`--mamba-cache-mode align`), but it does not make generation faster |
| vLLM + MTP 2 + prefix caching | 33.7 tok/s |
| AWQ instead of GPTQ in vLLM | not tried: in that fork AWQ runs through the same gfx906 GEMM path |
| Power 90 / 95 / 100 W instead of 85 | +1 % / +5 % / — ; blowers go from quiet to 62–66 %. Not worth the noise here |
| `--spec-draft-backend-sampling`, `-bs` (GPU sampling) | no change |
| `HSA_ENABLE_SDMA=0` | no change |
| MTP draft head on one GPU (`-devd ROCm1`) | no change |
| `--spec-draft-n-max 4` | more tokens accepted, but a 5-token verify costs more; no gain |
| n-gram speculation (`ngram-mod`, `ngram-map-k`) | nothing on "write from scratch" tasks |
| Fusing gate+up on 2–4 columns | slower: 56.6 vs 43.2 ms per 4-token pass (register spills) |
| LDS-staged Q4_0 mat-vec | single column already runs at ~595 GB/s of ~700; LDS variants slower |
| Synchronous `set_tensor` for small inputs | no change |

## Where the time goes now

One MTP step (~61 ms, ~3 tokens accepted): verify pass ~44 ms of GPU work, draft passes ~3 ms
each, and **10–17 ms of GPU idle time at pass boundaries** (copy logits to host, sampling over a
248K vocabulary on the CPU, writing graph inputs to each GPU). That host overhead is the next
target; see [how-it-works.md](how-it-works.md).
