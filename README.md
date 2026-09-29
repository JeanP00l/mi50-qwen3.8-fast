# mi50-qwen3.8-fast

**Qwen3.8-27B at ~48 tok/s on two AMD Instinct MI50 (gfx906) — up from 19.4 with stock llama.cpp.**
Same model, same Q4_0 file, bit-identical output. Patched llama.cpp in Docker, one-command setup.

[Русская версия](README.ru.md)

| Setup (2x MI50 16 GB, 85 W each) | Generation, agent-style request | Notes |
|---|---|---|
| llama.cpp b10621, ROCm 6.3, `-sm layer`, no MTP | 19.4 tok/s | typical starting point |
| b11235, `-sm tensor` + MTP, ROCm 6.3 | 39–45 | hangs about once per 8 long requests |
| b11235, `-sm tensor` + MTP, **ROCm 7.14** | 40–44 | 0 hangs in 30+ long requests |
| **+ gfx906 kernel patches (this repo)** | **48** | output bit-identical, KLD within noise |
| vLLM (gfx906 fork), GPTQ-Int4, TP=2 + MTP | 34.7 | slower prompt too, 64K context max |

Details and how each number was measured: [docs/en/results.md](docs/en/results.md).

## Quick start

Needs: Linux with the in-kernel `amdgpu` driver (any recent distro), Docker, ~30 GB of disk,
one or more MI50/MI60/Radeon VII.

```bash
git clone https://github.com/JeanP00l/mi50-qwen3.8-fast && cd mi50-qwen3.8-fast
./install.sh        # checks the system, builds the image (20-40 min once), downloads ~17.4 GB of models
./run.sh            # starts the server on http://127.0.0.1:8080 (OpenAI and Anthropic APIs)
```

Check it works (in another terminal): `./bench.sh`

`run.sh` picks the mode automatically:

| GPUs | What happens |
|---|---|
| 2 or more | two GPUs, `-sm tensor` + MTP speculative decoding — the measured setup |
| 1 × 32 GB | one GPU + MTP (should work; not measured by the author) |
| 1 × 16 GB | refuses: the 27B Q4_0 file is 16.1 GB. Options — [docs/en/single-gpu.md](docs/en/single-gpu.md) |

Useful settings: `API_KEY=secret HOST=0.0.0.0 PORT=8080 CTX=65536 ./run.sh`, `./run.sh --dry-run`
prints the exact `docker run` command. All options are listed at the top of `run.sh`.

Use with Claude Code (server started with `API_KEY=secret ./run.sh`):
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=secret \
       ANTHROPIC_MODEL=qwen3.8 ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8 \
       ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8 ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8 \
       CLAUDE_CODE_SUBAGENT_MODEL=qwen3.8 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
       NO_PROXY=127.0.0.1,localhost
claude
```
Keep one agent per server: `--parallel 1` lets a single conversation reuse its prompt cache
(40–140 new tokens per turn, <1 s); two agents evict each other and every turn re-reads 20–60K tokens.

## What is in here

| | |
|---|---|
| `docker/Dockerfile` | llama.cpp tag `b11235` on ROCm 7.14 with gfx906 support ([mixa3607/rocm-gfx906](https://hub.docker.com/r/mixa3607/rocm-gfx906)) + 3 patches |
| `patches/gfx906-mmvq-q4_0-multicol.py` | Q4_0 mat-vec for 2–8 columns: the MTP verify pass (1 + 3 draft tokens) 51 → 43 ms |
| `patches/gfx906-gdn-multicolumn.py` | Gated DeltaNet kernel (Qwen3.5/3.6/3.8 hybrids): one wavefront per 4 columns |
| `patches/gfx906-mmvf-few-rows.py` | F32 mat-vec for matrices with few rows (`ssm_alpha/beta`): 48 blocks on 60 CUs, GPU almost empty |
| `templates/qwen3.8-agent.jinja` | chat template that works with Claude Code (mid-conversation system messages, sane default effort) |
| `scripts/check-system.sh` | read-only check: driver, Docker, GPUs, **large BAR**, **GPU-to-GPU P2P** |

## The things that mattered most

1. **ROCm 7.14 instead of 6.3** for `-sm tensor`: 6.3 deadlocked in the all-reduce, the box died on GPU reset.
2. **BIOS: Above 4G Decoding + MMIO above 4G placed below 16 TiB** — Vega 20 DMA is 44-bit; with the
   aperture at 56 TiB there is no P2P and `-sm tensor` is 2x slower. [docs/en/hardware.md](docs/en/hardware.md)
3. **`-sm tensor` + MTP (draft 3)**: dense 27B is memory-bound, tensor split reads weights from both HBMs.
4. **`--cache-reuse 0 --cache-ram 0 --ctx-checkpoints 0`**: without them gfx906 hits
   `Memory access fault` under load. [docs/en/troubleshooting.md](docs/en/troubleshooting.md)
5. **Q4_0, not Q4_K_M**: K-quants have no fast gfx906 kernel.
6. The patches here: +13 % on top, with identical output. [docs/en/how-it-works.md](docs/en/how-it-works.md)

## Tested on

2x MI50 16 GB, Xeon E5-2666 v3 (X99), Ubuntu 24.04, kernel 6.8, Docker 29, power limit 85 W per card.
Other distros should work — only the kernel driver and Docker are used from the host. Reports welcome.

## License

MIT. llama.cpp is MIT; the chat template is derived from Qwen's (Apache-2.0) with Unsloth's fixes.
