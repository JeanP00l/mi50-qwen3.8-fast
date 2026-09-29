# Troubleshooting

**`Memory access fault by GPU node-N` under load, server restarts.**
The paths that save/restore conversation state crash on gfx906. Keep
`--cache-reuse 0 --cache-ram 0 --ctx-checkpoints 0` (run.sh does). Verified: 6 of 6 clean
900-second load runs with them; without them the fault came on the 7th request. Cost: the prompt
cache only works for one conversation per slot, hence `--parallel 1` and one agent per server.

**The whole machine freezes / reboots during generation with `-sm tensor`.**
Deadlock in the all-reduce on ROCm 6.3, followed by a failed GPU reset. Use this image (ROCm 7.14).
If it still happens, fall back to `EXTRA="--split-mode layer" ` or remove MTP (`MTP=0`) and report it.

**`-sm tensor` is barely faster than `layer` (or slower).**
No P2P between the cards. `scripts/check-system.sh` → fix MMIO placement in BIOS
([hardware.md](hardware.md) §2).

**Slow everything, `mem_info_vis_vram_total` is 268435456.**
`Above 4G Decoding` is off (or was reset by a dying CMOS battery).

**The model answers in a loop of thinking, or starts acting too late (agents).**
The stock Qwen template maps Claude Code's `effort: high` to `xhigh`; use the default
`TEMPLATE=agent`. Thinking is capped server-side by `REASONING_BUDGET` (8192): Claude Code does not
send `MAX_THINKING_TOKENS`. Do not go below ~768 — a cut thought silently gives a half-done answer.

**`System message must be at the beginning` error with Claude Code.**
Same fix: `TEMPLATE=agent`.

**K-quants (`Q4_K_M`, `IQ*`) are slow.** No fast gfx906 kernel for them. Use `Q4_0` (or `Q4_1`).

**`llama-bench` puts everything on one GPU and runs out of memory.**
`llama-bench -ts` takes a slash (`1/1`); a comma means "try each value". `llama-server` takes `1,1`.

**Docker cannot open `/dev/kfd`.** Add your user to `video` and `render` groups
(`sudo usermod -aG video,render $USER`, re-login) — run.sh adds both groups to the container.

**Build fails with `anchor ... found 0 times`.** The llama.cpp tag changed under a patch.
Build the tag the patches were written for (`LLAMA_TAG=b11235`, the default) or open an issue.

**Downloads are slow / interrupted.** `python3 scripts/download-models.py` resumes; set `HF_ENDPOINT`
to a mirror if needed.
