#!/usr/bin/env bash
# Checks that this machine can run the MI50 image. Read-only: changes nothing.
set -u
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=scripts/lib.sh
. scripts/lib.sh

fails=0
say "== System"
say "  kernel $(uname -r), $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown distro}")"

if [ -e /dev/kfd ]; then ok "/dev/kfd present (amdgpu + KFD in the kernel)"
else bad "/dev/kfd missing: amdgpu driver not loaded. Use a stock kernel >= 5.15; do NOT install amdgpu-dkms/amdgpu-install"; fails=$((fails+1)); fi

if ls /dev/dri/renderD* >/dev/null 2>&1; then ok "/dev/dri render nodes present"
else bad "/dev/dri/renderD* missing"; fails=$((fails+1)); fi

if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then ok "docker works for $(id -un)"
    else warn "docker installed but not usable by $(id -un): run with sudo or 'sudo usermod -aG docker $(id -un)' and re-login"; fi
else bad "docker not found: https://docs.docker.com/engine/install/"; fails=$((fails+1)); fi

command -v python3 >/dev/null 2>&1 && ok "python3 present" || { bad "python3 missing (needed for downloads/benchmarks)"; fails=$((fails+1)); }

say "== GPUs (gfx906 = MI50 / MI60 / Radeon VII)"
mapfile -t gpus < <(gfx906_gpus)
if [ "${#gpus[@]}" -eq 0 ]; then
    bad "no gfx906 GPU found"; fails=$((fails+1))
fi
small_bar=0
for g in "${gpus[@]}"; do
    read -r pci vram vis node <<<"$g"
    line="$pci  VRAM $(gib "$vram") GiB, CPU-visible $(gib "$vis") GiB"
    if [ "$vis" -lt "$vram" ]; then
        warn "$line - small BAR"
        small_bar=1
    else
        ok "$line - large BAR"
    fi
    cap=$(cat /sys/bus/pci/devices/"$pci"/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)
    [ -n "$cap" ] && say "         power limit $((cap / 1000000)) W"
done
if [ "$small_bar" = 1 ]; then
    say "  -> Enable 'Above 4G Decoding' (and Resizable BAR / large MMIO if offered) in BIOS."
    say "     Measured on 2x MI50: +18% generation, and GPU-to-GPU P2P needs it. See docs/en/hardware.md"
fi
if [ "${#gpus[@]}" -ge 2 ]; then
    read -r _ _ _ node <<<"${gpus[0]}"
    if [ -n "$(ls /sys/class/kfd/kfd/topology/nodes/"$node"/p2p_links 2>/dev/null)" ]; then
        ok "GPU-to-GPU P2P links present (needed for fast -sm tensor)"
    else
        warn "no P2P links between GPUs: -sm tensor will work but ~2x slower."
        say "     Usual cause: MMIO above 4G placed above the 44-bit DMA limit of Vega 20 (16 TiB)."
        say "     In BIOS set MMIOH base low (e.g. 4T instead of 56T). See docs/en/hardware.md"
    fi
fi

say "== Disk"
free=$(df -Pk "${MODELS_DIR:-./models}" 2>/dev/null | awk 'NR==2{print $4}')
[ -z "$free" ] && free=$(df -Pk . | awk 'NR==2{print $4}')
if [ "$free" -gt $((30 * 1024 * 1024)) ]; then ok "$((free / 1024 / 1024)) GiB free (need ~18 GiB models + ~12 GiB image)"
else warn "only $((free / 1024 / 1024)) GiB free; need ~30 GiB"; fi

say "== Verdict"
n=${#gpus[@]}
maxvram=0
for g in "${gpus[@]}"; do read -r _ v _ _ <<<"$g"; [ "$v" -gt "$maxvram" ] && maxvram=$v; done
if [ "$n" -ge 2 ]; then say "  $n GPUs: run.sh will use two of them with -sm tensor + MTP (the measured setup)."
elif [ "$n" -eq 1 ] && [ "$maxvram" -gt $((24 * 1073741824)) ]; then say "  1x 32 GB GPU: run.sh will run on one GPU with MTP (not measured by the author)."
elif [ "$n" -eq 1 ]; then say "  1x 16 GB GPU: Qwen3.8-27B Q4_0 (16.1 GB) does not fit. See docs/en/single-gpu.md"
fi
[ "$fails" -eq 0 ] && exit 0 || { say "  $fails blocking problem(s) above."; exit 1; }
