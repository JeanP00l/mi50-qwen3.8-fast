# shellcheck shell=bash
# Shared helpers: find gfx906 GPUs through sysfs (no ROCm needed on the host).

# Prints one line per gfx906 GPU: "<pci> <vram_bytes> <visible_vram_bytes> <kfd_node>"
gfx906_gpus() {
    local node props pci dom bus dev fn vram vis gpu_id
    for node in /sys/class/kfd/kfd/topology/nodes/*; do
        props="$node/properties"
        [ -r "$props" ] || continue
        # gfx906 = 9.0.6 -> 90006
        grep -q '^gfx_target_version 90006$' "$props" || continue
        gpu_id=$(cat "$node/gpu_id" 2>/dev/null)
        [ -n "$gpu_id" ] && [ "$gpu_id" != 0 ] || continue
        dom=$(awk '/^domain /{print $2}' "$props")
        bus=$(awk '/^location_id /{print $2}' "$props")
        # location_id = bus<<8 | dev<<3 | fn
        pci=$(printf '%04x:%02x:%02x.%x' "${dom:-0}" $((bus >> 8)) $(((bus >> 3) & 31)) $((bus & 7)))
        vram=$(cat "/sys/bus/pci/devices/$pci/mem_info_vram_total" 2>/dev/null || echo 0)
        vis=$(cat "/sys/bus/pci/devices/$pci/mem_info_vis_vram_total" 2>/dev/null || echo 0)
        echo "$pci $vram $vis $(basename "$node")"
    done
}

gib() { LC_ALL=C awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }

say()  { printf '%s\n' "$*"; }
ok()   { printf '  [ OK ] %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; }
