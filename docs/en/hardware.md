# Hardware and BIOS

## 1. Large BAR: `Above 4G Decoding`

Enable `Above 4G Decoding` (often under `Advanced → PCI Subsystem Settings`), and Resizable BAR if
the board offers it. `CSM` does not need to be disabled.

The trap: `mem_info_vram_total` shows 16 GB **even with a 256 MB BAR** — the driver reads it from
the card, not through the BAR. Check the *visible* VRAM instead:

```bash
cat /sys/bus/pci/devices/0000:XX:00.0/mem_info_vis_vram_total   # want 17163091968 (16 GiB), not 268435456
```

`scripts/check-system.sh` does this for every card. With a 256 MB BAR everything still runs, but on
2x MI50 enabling the large BAR gave **+18 % generation**, and it is a precondition for P2P below.

## 2. GPU-to-GPU P2P: MMIO above 4G must sit below 16 TiB

On Xeon E5 / X99 boards there are usually two more options: `MMIOH Base` and `MMIO High Size`
(names vary: `MMIO High Base`, `PCI 64-bit Resource Allocation`, under Chipset / North Bridge).

| Option | Value |
|---|---|
| `MMIOH Base` | **`4T`** (anything below `16T`; not `16T` itself, not `56T`) |
| `MMIO High Size` | `256G` |

Vega 20 DMA addressing is **44-bit (16 TiB)**. With the aperture at 56 TiB the kernel refuses P2P
between the cards (`amdgpu_device_is_peer_accessible`), `p2p_links` is empty, and `-sm tensor`
falls back to slow copies: measured 38.5 tok/s vs 79 on the same test after moving the base to `4T`.

```bash
ls /sys/class/kfd/kfd/topology/nodes/*/p2p_links         # want non-empty for GPU nodes
sudo dmesg | grep "amdgpu.*BAR 0.*assigned"             # addresses below 0x100000000000
```

There is no direct bridge between slots on most X99 boards; P2P goes through the CPU root complex
and that is enough.

**BIOS settings can silently reset** (weak CMOS battery). Losing `Above 4G` drops visible VRAM to
256 MB and costs ~18 % speed. Replace the CR2032 if the check above ever flips.

## 3. Platform

- **Xeon E5 v3/v4 (LGA2011-3) or newer.** Sandy/Ivy Bridge (E5 v1/v2, LGA2011) lack PCIe atomics:
  `amdkfd` logs `PCI rejects atomics` and does not initialise the card.
- PCIe 3.0 x16 per card is what was tested; the all-reduce traffic of `-sm tensor` is small.
- Use the **in-kernel amdgpu driver**. Do not install `amdgpu-dkms` / `amdgpu-install`; ROCm lives
  inside the container.

## 4. Power, heat, noise

MI50 is passively cooled; in a normal case it needs a blower (tested: a Delta BFB0712HF blower in a
shroud on each card). Factory power limit is 250 W, which such cooling cannot hold (~100 °C
junction). Measured here:

| Limit per card | Generation | Junction (hot card) | Blowers |
|---|---|---|---|
| 85 W (used) | 48.3 tok/s | ~80–83 °C | ~45–55 % |
| 90 W | ~48.8 (+1 %) | ~83 °C | avg 59 %, >60 % 12 % of time |
| 95 W | ~50.6 (+5 %) | ~83–85 °C | 62 % constantly |
| 100 W | — | 85–86 °C | 62–66 % within 2 min |

Set the limit (root, per MI50; the value is in microwatts, resets on reboot; `scripts/check-system.sh`
prints the PCI address of each card):

```bash
echo 85000000 | sudo tee /sys/bus/pci/devices/0000:XX:00.0/hwmon/hwmon*/power1_cap
```

DPM level `high` (`echo high | sudo tee /sys/class/drm/cardN/device/power_dpm_force_performance_level`)
removes clock parking between kernels; measured +12.7 % generation on another model on this box.

Undervolting (OverDrive curve) works on MI50 but each card has its own safe margin: here -50 mV was
fine and -75 mV **silently corrupted results** without crashing. Only do it with an output check.
