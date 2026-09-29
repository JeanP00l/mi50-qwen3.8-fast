# One MI50

**Nothing on this page was measured by the author** (the test box has two 16 GB cards).
It is what follows from the measurements and the memory math.

## 1x MI50 32 GB

Fits: model 16.1 GB + MTP head 1.4 GB + KV cache (16 full-attention layers; 128K context is
roughly 8 GB) + compute buffers. `./run.sh` runs it on one GPU with MTP; lower `CTX` if it does
not start.

Expected speed: a single MI50 reads the weights about as fast as `-sm layer` on two cards
(~22 tok/s without speculation). MTP multiplies that by the accepted tokens per step, and the
`mmvq` patch is what makes the 4-token verify pass cheap — it matters **more** on one GPU than on
two. Rough expectation: 30–40 tok/s. Please report your numbers (`./bench.sh`).

## 1x MI50 16 GB

The 27B Q4_0 file alone is 16.1 GB (16 GiB card = 17.18 GB, minus driver/compute buffers):
**it does not fit** with any useful context. Options:

- **Add a second card.** Two 16 GB MI50 are the measured setup.
- **Smaller quant of the same model** (e.g. `IQ3_S`, ~11.8 GB). Caveat: K- and I-quants have no
  fast gfx906 kernel, so it will be slower per byte than Q4_0 and the `mmvq` patch (Q4_0 only)
  will not apply. Quality drops too. Put the file into `models/` and run
  `FORCE=1 MODEL=your-file.gguf CTX=16384 ./run.sh` (the Q4_0 MTP head may still fit; `MTP=0` if not).
- **A smaller Q4_0 model with an MTP head** (e.g. a Qwen3.5/3.6 dense or MoE that fits): all three
  patches are generic — Gated DeltaNet ones apply to any Qwen3.5/3.6/3.8 hybrid, `mmvq` to any
  Q4_0 model when a verify/batch pass of 2–8 tokens happens.

## Using the image with any model

The image is plain llama.cpp `b11235` + patches, `llama-server` is the entrypoint:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined -v /path/to/models:/models:ro -p 127.0.0.1:8080:8080 \
  mi50-llama:b11235 --model /models/your-model-Q4_0.gguf -ngl 999 --flash-attn on \
  --cache-reuse 0 --cache-ram 0 --ctx-checkpoints 0 --ctx-size 32768
```
