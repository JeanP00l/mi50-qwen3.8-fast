#!/usr/bin/env python3
# ============================================================================
# gfx906-mmvf-few-rows.py - mul_mat_vec_f for matrices with very few rows.
#
# WHY [rocprofv3, Qwen3.8-27B, MI50]: ssm_alpha and ssm_beta are F32 matrices
# 5120 x 48, two per each of the 48 DeltaNet layers. Upstream launches one
# block per row, and on GCN the block is limited to 128 threads: 48 blocks of
# 2 wavefronts on 60 CUs - the GPU is almost empty, 13.4 us per matrix
# (~73 GB/s), 1.3 ms per token.
#
# WHAT: if there are fewer rows (x channels x samples) than 2 x CU count and
# rows are long, use 512/1024-thread blocks: 8-16 wavefronts per row, 8x more
# loads in flight. Same arithmetic; only the split of a row between threads
# changes (as with any other block size upstream may pick).
#
# WHERE: only wave64 (GCN/CDNA). NVIDIA and RDNA are unchanged.
#
# Usage:  python3 gfx906-mmvf-few-rows.py ggml/src/ggml-cuda/mmvf.cu
# ============================================================================
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "ggml/src/ggml-cuda/mmvf.cu"
src = open(path, encoding="utf-8").read()


def replace_once(old, new, what):
    global src
    n = src.count(old)
    if n != 1:
        sys.exit(f"gfx906-mmvf-few-rows: anchor '{what}' found {n} times, expected 1")
    src = src.replace(old, new, 1)


ANCHOR_CHOICE = ("    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;\n\n"
                 "    const int nbytes_shared = warp_size*sizeof(float) + (has_fusion ? warp_size*sizeof(float) : 0);\n")
replace_once(ANCHOR_CHOICE, ANCHOR_CHOICE + '''
    // [patch gfx906-mmvf-few-rows] few rows -> big block, otherwise the GPU sits idle
    if (warp_size == 64) {
        const int64_t nsm    = ggml_cuda_info().devices[device].nsm;
        const int64_t blocks = nrows * nchannels_dst * nsamples_or_ntokens;
        if (blocks < 2*nsm && ncols/2 >= 2048) {
            block_size_best = 1024;
        } else if (blocks < 2*nsm && ncols/2 >= 1024) {
            block_size_best = 512;
        }
    }
''', "has_fusion / nbytes_shared")

ANCHOR_CASE = '''        case  256: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 256, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
'''
extra = ""
for bs in (512, 1024):
    extra += ANCHOR_CASE.replace("case  256", "case %4d" % bs).replace("ncols_dst, 256,", "ncols_dst, %d," % bs)
replace_once(ANCHOR_CASE, ANCHOR_CASE + extra, "case 256")

open(path, "w", encoding="utf-8").write(src)
print("gfx906-mmvf-few-rows: applied")
