#!/usr/bin/env python3
# ============================================================================
# gfx906-mmvq-q4_0-multicol.py - mul_mat_vec_q for Q4_0 with 2..8 columns
# (verification of MTP draft tokens) on GCN.
#
# WHY [rocprofv3, Qwen3.8-27B, 2x MI50, -sm tensor]: with MTP speculative
# decoding every step verifies 1 + 3 draft tokens in one pass. That pass cost
# 51 ms vs 31 ms for a single token, although weights are read only once.
# For every 4 bytes of weights the kernel re-reads 8 bytes of activations for
# EACH column through L1, which is the bottleneck on Vega.
#
# WHAT: quants of all rows of the block and activations of all columns are
# loaded into registers once, then every (row, column) pair is computed.
# 4 rows per block x 2 wavefronts: with 2 wavefronts the per-thread split of
# the sum is the same as in the single-column kernel, so the model output is
# BIT-IDENTICAL to unpatched llama.cpp (and speculative decoding does not
# change the answer). 4 rows x 1 wavefront is ~4% faster on the verify pass
# but breaks that property - override with --rpb/--nw if you do not care.
#
# Result: 4-token pass 51.0 -> 43.2 ms; Qwen3.8-27B agent request 40 -> 48 tok/s.
#
# Usage:  python3 gfx906-mmvq-q4_0-multicol.py ggml/src/ggml-cuda/mmvq.cu [--rpb 4 --nw 2]
# ============================================================================
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--rpb", type=int, default=4, help="rows per block for 2..8 columns")
ap.add_argument("--nw", type=int, default=2, help="wavefronts per block for 2..8 columns")
a = ap.parse_args()
src = open(a.path, encoding="utf-8").read()


def replace_once(old, new, what):
    global src
    n = src.count(old)
    if n != 1:
        sys.exit(f"gfx906-mmvq-q4_0-multicol: anchor '{what}' found {n} times, expected 1")
    src = src.replace(old, new, 1)


replace_once(
    "static constexpr __host__ __device__ int calc_nwarps(",
    f"""// [patch gfx906-mmvq-q4_0-multicol] GCN parameters for 2..8 columns
#ifndef GCN_MMVQ_RPB
#define GCN_MMVQ_RPB {a.rpb}
#endif
#ifndef GCN_MMVQ_NW
#define GCN_MMVQ_NW {a.nw}
#endif

static constexpr __host__ __device__ int calc_nwarps(""",
    "calc_nwarps (macros)")

replace_once(
    "    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {",
    """    // [patch gfx906-mmvq-q4_0-multicol]
    if (table_id == MMVQ_PARAMETERS_GCN && ncols_dst >= 2 && ncols_dst <= 8) {
        return GCN_MMVQ_RPB;
    }
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {""",
    "calc_rows_per_block")

replace_once(
    """    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {""",
    """    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        // [patch gfx906-mmvq-q4_0-multicol]
        if (ncols_dst >= 2 && ncols_dst <= 8) {
            return GCN_MMVQ_NW;
        }
        switch (ncols_dst) {""",
    "calc_nwarps GCN")

OLD_LOOP = """#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);"""

replace_once(OLD_LOOP, """#if defined(GGML_USE_HIP) && defined(GCN)
        // [patch gfx906-mmvq-q4_0-multicol] row weights and column activations
        // go to registers once, then all pairs are computed.
        if constexpr (type == GGML_TYPE_Q4_0 && ncols_dst > 1 && !has_fusion) {
            int   v[rows_per_cuda_block][vdr];
            float d4[rows_per_cuda_block];
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                const block_q4_0 * bx = (const block_q4_0 *) vx + kbx_offset + i*stride_row_x + kbx;
#pragma unroll
                for (int l = 0; l < vdr; ++l) {
                    v[i][l] = get_int_b2(bx->qs, kqs + l);
                }
                d4[i] = bx->d;
            }
            int   u[ncols_dst][2*vdr];
            half2 ds[ncols_dst];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const block_q8_1 * by = &y[j*stride_col_y + kby];
#pragma unroll
                for (int l = 0; l < vdr; ++l) {
                    u[j][2*l+0] = get_int_b4(by->qs, kqs + l);
                    u[j][2*l+1] = get_int_b4(by->qs, kqs + l + QI4_0);
                }
                ds[j] = by->ds;
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp[j][i] += vec_dot_q4_0_q8_1_impl<vdr>(v[i], u[j], d4[i], ds[j]);
                }
            }
            continue;
        }
#endif // defined(GGML_USE_HIP) && defined(GCN)

""" + OLD_LOOP, "column loop")

open(a.path, "w", encoding="utf-8").write(src)
print(f"gfx906-mmvq-q4_0-multicol: applied (rpb={a.rpb}, nw={a.nw})")
