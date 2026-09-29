#!/usr/bin/env python3
# ============================================================================
# gfx906-gdn-multicolumn.py - Gated DeltaNet kernel where one wavefront handles
# SEVERAL state columns instead of one.
#
# WHY [rocprofv3, Qwen3.8-27B, MI50]: gated_delta_net_cuda<128> takes ~35 us
# per layer, 48 DeltaNet layers = 1.7 ms per token. The 128x128x48 float state
# (3.1 MB) is read and written at ~180 GB/s while HBM does ~680. The kernel is
# latency-bound: one wavefront per column, two sequential reductions each,
# ~6000 short wavefronts in 2.5 waves over 60 CUs.
#
# WHAT: a wavefront handles C=4 columns, so C independent reductions run in
# parallel (ILP) and all wavefronts fit into one wave. Arithmetic is
# bit-identical: same terms, same order for every column.
#
# WHERE: only wave64 (GCN/CDNA), only non-KDA (scalar g: Qwen3.5/3.6/3.8),
# only S_v = 128. Every other path is untouched.
#
# Usage:  python3 gfx906-gdn-multicolumn.py ggml/src/ggml-cuda/gated_delta_net.cu
# Every anchor is checked: a miss fails the build instead of silently
# producing an unpatched binary.
# ============================================================================
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "ggml/src/ggml-cuda/gated_delta_net.cu"
src = open(path, encoding="utf-8").read()

KERNEL = r'''
// [patch gfx906-gdn-multicolumn] one wavefront handles C state columns.
template <int S_v, bool keep_rs_t, int C>
__global__ void __launch_bounds__(64 * 4, 1)
gated_delta_net_mc_cuda(const float * q,
                        const float * k,
                        const float * v,
                        const float * g,
                        const float * beta,
                        const float * curr_state,
                        float *       dst,
                        float *       state,
                        int64_t       H,
                        int64_t       n_tokens,
                        int64_t       n_seqs,
                        int64_t       sq1,
                        int64_t       sq2,
                        int64_t       sq3,
                        int64_t       sv1,
                        int64_t       sv2,
                        int64_t       sv3,
                        int64_t       sb1,
                        int64_t       sb2,
                        int64_t       sb3,
                        const uint3   neqk1_magic,
                        const uint3   rq3_magic,
                        float         scale,
                        int64_t       state_slot_stride,
                        int           K) {
    constexpr int warp_size     = 64;
    constexpr int rows_per_lane = S_v / warp_size;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of 64");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col0     = (blockIdx.z * blockDim.y + threadIdx.y) * C;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst;

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_out_offset;
    curr_state += state_in_offset;
    attn_data  += (sequence * n_tokens * H + h_idx) * S_v;

    float s_shard[C][rows_per_lane];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < C; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[c][r] = curr_state[(col0 + c) * S_v + r * warp_size + lane];
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float beta_val = beta[gb_offset];
        const float g_val    = expf(g[gb_offset]);

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            k_reg[r] = k_t[r * warp_size + lane];
            q_reg[r] = q_t[r * warp_size + lane];
        }

        float kv[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            float acc = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                acc += s_shard[c][r] * k_reg[r];
            }
            kv[c] = acc;
        }
#pragma unroll
        for (int c = 0; c < C; c++) {
            kv[c] = warp_reduce_sum<warp_size>(kv[c]);
        }

        float at[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            const float delta_col = (v_t[col0 + c] - g_val * kv[c]) * beta_val;
            float acc = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[c][r] = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                acc += s_shard[c][r] * q_reg[r];
            }
            at[c] = acc;
        }
#pragma unroll
        for (int c = 0; c < C; c++) {
            at[c] = warp_reduce_sum<warp_size>(at[c]);
        }
        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < C; c++) {
                attn_data[col0 + c] = at[c] * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * cs = state + target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < C; c++) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        cs[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < C; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net('''


def replace_once(old, new, what):
    global src
    n = src.count(old)
    if n != 1:
        sys.exit(f"gfx906-gdn-multicolumn: anchor '{what}' found {n} times, expected 1")
    src = src.replace(old, new, 1)


replace_once("template <bool KDA, bool keep_rs_t>\nstatic void launch_gated_delta_net(",
             KERNEL.lstrip("\n"), "launch_gated_delta_net")

replace_once('''        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,''',
             '''        case 128: {
            if constexpr (!KDA) {
                constexpr int C = 4;
                if (warp_size == 64 && (S_v % (num_warps * C)) == 0) {
                    const dim3 grid_mc(H, n_seqs, S_v / (num_warps * C));
                    const ggml_cuda_kernel_launch_params lp_mc = ggml_cuda_kernel_launch_params(grid_mc, dim3(64, num_warps, 1), 0, stream);
                    ggml_cuda_kernel_launch(gated_delta_net_mc_cuda<128, keep_rs_t, C>, lp_mc,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                    break;
                }
            }
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,''',
             "case 128 branch")

open(path, "w", encoding="utf-8").write(src)
print("gfx906-gdn-multicolumn: applied")
