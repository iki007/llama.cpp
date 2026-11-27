#ifndef GGML_SYCL_MOE_Q4K_ESIMD_HPP
#define GGML_SYCL_MOE_Q4K_ESIMD_HPP

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdint>
#include "common.hpp"
#include "quants.hpp"

// Intel Arc Pro B70 (Xe2 Battlemage) Barrier-Free, Zero-SLM, Zero-Division ESIMD Hardware `dp4a (32|M0)` MoE Kernels
// Supports reordered SoA MoE experts: GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0
// paired with SoA Q8_1 activations ([qs: ncols B][ds: (ncols/32)*4 B]).
//
// Key design invariants:
// - ZERO workgroup barriers and ZERO SLM (hardware barrier deadlock is impossible).
// - ZERO integer divisions anywhere in prologue or hot loop (3D nd_range + incremental loops + precomputed byte offsets).
// - MAX_M=8 matching slots per single weight read -> fits in 128 GRFs with 0B spill.
// - Adaptive RI: RI=4 for small batches (n_tokens <= 4, higher EU occupancy) and RI=8 for larger batches (n_tokens > 4).
//
// Runtime selector: GGML_SYCL_MOE_Q4K_ESIMD
//   0 = disabled (baseline standard SYCL mul_mat_vec_q_moe_reorder / dedup)
//   1 = enabled, auto RI (RI=4 for n_tokens <= 4, RI=8 for n_tokens > 4) [default]
//   2 = forced RI=4
//   3 = forced RI=8

namespace ggml_sycl_moe_esimd {

namespace esimd = sycl::ext::intel::esimd;

static SYCL_ESIMD_FUNCTION inline esimd::simd<float, 4>
reduce32_to_4f(esimd::simd<int32_t, 32> v) {
    esimd::simd<int32_t, 16> s16 = v.select<16, 2>(0) + v.select<16, 2>(1);
    esimd::simd<int32_t, 8>  s8  = s16.select<8, 2>(0) + s16.select<8, 2>(1);
    esimd::simd<int32_t, 4>  s4  = s8.select<4, 2>(0)  + s8.select<4, 2>(1);
    return esimd::simd<float, 4>(s4);
}

static SYCL_ESIMD_FUNCTION inline esimd::simd<float, 8>
reduce32_to_8f(esimd::simd<int32_t, 32> v) {
    esimd::simd<int32_t, 16> s16 = v.select<16, 2>(0) + v.select<16, 2>(1);
    esimd::simd<int32_t, 8>  s8  = s16.select<8, 2>(0) + s16.select<8, 2>(1);
    return esimd::simd<float, 8>(s8);
}

static SYCL_ESIMD_FUNCTION inline float
reduce8_to_f(esimd::simd<int32_t, 8> v) {
    esimd::simd<int32_t, 4> s4 = v.select<4, 2>(0) + v.select<4, 2>(1);
    esimd::simd<int32_t, 2> s2 = s4.select<2, 2>(0) + s4.select<2, 2>(1);
    return float(int32_t(s2[0]) + int32_t(s2[1]));
}

static SYCL_ESIMD_FUNCTION inline void
decode_soa_k4_scales(const uint8_t * scales_ptr, uint32_t dm_u32,
                     esimd::simd<float, 4> & sc_ev, esimd::simd<float, 4> & sc_od,
                     esimd::simd<float, 4> & m_ev,  esimd::simd<float, 4> & m_od) {
    esimd::simd<uint32_t, 4> sc_raw = esimd::block_load<uint32_t, 4>(
        reinterpret_cast<const uint32_t *>(scales_ptr));
    esimd::simd<uint8_t, 16> sb = sc_raw.bit_cast_view<uint8_t>();

    esimd::simd<uint32_t, 1> dm_v(dm_u32);
    auto dm_h = dm_v.bit_cast_view<sycl::half>();
    const float dm_d = float(sycl::half(dm_h[0]));
    const float dm_m = float(sycl::half(dm_h[1]));

    const esimd::simd<uint32_t, 4> b03 = sb.select<4, 1>(0);
    const esimd::simd<uint32_t, 4> b47 = sb.select<4, 1>(4);
    const esimd::simd<uint32_t, 4> b8b = sb.select<4, 1>(8);

    esimd::simd<uint32_t, 8> sc_u, mn_u;
    sc_u.select<4, 1>(0) = b03 & 63u;
    mn_u.select<4, 1>(0) = b47 & 63u;
    sc_u.select<4, 1>(4) = (b8b & 0x0Fu) | ((b03 >> 6) << 4);
    mn_u.select<4, 1>(4) = (b8b >> 4)    | ((b47 >> 6) << 4);

    esimd::simd<float, 8> sc_f = esimd::simd<float, 8>(sc_u) * dm_d;
    esimd::simd<float, 8> mn_f = esimd::simd<float, 8>(mn_u) * dm_m;
    sc_ev = sc_f.select<4, 2>(0);
    sc_od = sc_f.select<4, 2>(1);
    m_ev  = mn_f.select<4, 2>(0);
    m_od  = mn_f.select<4, 2>(1);
}

static SYCL_ESIMD_FUNCTION inline float
eval_row_k4_dp4a(const esimd::simd<int32_t, 32> & zero32,
                 const esimd::simd<int32_t, 32> & v_lo, const esimd::simd<int32_t, 32> & v_hi,
                 const esimd::simd<int32_t, 32> & u_ev, const esimd::simd<int32_t, 32> & u_od,
                 const esimd::simd<float, 4> & d8_ev,   const esimd::simd<float, 4> & d8_od,
                 const esimd::simd<float, 4> & s2_ev,   const esimd::simd<float, 4> & s2_od,
                 const esimd::simd<float, 4> & sc_e,    const esimd::simd<float, 4> & sc_o,
                 const esimd::simd<float, 4> & m_e,     const esimd::simd<float, 4> & m_o) {
    const esimd::simd<float, 4> s1_ev = d8_ev * reduce32_to_4f(esimd::dp4a<int32_t>(zero32, v_lo, u_ev));
    const esimd::simd<float, 4> s1_od = d8_od * reduce32_to_4f(esimd::dp4a<int32_t>(zero32, v_hi, u_od));
    esimd::simd<float, 4> tot = (sc_e * s1_ev - m_e * s2_ev) + (sc_o * s1_od - m_o * s2_od);
    esimd::simd<float, 2> t2 = tot.select<2, 1>(0) + tot.select<2, 1>(2);
    return float(t2[0]) + float(t2[1]);
}

// Shared zero-division slot deduplication helper (returns 0 if this slot is handled by an earlier leader)
template <int MAX_M>
static SYCL_ESIMD_FUNCTION inline int
dedup_moe_slots(const int32_t * ids_dev, const int t0, const int k0, const int i02,
                const int n_tokens, const int n_experts_used, const size_t ids_row_stride,
                const size_t src1_tok_stride, const size_t src1_row_stride,
                const size_t dst_tok_stride,  const size_t dst_row_stride,
                size_t (&vy_off)[MAX_M], size_t (&dst_off)[MAX_M]) {
    int cnt = 1;
    vy_off[0]  = size_t(t0) * src1_tok_stride + size_t(k0) * src1_row_stride;
    dst_off[0] = size_t(t0) * dst_tok_stride  + size_t(k0) * dst_row_stride;

    if (n_tokens > 1 || n_experts_used > 1) {
        int rank = 0;
        for (int t = 0; t <= t0; ++t) {
            const int32_t * row_ids = ids_dev + size_t(t) * ids_row_stride;
            const int k_end = (t < t0) ? n_experts_used : k0;
            for (int k = 0; k < k_end; ++k) {
                if (row_ids[k] == i02) ++rank;
            }
        }
        if ((rank & (MAX_M - 1)) != 0) return 0;

        for (int t = t0; t < n_tokens && cnt < MAX_M; ++t) {
            const int32_t * row_ids = ids_dev + size_t(t) * ids_row_stride;
            const int k_start = (t == t0) ? (k0 + 1) : 0;
            for (int k = k_start; k < n_experts_used && cnt < MAX_M; ++k) {
                if (row_ids[k] == i02) {
                    vy_off[cnt]  = size_t(t) * src1_tok_stride + size_t(k) * src1_row_stride;
                    dst_off[cnt] = size_t(t) * dst_tok_stride  + size_t(k) * dst_row_stride;
                    ++cnt;
                }
            }
        }
    }
    return cnt;
}

template <int RI>
static SYCL_ESIMD_FUNCTION inline void
store_zero_rows(float * dst_s, const int r0, const int nrows) {
    if (r0 + RI <= nrows) {
        esimd::simd<float, RI> z(0.0f);
        esimd::block_store<float, RI>(dst_s + r0, z);
    } else {
        #pragma unroll
        for (int r = 0; r < RI; ++r) {
            if (r0 + r < nrows) dst_s[r0 + r] = 0.0f;
        }
    }
}

template <int RI, int MAX_M>
static SYCL_ESIMD_FUNCTION inline void
store_acc_rows(float * dst_base, const size_t (&dst_off)[MAX_M],
               esimd::simd<float, MAX_M> (&acc)[RI], const int cnt, const int r0, const int nrows) {
    for (int i = 0; i < cnt; ++i) {
        float * dst_s = reinterpret_cast<float *>(reinterpret_cast<char *>(dst_base) + dst_off[i]);
        if (r0 + RI <= nrows) {
            esimd::simd<float, RI> v_out;
            #pragma unroll
            for (int r = 0; r < RI; ++r) v_out[r] = float(acc[r][i]);
            esimd::block_store<float, RI>(dst_s + r0, v_out);
        } else {
            #pragma unroll
            for (int r = 0; r < RI; ++r) {
                if (r0 + r < nrows) dst_s[r0 + r] = float(acc[r][i]);
            }
        }
    }
}

// ============================================================================
// 1. Q4_K & Q5_K SoA MoE ESIMD `dp4a (32|M0)` Kernel (IS_Q5K = false / true)
// ============================================================================
template <bool IS_Q5K, int RI, size_t WG_THRS>
static void launch_moe_k4_k5_reorder_esimd_impl(
    const void *    vx_base,
    const void *    vy_base,
    const int32_t * ids_dev,
    float *         dst_base,
    const int       ncols,
    const int       nrows,
    const int       n_experts_used,
    const size_t    expert_weight_stride,
    const size_t    dst_row_stride,
    const size_t    src1_row_stride,
    dpct::queue_ptr stream,
    const int       n_tokens,
    const size_t    ids_row_stride,
    const size_t    dst_tok_stride,
    const size_t    src1_tok_stride) {

    constexpr int MAX_M = 8;
    const int    bpr         = ncols / QK_K;
    const size_t nblocks     = size_t(nrows) * size_t(bpr);
    const size_t rblocks     = size_t((nrows + RI - 1) / RI);
    const size_t wg_per_slot = (rblocks + WG_THRS - 1) / WG_THRS;

    const sycl::range<3> g_range(size_t(n_tokens), size_t(n_experts_used), wg_per_slot * WG_THRS);
    const sycl::range<3> l_range(1, 1, WG_THRS);

    stream->submit([&](sycl::handler & h) {
        h.parallel_for(
            sycl::nd_range<3>(g_range, l_range),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                const int    t0 = int(item.get_group(0));
                const int    k0 = int(item.get_group(1));
                const size_t rb = item.get_group(2) * WG_THRS + item.get_local_id(2);
                if (rb >= rblocks) return;

                const int i02 = ids_dev[size_t(t0) * ids_row_stride + size_t(k0)];
                const int r0  = int(rb) * RI;

                if (i02 < 0) {
                    float * dst_s = reinterpret_cast<float *>(
                        reinterpret_cast<char *>(dst_base) +
                        size_t(t0) * dst_tok_stride + size_t(k0) * dst_row_stride);
                    store_zero_rows<RI>(dst_s, r0, nrows);
                    return;
                }

                size_t vy_off[MAX_M], dst_off[MAX_M];
                const int cnt = dedup_moe_slots<MAX_M>(
                    ids_dev, t0, k0, i02, n_tokens, n_experts_used, ids_row_stride,
                    src1_tok_stride, src1_row_stride, dst_tok_stride, dst_row_stride, vy_off, dst_off);
                if (cnt == 0) return;

                const uint8_t *  vx          = reinterpret_cast<const uint8_t *>(vx_base) + size_t(i02) * expert_weight_stride;
                const uint8_t *  qs_base     = vx;
                const uint8_t *  qh_base     = IS_Q5K ? (vx + nblocks * (QK_K / 2)) : nullptr;
                const size_t     scales_off  = IS_Q5K ? (nblocks * (QK_K / 2 + QK_K / 8)) : (nblocks * (QK_K / 2));
                const uint8_t *  scales_base = vx + scales_off;
                const uint32_t * dm_base     = reinterpret_cast<const uint32_t *>(scales_base + nblocks * K_SCALE_SIZE);

                esimd::simd<float, MAX_M> acc[RI];
                #pragma unroll
                for (int r = 0; r < RI; ++r) acc[r] = 0.0f;

                const esimd::simd<int32_t, 32> zero32(0);
                const esimd::simd<int32_t, 32> ones32(0x01010101);

                for (int ib = 0; ib < bpr; ++ib) {
                    esimd::simd<int32_t, 32> v_lo[RI], v_hi[RI];
                    esimd::simd<float, 4> sc_ev[RI], sc_od[RI], m_ev[RI], m_od[RI];

                    #pragma unroll
                    for (int r = 0; r < RI; ++r) {
                        const int    rr  = (r0 + r < nrows) ? (r0 + r) : r0;
                        const size_t ibx = size_t(rr) * size_t(bpr) + size_t(ib);
                        const esimd::simd<int32_t, 32> qv = esimd::block_load<int32_t, 32>(
                            reinterpret_cast<const int32_t *>(qs_base + ibx * (QK_K / 2)));
                        v_lo[r] = qv & 0x0F0F0F0F;
                        v_hi[r] = (qv >> 4) & 0x0F0F0F0F;

                        if constexpr (IS_Q5K) {
                            const esimd::simd<int32_t, 8> qh = esimd::block_load<int32_t, 8>(
                                reinterpret_cast<const int32_t *>(qh_base + ibx * (QK_K / 8)));
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const esimd::simd<int32_t, 8> h_ev = ((qh >> (2 * j)) << 4) & 0x10101010;
                                const esimd::simd<int32_t, 8> h_od = ((qh >> (2 * j + 1)) << 4) & 0x10101010;
                                v_lo[r].template select<8, 1>(8 * j) |= h_ev;
                                v_hi[r].template select<8, 1>(8 * j) |= h_od;
                            }
                        }

                        decode_soa_k4_scales(scales_base + ibx * K_SCALE_SIZE, dm_base[ibx],
                                             sc_ev[r], sc_od[r], m_ev[r], m_od[r]);
                    }

                    for (int i = 0; i < cnt; ++i) {
                        const char * vy = reinterpret_cast<const char *>(vy_base) + vy_off[i];
                        esimd::simd<int32_t, 64> raw = esimd::block_load<int32_t, 64>(
                            reinterpret_cast<const int32_t *>(vy + size_t(ib) * QK_K));
                        esimd::simd<uint32_t, 8> ds_raw = esimd::block_load<uint32_t, 8>(
                            reinterpret_cast<const uint32_t *>(vy + ncols) + size_t(ib) * 8);

                        auto ds_h = ds_raw.bit_cast_view<sycl::half>();
                        const esimd::simd<float, 4> d8_ev = esimd::simd<sycl::half, 4>(ds_h.select<4, 4>(0));
                        const esimd::simd<float, 4> d8_od = esimd::simd<sycl::half, 4>(ds_h.select<4, 4>(2));

                        esimd::simd<int32_t, 32> u_ev, u_od;
                        u_ev.select<8, 1>(0)  = raw.select<8, 1>(0);
                        u_ev.select<8, 1>(8)  = raw.select<8, 1>(16);
                        u_ev.select<8, 1>(16) = raw.select<8, 1>(32);
                        u_ev.select<8, 1>(24) = raw.select<8, 1>(48);
                        u_od.select<8, 1>(0)  = raw.select<8, 1>(8);
                        u_od.select<8, 1>(8)  = raw.select<8, 1>(24);
                        u_od.select<8, 1>(16) = raw.select<8, 1>(40);
                        u_od.select<8, 1>(24) = raw.select<8, 1>(56);

                        const esimd::simd<float, 4> s2_ev = d8_ev * reduce32_to_4f(esimd::dp4a<int32_t>(zero32, ones32, u_ev));
                        const esimd::simd<float, 4> s2_od = d8_od * reduce32_to_4f(esimd::dp4a<int32_t>(zero32, ones32, u_od));

                        #pragma unroll
                        for (int r = 0; r < RI; ++r) {
                            const float prev = acc[r][i];
                            acc[r][i] = prev + eval_row_k4_dp4a(
                                zero32, v_lo[r], v_hi[r], u_ev, u_od,
                                d8_ev, d8_od, s2_ev, s2_od,
                                sc_ev[r], sc_od[r], m_ev[r], m_od[r]);
                        }
                    }
                }

                store_acc_rows<RI, MAX_M>(dst_base, dst_off, acc, cnt, r0, nrows);
            }
        );
    });
}

// ============================================================================
// 2. Q6_K SoA MoE ESIMD `dp4a (32|M0)` Kernel
// Layout: [ql: nblocks*128B] [qh: nblocks*64B] [scales: nblocks*16B (int8)] [d: nblocks*2B (half)]
// ============================================================================
static SYCL_ESIMD_FUNCTION inline esimd::simd<int32_t, 32>
sub32_bytes_32i(esimd::simd<int32_t, 32> v_u) {
    esimd::simd<int8_t, 128> b = v_u.bit_cast_view<int8_t>();
    b = b - int8_t(32);
    return b.bit_cast_view<int32_t>();
}

template <int RI, size_t WG_THRS>
static void launch_moe_q6k_reorder_esimd_impl(
    const void *    vx_base,
    const void *    vy_base,
    const int32_t * ids_dev,
    float *         dst_base,
    const int       ncols,
    const int       nrows,
    const int       n_experts_used,
    const size_t    expert_weight_stride,
    const size_t    dst_row_stride,
    const size_t    src1_row_stride,
    dpct::queue_ptr stream,
    const int       n_tokens,
    const size_t    ids_row_stride,
    const size_t    dst_tok_stride,
    const size_t    src1_tok_stride) {

    constexpr int MAX_M = 8;
    const int    bpr         = ncols / QK_K;
    const size_t nblocks     = size_t(nrows) * size_t(bpr);
    const size_t rblocks     = size_t((nrows + RI - 1) / RI);
    const size_t wg_per_slot = (rblocks + WG_THRS - 1) / WG_THRS;

    const sycl::range<3> g_range(size_t(n_tokens), size_t(n_experts_used), wg_per_slot * WG_THRS);
    const sycl::range<3> l_range(1, 1, WG_THRS);

    stream->submit([&](sycl::handler & h) {
        h.parallel_for(
            sycl::nd_range<3>(g_range, l_range),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                const int    t0 = int(item.get_group(0));
                const int    k0 = int(item.get_group(1));
                const size_t rb = item.get_group(2) * WG_THRS + item.get_local_id(2);
                if (rb >= rblocks) return;

                const int i02 = ids_dev[size_t(t0) * ids_row_stride + size_t(k0)];
                const int r0  = int(rb) * RI;

                if (i02 < 0) {
                    float * dst_s = reinterpret_cast<float *>(
                        reinterpret_cast<char *>(dst_base) +
                        size_t(t0) * dst_tok_stride + size_t(k0) * dst_row_stride);
                    store_zero_rows<RI>(dst_s, r0, nrows);
                    return;
                }

                size_t vy_off[MAX_M], dst_off[MAX_M];
                const int cnt = dedup_moe_slots<MAX_M>(
                    ids_dev, t0, k0, i02, n_tokens, n_experts_used, ids_row_stride,
                    src1_tok_stride, src1_row_stride, dst_tok_stride, dst_row_stride, vy_off, dst_off);
                if (cnt == 0) return;

                const uint8_t *    vx          = reinterpret_cast<const uint8_t *>(vx_base) + size_t(i02) * expert_weight_stride;
                const uint8_t *    ql_base     = vx;
                const uint8_t *    qh_base     = vx + nblocks * (QK_K / 2);
                const int8_t *     scales_base = reinterpret_cast<const int8_t *>(vx + nblocks * (QK_K / 2 + QK_K / 4));
                const sycl::half * d_base      = reinterpret_cast<const sycl::half *>(scales_base + nblocks * (QK_K / 16));

                esimd::simd<float, MAX_M> acc[RI];
                #pragma unroll
                for (int r = 0; r < RI; ++r) acc[r] = 0.0f;

                const esimd::simd<int32_t, 32> zero32(0);

                for (int ib = 0; ib < bpr; ++ib) {
                    // Unpack all 256 signed int8 weights for each row into 2 x 32-int32 vectors:
                    // v_lo = [half0 q8_0, half0 q8_1, half1 q8_4, half1 q8_5]
                    // v_hi = [half0 q8_2, half0 q8_3, half1 q8_6, half1 q8_7]
                    esimd::simd<int32_t, 32> v_lo[RI], v_hi[RI];
                    esimd::simd<float, 8> sc_lo[RI], sc_hi[RI];

                    #pragma unroll
                    for (int r = 0; r < RI; ++r) {
                        const int    rr  = (r0 + r < nrows) ? (r0 + r) : r0;
                        const size_t ibx = size_t(rr) * size_t(bpr) + size_t(ib);

                        esimd::simd<int32_t, 32> ql = esimd::block_load<int32_t, 32>(
                            reinterpret_cast<const int32_t *>(ql_base + ibx * (QK_K / 2)));
                        esimd::simd<int32_t, 16> qh = esimd::block_load<int32_t, 16>(
                            reinterpret_cast<const int32_t *>(qh_base + ibx * (QK_K / 4)));
                        const esimd::simd<int8_t, 16> sc_i8 = esimd::block_load<int8_t, 16>(
                            scales_base + ibx * (QK_K / 16));
                        const float d = float(d_base[ibx]);

                        esimd::simd<int32_t, 32> u_lo = ql & 0x0F0F0F0F;
                        esimd::simd<int32_t, 32> u_hi = (ql >> 4) & 0x0F0F0F0F;

                        const esimd::simd<int32_t, 8> qh0 = qh.select<8, 1>(0);
                        const esimd::simd<int32_t, 8> qh1 = qh.select<8, 1>(8);

                        u_lo.select<8, 1>(0)  |= ((qh0 << 4) & 0x30303030);
                        u_lo.select<8, 1>(8)  |= ((qh0 << 2) & 0x30303030);
                        u_lo.select<8, 1>(16) |= ((qh1 << 4) & 0x30303030);
                        u_lo.select<8, 1>(24) |= ((qh1 << 2) & 0x30303030);

                        u_hi.select<8, 1>(0)  |= ((qh0 >> 0) & 0x30303030);
                        u_hi.select<8, 1>(8)  |= ((qh0 >> 2) & 0x30303030);
                        u_hi.select<8, 1>(16) |= ((qh1 >> 0) & 0x30303030);
                        u_hi.select<8, 1>(24) |= ((qh1 >> 2) & 0x30303030);

                        v_lo[r] = sub32_bytes_32i(u_lo);
                        v_hi[r] = sub32_bytes_32i(u_hi);

                        esimd::simd<float, 16> sc_f = esimd::simd<float, 16>(sc_i8) * d;
                        sc_lo[r].template select<4, 1>(0) = sc_f.select<4, 1>(0);
                        sc_lo[r].template select<4, 1>(4) = sc_f.select<4, 1>(8);
                        sc_hi[r].template select<4, 1>(0) = sc_f.select<4, 1>(4);
                        sc_hi[r].template select<4, 1>(4) = sc_f.select<4, 1>(12);
                    }

                    for (int i = 0; i < cnt; ++i) {
                        const char * vy = reinterpret_cast<const char *>(vy_base) + vy_off[i];
                        esimd::simd<int32_t, 64> raw = esimd::block_load<int32_t, 64>(
                            reinterpret_cast<const int32_t *>(vy + size_t(ib) * QK_K));
                        esimd::simd<uint32_t, 8> ds_raw = esimd::block_load<uint32_t, 8>(
                            reinterpret_cast<const uint32_t *>(vy + ncols) + size_t(ib) * 8);

                        auto ds_h = ds_raw.bit_cast_view<sycl::half>();
                        esimd::simd<float, 8> d8_all = esimd::simd<sycl::half, 8>(ds_h.select<8, 2>(0));

                        // Duplicate each 32-element d8 across its two 16-element Q6_K sub-blocks
                        esimd::simd<float, 8> d8_lo, d8_hi;
                        d8_lo[0] = d8_all[0]; d8_lo[1] = d8_all[0];
                        d8_lo[2] = d8_all[1]; d8_lo[3] = d8_all[1];
                        d8_lo[4] = d8_all[4]; d8_lo[5] = d8_all[4];
                        d8_lo[6] = d8_all[5]; d8_lo[7] = d8_all[5];

                        d8_hi[0] = d8_all[2]; d8_hi[1] = d8_all[2];
                        d8_hi[2] = d8_all[3]; d8_hi[3] = d8_all[3];
                        d8_hi[4] = d8_all[6]; d8_hi[5] = d8_all[6];
                        d8_hi[6] = d8_all[7]; d8_hi[7] = d8_all[7];

                        esimd::simd<int32_t, 32> a_lo, a_hi;
                        a_lo.select<16, 1>(0)  = raw.select<16, 1>(0);
                        a_lo.select<16, 1>(16) = raw.select<16, 1>(32);
                        a_hi.select<16, 1>(0)  = raw.select<16, 1>(16);
                        a_hi.select<16, 1>(16) = raw.select<16, 1>(48);

                        #pragma unroll
                        for (int r = 0; r < RI; ++r) {
                            const esimd::simd<float, 8> p_lo = d8_lo * sc_lo[r] * reduce32_to_8f(esimd::dp4a<int32_t>(zero32, v_lo[r], a_lo));
                            const esimd::simd<float, 8> p_hi = d8_hi * sc_hi[r] * reduce32_to_8f(esimd::dp4a<int32_t>(zero32, v_hi[r], a_hi));
                            esimd::simd<float, 8> s8 = p_lo + p_hi;
                            esimd::simd<float, 4> s4 = s8.select<4, 1>(0) + s8.select<4, 1>(4);
                            esimd::simd<float, 2> s2 = s4.select<2, 1>(0) + s4.select<2, 1>(2);
                            const float prev = acc[r][i];
                            acc[r][i] = prev + (float(s2[0]) + float(s2[1]));
                        }
                    }
                }

                store_acc_rows<RI, MAX_M>(dst_base, dst_off, acc, cnt, r0, nrows);
            }
        );
    });
}

// ============================================================================
// 3. Q8_0 SoA MoE ESIMD `dp4a (32|M0)` Kernel
// Layout: [qs: nblocks*32B (int8)] [d: nblocks*2B (half)], bpr = ncols / 32
// Processes 4 Q8_0 blocks (128 weights) per 32-wide hardware `dp4a (32|M0)`!
// ============================================================================
template <int RI, size_t WG_THRS>
static void launch_moe_q8_0_reorder_esimd_impl(
    const void *    vx_base,
    const void *    vy_base,
    const int32_t * ids_dev,
    float *         dst_base,
    const int       ncols,
    const int       nrows,
    const int       n_experts_used,
    const size_t    expert_weight_stride,
    const size_t    dst_row_stride,
    const size_t    src1_row_stride,
    dpct::queue_ptr stream,
    const int       n_tokens,
    const size_t    ids_row_stride,
    const size_t    dst_tok_stride,
    const size_t    src1_tok_stride) {

    constexpr int MAX_M = 8;
    const int    bpr         = ncols / QK8_0;
    const size_t rblocks     = size_t((nrows + RI - 1) / RI);
    const size_t wg_per_slot = (rblocks + WG_THRS - 1) / WG_THRS;

    const sycl::range<3> g_range(size_t(n_tokens), size_t(n_experts_used), wg_per_slot * WG_THRS);
    const sycl::range<3> l_range(1, 1, WG_THRS);

    stream->submit([&](sycl::handler & h) {
        h.parallel_for(
            sycl::nd_range<3>(g_range, l_range),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                const int    t0 = int(item.get_group(0));
                const int    k0 = int(item.get_group(1));
                const size_t rb = item.get_group(2) * WG_THRS + item.get_local_id(2);
                if (rb >= rblocks) return;

                const int i02 = ids_dev[size_t(t0) * ids_row_stride + size_t(k0)];
                const int r0  = int(rb) * RI;

                if (i02 < 0) {
                    float * dst_s = reinterpret_cast<float *>(
                        reinterpret_cast<char *>(dst_base) +
                        size_t(t0) * dst_tok_stride + size_t(k0) * dst_row_stride);
                    store_zero_rows<RI>(dst_s, r0, nrows);
                    return;
                }

                size_t vy_off[MAX_M], dst_off[MAX_M];
                const int cnt = dedup_moe_slots<MAX_M>(
                    ids_dev, t0, k0, i02, n_tokens, n_experts_used, ids_row_stride,
                    src1_tok_stride, src1_row_stride, dst_tok_stride, dst_row_stride, vy_off, dst_off);
                if (cnt == 0) return;

                const uint8_t *    vx      = reinterpret_cast<const uint8_t *>(vx_base) + size_t(i02) * expert_weight_stride;
                const uint8_t *    qs_base = vx;
                const sycl::half * d_base  = reinterpret_cast<const sycl::half *>(vx + size_t(nrows) * size_t(ncols));

                esimd::simd<float, MAX_M> acc[RI];
                #pragma unroll
                for (int r = 0; r < RI; ++r) acc[r] = 0.0f;

                const esimd::simd<int32_t, 32> zero32(0);
                const esimd::simd<int32_t, 8>  zero8(0);

                // Main loop: 4 blocks (128 weights = 32 int32s) per 32-wide dp4a!
                const int bpr4 = bpr & ~3;
                for (int ib = 0; ib < bpr4; ib += 4) {
                    esimd::simd<int32_t, 32> w32[RI];
                    esimd::simd<float, 4> dw4[RI];

                    #pragma unroll
                    for (int r = 0; r < RI; ++r) {
                        const int    rr  = (r0 + r < nrows) ? (r0 + r) : r0;
                        const size_t ibx = size_t(rr) * size_t(bpr) + size_t(ib);
                        w32[r] = esimd::block_load<int32_t, 32>(
                            reinterpret_cast<const int32_t *>(qs_base + ibx * QK8_0));
                        esimd::simd<sycl::half, 4> dh = esimd::block_load<sycl::half, 4>(d_base + ibx);
                        dw4[r] = esimd::simd<float, 4>(dh);
                    }

                    for (int i = 0; i < cnt; ++i) {
                        const char * vy = reinterpret_cast<const char *>(vy_base) + vy_off[i];
                        const esimd::simd<int32_t, 32> a32 = esimd::block_load<int32_t, 32>(
                            reinterpret_cast<const int32_t *>(vy + size_t(ib) * QK8_0));
                        esimd::simd<uint32_t, 4> ds_raw = esimd::block_load<uint32_t, 4>(
                            reinterpret_cast<const uint32_t *>(vy + ncols) + size_t(ib));
                        auto ds_h = ds_raw.bit_cast_view<sycl::half>();
                        const esimd::simd<float, 4> d8_4 = esimd::simd<sycl::half, 4>(ds_h.select<4, 2>(0));

                        #pragma unroll
                        for (int r = 0; r < RI; ++r) {
                            esimd::simd<float, 4> prod = dw4[r] * d8_4 * reduce32_to_4f(esimd::dp4a<int32_t>(zero32, w32[r], a32));
                            esimd::simd<float, 2> s2   = prod.select<2, 1>(0) + prod.select<2, 1>(2);
                            const float prev = acc[r][i];
                            acc[r][i] = prev + (float(s2[0]) + float(s2[1]));
                        }
                    }
                }

                // Tail loop for bpr % 4 != 0 (e.g. Raon-1-mini ncols=1088 -> bpr=34 -> 2 tail blocks)
                for (int ib = bpr4; ib < bpr; ++ib) {
                    esimd::simd<int32_t, 8> w8[RI];
                    float dw[RI];

                    #pragma unroll
                    for (int r = 0; r < RI; ++r) {
                        const int    rr  = (r0 + r < nrows) ? (r0 + r) : r0;
                        const size_t ibx = size_t(rr) * size_t(bpr) + size_t(ib);
                        w8[r] = esimd::block_load<int32_t, 8>(
                            reinterpret_cast<const int32_t *>(qs_base + ibx * QK8_0));
                        dw[r] = float(d_base[ibx]);
                    }

                    for (int i = 0; i < cnt; ++i) {
                        const char * vy = reinterpret_cast<const char *>(vy_base) + vy_off[i];
                        const esimd::simd<int32_t, 8> a8 = esimd::block_load<int32_t, 8>(
                            reinterpret_cast<const int32_t *>(vy + size_t(ib) * QK8_0));
                        const uint32_t ds_u = *(reinterpret_cast<const uint32_t *>(vy + ncols) + size_t(ib));
                        esimd::simd<uint32_t, 1> ds_v(ds_u);
                        auto ds_h = ds_v.bit_cast_view<sycl::half>();
                        const float d8 = float(sycl::half(ds_h[0]));

                        #pragma unroll
                        for (int r = 0; r < RI; ++r) {
                            const float dot = reduce8_to_f(esimd::dp4a<int32_t>(zero8, w8[r], a8));
                            const float prev = acc[r][i];
                            acc[r][i] = prev + dw[r] * d8 * dot;
                        }
                    }
                }

                store_acc_rows<RI, MAX_M>(dst_base, dst_off, acc, cnt, r0, nrows);
            }
        );
    });
}

// Unified runtime dispatcher for Q4_K / Q5_K / Q6_K / Q8_0 SoA MoE ESIMD dp4a
static inline bool launch_moe_reorder_esimd(
    enum ggml_type  src0_type,
    const void *    vx_base,
    const void *    vy_base,
    const int32_t * ids_dev,
    float *         dst_base,
    const int       ncols,
    const int       nrows,
    const int       n_experts_used,
    const size_t    expert_weight_stride,
    const size_t    dst_row_stride,
    const size_t    src1_row_stride,
    dpct::queue_ptr stream,
    const int       n_tokens,
    const size_t    ids_row_stride,
    const size_t    dst_tok_stride,
    const size_t    src1_tok_stride) {

    static const int mode = ggml_sycl_get_env("GGML_SYCL_MOE_Q4K_ESIMD", 1);
    if (mode == 0 || ncols <= 0 || nrows <= 0 || n_tokens <= 0 || n_experts_used <= 0) {
        return false;
    }

    // Auto RI: RI=4 for n_tokens <= 4 (higher EU occupancy on decode/small verify), RI=8 for n_tokens > 4
    const bool use_ri8 = (mode == 3) || (mode == 1 && n_tokens > 4);

    switch (src0_type) {
        case GGML_TYPE_Q4_K:
            if ((ncols % QK_K) != 0) return false;
            if (use_ri8) {
                launch_moe_k4_k5_reorder_esimd_impl<false, 8, 16>(
                    vx_base, vy_base, ids_dev, dst_base, ncols, nrows, n_experts_used,
                    expert_weight_stride, dst_row_stride, src1_row_stride, stream,
                    n_tokens, ids_row_stride, dst_tok_stride, src1_tok_stride);
            } else {
                launch_moe_k4_k5_reorder_esimd_impl<false, 4, 16>(
                    vx_base, vy_base, ids_dev, dst_base, ncols, nrows, n_experts_used,
                    expert_weight_stride, dst_row_stride, src1_row_stride, stream,
                    n_tokens, ids_row_stride, dst_tok_stride, src1_tok_stride);
            }
            return true;

        case GGML_TYPE_Q5_K:
            if ((ncols % QK_K) != 0) return false;
            if (use_ri8) {
                launch_moe_k4_k5_reorder_esimd_impl<true, 8, 16>(
                    vx_base, vy_base, ids_dev, dst_base, ncols, nrows, n_experts_used,
                    expert_weight_stride, dst_row_stride, src1_row_stride, stream,
                    n_tokens, ids_row_stride, dst_tok_stride, src1_tok_stride);
            } else {
                launch_moe_k4_k5_reorder_esimd_impl<true, 4, 16>(
                    vx_base, vy_base, ids_dev, dst_base, ncols, nrows, n_experts_used,
                    expert_weight_stride, dst_row_stride, src1_row_stride, stream,
                    n_tokens, ids_row_stride, dst_tok_stride, src1_tok_stride);
            }
            return true;

        case GGML_TYPE_Q6_K:
            if ((ncols % QK_K) != 0) return false;
            // Q6_K holds 2x 32-int32 weights + 2x 8-float scales per row -> RI=4 keeps 0B spill at 128 GRFs
            launch_moe_q6k_reorder_esimd_impl<4, 16>(
                vx_base, vy_base, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream,
                n_tokens, ids_row_stride, dst_tok_stride, src1_tok_stride);
            return true;

        case GGML_TYPE_Q8_0:
            if ((ncols % QK8_0) != 0) return false;
            if (use_ri8) {
                launch_moe_q8_0_reorder_esimd_impl<8, 16>(
                    vx_base, vy_base, ids_dev, dst_base, ncols, nrows, n_experts_used,
                    expert_weight_stride, dst_row_stride, src1_row_stride, stream,
                    n_tokens, ids_row_stride, dst_tok_stride, src1_tok_stride);
            } else {
                launch_moe_q8_0_reorder_esimd_impl<4, 16>(
                    vx_base, vy_base, ids_dev, dst_base, ncols, nrows, n_experts_used,
                    expert_weight_stride, dst_row_stride, src1_row_stride, stream,
                    n_tokens, ids_row_stride, dst_tok_stride, src1_tok_stride);
            }
            return true;

        default:
            return false;
    }
}

} // namespace ggml_sycl_moe_esimd

#endif // GGML_SYCL_MOE_Q4K_ESIMD_HPP
