#include "fattn-xmx-dec.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/math.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstring>

#include "fattn-common.hpp"

// Decode flash attention on XMX over a q8_0 cache read in place, or an f16 cache: the decode path of the fork's fattn-xmx.cpp
// (sycl-decode-mmvq; Recursive Flint, Qwen3.8-27B), with the indexed bias of the Inkling 2 models added at the mask
// read. A work-group of D / 16 threads handles one KV head and one slice of the cache (split-K): per block of D tokens,
// thread t scores tokens 16t..16t+15 for all g heads x query rows (M <= 64 DPAS rows; the q8_0 blocks are dequantized
// in registers into the VNNI B operand), the row maxima and sums go through SLM, P = exp2(S - max) is stored to SLM
// as f16, and thread t accumulates dims 16t..16t+15 of O += P V. Each slice writes its unnormalized O with the row max
// and sum; a second kernel merges the slices into dst (and the sinks, if any).

namespace {

constexpr int FA_XMX_D  = 256;
constexpr int FA_XMX_T  = 16;             // threads per work-group; also the 16-dim slices of D
constexpr int FA_XMX_BK = FA_XMX_T * 16;  // KV tokens per block
static_assert(FA_XMX_D == FA_XMX_T * 16, "each thread owns 16 dims of O");

template <typename F> struct fa_xmx_grf256 {
    F f;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL { f(it); }

    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    }
};

// up to 3 row groups fit 128 registers per thread without spills: 8 threads per vector engine instead of 4 hide the
// per-block barriers and loads better (see fa_xmx_impl)
template <typename F> struct fa_xmx_grf128 {
    F f;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL { f(it); }

    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<128> };
    }
};

struct fa_xmx_args {
    const char * Q;
    const char * K;
    const char * V;
    const char * mask;
    const uint8_t * span_live;  // [n_kv / 16]: 0 = the 16-token span is masked for every query row
    // q8_0 cache read in place: per KV head, bytes added to the base (head-major) or dwords added to the 2D x
    // offset (heads interleaved in the token row)
    size_t       k_hb, v_hb;
    int          k_hx, v_hx;
    float *      part_o;   // [n_kvh][nsplit][R8][D]
    float *      part_ml;  // [n_kvh][nsplit][R8][2]: row max, row sum
    int          g;        // query heads per KV head
    int          nq;       // query rows
    int          n_kv;
    int          nsplit;
    int          bps;      // blocks per split
    size_t       q_nb1, q_nb2, k_nb1, k_nb2, v_nb1, v_nb2, m_nb1;
    float        qscale;   // scale * log2(e)
    // row chunks of fa_xmx_list_kernel: chunk c holds query rows [c nq_chunk, (c + 1) nq_chunk) and reads only
    // the spans in its list
    const int32_t * span_list;   // [n_chunks][list_stride]: live 16-token spans of the chunk, ascending
    const int32_t * span_count;  // [n_chunks]
    int             list_stride;
    int             nq_chunk;
    int             n_kvh;
    const uint32_t * mask_bits; // [nq][n_kv / 16]: selected cells in low bits, visible cells in high bits
    float            mask_fill, mask_zero;
    // decode rows of several streams (sequences, ne[3]) in one launch: work-group (stream, KV head, split); bytes
    // between streams (m_nb3 = 0: one mask for all) and span flags between streams
    int              n_str;
    size_t           q_nb3, k_nb3, v_nb3, m_nb3, live_nb3;
    // an indexed bias (fattn-common.hpp fattn_indexed_bias): the table [query][head][bin], the bins from the positions
    // (bpos [stream][bpos_nkv cells, nq_total queries], extent) or from indices [query][bidx_stride]; q_base: the
    // first query row of this launch in its stream
    const float *    btab;
    int              n_bins, n_head, extent, nq_total, q_base;
    const float *    bpos;
    size_t           bpos_row, bpos_nkv;
    const int32_t *  bidx;
    size_t           bidx_stride;
};

// row maxima of an [R][16] block
template <int R>
static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, R> fa_xmx_row_max(
    sycl::ext::intel::esimd::simd<float, R * 16> p) {
    using namespace sycl::ext::intel::esimd;
    auto               p2 = p.template bit_cast_view<float, R, 16>();
    simd<float, R * 8> s8 = max(simd<float, R * 8>(p2.template select<R, 1, 8, 1>(0, 0)),
                                simd<float, R * 8>(p2.template select<R, 1, 8, 1>(0, 8)));
    auto               v8 = s8.template bit_cast_view<float, R, 8>();
    simd<float, R * 4> s4 = max(simd<float, R * 4>(v8.template select<R, 1, 4, 1>(0, 0)),
                                simd<float, R * 4>(v8.template select<R, 1, 4, 1>(0, 4)));
    auto               v4 = s4.template bit_cast_view<float, R, 4>();
    simd<float, R * 2> s2 = max(simd<float, R * 2>(v4.template select<R, 1, 2, 1>(0, 0)),
                                simd<float, R * 2>(v4.template select<R, 1, 2, 1>(0, 2)));
    return max(simd<float, R>(s2.template select<R, 2>(0)), simd<float, R>(s2.template select<R, 2>(1)));
}

// row sums of an [R][16] block
template <int R>
static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, R> fa_xmx_row_sum(
    sycl::ext::intel::esimd::simd<float, R * 16> p) {
    using namespace sycl::ext::intel::esimd;
    auto               p2 = p.template bit_cast_view<float, R, 16>();
    simd<float, R * 8> s8 = p2.template select<R, 1, 8, 1>(0, 0) + p2.template select<R, 1, 8, 1>(0, 8);
    auto               v8 = s8.template bit_cast_view<float, R, 8>();
    simd<float, R * 4> s4 = v8.template select<R, 1, 4, 1>(0, 0) + v8.template select<R, 1, 4, 1>(0, 4);
    auto               v4 = s4.template bit_cast_view<float, R, 4>();
    simd<float, R * 2> s2 = v4.template select<R, 1, 2, 1>(0, 0) + v4.template select<R, 1, 2, 1>(0, 2);
    return s2.template select<R, 2>(0) + s2.template select<R, 2>(1);
}


// SKIP: skip the KV spans that span_live marks dead; without it the kernel reads every span
// Q8: K and V are a q8_0 cache read in place (see the loads below); otherwise f16
// D: the head size (256 or 128). T = D / 16 threads, each owning 16 dims of O and scoring 16 tokens of a block of
// BK = 16 T tokens; KS = D / 16 k-steps of 16 dims per score.
// DK = DV = 128 or 256, or the absorbed MLA shape DK = 576 (512 latent + 64), DV = 512 over an f16 cache whose V is the
// first 512 dims of the K rows (Raon-1 mini, 40 heads on one latent head): DV / 16 = 32 threads, 512-token blocks, the
// last block may be half full (the cache is padded to 256 tokens); its missing tokens score -inf
template <int DK, int DV, int RG, bool SKIP, bool Q8>
static ESIMD_INLINE void fa_xmx_kernel(const fa_xmx_args a, const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    namespace xmx     = sycl::ext::intel::esimd::xmx;
    namespace esimd_x = sycl::ext::intel::experimental::esimd;
    constexpr int R8 = RG * 8;  // rows padded to whole DPAS row groups
    constexpr int  D   = DV;  // O dims, per-thread slices
    constexpr int  T   = DV / 16;
    constexpr int  BK  = T * 16;
    constexpr int  KS  = DK / 16;
    constexpr bool MLA = DK != DV;
    static_assert((DK == DV && (DK == 256 || DK == 128)) || (DK == 576 && DV == 512 && !Q8 && !SKIP),
                  "XMX flash attention: head 256 or 128, or MLA 576/512 over f16");
    // SLM: Q as [KS k-steps][R8 rows][16 dims] f16, P as [T token-steps][R8 rows][16 tokens] f16, then the
    // per-thread row maxima and sums [T][R8] f32. Each 8x16 DPAS A operand is 256 contiguous bytes.
    constexpr uint32_t SLM_Q   = 0;
    constexpr uint32_t SLM_P   = SLM_Q + KS * R8 * 32;
    constexpr uint32_t SLM_MAX = SLM_P + T * R8 * 32;
    constexpr uint32_t SLM_SUM = SLM_MAX + T * R8 * 4;
    constexpr uint32_t SLM_END = SLM_SUM + T * R8 * 4;
    slm_init<SLM_END>();

    const properties a16{ alignment<16> };

    const int t     = it.get_local_id(0);
    const int wg    = it.get_group(0);
    const int split = wg % a.nsplit;
    const int kvh   = (wg / a.nsplit) % a.n_kvh;
    const int str   = wg / (a.nsplit * a.n_kvh);
    const int M     = a.g * a.nq;
    // this work-group's stream: its query rows, cache, mask and span flags
    const char *    Qs = a.Q + (size_t) str * a.q_nb3;
    const char *    Ms = a.mask + (size_t) str * a.m_nb3;
    const uint8_t * Ls = a.span_live + (size_t) str * a.live_nb3;

    // 1. Q rows r = j g + c (query j, head c of this KV head) to SLM, scaled, f16; padding rows are 0
    for (int r = t; r < R8; r += T) {
        simd<float, DK> q = 0.0f;
        if (r < M) {
            const int     j  = r / a.g;
            const int     c  = r - j * a.g;
            const float * qp = (const float *) (Qs + j * a.q_nb1 + (size_t) (kvh * a.g + c) * a.q_nb2);
#pragma unroll
            for (int i = 0; i < DK / 64; ++i) {
                q.template select<64, 1>(64 * i) = block_load<float, 64>(qp + 64 * i, a16);
            }
            q *= a.qscale;
        }
        simd<sycl::half, DK> qh = convert<sycl::half>(q);
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            slm_block_store<sycl::half, 16>(SLM_Q + (ks * R8 + r) * 32, qh.template select<16, 1>(16 * ks));
        }
    }
    barrier();

    // running row max (log2 domain) and sum, identical in every thread; this thread's 16 dims of O
    simd<float, R8>      m = -FLT_MAX;
    simd<float, R8>      l = 0.0f;
    simd<float, R8 * 16> o = 0.0f;

    // head slices of the caches: D f16 per row at the row pitch
    const uint32_t * Kh = (const uint32_t *) (a.K + (size_t) str * a.k_nb3 + (size_t) kvh * (Q8 ? a.k_hb : a.k_nb2));
    const sycl::half * Vh = (const sycl::half *) (a.V + (size_t) str * a.v_nb3 + (size_t) kvh * (Q8 ? a.v_hb : a.v_nb2));
    const unsigned surf_wk = DK * sizeof(sycl::half) - 1;
    const unsigned surf_wv = DV * sizeof(sycl::half) - 1;
    const unsigned surf_h  = (unsigned) a.n_kv - 1;
    esimd_x::config_2d_mem_access<uint32_t, 8, 16, 1>    k_desc(Kh, Q8 ? (unsigned) a.k_nb1 - 1 : surf_wk, surf_h,
                                                                (unsigned) a.k_nb1 - 1, 0, 0);
    esimd_x::config_2d_mem_access<sycl::half, 16, 16, 1> v_desc(Vh, surf_wv, surf_h, (unsigned) a.v_nb1 - 1, t * 16, 0);

    // q8_0 in place: a head row is 8 blocks of 34 bytes (a 2-byte f16 scale, 32 int8), loaded as dwords at
    // dword-aligned offsets and dequantized in f16 (int8 converts exactly, the product is rounded once), the same
    // values as the f16 copy of the cache. K: a pair of
    // blocks (64 dims) is 17 dwords, loaded transposed, so a dim pair of 16 tokens is a strided byte region of one
    // dword row. V: dims 16t..16t+15 are half (t & 1) of block j = t / 2; one [32 tokens][16 dwords] load per two
    // token steps from the dword holding the block's scale (its low half for an even block, high half for an odd
    // one) covers the scale and the 16 values, which start at byte 2 or 18 (even block) or 4 or 20 (odd block)
    const int k_x  = kvh * a.k_hx;
    const int vsel = t & 3;
    esimd_x::config_2d_mem_access<uint32_t, 1, 16, 1>  k1_desc(Kh, (unsigned) a.k_nb1 - 1, surf_h, (unsigned) a.k_nb1 - 1, 0, 0);
    esimd_x::config_2d_mem_access<uint32_t, 16, 32, 1> vq_desc((const uint32_t *) Vh, (unsigned) a.v_nb1 - 1, surf_h,
                                                               (unsigned) a.v_nb1 - 1, kvh * a.v_hx + 34 * (t >> 1) / 4, 0);

    const int blk0 = split * a.bps;
    const int nblk = (a.n_kv + BK - 1) / BK;
    const int blk1 = blk0 + a.bps < nblk ? blk0 + a.bps : nblk;
    for (int blk = blk0; blk < blk1; ++blk) {
        const int tok0   = blk * BK;
        const int my_tok = tok0 + t * 16;

        // masked spans add exactly 0 after the softmax, so their K and V are not read; every thread sees the same
        // flags, so a block without live spans is skipped by all of them
        simd<uint8_t, T> live = 1;
        if constexpr (SKIP) {
            live = block_load<uint8_t, T>(Ls + blk * T, properties{ alignment<T> });
            if (!(live != 0).any()) {
                continue;
            }
        }

        // 2. S[r][n] = Q[r] . K[my_tok + n] for this thread's 16 tokens
        simd<float, R8 * 16> s = 0.0f;
        auto score_ks = [&](int ks, const simd<sycl::half, 256> & b) {
#pragma unroll
            for (int rg = 0; rg < RG; ++rg) {
                simd<sycl::half, 128> qa = slm_block_load<sycl::half, 128>(SLM_Q + (ks * R8 + rg * 8) * 32);
                simd<float, 128>      c  = s.template select<128, 1>(rg * 128);
                s.template select<128, 1>(rg * 128) = xmx::dpas<8, 8, float, float>(c, b, qa);
            }
        };
        auto score = [&]() {
            k_desc.set_y(my_tok);
            if constexpr (Q8) {
                k1_desc.set_y(my_tok);
#pragma unroll
                for (int P = 0; P < D / 64; ++P) {
                    // [17 dwords][16 tokens]: block 2P = bytes 0..33, block 2P + 1 = bytes 34..67
                    simd<uint32_t, 17 * 16> w;
                    k_desc.set_x(k_x + 17 * P);
                    w.template select<128, 1>(0) = esimd_x::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(k_desc);
                    k_desc.set_x(k_x + 17 * P + 8);
                    w.template select<128, 1>(128) = esimd_x::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(k_desc);
                    k1_desc.set_x(k_x + 17 * P + 16);
                    w.template select<16, 1>(256) = esimd_x::lsc_load_2d<uint32_t, 1, 16, 1, true, false>(k1_desc);
                    auto                  wh = w.template bit_cast_view<sycl::half>();
                    auto                  wb = w.template bit_cast_view<int8_t, 17 * 16, 4>();
                    const simd<sycl::half, 16> d0 = wh.template select<16, 2>(0);
                    const simd<sycl::half, 16> d1 = wh.template select<16, 2>(257);
                    const simd<sycl::half, 32> d0r = d0.template replicate_vs_w_hs<16, 1, 2, 0>(0);
                    const simd<sycl::half, 32> d1r = d1.template replicate_vs_w_hs<16, 1, 2, 0>(0);
#pragma unroll
                    for (int s4 = 0; s4 < 4; ++s4) {
                        simd<sycl::half, 256> b;
#pragma unroll
                        for (int kp = 0; kp < 8; ++kp) {
                            // dim pair p of the block: bytes 2 + 2p (block 2P) or 36 + 2p (block 2P + 1)
                            const int p    = (s4 & 1) * 8 + kp;
                            const int byte = s4 < 2 ? 2 + 2 * p : 36 + 2 * p;
                            simd<int8_t, 32> qv = wb.template select<16, 1, 2, 1>((byte / 4) * 16, byte % 4);
                            b.template select<32, 1>(32 * kp) = convert<sycl::half>(qv) * (s4 < 2 ? d0r : d1r);
                        }
                        score_ks(4 * P + s4, b);
                    }
                }
            } else {
#pragma unroll
                for (int ks = 0; ks < KS; ++ks) {
                    k_desc.set_x(ks * 8);
                    simd<uint32_t, 128> kb = esimd_x::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(k_desc);
                    score_ks(ks, kb.template bit_cast_view<sycl::half>().read());
                }
            }
        };
        if constexpr (SKIP) {
            if (live[t]) {
                score();
            }
        } else {
            score();
        }

        // 3. mask (log2 domain), per query row j; an indexed bias joins the mask before its f16 rounding. MLA: the
        // tokens past the cache (2D loads read them as 0) of a half-full last block score -inf
        const bool past = MLA && my_tok >= a.n_kv;
#pragma unroll
        for (int r = 0; r < R8; ++r) {
            if (past) {
                s.template select<16, 1>(16 * r) = -FLT_MAX;
            } else if (r < M) {
                const int          jq = r / a.g;
                const sycl::half * mp = (const sycl::half *) (Ms + jq * a.m_nb1) + my_tok;
                simd<float, 16>    mk = convert<float>(block_load<sycl::half, 16>(mp, a16));
                if (a.btab) {
                    const size_t      q = (size_t) str * a.nq_total + a.q_base + jq;
                    simd<int32_t, 16> bin;
                    if (a.bpos) {
                        const float *         pr = a.bpos + (size_t) str * a.bpos_row;
                        const simd<float, 16> cp = block_load<float, 16>(pr + my_tok);
                        simd<int32_t, 16>     d  = convert<int32_t>(simd<float, 16>(pr[a.bpos_nkv + a.q_base + jq]) - cp);
                        d.merge(simd<int32_t, 16>(0), d < 0);
                        d.merge(simd<int32_t, 16>(a.extent), d > a.extent);
                        bin = d;
                    } else {
                        bin = block_load<int32_t, 16>(a.bidx + q * a.bidx_stride + my_tok);
                    }
                    const int             h   = kvh * a.g + (r - jq * a.g);
                    const simd_mask<16>   ok  = simd<uint32_t, 16>(bin) < (uint32_t) a.n_bins;
                    simd<uint32_t, 16>    off = simd<uint32_t, 16>(bin) * 4u;
                    off.merge(simd<uint32_t, 16>(0u), !ok);
                    const simd<float, 16> tv  = gather<float, 16>(a.btab + (q * a.n_head + h) * a.n_bins, off);
                    mk.merge(convert<float>(convert<sycl::half>(mk + tv)), ok);
                }
                s.template select<16, 1>(16 * r) += mk * 1.44269504088896341f;
            }
        }

        // 4. block row max over all threads, new running max, P = exp2(S - max)
        slm_block_store<float, R8>(SLM_MAX + t * R8 * 4, fa_xmx_row_max<R8>(s));
        barrier();
        simd<float, R8> bm = -FLT_MAX;
#pragma unroll
        for (int tt = 0; tt < T; ++tt) {
            bm = max(bm, slm_block_load<float, R8>(SLM_MAX + tt * R8 * 4));
        }
        const simd<float, R8> m_new = max(m, bm);
        const simd<float, R8> alpha = exp2(m - m_new);
        m                           = m_new;
#pragma unroll
        for (int r = 0; r < R8; ++r) {
            s.template select<16, 1>(16 * r) = exp2(s.template select<16, 1>(16 * r) - m_new[r]);
        }
        slm_block_store<float, R8>(SLM_SUM + t * R8 * 4, fa_xmx_row_sum<R8>(s));
        simd<sycl::half, R8 * 16> ph = convert<sycl::half>(s);
#pragma unroll
        for (int rg = 0; rg < RG; ++rg) {
            slm_block_store<sycl::half, 128>(SLM_P + (t * R8 + rg * 8) * 32, ph.template select<128, 1>(rg * 128));
        }
        barrier();

        // 5. running sum, rescale O, O += P V for this thread's 16 dims
        simd<float, R8> bs = 0.0f;
#pragma unroll
        for (int tt = 0; tt < T; ++tt) {
            bs += slm_block_load<float, R8>(SLM_SUM + tt * R8 * 4);
        }
        l = l * alpha + bs;
#pragma unroll
        for (int r = 0; r < R8; ++r) {
            o.template select<16, 1>(16 * r) *= alpha[r];
        }
        simd<uint32_t, 32 * 16> vwin;  // q8_0: the V window of token steps ts & ~1
        auto vload = [&](int ts) {
            if constexpr (Q8) {
                vq_desc.set_y(tok0 + ts * 16);
                vwin = esimd_x::lsc_load_2d<uint32_t, 16, 32, 1, false, false>(vq_desc);
            }
        };
        auto pv = [&](int ts) {
            simd<sycl::half, 256> vb;
            if constexpr (Q8) {
                // this step's 16 rows of the window, dequantized as [token][dim], then token pairs interleaved into
                // the VNNI layout
                auto                 vb8 = vwin.template bit_cast_view<int8_t, 32, 64>();
                auto                 vh  = vwin.template bit_cast_view<sycl::half, 32, 32>();
                const int            r0  = (ts & 1) * 16;
                simd<int8_t, 256>    q;
                simd<sycl::half, 16> dh;
                switch (vsel) {
                    case 0:  q = vb8.template select<16, 1, 16, 1>(r0, 2);  dh = vh.template select<16, 1, 1, 1>(r0, 0); break;
                    case 1:  q = vb8.template select<16, 1, 16, 1>(r0, 18); dh = vh.template select<16, 1, 1, 1>(r0, 0); break;
                    case 2:  q = vb8.template select<16, 1, 16, 1>(r0, 4);  dh = vh.template select<16, 1, 1, 1>(r0, 1); break;
                    default: q = vb8.template select<16, 1, 16, 1>(r0, 20); dh = vh.template select<16, 1, 1, 1>(r0, 1); break;
                }
                simd<sycl::half, 256> hv = convert<sycl::half>(q) * dh.template replicate_vs_w_hs<16, 1, 16, 0>(0);
                auto vb2 = vb.template bit_cast_view<sycl::half, 8, 32>();
                auto hv2 = hv.template bit_cast_view<sycl::half, 16, 16>();
                vb2.template select<8, 1, 16, 2>(0, 0) = hv2.template select<8, 2, 16, 1>(0, 0);
                vb2.template select<8, 1, 16, 2>(0, 1) = hv2.template select<8, 2, 16, 1>(1, 0);
            } else {
                v_desc.set_y(tok0 + ts * 16);
                vb = esimd_x::lsc_load_2d<sycl::half, 16, 16, 1, false, true>(v_desc);
            }
#pragma unroll
            for (int rg = 0; rg < RG; ++rg) {
                simd<sycl::half, 128> pa = slm_block_load<sycl::half, 128>(SLM_P + (ts * R8 + rg * 8) * 32);
                simd<float, 128>      c  = o.template select<128, 1>(rg * 128);
                o.template select<128, 1>(rg * 128) = xmx::dpas<8, 8, float, float>(c, vb, pa);
            }
        };
        // a fully live block keeps the branch-free loop, so dense masks run as fast as without skipping
        if constexpr (SKIP) {
            if ((live != 0).all()) {
#pragma unroll
                for (int ts = 0; ts < T; ++ts) {
                    if ((ts & 1) == 0) {
                        vload(ts);
                    }
                    pv(ts);
                }
            } else {
#pragma unroll
                for (int ts = 0; ts < T; ++ts) {
                    if ((ts & 1) == 0 && (live[ts] || live[ts + 1])) {
                        vload(ts);
                    }
                    if (live[ts]) {
                        pv(ts);
                    }
                }
            }
        } else {
#pragma unroll
            for (int ts = 0; ts < T; ++ts) {
                if ((ts & 1) == 0) {
                    vload(ts);
                }
                pv(ts);
            }
        }
        // the next block writes SLM_MAX only after every thread has passed the barrier above, and
        // SLM_P / SLM_SUM only after its first barrier, which every thread reaches after its PV loop
    }

    // 6. this slice's unnormalized O rows and their max and sum
    const size_t slice = ((size_t) str * a.n_kvh + kvh) * a.nsplit + split;
#pragma unroll
    for (int r = 0; r < R8; ++r) {
        block_store<float, 16>(a.part_o + (slice * R8 + r) * D + t * 16, o.template select<16, 1>(16 * r));
    }
    if (t == 0) {
        simd<float, R8 * 2> ml;
        ml.template select<R8, 2>(0) = m;
        ml.template select<R8, 2>(1) = l;
#pragma unroll
        for (int i = 0; i < R8 * 2 / 16; ++i) {
            block_store<float, 16>(a.part_ml + slice * R8 * 2 + 16 * i, ml.template select<16, 1>(16 * i));
        }
    }
}

template <int DK, int DV, int RG, bool SKIP, bool Q8>
static void fa_xmx_launch(const fa_xmx_args & a, int n_kvh, bool grf128, dpct::queue_ptr stream) {
    constexpr int           T = DV / 16;
    const sycl::nd_range<1> range(sycl::range<1>((size_t) a.n_str * n_kvh * a.nsplit * T), sycl::range<1>(T));
    auto kern = [=](sycl::nd_item<1> it) SYCL_ESIMD_FUNCTION { fa_xmx_kernel<DK, DV, RG, SKIP, Q8>(a, it); };
    if constexpr (RG <= 3) {
        if (grf128) {
            stream->parallel_for(range, fa_xmx_grf128<decltype(kern)>{ kern });
            return;
        }
    }
    stream->parallel_for(range, fa_xmx_grf256<decltype(kern)>{ kern });
}

template <int D, bool SKIP, bool Q8>
static void fa_xmx_launch_rg_d(int RG, const fa_xmx_args & a, int n_kvh, bool grf128, dpct::queue_ptr stream) {
    switch (RG) {
        case 1: fa_xmx_launch<D, D, 1, SKIP, Q8>(a, n_kvh, grf128, stream); break;
        case 2: fa_xmx_launch<D, D, 2, SKIP, Q8>(a, n_kvh, grf128, stream); break;
        case 3: fa_xmx_launch<D, D, 3, SKIP, Q8>(a, n_kvh, grf128, stream); break;
        case 4: fa_xmx_launch<D, D, 4, SKIP, Q8>(a, n_kvh, grf128, stream); break;
        case 5: fa_xmx_launch<D, D, 5, SKIP, Q8>(a, n_kvh, grf128, stream); break;
        case 6: fa_xmx_launch<D, D, 6, SKIP, Q8>(a, n_kvh, grf128, stream); break;
        case 7: fa_xmx_launch<D, D, 7, SKIP, Q8>(a, n_kvh, grf128, stream); break;
        case 8: fa_xmx_launch<D, D, 8, SKIP, Q8>(a, n_kvh, grf128, stream); break;
        default: GGML_ABORT("XMX flash attention: %d row groups", RG);
    }
}

template <bool SKIP, bool Q8 = false>
static void fa_xmx_launch_rg(int D, int RG, const fa_xmx_args & a, int n_kvh, bool grf128, dpct::queue_ptr stream) {
    if (D == 576) {
        // MLA (40 heads on one latent head, one token): 5 row groups only (ggml_sycl_flash_attn_ext_xmx_dec_supported)
        if constexpr (!SKIP && !Q8) {
            GGML_ASSERT(RG == 5);
            fa_xmx_launch<576, 512, 5, false, false>(a, n_kvh, grf128, stream);
            return;
        }
        GGML_ABORT("XMX flash attention: MLA over a q8_0 cache or with span skipping");
    } else if (D == 128) {
        fa_xmx_launch_rg_d<128, SKIP, Q8>(RG, a, n_kvh, grf128, stream);
    } else {
        fa_xmx_launch_rg_d<256, SKIP, Q8>(RG, a, n_kvh, grf128, stream);
    }
}

// a q8_0 cache that fa_xmx_kernel reads in place with 2D dword loads: 64-byte aligned base, 16-byte aligned row pitch,
// and each head either at a 64-byte aligned offset (hb) or inside the token row (hx, the KV cache's heads interleaved
// per token)
static bool fa_xmx_q8_layout(const ggml_tensor * t, size_t & hb, int & hx) {
    const size_t row = t->ne[0] / QK8_0 * sizeof(block_q8_0);
    if ((uintptr_t) t->data % 64 != 0 || t->nb[1] % 16 != 0 || t->nb[1] < 64 || t->nb[1] >= (1u << 24)) {
        return false;
    }
    if (t->nb[2] % 64 == 0 && t->nb[1] >= row) {
        hb = t->nb[2];
        hx = 0;
        return true;
    }
    if (t->nb[2] % 4 == 0 && (size_t) (t->ne[2] - 1) * t->nb[2] + row <= t->nb[1]) {
        hb = 0;
        hx = (int) (t->nb[2] / 4);
        return true;
    }
    return false;
}

// an f16 cache that fa_xmx_kernel reads with 2D loads, a surface per head: 64-byte aligned base and head offsets, a
// 16-byte aligned row pitch of 64 bytes or more
static bool fa_xmx_f16_layout(const ggml_tensor * t) {
    return (uintptr_t) t->data % 64 == 0 && t->nb[1] % 16 == 0 && t->nb[1] >= 64 && t->nb[1] < (1u << 24) &&
           t->nb[2] % 64 == 0;
}

static bool fa_xmx_kv_layout(const ggml_tensor * t) {
    size_t hb = 0;
    int    hx = 0;
    return t->type == GGML_TYPE_Q8_0 ? fa_xmx_q8_layout(t, hb, hx) : fa_xmx_f16_layout(t);
}

}  // namespace

bool ggml_sycl_flash_attn_ext_xmx_dec_supported(int device, const ggml_tensor * dst) {
    static const int enabled = ggml_sycl_get_env("GGML_SYCL_FA_XMX_DEC", 1);
    static const int max_q   = ggml_sycl_get_env("GGML_SYCL_FA_XMX_MAX_Q", 8);
    if (!enabled || !g_ggml_sycl_enable_esimd) {
        return false;
    }
    const gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    if (arch != gpu_arch::intel_gpu_bmg_g21 && arch != gpu_arch::intel_gpu_bmg_g31) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    // a q8_0 cache read in place or an f16 cache, a dense mask (no n_kv_max hint), decode and verify rows
    const bool q8  = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0;
    const bool f16 = K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;
    if (Q->type != GGML_TYPE_F32 || !(q8 || f16) || dst->type != GGML_TYPE_F32 ||
        !mask || mask->type != GGML_TYPE_F16 || (sinks && sinks->type != GGML_TYPE_F32) || max_bias != 0.0f ||
        softcap != 0.0f || dst->src[5] || ggml_get_op_params_i32(dst, 4) > 0) {
        return false;
    }
    const int64_t D   = K->ne[0];
    // absorbed MLA: f16 latent cache, V = the first 512 dims of the K rows, >= 32 heads on one latent head and one
    // token (5 DPAS row groups); GGML_SYCL_FA_XMX_DEC_MLA=0 leaves it to the MKL / tile kernels
    static const int mla_env = ggml_sycl_get_env("GGML_SYCL_FA_XMX_DEC_MLA", 1);
    const bool    mla = mla_env && D == 576 && V->ne[0] == 512 && Q->ne[0] == 576 && f16 && K->ne[2] == 1 &&
                     (Q->ne[2] * Q->ne[1] + 7) / 8 == 5 && Q->ne[2] >= 32 && K->ne[1] % 256 == 0;
    if (!mla && ((D != FA_XMX_D && D != 128) || V->ne[0] != D || Q->ne[0] != D)) {
        return false;
    }
    if (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        return false;
    }
    const int64_t ns = Q->ne[3];
    if (K->ne[3] != ns || V->ne[3] != ns || dst->ne[3] != ns || mask->ne[2] != 1 || (mask->ne[3] != 1 && mask->ne[3] != ns) ||
        K->ne[2] != V->ne[2] || K->ne[1] != V->ne[1] || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    const int64_t g  = Q->ne[2] / K->ne[2];
    const int64_t nq = Q->ne[1];
    if (nq > max_q || g > 64 || (!mla && K->ne[1] % D != 0) || mask->ne[0] < K->ne[1] || mask->ne[1] < nq) {
        return false;
    }
    if (!fa_xmx_kv_layout(K) || !fa_xmx_kv_layout(V)) {
        return false;
    }
    // an f16 V that is a view of K (MLA-style caches) keeps the other kernels: the fallback of the dispatch (DPAS)
    // reads only a q8_0 cache that way
    const ggml_tensor * k_src = K->view_src ? K->view_src : K;
    if (!mla && f16 && V->view_src == k_src && V->view_offs == (K->view_src ? K->view_offs : 0)) {
        return false;
    }
    if (ns > 1 && (Q->nb[3] % 16 != 0 || (mask->ne[3] > 1 && mask->nb[3] % 16 != 0) || K->nb[3] % 64 != 0 || V->nb[3] % 64 != 0)) {
        return false;
    }
    if (Q->nb[0] != 4 || Q->nb[1] % 16 != 0 || Q->nb[2] % 16 != 0 || mask->nb[1] % 16 != 0 || !ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
}

bool ggml_sycl_flash_attn_ext_xmx_dec_data_ok(const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    return (uintptr_t) Q->data % 16 == 0 && (uintptr_t) mask->data % 16 == 0 && (K->data != V->data || K->ne[0] == 576) &&
           fa_xmx_kv_layout(K) && fa_xmx_kv_layout(V);
}

void ggml_sycl_flash_attn_ext_xmx_dec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const float *       sinks = dst->src[4] ? (const float *) dst->src[4]->data : nullptr;

    dpct::queue_ptr stream = ctx.stream();
    ggml_sycl_pool & pool  = ctx.pool();
    fattn_indexed_bias          bias = {};
    if (dst->src[6]) {
        bias = ggml_sycl_fattn_indexed_bias(dst);
    }

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_kvh = (int) K->ne[2];
    const int g     = (int) (Q->ne[2] / K->ne[2]);
    const int nq    = (int) Q->ne[1];
    const int n_kv  = (int) K->ne[1];
    const int D     = (int) K->ne[0];   // K dims (576 for MLA)
    const int DV    = (int) V->ne[0];   // O dims, also the KV block (512 for MLA)
    const int nblk  = (n_kv + DV - 1) / DV;
    const int n_str = (int) Q->ne[3];
    const int m_str = (int) mask->ne[3];

    // at most 64 rows (g heads x query rows) per launch; more rows go in balanced chunks
    const int n_chunks = (nq + 64 / g - 1) / (64 / g);
    const int nq_chunk = (nq + n_chunks - 1) / n_chunks;
    const int R8_max   = (g * nq_chunk + 7) / 8 * 8;

    const bool q8   = K->type == GGML_TYPE_Q8_0;
    size_t     k_hb = 0, v_hb = 0;
    int        k_hx = 0, v_hx = 0;
    if (q8) {
        GGML_ASSERT(fa_xmx_q8_layout(K, k_hb, k_hx) && fa_xmx_q8_layout(V, v_hb, v_hx));
    }

    // a q8_0 cache read in place with up to 2 row groups (3 with several streams) runs with 128 registers a thread and
    // twice the work-groups (fattn-xmx.cpp of the fork); split the cache so that about GGML_SYCL_FA_XMX_WGS work-groups run
    // at head size 256. A work-group has D / 16 threads, so head size 128 takes twice the work-groups for the same
    // threads: the split then fills the card's thread slots once (Inkling 2 Mini, 8 KV heads, q8_0 at 256K: 4-row
    // verify 1401 -> 1213 us; 1.5x or 2x the threads leave a partial second wave, 1.4-1.6x slower)
    static const int target_wgs = ggml_sycl_get_env("GGML_SYCL_FA_XMX_WGS", 64);
    static const int grf128_env = ggml_sycl_get_env("GGML_SYCL_FA_XMX_GRF128", 1);
    const int        RG_max     = R8_max / 8;
    const bool       grf128     = grf128_env && (RG_max <= 2 || (RG_max == 3 && n_str > 1));
    const int        n_hs       = n_str * n_kvh;
    const int        n_wgs      = (grf128 ? 2 * target_wgs : target_wgs) * FA_XMX_D / DV;
    int              nsplit     = std::max(1, std::min(nblk, (n_wgs + n_hs - 1) / n_hs));
    const int        bps        = (nblk + nsplit - 1) / nsplit;
    nsplit                      = (nblk + bps - 1) / bps;

    ggml_sycl_pool_alloc<float> part_o(pool, (size_t) n_hs * nsplit * R8_max * DV);
    ggml_sycl_pool_alloc<float> part_ml(pool, (size_t) n_hs * nsplit * R8_max * 2);

    const size_t o_nb1 = dst->nb[1] / sizeof(float);
    const size_t o_nb2 = dst->nb[2] / sizeof(float);
    const size_t o_nb3 = dst->nb[3] / sizeof(float);

    fa_xmx_args a = {};
    a.n_kvh     = n_kvh;
    a.n_str     = n_str;
    a.q_nb3     = n_str > 1 ? Q->nb[3] : 0;
    a.k_nb3     = n_str > 1 ? K->nb[3] : 0;
    a.v_nb3     = n_str > 1 ? V->nb[3] : 0;
    a.m_nb3     = m_str > 1 ? mask->nb[3] : 0;
    a.live_nb3  = 0;
    a.K         = (const char *) K->data;
    a.V         = (const char *) V->data;
    a.span_live = nullptr;
    a.k_hb      = k_hb;
    a.v_hb      = v_hb;
    a.k_hx      = k_hx;
    a.v_hx      = v_hx;
    a.part_o    = part_o.get();
    a.part_ml   = part_ml.get();
    a.g         = g;
    a.n_kv      = n_kv;
    a.nsplit    = nsplit;
    a.bps       = bps;
    a.q_nb1     = Q->nb[1];
    a.q_nb2     = Q->nb[2];
    a.k_nb1     = K->nb[1];
    a.k_nb2     = K->nb[2];
    a.v_nb1     = V->nb[1];
    a.v_nb2     = V->nb[2];
    a.m_nb1     = mask->nb[1];
    a.qscale    = scale * 1.44269504088896341f;
    a.btab        = bias.table;
    a.n_bins      = (int) bias.n_bins;
    a.n_head      = (int) Q->ne[2];
    a.extent      = bias.extent;
    a.nq_total    = nq;
    a.bpos        = bias.pos;
    a.bpos_row    = (size_t) bias.pos_row;
    a.bpos_nkv    = (size_t) bias.pos_nkv;
    a.bidx        = bias.indices;
    a.bidx_stride = (size_t) bias.stride;

    const float * po  = part_o.get();
    const float * pml = part_ml.get();
    for (int j0 = 0; j0 < nq; j0 += nq_chunk) {
        const int nqc = std::min(nq_chunk, nq - j0);
        const int M   = g * nqc;
        const int RG  = (M + 7) / 8;
        const int R8  = RG * 8;
        a.Q      = (const char *) Q->data + j0 * Q->nb[1];
        a.mask   = (const char *) mask->data + j0 * mask->nb[1];
        a.nq     = nqc;
        a.q_base = j0;
        if (q8) {
            fa_xmx_launch_rg<false, true>(D, RG, a, n_kvh, grf128, stream);
        } else {
            fa_xmx_launch_rg<false, false>(D, RG, a, n_kvh, grf128, stream);
        }

        // merge the slices: dst[j][kvh g + c][d] = sum_s w_s O_s / sum_s w_s l_s, w_s = exp2(m_s - max_s m_s); an
        // attention sink (a logit with no value row, natural log units: times log2(e) here) joins the max and the sum
        float * out = (float *) dst->data + j0 * o_nb2;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n_hs * M * DV), sycl::range<1>(DV)),
                             [=](sycl::nd_item<1> it) {
                                 const int row = it.get_group(0);
                                 const int hs  = row / M;
                                 const int r   = row - hs * M;
                                 const int st  = hs / n_kvh;
                                 const int kvh = hs - st * n_kvh;
                                 const int d   = it.get_local_id(0);
                                 const int j   = r / g;
                                 const int c   = r - j * g;
                                 float     mx  = -FLT_MAX;
                                 for (int s = 0; s < nsplit; ++s) {
                                     mx = sycl::fmax(mx, pml[(((size_t) hs * nsplit + s) * R8 + r) * 2]);
                                 }
                                 const float sink = sinks ? sinks[kvh * g + c] * 1.44269504088896341f : -FLT_MAX;
                                 mx               = sycl::fmax(mx, sink);
                                 float num = 0.0f, den = sinks ? sycl::exp2(sink - mx) : 0.0f;
                                 for (int s = 0; s < nsplit; ++s) {
                                     const size_t i = ((size_t) hs * nsplit + s) * R8 + r;
                                     const float  w = sycl::exp2(pml[i * 2] - mx);
                                     num += w * po[i * DV + d];
                                     den += w * pml[i * 2 + 1];
                                 }
                                 out[st * o_nb3 + j * o_nb2 + (size_t) (kvh * g + c) * o_nb1 + d] = num / den;
                             });
    }
}
