#include "fattn-xmx.hpp"
#include "dpas.hpp"
#include "fattn.hpp"

#include <cfloat>
#include <climits>

#ifdef GGML_SYCL_HAS_DPAS

#include <sycl/ext/intel/experimental/esimd/math.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

// Flash attention on XMX for short query batches with grouped-query attention (Qwen3.8-27B: 24 query / 4 KV
// heads, head 256; decode and DFlash verify batches of up to 8 rows). The tile kernel scores these on vector
// units and is compute-bound at depth; here the g query heads of one KV head times the nq query rows form
// M = g*nq <= 64 rows of DPAS operands, so each K and V element is read once for all of them.
//
// A work-group of FA_XMX_T threads handles one KV head and one slice of the cache (split-K). Per block of
// FA_XMX_BK tokens, thread t scores tokens 16t..16t+15 for all rows (S = Q K^T: the K block is loaded
// transposed, which is the VNNI B operand), the row maxima and sums go through SLM, P = exp2(S - max) is
// stored to SLM as f16, and thread t accumulates dims 16t..16t+15 of O += P V (the V block loaded VNNI-
// transformed). Q is pre-scaled by scale*log2(e) so the softmax uses exp2. Each slice writes its
// unnormalized O with the row max and sum; a second kernel merges the slices into dst.
// A q8_0 cache is read in place, except by the prefill span lists (fa_xmx_list_kernel), which read an f16 copy made
// once per call. Batches over 64 rows run as row chunks.

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
template <int RG, bool SKIP, bool Q8>
static ESIMD_INLINE void fa_xmx_kernel(const fa_xmx_args a, const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    namespace xmx     = sycl::ext::intel::esimd::xmx;
    namespace esimd_x = sycl::ext::intel::experimental::esimd;
    constexpr int R8 = RG * 8;  // rows padded to whole DPAS row groups
    constexpr int T  = FA_XMX_T;
    // SLM: Q as [16 k-steps][R8 rows][16 dims] f16, P as [T token-steps][R8 rows][16 tokens] f16, then the
    // per-thread row maxima and sums [T][R8] f32. Each 8x16 DPAS A operand is 256 contiguous bytes.
    constexpr uint32_t SLM_Q   = 0;
    constexpr uint32_t SLM_P   = SLM_Q + 16 * R8 * 32;
    constexpr uint32_t SLM_MAX = SLM_P + T * R8 * 32;
    constexpr uint32_t SLM_SUM = SLM_MAX + T * R8 * 4;
    constexpr uint32_t SLM_END = SLM_SUM + T * R8 * 4;
    slm_init<SLM_END>();

    const properties a16{ alignment<16> };

    const int t     = it.get_local_id(0);
    const int wg    = it.get_group(0);
    const int kvh   = wg / a.nsplit;
    const int split = wg - kvh * a.nsplit;
    const int M     = a.g * a.nq;

    // 1. Q rows r = j g + c (query j, head c of this KV head) to SLM, scaled, f16; padding rows are 0
    for (int r = t; r < R8; r += T) {
        simd<float, FA_XMX_D> q = 0.0f;
        if (r < M) {
            const int     j  = r / a.g;
            const int     c  = r - j * a.g;
            const float * qp = (const float *) (a.Q + j * a.q_nb1 + (size_t) (kvh * a.g + c) * a.q_nb2);
#pragma unroll
            for (int i = 0; i < FA_XMX_D / 64; ++i) {
                q.template select<64, 1>(64 * i) = block_load<float, 64>(qp + 64 * i, a16);
            }
            q *= a.qscale;
        }
        simd<sycl::half, FA_XMX_D> qh = convert<sycl::half>(q);
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
            slm_block_store<sycl::half, 16>(SLM_Q + (ks * R8 + r) * 32, qh.template select<16, 1>(16 * ks));
        }
    }
    barrier();

    // running row max (log2 domain) and sum, identical in every thread; this thread's 16 dims of O
    simd<float, R8>      m = -FLT_MAX;
    simd<float, R8>      l = 0.0f;
    simd<float, R8 * 16> o = 0.0f;

    // head slices of the caches: 256 f16 per row at the row pitch
    const uint32_t * Kh = (const uint32_t *) (a.K + (size_t) kvh * (Q8 ? a.k_hb : a.k_nb2));
    const sycl::half * Vh = (const sycl::half *) (a.V + (size_t) kvh * (Q8 ? a.v_hb : a.v_nb2));
    const unsigned surf_w = FA_XMX_D * sizeof(sycl::half) - 1;
    const unsigned surf_h = (unsigned) a.n_kv - 1;
    esimd_x::config_2d_mem_access<uint32_t, 8, 16, 1>    k_desc(Kh, Q8 ? (unsigned) a.k_nb1 - 1 : surf_w, surf_h,
                                                                (unsigned) a.k_nb1 - 1, 0, 0);
    esimd_x::config_2d_mem_access<sycl::half, 16, 16, 1> v_desc(Vh, surf_w, surf_h, (unsigned) a.v_nb1 - 1, t * 16, 0);

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
    const int nblk = a.n_kv / FA_XMX_BK;
    const int blk1 = blk0 + a.bps < nblk ? blk0 + a.bps : nblk;
    for (int blk = blk0; blk < blk1; ++blk) {
        const int tok0   = blk * FA_XMX_BK;
        const int my_tok = tok0 + t * 16;

        // masked spans add exactly 0 after the softmax, so their K and V are not read; every thread sees the same
        // flags, so a block without live spans is skipped by all of them
        simd<uint8_t, 16> live = 1;
        if constexpr (SKIP) {
            live = block_load<uint8_t, 16>(a.span_live + blk * 16, a16);
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
                for (int P = 0; P < FA_XMX_D / 64; ++P) {
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
                for (int ks = 0; ks < 16; ++ks) {
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

        // 3. mask (log2 domain), per query row j
#pragma unroll
        for (int r = 0; r < R8; ++r) {
            if (r < M) {
                const sycl::half * mp = (const sycl::half *) (a.mask + (r / a.g) * a.m_nb1) + my_tok;
                simd<float, 16>    mk = convert<float>(block_load<sycl::half, 16>(mp, a16));
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
    const size_t slice = (size_t) kvh * a.nsplit + split;
#pragma unroll
    for (int r = 0; r < R8; ++r) {
        block_store<float, 16>(a.part_o + (slice * R8 + r) * FA_XMX_D + t * 16, o.template select<16, 1>(16 * r));
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

// Prefill over a sparse mask: work-group (chunk, KV head, split) walks the list of 16-token spans that the chunk's
// rows attend to, 16 spans per step (one per thread), so the work follows the live share of the mask, not n_kv.
// Same math as fa_xmx_kernel; a thread without a span in the step contributes -inf scores.
template <int RG, bool COMPACT> static ESIMD_INLINE void fa_xmx_list_kernel(const fa_xmx_args a, const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    namespace xmx     = sycl::ext::intel::esimd::xmx;
    namespace esimd_x = sycl::ext::intel::experimental::esimd;
    constexpr int R8 = RG * 8;
    constexpr int T  = FA_XMX_T;
    constexpr uint32_t SLM_Q   = 0;
    constexpr uint32_t SLM_P   = SLM_Q + 16 * R8 * 32;
    constexpr uint32_t SLM_MAX = SLM_P + T * R8 * 32;
    constexpr uint32_t SLM_SUM = SLM_MAX + T * R8 * 4;
    constexpr uint32_t SLM_END = SLM_SUM + T * R8 * 4;
    slm_init<SLM_END>();

    const properties a16{ alignment<16> };

    const int t     = it.get_local_id(0);
    const int wg    = it.get_group(0);
    const int split = wg % a.nsplit;
    const int kvh   = (wg / a.nsplit) % a.n_kvh;
    const int chunk = wg / (a.nsplit * a.n_kvh);
    const int j0    = chunk * a.nq_chunk;
    const int nq    = a.nq - j0 < a.nq_chunk ? a.nq - j0 : a.nq_chunk;
    const int M     = a.g * nq;
    const char * Qc = a.Q + (size_t) j0 * a.q_nb1;
    const char * Mc = a.mask + (size_t) j0 * a.m_nb1;

    for (int r = t; r < R8; r += T) {
        simd<float, FA_XMX_D> q = 0.0f;
        if (r < M) {
            const int     j  = r / a.g;
            const int     c  = r - j * a.g;
            const float * qp = (const float *) (Qc + j * a.q_nb1 + (size_t) (kvh * a.g + c) * a.q_nb2);
#pragma unroll
            for (int i = 0; i < FA_XMX_D / 64; ++i) {
                q.template select<64, 1>(64 * i) = block_load<float, 64>(qp + 64 * i, a16);
            }
            q *= a.qscale;
        }
        simd<sycl::half, FA_XMX_D> qh = convert<sycl::half>(q);
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
            slm_block_store<sycl::half, 16>(SLM_Q + (ks * R8 + r) * 32, qh.template select<16, 1>(16 * ks));
        }
    }
    barrier();

    simd<float, R8>      m = -FLT_MAX;
    simd<float, R8>      l = 0.0f;
    simd<float, R8 * 16> o = 0.0f;

    const uint32_t *   Kh     = (const uint32_t *) (a.K + (size_t) kvh * a.k_nb2);
    const sycl::half * Vh     = (const sycl::half *) (a.V + (size_t) kvh * a.v_nb2);
    const unsigned     surf_w = FA_XMX_D * sizeof(sycl::half) - 1;
    const unsigned     surf_h = (unsigned) a.n_kv - 1;
    esimd_x::config_2d_mem_access<uint32_t, 8, 16, 1>    k_desc(Kh, surf_w, surf_h, (unsigned) a.k_nb1 - 1, 0, 0);
    esimd_x::config_2d_mem_access<sycl::half, 16, 16, 1> v_desc(Vh, surf_w, surf_h, (unsigned) a.v_nb1 - 1, t * 16, 0);

    const int32_t * list   = a.span_list + (size_t) chunk * a.list_stride;
    const int       n_live = a.span_count[chunk];
    const int       n_grp  = (n_live + T - 1) / T;
    const int       gps    = (n_grp + a.nsplit - 1) / a.nsplit;
    const int       g0     = split * gps;
    const int       g1     = g0 + gps < n_grp ? g0 + gps : n_grp;
    const simd<int32_t, T> lane(0, 1);
    for (int gi = g0; gi < g1; ++gi) {
        // the list rows are padded to whole steps, so the last step reads past n_live inside its own row
        const simd<int32_t, T> sl    = block_load<int32_t, T>(list + gi * T, a16);
        const simd_mask<T>     valid = (lane + gi * T) < n_live;
        const bool             mine  = gi * T + t < n_live;
        const int              my_tok = mine ? (int) sl[t] * 16 : 0;

        simd<float, R8 * 16> s = -INFINITY;
        if (mine) {
            s = 0.0f;
            k_desc.set_y(my_tok);
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {
                k_desc.set_x(ks * 8);
                simd<uint32_t, 128>   kb = esimd_x::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(k_desc);
                simd<sycl::half, 256> b  = kb.template bit_cast_view<sycl::half>().read();
#pragma unroll
                for (int rg = 0; rg < RG; ++rg) {
                    simd<sycl::half, 128> qa = slm_block_load<sycl::half, 128>(SLM_Q + (ks * R8 + rg * 8) * 32);
                    simd<float, 128>      c  = s.template select<128, 1>(rg * 128);
                    s.template select<128, 1>(rg * 128) = xmx::dpas<8, 8, float, float>(c, b, qa);
                }
            }
            simd<float, 16> mk = 0.0f;
            int next_query = 0;
#pragma unroll
            for (int r = 0; r < R8; ++r) {
                if (r < M) {
                    if (!COMPACT || r == next_query) {
                        const sycl::half * mp = (const sycl::half *) (Mc + (r / a.g) * a.m_nb1) + my_tok;
                        mk = convert<float>(block_load<sycl::half, 16>(mp, a16));
                        if constexpr (COMPACT) {
                            next_query += a.g;
                            const uint32_t * bp = a.mask_bits + (size_t) (j0 + r / a.g) * (a.n_kv / 16) + my_tok / 16;
                            const simd<uint32_t, 1> bits = block_load<uint32_t, 1>(bp);
                            const simd<uint32_t, 16> lane(0, 1);
                            simd<float, 16> add = a.mask_fill;
                            add.merge(a.mask_zero, (bits[0] & (1u << lane)) != 0);
                            mk = convert<float>(convert<sycl::half>(add + mk));
                        }
                    }
                    if constexpr (COMPACT) {
                        // Keep the original fused rounding when the mask is reused.
                        s.template select<16, 1>(16 * r) = esimd_x::fma(mk, simd<float, 16>(1.44269504088896341f), s.template select<16, 1>(16 * r).read());
                    } else {
                        s.template select<16, 1>(16 * r) += mk * 1.44269504088896341f;
                    }
                }
            }
        }

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
        auto pv = [&](int ts) {
            v_desc.set_y((int) sl[ts] * 16);
            simd<sycl::half, 256> vb = esimd_x::lsc_load_2d<sycl::half, 16, 16, 1, false, true>(v_desc);
#pragma unroll
            for (int rg = 0; rg < RG; ++rg) {
                simd<sycl::half, 128> pa = slm_block_load<sycl::half, 128>(SLM_P + (ts * R8 + rg * 8) * 32);
                simd<float, 128>      c  = o.template select<128, 1>(rg * 128);
                o.template select<128, 1>(rg * 128) = xmx::dpas<8, 8, float, float>(c, vb, pa);
            }
        };
        if (valid.all()) {
#pragma unroll
            for (int ts = 0; ts < T; ++ts) {
                pv(ts);
            }
        } else {
#pragma unroll
            for (int ts = 0; ts < T; ++ts) {
                if (valid[ts]) {
                    pv(ts);
                }
            }
        }
    }

    const size_t slice = ((size_t) chunk * a.n_kvh + kvh) * a.nsplit + split;
#pragma unroll
    for (int r = 0; r < R8; ++r) {
        block_store<float, 16>(a.part_o + (slice * R8 + r) * FA_XMX_D + t * 16, o.template select<16, 1>(16 * r));
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

template <int RG, bool COMPACT> static void fa_xmx_list_launch(const fa_xmx_args & a, int n_chunks, dpct::queue_ptr stream) {
    const sycl::nd_range<1> range(sycl::range<1>((size_t) n_chunks * a.n_kvh * a.nsplit * FA_XMX_T),
                                  sycl::range<1>(FA_XMX_T));
    auto kern = [=](sycl::nd_item<1> it) SYCL_ESIMD_FUNCTION { fa_xmx_list_kernel<RG, COMPACT>(a, it); };
    stream->parallel_for(range, fa_xmx_grf256<decltype(kern)>{ kern });
}

template <int RG, bool SKIP, bool Q8> static void fa_xmx_launch(const fa_xmx_args & a, int n_kvh, dpct::queue_ptr stream) {
    const sycl::nd_range<1> range(sycl::range<1>((size_t) n_kvh * a.nsplit * FA_XMX_T), sycl::range<1>(FA_XMX_T));
    auto kern = [=](sycl::nd_item<1> it) SYCL_ESIMD_FUNCTION { fa_xmx_kernel<RG, SKIP, Q8>(a, it); };
    stream->parallel_for(range, fa_xmx_grf256<decltype(kern)>{ kern });
}

template <bool SKIP, bool Q8 = false>
static void fa_xmx_launch_rg(int RG, const fa_xmx_args & a, int n_kvh, dpct::queue_ptr stream) {
    switch (RG) {
        case 1: fa_xmx_launch<1, SKIP, Q8>(a, n_kvh, stream); break;
        case 2: fa_xmx_launch<2, SKIP, Q8>(a, n_kvh, stream); break;
        case 3: fa_xmx_launch<3, SKIP, Q8>(a, n_kvh, stream); break;
        case 4: fa_xmx_launch<4, SKIP, Q8>(a, n_kvh, stream); break;
        case 5: fa_xmx_launch<5, SKIP, Q8>(a, n_kvh, stream); break;
        case 6: fa_xmx_launch<6, SKIP, Q8>(a, n_kvh, stream); break;
        case 7: fa_xmx_launch<7, SKIP, Q8>(a, n_kvh, stream); break;
        case 8: fa_xmx_launch<8, SKIP, Q8>(a, n_kvh, stream); break;
        default: GGML_ABORT("XMX flash attention: %d row groups", RG);
    }
}

template <bool COMPACT>
static void fa_xmx_list_launch_rg(int RG, const fa_xmx_args & a, int n_chunks, dpct::queue_ptr stream) {
    switch (RG) {
        case 1: fa_xmx_list_launch<1, COMPACT>(a, n_chunks, stream); break;
        case 2: fa_xmx_list_launch<2, COMPACT>(a, n_chunks, stream); break;
        case 3: fa_xmx_list_launch<3, COMPACT>(a, n_chunks, stream); break;
        case 4: fa_xmx_list_launch<4, COMPACT>(a, n_chunks, stream); break;
        case 5: fa_xmx_list_launch<5, COMPACT>(a, n_chunks, stream); break;
        case 6: fa_xmx_list_launch<6, COMPACT>(a, n_chunks, stream); break;
        case 7: fa_xmx_list_launch<7, COMPACT>(a, n_chunks, stream); break;
        case 8: fa_xmx_list_launch<8, COMPACT>(a, n_chunks, stream); break;
        default: GGML_ABORT("XMX flash attention: %d row groups", RG);
    }
}

// the 16-token spans that any row of each chunk attends to (a mask entry above -inf), ascending; one work-group per
// chunk scans its rows in steps of WG spans and compacts the live ones
template <bool COMPACT>
static void fa_xmx_span_lists(const char * mask, const uint32_t * bits, size_t m_nb1, int nq, int nq_chunk, int n_chunks, int n_span,
                              int list_stride, int32_t * list, int32_t * count, dpct::queue_ptr stream) {
    constexpr int WG = 256;
    stream->parallel_for(sycl::nd_range<1>((size_t) n_chunks * WG, WG), [=](sycl::nd_item<1> it) {
        const int c   = it.get_group(0);
        const int tid = it.get_local_id(0);
        const int j0  = c * nq_chunk;
        const int nqc = sycl::min(nq_chunk, nq - j0);
        int       n   = 0;
        for (int s0 = 0; s0 < n_span; s0 += WG) {
            const int s    = s0 + tid;
            int       live = 0;
            if (s < n_span) {
                for (int j = 0; j < nqc && !live; ++j) {
                    if constexpr (COMPACT) {
                        live = (bits[(size_t) (j0 + j) * n_span + s] >> 16) != 0;
                    } else {
                        // two 16-byte loads: mask rows are only known to be 16-byte aligned
                        const sycl::vec<uint16_t, 8> * mv = (const sycl::vec<uint16_t, 8> *) (mask + (size_t) (j0 + j) * m_nb1) + 2 * s;
                        const sycl::vec<uint16_t, 8>   m0 = mv[0], m1 = mv[1];
                        for (int i = 0; i < 8; ++i) {
                            live |= m0[i] != 0xFC00 || m1[i] != 0xFC00;  // f16 -inf
                        }
                    }
                }
            }
            const int pos = sycl::exclusive_scan_over_group(it.get_group(), live, sycl::plus<int>());
            if (live) {
                list[(size_t) c * list_stride + n + pos] = s;
            }
            n += sycl::reduce_over_group(it.get_group(), live, sycl::plus<int>());
        }
        if (tid == 0) {
            count[c] = n;
        }
    });
}

// prefill-sized batches (more rows than GGML_SYCL_FA_XMX_MAX_Q) over a q8_0 cache with a sparse mask (QSA sets the
// n_kv_max hint) and at least GGML_SYCL_FA_XMX_LIST_MIN_KV cells: row chunks walk their own live spans
static bool fa_xmx_use_list(const ggml_tensor * dst, int max_q) {
    static const int min_kv = ggml_sycl_get_env("GGML_SYCL_FA_XMX_LIST_MIN_KV", 32768);
    static const int skip   = ggml_sycl_get_env("GGML_SYCL_FA_XMX_SKIP", 1);
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    return skip && min_kv > 0 && K->type == GGML_TYPE_Q8_0 && Q->ne[1] > max_q &&
           ggml_get_op_params_i32(dst, 4) > 0 && K->ne[1] >= min_kv;
}

// a q8_0 cache that fa_xmx_kernel reads in place with 2D dword loads: 64-byte aligned base, 16-byte aligned row pitch,
// and each head either at a 64-byte aligned offset (hb) or inside the token row (hx, the KV cache's heads interleaved
// per token)
static bool fa_xmx_q8_layout(const ggml_tensor * t, size_t & hb, int & hx) {
    const size_t row = FA_XMX_D / QK8_0 * sizeof(block_q8_0);
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

#endif // GGML_SYCL_HAS_DPAS

static bool fa_xmx_supported(int device, const ggml_tensor * dst, bool check_data) {
#ifdef GGML_SYCL_HAS_DPAS
    static const int enabled = ggml_sycl_get_env("GGML_SYCL_FA_XMX", 1);
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

    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // a q8_0 cache is read in place or converted to f16 once per call (see ggml_sycl_flash_attn_ext_xmx)
    const bool f16_kv = K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;
    const bool q8_kv  = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 && (!check_data || K->data != V->data) &&
                       K->nb[0] == ggml_type_size(GGML_TYPE_Q8_0) && V->nb[0] == ggml_type_size(GGML_TYPE_Q8_0);
    if (Q->type != GGML_TYPE_F32 || !(f16_kv || q8_kv) || dst->type != GGML_TYPE_F32 ||
        !mask || mask->type != GGML_TYPE_F16 || sinks || max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (K->ne[0] != FA_XMX_D || V->ne[0] != FA_XMX_D || Q->ne[0] != FA_XMX_D) {
        return false;
    }
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1 ||
        K->ne[2] != V->ne[2] || K->ne[1] != V->ne[1] || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    const int64_t g  = Q->ne[2] / K->ne[2];
    const int64_t nq = Q->ne[1];
    // rows beyond 64 / g run as further row chunks, each reading the cache once more
    if ((nq > max_q && !fa_xmx_use_list(dst, max_q)) || g > 64 || K->ne[1] % FA_XMX_BK != 0 || mask->ne[0] < K->ne[1] ||
        mask->ne[1] < nq) {
        return false;
    }
    // 2D block loads: 64-byte aligned head slices, 16-byte aligned row pitch of at least 64 bytes (the f16 copy
    // of a q8_0 cache is contiguous and meets them)
    for (const ggml_tensor * t : { K, V }) {
        if (f16_kv && ((check_data && (uintptr_t) t->data % 64 != 0) || t->nb[0] != 2 || t->nb[2] % 64 != 0 || t->nb[1] % 16 != 0 ||
                       t->nb[1] < 64)) {
            return false;
        }
    }
    if ((check_data && (uintptr_t) Q->data % 16 != 0) || Q->nb[0] != 4 || Q->nb[1] % 16 != 0 || Q->nb[2] % 16 != 0 ||
        (check_data && (uintptr_t) mask->data % 16 != 0) || mask->nb[1] % 16 != 0 || !ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
#else
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    GGML_UNUSED(check_data);
    return false;
#endif
}

bool ggml_sycl_flash_attn_ext_xmx_supported(int device, const ggml_tensor * dst) {
    return fa_xmx_supported(device, dst, true);
}

bool ggml_sycl_flash_attn_ext_xmx_qsa_supported(int device, const ggml_tensor * dst) {
#ifdef GGML_SYCL_HAS_DPAS
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2];
    static const int max_q = ggml_sycl_get_env("GGML_SYCL_FA_XMX_MAX_Q", 8);
    if (!Q || !K || !V || Q->ne[1] <= 0 || Q->ne[1] > INT_MAX - 64 || Q->ne[2] <= 0 || Q->ne[2] > INT_MAX ||
        K->ne[1] <= 0 || K->ne[1] > INT_MAX || K->ne[2] <= 0 || K->ne[2] > INT_MAX ||
        K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0 || K == V ||
        (K->view_src && K->view_src == V->view_src && K->view_offs == V->view_offs) ||
        Q->view_offs % 16 != 0 || (uintptr_t) Q->data % 16 != 0 ||
        !fa_xmx_use_list(dst, max_q) || !fa_xmx_supported(device, dst, false)) {
        return false;
    }
    const int64_t g = Q->ne[2] / K->ne[2];
    const int64_t chunks = (Q->ne[1] + 64 / g - 1) / (64 / g);
    return K->ne[2] <= INT_MAX / chunks;
#else
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#endif
}

#ifdef GGML_SYCL_HAS_DPAS
static void fa_xmx_qsa_bits(const ggml_tensor * indices, const ggml_tensor * causal, uint32_t * bits, dpct::queue_ptr stream) {
    const size_t nk = causal->ne[0], n_span = nk / 16, nw = n_span * ggml_nrows(causal), ni = ggml_nelements(indices);
    const int32_t * ids = (const int32_t *) indices->data;
    const uint16_t * mask = (const uint16_t *) causal->data;
    const size_t ns = indices->ne[0];
    stream->parallel_for(sycl::range<1>(nw), [=](sycl::id<1> id) {
        const size_t j = id[0];
        const sycl::vec<uint16_t, 8> * mv = (const sycl::vec<uint16_t, 8> *) mask + j * 2;
        const sycl::vec<uint16_t, 8> m0 = mv[0], m1 = mv[1];
        uint32_t visible = 0;
        // With a -inf background, only +inf and NaN produce visible entries.
        for (int b = 0; b < 8; ++b) {
            visible |= uint32_t((m0[b] & 0x7C00) == 0x7C00 && m0[b] != 0xFC00) << b;
            visible |= uint32_t((m1[b] & 0x7C00) == 0x7C00 && m1[b] != 0xFC00) << (b + 8);
        }
        bits[j] = visible << 16;
    });
    stream->parallel_for(sycl::range<1>(ni), [=](sycl::id<1> id) {
        const size_t j = id[0];
        const int32_t k = ids[j];
        if (k >= 0 && (size_t) k < nk) {
            const uint16_t v = mask[(j / ns) * nk + k];
            const uint32_t selected = 1u << (k % 16);
            const uint32_t visible = v != 0xFC00 ? selected << 16 : 0;
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> word(bits[(j / ns) * n_span + k / 16]);
            word.fetch_or(selected | visible);
        }
    });
}
#endif

static void fa_xmx_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const ggml_tensor * indices, const ggml_tensor * causal, float fill, float zero) {
#ifdef GGML_SYCL_HAS_DPAS
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_kvh = (int) K->ne[2];
    const int g     = (int) (Q->ne[2] / K->ne[2]);
    const int nq    = (int) Q->ne[1];
    const int n_kv  = (int) K->ne[1];
    const int nblk  = n_kv / FA_XMX_BK;

    // at most 64 rows (g heads x query rows) per launch; more rows go in balanced chunks
    const int n_chunks = (nq + 64 / g - 1) / (64 / g);
    const int nq_chunk = (nq + n_chunks - 1) / n_chunks;
    const int R8_max   = (g * nq_chunk + 7) / 8 * 8;

    dpct::queue_ptr stream = ctx.stream();

    // one flag per 16-token span: 1 if any query row of this call attends to it (a mask entry above -inf). Only a
    // q8_0 cache takes them: QSA's sparse masks (Qwen3.8-Flash-Next) come with one, and f16 caches keep the kernel
    // that reads every span, since the skipping build changed greedy outputs of f16 models at short context even
    // with skipping off (cause not found)
    static const int skip   = ggml_sycl_get_env("GGML_SYCL_FA_XMX_SKIP", 1);
    static const int max_q  = ggml_sycl_get_env("GGML_SYCL_FA_XMX_MAX_Q", 8);
    const bool       q8     = K->type == GGML_TYPE_Q8_0;
    const int        n_span = n_kv / 16;
    ggml_sycl_pool_alloc<uint32_t> mask_bits(ctx.pool());
    if (indices) {
        GGML_ASSERT(causal && causal->ne[0] == n_kv && ggml_nrows(causal) == nq);
        fa_xmx_qsa_bits(indices, causal, mask_bits.alloc((size_t) nq * n_span), stream);
    }
    // prefill: each row chunk walks its own list of live spans; the chunks together touch nearly every span, so the
    // whole cache is converted
    const bool list = q8 && fa_xmx_use_list(dst, max_q);
    ggml_sycl_pool_alloc<uint8_t> span_live(ctx.pool());
    uint8_t * live = q8 ? span_live.alloc(n_span) : nullptr;
    if (q8 && skip && !list) {
        const char * mp    = (const char *) mask->data;
        const size_t m_nb1 = mask->nb[1];
        stream->parallel_for(sycl::range<1>(n_span), [=](sycl::id<1> id) {
            const int s   = id[0];
            bool      any = false;
            for (int j = 0; j < nq; ++j) {
                // two 16-byte loads: mask rows are only known to be 16-byte aligned
                const sycl::vec<uint16_t, 8> * mv = (const sycl::vec<uint16_t, 8> *) (mp + j * m_nb1) + 2 * s;
                const sycl::vec<uint16_t, 8>   m0 = mv[0], m1 = mv[1];
                for (int i = 0; i < 8; ++i) {
                    any = any || m0[i] != 0xFC00 || m1[i] != 0xFC00;  // f16 -inf
                }
            }
            live[s] = any;
        });
    } else if (q8) {
        stream->memset(live, 1, n_span);
    }

    // a q8_0 cache is read in place (half the bytes of an f16 cache); for the span lists, or a layout the 2D loads
    // cannot take, the live spans of K and V are converted to contiguous f16 once and every chunk reads the copy
    static const int q8_direct = ggml_sycl_get_env("GGML_SYCL_FA_XMX_Q8_DIRECT", 1);
    size_t           k_hb = 0, v_hb = 0;
    int              k_hx = 0, v_hx = 0;
    const bool       direct = q8 && !list && q8_direct && fa_xmx_q8_layout(K, k_hb, k_hx) && fa_xmx_q8_layout(V, v_hb, v_hx);
    const char * K_data = (const char *) K->data;
    const char * V_data = (const char *) V->data;
    size_t       k_nb1 = K->nb[1], k_nb2 = K->nb[2], v_nb1 = V->nb[1], v_nb2 = V->nb[2];
    ggml_sycl_fattn_alloc K_f16(ctx.fattn_buffers().K);
    ggml_sycl_fattn_alloc V_f16(ctx.fattn_buffers().V);
    if (q8 && !direct) {
        const ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);
        sycl::half * Kh = extra.K_buffer_ptr ? (sycl::half *) extra.K_buffer_ptr : K_f16.alloc(ggml_nelements(K));
        sycl::half * Vh = extra.V_buffer_ptr ? (sycl::half *) extra.V_buffer_ptr : V_f16.alloc(ggml_nelements(V));
        GGML_ASSERT((uintptr_t) Kh % 64 == 0 && (uintptr_t) Vh % 64 == 0);
        // one work-item per 8 values of a row (head, token, group); rows of masked spans stay stale and are never read
        for (const ggml_tensor * t : { K, V }) {
            const char * src = (const char *) t->data;
            sycl::half * out = t == K ? Kh : Vh;
            const size_t nb1 = t->nb[1], nb2 = t->nb[2];
            stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(n_kvh, n_kv, FA_XMX_D / 8), sycl::range<3>(1, 8, FA_XMX_D / 8)),
                                 [=](sycl::nd_item<3> it) {
                                     const int h   = it.get_global_id(0);
                                     const int tok = it.get_global_id(1);
                                     const int c   = it.get_global_id(2);
                                     if (!live[tok / 16]) {
                                         return;
                                     }
                                     const block_q8_0 * b = (const block_q8_0 *) (src + h * nb2 + (size_t) tok * nb1) + c / 4;
                                     const float        d = b->d;
                                     sycl::vec<sycl::half, 8> v;
                                     for (int k = 0; k < 8; ++k) {
                                         v[k] = d * b->qs[(c % 4) * 8 + k];
                                     }
                                     *(sycl::vec<sycl::half, 8> *) (out + ((size_t) h * n_kv + tok) * FA_XMX_D + c * 8) = v;
                                 });
        }
        K_data = (const char *) Kh;
        V_data = (const char *) Vh;
        k_nb1  = v_nb1 = FA_XMX_D * sizeof(sycl::half);
        k_nb2  = v_nb2 = (size_t) n_kv * k_nb1;
    }

    static const int target_wgs = ggml_sycl_get_env("GGML_SYCL_FA_XMX_WGS", 64);
    const size_t     o_nb1      = dst->nb[1] / sizeof(float);
    const size_t     o_nb2      = dst->nb[2] / sizeof(float);

    if (list) {
        const int list_stride = (n_span + FA_XMX_T - 1) / FA_XMX_T * FA_XMX_T;
        ggml_sycl_pool_alloc<int32_t> span_list(ctx.pool(), (size_t) n_chunks * list_stride);
        ggml_sycl_pool_alloc<int32_t> span_count(ctx.pool(), n_chunks);
        if (indices) {
            fa_xmx_span_lists<true>(nullptr, mask_bits.get(), causal->nb[1], nq, nq_chunk, n_chunks, n_span, list_stride,
                                   span_list.get(), span_count.get(), stream);
        } else {
            fa_xmx_span_lists<false>((const char *) mask->data, nullptr, mask->nb[1], nq, nq_chunk, n_chunks, n_span, list_stride,
                                    span_list.get(), span_count.get(), stream);
        }

        // enough work-groups from the chunks alone: split a chunk's list only when there are few chunks
        const int ns = std::max(1, target_wgs / (n_kvh * n_chunks));
        const int R8 = R8_max;
        ggml_sycl_pool_alloc<float> lpart_o(ctx.pool(), (size_t) n_chunks * n_kvh * ns * R8 * FA_XMX_D);
        ggml_sycl_pool_alloc<float> lpart_ml(ctx.pool(), (size_t) n_chunks * n_kvh * ns * R8 * 2);

        fa_xmx_args a = {};
        a.Q           = (const char *) Q->data;
        a.K           = K_data;
        a.V           = V_data;
        a.mask        = (const char *) (indices ? causal->data : mask->data);
        a.part_o      = lpart_o.get();
        a.part_ml     = lpart_ml.get();
        a.g           = g;
        a.nq          = nq;
        a.n_kv        = n_kv;
        a.nsplit      = ns;
        a.q_nb1       = Q->nb[1];
        a.q_nb2       = Q->nb[2];
        a.k_nb1       = k_nb1;
        a.k_nb2       = k_nb2;
        a.v_nb1       = v_nb1;
        a.v_nb2       = v_nb2;
        a.m_nb1       = indices ? causal->nb[1] : mask->nb[1];
        a.qscale      = scale * 1.44269504088896341f;
        a.span_list   = span_list.get();
        a.span_count  = span_count.get();
        a.list_stride = list_stride;
        a.nq_chunk    = nq_chunk;
        a.n_kvh       = n_kvh;
        a.mask_bits   = mask_bits.get();
        a.mask_fill   = fill;
        a.mask_zero   = zero;
        if (indices) {
            fa_xmx_list_launch_rg<true>(R8 / 8, a, n_chunks, stream);
        } else {
            fa_xmx_list_launch_rg<false>(R8 / 8, a, n_chunks, stream);
        }

        const float * po = lpart_o.get();
        const float * pml = lpart_ml.get();
        float *       out = (float *) dst->data;
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) n_chunks * n_kvh * R8 * FA_XMX_D), sycl::range<1>(FA_XMX_D)),
            [=](sycl::nd_item<1> it) {
                const int row = it.get_group(0);
                const int r   = row % R8;
                const int kvh = (row / R8) % n_kvh;
                const int c   = row / (R8 * n_kvh);
                const int j0  = c * nq_chunk;
                if (r >= g * sycl::min(nq_chunk, nq - j0)) {
                    return;
                }
                const int    d    = it.get_local_id(0);
                const size_t base = ((size_t) c * n_kvh + kvh) * ns;
                float        mx   = -FLT_MAX;
                for (int s = 0; s < ns; ++s) {
                    mx = sycl::fmax(mx, pml[((base + s) * R8 + r) * 2]);
                }
                float num = 0.0f, den = 0.0f;
                for (int s = 0; s < ns; ++s) {
                    const size_t i = (base + s) * R8 + r;
                    const float  w = sycl::exp2(pml[i * 2] - mx);
                    num += w * po[i * FA_XMX_D + d];
                    den += w * pml[i * 2 + 1];
                }
                const int j = r / g;
                const int h = r - j * g;
                out[(size_t) (j0 + j) * o_nb2 + (size_t) (kvh * g + h) * o_nb1 + d] = num / den;
            });
        return;
    }

    // split the cache so that about GGML_SYCL_FA_XMX_WGS work-groups run
    int nsplit = std::max(1, std::min(nblk, (target_wgs + n_kvh - 1) / n_kvh));
    const int bps = (nblk + nsplit - 1) / nsplit;
    nsplit        = (nblk + bps - 1) / bps;

    ggml_sycl_pool_alloc<float> part_o(ctx.pool(), (size_t) n_kvh * nsplit * R8_max * FA_XMX_D);
    ggml_sycl_pool_alloc<float> part_ml(ctx.pool(), (size_t) n_kvh * nsplit * R8_max * 2);

    fa_xmx_args a;
    a.K         = K_data;
    a.V         = V_data;
    a.span_live = live;
    a.k_hb      = k_hb;
    a.v_hb      = v_hb;
    a.k_hx      = k_hx;
    a.v_hx      = v_hx;
    a.part_o  = part_o.get();
    a.part_ml = part_ml.get();
    a.g       = g;
    a.n_kv    = n_kv;
    a.nsplit  = nsplit;
    a.bps     = bps;
    a.q_nb1   = Q->nb[1];
    a.q_nb2   = Q->nb[2];
    a.k_nb1   = k_nb1;
    a.k_nb2   = k_nb2;
    a.v_nb1   = v_nb1;
    a.v_nb2   = v_nb2;
    a.m_nb1   = mask->nb[1];
    a.qscale  = scale * 1.44269504088896341f;

    const float * po    = part_o.get();
    const float * pml   = part_ml.get();
    for (int j0 = 0; j0 < nq; j0 += nq_chunk) {
        const int nqc = std::min(nq_chunk, nq - j0);
        const int M   = g * nqc;
        const int RG  = (M + 7) / 8;
        const int R8  = RG * 8;
        a.Q    = (const char *) Q->data + j0 * Q->nb[1];
        a.mask = (const char *) mask->data + j0 * mask->nb[1];
        a.nq   = nqc;
        if (direct) {
            fa_xmx_launch_rg<true, true>(RG, a, n_kvh, stream);
        } else if (q8) {
            fa_xmx_launch_rg<true>(RG, a, n_kvh, stream);
        } else {
            fa_xmx_launch_rg<false>(RG, a, n_kvh, stream);
        }

        // merge the slices: dst[j][kvh g + c][d] = sum_s w_s O_s / sum_s w_s l_s, w_s = exp2(m_s - max_s m_s)
        float * out = (float *) dst->data + j0 * o_nb2;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n_kvh * M * FA_XMX_D), sycl::range<1>(FA_XMX_D)),
                             [=](sycl::nd_item<1> it) {
                                 const int row = it.get_group(0);
                                 const int kvh = row / M;
                                 const int r   = row - kvh * M;
                                 const int d   = it.get_local_id(0);
                                 float     mx  = -FLT_MAX;
                                 for (int s = 0; s < nsplit; ++s) {
                                     mx = sycl::fmax(mx, pml[(((size_t) kvh * nsplit + s) * R8 + r) * 2]);
                                 }
                                 float num = 0.0f, den = 0.0f;
                                 for (int s = 0; s < nsplit; ++s) {
                                     const size_t i = ((size_t) kvh * nsplit + s) * R8 + r;
                                     const float  w = sycl::exp2(pml[i * 2] - mx);
                                     num += w * po[i * FA_XMX_D + d];
                                     den += w * pml[i * 2 + 1];
                                 }
                                 const int j = r / g;
                                 const int c = r - j * g;
                                 out[j * o_nb2 + (size_t) (kvh * g + c) * o_nb1 + d] = num / den;
                             });
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_UNUSED(indices);
    GGML_UNUSED(causal);
    GGML_UNUSED(fill);
    GGML_UNUSED(zero);
    GGML_ABORT("ESIMD not available");
#endif
}

void ggml_sycl_flash_attn_ext_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    fa_xmx_impl(ctx, dst, nullptr, nullptr, -INFINITY, 0.0f);
}

void ggml_sycl_flash_attn_ext_xmx_qsa(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const ggml_tensor * indices, const ggml_tensor * causal, float fill, float zero) {
    fa_xmx_impl(ctx, dst, indices, causal, fill, zero);
}
