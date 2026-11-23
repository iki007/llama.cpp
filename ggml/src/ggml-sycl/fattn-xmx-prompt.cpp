#include "fattn-xmx-prompt.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <type_traits>

#include "convert.hpp"
#include "fattn-common.hpp"

// Thread layout (FlashAttention-2): rows r = j g + c (query j, head c of the KV head), 8 rows a thread. Per block of
// XP_BK tokens the thread scores its rows against the block (S = Q K^T, Q kept in registers as the DPAS A operand,
// K loaded transposed, which is the VNNI B operand), keeps the running row max and sum itself (no SLM, no
// barriers), turns S into P = exp2(S - max) in place, which is already the A operand of O += P V (V loaded
// VNNI-transformed), and owns all D dims of O. Q is scaled by scale*log2(e), the mask and bias by log2(e).
//
// Block classes (one pre-pass a call, [n_q][n_blk]): XP_DEAD = the mask is -inf for the whole block (skipped),
// >= 0 = the mask is 0 and every key has that bin (one bias value a row: at depth, every key past the relative extent
// has the last bin), XP_MIXED = mask and bias per token.
//
// A q8_0 cache (Q8) is read in place: all 16 threads share each block, so dequantizing it in every thread would cost
// ~2x the DPAS time in ALU work. Instead each thread dequantizes one q8_0 block (32 dims) of 16 tokens of K and of V,
// once a work-group, into SLM as the B operands the f16 path loads (two 32 KB buffers: block b + 1 is written while
// block b is read, one barrier a block), and every thread reads its operands from SLM.

namespace {

namespace esimd   = sycl::ext::intel::esimd;
namespace xmx     = sycl::ext::intel::esimd::xmx;
namespace esimd_x = sycl::ext::intel::experimental::esimd;

constexpr int     XP_D     = 128;
constexpr int     XP_BK    = 64;  // KV tokens a block
constexpr int     XP_NT    = 16;  // threads a work-group
constexpr int32_t XP_DEAD  = -1;
constexpr int32_t XP_MIXED = -2;
constexpr float   XP_LOG2E = 1.44269504088896341f;

struct xp_args {
    const char *       Q;
    size_t             q_nb1, q_nb2;
    const sycl::half * K;
    const sycl::half * V;
    size_t             k_nb1, k_nb2, v_nb1, v_nb2;
    const char *       mask;
    size_t             m_nb1;
    const int32_t *    cls;    // [nq][n_blk]
    const float *      table;  // [nq][n_head][n_bins] (bins innermost) or null
    const int32_t *    idx;    // [nq][idx_stride], or null with pos
    int64_t            idx_stride;
    const float *      pos;    // bins from positions: this stream's [pos_nkv cells, nq queries], extent
    int64_t            pos_nkv;
    int                extent;
    int                n_bins;
    const float *      sinks;  // [n_head] or null
    float *            dst;    // [nq][n_head][D]
    int                nq, n_kv, n_head, g, n_blk, n_rg, n_wg_kvh;
    float              qscale;
    // split-K: work-group (KV head, split, row chunk); with nsplit > 1 each split writes its unnormalized rows and
    // their max and sum, [nsplit][n_head * nq rows (KV head major)][D] and [..][2], and xp_merge finishes them
    int                nsplit, bps;
    float *            part_o;
    float *            part_ml;
    int                prefetch;  // the next block's K and V to L1, one 64-byte x 32-row tile a thread
    // a quantized cache runs in chunks converted to f16: this launch's K / V are the chunk (n_kv cells), its block 0
    // is global block kv_blk0 (classes, mask, bias), it continues the rows of the earlier chunks (cont) and always
    // leaves partial rows (part_always) for xp_merge
    int                n_blk_loc, kv_blk0, cont, part_always;
    // a q8_0 cache read in place: per KV head, bytes added to the base (head-major) or dwords added to the 2D x offset
    // (heads interleaved in the token row), as fattn-xmx-dec.cpp
    size_t             k_hb, v_hb;
    int                k_hx, v_hx;
};

template <typename F> struct xp_grf256 {
    F f;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL { f(it); }

    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    }
};

// row maxima / sums of an [8][16] block
ESIMD_INLINE esimd::simd<float, 8> xp_row_max(esimd::simd<float, 128> p) {
    using namespace esimd;
    auto            p2 = p.bit_cast_view<float, 8, 16>();
    simd<float, 64> s8 = max(simd<float, 64>(p2.select<8, 1, 8, 1>(0, 0)), simd<float, 64>(p2.select<8, 1, 8, 1>(0, 8)));
    auto            v8 = s8.bit_cast_view<float, 8, 8>();
    simd<float, 32> s4 = max(simd<float, 32>(v8.select<8, 1, 4, 1>(0, 0)), simd<float, 32>(v8.select<8, 1, 4, 1>(0, 4)));
    auto            v4 = s4.bit_cast_view<float, 8, 4>();
    simd<float, 16> s2 = max(simd<float, 16>(v4.select<8, 1, 2, 1>(0, 0)), simd<float, 16>(v4.select<8, 1, 2, 1>(0, 2)));
    return max(simd<float, 8>(s2.select<8, 2>(0)), simd<float, 8>(s2.select<8, 2>(1)));
}

ESIMD_INLINE esimd::simd<float, 8> xp_row_sum(esimd::simd<float, 128> p) {
    using namespace esimd;
    auto            p2 = p.bit_cast_view<float, 8, 16>();
    simd<float, 64> s8 = p2.select<8, 1, 8, 1>(0, 0) + p2.select<8, 1, 8, 1>(0, 8);
    auto            v8 = s8.bit_cast_view<float, 8, 8>();
    simd<float, 32> s4 = v8.select<8, 1, 4, 1>(0, 0) + v8.select<8, 1, 4, 1>(0, 4);
    auto            v4 = s4.bit_cast_view<float, 8, 4>();
    simd<float, 16> s2 = v4.select<8, 1, 2, 1>(0, 0) + v4.select<8, 1, 2, 1>(0, 2);
    return s2.select<8, 2>(0) + s2.select<8, 2>(1);
}

template <int D, bool Q8> ESIMD_INLINE void xp_kernel(const xp_args & a, const sycl::nd_item<1> & it) {
    using namespace esimd;
    constexpr int KS = D / 16;      // k-steps of a score, dim groups of O
    constexpr int NB = XP_BK / 16;  // 16-token groups of a block
    // Q8 SLM: per buffer the K tiles [n][ks] then the V tiles [n][dg], each one 16 x 16 f16 B operand
    constexpr uint32_t XQ_TILE = 512;
    constexpr uint32_t XQ_V    = NB * KS * XQ_TILE;
    constexpr uint32_t XQ_BUF  = 2 * XQ_V;
    static_assert(!Q8 || NB * (D / 32) == XP_NT, "a thread dequantizes one q8_0 block of 16 tokens");
    if constexpr (Q8) {
        slm_init<2 * XQ_BUF>();
    }

    const int t     = (int) it.get_local_id(0);
    const int wg    = (int) it.get_group(0);
    const int chunk = wg % a.n_wg_kvh;  // the row chunks of one (KV head, split) are neighbours: they share K / V in L2
    const int split = (wg / a.n_wg_kvh) % a.nsplit;
    const int kvh   = wg / (a.n_wg_kvh * a.nsplit);
    const int rgi   = chunk * XP_NT + t;
    // Q8: a thread without rows still dequantizes its share and joins the barriers (its rows are all invalid)
    if (!Q8 && rgi >= a.n_rg) {
        return;
    }
    const int M = a.nq * a.g;

    // the thread's rows: query, head, the byte offset of the query's class row
    int               jr[8], hr[8];
    simd<uint32_t, 8> cls_off, tab_off;  // byte offsets: the query's class row, the row's bias table
    simd_mask<8>      valid;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r  = rgi * 8 + i;
        const int rv = r < M ? r : M - 1;
        jr[i]        = rv / a.g;
        hr[i]        = kvh * a.g + (rv - jr[i] * a.g);
        cls_off[i]   = (uint32_t) jr[i] * (uint32_t) a.n_blk * 4u;
        tab_off[i]   = ((uint32_t) jr[i] * (uint32_t) a.n_head + (uint32_t) hr[i]) * (uint32_t) a.n_bins * 4u;
        valid[i]     = r < M;
    }

    // Q rows, scaled, f16, as the A operands of the KS k-steps: [ks][8 rows][16 dims]
    simd<sycl::half, KS * 128> qa;
    {
        simd<sycl::half, 8 * D> qall = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (valid[i]) {
                const float *  qp = (const float *) (a.Q + (size_t) jr[i] * a.q_nb1 + (size_t) hr[i] * a.q_nb2);
                simd<float, D> q;
#pragma unroll
                for (int k = 0; k < D / 64; ++k) {
                    q.template select<64, 1>(64 * k) = block_load<float, 64>(qp + 64 * k);
                }
                qall.template select<D, 1>(D * i) = convert<sycl::half>(q * a.qscale);
            }
        }
        auto q2 = qall.template bit_cast_view<sycl::half, 8, D>();
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            qa.template select<128, 1>(128 * ks) = q2.template select<8, 1, 16, 1>(0, 16 * ks);
        }
    }

    const sycl::half * Kh     = (const sycl::half *) ((const char *) a.K + (size_t) kvh * a.k_nb2);
    const sycl::half * Vh     = (const sycl::half *) ((const char *) a.V + (size_t) kvh * a.v_nb2);
    const unsigned     surf_w = D * sizeof(sycl::half) - 1;
    const unsigned     surf_h = (unsigned) a.n_kv - 1;
    // K: a transposed load of 8 dim pairs x 32 tokens = the VNNI B operands of two 16-token groups
    esimd_x::config_2d_mem_access<uint32_t, 8, 32, 1>    k_desc((const uint32_t *) Kh, surf_w, surf_h, (unsigned) a.k_nb1 - 1, 0, 0);
    // V: one VNNI-transformed load of 32 tokens x 2 blocks of 16 dims = four B operands
    esimd_x::config_2d_mem_access<sycl::half, 16, 32, 2> v_desc(Vh, surf_w, surf_h, (unsigned) a.v_nb1 - 1, 0, 0);

    // Q8: thread t dequantizes q8_0 block qj (dims 32 qj..32 qj + 31, 34 bytes at byte 34 qj of the head row) of the
    // 16 tokens qn of a block. Its 2D dword loads start at the dword holding the block's scale, which is at byte 0
    // (even block) or 2 (odd block) of that dword: K as [9 dwords][16 tokens] (transposed, so a dim pair of the 16
    // tokens is a strided byte region of one dword row, already the VNNI B layout), V as [16 tokens][16 dwords]
    const int        qn  = t >> 2;
    const int        qj  = t & 3;
    const uint32_t * Kq  = (const uint32_t *) ((const char *) a.K + (size_t) kvh * a.k_hb);
    const uint32_t * Vq  = (const uint32_t *) ((const char *) a.V + (size_t) kvh * a.v_hb);
    const int        qkx = kvh * a.k_hx + 34 * qj / 4;
    const int        qvx = kvh * a.v_hx + 34 * qj / 4;
    esimd_x::config_2d_mem_access<uint32_t, 8, 16, 1>  kq8_desc(Kq, (unsigned) a.k_nb1 - 1, surf_h, (unsigned) a.k_nb1 - 1, qkx, 0);
    esimd_x::config_2d_mem_access<uint32_t, 1, 16, 1>  kq1_desc(Kq, (unsigned) a.k_nb1 - 1, surf_h, (unsigned) a.k_nb1 - 1, qkx + 8, 0);
    esimd_x::config_2d_mem_access<uint32_t, 16, 16, 1> vq_desc(Vq, (unsigned) a.v_nb1 - 1, surf_h, (unsigned) a.v_nb1 - 1, qvx, 0);
    simd<uint32_t, 9 * 16>  kw;  // this thread's K window of the block to dequantize next
    simd<uint32_t, 16 * 16> vw;  // and its V window
    auto q8_load = [&](int b) {
        const int y = b * XP_BK + 16 * qn;
        kq8_desc.set_y(y);
        kq1_desc.set_y(y);
        vq_desc.set_y(y);
        kw.template select<128, 1>(0)  = esimd_x::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(kq8_desc);
        kw.template select<16, 1>(128) = esimd_x::lsc_load_2d<uint32_t, 1, 16, 1, true, false>(kq1_desc);
        vw                             = esimd_x::lsc_load_2d<uint32_t, 16, 16, 1, false, false>(vq_desc);
    };
    // dequantized like the f16 copy (dequantize_q8_0: f16(int8) * f16 scale, the exact int8 rounded once by the
    // product), so the B operands equal the f16 path's bit for bit; ODD: the scale at byte 2 of the first dword
    auto q8_store_t = [&](int b, auto odd) {
        constexpr int  O    = decltype(odd)::value ? 2 : 0;
        const uint32_t base = (uint32_t) (b & 1) * XQ_BUF + (uint32_t) (qn * KS + 2 * qj) * XQ_TILE;
        // K: tile c2 = dim pairs 8 c2..8 c2 + 7 of the block, a pair = bytes O + 2 + 2p of the 16 tokens
        auto                       kh = kw.template bit_cast_view<sycl::half>();
        auto                       kb = kw.template bit_cast_view<int8_t, 9 * 16, 4>();
        const simd<sycl::half, 16> kd = kh.template select<16, 2>(O / 2);
        const simd<sycl::half, 32> kr = kd.template replicate_vs_w_hs<16, 1, 2, 0>(0);
#pragma unroll
        for (int c2 = 0; c2 < 2; ++c2) {
            simd<sycl::half, 256> bt;
#pragma unroll
            for (int kp = 0; kp < 8; ++kp) {
                const int              byte = O + 2 + 2 * (8 * c2 + kp);
                const simd<int8_t, 32> qv   = kb.template select<16, 1, 2, 1>((byte / 4) * 16, byte % 4);
                bt.template select<32, 1>(32 * kp) = convert<sycl::half>(qv) * kr;
            }
            slm_block_store<sycl::half, 256>(base + c2 * XQ_TILE, bt);
        }
        // V: [16 tokens][32 dims] from byte O + 2 of each row, then token pairs interleaved into the VNNI layout
        auto                       vb8 = vw.template bit_cast_view<int8_t, 16, 64>();
        auto                       vh  = vw.template bit_cast_view<sycl::half, 16, 32>();
        const simd<int8_t, 512>    vq  = vb8.template select<16, 1, 32, 1>(0, O + 2);
        const simd<sycl::half, 16> vd  = vh.template select<16, 1, 1, 1>(0, O / 2);
        simd<sycl::half, 512>      hv  = convert<sycl::half>(vq) * vd.template replicate_vs_w_hs<16, 1, 32, 0>(0);
        auto                       hv2 = hv.template bit_cast_view<sycl::half, 16, 32>();
#pragma unroll
        for (int c2 = 0; c2 < 2; ++c2) {
            simd<sycl::half, 256> vt;
            auto                  vt2 = vt.template bit_cast_view<sycl::half, 8, 32>();
            vt2.template select<8, 1, 16, 2>(0, 0) = hv2.template select<8, 2, 16, 1>(0, 16 * c2);
            vt2.template select<8, 1, 16, 2>(0, 1) = hv2.template select<8, 2, 16, 1>(1, 16 * c2);
            slm_block_store<sycl::half, 256>(base + XQ_V + c2 * XQ_TILE, vt);
        }
    };
    auto q8_store = [&](int b) {
        if (qj & 1) {
            q8_store_t(b, std::true_type{});
        } else {
            q8_store_t(b, std::false_type{});
        }
    };

    simd<float, 8>       m = -FLT_MAX;  // running row max (log2 domain)
    simd<float, 8>       l = 0.0f;      // running row sum
    simd<float, KS * 128> o = 0.0f;     // [dim group][8 rows][16 dims]

    const bool   to_part = a.nsplit > 1 || a.part_always;
    const size_t row0    = (size_t) split * a.n_head * a.nq + (size_t) kvh * M + (size_t) rgi * 8;
    if (a.cont) {
        // the running rows of the earlier chunks
        simd<float, 16> ml = 0.0f;
        if (rgi * 8 + 8 <= M) {
            ml = block_load<float, 16>(a.part_ml + row0 * 2);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (valid[i]) {
                    ml.template select<2, 1>(2 * i) = block_load<float, 2>(a.part_ml + (row0 + i) * 2);
                }
            }
        }
        m = ml.template select<8, 2>(0);
        l = ml.template select<8, 2>(1);
        m.merge(simd<float, 8>(-FLT_MAX), !valid);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (valid[i]) {
#pragma unroll
                for (int dg = 0; dg < KS; ++dg) {
                    o.template select<16, 1>(128 * dg + 16 * i) = block_load<float, 16>(a.part_o + (row0 + i) * D + 16 * dg);
                }
            }
        }
    }

    const int b0 = split * a.bps;
    const int b1 = b0 + a.bps < a.n_blk_loc ? b0 + a.bps : a.n_blk_loc;
    if constexpr (Q8) {
        if (b0 < b1) {
            q8_load(b0);
        }
    }
    for (int b = b0; b < b1; ++b) {
        if constexpr (Q8) {
            // block b to SLM buffer b & 1, whose last readers (block b - 2) all passed the barrier of block b - 1; the
            // load of block b + 1 runs during the barrier and the compute of block b. Every thread takes part in every
            // block, also one dead for its rows
            q8_store(b);
            if (b + 1 < b1) {
                q8_load(b + 1);
            }
            barrier();
        }
        simd<int32_t, 8> cl = gather<int32_t, 8>(a.cls, cls_off + (uint32_t) (b + a.kv_blk0) * 4u);
        cl.merge(simd<int32_t, 8>(XP_DEAD), !valid);
        if ((cl == XP_DEAD).all()) {
            continue;
        }
        const int tok0  = b * XP_BK;                // in this launch's K / V
        const int gtok0 = (b + a.kv_blk0) * XP_BK;  // in the mask and the bias indices
        if (!Q8 && a.prefetch && b + 1 < b1) {
            // 16 tiles: K or V (t / 8), 64-byte column (t % 4), 32-token half of the block ((t / 4) % 2)
            const int part = t & 7;
            esimd_x::lsc_prefetch_2d<uint32_t, 16, 32, 1, cache_hint::cached, cache_hint::cached>(
                (const uint32_t *) (t < 8 ? Kh : Vh), surf_w, surf_h, (unsigned) (t < 8 ? a.k_nb1 : a.v_nb1) - 1,
                16 * (part & 3), tok0 + XP_BK + 32 * (part >> 2));
        }

        // S = Q K^T
        simd<float, NB * 128> s = 0.0f;
        const uint32_t        xq = (uint32_t) (b & 1) * XQ_BUF;  // Q8: this block's SLM buffer
        if constexpr (Q8) {
            // the same DPAS sequence as the f16 loads below: per 16-token group, k-steps in order
#pragma unroll
            for (int n = 0; n < NB; ++n) {
#pragma unroll
                for (int ks = 0; ks < KS; ++ks) {
                    const simd<sycl::half, 256> bk = slm_block_load<sycl::half, 256>(xq + (uint32_t) (n * KS + ks) * XQ_TILE);
                    s.template select<128, 1>(128 * n) = xmx::dpas<8, 8, float, float>(
                        simd<float, 128>(s.template select<128, 1>(128 * n)), bk, simd<sycl::half, 128>(qa.template select<128, 1>(128 * ks)));
                }
            }
        } else {
#pragma unroll
            for (int n2 = 0; n2 < NB / 2; ++n2) {
                k_desc.set_y(tok0 + 32 * n2);
#pragma unroll
                for (int ks = 0; ks < KS; ++ks) {
                    k_desc.set_x(ks * 8);
                    simd<uint32_t, 256> kb = esimd_x::lsc_load_2d<uint32_t, 8, 32, 1, true, false>(k_desc);
                    auto                k2 = kb.template bit_cast_view<uint32_t, 8, 32>();
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int           n  = 2 * n2 + c;
                        simd<uint32_t, 128> bk = k2.template select<8, 1, 16, 1>(0, 16 * c);
                        s.template select<128, 1>(128 * n) = xmx::dpas<8, 8, float, float>(
                            simd<float, 128>(s.template select<128, 1>(128 * n)), bk.template bit_cast_view<sycl::half>().read(),
                            simd<sycl::half, 128>(qa.template select<128, 1>(128 * ks)));
                    }
                }
            }
        }

        // mask and bias: one value a row for dead and one-bin rows, per token for mixed rows
        simd<float, 8> add = 0.0f;
        if (a.table) {
            const simd_mask<8> one_bin = (cl >= 0) & (cl < a.n_bins);
            simd<uint32_t, 8>  toff    = tab_off + simd<uint32_t, 8>(cl) * 4u;
            toff.merge(tab_off, !one_bin);
            const simd<float, 8> tv = convert<float>(convert<sycl::half>(gather<float, 8>(a.table, toff)));
            add.merge(tv * XP_LOG2E, one_bin);
        }
        add.merge(simd<float, 8>(-INFINITY), cl == XP_DEAD);
        const simd<float, 128> addr = add.template replicate_vs_w_hs<8, 1, 16, 0>(0);
#pragma unroll
        for (int n = 0; n < NB; ++n) {
            s.template select<128, 1>(128 * n) += addr;
        }
        if ((cl == XP_MIXED).any()) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (cl[i] == XP_MIXED) {
                    const sycl::half * mp = (const sycl::half *) (a.mask + (size_t) jr[i] * a.m_nb1) + gtok0;
                    simd<float, XP_BK> mk = convert<float>(block_load<sycl::half, XP_BK>(mp));
                    if (a.table) {
                        simd<int32_t, XP_BK> bin;
                        if (a.pos) {
                            // clamp(query position - cell position, 0, extent)
                            const simd<float, XP_BK> cp = block_load<float, XP_BK>(a.pos + gtok0);
                            simd<int32_t, XP_BK>     d  = convert<int32_t>(simd<float, XP_BK>(a.pos[a.pos_nkv + jr[i]]) - cp);
                            d.merge(simd<int32_t, XP_BK>(0), d < 0);
                            d.merge(simd<int32_t, XP_BK>(a.extent), d > a.extent);
                            bin = d;
                        } else {
                            bin = block_load<int32_t, XP_BK>(a.idx + (size_t) jr[i] * a.idx_stride + gtok0);
                        }
                        const simd_mask<XP_BK>     ok  = simd<uint32_t, XP_BK>(bin) < (uint32_t) a.n_bins;
                        simd<uint32_t, XP_BK>      off = simd<uint32_t, XP_BK>(bin) * 4u;
                        off.merge(simd<uint32_t, XP_BK>(0u), !ok);
                        const float * tp = a.table + ((size_t) jr[i] * a.n_head + hr[i]) * a.n_bins;
                        simd<float, XP_BK> tv;
#pragma unroll
                        for (int h = 0; h < XP_BK / 32; ++h) {
                            tv.template select<32, 1>(32 * h) = gather<float, 32>(tp, simd<uint32_t, 32>(off.template select<32, 1>(32 * h)));
                        }
                        const simd<float, XP_BK> mb = convert<float>(convert<sycl::half>(mk + tv));
                        mk.merge(mb, ok);
                    }
                    mk *= XP_LOG2E;
#pragma unroll
                    for (int n = 0; n < NB; ++n) {
                        s.template select<16, 1>(128 * n + 16 * i) += mk.template select<16, 1>(16 * n);
                    }
                }
            }
        }

        // online softmax
        simd<float, 8> bm = xp_row_max(s.template select<128, 1>(0));
#pragma unroll
        for (int n = 1; n < NB; ++n) {
            bm = max(bm, xp_row_max(s.template select<128, 1>(128 * n)));
        }
        const simd<float, 8>   m_new = max(m, bm);
        const simd<float, 8>   alpha = exp2(m - m_new);
        const simd<float, 128> mrep  = m_new.template replicate_vs_w_hs<8, 1, 16, 0>(0);
        m                            = m_new;
        simd<float, 8> bs            = 0.0f;
#pragma unroll
        for (int n = 0; n < NB; ++n) {
            simd<float, 128> p = exp2(s.template select<128, 1>(128 * n) - mrep);
            s.template select<128, 1>(128 * n) = p;
            bs += xp_row_sum(p);
        }
        l = l * alpha + bs;
        const simd<float, 128> arep = alpha.template replicate_vs_w_hs<8, 1, 16, 0>(0);
#pragma unroll
        for (int dg = 0; dg < KS; ++dg) {
            o.template select<128, 1>(128 * dg) *= arep;
        }

        // O += P V; Q8: tile (n, dg) of SLM, in the order of the f16 path (per dim group: tokens 0..15 then 16..31 of
        // each 32-token pair); f16: a load's block c holds dims 32 dg2 + 16 c, its tokens 0..15 then 16..31 (each 256
        // halves)
        if constexpr (Q8) {
#pragma unroll
            for (int n2 = 0; n2 < NB / 2; ++n2) {
                const simd<sycl::half, 128> pa0 = convert<sycl::half>(simd<float, 128>(s.template select<128, 1>(128 * (2 * n2))));
                const simd<sycl::half, 128> pa1 = convert<sycl::half>(simd<float, 128>(s.template select<128, 1>(128 * (2 * n2 + 1))));
#pragma unroll
                for (int dg = 0; dg < KS; ++dg) {
                    const simd<sycl::half, 256> vb0 = slm_block_load<sycl::half, 256>(xq + XQ_V + (uint32_t) (2 * n2 * KS + dg) * XQ_TILE);
                    const simd<sycl::half, 256> vb1 = slm_block_load<sycl::half, 256>(xq + XQ_V + (uint32_t) ((2 * n2 + 1) * KS + dg) * XQ_TILE);
                    o.template select<128, 1>(128 * dg) = xmx::dpas<8, 8, float, float>(simd<float, 128>(o.template select<128, 1>(128 * dg)), vb0, pa0);
                    o.template select<128, 1>(128 * dg) = xmx::dpas<8, 8, float, float>(simd<float, 128>(o.template select<128, 1>(128 * dg)), vb1, pa1);
                }
            }
        } else {
#pragma unroll
            for (int n2 = 0; n2 < NB / 2; ++n2) {
                const simd<sycl::half, 128> pa0 = convert<sycl::half>(simd<float, 128>(s.template select<128, 1>(128 * (2 * n2))));
                const simd<sycl::half, 128> pa1 = convert<sycl::half>(simd<float, 128>(s.template select<128, 1>(128 * (2 * n2 + 1))));
                v_desc.set_y(tok0 + 32 * n2);
#pragma unroll
                for (int dg2 = 0; dg2 < KS / 2; ++dg2) {
                    v_desc.set_x(32 * dg2);
                    simd<sycl::half, 1024> vb = esimd_x::lsc_load_2d<sycl::half, 16, 32, 2, false, true>(v_desc);
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int dg = 2 * dg2 + c;
                        o.template select<128, 1>(128 * dg) = xmx::dpas<8, 8, float, float>(
                            simd<float, 128>(o.template select<128, 1>(128 * dg)), simd<sycl::half, 256>(vb.template select<256, 1>(512 * c)), pa0);
                        o.template select<128, 1>(128 * dg) = xmx::dpas<8, 8, float, float>(
                            simd<float, 128>(o.template select<128, 1>(128 * dg)), simd<sycl::half, 256>(vb.template select<256, 1>(512 * c + 256)), pa1);
                    }
                }
            }
        }
    }

    if (to_part) {
        // this split's unnormalized rows and their max and sum
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (valid[i]) {
#pragma unroll
                for (int dg = 0; dg < KS; ++dg) {
                    block_store<float, 16>(a.part_o + (row0 + i) * D + 16 * dg, simd<float, 16>(o.template select<16, 1>(128 * dg + 16 * i)));
                }
            }
        }
        simd<float, 16> ml;
        ml.template select<8, 2>(0) = m;
        ml.template select<8, 2>(1) = l;
        if (rgi * 8 + 8 <= M) {
            block_store<float, 16>(a.part_ml + row0 * 2, ml);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (valid[i]) {
                    block_store<float, 2>(a.part_ml + (row0 + i) * 2, simd<float, 2>(ml.template select<2, 1>(2 * i)));
                }
            }
        }
        return;
    }

    // normalize (with the sink logit of the head in the denominator) and store the rows
    simd<float, 8> sc = 1.0f;
    simd<float, 8> li = l;
    if (a.sinks) {
        simd<float, 8> sk;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            sk[i] = a.sinks[hr[i]];
        }
        sk *= XP_LOG2E;
        const simd<float, 8> mf = max(m, sk);
        sc                      = exp2(m - mf);
        li                      = l * sc + exp2(sk - mf);
    }
    simd<float, 8> fac = sc / li;
    fac.merge(simd<float, 8>(0.0f), li <= 0.0f);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (!valid[i]) {
            continue;
        }
        const float f = fac[i];
        float *     d = a.dst + ((size_t) jr[i] * a.n_head + hr[i]) * D;
#pragma unroll
        for (int dg = 0; dg < KS; ++dg) {
            block_store<float, 16>(d + 16 * dg, simd<float, 16>(o.template select<16, 1>(128 * dg + 16 * i)) * f);
        }
    }
}

// the class of every (query, block): reads the mask and the bin indices once
void xp_classes(const sycl::half * mask, const size_t m_stride, const fattn_indexed_bias bias, const int64_t q0,
                const int nq, const int n_blk, int32_t * cls, dpct::queue_ptr stream) {
    const bool has_bias = bias.table != nullptr;
    stream->parallel_for(sycl::range<2>((size_t) nq, (size_t) n_blk), [=](sycl::id<2> id) {
        const int        j  = (int) id[0];
        const int        b  = (int) id[1];
        const uint16_t * mp = (const uint16_t *) mask + (size_t) j * m_stride + (size_t) b * XP_BK;
        const int64_t    k0 = (int64_t) b * XP_BK;
        const int32_t    b0 = has_bias ? bias.bin(q0 + j, k0) : 0;
        bool             dead = true, zero = true, same = true;
        for (int k = 0; k < XP_BK; ++k) {
            const uint16_t mv = mp[k];
            dead &= mv == 0xFC00;              // f16 -inf
            zero &= (mv & 0x7FFF) == 0;        // +-0
            if (has_bias) {
                same &= bias.bin(q0 + j, k0 + k) == b0;
            }
        }
        cls[(size_t) j * n_blk + b] = dead ? XP_DEAD : (zero && same && b0 >= 0 ? b0 : XP_MIXED);
    });
}

// the splits of every row: rescale to the common max, add up, the sink in the denominator, normalize; one
// work-group of D work-items a row (KV head major: row = kvh * M + j * g + c)
void xp_merge(const float * part_o, const float * part_ml, const float * sinks, float * dst, const int nsplit,
              const int nq, const int n_head, const int g, dpct::queue_ptr stream) {
    const int n_rows = n_head * nq;
    stream->parallel_for(sycl::nd_range<1>((size_t) n_rows * XP_D, XP_D), [=](sycl::nd_item<1> it) {
        const int    row = (int) it.get_group(0);
        const int    d   = (int) it.get_local_id(0);
        const int    M   = nq * g;
        const int    kvh = row / M;
        const int    r   = row - kvh * M;
        const int    j   = r / g;
        const int    h   = kvh * g + (r - j * g);
        float        mx  = -FLT_MAX;
        for (int s = 0; s < nsplit; ++s) {
            mx = sycl::fmax(mx, part_ml[((size_t) s * n_rows + row) * 2]);
        }
        const float sk = sinks ? sinks[h] * XP_LOG2E : -FLT_MAX;
        mx             = sycl::fmax(mx, sk);
        float acc = 0.0f, sum = 0.0f;
        for (int s = 0; s < nsplit; ++s) {
            const size_t q = (size_t) s * n_rows + row;
            const float  w = sycl::exp2(part_ml[q * 2] - mx);
            acc += part_o[q * XP_D + d] * w;
            sum += part_ml[q * 2 + 1] * w;
        }
        if (sinks) {
            sum += sycl::exp2(sk - mx);
        }
        dst[((size_t) j * n_head + h) * XP_D + d] = sum > 0.0f ? acc / sum : 0.0f;
    });
}

// a q8_0 cache the 2D dword loads read in place (as fa_xmx_q8_layout of fattn-xmx-dec.cpp): contiguous blocks,
// 64-byte aligned stream bases, a 16-byte aligned row pitch, each head at a 64-byte aligned offset (hb) or inside the
// token row (hx, the KV cache's heads interleaved per token). Depends on the data pointers: checked at the launch
static bool xp_q8_layout(const ggml_tensor * t, size_t & hb, int & hx) {
    const size_t row = XP_D / QK8_0 * sizeof(block_q8_0);
    if (t->type != GGML_TYPE_Q8_0 || t->nb[0] != sizeof(block_q8_0) || t->nb[1] % 16 != 0 || t->nb[1] < 64 ||
        t->nb[1] >= (1u << 24)) {
        return false;
    }
    for (int64_t s = 0; s < t->ne[3]; ++s) {
        if (((uintptr_t) t->data + s * t->nb[3]) % 64 != 0) {
            return false;
        }
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

}  // namespace

bool ggml_sycl_flash_attn_ext_xmx_prompt_supported(int device, const ggml_tensor * dst) {
    static const int enabled = ggml_sycl_get_env("GGML_SYCL_FA_XMX_PROMPT", 1);
    static const int min_q   = ggml_sycl_get_env("GGML_SYCL_FA_XMX_PROMPT_MIN_Q", 32);
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
    float               max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    // f16 is read in place; a quantized cache with an f16 converter runs in chunks converted to f16
    const bool f16 = K->type == GGML_TYPE_F16;
    if (K->type != V->type || (!f16 && (ggml_get_to_fp16_nc_sycl(K->type) == nullptr || K->nb[1] % ggml_type_size(K->type) != 0 ||
                                        K->nb[2] % ggml_type_size(K->type) != 0 || V->nb[1] % ggml_type_size(V->type) != 0 ||
                                        V->nb[2] % ggml_type_size(V->type) != 0 || K->ne[0] % ggml_blck_size(K->type) != 0))) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        !mask || mask->type != GGML_TYPE_F16 || (sinks && sinks->type != GGML_TYPE_F32) || max_bias != 0.0f ||
        softcap != 0.0f || dst->src[5]) {
        return false;
    }
    if (Q->ne[0] != XP_D || K->ne[0] != XP_D || V->ne[0] != XP_D || Q->ne[1] < min_q || K->ne[1] % XP_BK != 0 ||
        K->ne[1] != V->ne[1] || K->ne[2] != V->ne[2] || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    const int64_t ns = Q->ne[3];
    if (K->ne[3] != ns || V->ne[3] != ns || dst->ne[3] != ns || mask->ne[2] != 1 || (mask->ne[3] != 1 && mask->ne[3] != ns) ||
        mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || mask->nb[0] != sizeof(sycl::half)) {
        return false;
    }
    // 2D block loads: 64-byte aligned head slices, a 16-byte aligned row pitch of at least 64 bytes (the data
    // pointers are checked at the launch, which falls back to the tile kernel)
    for (const ggml_tensor * t : { K, V }) {
        if (f16 && (t->nb[0] != sizeof(sycl::half) || t->nb[1] % 16 != 0 || t->nb[1] < 64 || t->nb[2] % 64 != 0)) {
            return false;
        }
    }
    if (Q->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
}

bool ggml_sycl_flash_attn_ext_xmx_prompt_data_ok(const ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (K->type != GGML_TYPE_F16) {
        return true;  // converted to a 64-byte aligned f16 chunk
    }
    for (int64_t s = 0; s < K->ne[3]; ++s) {
        if (((uintptr_t) K->data + s * K->nb[3]) % 64 != 0 || ((uintptr_t) V->data + s * V->nb[3]) % 64 != 0) {
            return false;
        }
    }
    return true;
}

void ggml_sycl_flash_attn_ext_xmx_prompt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    dpct::queue_ptr stream = ctx.stream();
    ggml_sycl_pool & pool  = ctx.pool();
    // the pool is a stack: the bias table, allocated first, is declared first
    ggml_sycl_pool_alloc<float>   bias_table_t(pool);
    ggml_sycl_pool_alloc<int32_t> cls(pool);
    ggml_sycl_pool_alloc<float>   part_o(pool);
    ggml_sycl_pool_alloc<float>   part_ml(pool);
    ggml_sycl_pool_alloc<char>    kv16(pool);  // a quantized cache: the f16 K and V of one chunk, head major
    fattn_indexed_bias            bias = {};
    if (dst->src[6]) {
        bias = ggml_sycl_fattn_indexed_bias(stream, dst, bias_table_t);
    }

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int nq     = (int) Q->ne[1];
    const int n_kv   = (int) K->ne[1];
    const int n_head = (int) Q->ne[2];
    const int n_kvh  = (int) K->ne[2];
    const int g      = n_head / n_kvh;
    const int n_blk  = n_kv / XP_BK;
    const int n_rg   = (nq * g + 7) / 8;
    const int n_wg   = (n_rg + XP_NT - 1) / XP_NT;
    cls.alloc((size_t) nq * n_blk);
    // split-K for occupancy: up to GGML_SYCL_FA_XMX_PROMPT_SPLIT splits of at least GGML_SYCL_FA_XMX_PROMPT_SPLIT_MIN
    // blocks (32 = 2048 keys; small values make the op tests take the split path)
    static const int max_split = ggml_sycl_get_env("GGML_SYCL_FA_XMX_PROMPT_SPLIT", 8);
    static const int prefetch  = ggml_sycl_get_env("GGML_SYCL_FA_XMX_PROMPT_PREFETCH", 1);
    static const int min_bps   = std::max(1, ggml_sycl_get_env("GGML_SYCL_FA_XMX_PROMPT_SPLIT_MIN", 32));
    // a quantized cache: chunks of GGML_SYCL_FA_XMX_PROMPT_CHUNK cells converted to f16 (32768: 128 MiB for 8 KV heads)
    static const int chunk_env = ggml_sycl_get_env("GGML_SYCL_FA_XMX_PROMPT_CHUNK", 32768);
    // a q8_0 cache in a layout the 2D loads take is read in place (GGML_SYCL_FA_XMX_PROMPT_Q8=0: converted like the
    // other types)
    static const int q8_env = ggml_sycl_get_env("GGML_SYCL_FA_XMX_PROMPT_Q8", 1);
    size_t           k_hb = 0, v_hb = 0;
    int              k_hx = 0, v_hx = 0;
    const bool       q8   = q8_env && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 && xp_q8_layout(K, k_hb, k_hx) &&
                            xp_q8_layout(V, v_hb, v_hx);
    const bool       conv      = K->type != GGML_TYPE_F16 && !q8;
    const int        chunk     = conv ? std::min(n_kv, std::max(XP_BK, chunk_env / XP_BK * XP_BK)) : n_kv;
    const int        n_chunk   = (n_kv + chunk - 1) / chunk;
    const int        n_blk_c   = chunk / XP_BK;  // blocks of a (full) chunk
    int nsplit = n_blk_c / min_bps;
    nsplit     = nsplit < 1 ? 1 : (nsplit > max_split ? (max_split < 1 ? 1 : max_split) : nsplit);
    const int bps = (n_blk_c + nsplit - 1) / nsplit;
    nsplit        = (n_blk_c + bps - 1) / bps;
    if (nsplit > 1 || n_chunk > 1) {
        part_o.alloc((size_t) nsplit * n_head * nq * XP_D);
        part_ml.alloc((size_t) nsplit * n_head * nq * 2);
    }
    if (conv) {
        kv16.alloc((size_t) 2 * n_kvh * chunk * XP_D * sizeof(sycl::half) + 64);
    }

    for (int64_t s = 0; s < Q->ne[3]; ++s) {
        const char * Ms = (const char *) mask->data + (mask->ne[3] > 1 ? s * mask->nb[3] : 0);
        const int64_t q0 = s * nq;  // the stream's first query in the flattened bias rows
        xp_classes((const sycl::half *) Ms, mask->nb[1] / sizeof(sycl::half), bias, q0, nq, n_blk, cls.get(), stream);

        xp_args a{};
        a.Q          = (const char *) Q->data + s * Q->nb[3];
        a.q_nb1      = Q->nb[1];
        a.q_nb2      = Q->nb[2];
        a.K          = (const sycl::half *) ((const char *) K->data + s * K->nb[3]);
        a.V          = (const sycl::half *) ((const char *) V->data + s * V->nb[3]);
        a.k_nb1      = K->nb[1];
        a.k_nb2      = K->nb[2];
        a.v_nb1      = V->nb[1];
        a.v_nb2      = V->nb[2];
        a.mask       = Ms;
        a.m_nb1      = mask->nb[1];
        a.cls        = cls.get();
        a.table      = bias.table ? bias.table + (size_t) q0 * bias.n_head * bias.n_bins : nullptr;
        a.idx        = bias.indices ? bias.indices + q0 * bias.stride : nullptr;
        a.idx_stride = bias.stride;
        a.pos        = bias.pos ? bias.pos + s * bias.pos_row : nullptr;
        a.pos_nkv    = bias.pos_nkv;
        a.extent     = bias.extent;
        a.n_bins     = (int) bias.n_bins;
        a.sinks      = sinks ? (const float *) sinks->data : nullptr;
        a.dst        = (float *) dst->data + (size_t) s * nq * n_head * XP_D;
        a.nq         = nq;
        a.n_kv       = n_kv;
        a.n_head     = n_head;
        a.g          = g;
        a.n_blk      = n_blk;
        a.n_rg       = n_rg;
        a.n_wg_kvh   = n_wg;
        a.qscale     = scale * XP_LOG2E;
        a.nsplit     = nsplit;
        a.bps        = bps;
        a.part_o      = nsplit > 1 || n_chunk > 1 ? part_o.get() : nullptr;
        a.part_ml     = nsplit > 1 || n_chunk > 1 ? part_ml.get() : nullptr;
        a.prefetch    = prefetch;
        a.part_always = n_chunk > 1;
        a.k_hb        = k_hb;
        a.v_hb        = v_hb;
        a.k_hx        = k_hx;
        a.v_hx        = v_hx;

        const sycl::nd_range<1> range(sycl::range<1>((size_t) n_kvh * nsplit * n_wg * XP_NT), sycl::range<1>(XP_NT));
        for (int c = 0; c < n_chunk; ++c) {
            const int c0 = c * chunk;
            const int nc = std::min(chunk, n_kv - c0);
            a.n_kv       = nc;
            a.n_blk_loc  = nc / XP_BK;
            a.kv_blk0    = c0 / XP_BK;
            a.cont       = c > 0;
            if (conv) {
                // [n_kvh][nc][D] f16 for K, then for V; the 2D loads want 64-byte aligned head slices
                const size_t head  = (size_t) nc * XP_D * sizeof(sycl::half);
                char *       base  = (char *) GGML_PAD((uintptr_t) kv16.get(), 64);
                sycl::half * k16   = (sycl::half *) base;
                sycl::half * v16   = (sycl::half *) (base + (size_t) n_kvh * head);
                const size_t ts    = ggml_type_size(K->type);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
                to_fp16((const char *) K->data + s * K->nb[3] + (size_t) c0 * K->nb[1], k16, XP_D, nc, n_kvh, 1,
                        K->nb[1] / ts, K->nb[2] / ts, 0, stream);
                to_fp16((const char *) V->data + s * V->nb[3] + (size_t) c0 * V->nb[1], v16, XP_D, nc, n_kvh, 1,
                        V->nb[1] / ts, V->nb[2] / ts, 0, stream);
                a.K     = k16;
                a.V     = v16;
                a.k_nb1 = a.v_nb1 = XP_D * sizeof(sycl::half);
                a.k_nb2 = a.v_nb2 = head;
            }
            if (q8) {
                auto kern = [=](sycl::nd_item<1> it) SYCL_ESIMD_FUNCTION { xp_kernel<XP_D, true>(a, it); };
                stream->parallel_for(range, xp_grf256<decltype(kern)>{ kern });
            } else {
                auto kern = [=](sycl::nd_item<1> it) SYCL_ESIMD_FUNCTION { xp_kernel<XP_D, false>(a, it); };
                stream->parallel_for(range, xp_grf256<decltype(kern)>{ kern });
            }
        }
        if (nsplit > 1 || n_chunk > 1) {
            xp_merge(a.part_o, a.part_ml, a.sinks, a.dst, nsplit, nq, n_head, g, stream);
        }
    }
}
