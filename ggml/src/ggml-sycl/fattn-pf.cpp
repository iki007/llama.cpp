// Prompt-size flash attention on Xe2 XMX for head 256: f16 or q8_0 caches, every stream (sequence) in one launch.
// The oneDNN SDPA takes one stream only, so the prompt rows of several sessions (a non-unified cache's streams) fell
// to the MKL / tile kernels (Qwen3.8-27B per card, 4 streams x 512 rows at 30k keys: 29.3 ms against 4.3 ms for one).
// A work-group of 16 threads takes 128 query rows of one head (one stream) and walks the keys in blocks of 128.
// Thread t (row group rg = t / 4, column c = t % 4) scores rows 32 rg.. against keys 32 c.. of a block (S = Q K^T over
// the 256 dims: Q 2D-loaded from a scaled f16 copy, K loaded transposed = the VNNI B operand), the row maxima and sums
// of the 4 threads of a row group meet in SLM, P = exp2(S - max) goes to SLM as f16, and the thread accumulates dims
// 64 c.. of O += P V for its 32 rows (V loaded VNNI-transformed). Q is pre-scaled by scale * log2(e). The 6 query heads
// of a KV head run in adjacent work-groups, so each K / V block is fetched once into L2 for all of them.
#include "fattn-pf.hpp"
#include "dpas.hpp"
#include "convert.hpp"

#include <cfloat>

#ifdef GGML_SYCL_HAS_DPAS
#include <sycl/ext/intel/experimental/esimd/math.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace {

template <typename F> struct pf_grf256 {
    F f;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL { f(it); }
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    }
};

struct pf_args {
    const sycl::half * Q;  // [ns][H][nr][D], scaled
    const char *       K;  // plane (s, kvh) at s k_s + kvh k_h bytes, rows of D f16 at pitch k_p
    const char *       V;
    const char *       M;  // f16 [nr][n_kv] at pitch m_p, stream s at s m_s (0: one mask)
    char *             dst;
    size_t             k_s, k_h, v_s, v_h, m_s, d_nb1, d_nb2, d_nb3;
    int                k_p, v_p, m_p;
    int                H, G, n_kvh, nr, n_kv, n_rt, ns;
    // split-K: work-group split sp walks key blocks [sp bps, (sp + 1) bps) and, with nsplit > 1, writes its
    // unnormalized rows to part_o [nsplit][ns][H][nr][D] and their max and sum to part_ml [nsplit][ns][H][nr][2]
    int                nsplit, bps;
    float *            part_o;
    float *            part_ml;
    // per (stream of the mask, row tile, key block): 0 all visible (0), 1 all masked (-inf), 2 mixed; null: read the mask
    const uint8_t *    mcls;
    int                m_ns;
};

constexpr int PF_D = 256, PF_BR = 128, PF_BC = 128, PF_T = 16;

// row maxima / sums of an [8][16] block
static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 8> pf_row_max8(sycl::ext::intel::esimd::simd<float, 128> p) {
    using namespace sycl::ext::intel::esimd;
    auto            p2 = p.template bit_cast_view<float, 8, 16>();
    simd<float, 64> s8 = max(simd<float, 64>(p2.template select<8, 1, 8, 1>(0, 0)), simd<float, 64>(p2.template select<8, 1, 8, 1>(0, 8)));
    auto            v8 = s8.template bit_cast_view<float, 8, 8>();
    simd<float, 32> s4 = max(simd<float, 32>(v8.template select<8, 1, 4, 1>(0, 0)), simd<float, 32>(v8.template select<8, 1, 4, 1>(0, 4)));
    auto            v4 = s4.template bit_cast_view<float, 8, 4>();
    simd<float, 16> s2 = max(simd<float, 16>(v4.template select<8, 1, 2, 1>(0, 0)), simd<float, 16>(v4.template select<8, 1, 2, 1>(0, 2)));
    return max(simd<float, 8>(s2.template select<8, 2>(0)), simd<float, 8>(s2.template select<8, 2>(1)));
}

static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 8> pf_row_sum8(sycl::ext::intel::esimd::simd<float, 128> p) {
    using namespace sycl::ext::intel::esimd;
    auto            p2 = p.template bit_cast_view<float, 8, 16>();
    simd<float, 64> s8 = p2.template select<8, 1, 8, 1>(0, 0) + p2.template select<8, 1, 8, 1>(0, 8);
    auto            v8 = s8.template bit_cast_view<float, 8, 8>();
    simd<float, 32> s4 = v8.template select<8, 1, 4, 1>(0, 0) + v8.template select<8, 1, 4, 1>(0, 4);
    auto            v4 = s4.template bit_cast_view<float, 8, 4>();
    simd<float, 16> s2 = v4.template select<8, 1, 2, 1>(0, 0) + v4.template select<8, 1, 2, 1>(0, 2);
    return s2.template select<8, 2>(0) + s2.template select<8, 2>(1);
}

// v[r] repeated 16 times for each of the 8 rows: the per-row factor of an [8][16] accumulator
static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 128> pf_rows16(sycl::ext::intel::esimd::simd<float, 8> v) {
    return v.template replicate_vs_w_hs<8, 1, 16, 0>(0);
}

// K / V as dense f16 [n_kv][D] planes: 2D loads, K transposed (the VNNI B operand of S), V VNNI-transformed (of P V)
ESIMD_INLINE void pf_kernel(const pf_args a, const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;
    namespace esx = sycl::ext::intel::experimental::esimd;
    constexpr int      D  = PF_D;
    constexpr int      KS = D / 16;
    constexpr float    LOG2E   = 1.44269504088896341f;
    // SLM: P as [8 key steps][16 row groups of 8] A operands of 256 bytes, then the row maxima and sums [4 c][128]
    constexpr uint32_t SLM_P   = 0;
    constexpr uint32_t SLM_MAX = PF_BR * PF_BC * 2;
    constexpr uint32_t SLM_SUM = SLM_MAX + 4 * PF_BR * 4;
    constexpr uint32_t SLM_END = SLM_SUM + 4 * PF_BR * 4;
    slm_init<SLM_END>();

    const int t  = it.get_local_id(0);
    int       wg = it.get_group(0);
    const int g  = wg % a.G;
    wg /= a.G;
    const int sp = wg % a.nsplit;
    wg /= a.nsplit;
    const int rt = wg % a.n_rt;
    wg /= a.n_rt;
    const int kvh = wg % a.n_kvh;
    const int s   = wg / a.n_kvh;
    const int h   = kvh * a.G + g;
    const int rg = t / 4, c = t % 4;
    const int r0 = rt * PF_BR + 32 * rg;

    const sycl::half * Qp = a.Q + (size_t) (s * a.H + h) * a.nr * D;
    const uint32_t *   Kp = (const uint32_t *) (a.K + s * a.k_s + kvh * a.k_h);
    const sycl::half * Vp = (const sycl::half *) (a.V + s * a.v_s + kvh * a.v_h);
    const sycl::half * Mp = (const sycl::half *) (a.M + s * a.m_s);
    esx::config_2d_mem_access<sycl::half, 16, 32, 1> qd(Qp, D * 2 - 1, (unsigned) a.nr - 1, D * 2 - 1, 0, r0);
    esx::config_2d_mem_access<uint32_t, 8, 16, 1>    kd2(Kp, D * 2 - 1, (unsigned) a.n_kv - 1, D * 2 - 1, 0, 0);
    esx::config_2d_mem_access<sycl::half, 16, 16, 1> vd2(Vp, D * 2 - 1, (unsigned) a.n_kv - 1, D * 2 - 1, 0, 0);
    esx::config_2d_mem_access<sycl::half, 32, 32, 1> md(Mp, (unsigned) a.n_kv * 2 - 1, (unsigned) a.nr - 1, (unsigned) a.m_p - 1, 0,
                                                        r0);

    // o[4 i + dg]: rows 8 i.., dims 64 c + 16 dg..; m, l: this thread's 32 rows (the same in the 4 threads of a group)
    simd<float, 128> o[16];
#pragma unroll
    for (int q = 0; q < 16; ++q) {
        o[q] = 0.0f;
    }
    simd<float, 32> m = -1.0e30f;
    simd<float, 32> l = 0.0f;

    const int nblk = a.n_kv / PF_BC;
    const int kb1  = (sp + 1) * a.bps < nblk ? (sp + 1) * a.bps : nblk;
    const uint8_t * cls = a.mcls ? a.mcls + ((size_t) (a.m_ns > 1 ? s : 0) * a.n_rt + rt) * nblk : nullptr;
    for (int kb = sp * a.bps; kb < kb1; ++kb) {
        const int k0 = kb * PF_BC;
        // a block every row of the tile masks adds exactly 0 (causal prompts at depth: the blocks past the diagonal); the
        // class byte is the same for all threads of the work-group, so skipping keeps the barriers paired
        const int mc = cls ? cls[kb] : 2;
        if (mc == 1) {
            continue;
        }
        // 1. S: sa[2 i + j] = rows 8 i.., keys 32 c + 16 j..
        simd<float, 128> sa[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            sa[q] = 0.0f;
        }
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            qd.set_x(16 * ks);
            simd<sycl::half, 512> qa = esx::lsc_load_2d<sycl::half, 16, 32, 1, false, false, cache_hint::cached, cache_hint::cached>(qd);
            kd2.set_x(8 * ks);
            kd2.set_y(k0 + 32 * c);
            simd<uint32_t, 128> kb0 = esx::lsc_load_2d<uint32_t, 8, 16, 1, true, false, cache_hint::cached, cache_hint::cached>(kd2);
            kd2.set_y(k0 + 32 * c + 16);
            simd<uint32_t, 128> kb1 = esx::lsc_load_2d<uint32_t, 8, 16, 1, true, false, cache_hint::cached, cache_hint::cached>(kd2);
            simd<sycl::half, 256> b0 = kb0.template bit_cast_view<sycl::half>();
            simd<sycl::half, 256> b1 = kb1.template bit_cast_view<sycl::half>();
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // (one 8-row A operand per load avoids the spills but costs more: 4x the messages, 10.5 -> 12.7 ms)
                simd<sycl::half, 128> ai = qa.template select<128, 1>(128 * i);
                sa[2 * i]     = xmx::dpas<8, 8, float, float>(sa[2 * i], b0, ai);
                sa[2 * i + 1] = xmx::dpas<8, 8, float, float>(sa[2 * i + 1], b1, ai);
            }
        }
        // 2. mask (log2 domain): [32 rows][32 keys]
        // (a runtime branch around the mask load, to skip all-visible blocks, costs 50%: 10.7 -> 15.8 ms)
        md.set_x(k0 + 32 * c);
        simd<sycl::half, 1024> mk  = esx::lsc_load_2d<sycl::half, 32, 32, 1, false, false>(md);
        auto                   mk2 = mk.template bit_cast_view<sycl::half, 32, 32>();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                simd<sycl::half, 128> mv = mk2.template select<8, 1, 16, 1>(8 * i, 16 * j);
                sa[2 * i + j] += convert<float>(mv) * LOG2E;
            }
        }
        // 3. block row max over the 4 threads of the row group, new running max, P = exp2(S - max)
        simd<float, 32> mx;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            mx.template select<8, 1>(8 * i) = pf_row_max8(max(sa[2 * i], sa[2 * i + 1]));
        }
        slm_block_store<float, 32>(SLM_MAX + (c * PF_BR + 32 * rg) * 4, mx);
        barrier();
        simd<float, 32> bm = slm_block_load<float, 32>(SLM_MAX + (32 * rg) * 4);
#pragma unroll
        for (int cc = 1; cc < 4; ++cc) {
            bm = max(bm, slm_block_load<float, 32>(SLM_MAX + (cc * PF_BR + 32 * rg) * 4));
        }
        simd<float, 32>       mn    = max(m, bm);
        simd<float, 32>       alpha = exp2(m - mn);
        m                           = mn;
        simd<float, 32> sm;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const simd<float, 128> mr = pf_rows16(mn.template select<8, 1>(8 * i));
            sa[2 * i]                 = exp2(sa[2 * i] - mr);
            sa[2 * i + 1]             = exp2(sa[2 * i + 1] - mr);
            sm.template select<8, 1>(8 * i) = pf_row_sum8(sa[2 * i] + sa[2 * i + 1]);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                slm_block_store<sycl::half, 128>(SLM_P + ((2 * c + j) * 16 + 4 * rg + i) * 256, convert<sycl::half>(sa[2 * i + j]));
            }
        }
        slm_block_store<float, 32>(SLM_SUM + (c * PF_BR + 32 * rg) * 4, sm);
        barrier();
        // 4. running sum, rescale O, O += P V for dims 64 c..
        simd<float, 32> bs = slm_block_load<float, 32>(SLM_SUM + (32 * rg) * 4);
#pragma unroll
        for (int cc = 1; cc < 4; ++cc) {
            bs += slm_block_load<float, 32>(SLM_SUM + (cc * PF_BR + 32 * rg) * 4);
        }
        l = l * alpha + bs;
        {  // (skipping this when no row max moved adds a branch that costs 7%: 10.67 -> 11.38 ms)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const simd<float, 128> ar = pf_rows16(alpha.template select<8, 1>(8 * i));
#pragma unroll
                for (int dg = 0; dg < 4; ++dg) {
                    o[4 * i + dg] *= ar;
                }
            }
        }
#pragma unroll
        for (int ks = 0; ks < PF_BC / 16; ++ks) {
            simd<sycl::half, 128> pa[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                pa[i] = slm_block_load<sycl::half, 128>(SLM_P + (ks * 16 + 4 * rg + i) * 256);
            }
#pragma unroll
            for (int dg = 0; dg < 4; ++dg) {
                vd2.set_x(64 * c + 16 * dg);
                vd2.set_y(k0 + 16 * ks);
                simd<sycl::half, 256> vb = esx::lsc_load_2d<sycl::half, 16, 16, 1, false, true, cache_hint::cached, cache_hint::cached>(vd2);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    o[4 * i + dg] = xmx::dpas<8, 8, float, float>(o[4 * i + dg], vb, pa[i]);
                }
            }
        }
        // the next block writes SLM_MAX before its first barrier, which every thread reaches only after it has read
        // this block's maxima (before the second barrier); SLM_P / SLM_SUM after it, when every thread is past its PV
    }

    // 5. split-K: this split's unnormalized rows, max and sum
    if (a.nsplit > 1) {
        const size_t plane = (((size_t) sp * a.ns + s) * a.H + h) * a.nr;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int row = r0 + 8 * i + r;
                if (row < a.nr) {
                    float * op = a.part_o + (plane + row) * D + 64 * c;
#pragma unroll
                    for (int dg = 0; dg < 4; ++dg) {
                        block_store<float, 16>(op + 16 * dg, o[4 * i + dg].template select<16, 1>(16 * r));
                    }
                }
            }
        }
        if (c == 0) {
#pragma unroll
            for (int r = 0; r < 32; ++r) {
                const int row = r0 + r;
                if (row < a.nr) {
                    a.part_ml[(plane + row) * 2]     = m[r];
                    a.part_ml[(plane + row) * 2 + 1] = l[r];
                }
            }
        }
        return;
    }

    // 5. normalized rows to dst (d, h, row, s)
    const simd<float, 32> inv = 1.0f / max(l, 1.0e-30f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row = r0 + 8 * i + r;
            if (row < a.nr) {
                float * dp = (float *) (a.dst + s * a.d_nb3 + (size_t) row * a.d_nb2 + (size_t) h * a.d_nb1) + 64 * c;
                const float f = l[8 * i + r] > 0.0f ? inv[8 * i + r] : 0.0f;
#pragma unroll
                for (int dg = 0; dg < 4; ++dg) {
                    block_store<float, 16>(dp + 16 * dg, o[4 * i + dg].template select<16, 1>(16 * r) * f);
                }
            }
        }
    }
}

}  // namespace

bool ggml_sycl_fa_prompt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    static const bool enabled = ggml_sycl_get_env("GGML_SYCL_FA_PROMPT", 1) != 0;
    const gpu_arch    arch    = ggml_sycl_info().devices[ctx.device].hw_info.arch;
    if (!enabled || !g_ggml_sycl_enable_esimd || (arch != gpu_arch::intel_gpu_bmg_g21 && arch != gpu_arch::intel_gpu_bmg_g31)) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    float scale = 1.0f, max_bias = 0.0f, softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int64_t D = Q->ne[0], nr = Q->ne[1], H = Q->ne[2], ns = Q->ne[3];
    const int64_t n_kv = K->ne[1], n_kvh = K->ne[2];
    if (D != PF_D || K->ne[0] != D || V->ne[0] != D || Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || sinks ||
        max_bias != 0.0f || softcap != 0.0f || !mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 ||
        (mask->ne[3] != 1 && mask->ne[3] != ns) || mask->ne[0] < n_kv || mask->ne[1] < nr || nr < 32 || n_kv % PF_BC != 0 ||
        n_kv < PF_BC || H % n_kvh != 0 || K->ne[3] != ns || V->ne[3] != ns || V->ne[1] != n_kv || V->ne[2] != n_kvh ||
        dst->ne[0] != D || dst->ne[1] != H || dst->ne[2] != nr || dst->ne[3] != ns ||
        !((K->type == V->type && (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0)) ||
          (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0)) || (uintptr_t) mask->data % 64 != 0 || mask->nb[1] % 16 != 0 ||
        mask->nb[1] < 64 || (ns > 1 && mask->ne[3] == ns && mask->nb[3] % 64 != 0) || Q->nb[0] != sizeof(float)) {
        return false;
    }
    dpct::queue_ptr stream = ctx.stream();
    const float     qscale = scale * 1.44269504088896341f;

    // Q -> [ns][H][nr][D] f16, scaled
    ggml_sycl_pool_alloc<sycl::half> qh(ctx.pool(), (size_t) ns * H * nr * D);
    {
        sycl::half *  out = qh.get();
        const char *  qd  = (const char *) Q->data;
        const size_t  nb1 = Q->nb[1], nb2 = Q->nb[2], nb3 = Q->nb[3];
        const int64_t n4  = ns * H * nr * D / 4;
        const int     Hi = (int) H, nri = (int) nr;
        stream->parallel_for(sycl::range<1>((size_t) n4), [=](sycl::id<1> id) {
            const int64_t e = (int64_t) id[0] * 4;
            const int     d = (int) (e % D);
            int64_t       r = e / D;
            const int     row = (int) (r % nri);
            r /= nri;
            const int h = (int) (r % Hi);
            const int s = (int) (r / Hi);
            const float * qp = (const float *) (qd + s * nb3 + h * nb2 + row * nb1) + d;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                out[e + k] = sycl::half(qp[k] * qscale);
            }
        });
    }

    // K / V as dense f16 [ns][n_kvh][n_kv][D] copies for the 2D loads: q8_0 through the library's nc conversion (149 us
    // per tensor at 32k keys, faster than a kernel of our own), f16 at the cache's byte strides one work-item per 2 dims
    pf_args a{};
    ggml_sycl_pool_alloc<uint32_t> kt(ctx.pool(), (size_t) ns * n_kvh * n_kv * D / 2);
    ggml_sycl_pool_alloc<uint32_t> vp(ctx.pool(), (size_t) ns * n_kvh * n_kv * D / 2);
    for (int w = 0; w < 2; ++w) {
        const ggml_tensor * T = w ? V : K;
        if (T->type != GGML_TYPE_F16) {  // q8_0, or a q4_0 V
            const size_t ts = ggml_type_size(T->type);
            ggml_get_to_fp16_nc_sycl(T->type)(T->data, (sycl::half *) (w ? vp.get() : kt.get()), T->ne[0], T->ne[1], T->ne[2],
                                              T->ne[3], (int64_t) (T->nb[1] / ts), (int64_t) (T->nb[2] / ts),
                                              (int64_t) (T->nb[3] / ts), stream);
            continue;
        }
        const char * src = (const char *) T->data;
        const size_t nb1 = T->nb[1], nb2 = T->nb[2], nb3 = T->nb[3];
        uint32_t *   out = w ? vp.get() : kt.get();
        const int    nkv = (int) n_kv, nh = (int) n_kvh;
        stream->parallel_for(sycl::range<1>((size_t) ns * n_kvh * n_kv * D / 2), [=](sycl::id<1> id) {
            const int    dp  = (int) (id[0] % (D / 2));
            size_t       r   = id[0] / (D / 2);
            const int    key = (int) (r % nkv);
            r /= nkv;
            const int    h   = (int) (r % nh);
            const int    s   = (int) (r / nh);
            const uint32_t * row = (const uint32_t *) (src + s * nb3 + h * nb2 + (size_t) key * nb1);
            out[id[0]] = row[dp];
        });
    }
    a.K   = (const char *) kt.get();
    a.V   = (const char *) vp.get();
    a.k_h = (size_t) n_kv * D * 2;
    a.k_s = a.k_h * n_kvh;
    a.v_h = a.k_h;
    a.v_s = a.k_s;
    a.k_p = a.v_p = 0;
    a.Q     = qh.get();
    a.M     = (const char *) mask->data;
    a.m_s   = mask->ne[3] == ns && ns > 1 ? mask->nb[3] : 0;
    a.m_p   = (int) mask->nb[1];
    a.dst   = (char *) dst->data;
    a.d_nb1 = dst->nb[1];
    a.d_nb2 = dst->nb[2];
    a.d_nb3 = dst->nb[3];
    a.H     = (int) H;
    a.G     = (int) (H / n_kvh);
    a.n_kvh = (int) n_kvh;
    a.nr    = (int) nr;
    a.n_kv  = (int) n_kv;
    a.n_rt  = (int) ((nr + PF_BR - 1) / PF_BR);
    a.ns    = (int) ns;
    // split the keys when the work-groups would not fill whole waves of the GPU: the split count with the best wave fill,
    // at least 4 key blocks per split, fewer splits unless 5% better (1 x 128 rows at 32k keys: 2.68 -> 0.98 ms). 64
    // resident work-groups = 2 per Xe-core of an Arc Pro B70 (the device's reported unit count does not give it)
    constexpr int slots = 64;
    const int        nblk  = (int) (n_kv / PF_BC);
    const int        tiles = (int) (ns * H * a.n_rt);
    auto fill = [&](int k) { const int w = tiles * k; return (double) w / ((double) ((w + slots - 1) / slots) * slots); };
    int    nsplit = 1;
    double best   = fill(1);
    for (int k = 2; k <= 32 && nblk / k >= 4; ++k) {
        if (fill(k) > best + 0.05) {
            nsplit = k;
            best   = fill(k);
        }
    }
    a.bps    = (nblk + nsplit - 1) / nsplit;
    a.nsplit = (nblk + a.bps - 1) / a.bps;
    // mask classes: one work-group of 128 work-items per (mask stream, row tile, key block); work-item i scans key i
    const int m_ns = (int) mask->ne[3];
    ggml_sycl_pool_alloc<uint8_t> mc(ctx.pool());
    a.m_ns = m_ns;
    {
        mc.alloc((size_t) m_ns * a.n_rt * nblk);
        uint8_t *     out  = mc.get();
        const char *  mp   = (const char *) mask->data;
        const size_t  mnb1 = mask->nb[1], mnb3 = mask->nb[3];
        const int     n_rt = a.n_rt, nri = (int) nr;
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>((size_t) m_ns * n_rt * nblk * PF_BR, PF_BR), [=](sycl::nd_item<1> it) {
                const size_t g   = it.get_group(0);
                const int    kb  = (int) (g % nblk);
                const int    rt  = (int) ((g / nblk) % n_rt);
                const int    ms  = (int) (g / ((size_t) nblk * n_rt));
                // work-item i reads key kb * 128 + i of every row of the tile: each row is one coalesced 256-byte read
                const int key = kb * PF_BC + (int) it.get_local_id(0);
                int zero = 1, inf = 1;
                const int r1 = rt * PF_BR + PF_BR < nri ? rt * PF_BR + PF_BR : nri;
                for (int row = rt * PF_BR; row < r1; ++row) {
                    const float v = (float) ((const sycl::half *) (mp + ms * mnb3 + (size_t) row * mnb1))[key];
                    zero &= v == 0.0f;
                    inf &= v == -INFINITY;
                }
                zero = sycl::all_of_group(it.get_group(), zero != 0);
                inf  = sycl::all_of_group(it.get_group(), inf != 0);
                if (it.get_local_id(0) == 0) {
                    out[g] = inf ? 1 : zero ? 0 : 2;
                }
            });
        });
        a.mcls = mc.get();
    }
    ggml_sycl_pool_alloc<float> po(ctx.pool()), pml(ctx.pool());
    if (a.nsplit > 1) {
        po.alloc((size_t) a.nsplit * ns * H * nr * D);
        pml.alloc((size_t) a.nsplit * ns * H * nr * 2);
        a.part_o  = po.get();
        a.part_ml = pml.get();
    }
    const size_t n_wg = (size_t) ns * n_kvh * a.n_rt * a.nsplit * a.G;
    stream->submit([&](sycl::handler & cgh) {
        auto kern = [=](sycl::nd_item<1> it) SYCL_ESIMD_FUNCTION { pf_kernel(a, it); };
        cgh.parallel_for(sycl::nd_range<1>(n_wg * PF_T, PF_T), pf_grf256<decltype(kern)>{ kern });
    });
    if (a.nsplit > 1) {
        // merge: O = sum_i O_i 2^(m_i - M) / sum_i l_i 2^(m_i - M), M = max_i m_i; one work-item per 4 dims of a row
        const float * pof = a.part_o;
        const float * pmf = a.part_ml;
        char *        dd  = a.dst;
        const int     nsp = a.nsplit, Hi = (int) H, nri = (int) nr;
        const size_t  nb1 = a.d_nb1, nb2 = a.d_nb2, nb3 = a.d_nb3;
        const size_t  plane = (size_t) ns * H * nr;
        stream->parallel_for(sycl::range<1>(plane * D / 4), [=](sycl::id<1> id) {
            const size_t row = id[0] / (D / 4);
            const int    d4  = (int) (id[0] % (D / 4)) * 4;
            float        M   = -FLT_MAX;
            for (int i = 0; i < nsp; ++i) {
                M = sycl::fmax(M, pmf[(i * plane + row) * 2]);
            }
            float L = 0.0f, o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
            for (int i = 0; i < nsp; ++i) {
                const float   w  = sycl::exp2(pmf[(i * plane + row) * 2] - M);
                const float * op = pof + (i * plane + row) * D + d4;
                L += pmf[(i * plane + row) * 2 + 1] * w;
                o0 += op[0] * w;
                o1 += op[1] * w;
                o2 += op[2] * w;
                o3 += op[3] * w;
            }
            const float inv = L > 0.0f ? 1.0f / L : 0.0f;
            const int   r   = (int) (row % nri);
            const int   h   = (int) ((row / nri) % Hi);
            const int   s   = (int) (row / ((size_t) nri * Hi));
            float *     dp  = (float *) (dd + s * nb3 + (size_t) r * nb2 + (size_t) h * nb1) + d4;
            dp[0] = o0 * inv;
            dp[1] = o1 * inv;
            dp[2] = o2 * inv;
            dp[3] = o3 * inv;
        });
    }
    return true;
}

#else
bool ggml_sycl_fa_prompt(ggml_backend_sycl_context &, ggml_tensor *) {
    return false;
}
#endif
