#include "fattn-xmx.hpp"
#include "dpas.hpp"
#include "fattn.hpp"

#include <cfloat>

#ifdef GGML_SYCL_HAS_DPAS

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
// A q8_0 cache is converted to f16 once per call; batches over 64 rows run as row chunks over that copy.

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
    float *      part_o;   // [n_kvh][nsplit][R8][D]
    float *      part_ml;  // [n_kvh][nsplit][R8][2]: row max, row sum
    int          g;        // query heads per KV head
    int          nq;       // query rows
    int          n_kv;
    int          nsplit;
    int          bps;      // blocks per split
    size_t       q_nb1, q_nb2, k_nb1, k_nb2, v_nb1, v_nb2, m_nb1;
    float        qscale;   // scale * log2(e)
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
template <int RG, bool SKIP> static ESIMD_INLINE void fa_xmx_kernel(const fa_xmx_args a, const sycl::nd_item<1> & it) {
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
    const uint32_t * Kh = (const uint32_t *) (a.K + (size_t) kvh * a.k_nb2);
    const sycl::half * Vh = (const sycl::half *) (a.V + (size_t) kvh * a.v_nb2);
    const unsigned surf_w = FA_XMX_D * sizeof(sycl::half) - 1;
    const unsigned surf_h = (unsigned) a.n_kv - 1;
    esimd_x::config_2d_mem_access<uint32_t, 8, 16, 1>    k_desc(Kh, surf_w, surf_h, (unsigned) a.k_nb1 - 1, 0, 0);
    esimd_x::config_2d_mem_access<sycl::half, 16, 16, 1> v_desc(Vh, surf_w, surf_h, (unsigned) a.v_nb1 - 1, t * 16, 0);

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
        auto score = [&]() {
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
        auto pv = [&](int ts) {
            v_desc.set_y(tok0 + ts * 16);
            simd<sycl::half, 256> vb = esimd_x::lsc_load_2d<sycl::half, 16, 16, 1, false, true>(v_desc);
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
                    pv(ts);
                }
            } else {
#pragma unroll
                for (int ts = 0; ts < T; ++ts) {
                    if (live[ts]) {
                        pv(ts);
                    }
                }
            }
        } else {
#pragma unroll
            for (int ts = 0; ts < T; ++ts) {
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

template <int RG, bool SKIP> static void fa_xmx_launch(const fa_xmx_args & a, int n_kvh, dpct::queue_ptr stream) {
    const sycl::nd_range<1> range(sycl::range<1>((size_t) n_kvh * a.nsplit * FA_XMX_T), sycl::range<1>(FA_XMX_T));
    auto kern = [=](sycl::nd_item<1> it) SYCL_ESIMD_FUNCTION { fa_xmx_kernel<RG, SKIP>(a, it); };
    stream->parallel_for(range, fa_xmx_grf256<decltype(kern)>{ kern });
}

template <bool SKIP> static void fa_xmx_launch_rg(int RG, const fa_xmx_args & a, int n_kvh, dpct::queue_ptr stream) {
    switch (RG) {
        case 1: fa_xmx_launch<1, SKIP>(a, n_kvh, stream); break;
        case 2: fa_xmx_launch<2, SKIP>(a, n_kvh, stream); break;
        case 3: fa_xmx_launch<3, SKIP>(a, n_kvh, stream); break;
        case 4: fa_xmx_launch<4, SKIP>(a, n_kvh, stream); break;
        case 5: fa_xmx_launch<5, SKIP>(a, n_kvh, stream); break;
        case 6: fa_xmx_launch<6, SKIP>(a, n_kvh, stream); break;
        case 7: fa_xmx_launch<7, SKIP>(a, n_kvh, stream); break;
        case 8: fa_xmx_launch<8, SKIP>(a, n_kvh, stream); break;
        default: GGML_ABORT("XMX flash attention: %d row groups", RG);
    }
}

#endif // GGML_SYCL_HAS_DPAS

bool ggml_sycl_flash_attn_ext_xmx_supported(int device, const ggml_tensor * dst) {
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

    // a q8_0 cache is converted to f16 once per call (see ggml_sycl_flash_attn_ext_xmx)
    const bool f16_kv = K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;
    const bool q8_kv  = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 && K->data != V->data &&
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
    if (nq > max_q || g > 64 || K->ne[1] % FA_XMX_BK != 0 || mask->ne[0] < K->ne[1] || mask->ne[1] < nq) {
        return false;
    }
    // 2D block loads: 64-byte aligned head slices, 16-byte aligned row pitch of at least 64 bytes (the f16 copy
    // of a q8_0 cache is contiguous and meets them)
    for (const ggml_tensor * t : { K, V }) {
        if (f16_kv && ((uintptr_t) t->data % 64 != 0 || t->nb[0] != 2 || t->nb[2] % 64 != 0 || t->nb[1] % 16 != 0 ||
                       t->nb[1] < 64)) {
            return false;
        }
    }
    if ((uintptr_t) Q->data % 16 != 0 || Q->nb[0] != 4 || Q->nb[1] % 16 != 0 || Q->nb[2] % 16 != 0 ||
        (uintptr_t) mask->data % 16 != 0 || mask->nb[1] % 16 != 0 || !ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
#else
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#endif
}

void ggml_sycl_flash_attn_ext_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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
    const bool       q8     = K->type == GGML_TYPE_Q8_0;
    const int        n_span = n_kv / 16;
    ggml_sycl_pool_alloc<uint8_t> span_live(ctx.pool());
    uint8_t * live = q8 ? span_live.alloc(n_span) : nullptr;
    if (q8 && skip) {
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

    // a q8_0 cache: convert the live spans of K and V to contiguous f16 once, then every chunk reads the copy
    const char * K_data = (const char *) K->data;
    const char * V_data = (const char *) V->data;
    size_t       k_nb1 = K->nb[1], k_nb2 = K->nb[2], v_nb1 = V->nb[1], v_nb2 = V->nb[2];
    ggml_sycl_fattn_alloc K_f16(ctx.fattn_buffers().K);
    ggml_sycl_fattn_alloc V_f16(ctx.fattn_buffers().V);
    if (q8) {
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

    // split the cache so that about GGML_SYCL_FA_XMX_WGS work-groups run
    static const int target_wgs = ggml_sycl_get_env("GGML_SYCL_FA_XMX_WGS", 64);
    int nsplit = std::max(1, std::min(nblk, (target_wgs + n_kvh - 1) / n_kvh));
    const int bps = (nblk + nsplit - 1) / nsplit;
    nsplit        = (nblk + bps - 1) / bps;

    ggml_sycl_pool_alloc<float> part_o(ctx.pool(), (size_t) n_kvh * nsplit * R8_max * FA_XMX_D);
    ggml_sycl_pool_alloc<float> part_ml(ctx.pool(), (size_t) n_kvh * nsplit * R8_max * 2);

    fa_xmx_args a;
    a.K         = K_data;
    a.V         = V_data;
    a.span_live = live;
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
    const size_t  o_nb1 = dst->nb[1] / sizeof(float);
    const size_t  o_nb2 = dst->nb[2] / sizeof(float);
    for (int j0 = 0; j0 < nq; j0 += nq_chunk) {
        const int nqc = std::min(nq_chunk, nq - j0);
        const int M   = g * nqc;
        const int RG  = (M + 7) / 8;
        const int R8  = RG * 8;
        a.Q    = (const char *) Q->data + j0 * Q->nb[1];
        a.mask = (const char *) mask->data + j0 * mask->nb[1];
        a.nq   = nqc;
        if (q8) {
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
    GGML_ABORT("ESIMD not available");
#endif
}
