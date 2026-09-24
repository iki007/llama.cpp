#include "gated_delta_net_esimd.hpp"
#include "dpas.hpp"

#ifdef GGML_SYCL_HAS_DPAS

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

// The gated delta rule recurrence of gated_delta_net_sycl, in ESIMD for prefill on Xe2: the same f32 arithmetic per
// token, laid out so that the reductions stop dominating. A thread owns NC whole state columns of one head in
// registers; per token it forms S.k and S.q for each of its columns and k.q as 16-wide partial sums and reduces all
// 2 NC + 1 of them in one pass, where the SIMT kernel spends a sub-group reduction per column. The inputs of the
// next token are loaded while the current one is computed.
static constexpr int GDN_E_SV = 128;  // state size

template <typename F> struct gdn_esimd_grf256 {
    F f;

    void operator()(sycl::nd_item<3> it) const SYCL_ESIMD_KERNEL { f(it); }

    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    }
};

// row sums of an [R][16] block
template <int R> static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, R> gdn_row_sums(
    sycl::ext::intel::esimd::simd<float, R * 16> p) {
    using namespace sycl::ext::intel::esimd;
    auto            p2 = p.template bit_cast_view<float, R, 16>();
    simd<float, R * 8> s8 = p2.template select<R, 1, 8, 1>(0, 0) + p2.template select<R, 1, 8, 1>(0, 8);
    auto            v8 = s8.template bit_cast_view<float, R, 8>();
    simd<float, R * 4> s4 = v8.template select<R, 1, 4, 1>(0, 0) + v8.template select<R, 1, 4, 1>(0, 4);
    auto            v4 = s4.template bit_cast_view<float, R, 4>();
    simd<float, R * 2> s2 = v4.template select<R, 1, 2, 1>(0, 0) + v4.template select<R, 1, 2, 1>(0, 2);
    return s2.template select<R, 2>(0) + s2.template select<R, 2>(1);
}

template <int NC>
static ESIMD_INLINE void gdn_esimd_thread(const float * q, const float * k, const float * v, const float * g,
                                          const float * beta, const float * state_in, float * attn,
                                          float * state_out, const ggml_sycl_gdn_esimd_args a,
                                          const sycl::nd_item<3> & it) {
    using namespace sycl::ext::intel::esimd;
    constexpr int SV = GDN_E_SV;
    constexpr int NR = 2 * NC + 1;  // partial rows: S.k per column, S.q per column, k.q
    const properties a16{ alignment<16> };

    const int     col0 = (int) it.get_local_id(2) * NC;
    const int64_t h    = it.get_group(1);
    const int64_t s    = it.get_group(0);

    // S[c SV + i] = S[i][col0 + c]: columns are contiguous in ggml's (transposed) state layout
    const float *        s0 = state_in + ((s * a.H + h) * SV + col0) * SV;
    simd<float, NC * SV> S;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        S.template select<SV, 1>(c * SV) = block_load<float, SV>(s0 + c * SV, a16);
    }

    const float * qp = q + (s / a.rq3) * a.sq3 + (h % a.neqk1) * a.sq1;
    const float * kp = k + (s / a.rq3) * a.sq3 + (h % a.neqk1) * a.sq1;
    const float * vp = v + s * a.sv3 + h * a.sv1 + col0;
    const float * gp = g + s * a.sb3 + h * a.sb1;
    const float * bp = beta + s * a.sb3 + h * a.sb1;
    float *       op = attn + (s * a.n_tokens * a.H + h) * SV + col0;

    simd<float, SV> qt = block_load<float, SV>(qp, a16);
    simd<float, SV> kt = block_load<float, SV>(kp, a16);
    simd<float, NC> vt = block_load<float, NC>(vp, a16);
    float           gt = gp[0];
    float           bt = bp[0];
    float           gm = 1.0f;  // S = gm * (what the registers hold)

    // one token; the next token's inputs are loaded first so that they arrive during this one
    auto step = [&](const bool next) {
        simd<float, SV> qn, kn;
        simd<float, NC> vn;
        float           gn = 0.0f, bn = 0.0f;
        if (next) {
            qn = block_load<float, SV>(qp + a.sq2, a16);
            kn = block_load<float, SV>(kp + a.sq2, a16);
            vn = block_load<float, NC>(vp + a.sv2, a16);
            gn = gp[a.sb2];
            bn = bp[a.sb2];
        }

        simd<float, NR * 16> p;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            simd<float, 16> pk = S.template select<16, 1>(c * SV) * kt.select<16, 1>(0);
            simd<float, 16> pq = S.template select<16, 1>(c * SV) * qt.select<16, 1>(0);
#pragma unroll
            for (int r = 1; r < SV / 16; ++r) {
                pk += S.template select<16, 1>(c * SV + r * 16) * kt.select<16, 1>(r * 16);
                pq += S.template select<16, 1>(c * SV + r * 16) * qt.select<16, 1>(r * 16);
            }
            p.template select<16, 1>(c * 16)        = pk;
            p.template select<16, 1>((NC + c) * 16) = pq;
        }
        {
            simd<float, 16> pkq = kt.select<16, 1>(0) * qt.select<16, 1>(0);
#pragma unroll
            for (int r = 1; r < SV / 16; ++r) {
                pkq += kt.select<16, 1>(r * 16) * qt.select<16, 1>(r * 16);
            }
            p.template select<16, 1>(2 * NC * 16) = pkq;
        }
        simd<float, NR> sums = gdn_row_sums<NR>(p);

        // as gated_delta_net_sycl: delta = (v - g S.k) beta, attn = (g S.q + delta k.q) scale, S = g S + k delta^T,
        // with the registers holding S / gm so that the decay costs one multiply per token instead of one per element
        const float     gd    = sycl::ext::intel::esimd::exp(simd<float, 1>(gt))[0];
        const float     gs    = gm * gd;
        simd<float, NC> delta = (vt - gs * sums.template select<NC, 1>(0)) * bt;
        const float     kq    = sums[2 * NC];
        simd<float, NC> o     = (gs * sums.template select<NC, 1>(NC) + delta * kq) * a.scale;
        block_store<float, NC>(op, o, a16);
        if (gs >= 0x1p-64f) {
            gm                        = gs;
            const simd<float, NC> dsc = delta * (1.0f / gs);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const float dc                   = dsc[c];
                S.template select<SV, 1>(c * SV) = S.template select<SV, 1>(c * SV) + kt * dc;
            }
        } else {
            // fold the scale back in before it leaves f32 (or when g is 0)
            gm = 1.0f;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const float dc                   = delta[c];
                S.template select<SV, 1>(c * SV) = S.template select<SV, 1>(c * SV) * gs + kt * dc;
            }
        }

        op += a.H * SV;
        if (next) {
            qt = qn;
            kt = kn;
            vt = vn;
            gt = gn;
            bt = bn;
            qp += a.sq2;
            kp += a.sq2;
            vp += a.sv2;
            gp += a.sb2;
            bp += a.sb2;
        }
    };
    for (int64_t t = 1; t < a.n_tokens; ++t) {
        step(true);
    }
    step(false);

    float * s1 = state_out + ((s * a.H + h) * SV + col0) * SV;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        block_store<float, SV>(s1 + c * SV, S.template select<SV, 1>(c * SV) * gm, a16);
    }
}

#endif  // GGML_SYCL_HAS_DPAS

bool ggml_sycl_gdn_esimd_supported(const ggml_backend_sycl_context & ctx, const int64_t S_v, const bool kda,
                                   const int K, const float * q, const float * k, const float * v,
                                   const float * state_in, const float * attn, const float * state_out,
                                   const ggml_sycl_gdn_esimd_args & a) {
#ifdef GGML_SYCL_HAS_DPAS
    // below this the SIMT kernel wins (B70, 32 / 48 heads: 4 tokens 5.1 / 6.9 us against 9.3 / 10.6, 16 tokens
    // 13.0 / 19.5 against 12.8 / 16.3); 0 turns the ESIMD kernel off
    static const int min_tokens = ggml_sycl_get_env("GGML_SYCL_GDN_ESIMD_MIN", 16);
    const auto       arch       = ggml_sycl_info().devices[ctx.device].hw_info.arch;
    const auto       aligned    = [](const void * p) { return (uintptr_t) p % 16 == 0; };
    return min_tokens > 0 && a.n_tokens >= min_tokens && S_v == GDN_E_SV && !kda && K == 1 &&
           (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31) && aligned(q) && aligned(k) &&
           aligned(v) && aligned(state_in) && aligned(attn) && aligned(state_out) && a.sq1 % 4 == 0 &&
           a.sq2 % 4 == 0 && a.sq3 % 4 == 0 && a.sv1 % 4 == 0 && a.sv2 % 4 == 0 && a.sv3 % 4 == 0;
#else
    GGML_UNUSED_VARS(ctx, S_v, kda, K, q, k, v, state_in, attn, state_out, a);
    return false;
#endif
}

void ggml_sycl_gdn_esimd(ggml_backend_sycl_context & ctx, const float * q, const float * k, const float * v,
                         const float * g, const float * beta, const float * state_in, float * attn, float * state_out,
                         const ggml_sycl_gdn_esimd_args & a) {
#ifdef GGML_SYCL_HAS_DPAS
    // the fewest columns per thread whose threads fit in one wave of 256-GRF threads (4 per XVE; nsm counts 16 XVEs):
    // Arc Pro B70, 2048 tokens: 32 heads 960 us at 4 columns, 1111 at 8; 48 heads 1322 at 8, 1933 at 4 (two waves),
    // measured before the pointer walk and the scaled state
    const int64_t wave = (int64_t) ggml_sycl_info().devices[ctx.device].nsm * 16 * 4;
    auto launch = [&](auto ncv) {
        constexpr int           NC = decltype(ncv)::value;
        const sycl::nd_range<3> range(sycl::range<3>(a.n_seqs, a.H, GDN_E_SV / NC), sycl::range<3>(1, 1, GDN_E_SV / NC));
        auto kern = [=](sycl::nd_item<3> it) SYCL_ESIMD_FUNCTION {
            gdn_esimd_thread<NC>(q, k, v, g, beta, state_in, attn, state_out, a, it);
        };
        ctx.stream()->parallel_for(range, gdn_esimd_grf256<decltype(kern)>{ kern });
    };
    const int64_t n_cols = a.n_seqs * a.H * GDN_E_SV;
    if (n_cols / 4 <= wave) {
        launch(std::integral_constant<int, 4>());
    } else if (n_cols / 8 <= wave) {
        launch(std::integral_constant<int, 8>());
    } else {
        launch(std::integral_constant<int, 16>());
    }
#else
    GGML_UNUSED_VARS(ctx, q, k, v, g, beta, state_in, attn, state_out, a);
    GGML_ABORT("the ESIMD gated delta net needs an Intel compiler");
#endif
}
