// Prompt-size GEMM with the weights dequantized inside the kernel: Q4_K, Q5_K, IQ4_XS on Xe2 (Battlemage) XMX.
//
// dst[t * ldd + r] = sum_k W[r][k] * Y[t][k], W quantized (plain blocks or the reordered layout), Y f16 [ntok][K].
// The library route expands W to f16 in memory and runs oneMKL; at prompt sizes below ~1.5k tokens the expansion
// dominates. Here a work-group of 16 threads computes 128 rows x 256 tokens:
//  - weights: each 32-k chunk of the 128 rows is decoded once into SLM (thread t: row block t % 8, k step t / 8, f32
//    math rounded to f16 once), three SLM buffers with a split barrier (decode chunk c + 1, fence, signal, compute
//    chunk c, wait), so the decode overlaps the DPAS work and no full barrier runs per chunk;
//  - activations: each thread loads its [64 tokens x 16 k] DPAS operands straight from global memory with L1-cached
//    2D loads (the 4 threads of a token group share them through L1); staging them through SLM cost ~25%;
//  - the qs bytes of a chunk pair are gathered two chunks ahead, the next block's scales one block ahead.
// Q4_K / Q5_K nibbles come out of 4 bytes x 16 rows with one shift and mask on dwords (Q5_K's fifth bit likewise).
// Arc Pro B70, TFLOPS at 17408 x 2048 x 5120 / 5120 x 2048 x 17408 against expand + oneMKL: Q4_K 109 / 104 vs
// 107 / 100, IQ4_XS 96.3 / 90.8 vs 96.6 / 90.3, Q5_K 106.1 / 101.2 vs 106.7 / 99.2; +12..23% at 1024 tokens.
// Qwen3.8-27B one card: pp256 685 -> 1147, pp512 1047 -> 1347, pp1024 1340 -> 1500, pp2048 1546 -> 1551 t/s.
#include "pgemm.hpp"
#include "dpas.hpp"

#if defined(__INTEL_LLVM_COMPILER)
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace {

template <typename F> struct pg_grf256 {
    F f;
    void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL { f(it); }
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    }
};

constexpr int PG_BR  = 128;   // weight rows per work-group
constexpr int PG_BT  = 256;   // tokens per work-group
constexpr int PG_BUF = 8192;  // one decoded chunk: 128 rows x 32 k f16

// The work-group's 16 threads multiply against the SLM chunk: thread (tr, tt) = (t / 4, t % 4) owns rows 32 tr..
// and tokens 64 tt.. of the tile.
struct pg_compute {
    sycl::ext::intel::experimental::esimd::config_2d_mem_access<sycl::half, 16, 32, 1> ya, yb;
    sycl::ext::intel::esimd::simd<float, 128> acc0[8], acc1[8];
    int tr;

    ESIMD_INLINE pg_compute(const sycl::half * Y, int K, int ntok, int tok_base, int t)
        : ya(Y, (unsigned) K * 2 - 1, (unsigned) ntok - 1, (unsigned) K * 2 - 1, 0, tok_base + 64 * (t % 4)),
          yb(Y, (unsigned) K * 2 - 1, (unsigned) ntok - 1, (unsigned) K * 2 - 1, 0, tok_base + 64 * (t % 4) + 32),
          tr(t / 4) {
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            acc0[g] = 0.0f;
            acc1[g] = 0.0f;
        }
    }

    // chunk c (k 32 c..) from SLM buffer buf
    ESIMD_INLINE void chunk(int c, int buf) {
        using namespace sycl::ext::intel::esimd;
        namespace xmx = sycl::ext::intel::esimd::xmx;
        namespace esx = sycl::ext::intel::experimental::esimd;
        const uint32_t base = buf * PG_BUF;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            ya.set_x(32 * c + 16 * ks);
            yb.set_x(32 * c + 16 * ks);
            simd<sycl::half, 512> y0 = esx::lsc_load_2d<sycl::half, 16, 32, 1, false, false, cache_hint::cached, cache_hint::cached>(ya);
            simd<sycl::half, 512> y1 = esx::lsc_load_2d<sycl::half, 16, 32, 1, false, false, cache_hint::cached, cache_hint::cached>(yb);
            simd<uint32_t, 128>   bd0 = slm_block_load<uint32_t, 128>(base + (uint32_t) (ks * 8 + 2 * tr) * 512);
            simd<uint32_t, 128>   bd1 = slm_block_load<uint32_t, 128>(base + (uint32_t) (ks * 8 + 2 * tr + 1) * 512);
            simd<sycl::half, 256> b0  = bd0.template bit_cast_view<sycl::half>();
            simd<sycl::half, 256> b1  = bd1.template bit_cast_view<sycl::half>();
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                simd<sycl::half, 128> a = g < 4 ? simd<sycl::half, 128>(y0.template select<128, 1>((g % 4) * 128))
                                                : simd<sycl::half, 128>(y1.template select<128, 1>((g % 4) * 128));
                acc0[g] = xmx::dpas<8, 8, float, float>(acc0[g], b0, a);
                acc1[g] = xmx::dpas<8, 8, float, float>(acc1[g], b1, a);
            }
        }
    }

    ESIMD_INLINE void store(float * dst, int ntok, int64_t ldd, int row_base, int tok_base, int t) {
        using namespace sycl::ext::intel::esimd;
        const int row0 = row_base + 32 * tr;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int tok = tok_base + 64 * (t % 4) + 8 * g + c;
                if (tok < ntok) {
                    block_store<float, 16>(dst + (int64_t) tok * ldd + row0, acc0[g].template select<16, 1>(c * 16).read());
                    block_store<float, 16>(dst + (int64_t) tok * ldd + row0 + 16, acc1[g].template select<16, 1>(c * 16).read());
                }
            }
        }
    }
};

// Q4_K (QH_STRIDE 0) / Q5_K: chunk j of a block has scale d * sc_j and min -dmin * m_j; chunks 2p and 2p + 1 share the
// 32 qs bytes p * 32.. (low / high nibbles); Q5_K adds bit j of qh byte l as bit 4 of chunk j's weight at k l. The
// strides are the plain block size, or the reordered layout's per-block sizes of [qs] [qh] [scales] [dm].
template <uint32_t QS_STRIDE, uint32_t SC_STRIDE, uint32_t DM_STRIDE, uint32_t QH_STRIDE = 0>
ESIMD_INLINE void pgemm_k4(const uint8_t * qs, const uint8_t * scales, const uint8_t * dm, const uint8_t * qh,
                           const sycl::half * Y, float * dst, int K, int ntok, int64_t ldd, int row_base, int tok_base, int t) {
    using namespace sycl::ext::intel::esimd;
    namespace esx = sycl::ext::intel::experimental::esimd;
    slm_init<3 * PG_BUF>();
    const int rb = t % 8, s = t / 8;  // decode share: row block rb, k step s of every chunk
    const int nblk = K / QK_K, npair = K / 64;
    const simd<uint32_t, 16> bi0 = (simd<uint32_t, 16>(0, 1) + (uint32_t) (row_base + 16 * rb)) * (uint32_t) nblk;
    simd<uint32_t, 64> wq[2];          // qs bytes of pair P in slot P & 1
    simd<uint32_t, 48> sw, swn;        // scale bytes of the current / next block
    simd<uint32_t, 16> dmw, dmwn;
    simd<uint32_t, 64> hq, hqn;        // Q5_K: qh bytes 16 s.. of the current / next block
    auto gather_pair = [&](int P, int slot) {
        if (P < npair) {
            const simd<uint32_t, 16> bi = bi0 + (uint32_t) (P / 4);
            wq[slot] = gather<uint32_t, 64, 4>((const uint32_t *) qs, bi * QS_STRIDE + (uint32_t) ((P % 4) * 32 + 16 * s));
        }
    };
    auto gather_scales = [&](int ib) {
        if (ib < nblk) {
            const simd<uint32_t, 16> bi = bi0 + (uint32_t) ib;
            swn  = gather<uint32_t, 48, 3>((const uint32_t *) scales, bi * SC_STRIDE);
            dmwn = gather<uint32_t, 16>((const uint32_t *) dm, bi * DM_STRIDE);
            if constexpr (QH_STRIDE > 0) {
                hqn = gather<uint32_t, 64, 4>((const uint32_t *) qh, bi * QH_STRIDE + (uint32_t) (16 * s));
            }
        }
    };
    // chunk j (0..7 of its block) from slot into SLM buffer buf
    auto decode = [&](int buf, int j, int slot) {
        simd<sycl::half, 32> dmh  = dmw.template bit_cast_view<sycl::half>().read();
        simd<float, 16>      dall = convert<float>(simd<sycl::half, 16>(dmh.template select<16, 2>(0)));
        simd<float, 16>      dmin = convert<float>(simd<sycl::half, 16>(dmh.template select<16, 2>(1)));
        auto byte_of = [&](int i) -> simd<uint16_t, 16> {
            return convert<uint16_t>((simd<uint32_t, 16>(sw.template select<16, 1>(16 * (i / 4))) >> (8 * (i % 4))) & 0xFF);
        };
        simd<uint16_t, 16> a, m;
        if (j < 4) {
            a = byte_of(j) & 63;
            m = byte_of(j + 4) & 63;
        } else {
            a = (byte_of(j + 4) & 0xF) | ((byte_of(j - 4) >> 6) << 4);
            m = (byte_of(j + 4) >> 4) | ((byte_of(j) >> 6) << 4);
        }
        simd<float, 16> sch = convert<float>(a) * dall;
        simd<float, 16> mnh = convert<float>(m) * (-dmin);
        simd<float, 32> sc2, mn2;
        sc2.template select<16, 2>(0) = sch;
        sc2.template select<16, 2>(1) = sch;
        mn2.template select<16, 2>(0) = mnh;
        mn2.template select<16, 2>(1) = mnh;
        simd<sycl::half, 256> wb;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            // the 4-bit values of all 4 bytes of 16 rows with one shift and mask on dwords; Q5_K's bit 4 the same way
            simd<uint32_t, 16> tv = (wq[slot].template select<16, 1>(16 * g) >> (4 * (j & 1))) & 0x0F0F0F0Fu;
            if constexpr (QH_STRIDE > 0) {
                tv |= ((hq.template select<16, 1>(16 * g) >> j) & 0x01010101u) << 4;
            }
            auto tm = tv.template bit_cast_view<uint8_t, 16, 4>();
#pragma unroll
            for (int jj = 0; jj < 4; jj += 2) {
                simd<uint8_t, 32> v = tm.template select<16, 1, 2, 1>(0, jj);
                wb.template select<32, 1>((2 * g + jj / 2) * 32) = convert<sycl::half>(convert<float>(v) * sc2 + mn2);
            }
        }
        slm_block_store<sycl::half, 256>(buf * PG_BUF + (uint32_t) (s * 8 + rb) * 512, wb);
    };
    pg_compute cmp(Y, K, ntok, tok_base, t);
    gather_pair(0, 0);
    gather_pair(1, 1);
    gather_scales(0);
    sw  = swn;
    dmw = dmwn;
    hq  = hqn;
    decode(0, 0, 0);
    barrier();
    const int nchunk = K / 32;
    for (int c0 = 0; c0 < nchunk; c0 += 8) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int c = c0 + j;
            // decode chunk c + 1 into buffer (c + 1) % 3, signal, compute chunk c, wait: a thread past wait(c - 1) knows
            // every thread has finished chunk c - 2, the last reader of buffer (c + 1) % 3
            if (c + 1 < nchunk) {
                if (j == 7) {
                    sw  = swn;
                    dmw = dmwn;
                    hq  = hqn;
                }
                decode((c + 1) % 3, (j + 1) % 8, ((j + 1) / 2) & 1);
            }
            fence<fence_mask::local_barrier>();
            esx::split_barrier<esx::split_barrier_action::signal>();
            cmp.chunk(c, c % 3);
            if (j % 2 == 0) {
                gather_pair(c / 2 + 2, (j / 2) & 1);
            }
            if (j == 0) {
                gather_scales(c0 / 8 + 1);
            }
            esx::split_barrier<esx::split_barrier_action::wait>();
        }
    }
    cmp.store(dst, ntok, ldd, row_base, tok_base, t);
}

// IQ4_XS: chunk c is sub-block j = c % 8 of block c / 8; its 16 qs bytes j * 16.. hold k 0..15 (low nibbles) and
// 16..31 (high nibbles). Every chunk gathers its own bytes, two chunks ahead. PLAIN: block_iq4_xs rows (d and
// scales_h in the block's first dword); else the reordered layout [qs] [scales_l] [scales_h] [d].
template <bool PLAIN>
ESIMD_INLINE void pgemm_iq4xs(const uint8_t * vx, size_t nb, const sycl::half * Y, float * dst, int K, int ntok, int64_t ldd,
                              int row_base, int tok_base, int t) {
    using namespace sycl::ext::intel::esimd;
    namespace esx = sycl::ext::intel::experimental::esimd;
    constexpr uint32_t QS_STRIDE = PLAIN ? sizeof(block_iq4_xs) : QK_K / 2;
    constexpr uint32_t SL_STRIDE = PLAIN ? sizeof(block_iq4_xs) : QK_K / 64;
    const uint8_t * qs       = PLAIN ? vx + offsetof(block_iq4_xs, qs) : vx;
    const uint8_t * scales_l = PLAIN ? vx + offsetof(block_iq4_xs, scales_l) : vx + nb * (QK_K / 2);
    const uint8_t * scales_h = scales_l + nb * (QK_K / 64);
    const uint8_t * dd       = scales_h + nb * sizeof(uint16_t);
    slm_init<3 * PG_BUF>();
    const int rb = t % 8, s = t / 8;
    const int nblk = K / QK_K;
    const simd<uint32_t, 16> bi0 = (simd<uint32_t, 16>(0, 1) + (uint32_t) (row_base + 16 * rb)) * (uint32_t) nblk;
    simd<uint32_t, 64> wq[2];
    simd<uint32_t, 16> sl, sh, sln, shn;
    simd<float, 16>    drow, drown;
    auto gather_chunk = [&](int c, int slot) {
        if (c < K / 32) {
            const simd<uint32_t, 16> bi = bi0 + (uint32_t) (c / 8);
            wq[slot] = gather<uint32_t, 64, 4>((const uint32_t *) qs, bi * QS_STRIDE + (uint32_t) ((c % 8) * 16));
        }
    };
    auto gather_scales = [&](int ib) {
        if (ib < nblk) {
            const simd<uint32_t, 16> bi = bi0 + (uint32_t) ib;
            sln = gather<uint32_t, 16>((const uint32_t *) scales_l, bi * SL_STRIDE);
            simd<uint32_t, 16> dbits;
            if constexpr (PLAIN) {
                const simd<uint32_t, 16> w = gather<uint32_t, 16>((const uint32_t *) vx, bi * (uint32_t) sizeof(block_iq4_xs));
                dbits = w & 0xFFFF;
                shn   = w >> 16;
            } else {
                dbits = dpas_gather_u16(dd, bi);
                shn   = dpas_gather_u16(scales_h, bi);
            }
            simd<uint16_t, 16> d16 = convert<uint16_t>(dbits);
            drown = convert<float>(simd<sycl::half, 16>(d16.template bit_cast_view<sycl::half>().read()));
        }
    };
    auto decode = [&](int buf, int j, int slot) {
        const simd<uint32_t, 16> ls = ((sl >> (4 * j)) & 0xF) | (((sh >> (2 * j)) & 3) << 4);
        const simd<float, 16>    sc = (convert<float>(ls) - 32.0f) * drow;
        simd<float, 32>          sc2;
        sc2.template select<16, 2>(0) = sc;
        sc2.template select<16, 2>(1) = sc;
        const uint8_t         sh4 = (uint8_t) (4 * s);
        simd<sycl::half, 256> wb;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            simd<uint32_t, 16> tv = (wq[slot].template select<16, 1>(16 * g) >> (uint32_t) sh4) & 0x0F0F0F0Fu;
            auto               tm = tv.template bit_cast_view<uint8_t, 16, 4>();
#pragma unroll
            for (int jj = 0; jj < 4; jj += 2) {
                simd<uint8_t, 32> v = tm.template select<16, 1, 2, 1>(0, jj);
                wb.template select<32, 1>((2 * g + jj / 2) * 32) =
                    convert<sycl::half>(dpas_tile_traits<GGML_TYPE_IQ4_XS>::codebook(convert<float>(v)) * sc2);
            }
        }
        slm_block_store<sycl::half, 256>(buf * PG_BUF + (uint32_t) (s * 8 + rb) * 512, wb);
    };
    pg_compute cmp(Y, K, ntok, tok_base, t);
    gather_chunk(0, 0);
    gather_chunk(1, 1);
    gather_scales(0);
    sl   = sln;
    sh   = shn;
    drow = drown;
    decode(0, 0, 0);
    barrier();
    const int nchunk = K / 32;
    for (int c0 = 0; c0 < nchunk; c0 += 8) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int c = c0 + j;
            if (c + 1 < nchunk) {
                if (j == 7) {
                    sl   = sln;
                    sh   = shn;
                    drow = drown;
                }
                decode((c + 1) % 3, (j + 1) % 8, (j + 1) & 1);
            }
            fence<fence_mask::local_barrier>();
            esx::split_barrier<esx::split_barrier_action::signal>();
            cmp.chunk(c, c % 3);
            gather_chunk(c + 2, j & 1);
            if (j == 0) {
                gather_scales(c0 / 8 + 1);
            }
            esx::split_barrier<esx::split_barrier_action::wait>();
        }
    }
    cmp.store(dst, ntok, ldd, row_base, tok_base, t);
}

template <typename F> static void pgemm_launch(int nrows, int ntok, dpct::queue_ptr stream, F body) {
    const int row_tiles = nrows / PG_BR;
    const int tok_tiles = (ntok + PG_BT - 1) / PG_BT;
    stream->submit([&](sycl::handler & h) {
        auto kern = [=](sycl::nd_item<2> it) SYCL_ESIMD_FUNCTION {
            body(it.get_group(0) * PG_BR, it.get_group(1) * PG_BT, (int) it.get_local_id(1));
        };
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(row_tiles, (size_t) tok_tiles * 16), sycl::range<2>(1, 16)),
                       pg_grf256<decltype(kern)>{ kern });
    });
}

}  // namespace

bool ggml_sycl_pgemm_supported(int device, ggml_type type) {
    static const bool enabled = ggml_sycl_get_env("GGML_SYCL_PGEMM", 1) != 0;
    const gpu_arch    arch    = ggml_sycl_info().devices[device].hw_info.arch;
    return enabled && g_ggml_sycl_enable_esimd && (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31) &&
           (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_IQ4_XS);
}

bool ggml_sycl_pgemm(ggml_type type, const void * vx, bool reordered, const sycl::half * Y, float * dst, int K, int nrows, int ntok,
                     int64_t ldd, dpct::queue_ptr stream) {
    if (K % QK_K != 0 || nrows % PG_BR != 0 || ntok < 64 || (uintptr_t) Y % 64 != 0 || (uintptr_t) vx % 4 != 0 || ldd % 4 != 0) {
        return false;
    }
    const uint8_t * base = (const uint8_t *) vx;
    const size_t    nb   = (size_t) nrows * (K / QK_K);
    if (type == GGML_TYPE_IQ4_XS) {
        if (reordered) {
            pgemm_launch(nrows, ntok, stream, [=](int r0, int t0, int t) {
                pgemm_iq4xs<false>(base, nb, Y, dst, K, ntok, ldd, r0, t0, t);
            });
        } else {
            pgemm_launch(nrows, ntok, stream, [=](int r0, int t0, int t) {
                pgemm_iq4xs<true>(base, nb, Y, dst, K, ntok, ldd, r0, t0, t);
            });
        }
        return true;
    }
    if (type == GGML_TYPE_Q5_K) {
        constexpr uint32_t BS = sizeof(block_q5_K);
        if (reordered) {
            const uint8_t * qh = base + nb * (QK_K / 2);
            const uint8_t * sc = qh + nb * (QK_K / 8);
            const uint8_t * dm = sc + nb * K_SCALE_SIZE;
            pgemm_launch(nrows, ntok, stream, [=](int r0, int t0, int t) {
                pgemm_k4<QK_K / 2, K_SCALE_SIZE, 4, QK_K / 8>(base, sc, dm, qh, Y, dst, K, ntok, ldd, r0, t0, t);
            });
        } else {
            pgemm_launch(nrows, ntok, stream, [=](int r0, int t0, int t) {
                pgemm_k4<BS, BS, BS, BS>(base + offsetof(block_q5_K, qs), base + offsetof(block_q5_K, scales), base,
                                         base + offsetof(block_q5_K, qh), Y, dst, K, ntok, ldd, r0, t0, t);
            });
        }
        return true;
    }
    if (type == GGML_TYPE_Q4_K) {
        constexpr uint32_t BS = sizeof(block_q4_K);
        if (reordered) {
            const uint8_t * sc = base + nb * (QK_K / 2);
            const uint8_t * dm = sc + nb * K_SCALE_SIZE;
            pgemm_launch(nrows, ntok, stream, [=](int r0, int t0, int t) {
                pgemm_k4<QK_K / 2, K_SCALE_SIZE, 4>(base, sc, dm, nullptr, Y, dst, K, ntok, ldd, r0, t0, t);
            });
        } else {
            pgemm_launch(nrows, ntok, stream, [=](int r0, int t0, int t) {
                pgemm_k4<BS, BS, BS>(base + offsetof(block_q4_K, qs), base + offsetof(block_q4_K, scales), base, nullptr, Y, dst,
                                     K, ntok, ldd, r0, t0, t);
            });
        }
        return true;
    }
    return false;
}

#else
bool ggml_sycl_pgemm_supported(int, ggml_type) {
    return false;
}
bool ggml_sycl_pgemm(ggml_type, const void *, bool, const sycl::half *, float *, int, int, int, int64_t, dpct::queue_ptr) {
    return false;
}
#endif
