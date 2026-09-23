#ifndef GGML_SYCL_DPAS_HPP
#define GGML_SYCL_DPAS_HPP

// Dequantization of reordered weight tiles into XMX (DPAS) operands, shared by the DPAS mat-vec
// (dmmv.cpp) and the grouped MoE expert GEMM (mmid-dpas.cpp). Written for 16-wide DPAS (Xe2).
//
// A tile is 16 output rows. dpas_tile_traits<T>::block(vx, nb, bi, fn) dequantizes one QK_K block
// of the tile, bi holding the 16 rows' block indices, and calls fn(k offset in the block, B) for
// its 16 steps of 16 k, B being the f16 DPAS B operand (16 k x 16 rows, VNNI). The rows are
// gathered with one lane per row, so a 2D byte view of the gathered words holds k positions
// (k, k+1) across the 16 rows, i.e. one VNNI row pair. Values are computed in f32 and rounded to
// f16 once: an f16 scale or min would carry its rounding error into every weight of a chunk.

#include "common.hpp"

#if defined(__INTEL_LLVM_COMPILER)
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

#define GGML_SYCL_HAS_DPAS

constexpr int GGML_SYCL_DPAS_ROWS = 16;  // DPAS execution size on Xe2: output rows per tile

template <ggml_type T> struct dpas_tile_traits;

// Q4_K / Q5_K chunk scales and mins of the tile's 16 rows (get_scale_min_k4 layout, 12 bytes per
// row at scales + 12 bi, d and dmin as half2 at dm + 4 bi): sc2[j] / mn2[j] hold each row's
// d*sc_j / -dmin*m_j twice, for the two k of a VNNI pair.
static ESIMD_INLINE void dpas_scale_min_k4(const uint8_t * scales, const uint8_t * dm,
                                                sycl::ext::intel::esimd::simd<uint32_t, 16> bi,
                                                sycl::ext::intel::esimd::simd<float, 32> (&sc2)[8],
                                                sycl::ext::intel::esimd::simd<float, 32> (&mn2)[8]) {
    using namespace sycl::ext::intel::esimd;
    // element-major: dword i of all 16 rows at [16 i]
    simd<uint32_t, 48>   sw   = gather<uint32_t, 48, 3>((const uint32_t *) scales, bi * (uint32_t) K_SCALE_SIZE);
    simd<uint32_t, 16>   dmw  = gather<uint32_t, 16>((const uint32_t *) dm, bi * 4u);
    simd<sycl::half, 32> dmh  = dmw.bit_cast_view<sycl::half>().read();
    simd<float, 16>      dall = convert<float>(simd<sycl::half, 16>(dmh.select<16, 2>(0)));
    simd<float, 16>      dmin = convert<float>(simd<sycl::half, 16>(dmh.select<16, 2>(1)));
    auto byte_of = [&](int i) -> simd<uint16_t, 16> {
        return convert<uint16_t>((simd<uint32_t, 16>(sw.select<16, 1>(16 * (i / 4))) >> (8 * (i % 4))) & 0xFF);
    };
#pragma unroll
    for (int j = 0; j < 8; ++j) {
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
        sc2[j].select<16, 2>(0) = sch;
        sc2[j].select<16, 2>(1) = sch;
        mn2[j].select<16, 2>(0) = mnh;
        mn2[j].select<16, 2>(1) = mnh;
    }
}

// Q4_K, reorder layout [qs: nb*(QK_K/2)] [scales: nb*12] [dm: nb*half2]. Chunk j (32 k)
// of a block has scale d*sc_j and min -dmin*m_j; chunks 2p and 2p+1 share the 32 qs bytes
// p*32.. (low and high nibbles).
template <> struct dpas_tile_traits<GGML_TYPE_Q4_K> {
    // calls fn(k offset in the block, B) for the block's 16 k steps; bi = the rows' block indices
    template <typename F>
    static ESIMD_INLINE void block(const void * vx, size_t nb, sycl::ext::intel::esimd::simd<uint32_t, 16> bi, F && fn) {
        using namespace sycl::ext::intel::esimd;
        const uint8_t * qs     = (const uint8_t *) vx;
        const uint8_t * scales = qs + nb * (QK_K / 2);
        const uint8_t * dm     = scales + nb * K_SCALE_SIZE;

        simd<float, 32> sc2[8];
        simd<float, 32> mn2[8];
        dpas_scale_min_k4(scales, dm, bi, sc2, mn2);

#pragma unroll
        for (int p = 0; p < 4; ++p) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                // 16 qs bytes per row: k 16h.. of chunks 2p (low nibbles) and 2p+1 (high)
                simd<uint32_t, 64> w4 = gather<uint32_t, 64, 4>((const uint32_t *) qs, bi * (uint32_t) (QK_K / 2) + (p * 32 + h * 16));
                simd<sycl::half, 256> b_lo;
                simd<sycl::half, 256> b_hi;
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    simd<uint32_t, 16> w  = w4.select<16, 1>(16 * g);
                    auto               wm = w.bit_cast_view<uint8_t, 16, 4>();
#pragma unroll
                    for (int j = 0; j < 4; j += 2) {
                        simd<uint8_t, 32> q  = wm.select<16, 1, 2, 1>(0, j);
                        const int         kp = (4 * g + j) / 2;
                        // f32 math, one rounding to f16 per weight: an f16 min term would carry
                        // its rounding error into every weight of the chunk
                        b_lo.select<32, 1>(kp * 32) = convert<sycl::half>(convert<float>(simd<uint8_t, 32>(q & 0x0F)) * sc2[2 * p] + mn2[2 * p]);
                        b_hi.select<32, 1>(kp * 32) = convert<sycl::half>(convert<float>(simd<uint8_t, 32>(q >> 4)) * sc2[2 * p + 1] + mn2[2 * p + 1]);
                    }
                }
                fn((2 * p) * 32 + h * 16, b_lo);
                fn((2 * p + 1) * 32 + h * 16, b_hi);
            }
        }
    }
};

// Q6_K, reorder layout [ql: nb*(QK_K/2)] [qh: nb*(QK_K/4)] [scales: nb*(QK_K/16) int8] [d: nb*half].
// Half n (128 k) of a block: group j (32 k, j = 0..3) at k 128n + 32j takes the low (j < 2) or
// high (j >= 2) nibbles of ql[64n + 32(j%2) + l] and bits 2j, 2j+1 of qh[32n + l] for l = 0..31;
// value = d * scales[8n + l/16 + 2j] * (q - 32), so a 16-k step has one scale per row.
template <> struct dpas_tile_traits<GGML_TYPE_Q6_K> {
    template <typename F>
    static ESIMD_INLINE void block(const void * vx, size_t nb, sycl::ext::intel::esimd::simd<uint32_t, 16> bi, F && fn) {
        using namespace sycl::ext::intel::esimd;
        const uint8_t *    ql     = (const uint8_t *) vx;
        const uint8_t *    qh     = ql + nb * (QK_K / 2);
        const int8_t *     scales = (const int8_t *) (qh + nb * (QK_K / 4));
        const sycl::half * d      = (const sycl::half *) (scales + nb * (QK_K / 16));

        // the 16 int8 scales of every row, element-major: dword i (scales 4i..4i+3) of all rows at [16 i]
        simd<uint32_t, 64> sw = gather<uint32_t, 64, 4>((const uint32_t *) scales, bi * (uint32_t) (QK_K / 16));
        // d of every row: 2-byte loads through the enclosing dword
        simd<uint32_t, 16> dw  = gather<uint32_t, 16>((const uint32_t *) d, (bi & ~1u) * 2u);
        simd<uint16_t, 16> dbits = convert<uint16_t>((dw >> ((bi & 1u) * 16u)) & 0xFFFF);
        simd<float, 16>    drow  = convert<float>(simd<sycl::half, 16>(dbits.bit_cast_view<sycl::half>().read()));
        auto scale2 = [&](int i) -> simd<float, 32> {
            simd<int32_t, 16> b = (simd<int32_t, 16>(sw.select<16, 1>(16 * (i / 4)).bit_cast_view<int32_t>().read()) << (24 - 8 * (i % 4))) >> 24;
            simd<float, 16>   f = convert<float>(b) * drow;
            simd<float, 32>   r;
            r.select<16, 2>(0) = f;
            r.select<16, 2>(1) = f;
            return r;
        };

#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                // qh bytes 32n + 16h.. : 2 high bits for each of the 4 groups
                simd<uint32_t, 64> hw4 = gather<uint32_t, 64, 4>((const uint32_t *) qh, bi * (uint32_t) (QK_K / 4) + (32 * n + 16 * h));
#pragma unroll
                for (int jp = 0; jp < 2; ++jp) {
                    // ql bytes 64n + 32jp + 16h.. : low nibbles for group jp, high nibbles for group jp+2
                    simd<uint32_t, 64> w4 = gather<uint32_t, 64, 4>((const uint32_t *) ql, bi * (uint32_t) (QK_K / 2) + (64 * n + 32 * jp + 16 * h));
                    const simd<float, 32> s_lo = scale2(8 * n + h + 2 * jp);
                    const simd<float, 32> s_hi = scale2(8 * n + h + 2 * (jp + 2));
                    simd<sycl::half, 256> b_lo;
                    simd<sycl::half, 256> b_hi;
#pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        simd<uint32_t, 16> w   = w4.select<16, 1>(16 * g);
                        simd<uint32_t, 16> hw  = hw4.select<16, 1>(16 * g);
                        auto               wm  = w.bit_cast_view<uint8_t, 16, 4>();
                        auto               hwm = hw.bit_cast_view<uint8_t, 16, 4>();
#pragma unroll
                        for (int j = 0; j < 4; j += 2) {
                            simd<uint8_t, 32> q  = wm.select<16, 1, 2, 1>(0, j);
                            simd<uint8_t, 32> qb = hwm.select<16, 1, 2, 1>(0, j);
                            simd<uint8_t, 32> lo = (q & 0x0F) | (((qb >> (2 * jp)) & 3) << 4);
                            simd<uint8_t, 32> hi = (q >> 4) | (((qb >> (2 * (jp + 2))) & 3) << 4);
                            const int         kp = (4 * g + j) / 2;
                            b_lo.select<32, 1>(kp * 32) = convert<sycl::half>((convert<float>(lo) - 32.0f) * s_lo);
                            b_hi.select<32, 1>(kp * 32) = convert<sycl::half>((convert<float>(hi) - 32.0f) * s_hi);
                        }
                    }
                    fn(128 * n + 32 * jp + 16 * h, b_lo);
                    fn(128 * n + 32 * (jp + 2) + 16 * h, b_hi);
                }
            }
        }
    }
};

#endif // __INTEL_LLVM_COMPILER

#endif // GGML_SYCL_DPAS_HPP
