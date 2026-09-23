#ifndef GGML_SYCL_ESIMD_HPP
#define GGML_SYCL_ESIMD_HPP

#include <sycl/ext/intel/esimd.hpp>

#include "common.hpp"

namespace ggml_sycl_esimd {

constexpr int GGML_SYCL_DMMV_ESIMD_WG_SIZE = 4;

//
// Shared ESIMD building blocks for the reordered K-quant dequantize-matvec
// kernels.
//
// The reordered K-quant ESIMD matvec kernels share one skeleton: per super-block,
// load a 256-float activation slice, load one weight block, dequantize it into 8
// chunks of 32 and MAC each chunk against the matching activation slice, then
// reduce and run a lane-0 epilogue.
//
// Each K-quant kernel emits exactly 8 chunks of 32 mapping to activation slices
// 0..7, so the per-block work is captured by esimd_reorder_q_traits<T>::mac_pair_nc,
// which dequantizes two weight blocks once and MACs both against NC activation
// columns with the two FMA chains interleaved (co-scheduled to hide FMA latency).
// The "pair" is the (row0,row1) row pair owned by one work-group, so the
// layout+dequant is written once per quant type here.
//

template <ggml_type T> struct esimd_reorder_q_traits;

// build a 32-lane vector whose low 16 lanes are `lo` and high 16 are `hi`
// (a super-chunk splits into two 16-wide halves with distinct scale/min codes).
static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 32> splat_lo_hi(float lo, float hi) {
    using namespace sycl::ext::intel::esimd;
    simd<float, 32> v;
    v.select<16, 1>(0)  = lo;
    v.select<16, 1>(16) = hi;
    return v;
}

// unpack one block of Q4_K/Q5_K scale/min codes (get_scale_min_k4 layout) into 8
// float scales (dall * sc) and 8 float mins (-dmin * m); the min carries the
// negation so the dequant epilogue adds.
static ESIMD_INLINE void unpack_scale_min_k4(
        sycl::ext::intel::esimd::simd<uint8_t, 12> scales, float dall, float dmin,
        sycl::ext::intel::esimd::simd<float, 8> & scale_f,
        sycl::ext::intel::esimd::simd<float, 8> & min_f) {
    using namespace sycl::ext::intel::esimd;
    simd<uint8_t, 8> sc = 0;
    simd<uint8_t, 8> m  = 0;
    simd<uint8_t, 4> scale_lo = scales.select<4, 1>(0);
    simd<uint8_t, 4> min_lo   = scales.select<4, 1>(4);
    simd<uint8_t, 4> hi_bits  = scales.select<4, 1>(8);
    sc.select<4, 1>(0) = scale_lo & simd<uint8_t, 4>(0x3F);
    sc.select<4, 1>(4) = (hi_bits & simd<uint8_t, 4>(0x0F)) |
                         ((scale_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
    m.select<4, 1>(0)  = min_lo & simd<uint8_t, 4>(0x3F);
    m.select<4, 1>(4)  = (hi_bits >> simd<uint8_t, 4>(4)) |
                         ((min_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
    scale_f = convert<float>(sc) * dall;
    min_f   = convert<float>(m) * (-dmin);
}

// ---------------------------------------------------------------------------
// Q2_K, SOA reorder layout produced by reorder_qw_q2_k:
//   [qs: nb*(QK_K/4)] [scales: nb*(QK_K/16)] [dm: nb*sizeof(half2)]
// with nb = nrows*num_blocks_per_row.
//
// 2 bits per weight. The 8 output chunks of 32 (matching dequantize_row_q2_K)
// map to super-chunk s (0..7): byte base 32*(s/4) into the 64-byte qs array,
// bit shift 2*(s%4); the low 16 lanes use scales[2s], the high 16 use
// scales[2s+1], with dl = d*(sc & 0xF), ml = dmin*(sc >> 4), deq = dl*q - ml.
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q2_K> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    scales;
        const sycl::half * dm;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs     = (const uint8_t *) vx;
        const uint8_t * scales = qs + nb * (QK_K / 4);
        const sycl::half * dm  = (const sycl::half *) (scales + nb * (QK_K / 16));
        return { qs, scales, dm };
    }

    // NC activation columns share one dequantization of the two weight blocks
    template <int NC>
    static ESIMD_INLINE void mac_pair_nc(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            const float * y_blk, int64_t y_stride,
            sycl::ext::intel::esimd::simd<float, 32> (&acc_a)[NC],
            sycl::ext::intel::esimd::simd<float, 32> (&acc_b)[NC]) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 64> qs_a     = block_load<uint8_t, 64>(pa.qs + bia * (QK_K / 4));
        simd<uint8_t, 64> qs_b     = 0;
        simd<uint8_t, 16> scales_a = block_load<uint8_t, 16>(pa.scales + bia * (QK_K / 16));
        simd<uint8_t, 16> scales_b = 0;

        const float dall_a = (float) pa.dm[bia * 2 + 0];
        const float dmin_a = (float) pa.dm[bia * 2 + 1];
        float dall_b = 0.0f;
        float dmin_b = 0.0f;
        if (has_b) {
            qs_b     = block_load<uint8_t, 64>(pb.qs + bib * (QK_K / 4));
            scales_b = block_load<uint8_t, 16>(pb.scales + bib * (QK_K / 16));
            dall_b = (float) pb.dm[bib * 2 + 0];
            dmin_b = (float) pb.dm[bib * 2 + 1];
        }

        // per-chunk scale (d * (sc & 0xF)) and min (-dmin * (sc >> 4)), all 16 codes;
        // min carries the negation so the dequant epilogue adds (matches Q4_K/Q5_K)
        simd<float, 16> scale_f_a = convert<float>(scales_a & simd<uint8_t, 16>(0x0F)) * dall_a;
        simd<float, 16> min_f_a   = convert<float>(scales_a >> simd<uint8_t, 16>(4))  * (-dmin_a);
        simd<float, 16> scale_f_b = convert<float>(scales_b & simd<uint8_t, 16>(0x0F)) * dall_b;
        simd<float, 16> min_f_b   = convert<float>(scales_b >> simd<uint8_t, 16>(4))  * (-dmin_b);

#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int     byte_base = 32 * (s / 4);
            const uint8_t shift     = (uint8_t) (2 * (s % 4));

            simd<uint8_t, 32> qa = (qs_a.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3);
            simd<uint8_t, 32> qb = (qs_b.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3);

            const float scale_a_lo = scale_f_a[2 * s + 0];
            const float scale_a_hi = scale_f_a[2 * s + 1];
            const float min_a_lo   = min_f_a[2 * s + 0];
            const float min_a_hi   = min_f_a[2 * s + 1];
            const float scale_b_lo = scale_f_b[2 * s + 0];
            const float scale_b_hi = scale_f_b[2 * s + 1];
            const float min_b_lo   = min_f_b[2 * s + 0];
            const float min_b_hi   = min_f_b[2 * s + 1];

            simd<float, 32> scale_vec_a = splat_lo_hi(scale_a_lo, scale_a_hi);
            simd<float, 32> min_vec_a   = splat_lo_hi(min_a_lo, min_a_hi);
            simd<float, 32> scale_vec_b = splat_lo_hi(scale_b_lo, scale_b_hi);
            simd<float, 32> min_vec_b   = splat_lo_hi(min_b_lo, min_b_hi);

            simd<float, 32> deq_a = convert<float>(qa) * scale_vec_a + min_vec_a;
            simd<float, 32> deq_b = convert<float>(qb) * scale_vec_b + min_vec_b;

#pragma unroll
            for (int c = 0; c < NC; ++c) {
                simd<float, 32> y_s = block_load<float, 32>(y_blk + c * y_stride + s * 32);
                acc_a[c] += y_s * deq_a;
                acc_b[c] += y_s * deq_b;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Q3_K, SOA reorder layout produced by reorder_qw_q3_k:
//   [qs: nb*(QK_K/4)] [hmask: nb*(QK_K/8)] [scales: nb*12] [d: nb*sizeof(half)]
// with nb = nrows*num_blocks_per_row. Single super-block scale d, no dmin.
//
// 3 bits per weight: 2 low bits in qs, 1 high bit in hmask. The 8 output chunks
// of 32 (matching dequantize_row_q3_K) map to super-chunk s (0..7): byte base
// 32*(s/4) into the 64-byte qs array, bit shift 2*(s%4); the low 16 lanes use
// scale code 2s, the high 16 use 2s+1. hmask is a 32-byte array (like Q5_K's
// qh) where chunk s uses bit s of the same 32 bytes, but INVERTED: the value is
// (q & 3) - (hmask_bit_set ? 0 : 4), i.e. (q & 3) + 4*bit - 4.
//
// The 16 6-bit scale codes are packed into 12 bytes (get_scale_min layout for
// Q3_K): low nibbles from bytes 0..7, high 2 bits from bytes 8..11 shifted by
// 0/2/4/6; the dequant scale is d * (code - 32).
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q3_K> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    hmask;
        const uint8_t *    scales;
        const sycl::half * d;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs     = (const uint8_t *) vx;
        const uint8_t * hmask  = qs + nb * (QK_K / 4);
        const uint8_t * scales = hmask + nb * (QK_K / 8);
        const sycl::half * d   = (const sycl::half *) (scales + nb * 12);
        return { qs, hmask, scales, d };
    }

    // unpack the 12 packed bytes into 16 6-bit scale codes (dequantize_row_q3_K
    // aux layout), returned as float scale = d * (code - 32).
    // done with wide (8/16-lane) ops rather than four 4-lane groups.
    static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 16> unpack_scales(
            sycl::ext::intel::esimd::simd<uint8_t, 12> in, float d) {
        using namespace sycl::ext::intel::esimd;

        // low 6-bit part: codes 0..7 = low nibble of bytes 0..7,
        //                 codes 8..15 = high nibble of bytes 0..7
        simd<uint8_t, 8>  lo8 = in.select<8, 1>(0);
        simd<uint8_t, 16> code;
        code.select<8, 1>(0) = lo8 & simd<uint8_t, 8>(0x0F);
        code.select<8, 1>(8) = lo8 >> simd<uint8_t, 8>(4);

        // high 2-bit part: bytes 8..11 replicated 4x, group g (0..3) shifted 2*g
        simd<uint8_t, 16> hib;
        hib.select<4, 1>(0)  = in.select<4, 1>(8);
        hib.select<4, 1>(4)  = in.select<4, 1>(8);
        hib.select<4, 1>(8)  = in.select<4, 1>(8);
        hib.select<4, 1>(12) = in.select<4, 1>(8);
        simd<uint8_t, 16> hshift;
        hshift.select<4, 1>(0)  = 0;
        hshift.select<4, 1>(4)  = 2;
        hshift.select<4, 1>(8)  = 4;
        hshift.select<4, 1>(12) = 6;
        hib = (hib >> hshift) & simd<uint8_t, 16>(0x03);

        code = code | (hib << simd<uint8_t, 16>(4));
        return (convert<float>(code) - 32.0f) * d;
    }

    // NC activation columns share one dequantization of the two weight blocks
    template <int NC>
    static ESIMD_INLINE void mac_pair_nc(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            const float * y_blk, int64_t y_stride,
            sycl::ext::intel::esimd::simd<float, 32> (&acc_a)[NC],
            sycl::ext::intel::esimd::simd<float, 32> (&acc_b)[NC]) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 64> qs_a     = block_load<uint8_t, 64>(pa.qs + bia * (QK_K / 4));
        simd<uint8_t, 64> qs_b     = 0;
        simd<uint8_t, 32> hmask_a  = block_load<uint8_t, 32>(pa.hmask + bia * (QK_K / 8));
        simd<uint8_t, 32> hmask_b  = 0;
        simd<uint8_t, 12> scales_a = block_load<uint8_t, 12>(pa.scales + bia * 12);
        simd<uint8_t, 12> scales_b = 0;

        const float d_a = (float) pa.d[bia];
        float d_b = 0.0f;
        if (has_b) {
            qs_b     = block_load<uint8_t, 64>(pb.qs + bib * (QK_K / 4));
            hmask_b  = block_load<uint8_t, 32>(pb.hmask + bib * (QK_K / 8));
            scales_b = block_load<uint8_t, 12>(pb.scales + bib * 12);
            d_b = (float) pb.d[bib];
        }

        simd<float, 16> scale_f_a = unpack_scales(scales_a, d_a);
        simd<float, 16> scale_f_b = unpack_scales(scales_b, d_b);

#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int     byte_base = 32 * (s / 4);
            const uint8_t shift     = (uint8_t) (2 * (s % 4));

            // 2 low bits from qs, high bit from hmask (bit s of the same 32 bytes);
            // value = (q & 3) + 4*bit - 4  (inverted hmask: subtract 4 when bit clear).
            // merge in the integer domain: q3 = (q & 3) | (bit << 2) in {0..7},
            // then a single convert + subtract yields q3 - 4 (one convert, not two)
            simd<uint16_t, 32> q3_a = convert<uint16_t>(
                    (qs_a.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3));
            q3_a |= convert<uint16_t>(
                    ((hmask_a >> simd<uint8_t, 32>((uint8_t) s)) & simd<uint8_t, 32>(1)) << simd<uint8_t, 32>(2));
            simd<uint16_t, 32> q3_b = convert<uint16_t>(
                    (qs_b.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3));
            q3_b |= convert<uint16_t>(
                    ((hmask_b >> simd<uint8_t, 32>((uint8_t) s)) & simd<uint8_t, 32>(1)) << simd<uint8_t, 32>(2));

            simd<float, 32> qf_a = convert<float>(q3_a) - 4.0f;
            simd<float, 32> qf_b = convert<float>(q3_b) - 4.0f;

            const float scale_a_lo = scale_f_a[2 * s + 0];
            const float scale_a_hi = scale_f_a[2 * s + 1];
            const float scale_b_lo = scale_f_b[2 * s + 0];
            const float scale_b_hi = scale_f_b[2 * s + 1];

            simd<float, 32> scale_vec_a = splat_lo_hi(scale_a_lo, scale_a_hi);
            simd<float, 32> scale_vec_b = splat_lo_hi(scale_b_lo, scale_b_hi);

            simd<float, 32> deq_a = qf_a * scale_vec_a;
            simd<float, 32> deq_b = qf_b * scale_vec_b;

#pragma unroll
            for (int c = 0; c < NC; ++c) {
                simd<float, 32> y_s = block_load<float, 32>(y_blk + c * y_stride + s * 32);
                acc_a[c] += y_s * deq_a;
                acc_b[c] += y_s * deq_b;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Q4_K, SOA reorder layout produced by reorder_qw_q4_k:
//   [qs: nb*(QK_K/2)] [scales: nb*K_SCALE_SIZE] [dm: nb*sizeof(half2)]
// with nb = nrows*num_blocks_per_row.
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q4_K> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    scales;
        const sycl::half * dm;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs     = (const uint8_t *) vx;
        const uint8_t * scales = qs + nb * (QK_K / 2);
        const sycl::half * dm  = (const sycl::half *) (scales + nb * K_SCALE_SIZE);
        return { qs, scales, dm };
    }

    // NC activation columns share one dequantization of the two weight blocks;
    // the activation chunks are read per column from y_blk + c * y_stride
    template <int NC>
    static ESIMD_INLINE void mac_pair_nc(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            const float * y_blk, int64_t y_stride,
            sycl::ext::intel::esimd::simd<float, 32> (&acc_a)[NC],
            sycl::ext::intel::esimd::simd<float, 32> (&acc_b)[NC]) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> qs_a     = block_load<uint8_t, 128>(pa.qs + bia * (QK_K / 2));
        simd<uint8_t, 128> qs_b     = 0;
        simd<uint8_t, 12>  scales_a = block_load<uint8_t, 12>(pa.scales + bia * K_SCALE_SIZE);
        simd<uint8_t, 12>  scales_b = 0;

        const float dall_a = (float) pa.dm[bia * 2 + 0];
        const float dmin_a = (float) pa.dm[bia * 2 + 1];
        float dall_b = 0.0f;
        float dmin_b = 0.0f;
        if (has_b) {
            qs_b     = block_load<uint8_t, 128>(pb.qs + bib * (QK_K / 2));
            scales_b = block_load<uint8_t, 12>(pb.scales + bib * K_SCALE_SIZE);
            dall_b = (float) pb.dm[bib * 2 + 0];
            dmin_b = (float) pb.dm[bib * 2 + 1];
        }

        simd<float, 8> scale_f_a, min_f_a, scale_f_b, min_f_b;
        unpack_scale_min_k4(scales_a, dall_a, dmin_a, scale_f_a, min_f_a);
        unpack_scale_min_k4(scales_b, dall_b, dmin_b, scale_f_b, min_f_b);

        simd<uint8_t, 128> qs_lo_a = qs_a & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_a = qs_a >> simd<uint8_t, 128>(4);
        simd<uint8_t, 128> qs_lo_b = qs_b & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_b = qs_b >> simd<uint8_t, 128>(4);

#pragma unroll
        for (int sb = 0; sb < 8; sb += 2) {
            const int q_offset = sb * 16;

            const float scale_a_lo = scale_f_a[sb];
            const float scale_a_hi = scale_f_a[sb + 1];
            const float min_a_lo   = min_f_a[sb];
            const float min_a_hi   = min_f_a[sb + 1];
            const float scale_b_lo = scale_f_b[sb];
            const float scale_b_hi = scale_f_b[sb + 1];
            const float min_b_lo   = min_f_b[sb];
            const float min_b_hi   = min_f_b[sb + 1];

            simd<uint8_t, 32> qa_lo = qs_lo_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qa_hi = qs_hi_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_lo = qs_lo_b.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_hi = qs_hi_b.select<32, 1>(q_offset);

            simd<float, 32> deq_a_lo = convert<float>(qa_lo) * scale_a_lo + min_a_lo;
            simd<float, 32> deq_a_hi = convert<float>(qa_hi) * scale_a_hi + min_a_hi;
            simd<float, 32> deq_b_lo = convert<float>(qb_lo) * scale_b_lo + min_b_lo;
            simd<float, 32> deq_b_hi = convert<float>(qb_hi) * scale_b_hi + min_b_hi;

#pragma unroll
            for (int c = 0; c < NC; ++c) {
                simd<float, 32> y_lo = block_load<float, 32>(y_blk + c * y_stride + sb * 32);
                simd<float, 32> y_hi = block_load<float, 32>(y_blk + c * y_stride + (sb + 1) * 32);
                acc_a[c] += y_lo * deq_a_lo;
                acc_b[c] += y_lo * deq_b_lo;
                acc_a[c] += y_hi * deq_a_hi;
                acc_b[c] += y_hi * deq_b_hi;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Q5_K, SOA reorder layout produced by reorder_qw_q5_k:
//   [qs: nb*(QK_K/2)] [qh: nb*(QK_K/8)] [scales: nb*K_SCALE_SIZE] [dm: nb*sizeof(half2)]
// with nb = nrows*num_blocks_per_row.
//
// Identical to Q4_K except each 4-bit quant gains a 5th (high) bit from qh:
// output chunk c (0..7) adds 16 when bit c of qh[l] is set, where qh[l] indexes
// the same 32 bytes for every chunk (matches dequantize_row_q5_K).
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q5_K> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    qh;
        const uint8_t *    scales;
        const sycl::half * dm;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs     = (const uint8_t *) vx;
        const uint8_t * qh     = qs + nb * (QK_K / 2);
        const uint8_t * scales = qh + nb * (QK_K / 8);
        const sycl::half * dm  = (const sycl::half *) (scales + nb * K_SCALE_SIZE);
        return { qs, qh, scales, dm };
    }

    // extract bit `bit` (0..7) of each lane and move it to bit position 4,
    // e.g. for the 4-bit base quant's 5th (high) bit. `bit` is always a
    // compile-time-known unrolled loop constant at call sites, so this folds
    // to a single mask (bit==4), mask+left-shift (bit<4), or mask+right-shift
    // (bit>4) instead of the shift+mask+shift a naive `(qh>>bit & 1) << 4` emits.
    static ESIMD_INLINE sycl::ext::intel::esimd::simd<uint16_t, 32> extract_bit_to_pos4(
            sycl::ext::intel::esimd::simd<uint8_t, 32> qh, int bit) {
        using namespace sycl::ext::intel::esimd;
        simd<uint16_t, 32> masked = convert<uint16_t>(qh & simd<uint8_t, 32>((uint8_t) (1u << bit)));
        if (bit < 4) {
            return masked << simd<uint16_t, 32>((uint16_t) (4 - bit));
        } else if (bit > 4) {
            return masked >> simd<uint16_t, 32>((uint16_t) (bit - 4));
        }
        return masked;
    }

    // NC activation columns share one dequantization of the two weight blocks
    template <int NC>
    static ESIMD_INLINE void mac_pair_nc(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            const float * y_blk, int64_t y_stride,
            sycl::ext::intel::esimd::simd<float, 32> (&acc_a)[NC],
            sycl::ext::intel::esimd::simd<float, 32> (&acc_b)[NC]) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> qs_a     = block_load<uint8_t, 128>(pa.qs + bia * (QK_K / 2));
        simd<uint8_t, 128> qs_b     = 0;
        simd<uint8_t, 32>  qh_a     = block_load<uint8_t, 32>(pa.qh + bia * (QK_K / 8));
        simd<uint8_t, 32>  qh_b     = 0;
        simd<uint8_t, 12>  scales_a = block_load<uint8_t, 12>(pa.scales + bia * K_SCALE_SIZE);
        simd<uint8_t, 12>  scales_b = 0;

        const float dall_a = (float) pa.dm[bia * 2 + 0];
        const float dmin_a = (float) pa.dm[bia * 2 + 1];
        float dall_b = 0.0f;
        float dmin_b = 0.0f;
        if (has_b) {
            qs_b     = block_load<uint8_t, 128>(pb.qs + bib * (QK_K / 2));
            qh_b     = block_load<uint8_t, 32>(pb.qh + bib * (QK_K / 8));
            scales_b = block_load<uint8_t, 12>(pb.scales + bib * K_SCALE_SIZE);
            dall_b = (float) pb.dm[bib * 2 + 0];
            dmin_b = (float) pb.dm[bib * 2 + 1];
        }

        simd<float, 8> scale_f_a, min_f_a, scale_f_b, min_f_b;
        unpack_scale_min_k4(scales_a, dall_a, dmin_a, scale_f_a, min_f_a);
        unpack_scale_min_k4(scales_b, dall_b, dmin_b, scale_f_b, min_f_b);

        simd<uint8_t, 128> qs_lo_a = qs_a & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_a = qs_a >> simd<uint8_t, 128>(4);
        simd<uint8_t, 128> qs_lo_b = qs_b & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_b = qs_b >> simd<uint8_t, 128>(4);

#pragma unroll
        for (int sb = 0; sb < 8; sb += 2) {
            const int q_offset = sb * 16;

            const float scale_a_lo = scale_f_a[sb];
            const float scale_a_hi = scale_f_a[sb + 1];
            const float min_a_lo   = min_f_a[sb];
            const float min_a_hi   = min_f_a[sb + 1];
            const float scale_b_lo = scale_f_b[sb];
            const float scale_b_hi = scale_f_b[sb + 1];
            const float min_b_lo   = min_f_b[sb];
            const float min_b_hi   = min_f_b[sb + 1];

            simd<uint8_t, 32> qa_lo_u8 = qs_lo_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qa_hi_u8 = qs_hi_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_lo_u8 = qs_lo_b.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_hi_u8 = qs_hi_b.select<32, 1>(q_offset);
            simd<uint16_t, 32> qa_lo = convert<uint16_t>(qa_lo_u8);
            simd<uint16_t, 32> qa_hi = convert<uint16_t>(qa_hi_u8);
            simd<uint16_t, 32> qb_lo = convert<uint16_t>(qb_lo_u8);
            simd<uint16_t, 32> qb_hi = convert<uint16_t>(qb_hi_u8);

            // add the 5th bit: chunk sb uses qh bit sb, chunk sb+1 uses qh bit sb+1;
            // qh always indexes the same 32 bytes regardless of chunk
            qa_lo += extract_bit_to_pos4(qh_a, sb);
            qa_hi += extract_bit_to_pos4(qh_a, sb + 1);
            qb_lo += extract_bit_to_pos4(qh_b, sb);
            qb_hi += extract_bit_to_pos4(qh_b, sb + 1);

            simd<float, 32> deq_a_lo = convert<float>(qa_lo) * scale_a_lo + min_a_lo;
            simd<float, 32> deq_a_hi = convert<float>(qa_hi) * scale_a_hi + min_a_hi;
            simd<float, 32> deq_b_lo = convert<float>(qb_lo) * scale_b_lo + min_b_lo;
            simd<float, 32> deq_b_hi = convert<float>(qb_hi) * scale_b_hi + min_b_hi;

#pragma unroll
            for (int c = 0; c < NC; ++c) {
                simd<float, 32> y_lo = block_load<float, 32>(y_blk + c * y_stride + sb * 32);
                simd<float, 32> y_hi = block_load<float, 32>(y_blk + c * y_stride + (sb + 1) * 32);
                acc_a[c] += y_lo * deq_a_lo;
                acc_b[c] += y_lo * deq_b_lo;
                acc_a[c] += y_hi * deq_a_hi;
                acc_b[c] += y_hi * deq_b_hi;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Q6_K, SOA reorder layout:
//   [ql: nb*(QK_K/2)] [qh: nb*(QK_K/4)] [scales(int8): nb*(QK_K/16)] [d: nb*half]
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q6_K> {
    struct ptrs {
        const uint8_t *    ql;
        const uint8_t *    qh;
        const int8_t *     scales;
        const sycl::half * d;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t *    ql     = (const uint8_t *) vx;
        const uint8_t *    qh     = ql + nb * (QK_K / 2);
        const int8_t *     scales = (const int8_t *) (qh + nb * (QK_K / 4));
        const sycl::half * d      = (const sycl::half *) (scales + nb * (QK_K / 16));
        return { ql, qh, scales, d };
    }

    // NC activation columns share one dequantization of the two weight blocks
    template <int NC>
    static ESIMD_INLINE void mac_pair_nc(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            const float * y_blk, int64_t y_stride,
            sycl::ext::intel::esimd::simd<float, 32> (&acc_a)[NC],
            sycl::ext::intel::esimd::simd<float, 32> (&acc_b)[NC]) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> ql_a     = block_load<uint8_t, 128>(pa.ql + bia * (QK_K / 2));
        simd<uint8_t, 128> ql_b     = 0;
        simd<uint8_t, 64>  qh_a     = block_load<uint8_t, 64>(pa.qh + bia * (QK_K / 4));
        simd<uint8_t, 64>  qh_b     = 0;
        simd<int8_t, 16>   scales_a = block_load<int8_t, 16>(pa.scales + bia * (QK_K / 16));
        simd<int8_t, 16>   scales_b = 0;

        const float d_a = (float) pa.d[bia];
        float d_b = 0.0f;
        if (has_b) {
            ql_b     = block_load<uint8_t, 128>(pb.ql + bib * (QK_K / 2));
            qh_b     = block_load<uint8_t, 64>(pb.qh + bib * (QK_K / 4));
            scales_b = block_load<int8_t, 16>(pb.scales + bib * (QK_K / 16));
            d_b = (float) pb.d[bib];
        }

        simd<float, 16> sc_a = convert<float>(scales_a);
        simd<float, 16> sc_b = convert<float>(scales_b);

#pragma unroll
        for (int im = 0; im < 2; ++im) {
            simd<uint8_t, 32> ql_lo_a   = ql_a.select<32, 1>(64 * im);
            simd<uint8_t, 32> ql_hi_a   = ql_a.select<32, 1>(64 * im + 32);
            simd<uint8_t, 32> qh_bits_a = qh_a.select<32, 1>(32 * im);
            simd<uint8_t, 32> ql_lo_b   = ql_b.select<32, 1>(64 * im);
            simd<uint8_t, 32> ql_hi_b   = ql_b.select<32, 1>(64 * im + 32);
            simd<uint8_t, 32> qh_bits_b = qh_b.select<32, 1>(32 * im);

            // reconstruct each 32-wide 6-bit group (matches dequantize_row_q6_K)
#pragma unroll
            for (int g = 0; g < 4; ++g) {

                const float scale_a_lo = sc_a[8 * im + 2 * g + 0] * d_a;
                const float scale_a_hi = sc_a[8 * im + 2 * g + 1] * d_a;
                const float scale_b_lo = sc_b[8 * im + 2 * g + 0] * d_b;
                const float scale_b_hi = sc_b[8 * im + 2 * g + 1] * d_b;

                simd<float, 32> scale_vec_a = splat_lo_hi(scale_a_lo, scale_a_hi);
                simd<float, 32> scale_vec_b = splat_lo_hi(scale_b_lo, scale_b_hi);

                simd<uint8_t, 32> qa;
                simd<uint8_t, 32> qb;
                switch (g) {
                    case 0:
                        qa = (ql_lo_a & simd<uint8_t, 32>(0x0F)) | ((qh_bits_a & simd<uint8_t, 32>(0x03)) << simd<uint8_t, 32>(4));
                        qb = (ql_lo_b & simd<uint8_t, 32>(0x0F)) | ((qh_bits_b & simd<uint8_t, 32>(0x03)) << simd<uint8_t, 32>(4));
                        break;
                    case 1:
                        qa = (ql_hi_a & simd<uint8_t, 32>(0x0F)) | ((qh_bits_a & simd<uint8_t, 32>(0x0C)) << simd<uint8_t, 32>(2));
                        qb = (ql_hi_b & simd<uint8_t, 32>(0x0F)) | ((qh_bits_b & simd<uint8_t, 32>(0x0C)) << simd<uint8_t, 32>(2));
                        break;
                    case 2:
                        qa = (ql_lo_a >> simd<uint8_t, 32>(4)) | (qh_bits_a & simd<uint8_t, 32>(0x30));
                        qb = (ql_lo_b >> simd<uint8_t, 32>(4)) | (qh_bits_b & simd<uint8_t, 32>(0x30));
                        break;
                    default:
                        qa = (ql_hi_a >> simd<uint8_t, 32>(4)) | ((qh_bits_a & simd<uint8_t, 32>(0xC0)) >> simd<uint8_t, 32>(2));
                        qb = (ql_hi_b >> simd<uint8_t, 32>(4)) | ((qh_bits_b & simd<uint8_t, 32>(0xC0)) >> simd<uint8_t, 32>(2));
                        break;
                }

                simd<float, 32> deq_a = (convert<float>(qa) - 32.0f) * scale_vec_a;
                simd<float, 32> deq_b = (convert<float>(qb) - 32.0f) * scale_vec_b;

#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    simd<float, 32> y_g = block_load<float, 32>(y_blk + c * y_stride + 32 * (4 * im + g));
                    acc_a[c] += y_g * deq_a;
                    acc_b[c] += y_g * deq_b;
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// IQ4_XS, SOA reorder layout produced by reorder_qw_iq4_xs:
//   [qs: nb*(QK_K/2)] [scales_l: nb*(QK_K/64)] [scales_h: nb*uint16] [d: nb*half]
// with nb = nrows*num_blocks_per_row.
//
// Output chunk s (0..7) is sub-block s: the low nibbles of qs[16s..16s+15] give its
// first 16 values, the high nibbles the last 16, each an index into the 16-entry
// kvalues_iq4nl codebook. Its scale is d * (ls - 32) with the 6-bit ls taken from
// nibble s%2 of scales_l[s/2] (low 4 bits) and bits 2s..2s+1 of scales_h (high 2).
//
// The codebook is evaluated, not looked up: a register-indirect gather (iselect) per
// lane made the kernel 2.6x slower than MMVQ, while this degree-4 polynomial rounded
// to nearest reproduces all 16 entries exactly in float32 (worst error 0.476).
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_IQ4_XS> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    scales_l;
        const uint16_t *   scales_h;
        const sycl::half * d;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t *    qs       = (const uint8_t *) vx;
        const uint8_t *    scales_l = qs + nb * (QK_K / 2);
        const uint16_t *   scales_h = (const uint16_t *) (scales_l + nb * (QK_K / 64));
        const sycl::half * d        = (const sycl::half *) (scales_h + nb);
        return { qs, scales_l, scales_h, d };
    }

    // the eight sub-block scales d * (ls - 32) of one block
    static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 8> unpack_scales(const ptrs & p, size_t bi) {
        using namespace sycl::ext::intel::esimd;
        simd<uint8_t, 4>   sl8 = block_load<uint8_t, 4>(p.scales_l + bi * (QK_K / 64));
        simd<uint16_t, 4>  sl  = convert<uint16_t>(sl8);
        simd<uint16_t, 8>  lo;
        lo.select<4, 2>(0) = sl & simd<uint16_t, 4>(0x0F);
        lo.select<4, 2>(1) = sl >> simd<uint16_t, 4>(4);
        simd<uint16_t, 8>  hi = (simd<uint16_t, 8>(p.scales_h[bi]) >> simd<uint16_t, 8>(0, 2)) & simd<uint16_t, 8>(3);
        simd<uint16_t, 8>  ls = lo | (hi << simd<uint16_t, 8>(4));
        return (convert<float>(ls) - 32.0f) * (float) p.d[bi];
    }

    // kvalues_iq4nl[q] for q = 0..15
    static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 32> codebook(
            sycl::ext::intel::esimd::simd<float, 32> q) {
        using namespace sycl::ext::intel::esimd;
        simd<float, 32> v = q * 0.00132472022f + 0.0412168242f;
        v = v * q - 1.50018358f;
        v = v * q + 24.7432461f;
        v = v * q - 127.043602f;
        return rnde<float>(v);
    }

    // NC activation columns share one dequantization of the two weight blocks
    template <int NC>
    static ESIMD_INLINE void mac_pair_nc(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            const float * y_blk, int64_t y_stride,
            sycl::ext::intel::esimd::simd<float, 32> (&acc_a)[NC],
            sycl::ext::intel::esimd::simd<float, 32> (&acc_b)[NC]) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> qs_a    = block_load<uint8_t, 128>(pa.qs + bia * (QK_K / 2));
        simd<uint8_t, 128> qs_b    = 0;
        simd<float, 8>     scale_a = unpack_scales(pa, bia);
        simd<float, 8>     scale_b = 0.0f;
        if (has_b) {
            qs_b    = block_load<uint8_t, 128>(pb.qs + bib * (QK_K / 2));
            scale_b = unpack_scales(pb, bib);
        }

        simd<uint8_t, 128> qs_lo_a = qs_a & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_a = qs_a >> simd<uint8_t, 128>(4);
        simd<uint8_t, 128> qs_lo_b = qs_b & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_b = qs_b >> simd<uint8_t, 128>(4);

#pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            simd<float, 32> q_a;
            simd<float, 32> q_b;
            q_a.select<16, 1>(0)  = convert<float>(simd<uint8_t, 16>(qs_lo_a.select<16, 1>(16 * sb)));
            q_a.select<16, 1>(16) = convert<float>(simd<uint8_t, 16>(qs_hi_a.select<16, 1>(16 * sb)));
            q_b.select<16, 1>(0)  = convert<float>(simd<uint8_t, 16>(qs_lo_b.select<16, 1>(16 * sb)));
            q_b.select<16, 1>(16) = convert<float>(simd<uint8_t, 16>(qs_hi_b.select<16, 1>(16 * sb)));

            const float d_a = scale_a[sb];
            const float d_b = scale_b[sb];
            simd<float, 32> deq_a = codebook(q_a) * d_a;
            simd<float, 32> deq_b = codebook(q_b) * d_b;

#pragma unroll
            for (int c = 0; c < NC; ++c) {
                simd<float, 32> y_s = block_load<float, 32>(y_blk + c * y_stride + sb * 32);
                acc_a[c] += y_s * deq_a;
                acc_b[c] += y_s * deq_b;
            }
        }
    }
};

// the 32 magnitudes of one sub-block for each of two rows (IQ3_S / IQ3_XXS): 8 grid entries
// of 4 bytes per row, gathered by index in one 16-lane gather
static ESIMD_INLINE void grid_values_pair(
        const uint32_t * grid,
        sycl::ext::intel::esimd::simd<uint32_t, 8> idx_a, sycl::ext::intel::esimd::simd<uint32_t, 8> idx_b,
        sycl::ext::intel::esimd::simd<float, 32> & va, sycl::ext::intel::esimd::simd<float, 32> & vb) {
    using namespace sycl::ext::intel::esimd;
    simd<uint32_t, 16> idx;
    idx.select<8, 1>(0) = idx_a;
    idx.select<8, 1>(8) = idx_b;
    simd<uint32_t, 16> g = gather<uint32_t, 16>(grid, idx * (uint32_t) sizeof(uint32_t));
    simd<uint8_t, 64>  m = g.bit_cast_view<uint8_t>().read();
    va = convert<float>(simd<uint8_t, 32>(m.select<32, 1>(0)));
    vb = convert<float>(simd<uint8_t, 32>(m.select<32, 1>(32)));
}

// negate lane v where bit v of sw is set
static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 32> apply_signs(
        sycl::ext::intel::esimd::simd<float, 32> v, uint32_t sw) {
    using namespace sycl::ext::intel::esimd;
    simd<uint32_t, 32> shift = 31 - simd<uint32_t, 32>(0, 1);
    simd<uint32_t, 32> neg   = (simd<uint32_t, 32>(sw) << shift) & 0x80000000u;
    simd<uint32_t, 32> bits  = v.bit_cast_view<uint32_t>().read() ^ neg;
    return bits.bit_cast_view<float>().read();
}

// ---------------------------------------------------------------------------
// IQ3_S, SOA reorder layout produced by reorder_qw_iq3_s:
//   [qs: nb*(QK_K/4)] [qh: nb*(QK_K/32)] [signs: nb*(QK_K/8)] [scales(4 bytes) + d(half): nb*6]
// with nb = nrows*num_blocks_per_row.
//
// Output chunk s (0..7) is sub-block s: 8 grid indices qs[8s+j] | (bit j of qh[s]) << 8
// each select a 4-byte entry of the 512-entry iq3s_grid, giving the 32 magnitudes in
// order; bit v of the s-th 32-bit signs word negates value v; the scale is
// d * (1 + 2 * sc) with sc the nibble s%2 of scales[s/2]. The grid (an L1-resident
// 2 KiB table) is read with one 16-lane gather per sub-block for both rows.
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_IQ3_S> {
    struct ptrs {
        const uint8_t * qs;
        const uint8_t * qh;
        const uint8_t * signs;
        const uint8_t * sd;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs    = (const uint8_t *) vx;
        const uint8_t * qh    = qs + nb * (QK_K / 4);
        const uint8_t * signs = qh + nb * (QK_K / 32);
        const uint8_t * sd    = signs + nb * (QK_K / 8);
        return { qs, qh, signs, sd };
    }

    struct block {
        sycl::ext::intel::esimd::simd<uint8_t, 64> qs;
        sycl::ext::intel::esimd::simd<uint8_t, 8>  qh;
        sycl::ext::intel::esimd::simd<uint32_t, 8> signs;
        sycl::ext::intel::esimd::simd<float, 8>    scale;
    };

    static ESIMD_INLINE block load(const ptrs & p, size_t bi) {
        using namespace sycl::ext::intel::esimd;
        block b;
        b.qs    = block_load<uint8_t, 64>(p.qs + bi * (QK_K / 4));
        b.qh    = block_load<uint8_t, 8>(p.qh + bi * (QK_K / 32));
        b.signs = block_load<uint32_t, 8>((const uint32_t *) (p.signs + bi * (QK_K / 8)));
        const uint8_t * sd = p.sd + bi * (QK_K / 64 + sizeof(ggml_half));
        simd<uint16_t, 8> sc;
#pragma unroll
        for (int k = 0; k < QK_K / 64; ++k) {
            sc[2 * k + 0] = sd[k] & 0xF;
            sc[2 * k + 1] = sd[k] >> 4;
        }
        const float d = (float) *(const sycl::half *) (sd + QK_K / 64);
        b.scale = (convert<float>(sc) * 2.0f + 1.0f) * d;
        return b;
    }


    static ESIMD_INLINE sycl::ext::intel::esimd::simd<uint32_t, 8> indices(block & b, int s) {
        using namespace sycl::ext::intel::esimd;
        simd<uint32_t, 8> idx = convert<uint32_t>(simd<uint8_t, 8>(b.qs.select<8, 1>(8 * s)));
        const uint32_t    qh  = b.qh[s];
        return idx | (((simd<uint32_t, 8>(qh) >> simd<uint32_t, 8>(0, 1)) & 1) << 8);
    }

    // NC activation columns share one dequantization of the two weight blocks
    template <int NC>
    static ESIMD_INLINE void mac_pair_nc(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            const float * y_blk, int64_t y_stride,
            sycl::ext::intel::esimd::simd<float, 32> (&acc_a)[NC],
            sycl::ext::intel::esimd::simd<float, 32> (&acc_b)[NC]) {
        using namespace sycl::ext::intel::esimd;

        block ba = load(pa, bia);
        block bb;
        bb.qs    = 0;
        bb.qh    = 0;
        bb.signs = 0;
        bb.scale = 0.0f;
        if (has_b) {
            bb = load(pb, bib);
        }

#pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            simd<float, 32> mag_a;
            simd<float, 32> mag_b;
            grid_values_pair(iq3s_grid, indices(ba, sb), indices(bb, sb), mag_a, mag_b);
            simd<float, 32> deq_a = apply_signs(mag_a * (float) ba.scale[sb], ba.signs[sb]);
            simd<float, 32> deq_b = apply_signs(mag_b * (float) bb.scale[sb], bb.signs[sb]);

#pragma unroll
            for (int c = 0; c < NC; ++c) {
                simd<float, 32> y_s = block_load<float, 32>(y_blk + c * y_stride + sb * 32);
                acc_a[c] += y_s * deq_a;
                acc_b[c] += y_s * deq_b;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// IQ3_XXS, SOA reorder layout produced by reorder_qw_iq3_xxs:
//   [qs: nb*(3*QK_K/8)] [d: nb*half], with nb = nrows*num_blocks_per_row.
//
// Per block qs holds 64 grid index bytes and then 8 32-bit words, one per sub-block
// s (0..7): its 8 indices qs[8s+j] select 4-byte entries of the 256-entry
// iq3xxs_grid; word s carries the scale d * (0.5 + (w >> 28)) * 0.5 and four 7-bit
// sign groups (w >> 7g) & 127, each completed to 8 bits by its parity bit (the
// ksigns_iq2xs table), which negate values 8g..8g+7.
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_IQ3_XXS> {
    struct ptrs {
        const uint8_t *    qs;
        const sycl::half * d;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t *    qs = (const uint8_t *) vx;
        const sycl::half * d  = (const sycl::half *) (qs + nb * (3 * QK_K / 8));
        return { qs, d };
    }

    struct block {
        sycl::ext::intel::esimd::simd<uint8_t, 64> qs;
        sycl::ext::intel::esimd::simd<uint32_t, 8> signs;
        sycl::ext::intel::esimd::simd<float, 8>    scale;
    };

    static ESIMD_INLINE block load(const ptrs & p, size_t bi) {
        using namespace sycl::ext::intel::esimd;
        block b;
        const uint8_t * qs = p.qs + bi * (3 * QK_K / 8);
        b.qs = block_load<uint8_t, 64>(qs);
        simd<uint32_t, 8> w = block_load<uint32_t, 8>((const uint32_t *) (qs + QK_K / 4));
        b.signs = 0;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            simd<uint32_t, 8> s7 = (w >> (7 * g)) & 127;
            b.signs |= (s7 | ((cbit(s7) & 1) << 7)) << (8 * g);
        }
        b.scale = (convert<float>(simd<uint32_t, 8>(w >> 28)) + 0.5f) * (0.5f * (float) p.d[bi]);
        return b;
    }


    static ESIMD_INLINE sycl::ext::intel::esimd::simd<uint32_t, 8> indices(block & b, int s) {
        using namespace sycl::ext::intel::esimd;
        return convert<uint32_t>(simd<uint8_t, 8>(b.qs.select<8, 1>(8 * s)));
    }

    // NC activation columns share one dequantization of the two weight blocks
    template <int NC>
    static ESIMD_INLINE void mac_pair_nc(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            const float * y_blk, int64_t y_stride,
            sycl::ext::intel::esimd::simd<float, 32> (&acc_a)[NC],
            sycl::ext::intel::esimd::simd<float, 32> (&acc_b)[NC]) {
        using namespace sycl::ext::intel::esimd;

        block ba = load(pa, bia);
        block bb;
        bb.qs    = 0;
        bb.signs = 0;
        bb.scale = 0.0f;
        if (has_b) {
            bb = load(pb, bib);
        }

#pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            simd<float, 32> mag_a;
            simd<float, 32> mag_b;
            grid_values_pair(iq3xxs_grid, indices(ba, sb), indices(bb, sb), mag_a, mag_b);
            simd<float, 32> deq_a = apply_signs(mag_a * (float) ba.scale[sb], ba.signs[sb]);
            simd<float, 32> deq_b = apply_signs(mag_b * (float) bb.scale[sb], bb.signs[sb]);

#pragma unroll
            for (int c = 0; c < NC; ++c) {
                simd<float, 32> y_s = block_load<float, 32>(y_blk + c * y_stride + sb * 32);
                acc_a[c] += y_s * deq_a;
                acc_b[c] += y_s * deq_b;
            }
        }
    }
};

} // namespace ggml_sycl_esimd

#endif // GGML_SYCL_ESIMD_HPP
