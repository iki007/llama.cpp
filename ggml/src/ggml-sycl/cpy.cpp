#include "cpy.hpp"

#include <float.h>
#include <vector>

#include "convert.hpp"
#include "dequantize.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml.h"

static void cpy_1_f32_f32(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    float *       dsti = (float *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f32_f16(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    sycl::half *  dsti = (sycl::half *) cdsti;

    *dsti = sycl::vec<float, 1>(*xi).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

static void cpy_1_f16_f16(const char * cxi, char * cdsti) {
    const sycl::half * xi   = (const sycl::half *) cxi;
    sycl::half *       dsti = (sycl::half *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f16_f32(const char * cxi, char * cdsti) {
    const sycl::half * xi   = (const sycl::half *) cxi;
    float *            dsti = (float *) cdsti;

    *dsti = *xi;
}

static void cpy_1_i16_i16(const char * cxi, char * cdsti) {
    const int16_t * xi   = (const int16_t *) cxi;
    int16_t *       dsti = (int16_t *) cdsti;

    *dsti = *xi;
}

static void cpy_1_i32_i32(const char * cxi, char * cdsti) {
    const int32_t * xi   = (const int32_t *) cxi;
    int32_t *       dsti = (int32_t *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f32_i32(const char * cxi, char * cdsti) {
    const float * xi     = (const float *) cxi;
    int32_t *     dsti   = (int32_t *) cdsti;

    *dsti = (int32_t) *xi;
}

static void cpy_1_i32_f32(const char * cxi, char * cdsti) {
    const int32_t * xi   = (const int32_t *) cxi;
    float *         dsti = (float *) cdsti;

    *dsti = (float) *xi;
}

static void cpy_1_f8_e4m3_f8_e4m3(const char * cxi, char * cdsti) {
    const ggml_fp8_e4m3_t * xi   = (const ggml_fp8_e4m3_t *) cxi;
    ggml_fp8_e4m3_t *       dsti = (ggml_fp8_e4m3_t *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f8_e4m3_f32(const char * cxi, char * cdsti) {
    const ggml_fp8_e4m3_t * xi   = (const ggml_fp8_e4m3_t *) cxi;
    float *                 dsti = (float *) cdsti;

    *dsti = ggml_sycl_cast<float>(*xi);
}

static void cpy_1_f32_f8_e4m3(const char * cxi, char * cdsti) {
    const float *         xi   = (const float *) cxi;
    ggml_fp8_e4m3_t *     dsti = (ggml_fp8_e4m3_t *) cdsti;

    *dsti = ggml_sycl_cast<ggml_fp8_e4m3_t>(*xi);
}

static void cpy_1_f8_e4m3_f16(const char * cxi, char * cdsti) {
    const ggml_fp8_e4m3_t * xi   = (const ggml_fp8_e4m3_t *) cxi;
    sycl::half *            dsti = (sycl::half *) cdsti;

    *dsti = ggml_sycl_cast<sycl::half>(*xi);
}

static void cpy_1_f16_f8_e4m3(const char * cxi, char * cdsti) {
    const sycl::half *    xi   = (const sycl::half *) cxi;
    ggml_fp8_e4m3_t *     dsti = (ggml_fp8_e4m3_t *) cdsti;

    *dsti = ggml_sycl_cast<ggml_fp8_e4m3_t>(*xi);
}

#ifdef GGML_SYCL_HAS_BF16
static void cpy_1_f32_bf16(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(*xi);
}

static void cpy_1_bf16_f32(const char * cxi, char * cdsti) {
    const sycl::ext::oneapi::bfloat16 * xi   = (const sycl::ext::oneapi::bfloat16 *) cxi;
    float *            dsti = (float *) cdsti;

    *dsti = static_cast<float>(*xi);
}

static void cpy_1_bf16_bf16(const char * cxi, char * cdsti) {
    const sycl::ext::oneapi::bfloat16 * xi   = (const sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::ext::oneapi::bfloat16 *       dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f16_bf16(const char * cxi, char * cdsti) {
    const sycl::half * xi   = (const sycl::half *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(static_cast<float>(*xi));
}

static void cpy_1_bf16_f16(const char * cxi, char * cdsti) {
    const sycl::ext::oneapi::bfloat16 * xi   = (const sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::half *            dsti = (sycl::half *) cdsti;

    *dsti = sycl::half(static_cast<float>(*xi));
}

static void cpy_1_f8_e4m3_bf16(const char * cxi, char * cdsti) {
    const ggml_fp8_e4m3_t *         xi   = (const ggml_fp8_e4m3_t *) cxi;
    sycl::ext::oneapi::bfloat16 *   dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = ggml_sycl_cast<sycl::ext::oneapi::bfloat16>(*xi);
}

static void cpy_1_bf16_f8_e4m3(const char * cxi, char * cdsti) {
    const sycl::ext::oneapi::bfloat16 * xi   = (const sycl::ext::oneapi::bfloat16 *) cxi;
    ggml_fp8_e4m3_t *                   dsti = (ggml_fp8_e4m3_t *) cdsti;

    *dsti = ggml_sycl_cast<ggml_fp8_e4m3_t>(*xi);
}
#endif

template <cpy_kernel_t cpy_1>
static void cpy_f32_f16(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01, const int64_t ne02,
                        const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03, const int64_t ne10, const int64_t ne11,
                        const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12, const size_t nb13,
                        const sycl::nd_item<3> & item_ct1) {
    const int64_t tid    = (int64_t) item_ct1.get_local_range(2) * (int64_t) item_ct1.get_group(2) + (int64_t) item_ct1.get_local_id(2);
    const int64_t stride = (int64_t) item_ct1.get_local_range(2) * (int64_t) item_ct1.get_group_range(2);

    if (ne <= UINT32_MAX) {
        const uint32_t ne_u      = (uint32_t) ne;
        const uint32_t stride_u  = (uint32_t) stride;
        const uint32_t ne00_u    = (uint32_t) ne00;
        const uint32_t ne0_01_u  = (uint32_t) (ne00 * ne01);
        const uint32_t ne0_012_u = (uint32_t) (ne00 * ne01 * ne02);
        const uint32_t ne10_u    = (uint32_t) ne10;
        const uint32_t ne1_01_u  = (uint32_t) (ne10 * ne11);
        const uint32_t ne1_012_u = (uint32_t) (ne10 * ne11 * ne12);

        for (uint64_t i64 = (uint64_t) tid; i64 < (uint64_t) ne_u; i64 += (uint64_t) stride_u) {
            const uint32_t i        = (uint32_t) i64;
            const uint32_t i03      = i / ne0_012_u;
            const uint32_t r03      = i - i03 * ne0_012_u;
            const uint32_t i02      = r03 / ne0_01_u;
            const uint32_t r02      = r03 - i02 * ne0_01_u;
            const uint32_t i01      = r02 / ne00_u;
            const uint32_t i00      = r02 - i01 * ne00_u;
            const size_t   x_offset = (size_t) i00 * nb00 + (size_t) i01 * nb01 + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            const uint32_t i13        = i / ne1_012_u;
            const uint32_t r13        = i - i13 * ne1_012_u;
            const uint32_t i12        = r13 / ne1_01_u;
            const uint32_t r12        = r13 - i12 * ne1_01_u;
            const uint32_t i11        = r12 / ne10_u;
            const uint32_t i10        = r12 - i11 * ne10_u;
            const size_t   dst_offset = (size_t) i10 * nb10 + (size_t) i11 * nb11 + (size_t) i12 * nb12 + (size_t) i13 * nb13;

            cpy_1(cx + x_offset, cdst + dst_offset);
        }
        return;
    }

    const int64_t ne0_01  = ne00 * ne01;
    const int64_t ne0_012 = ne0_01 * ne02;
    const int64_t ne1_01  = ne10 * ne11;
    const int64_t ne1_012 = ne1_01 * ne12;

    for (int64_t i = tid; i < ne; i += stride) {
        const int64_t i03      = i / ne0_012;
        const int64_t r03      = i - i03 * ne0_012;
        const int64_t i02      = r03 / ne0_01;
        const int64_t r02      = r03 - i02 * ne0_01;
        const int64_t i01      = r02 / ne00;
        const int64_t i00      = r02 - i01 * ne00;
        const size_t  x_offset = (size_t) i00 * nb00 + (size_t) i01 * nb01 + (size_t) i02 * nb02 + (size_t) i03 * nb03;

        const int64_t i13        = i / ne1_012;
        const int64_t r13        = i - i13 * ne1_012;
        const int64_t i12        = r13 / ne1_01;
        const int64_t r12        = r13 - i12 * ne1_01;
        const int64_t i11        = r12 / ne10;
        const int64_t i10        = r12 - i11 * ne10;
        const size_t  dst_offset = (size_t) i10 * nb10 + (size_t) i11 * nb11 + (size_t) i12 * nb12 + (size_t) i13 * nb13;

        cpy_1(cx + x_offset, cdst + dst_offset);
    }
}

template <typename SrcT, typename DstT>
static inline DstT cpy_convert_scalar(SrcT v) {
    if constexpr (std::is_same_v<SrcT, DstT>) {
        return v;
    } else if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, sycl::half>) {
        return sycl::vec<float, 1>(v).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    } else {
        return static_cast<DstT>(v);
    }
}

// 32x32 SLM tiled transpose for strided 2D/3D permutes where src has unit stride along N (dim 1 or dim 2)
// and dst has unit stride along M (dim 0). Uses 32-bit tile indexing with 64-bit byte offsets.
template <typename SrcT, typename DstT>
static bool ggml_cpy_try_transpose_sycl(const char * cx, char * cdst, const int64_t ne,
                                        const int64_t ne00, const int64_t ne01, const int64_t ne02,
                                        const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                        const int64_t ne10, const int64_t ne11, const int64_t ne12,
                                        const size_t nb10, const size_t nb11, const size_t nb12, const size_t nb13,
                                        queue_ptr stream) {
    if (cx == cdst || ne <= 0 || ne00 != ne10 || ne01 != ne11 || ne02 != ne12) {
        return false;
    }
    if (nb10 != sizeof(DstT)) {
        return false;
    }
    if (((uintptr_t) cx % alignof(SrcT)) != 0 || ((uintptr_t) cdst % alignof(DstT)) != 0) {
        return false;
    }
    if ((nb00 % sizeof(SrcT)) != 0 || (nb01 % sizeof(SrcT)) != 0 || (nb02 % sizeof(SrcT)) != 0 || (nb03 % sizeof(SrcT)) != 0 ||
        (nb11 % sizeof(DstT)) != 0 || (nb12 % sizeof(DstT)) != 0 || (nb13 % sizeof(DstT)) != 0) {
        return false;
    }

    int64_t M = 0, N = 0, B0 = 0, B1 = 0;
    size_t src_sM = 0, src_sB0 = 0, src_sB1 = 0;
    size_t dst_sN = 0, dst_sB0 = 0, dst_sB1 = 0;

    const int64_t ne03 = ne / (ne00 * ne01 * ne02);
    if (nb01 == sizeof(SrcT) && nb00 > sizeof(SrcT) && ne00 >= 16 && ne01 >= 16) {
        // Case A: swap dim 0 <-> dim 1
        M       = ne00;
        N       = ne01;
        B0      = ne02;
        B1      = ne03;
        src_sM  = nb00 / sizeof(SrcT);
        src_sB0 = nb02 / sizeof(SrcT);
        src_sB1 = nb03 / sizeof(SrcT);
        dst_sN  = nb11 / sizeof(DstT);
        dst_sB0 = nb12 / sizeof(DstT);
        dst_sB1 = nb13 / sizeof(DstT);
    } else if (nb02 == sizeof(SrcT) && nb00 > sizeof(SrcT) && ne00 >= 16 && ne02 >= 16) {
        // Case B: swap dim 0 <-> dim 2 (e.g. Inkling 2 kq_b permute (2, 0, 1, 3))
        M       = ne00;
        N       = ne02;
        B0      = ne01;
        B1      = ne03;
        src_sM  = nb00 / sizeof(SrcT);
        src_sB0 = nb01 / sizeof(SrcT);
        src_sB1 = nb03 / sizeof(SrcT);
        dst_sN  = nb12 / sizeof(DstT);
        dst_sB0 = nb11 / sizeof(DstT);
        dst_sB1 = nb13 / sizeof(DstT);
    } else {
        return false;
    }

    constexpr int TILE = 32;
    constexpr int ROWS = 8;

    const int64_t tiles_m       = (M + TILE - 1) / TILE;
    const int64_t tiles_n       = (N + TILE - 1) / TILE;
    const int64_t tiles_mn      = tiles_m * tiles_n;
    const int64_t total_batches = B0 * B1;
    const int64_t total_tiles64 = tiles_mn * total_batches;
    if (total_tiles64 <= 0 || total_tiles64 > UINT32_MAX || B0 > UINT32_MAX || tiles_n > UINT32_MAX || tiles_mn > UINT32_MAX) {
        return false;
    }

    const uint32_t total_tiles = (uint32_t) total_tiles64;
    const uint32_t tiles_mn_u  = (uint32_t) tiles_mn;
    const uint32_t tiles_n_u   = (uint32_t) tiles_n;
    const uint32_t B0_u        = (uint32_t) B0;
    const int      num_wgs     = (int) std::min<int64_t>(total_tiles64, 65536);

    const SrcT * src = (const SrcT *) cx;
    DstT *       dst = (DstT *) cdst;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<DstT, 1> tile_mem(sycl::range<1>(TILE * (TILE + 1)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, ROWS, (size_t) num_wgs * TILE),
                              sycl::range<3>(1, ROWS, TILE)),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int      tx        = (int) item.get_local_id(2);
                const int      ty        = (int) item.get_local_id(1);
                const uint32_t wg_id     = (uint32_t) item.get_group(2);
                const uint32_t wg_stride = (uint32_t) item.get_group_range(2);

                for (uint64_t tile_idx64 = (uint64_t) wg_id; tile_idx64 < (uint64_t) total_tiles; tile_idx64 += (uint64_t) wg_stride) {
                    const uint32_t tile_idx  = (uint32_t) tile_idx64;
                    const uint32_t batch_idx = tile_idx / tiles_mn_u;
                    const uint32_t mn_idx    = tile_idx - batch_idx * tiles_mn_u;
                    const uint32_t tm        = mn_idx / tiles_n_u;
                    const uint32_t tn        = mn_idx - tm * tiles_n_u;
                    const uint32_t b1        = batch_idx / B0_u;
                    const uint32_t b0        = batch_idx - b1 * B0_u;

                    const int64_t m_base = (int64_t) tm * TILE;
                    const int64_t n_base = (int64_t) tn * TILE;

                    const SrcT * src_batch = src + (size_t) b0 * src_sB0 + (size_t) b1 * src_sB1;
                    DstT *       dst_batch = dst + (size_t) b0 * dst_sB0 + (size_t) b1 * dst_sB1;

                    // Coalesced load along N into padded SLM tile
                    const int64_t n_in = n_base + tx;
                    if (n_in < N) {
                        #pragma unroll
                        for (int k = 0; k < TILE; k += ROWS) {
                            const int     m_loc = ty + k;
                            const int64_t m_in  = m_base + m_loc;
                            if (m_in < M) {
                                tile_mem[m_loc * (TILE + 1) + tx] =
                                    cpy_convert_scalar<SrcT, DstT>(src_batch[(size_t) m_in * src_sM + (size_t) n_in]);
                            }
                        }
                    }

                    sycl::group_barrier(item.get_group());

                    // Coalesced store along M from padded SLM tile
                    const int64_t m_out = m_base + tx;
                    if (m_out < M) {
                        #pragma unroll
                        for (int k = 0; k < TILE; k += ROWS) {
                            const int     n_loc = ty + k;
                            const int64_t n_out = n_base + n_loc;
                            if (n_out < N) {
                                dst_batch[(size_t) n_out * dst_sN + (size_t) m_out] =
                                    tile_mem[tx * (TILE + 1) + n_loc];
                            }
                        }
                    }

                    sycl::group_barrier(item.get_group());
                }
            });
    });

    return true;
}

static void ggml_cpy_contig_f32_f16_sycl(const char * cx, char * cdst, const int64_t ne, queue_ptr stream) {
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    constexpr int BS = SYCL_CPY_BLOCK_SIZE;
    const bool aligned = (((uintptr_t) cx & 15) == 0) && (((uintptr_t) cdst & 7) == 0);
    if (aligned && ne >= 4) {
        const int64_t ne4        = ne / 4;
        const int     num_blocks = (int) std::min<int64_t>((ne4 + BS - 1) / BS, 65536);
        const sycl::float4 * src4 = (const sycl::float4 *) cx;
        sycl::half4 *        dst4 = (sycl::half4 *) cdst;
        const float *        src1 = (const float *) cx;
        sycl::half *         dst1 = (sycl::half *) cdst;
        stream->parallel_for(
            sycl::nd_range<1>((size_t) num_blocks * BS, BS),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int64_t tid    = (int64_t) item.get_global_linear_id();
                const int64_t stride = (int64_t) item.get_global_range(0);
                for (int64_t i = tid; i < ne4; i += stride) {
                    dst4[i] = src4[i].convert<sycl::half, sycl::rounding_mode::automatic>();
                }
                const int64_t tail = ne - ne4 * 4;
                if (tid < tail) {
                    const int64_t idx = ne4 * 4 + tid;
                    dst1[idx] = sycl::vec<float, 1>(src1[idx]).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
                }
            });
    } else {
        const int     num_blocks = (int) std::min<int64_t>((ne + BS - 1) / BS, 65536);
        const float * src1       = (const float *) cx;
        sycl::half *  dst1       = (sycl::half *) cdst;
        stream->parallel_for(
            sycl::nd_range<1>((size_t) num_blocks * BS, BS),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int64_t tid    = (int64_t) item.get_global_linear_id();
                const int64_t stride = (int64_t) item.get_global_range(0);
                for (int64_t i = tid; i < ne; i += stride) {
                    dst1[i] = sycl::vec<float, 1>(src1[i]).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
                }
            });
    }
}

static void ggml_cpy_contig_f16_f32_sycl(const char * cx, char * cdst, const int64_t ne, queue_ptr stream) {
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    constexpr int BS = SYCL_CPY_BLOCK_SIZE;
    const bool aligned = (((uintptr_t) cx & 7) == 0) && (((uintptr_t) cdst & 15) == 0);
    if (aligned && ne >= 4) {
        const int64_t ne4         = ne / 4;
        const int     num_blocks  = (int) std::min<int64_t>((ne4 + BS - 1) / BS, 65536);
        const sycl::half4 * src4  = (const sycl::half4 *) cx;
        sycl::float4 *      dst4  = (sycl::float4 *) cdst;
        const sycl::half *  src1  = (const sycl::half *) cx;
        float *             dst1  = (float *) cdst;
        stream->parallel_for(
            sycl::nd_range<1>((size_t) num_blocks * BS, BS),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int64_t tid    = (int64_t) item.get_global_linear_id();
                const int64_t stride = (int64_t) item.get_global_range(0);
                for (int64_t i = tid; i < ne4; i += stride) {
                    dst4[i] = src4[i].convert<float>();
                }
                const int64_t tail = ne - ne4 * 4;
                if (tid < tail) {
                    const int64_t idx = ne4 * 4 + tid;
                    dst1[idx] = (float) src1[idx];
                }
            });
    } else {
        const int          num_blocks = (int) std::min<int64_t>((ne + BS - 1) / BS, 65536);
        const sycl::half * src1       = (const sycl::half *) cx;
        float *            dst1       = (float *) cdst;
        stream->parallel_for(
            sycl::nd_range<1>((size_t) num_blocks * BS, BS),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int64_t tid    = (int64_t) item.get_global_linear_id();
                const int64_t stride = (int64_t) item.get_global_range(0);
                for (int64_t i = tid; i < ne; i += stride) {
                    dst1[i] = (float) src1[i];
                }
            });
    }
}


/* quantized type same copy */
template<typename T>
static void cpy_blck_q_q(const char * cxi, char * cdsti) {
    const T * xi = (const T *) cxi;
    T * dsti = (T *) cdsti;
    *dsti = *xi;
}


static void cpy_blck_q8_0_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *) (cdsti);

    for (int j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j)     = dq.x();
        *(cdstf + j + 1) = dq.y();
    }
}

static void cpy_blck_q2_0_f32(const char * cxi, char * cdsti) {
    const block_q2_0 * xi = (const block_q2_0 *) cxi;
    float * cdstf = (float *) cdsti;

    const float d = xi->d;

    for (int j = 0; j < QK2_0; ++j) {
        const int byte_index = j / 4;
        const int bit_offset = (j % 4) * 2;
        const int q = (xi->qs[byte_index] >> bit_offset) & 0x3;
        cdstf[j] = (float) (q - 1) * d;
    }
}



template <dequantize_kernel_t dequant, int qk> static void cpy_blck_q_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *) (cdsti);

    for (int j = 0; j < qk / 2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j)          = dq.x();
        *(cdstf + j + qk / 2) = dq.y();
    }
}


template <typename T, int qk>
static void cpy_q_q(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                      const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                      const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    if (i >= ne) {
        return;
    }

    const int i03         = i / (ne00 * ne01 * ne02);
    const int i02         = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01         = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00         = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const size_t x_offset = (size_t) (i00 / qk) * (size_t) nb00 + (size_t) i01 * (size_t) nb01 + (size_t) i02 * (size_t) nb02 + (size_t) i03 * (size_t) nb03;


    const int i13           = i / (ne10 * ne11 * ne12);
    const int i12           = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11           = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10           = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const size_t dst_offset = (size_t) (i10 / qk) * (size_t) nb10 + (size_t) i11 * (size_t) nb11 + (size_t) i12 * (size_t) nb12 + (size_t) i13 * (size_t) nb13;

    cpy_blck_q_q<T>(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static void cpy_f32_q(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                      const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                      const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    if (i >= ne) {
        return;
    }


    const int i03         = i / (ne00 * ne01 * ne02);
    const int i02         = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01         = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00         = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const size_t x_offset = (size_t) i00 * (size_t) nb00 + (size_t) i01 * (size_t) nb01 + (size_t) i02 * (size_t) nb02 + (size_t) i03 * (size_t) nb03;

    const int i13           = i / (ne10 * ne11 * ne12);
    const int i12           = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11           = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10           = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const size_t dst_offset = (size_t) (i10 / qk) * (size_t) nb10 + (size_t) i11 * (size_t) nb11 + (size_t) i12 * (size_t) nb12 + (size_t) i13 * (size_t) nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static void cpy_q_f32(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                      const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                      const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    if (i >= ne) {
        return;
    }

    const int i03         = i / (ne00 * ne01 * ne02);
    const int i02         = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01         = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00         = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const size_t x_offset = (size_t) (i00 / qk) * (size_t) nb00 + (size_t) i01 * (size_t) nb01 + (size_t) i02 * (size_t) nb02 + (size_t) i03 * (size_t) nb03;

    const int i13           = i / (ne10 * ne11 * ne12);
    const int i12           = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11           = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10           = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const size_t dst_offset = (size_t) i10 * (size_t) nb10 + (size_t) i11 * (size_t) nb11 + (size_t) i12 * (size_t) nb12 + (size_t) i13 * (size_t) nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

static void ggml_cpy_f16_f32_sycl(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01,
                                  const int64_t ne02, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                  const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11,
                                  const size_t nb12, const size_t nb13, queue_ptr stream) {
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    if (ggml_cpy_try_transpose_sycl<sycl::half, float>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                       ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream)) {
        return;
    }
    const int num_blocks = (int) std::min<int64_t>((ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE, 65536);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                       nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_f32_f32_sycl(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01,
                                  const int64_t ne02, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                  const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11,
                                  const size_t nb12, const size_t nb13, queue_ptr stream) {
    if (ggml_cpy_try_transpose_sycl<float, float>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                  ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream)) {
        return;
    }
    const int num_blocks = (int) std::min<int64_t>((ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE, 65536);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                       nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_f32_f16_sycl(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01,
                                  const int64_t ne02, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                  const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11,
                                  const size_t nb12, const size_t nb13, queue_ptr stream) {
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    if (ggml_cpy_try_transpose_sycl<float, sycl::half>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                       ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream)) {
        return;
    }
    const int num_blocks = (int) std::min<int64_t>((ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE, 65536);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                       nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_f32_i32_sycl(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01,
                                  const int64_t ne02, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                  const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11,
                                  const size_t nb12, const size_t nb13, queue_ptr stream) {
    const int num_blocks = (int) std::min<int64_t>((ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE, 65536);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                       nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_i32_f32_sycl(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01,
                                  const int64_t ne02, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                  const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11,
                                  const size_t nb12, const size_t nb13, queue_ptr stream) {
    const int num_blocks = (int) std::min<int64_t>((ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE, 65536);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_i32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                       nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_f8_e4m3_f8_e4m3_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                          const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                          const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                          const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f8_e4m3_f8_e4m3>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                                   nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_f8_e4m3_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                      const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                      const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                      const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f8_e4m3_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                               nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_f32_f8_e4m3_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                      const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                      const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                      const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_f8_e4m3>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                               nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_f8_e4m3_f16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                      const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                      const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                      const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f8_e4m3_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                               nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_f16_f8_e4m3_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                      const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                      const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                      const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f16_f8_e4m3>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                               nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_f32_q8_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    const int num_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q8_0_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    const int num_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q2_0_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    const int num_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_f32<cpy_blck_q2_0_f32, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_f32_q4_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q4_0_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_q4_1_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q4_1_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_q5_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    const int num_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q5_0_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    const int num_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_q5_1_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    const int num_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q5_1_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    const int num_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_mxfp4_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                    const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                    const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                    const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    const int num_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_f32<cpy_blck_q_f32<dequantize_mxfp4, QK_MXFP4>, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                             nb01, nb02, nb03, ne10, ne11, ne12,
                                                                             nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_f32_iq4_nl_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                     const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                     const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                     const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    const int num_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                   ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void cpy_blck_f16_q4_0(const char * cxi, char * cdsti) {
    const sycl::half * xi = (const sycl::half *) cxi;
    float xf[QK4_0];

    for (int j = 0; j < QK4_0; ++j) {
        xf[j] = (float) xi[j];
    }

    cpy_blck_f32_q4_0((const char *) xf, cdsti);
}

static void cpy_blck_f16_q4_1(const char * cxi, char * cdsti) {
    const sycl::half * xi = (const sycl::half *) cxi;
    float xf[QK4_1];

    for (int j = 0; j < QK4_1; ++j) {
        xf[j] = (float) xi[j];
    }

    cpy_blck_f32_q4_1((const char *) xf, cdsti);
}

static void cpy_blck_f16_q5_0(const char * cxi, char * cdsti) {
    const sycl::half * xi = (const sycl::half *) cxi;
    float xf[QK5_0];

    for (int j = 0; j < QK5_0; ++j) {
        xf[j] = (float) xi[j];
    }

    cpy_blck_f32_q5_0((const char *) xf, cdsti);
}

static void ggml_cpy_f16_q4_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_f16_q4_1_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_f16_q5_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    const int num_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static bool ggml_sycl_is_quantized_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static bool ggml_sycl_can_quantize_rows_sycl(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

template <typename SrcScalar>
static inline float ggml_sycl_src_to_f32(const SrcScalar & x) {
    return (float) x;
}

#ifdef GGML_SYCL_HAS_BF16
template <>
inline float ggml_sycl_src_to_f32<sycl::ext::oneapi::bfloat16>(const sycl::ext::oneapi::bfloat16 & x) {
    return static_cast<float>(x);
}

template <>
inline float ggml_sycl_src_to_f32<ggml_bf16_t>(const ggml_bf16_t & x) {
    union {
        uint32_t u32;
        float f32;
    } value;

    value.u32 = (uint32_t) x.bits << 16;
    return value.f32;
}
#endif

template <typename SrcScalar, cpy_kernel_t quantize_block, int qk>
static void ggml_sycl_quantize_rows_q(const char * cx, char * cdst, const int64_t ne,
                                      const int64_t ne00, const int64_t ne01, const int64_t ne02,
                                      const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                      const int64_t ne10, const int64_t ne11, const int64_t ne12,
                                      const size_t nb10, const size_t nb11, const size_t nb12, const size_t nb13,
                                      queue_ptr stream) {
    GGML_ASSERT(ne % qk == 0);
    GGML_ASSERT(ne00 % qk == 0);

    const int64_t total_blocks = ne / qk;
    constexpr int block_size = 256;
    const int64_t grid_size = ceil_div(total_blocks, (int64_t) block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size),
                        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        const int64_t block_idx = item_ct1.get_global_linear_id();
        if (block_idx >= total_blocks) {
            return;
        }

        const int64_t i = block_idx * qk;

        const int64_t i03 = i / (ne00 * ne01 * ne02);
        const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
        const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
        const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
        const size_t x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

        const int64_t i13 = i / (ne10 * ne11 * ne12);
        const int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
        const int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
        const int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
        const size_t dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

        float xf[qk];
        if (nb00 == sizeof(SrcScalar)) {
            const SrcScalar * src_row = (const SrcScalar *) (cx + x_offset);
            for (int j = 0; j < qk; ++j) {
                xf[j] = ggml_sycl_src_to_f32(src_row[j]);
            }
        } else {
            for (int j = 0; j < qk; ++j) {
                const SrcScalar * src_val = (const SrcScalar *) (cx + x_offset + j * nb00);
                xf[j] = ggml_sycl_src_to_f32(*src_val);
            }
        }

        quantize_block((const char *) xf, cdst + dst_offset);
    });
}

template <typename SrcScalar>
static void ggml_sycl_quantize_rows_sycl(const char * cx, char * cdst, const ggml_tensor * src0, const ggml_tensor * src1,
                                         const int64_t ne, const int64_t ne00, const int64_t ne01, const int64_t ne02,
                                         const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                         const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10,
                                         const size_t nb11, const size_t nb12, const size_t nb13, queue_ptr stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);

    switch (src1->type) {
        case GGML_TYPE_Q8_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            break;
        case GGML_TYPE_Q1_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            break;
        case GGML_TYPE_Q2_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            break;
        case GGML_TYPE_Q5_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            break;
        case GGML_TYPE_Q5_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            break;
        case GGML_TYPE_Q4_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            break;
        case GGML_TYPE_Q4_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                               nb01, nb02, nb03, ne10, ne11, ne12,
                                                                               nb10, nb11, nb12, nb13, stream);
            break;
        case GGML_TYPE_MXFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            break;
        case GGML_TYPE_NVFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            break;
        default:
            GGML_ABORT("unsupported quantized target type in sycl quantizer src1->type=%s\n",
                       ggml_type_name(src1->type));
    }
}

static void ggml_cpy_f16_f16_sycl(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01,
                                  const int64_t ne02, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                  const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11,
                                  const size_t nb12, const size_t nb13, queue_ptr stream) {
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    if (ggml_cpy_try_transpose_sycl<sycl::half, sycl::half>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                            ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream)) {
        return;
    }
    const int num_blocks = (int) std::min<int64_t>((ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE, 65536);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                       nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_i16_i16_sycl(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01,
                                  const int64_t ne02, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                  const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11,
                                  const size_t nb12, const size_t nb13, queue_ptr stream) {
    const int num_blocks = (int) std::min<int64_t>((ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE, 65536);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                       nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_i32_i32_sycl(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01,
                                  const int64_t ne02, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                  const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11,
                                  const size_t nb12, const size_t nb13, queue_ptr stream) {
    const int num_blocks = (int) std::min<int64_t>((ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE, 65536);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                       nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q8_0_q8_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    const int num_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q5_0_q5_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    const int num_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q5_1_q5_1(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    const int num_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q4_0_q4_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q4_1_q4_1(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q1_0_q1_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK1_0 == 0);
    const int num_blocks = ceil_div(ne / QK1_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_q<block_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q2_0_q2_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    const int num_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_mxfp4_mxfp4(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    const int num_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_q<block_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_nvfp4_nvfp4(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_NVFP4 == 0);
    const int num_blocks = ceil_div(ne / QK_NVFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q2_K_q2_K(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    const int num_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q2_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q3_K_q3_K(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    const int num_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q3_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q4_K_q4_K(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    const int num_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q5_K_q5_K(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    const int num_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q6_K_q6_K(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    const int num_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q6_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq2_xxs_iq2_xxs(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    const int num_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq2_xs_iq2_xs(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    const int num_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq2_s_iq2_s(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    const int num_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq3_xxs_iq3_xxs(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    const int num_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq3_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq1_s_iq1_s(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    const int num_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq1_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq1_m_iq1_m(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    const int num_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq1_m, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq4_nl_iq4_nl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    const int num_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq3_s_iq3_s(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    const int num_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq3_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq4_xs_iq4_xs(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    const int num_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq4_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

#ifdef GGML_SYCL_HAS_BF16
static void ggml_cpy_f32_bf16_sycl(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01,
                                  const int64_t ne02, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                  const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11,
                                  const size_t nb12, const size_t nb13, queue_ptr stream) {
    const int num_blocks = (int) std::min<int64_t>((ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE, 65536);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f32_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_bf16_f32_sycl(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01,
                                  const int64_t ne02, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                  const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11,
                                  const size_t nb12, const size_t nb13, queue_ptr stream) {
    const int num_blocks = (int) std::min<int64_t>((ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE, 65536);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_bf16_bf16_sycl(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01,
                                  const int64_t ne02, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                  const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11,
                                  const size_t nb12, const size_t nb13, queue_ptr stream) {
    const int num_blocks = (int) std::min<int64_t>((ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE, 65536);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                            nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_f16_bf16_sycl(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01,
                                  const int64_t ne02, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                  const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11,
                                  const size_t nb12, const size_t nb13, queue_ptr stream) {
    const int num_blocks = (int) std::min<int64_t>((ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE, 65536);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_bf16_f16_sycl(const char * cx, char * cdst, const int64_t ne, const int64_t ne00, const int64_t ne01,
                                  const int64_t ne02, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                  const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11,
                                  const size_t nb12, const size_t nb13, queue_ptr stream) {
    const int num_blocks = (int) std::min<int64_t>((ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE, 65536);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_f8_e4m3_bf16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                       const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                       const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                       const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f8_e4m3_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                             nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_bf16_f8_e4m3_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                       const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                       const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                       const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_f8_e4m3>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                             nb10, nb11, nb12, nb13, item_ct1);
        });
}
#endif

void ggml_sycl_cpy(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1) try {
    // Unlike other operators ggml_sycl_cpy takes 2 distinct tensors instead of a dst ggml_tensor and rely on its src field
    GGML_SYCL_DEBUG("ggml_sycl_cpy: src0->type=%s, src1->type=%s\n",
        ggml_type_name(src0->type), ggml_type_name(src1->type));
    scope_op_debug_print scope_dbg_print(__func__, src1, /*num_src=*/0, debug_get_tensor_str("\tsrc0", src0));
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    GGML_TENSOR_BINARY_OP_LOCALS01;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;
    const bool both_contig = ggml_is_contiguous(src0) && ggml_is_contiguous(src1);
    if ((src0->type == src1->type) && both_contig) {
        GGML_SYCL_DEBUG("%s: memcpy path\n", __func__);
        main_stream->memcpy(src1_ddc, src0_ddc, ggml_nbytes(src0));
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && both_contig && src0_ddc != src1_ddc) {
        ggml_cpy_contig_f32_f16_sycl(src0_ddc, src1_ddc, ne, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && both_contig && src0_ddc != src1_ddc) {
        ggml_cpy_contig_f16_f32_sycl(src0_ddc, src1_ddc, ne, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<float>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00, nb01,
                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<sycl::half>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00,
                                                 nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                 main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } else if (src0->type == GGML_TYPE_BF16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<ggml_bf16_t>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02,
                                                  nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                  nb12, nb13, main_stream);
#endif
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_f32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        GGML_ASSERT(ne <= INT32_MAX);
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        GGML_ASSERT(ne <= INT32_MAX);
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        GGML_ASSERT(ne <= INT32_MAX);
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f16_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f16_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f16_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_i32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q2_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                               nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_mxfp4_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                 nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_q8_0_q8_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_q5_0_q5_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_q5_1_q5_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_q4_0_q4_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_q4_1_q4_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q1_0 && src1->type == GGML_TYPE_Q1_0) {
        ggml_cpy_q1_0_q1_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_Q2_0) {
        ggml_cpy_q2_0_q2_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_MXFP4) {
        ggml_cpy_mxfp4_mxfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_NVFP4 && src1->type == GGML_TYPE_NVFP4) {
        ggml_cpy_nvfp4_nvfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q2_K && src1->type == GGML_TYPE_Q2_K) {
        ggml_cpy_q2_K_q2_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q3_K && src1->type == GGML_TYPE_Q3_K) {
        ggml_cpy_q3_K_q3_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_K && src1->type == GGML_TYPE_Q4_K) {
        ggml_cpy_q4_K_q4_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_K && src1->type == GGML_TYPE_Q5_K) {
        ggml_cpy_q5_K_q5_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_Q6_K) {
        ggml_cpy_q6_K_q6_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ2_XXS && src1->type == GGML_TYPE_IQ2_XXS) {
        ggml_cpy_iq2_xxs_iq2_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ2_XS && src1->type == GGML_TYPE_IQ2_XS) {
        ggml_cpy_iq2_xs_iq2_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ2_S && src1->type == GGML_TYPE_IQ2_S) {
        ggml_cpy_iq2_s_iq2_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ3_XXS && src1->type == GGML_TYPE_IQ3_XXS) {
        ggml_cpy_iq3_xxs_iq3_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ1_S && src1->type == GGML_TYPE_IQ1_S) {
        ggml_cpy_iq1_s_iq1_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ1_M && src1->type == GGML_TYPE_IQ1_M) {
        ggml_cpy_iq1_m_iq1_m(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_iq4_nl_iq4_nl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ3_S && src1->type == GGML_TYPE_IQ3_S) {
        ggml_cpy_iq3_s_iq3_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ4_XS && src1->type == GGML_TYPE_IQ4_XS) {
        ggml_cpy_iq4_xs_iq4_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f32_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_bf16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_bf16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_bf16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
#endif
    } else if (src0->type == GGML_TYPE_F8_E4M3 && src1->type == GGML_TYPE_F8_E4M3) {
        ggml_cpy_f8_e4m3_f8_e4m3_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                                      nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F8_E4M3 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f8_e4m3_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                                  nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F8_E4M3) {
        ggml_cpy_f32_f8_e4m3_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                                  nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F8_E4M3 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f8_e4m3_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                                  nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F8_E4M3) {
        ggml_cpy_f16_f8_e4m3_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                                  nb11, nb12, nb13, main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } else if (src0->type == GGML_TYPE_F8_E4M3 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f8_e4m3_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                                   nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F8_E4M3) {
        ggml_cpy_bf16_f8_e4m3_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                                   nb11, nb12, nb13, main_stream);
#endif
    } else {
        GGML_LOG_ERROR("%s: unsupported type combination (%s to %s)\n", __func__, ggml_type_name(src0->type),
                       ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_cpy(ctx, dst->src[0], dst);
}
