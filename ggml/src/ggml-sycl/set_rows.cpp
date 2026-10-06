#include "set_rows.hpp"
#include "cpy.hpp"
#include "fattn-xmx.hpp"

#include "ggml-quants.h"
#include "ggml-impl.h"

#include <climits>
#include <vector>

namespace utils {
template<typename T>
static constexpr bool is_arithmetic_v() {
    return std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#ifdef GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#endif
        ;
}
}

int ggml_sycl_qsa_mask_absorbs(const ggml_cgraph * graph, int i) {
    static const bool enabled = ggml_sycl_get_env("GGML_SYCL_QSA_MASK", 1) != 0;
    if (!enabled || !g_ggml_sycl_enable_fusion || i + 8 >= graph->n_nodes) {
        return 0;
    }
    const ggml_op ops[] = { GGML_OP_FILL, GGML_OP_REPEAT, GGML_OP_RESHAPE, GGML_OP_FILL, GGML_OP_REPEAT,
                            GGML_OP_RESHAPE, GGML_OP_SET_ROWS, GGML_OP_VIEW, GGML_OP_ADD };
    const int outputs[] = { i + 8 };
    if (!ggml_can_fuse_subgraph(graph, i, 9, ops, outputs, 1)) {
        return 0;
    }
    const auto n = graph->nodes + i;
    const ggml_tensor * idx = n[6]->src[1];
    const ggml_tensor * mask = n[8]->src[1];
    const ggml_tensor * dst = n[8];
    const int64_t nk = dst->ne[0], ns = n[0]->ne[0], nt = ggml_nrows(dst);
    if (!idx || !mask || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32) ||
        nk <= 0 || ns <= 0 || nt < 1 || nk > INT_MAX - ns || nt > INT_MAX ||
        idx->type != GGML_TYPE_I32 || idx->ne[0] != ns || idx->ne[1] != nt ||
        idx->ne[2] != 1 || idx->ne[3] != 1 || !ggml_is_contiguous(idx) ||
        mask->type != dst->type || !ggml_are_same_shape(mask, dst) || !ggml_is_contiguous(mask) ||
        !ggml_is_contiguous(dst) || dst->view_src || n[0]->view_src || n[3]->view_src || n[8]->src[0] != n[7]) {
        return 0;
    }
    for (int j = 0; j < 8; ++j) {
        if (n[j]->type != dst->type || n[j]->flags & GGML_TENSOR_FLAG_INPUT) {
            return 0;
        }
    }
    float fill, zero;
    memcpy(&fill, n[3]->op_params, sizeof(fill));
    memcpy(&zero, n[0]->op_params, sizeof(zero));
    if (fill != -INFINITY || zero != 0.0f || std::signbit(zero) ||
        n[3]->ne[0] != nk + ns || ggml_nrows(n[0]) != 1 || ggml_nrows(n[3]) != 1 ||
        n[1]->src[0] != n[0] || n[1]->ne[0] != ns || n[1]->ne[1] != nt || ggml_nrows(n[1]) != nt ||
        n[2]->src[0] != n[1] || n[2]->ne[0] != 1 || n[2]->ne[1] != ns || n[2]->ne[2] != nt || n[2]->ne[3] != 1 ||
        n[4]->src[0] != n[3] || n[4]->ne[0] != nk + ns || n[4]->ne[1] != nt || ggml_nrows(n[4]) != nt ||
        n[5]->src[0] != n[4] || n[5]->ne[0] != 1 || n[5]->ne[1] != nk + ns || n[5]->ne[2] != nt || n[5]->ne[3] != 1 ||
        n[6]->src[0] != n[2] || n[6]->src[2] != n[5] || n[6]->view_src != n[4] ||
        n[7]->src[0] != n[6] || n[7]->view_src != n[4] || n[7]->view_offs != 0 ||
        !ggml_are_same_shape(n[7], dst) || n[7]->nb[0] != ggml_type_size(dst->type)) {
        return 0;
    }
    size_t stride = (nk + ns) * ggml_type_size(dst->type);
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (n[7]->nb[d] != stride) {
            return 0;
        }
        stride *= dst->ne[d];
    }
    return 8;
}

int ggml_sycl_qsa_attn_absorbs(int device, const ggml_cgraph * graph, int i) {
    static const bool enabled = ggml_sycl_get_env("GGML_SYCL_QSA_COMPACT", 0) != 0;
    if (!enabled || !ggml_sycl_qsa_mask_absorbs(graph, i) || i + 10 >= graph->n_nodes) {
        return 0;
    }
    const ggml_op ops[] = { GGML_OP_FILL, GGML_OP_REPEAT, GGML_OP_RESHAPE, GGML_OP_FILL, GGML_OP_REPEAT,
                            GGML_OP_RESHAPE, GGML_OP_SET_ROWS, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_RESHAPE, GGML_OP_FLASH_ATTN_EXT };
    const int outputs[] = { i + 10 };
    if (!ggml_can_fuse_subgraph(graph, i, 11, ops, outputs, 1)) {
        return 0;
    }
    const auto n = graph->nodes + i;
    const ggml_tensor * causal = n[8]->src[1];
    if (!n[10]->src[0] || !n[10]->src[1] || !n[10]->src[2] ||
        n[8]->type != GGML_TYPE_F16 || n[9]->src[0] != n[8] || n[9]->view_src != n[8] || n[9]->view_offs != 0 ||
        !ggml_are_same_shape(n[8], n[9]) || n[10]->src[3] != n[9] || n[9]->flags & GGML_TENSOR_FLAG_INPUT ||
        causal->view_offs % 16 != 0 || (uintptr_t) causal->data % 16 != 0 ||
        n[8]->ne[0] != n[10]->src[1]->ne[1] || ggml_nrows(n[8]) != n[10]->src[0]->ne[1] ||
        !ggml_sycl_flash_attn_ext_xmx_qsa_supported(device, n[10])) {
        return 0;
    }
    return 10;
}

int ggml_sycl_fuse_qsa_attn(ggml_backend_sycl_context & ctx, ggml_cgraph * graph, int i) {
    const int skip = ggml_sycl_qsa_attn_absorbs(ctx.device, graph, i);
    if (!skip) {
        return 0;
    }
    GGML_SYCL_DEBUG("%s: %s\n", __func__, graph->nodes[i + skip]->name);
    ggml_sycl_flash_attn_ext_xmx_qsa(ctx, graph->nodes[i + skip], graph->nodes[i + 6]->src[1], graph->nodes[i + 8]->src[1],
                                  ggml_get_op_params_f32(graph->nodes[i + 3], 0), ggml_get_op_params_f32(graph->nodes[i], 0));
    return skip;
}

template <typename T>
static void qsa_mask_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * idx,
                           const ggml_tensor * mask, ggml_tensor * dst, float fill, float zero) {
    const int64_t nk = dst->ne[0], ns = idx->ne[0];
    const size_t ni = ggml_nelements(idx), nd = ggml_nelements(dst);
    const int32_t * indices = (const int32_t *) idx->data;
    const T * causal = (const T *) mask->data;
    T * out = (T *) dst->data;
    const auto stream = ctx.stream();
    constexpr size_t block = 256;
    ggml_sycl_pool_alloc<T> saved(ctx.pool());
    const T * selected = nullptr;
    // ADD may reuse its causal input. Save the selected entries before filling it.
    if (out == causal) {
        selected = saved.alloc(ni);
        T * values = saved.get();
        stream->parallel_for(sycl::nd_range<1>((ni + block - 1) / block * block, block), [=](sycl::nd_item<1> it) {
            const size_t j = it.get_global_id(0);
            if (j < ni) {
                const int32_t k = indices[j];
                values[j] = k >= 0 && k < nk ? causal[(j / ns) * nk + k] : T(0.0f);
            }
        });
    }
    stream->parallel_for(sycl::nd_range<1>((nd + block - 1) / block * block, block), [=](sycl::nd_item<1> it) {
        const size_t j = it.get_global_id(0);
        if (j < nd) {
            out[j] = T(fill + float(causal[j]));
        }
    });
    stream->parallel_for(sycl::nd_range<1>((ni + block - 1) / block * block, block), [=](sycl::nd_item<1> it) {
        const size_t j = it.get_global_id(0);
        if (j < ni) {
            const int32_t k = indices[j];
            if (k >= 0 && k < nk) {
                const size_t at = (j / ns) * nk + k;
                const float v = float(selected ? selected[j] : causal[at]);
                out[at] = T(zero + v);
            }
        }
    });
}

int ggml_sycl_fuse_qsa_mask(ggml_backend_sycl_context & ctx, ggml_cgraph * graph, int i) {
    const int skip = ggml_sycl_qsa_mask_absorbs(graph, i);
    if (!skip) {
        return 0;
    }
    const ggml_tensor * idx = graph->nodes[i + 6]->src[1];
    const ggml_tensor * mask = graph->nodes[i + 8]->src[1];
    ggml_tensor * dst = graph->nodes[i + 8];
    const float fill = ggml_get_op_params_f32(graph->nodes[i + 3], 0);
    const float zero = ggml_get_op_params_f32(graph->nodes[i], 0);
    GGML_SYCL_DEBUG("%s: %lld cells, %lld rows\n", __func__, (long long) dst->ne[0], (long long) ggml_nrows(dst));
    if (dst->type == GGML_TYPE_F16) {
        qsa_mask_sycl<sycl::half>(ctx, idx, mask, dst, fill, zero);
    } else {
        qsa_mask_sycl<float>(ctx, idx, mask, dst, fill, zero);
    }
    return skip;
}

template<typename TIn, typename TOut>
static inline std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), void>
convert (const char* src, char* dst) {
    auto src_val = *reinterpret_cast<const TIn*>(src);
    auto dst_val = sycl::vec<TIn, 1>(src_val).template convert<TOut, sycl::rounding_mode::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

#ifdef GGML_SYCL_HAS_BF16
// sycl::vec::convert does not provide a half -> bfloat16 path, so route through float.
template<>
inline void convert<sycl::half, sycl::ext::oneapi::bfloat16>(const char* src, char* dst) {
    const float tmp = sycl::vec<sycl::half, 1>(*reinterpret_cast<const sycl::half*>(src))
                          .template convert<float, sycl::rounding_mode::automatic>()[0];
    *reinterpret_cast<sycl::ext::oneapi::bfloat16*>(dst) = sycl::ext::oneapi::bfloat16(tmp);
}
#endif

template <typename TIn, typename TIdx, typename blockType, int qk, cpy_kernel_t cpyblck>
static void set_rows_sycl_q(const char * __restrict__ src0_d,
                            const TIdx * __restrict__ src1_d,
                            blockType * __restrict__ dst_d,
                            // tensor dimensions src0 and src1
                            const int64_t ne00,
                            const int64_t ne01,
                            const int64_t ne02,
                            const int64_t ne03,
                            const int64_t ne10,
                            const int64_t ne11,
                            const int64_t ne12,
                            const int64_t ne13,
                            // strides for src0
                            const size_t  nb00,
                            const size_t  nb01,
                            const size_t  nb02,
                            const size_t  nb03,
                            // strides for src1
                            const size_t  nb10,
                            const size_t  nb11,
                            const size_t  nb12,
                            const size_t  nb13,
                            // strides for dst
                            const size_t  nb1,
                            const size_t  nb2,
                            const size_t  nb3,
                            queue_ptr     stream) {
    const int64_t total_blocks = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr int block_size   = 256;
    const int64_t grid_size    = ceil_div(total_blocks, block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item_ct1) {
        const int64_t i = item_ct1.get_global_linear_id();
        if (i >= total_blocks) {
            return;
        }
        const int64_t i_base      = i * qk;
        const int64_t i03         = i_base / (ne00 * ne01 * ne02);
        const int64_t rem1        = i_base - i03 * (ne00 * ne01 * ne02);
        const int64_t i02         = rem1 / (ne00 * ne01);
        const int64_t rem2        = rem1 - i02 * ne00 * ne01;
        const int64_t i01         = rem2 / ne00;
        const int64_t i00         = rem2 - i01 * ne00;
        const int64_t i12         = i03 % ne12;
        const int64_t i11         = i02 % ne11;
        const int64_t i10         = i01;
        const size_t  src_offset  = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        const char *  src_block   = src0_d + src_offset + i00 * sizeof(TIn);
        const size_t  src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        const int64_t dst_row     = src1_d[src1_offset / sizeof(TIdx)];
        const size_t  dst_offset =
            calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 }) + (i00 / qk) * sizeof(blockType);
        char * dst_block = reinterpret_cast<char *>(reinterpret_cast<char *>(dst_d) + dst_offset);
        if constexpr (std::is_same_v<TIn, float>) {
            cpyblck(src_block, dst_block);
        } else {
            float src_block_f32[qk];
            const TIn * src_block_t = reinterpret_cast<const TIn *>(src_block);
            for (int j = 0; j < qk; ++j) {
                src_block_f32[j] = (float) src_block_t[j];
            }
            cpyblck(reinterpret_cast<const char *>(src_block_f32), dst_block);
        }
    });
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(nb00);
    GGML_UNUSED(nb13);
}

template<typename blockType>
using quantize_row_qk_t = void (*)(const float *, blockType *, int64_t);

using quantize_rows_f_t = size_t (*)(const float *, void *, int64_t, int64_t, const float *);

template <typename TIn, typename TIdx, typename blockType, int qk, quantize_row_qk_t<blockType> quantize_row>
static void set_rows_sycl_qk_host(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
        const int64_t ne11,
        const int64_t ne12,
        const size_t nb01,
        const size_t nb02,
        const size_t nb03,
        const size_t nb10,
        const size_t nb11,
        const size_t nb12,
        const size_t nb1,
        const size_t nb2,
        const size_t nb3,
        queue_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    const size_t src0_bytes = ggml_nbytes(src0);
    const size_t src1_bytes = ggml_nbytes(src1);

    std::vector<char> src0_host(src0_bytes);
    std::vector<char> src1_host(src1_bytes);

    stream->memcpy(src0_host.data(), src0->data, src0_bytes);
    stream->memcpy(src1_host.data(), src1->data, src1_bytes);
    stream->wait();

    std::vector<float> src_row_f32(ne00);
    const int64_t nblocks = ne00 / qk;
    std::vector<blockType> dst_row_q(nblocks);

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i01 = 0; i01 < ne01; ++i01) {
                const int64_t i12 = i03 % ne12;
                const int64_t i11 = i02 % ne11;
                const int64_t i10 = i01;

                const size_t src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                const int64_t dst_row = *(const TIdx *) (src1_host.data() + src1_offset);

                const size_t src0_row_offset = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                const TIn * src_row = reinterpret_cast<const TIn *>(src0_host.data() + src0_row_offset);

                for (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_row_f32[i00] = (float) src_row[i00];
                }

                quantize_row(src_row_f32.data(), dst_row_q.data(), ne00);

                const size_t dst_offset = calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 });
                stream->memcpy((char *) dst->data + dst_offset, dst_row_q.data(), nblocks * sizeof(blockType));
                stream->wait();
            }
        }
    }
}

template <typename TIn, typename TIdx, typename blockType, int qk, quantize_rows_f_t quantize_rows>
static void set_rows_sycl_iq_host(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
        const int64_t ne11,
        const int64_t ne12,
        const size_t nb01,
        const size_t nb02,
        const size_t nb03,
        const size_t nb10,
        const size_t nb11,
        const size_t nb12,
        const size_t nb1,
        const size_t nb2,
        const size_t nb3,
        queue_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    const size_t src0_bytes = ggml_nbytes(src0);
    const size_t src1_bytes = ggml_nbytes(src1);

    std::vector<char> src0_host(src0_bytes);
    std::vector<char> src1_host(src1_bytes);

    stream->memcpy(src0_host.data(), src0->data, src0_bytes);
    stream->memcpy(src1_host.data(), src1->data, src1_bytes);
    stream->wait();

    std::vector<float> src_row_f32(ne00);
    const int64_t nblocks = ne00 / qk;
    std::vector<blockType> dst_row_q(nblocks);

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i01 = 0; i01 < ne01; ++i01) {
                const int64_t i12 = i03 % ne12;
                const int64_t i11 = i02 % ne11;
                const int64_t i10 = i01;

                const size_t src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                const int64_t dst_row = *(const TIdx *) (src1_host.data() + src1_offset);

                const size_t src0_row_offset = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                const TIn * src_row = reinterpret_cast<const TIn *>(src0_host.data() + src0_row_offset);

                for (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_row_f32[i00] = (float) src_row[i00];
                }

                quantize_rows(src_row_f32.data(), dst_row_q.data(), 1, ne00, nullptr);

                const size_t dst_offset = calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 });
                stream->memcpy((char *) dst->data + dst_offset, dst_row_q.data(), nblocks * sizeof(blockType));
                stream->wait();
            }
        }
    }
}

template<typename TIn, typename TIdx, typename TOut>
static void k_set_rows(
        const char * __restrict__ src0, const TIdx * __restrict__ src1, char * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne11, const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size,
        const int64_t total_elements,
        const sycl::nd_item<1> & item_ct1) {

    const int64_t i = item_ct1.get_global_linear_id();
    if (i >= total_elements) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(const TIdx *)((const char *)src1 + calculate_offset<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    const char * src0_row = src0 + calculate_offset<3>({nb01, nb02, nb03}, {i01, i02, i03});
    const char * src_elem = src0_row + i00 * src_type_size;
    char * dst_row_ptr = dst + dst_row*nb1 + i02*nb2 + i03*nb3;
    char * dst_elem = dst_row_ptr + i00 * dst_type_size;

    convert<TIn, TOut>(src_elem, dst_elem);
}

template<typename TIn, typename TIdx, typename TOut>
static void set_rows_sycl(
        const char * src0_d, const TIdx * src1_d, char * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne11, const int64_t ne12, const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size,
        queue_ptr stream) {

    const int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    constexpr int block_size = 64;
    const int64_t grid_size = ceil_div(total_elements, block_size);

    stream->parallel_for(
        sycl::nd_range<1>(grid_size * block_size, block_size),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            k_set_rows<TIn, TIdx, TOut>(
                src0_d, src1_d, dst_d,
                ne00, ne01, ne02,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                src_type_size, dst_type_size,
                total_elements,
                item_ct1
            );
        }
    );
}

template<typename TIn, typename TIdx>
static void set_rows_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const char * src0_d = (const char *)src0->data;
    const TIdx * src1_d = (const TIdx *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    dpct::queue_ptr stream = ctx.stream();
    switch (dst->type) {
        case GGML_TYPE_F32:
            set_rows_sycl<TIn, TIdx, float>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(TIn), sizeof(float),
                stream
            );
            break;
        case GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            set_rows_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(TIn), sizeof(sycl::half),
                stream
            );
            break;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            set_rows_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(TIn), sizeof(sycl::ext::oneapi::bfloat16),
                stream
            );
            break;
#endif
        case GGML_TYPE_Q8_0:
            set_rows_sycl_q<TIn, TIdx, block_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (block_q8_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q1_0:
            set_rows_sycl_q<TIn, TIdx, block_q1_0, QK1_0, cpy_blck_f32_q1_0>(
                src0_d, src1_d, (block_q1_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_0:
            set_rows_sycl_q<TIn, TIdx, block_q2_0, QK2_0, cpy_blck_f32_q2_0>(
                src0_d, src1_d, (block_q2_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            set_rows_sycl_q<TIn, TIdx, block_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (block_q5_1 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            set_rows_sycl_q<TIn, TIdx, block_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (block_q5_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            set_rows_sycl_q<TIn, TIdx, block_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (block_q4_1 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            set_rows_sycl_q<TIn, TIdx, block_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (block_q4_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            set_rows_sycl_q<TIn, TIdx, block_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (block_iq4_nl *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_MXFP4:
            set_rows_sycl_q<TIn, TIdx, block_mxfp4, QK_MXFP4, cpy_blck_f32_mxfp4>(
                src0_d, src1_d, (block_mxfp4 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_NVFP4:
            set_rows_sycl_q<TIn, TIdx, block_nvfp4, QK_NVFP4, cpy_blck_f32_nvfp4>(
                src0_d, src1_d, (block_nvfp4 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q2_K, QK_K, quantize_row_q2_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            break;
        case GGML_TYPE_Q3_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q3_K, QK_K, quantize_row_q3_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            break;
        case GGML_TYPE_Q4_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q4_K, QK_K, quantize_row_q4_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            break;
        case GGML_TYPE_Q5_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q5_K, QK_K, quantize_row_q5_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            break;
        case GGML_TYPE_Q6_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q6_K, QK_K, quantize_row_q6_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_xxs, QK_K, quantize_iq2_xxs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            break;
        case GGML_TYPE_IQ2_XS:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_xs, QK_K, quantize_iq2_xs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            break;
        case GGML_TYPE_IQ2_S:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_s, QK_K, quantize_iq2_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq3_xxs, QK_K, quantize_row_iq3_xxs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            break;
        case GGML_TYPE_IQ3_S:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq3_s, QK_K, quantize_row_iq3_s_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            break;
        case GGML_TYPE_IQ1_S:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq1_s, QK_K, quantize_iq1_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            break;
        case GGML_TYPE_IQ1_M:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq1_m, QK_K, quantize_iq1_m>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            break;
        case GGML_TYPE_IQ4_XS:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq4_xs, QK_K, quantize_row_iq4_xs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            break;
        default:
            GGML_ABORT("Unsupported tensor type: src0 %s src1 %s dst %s", ggml_type_name(dst->src[0]->type),
                ggml_type_name(dst->src[1]->type), ggml_type_name(dst->type));
            break;
    }
}

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    // dispatch on the index type (src1) and the source value type (src0)
    if (src0->type == GGML_TYPE_F16) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_sycl<sycl::half, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_sycl<sycl::half, int32_t>(ctx, src0, src1, dst);
        }
    } else {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_sycl<float, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_sycl<float, int32_t>(ctx, src0, src1, dst);
        }
    }
}
