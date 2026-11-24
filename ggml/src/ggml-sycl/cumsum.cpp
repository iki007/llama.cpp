#include "cumsum.hpp"
#include "common.hpp"

#include <algorithm>

#define SYCL_CUMSUM_BLOCK_SIZE 256

static __dpct_inline__ float warp_prefix_inclusive_sum_f32(float x, const sycl::nd_item<3> & item) {
    return sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<float>());
}

static void cumsum_f32_kernel(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t  d1, const int64_t  d2, const int64_t  d3,
        const sycl::nd_item<3> & item, float * smem) {

    const int tid = item.get_local_id(2);
    const int block_size = item.get_local_range(2);
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int warps_per_block = block_size / WARP_SIZE;

    float * s_vals      = smem;
    float * s_warp_sums = smem + block_size;
    float * s_carry     = smem + block_size + warps_per_block;

    if (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int64_t i3 = item.get_group(0);
    const int64_t i2 = item.get_group(1);
    const int64_t i1 = item.get_group(2);
    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    const float * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    float       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr int num_unroll = 4;
    float temp[num_unroll];

    for (int64_t i = 0; i < ne00; i += num_unroll * block_size) {
        int64_t idx = i + tid * num_unroll;

        temp[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
        for (int j = 1; j < num_unroll; j++) {
            temp[j] = temp[j - 1];
            if (idx + j < ne00) {
                temp[j] += src_row[idx + j];
            }
        }

        float val = (idx < ne00) ? temp[num_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_sum_f32(val, item);
        s_vals[tid] = val;

        if (lane == WARP_SIZE - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        if (warp == 0) {
            float w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            float inc = warp_prefix_inclusive_sum_f32(w, item);
            if (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            if (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        float carry = s_carry[0];
        float final_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[num_unroll - 1];

#pragma unroll
        for (int j = 0; j < num_unroll; j++) {
            if (idx + j < ne00) {
                dst_row[idx + j] = temp[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        if (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

// A few long rows (the Inkling2 DSpark drafter's draws scan its 200K vocabulary three times a draft step): one
// work-group a row walked it in ~200 steps with three barriers each (~0.14 ms a call). Chunks of CUMSUM_CHUNK values
// take a work-group each: the chunk sums, their exclusive scan per row (at most CUMSUM_WG chunks), then every chunk's
// scan from its offset. The sums of integer-valued rows (the drafter's radix parts) come out the same in any order.
static constexpr int CUMSUM_CHUNK = 4096;
static constexpr int CUMSUM_WG    = 256;  // CUMSUM_CHUNK / CUMSUM_WG values a work-item

static bool cumsum_chunked(ggml_backend_sycl_context & ctx, const float * src, float * dst, int64_t ne00, int64_t ne01,
                           int64_t ne02, int64_t ne03, int64_t s01, int64_t s02, int64_t s03, int64_t d1, int64_t d2,
                           int64_t d3) {
    const int64_t nrows   = ne01 * ne02 * ne03;
    const int64_t nchunks = (ne00 + CUMSUM_CHUNK - 1) / CUMSUM_CHUNK;
    if (ne00 < 4 * CUMSUM_CHUNK || nrows >= 64 || nchunks > CUMSUM_WG) {
        return false;
    }
    constexpr int VPT = CUMSUM_CHUNK / CUMSUM_WG;
    dpct::queue_ptr stream = ctx.stream();
    ggml_sycl_pool_alloc<float> part(ctx.pool(), (size_t) nrows * nchunks);
    float * p = part.get();
    auto row_ptr = [=](int64_t r, const float * base, int64_t a, int64_t b, int64_t c) {
        const int64_t i1 = r % ne01;
        const int64_t i2 = (r / ne01) % ne02;
        const int64_t i3 = r / (ne01 * ne02);
        return base + i1 * a + i2 * b + i3 * c;
    };
    // 1. the chunk sums
    stream->parallel_for(sycl::nd_range<2>(sycl::range<2>(nrows, nchunks * CUMSUM_WG), sycl::range<2>(1, CUMSUM_WG)),
                         [=](sycl::nd_item<2> it) {
                             const int64_t r  = it.get_group(0);
                             const int64_t c  = it.get_group(1);
                             const float * x  = row_ptr(r, src, s01, s02, s03);
                             const int64_t i0 = c * CUMSUM_CHUNK + (int64_t) it.get_local_id(1) * VPT;
                             float         s  = 0.0f;
                             for (int j = 0; j < VPT; ++j) {
                                 s += i0 + j < ne00 ? x[i0 + j] : 0.0f;
                             }
                             s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
                             if (it.get_local_id(1) == 0) {
                                 p[r * nchunks + c] = s;
                             }
                         });
    // 2. their exclusive scan per row, in place
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nrows * CUMSUM_WG), sycl::range<1>(CUMSUM_WG)),
                         [=](sycl::nd_item<1> it) {
                             const int64_t r = it.get_group(0);
                             const int64_t c = it.get_local_id(0);
                             const float   v = c < nchunks ? p[r * nchunks + c] : 0.0f;
                             const float   e = sycl::exclusive_scan_over_group(it.get_group(), v, sycl::plus<float>());
                             if (c < nchunks) {
                                 p[r * nchunks + c] = e;
                             }
                         });
    // 3. every chunk's scan from its offset
    stream->parallel_for(sycl::nd_range<2>(sycl::range<2>(nrows, nchunks * CUMSUM_WG), sycl::range<2>(1, CUMSUM_WG)),
                         [=](sycl::nd_item<2> it) {
                             const int64_t r  = it.get_group(0);
                             const int64_t c  = it.get_group(1);
                             const float * x  = row_ptr(r, src, s01, s02, s03);
                             float *       y  = const_cast<float *>(row_ptr(r, dst, d1, d2, d3));
                             const int64_t i0 = c * CUMSUM_CHUNK + (int64_t) it.get_local_id(1) * VPT;
                             float         v[VPT];
                             float         run = 0.0f;
                             for (int j = 0; j < VPT; ++j) {
                                 run += i0 + j < ne00 ? x[i0 + j] : 0.0f;
                                 v[j] = run;
                             }
                             const float before =
                                 sycl::exclusive_scan_over_group(it.get_group(), run, sycl::plus<float>()) + p[r * nchunks + c];
                             for (int j = 0; j < VPT; ++j) {
                                 if (i0 + j < ne00) {
                                     y[i0 + j] = v[j] + before;
                                 }
                             }
                         });
    return true;
}

inline void ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src_d = static_cast<const float *>(src0->data);
    float       * dst_d = static_cast<float *>(dst->data);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const size_t ts = sizeof(float);
    const int64_t s01 = src0->nb[1] / ts;
    const int64_t s02 = src0->nb[2] / ts;
    const int64_t s03 = src0->nb[3] / ts;
    const int64_t d1  = dst->nb[1] / ts;
    const int64_t d2  = dst->nb[2] / ts;
    const int64_t d3  = dst->nb[3] / ts;

    if (cumsum_chunked(ctx, src_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, d1, d2, d3)) {
        return;
    }

    const int num_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    int block_size = num_warps * WARP_SIZE;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);
    const int warps_per_block = block_size / WARP_SIZE;
    const int smem_size = block_size + warps_per_block + 2;

    const sycl::range<3> grid(ne03, ne02, ne01);
    const sycl::range<3> block(1, 1, block_size);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> smem_acc(sycl::range<1>(smem_size), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

void ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
