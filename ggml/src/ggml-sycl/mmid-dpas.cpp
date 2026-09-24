#include "mmid-dpas.hpp"
#include "dpas.hpp"

#ifdef GGML_SYCL_HAS_DPAS

#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

// A work-group of GGML_SYCL_MMID_DPAS_TPW threads covers 16*TPW rows of one expert x one token tile
// (NG*8 expert-sorted tokens). Each thread dequantizes its 16 rows once per 16 k and runs NG DPAS,
// one per 8 tokens, whose A operands come from one 2D block load of the tile's sorted f16 activations
// (rows past the end of the sorted buffer read as zero; tokens past the tile's count are computed, not
// stored). Each output row goes straight to its (slot, token) place in dst, so no scatter pass follows.
constexpr int GGML_SYCL_MMID_DPAS_TPW = 4;

template <ggml_type T, int NG>
ESIMD_INLINE void mul_mat_id_dpas(const void * weights, const size_t expert_bytes, const int ncols, const int nrows,
                                  const sycl::half * y, const int y_rows, float * dst, const size_t dst_nb1,
                                  const size_t dst_nb2, const mmid_row_mapping * row_mapping,
                                  const ggml_sycl_mmid_tile * tiles, const int n_row_wgs, const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    namespace xmx     = sycl::ext::intel::esimd::xmx;
    namespace esimd_x = sycl::ext::intel::experimental::esimd;
    using traits = dpas_tile_traits<T>;
    constexpr int ROWS = GGML_SYCL_DPAS_ROWS;

    const int wg   = it.get_group(0);
    const int row0 = ((wg % n_row_wgs) * GGML_SYCL_MMID_DPAS_TPW + (int) it.get_local_id(0)) * ROWS;
    if (row0 >= nrows) {
        return;
    }
    const ggml_sycl_mmid_tile t = tiles[wg / n_row_wgs];

    const int    nb_row = ncols / QK_K;
    const size_t nb     = (size_t) nrows * nb_row;
    const void * w      = (const char *) weights + (size_t) t.expert * expert_bytes;

    simd<uint32_t, ROWS> rows(row0, 1);
    rows.merge(simd<uint32_t, ROWS>(nrows - 1), rows >= (uint32_t) nrows);
    const simd<uint32_t, ROWS> row_blk = rows * (uint32_t) nb_row;

    // acc[(8 g + i) * 16 + n]: token 8 g + i of the tile, row row0 + n
    simd<float, NG * 8 * ROWS> acc = 0.0f;
    const unsigned surf_w = (unsigned) ncols * sizeof(sycl::half) - 1;
    const unsigned surf_h = (unsigned) y_rows - 1;
    // one load per 16 k covers the whole tile; reusing the descriptor and moving only x is cheaper than
    // building it for every load
    static_assert(NG * 8 <= 32, "a 2D block load covers at most 32 rows");
    esimd_x::config_2d_mem_access<sycl::half, 16, NG * 8, 1> a_desc(y, surf_w, surf_h, surf_w, 0, t.row_begin);
    for (int ib = 0; ib < nb_row; ++ib) {
        traits::block(w, nb, row_blk + (uint32_t) ib, [&](int koff, simd<sycl::half, 256> & b) {
            a_desc.set_x(ib * QK_K + koff);
            simd<sycl::half, NG * 128> a = esimd_x::lsc_load_2d<sycl::half, 16, NG * 8, 1, false, false>(a_desc);
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                simd<float, 128> c = acc.template select<128, 1>(g * 128);
                acc.template select<128, 1>(g * 128) =
                    xmx::dpas<8, 8, float, float>(c, b, a.template select<128, 1>(g * 128).read());
            }
        });
    }

#pragma unroll
    for (int i = 0; i < NG * 8; ++i) {
        if (i < t.row_count) {
            const mmid_row_mapping m = row_mapping[t.row_begin + i];
            float * d = (float *) ((char *) dst + m.i1 * dst_nb1 + m.i2 * dst_nb2) + row0;
            if (row0 + ROWS <= nrows) {
                block_store<float, ROWS>(d, acc.template select<ROWS, 1>(i * ROWS).read());
            } else {
                for (int n = 0; n < ROWS && row0 + n < nrows; ++n) {
                    d[n] = acc[i * ROWS + n];
                }
            }
        }
    }
}

// Runs an ESIMD kernel with 256 GRF: half the threads per XVE, but each keeps more loads in flight.
template <typename F> struct mul_mat_id_dpas_grf256 {
    F f;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL { f(it); }
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    }
};

template <ggml_type T, int NG>
static void mul_mat_id_dpas_sycl(const void * weights, size_t expert_bytes, int ncols, int nrows, const sycl::half * y,
                                 int y_rows, float * dst, size_t dst_nb1, size_t dst_nb2,
                                 const mmid_row_mapping * row_mapping, const ggml_sycl_mmid_tile * tiles, int n_tiles,
                                 dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr int TPW       = GGML_SYCL_MMID_DPAS_TPW;
    const int     n_row_wgs = (nrows + GGML_SYCL_DPAS_ROWS * TPW - 1) / (GGML_SYCL_DPAS_ROWS * TPW);
    const size_t  wgs       = (size_t) n_tiles * n_row_wgs;
    const sycl::nd_range<1> range(sycl::range<1>(wgs * TPW), sycl::range<1>(TPW));
    auto kernel = [=](sycl::nd_item<1> it) SYCL_ESIMD_FUNCTION {
        mul_mat_id_dpas<T, NG>(weights, expert_bytes, ncols, nrows, y, y_rows, dst, dst_nb1, dst_nb2, row_mapping,
                               tiles, n_row_wgs, it);
    };
    // 256 GRF pays off for long k loops (Arc Pro B70: -16% at 2048 k) and loses for short ones (+14% at 512 k)
    if (ncols >= 1024) {
        stream->parallel_for(range, mul_mat_id_dpas_grf256<decltype(kernel)>{ kernel });
    } else {
        stream->parallel_for(range, [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL { kernel(it); });
    }
}

#endif // GGML_SYCL_HAS_DPAS

bool ggml_sycl_mul_mat_id_dpas_supported(const ggml_backend_sycl_context & ctx, ggml_type type) {
#ifdef GGML_SYCL_HAS_DPAS
    const auto arch = ggml_sycl_info().devices[ctx.device].hw_info.arch;
    return g_ggml_sycl_enable_esimd && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K) &&
           (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31);
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(type);
    return false;
#endif
}

void ggml_sycl_mul_mat_id_dpas(ggml_type type, const void * weights, size_t expert_bytes, int ncols, int nrows,
                               const sycl::half * y, int y_rows, float * dst, size_t dst_nb1, size_t dst_nb2,
                               const mmid_row_mapping * row_mapping, const ggml_sycl_mmid_tile * tiles, int n_tiles,
                               dpct::queue_ptr stream) {
#ifdef GGML_SYCL_HAS_DPAS
    static_assert(GGML_SYCL_MMID_DPAS_TILE_TOKENS == 4 * 8, "tile tokens must match NG");
    switch (type) {
        case GGML_TYPE_Q4_K:
            mul_mat_id_dpas_sycl<GGML_TYPE_Q4_K, 4>(weights, expert_bytes, ncols, nrows, y, y_rows, dst, dst_nb1, dst_nb2, row_mapping, tiles, n_tiles, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_id_dpas_sycl<GGML_TYPE_Q6_K, 4>(weights, expert_bytes, ncols, nrows, y, y_rows, dst, dst_nb1, dst_nb2, row_mapping, tiles, n_tiles, stream);
            break;
        default:
            GGML_ABORT("no XMX mul_mat_id for %s", ggml_type_name(type));
    }
#else
    GGML_UNUSED(type); GGML_UNUSED(weights); GGML_UNUSED(expert_bytes); GGML_UNUSED(ncols); GGML_UNUSED(nrows);
    GGML_UNUSED(y); GGML_UNUSED(y_rows); GGML_UNUSED(dst); GGML_UNUSED(dst_nb1); GGML_UNUSED(dst_nb2); GGML_UNUSED(row_mapping);
    GGML_UNUSED(tiles); GGML_UNUSED(n_tiles); GGML_UNUSED(stream);
    GGML_ABORT("ESIMD not available");
#endif
}
