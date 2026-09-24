#ifndef GGML_SYCL_MMID_DPAS_HPP
#define GGML_SYCL_MMID_DPAS_HPP

// Grouped MoE expert GEMM on XMX (DPAS) for the expert-sorted rows of a mul_mat_id: one launch runs
// every expert's token tiles, where the per-expert loop dequantizes each expert to f16 and calls
// oneMKL once per expert.

#include "common.hpp"

// token tile of one expert in the expert-sorted rows of a mul_mat_id: rows [row_begin, row_begin + row_count)
struct ggml_sycl_mmid_tile {
    int32_t expert;
    int32_t row_begin;
    int32_t row_count;
};

// tokens per ggml_sycl_mmid_tile for ggml_sycl_mul_mat_id_dpas
constexpr int GGML_SYCL_MMID_DPAS_TILE_TOKENS = 32;

// whether ggml_sycl_mul_mat_id_dpas() can run weights of this type on this device
bool ggml_sycl_mul_mat_id_dpas_supported(const ggml_backend_sycl_context & ctx, ggml_type type);

// out(r)[0..nrows) = expert(tile of r) weights x y[r] for every sorted row r covered by a tile, where out(r) is the
// f32 row at (char *) dst + row_mapping[r].i1*dst_nb1 + row_mapping[r].i2*dst_nb2.
// weights: n_expert x (nrows x ncols) reordered blocks, expert_bytes apart; y: sorted rows, f16, ncols each.
void ggml_sycl_mul_mat_id_dpas(ggml_type type, const void * weights, size_t expert_bytes, int ncols, int nrows,
                               const sycl::half * y, int y_rows, float * dst, size_t dst_nb1, size_t dst_nb2,
                               const mmid_row_mapping * row_mapping, const ggml_sycl_mmid_tile * tiles, int n_tiles,
                               dpct::queue_ptr stream);

#endif // GGML_SYCL_MMID_DPAS_HPP
