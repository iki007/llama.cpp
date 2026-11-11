#ifndef GGML_SYCL_SET_ROWS_HPP
#define GGML_SYCL_SET_ROWS_HPP

#include "common.hpp"

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
int ggml_sycl_qsa_mask_absorbs(const ggml_cgraph * cgraph, int node_idx);
int ggml_sycl_fuse_qsa_mask(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);
int ggml_sycl_qsa_attn_absorbs(int device, const ggml_cgraph * cgraph, int node_idx);
int ggml_sycl_fuse_qsa_attn(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

#endif // GGML_SYCL_SET_ROWS_HPP
