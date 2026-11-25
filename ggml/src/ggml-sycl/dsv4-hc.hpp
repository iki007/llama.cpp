#ifndef GGML_SYCL_DSV4_HC_HPP
#define GGML_SYCL_DSV4_HC_HPP

#include "common.hpp"

void ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_raon_hc_mix(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
// DSV4_HC_POST whose branch is an ADD read only by it: adds the two itself (false: not this shape)
bool ggml_sycl_op_dsv4_hc_post_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const ggml_tensor * add);

#endif // GGML_SYCL_DSV4_HC_HPP
