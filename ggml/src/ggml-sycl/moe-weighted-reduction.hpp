#ifndef GGML_SYCL_MOE_WEIGHTED_REDUCTION_HPP
#define GGML_SYCL_MOE_WEIGHTED_REDUCTION_HPP

#include "common.hpp"

// MoE combine tail: experts * [expert_scale *] router_weights, split into per-expert views and
// summed left to right by an ADD chain. Ported from the CUDA fusion.
struct ggml_sycl_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

// structural match starting at the first MUL; does not check memory ranges
bool ggml_sycl_match_moe_weighted_reduction(const ggml_cgraph * cgraph, int node_idx,
                                            ggml_sycl_moe_weighted_reduction_match & match);

void ggml_sycl_op_moe_weighted_reduction(ggml_backend_sycl_context & ctx,
                                         const ggml_sycl_moe_weighted_reduction_match & match);

#endif  // GGML_SYCL_MOE_WEIGHTED_REDUCTION_HPP
