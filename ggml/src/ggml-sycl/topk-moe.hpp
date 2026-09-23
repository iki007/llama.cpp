#ifndef GGML_SYCL_TOPK_MOE_HPP
#define GGML_SYCL_TOPK_MOE_HPP

#include "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
int ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

int ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

// Length of the fusable top-k MoE router subgraph starting at node `i`, or 0. Checks graph shape and
// tensor properties only, so graph_optimize can keep the router's inputs allocated for the fusion.
int ggml_sycl_topk_moe_node_count(const ggml_cgraph * cgraph, int i);

#endif  // GGML_SYCL_TOPK_MOE_HPP
