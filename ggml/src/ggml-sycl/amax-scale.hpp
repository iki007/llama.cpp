#ifndef GGML_SYCL_AMAX_SCALE_HPP
#define GGML_SYCL_AMAX_SCALE_HPP

#include "common.hpp"

// The per-row power-of-two scale of llm_graph_context::build_amax_src1_f16() (the MoE down-projection input of
// Recursive Flint): abs -> pool_1d(max over the row) -> scale -> log -> scale -> ceil -> clamp -> scale -> exp,
// then div(x, e). Ten launches per MoE layer, most of them on one value per row; fused into one kernel that
// writes both outputs, e (the exp node, used again after the expert mat-vec) and x / e.
struct ggml_sycl_amax_scale_match {
    const ggml_tensor * x          = nullptr;  // the row source, src of the abs and of the div
    ggml_tensor *       e          = nullptr;  // the exp node: one value per row
    ggml_tensor *       res        = nullptr;  // the div node: x / e
    float               s[3]       = {};       // the three scales ...
    float               b[3]       = {};       // ... and their biases
    float               lo         = 0.0f;     // the clamp range
    float               hi         = 0.0f;
    int                 node_count = 0;
};

// structural match starting at the abs; does not check memory ranges
bool ggml_sycl_match_amax_scale(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_amax_scale_match & match);

bool ggml_sycl_amax_scale_ranges_ok(const ggml_sycl_amax_scale_match & match);

void ggml_sycl_op_amax_scale(ggml_backend_sycl_context & ctx, const ggml_sycl_amax_scale_match & match);

#endif  // GGML_SYCL_AMAX_SCALE_HPP
