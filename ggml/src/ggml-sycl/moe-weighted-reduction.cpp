#include <vector>

#include "ggml.h"
#include "ggml-impl.h"
#include "moe-weighted-reduction.hpp"

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most 31 nodes, so k <= 15;
// larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

bool ggml_sycl_match_moe_weighted_reduction(const ggml_cgraph * cgraph, int node_idx,
                                            ggml_sycl_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                   tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                   ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // (experts * expert_scale) * router_weight, or experts * router_weight
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
            split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS || n_tokens <= 0) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int                 n_adds   = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset]                   = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] || previous->ne[1] != n_tokens ||
        previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}

void ggml_sycl_op_moe_weighted_reduction(ggml_backend_sycl_context &                     ctx,
                                         const ggml_sycl_moe_weighted_reduction_match & match) {
    const ggml_tensor * dst = match.dst;
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0, " : fused mul + views + add chain");

    const float *  experts       = (const float *) match.experts->data;
    const float *  expert_scale  = match.expert_scale ? (const float *) match.expert_scale->data : nullptr;
    const float *  weights       = (const float *) match.weights->data;
    float *        out           = (float *) dst->data;
    const int64_t  n_embd        = match.experts->ne[0];
    const int      n_expert_used = (int) match.experts->ne[1];
    const int64_t  n_tokens      = match.experts->ne[2] * match.experts->ne[3];

    // same left-to-right order as the ADD chain it replaces
    ctx.stream()->parallel_for(sycl::range<2>((size_t) n_tokens, (size_t) n_embd), [=](sycl::id<2> id) {
        const size_t token     = id[0];
        const size_t col       = id[1];
        const size_t first_row = token * n_expert_used;

        float sum = (experts[first_row * n_embd + col] * (expert_scale ? expert_scale[first_row] : 1.0f)) *
                    weights[first_row];
        for (int expert = 1; expert < n_expert_used; ++expert) {
            const size_t row = first_row + expert;
            sum += (experts[row * n_embd + col] * (expert_scale ? expert_scale[row] : 1.0f)) * weights[row];
        }
        out[token * n_embd + col] = sum;
    });
}
