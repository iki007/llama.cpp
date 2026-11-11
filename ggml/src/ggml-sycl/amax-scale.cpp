#include <cfloat>
#include <cmath>

#include "ggml.h"
#include "ggml-impl.h"
#include "amax-scale.hpp"

static constexpr int AMAX_SCALE_NODES = 10;

bool ggml_sycl_match_amax_scale(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_amax_scale_match & match) {
    if (node_idx + AMAX_SCALE_NODES > cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * const * n = cgraph->nodes + node_idx;

    auto is_unary = [](const ggml_tensor * t, ggml_unary_op op) {
        return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
    };
    // abs -> pool_1d -> scale -> log -> scale -> ceil -> clamp -> scale -> exp -> div, each reading the one before
    if (!is_unary(n[0], GGML_UNARY_OP_ABS) || n[1]->op != GGML_OP_POOL_1D || n[2]->op != GGML_OP_SCALE ||
        n[3]->op != GGML_OP_LOG || n[4]->op != GGML_OP_SCALE || !is_unary(n[5], GGML_UNARY_OP_CEIL) ||
        n[6]->op != GGML_OP_CLAMP || n[7]->op != GGML_OP_SCALE || !is_unary(n[8], GGML_UNARY_OP_EXP) ||
        n[9]->op != GGML_OP_DIV) {
        return false;
    }
    for (int i = 1; i < AMAX_SCALE_NODES - 1; ++i) {
        if (n[i]->src[0] != n[i - 1]) {
            return false;
        }
    }

    const ggml_tensor * x = n[0]->src[0];
    if (n[9]->src[0] != x || n[9]->src[1] != n[8]) {
        return false;
    }
    // F32 throughout, the rows of x contiguous, one pooled value per row
    for (int i = 0; i < AMAX_SCALE_NODES; ++i) {
        if (n[i]->type != GGML_TYPE_F32 || !ggml_is_contiguous(n[i])) {
            return false;
        }
    }
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_are_same_shape(n[0], x) ||
        !ggml_are_same_shape(n[9], x)) {
        return false;
    }
    const int32_t * pool = (const int32_t *) n[1]->op_params;  // op, kernel, stride, padding
    if (pool[0] != GGML_OP_POOL_MAX || pool[1] != x->ne[0] || pool[2] != x->ne[0] || pool[3] != 0) {
        return false;
    }
    for (int i = 1; i < AMAX_SCALE_NODES - 1; ++i) {
        if (n[i]->ne[0] != 1 || n[i]->ne[1] != x->ne[1] || n[i]->ne[2] != x->ne[2] || n[i]->ne[3] != x->ne[3]) {
            return false;
        }
    }

    const ggml_op ops[AMAX_SCALE_NODES] = { GGML_OP_UNARY, GGML_OP_POOL_1D, GGML_OP_SCALE, GGML_OP_LOG, GGML_OP_SCALE,
                                            GGML_OP_UNARY, GGML_OP_CLAMP,   GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_DIV };
    const int outputs[2] = { node_idx + 8, node_idx + 9 };
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, AMAX_SCALE_NODES, ops, outputs, 2)) {
        return false;
    }

    const int scales[3] = { 2, 4, 7 };
    for (int i = 0; i < 3; ++i) {
        match.s[i] = ggml_get_op_params_f32(n[scales[i]], 0);
        match.b[i] = ggml_get_op_params_f32(n[scales[i]], 1);
    }
    match.lo         = ggml_get_op_params_f32(n[6], 0);
    match.hi         = ggml_get_op_params_f32(n[6], 1);
    match.x          = x;
    match.e          = n[8];
    match.res        = n[9];
    match.node_count = AMAX_SCALE_NODES;
    return true;
}

// The kernel reads each row whole before writing it, so x / e may overwrite x (the allocator places the div in
// place); e must not overlap either, and an out-of-place x / e must not overlap x.
bool ggml_sycl_amax_scale_ranges_ok(const ggml_sycl_amax_scale_match & m) {
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data;
        const char * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    return !overlap(m.e, m.x) && !overlap(m.e, m.res) && (m.res->data == m.x->data || !overlap(m.res, m.x));
}

// One sub-group per row: the row's absolute maximum, then the unfused kernels' per-value math in their order
// (scale_f32, op_log, op_ceil, op_clamp, op_exp, op_div), so the outputs are those of the ten launches.
void ggml_sycl_op_amax_scale(ggml_backend_sycl_context & ctx, const ggml_sycl_amax_scale_match & m) {
    const int64_t ne0    = m.x->ne[0];
    const int64_t n_rows = ggml_nrows(m.x);
    const float * x      = (const float *) m.x->data;
    float *       e_out  = (float *) m.e->data;
    float *       res    = (float *) m.res->data;
    const float   s0 = m.s[0], s1 = m.s[1], s2 = m.s[2], b0 = m.b[0], b1 = m.b[1], b2 = m.b[2], lo = m.lo, hi = m.hi;

    constexpr int rows_per_group = 8;
    const int64_t n_groups       = (n_rows + rows_per_group - 1) / rows_per_group;
    ctx.stream()->parallel_for(
        sycl::nd_range<1>(n_groups * rows_per_group * WARP_SIZE, rows_per_group * WARP_SIZE),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const sycl::sub_group sg  = it.get_sub_group();
            const int64_t         row = it.get_group(0) * rows_per_group + sg.get_group_linear_id();
            if (row >= n_rows) {
                return;
            }
            const int     lane = sg.get_local_linear_id();
            const float * xr   = x + row * ne0;
            float         amax = -FLT_MAX;
            for (int64_t i = lane; i < ne0; i += WARP_SIZE) {
                amax = sycl::max(amax, sycl::fabs(xr[i]));
            }
            amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());

            float v = s0 * amax + b0;
            v       = v <= 0.0f ? -INFINITY : sycl::log(v);
            v       = s1 * v + b1;
            v       = sycl::ceil(v);
            v       = v < lo ? lo : (v > hi ? hi : v);
            v       = s2 * v + b2;
            v       = sycl::exp(v);

            if (lane == 0) {
                e_out[row] = v;
            }
            float * rr = res + row * ne0;
            for (int64_t i = lane; i < ne0; i += WARP_SIZE) {
                rr[i] = xr[i] / v;
            }
        });
}
