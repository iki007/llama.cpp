#include "fusion.hpp"
#include "binbcast.hpp"

#include <algorithm>

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_op_mul_mat_glu_fused().
static bool ggml_sycl_should_fuse_mul_mat_glu(const ggml_tensor * gate, const ggml_tensor * up,
                                              const ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    const ggml_glu_op glu_op = ggml_get_glu_op(glu);
    if (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        return false;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    if (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        return false;
    }

    const ggml_tensor * wu  = up->src[0];
    const ggml_tensor * wg  = gate->src[0];
    const ggml_tensor * act = up->src[1];

    // one activation and one output indexing must serve both weights; the block types
    // may differ, since the plain-layout fused kernel runs each operand's own vec_dot
    // (different types then imply different byte strides, so only the shape must agree)
    if (!ggml_are_same_shape(wu, wg)) {
        return false;
    }
    if (act != gate->src[1]) {
        return false;
    }

    // fused K-quant GEMVs walk whole QK_K super-blocks: the reorder kernel covers same-type
    // q4_K, the plain-layout kernel covers q5_K / iq4_xs pairs incl. mixed gate/up types
    const bool reorder_pair = wu->type == GGML_TYPE_Q4_K && wg->type == GGML_TYPE_Q4_K;
    const bool plain_pair   = (wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_IQ4_XS) &&
                            (wg->type == GGML_TYPE_Q5_K || wg->type == GGML_TYPE_IQ4_XS);
#ifdef GGML_SYCL_MMVQ_HAS_XMX
    // same-type pairs of the types of the XMX kernels, which also take a few more columns
    // (the XMX kernels walk whole QK_K super-blocks, so they need rows of a multiple of QK_K)
    const bool xmx_pair = wu->type == wg->type && ggml_sycl_xmx_supports_type(wu->type) && wu->ne[0] % QK_K == 0;
#else
    const bool xmx_pair = false;
#endif // GGML_SYCL_MMVQ_HAS_XMX
    // the ESIMD kernel covers same-type q2_K..q6_K and q8_0, single column only (as unfused DMMV)
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    const bool esimd_pair   = g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
                            (wu->type == GGML_TYPE_Q2_K || wu->type == GGML_TYPE_Q3_K || wu->type == GGML_TYPE_Q4_K ||
                             wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_Q6_K || wu->type == GGML_TYPE_Q8_0);
#else
    const bool esimd_pair   = false;
#endif
    // q8_0 rows are walked in 32-wide blocks, so they only need to be a multiple of QK8_0
    const int64_t row_align = esimd_pair && wu->type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    if ((!reorder_pair && !plain_pair && !xmx_pair && !esimd_pair) || wu->ne[0] % row_align != 0) {
        return false;
    }

    // one 2D reorder-layout matrix in, a plain column stride out: no broadcast or padding
    if (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        return false;
    }
    if (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        return false;
    }
    if (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        return false;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    if (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        return false;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    const int max_cols = xmx_pair ? std::max(MMVQ_MAX_BATCH_SIZE, g_ggml_sycl_xmx_glu_max_cols) : MMVQ_MAX_BATCH_SIZE;
    if (act->ne[1] > max_cols) {
        return false;
    }

    return true;
}

// the unary ops the fused unary chains in element_wise.cpp have a functor for
static bool ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
    return unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
           unary_op == GGML_UNARY_OP_SOFTPLUS || unary_op == GGML_UNARY_OP_TANH;
}

bool ggml_sycl_can_fuse(const ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops,
                        std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    if (!g_ggml_sycl_enable_fusion) {
        return false;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
            return false;
        }

        const ggml_tensor * glu  = cgraph->nodes[node_idx + 2];
        const ggml_tensor * gate = glu->src[0];
        const ggml_tensor * up   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        const bool ok = (gate == cgraph->nodes[node_idx] && up == cgraph->nodes[node_idx + 1]) ||
                        (gate == cgraph->nodes[node_idx + 1] && up == cgraph->nodes[node_idx]);
        if (!ok) {
            return false;
        }

        return ggml_sycl_should_fuse_mul_mat_glu(gate, up, glu);
    }

    // the residual sum stays materialised (a later residual add reads it), so ggml_can_fuse cannot
    // express this either: the add and the mul are the subgraph's outputs
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_RMS_NORM &&
        ops.begin()[2] == GGML_OP_MUL) {
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx, node_idx + 2 })) {
            return false;
        }

        const ggml_tensor * add = cgraph->nodes[node_idx];
        if (cgraph->nodes[node_idx + 1]->src[0] != add) {
            return false;
        }

        // the fused kernel does not broadcast the add
        if (add->src[0]->type != GGML_TYPE_F32 || add->src[1]->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(add->src[0], add) || !ggml_are_same_shape(add->src[1], add)) {
            return false;
        }

        return ggml_sycl_can_fuse(cgraph, node_idx + 1, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {});
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        if (ops.size() == 3 && ops.begin()[2] != GGML_OP_ADD) {
            return false;
        }

        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * add      = ops.size() == 3 ? cgraph->nodes[node_idx + 2] : nullptr;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        const ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        if (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_size(mul_w->type)) {
            return false;
        }

        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add != nullptr) {
            if (add->src[0]->type != GGML_TYPE_F32 ||
                add->src[1]->type != GGML_TYPE_F32 ||
                add->type != GGML_TYPE_F32) {
                return false;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            const ggml_tensor * add_w = (add->src[0] == mul) ? add->src[1] : add->src[0];
            if (!ggml_are_same_shape(add_w, add)) {
                return false;
            }

            if (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1])) {
                return false;
            }
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_ADD) {
        const ggml_tensor * add0 = cgraph->nodes[node_idx];
        const ggml_tensor * add1 = cgraph->nodes[node_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // Keep the CUDA association: the running sum is src0 of the next ADD so the fused
        // float fold matches two sequential add() launches.
        if (add1->src[0] != add0) {
            return false;
        }

        const ggml_tensor * c = add1->src[1];
        if (!ggml_sycl_add_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_add_kernel_supports(add0->type, c->type, add1->type)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 1];

        const ggml_unary_op unary_op = ggml_get_unary_op(unary);
        if (unary_op != unary_ops.begin()[0]) {
            return false;
        }

        if (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        if (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            return false;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        if (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx + 1];

        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }
        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        if (!ggml_is_contiguous(silu)) {
            return false;
        }

        return true;
    }

    // ADD(bias) + UNARY + MUL(scale): the delta-net alpha gate, softplus(alpha + dt) * a.
    // The broadcast is what stops the same-shape UNARY + MUL branch above firing past one token.
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
        const ggml_tensor * add   = cgraph->nodes[node_idx];
        const ggml_tensor * unary = cgraph->nodes[node_idx + 1];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 2];

        const ggml_unary_op unary_op = ggml_get_unary_op(unary);
        if (unary_op != unary_ops.begin()[0]) {
            return false;
        }

        if (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            return false;
        }

        // ggml_can_fuse() has already pinned the chain: unary consumes add, mul consumes
        // unary, add and unary have one use each, and all three have the same shape
        const ggml_tensor * a     = add->src[0];
        const ggml_tensor * bias  = add->src[1];
        const ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];

        if (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            return false;
        }

        // the activation and the destination are indexed flat
        if (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
            return false;
        }

        // the kernel reads the bias and the scale as v[col], so each must be a single
        // contiguous row spanning ne0
        if (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            return false;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        if (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD &&
        ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx + 2];

        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }
        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        if (!ggml_is_contiguous(silu)) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx + 1];
        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }
        // the fused kernel reads/writes rows flat like the unfused pair
        if (!ggml_is_contiguous_rows(rms_norm) || !ggml_is_contiguous_rows(scale)) {
            return false;
        }
        return true;
    }

    auto elementwise_alias_safe = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        if (!src || !dst || !src->data || !dst->data) {
            return false;
        }
        if (dst->data == src->data) {
            return true; // exact element-for-element in-place
        }
        const uintptr_t ds = (uintptr_t) dst->data;
        const uintptr_t de = ds + ggml_nbytes(dst);
        const uintptr_t ss = (uintptr_t) src->data;
        const uintptr_t se = ss + ggml_nbytes(src);
        return de <= ss || se <= ds;
    };

    // UNARY + SCALE + MUL: dst = (s * op(x) + b) * other
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SCALE &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * scale = cgraph->nodes[node_idx + 1];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 2];

        const ggml_unary_op unary_op = ggml_get_unary_op(unary);
        if (unary_op != unary_ops.begin()[0] || !ggml_sycl_fused_unary_has_kernel(unary_op)) {
            return false;
        }
        if (scale->src[0] != unary) {
            return false;
        }
        const ggml_tensor * x     = unary->src[0];
        const ggml_tensor * other = (mul->src[0] == scale) ? mul->src[1] : mul->src[0];
        if (!x || !other || other == scale) {
            return false;
        }
        if (x->type != GGML_TYPE_F32 || other->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            return false;
        }
        if (!ggml_is_contiguous(x) || !ggml_is_contiguous(other) || !ggml_is_contiguous(mul)) {
            return false;
        }
        if (!ggml_are_same_shape(x, other) || !ggml_are_same_shape(x, mul)) {
            return false;
        }
        if (!elementwise_alias_safe(mul, x) || !elementwise_alias_safe(mul, other)) {
            return false;
        }
        return true;
    }

    // UNARY + MUL + ADD: dst = acc + op(x) * other
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        ops.begin()[2] == GGML_OP_ADD && unary_ops.size() == 1) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * add   = cgraph->nodes[node_idx + 2];

        const ggml_unary_op unary_op = ggml_get_unary_op(unary);
        if (unary_op != unary_ops.begin()[0] || !ggml_sycl_fused_unary_has_kernel(unary_op)) {
            return false;
        }
        const ggml_tensor * x     = unary->src[0];
        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        const ggml_tensor * acc   = (add->src[0] == mul)   ? add->src[1] : add->src[0];
        if (!x || !other || !acc || other == unary || acc == mul) {
            return false;
        }
        if (x->type != GGML_TYPE_F32 || other->type != GGML_TYPE_F32 ||
            acc->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32) {
            return false;
        }
        if (!ggml_is_contiguous(x) || !ggml_is_contiguous(other) ||
            !ggml_is_contiguous(acc) || !ggml_is_contiguous(add)) {
            return false;
        }
        if (!ggml_are_same_shape(x, other) || !ggml_are_same_shape(x, acc) || !ggml_are_same_shape(x, add)) {
            return false;
        }
        if (!elementwise_alias_safe(add, x) || !elementwise_alias_safe(add, other) || !elementwise_alias_safe(add, acc)) {
            return false;
        }
        return true;
    }

    return false;
}

bool ggml_sycl_can_fuse_l2_clip(const ggml_cgraph * cgraph, int node_idx, bool has_scale) {
    if (!g_ggml_sycl_enable_fusion) {
        return false;
    }

    static constexpr ggml_op ops_with_scale[9] = {
        GGML_OP_MUL, GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT,
        GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL,
    };
    static constexpr ggml_op ops_no_scale[8] = {
        GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT,
        GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL,
    };

    const int n_ops = has_scale ? 9 : 8;
    const int out_idx = node_idx + n_ops - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, n_ops, has_scale ? ops_with_scale : ops_no_scale, &out_idx, 1)) {
        return false;
    }

    const int b = has_scale ? 1 : 0;
    const ggml_tensor * mul0     = has_scale ? cgraph->nodes[node_idx] : nullptr;
    const ggml_tensor * sqr_n    = cgraph->nodes[node_idx + b + 0];
    const ggml_tensor * sum_n    = cgraph->nodes[node_idx + b + 1];
    const ggml_tensor * sqrt_n   = cgraph->nodes[node_idx + b + 2];
    const ggml_tensor * scale_n  = cgraph->nodes[node_idx + b + 3];
    const ggml_tensor * fill_n   = cgraph->nodes[node_idx + b + 4];
    const ggml_tensor * div_n    = cgraph->nodes[node_idx + b + 5];
    const ggml_tensor * clamp_n  = cgraph->nodes[node_idx + b + 6];
    const ggml_tensor * out_mul  = cgraph->nodes[node_idx + b + 7];

    const ggml_tensor * cur_in   = has_scale ? mul0->src[0] : sqr_n->src[0];
    const ggml_tensor * scale_in = has_scale ? mul0->src[1] : nullptr;
    const ggml_tensor * cur_mid  = has_scale ? mul0 : cur_in;

    if (!cur_in || (has_scale && !scale_in)) {
        return false;
    }
    if (sqr_n->src[0] != cur_mid || sum_n->src[0] != sqr_n || sqrt_n->src[0] != sum_n ||
        scale_n->src[0] != sqrt_n || fill_n->src[0] != scale_n ||
        div_n->src[0] != fill_n || div_n->src[1] != scale_n ||
        clamp_n->src[0] != div_n) {
        return false;
    }
    if (!((out_mul->src[0] == cur_mid && out_mul->src[1] == clamp_n) ||
          (out_mul->src[1] == cur_mid && out_mul->src[0] == clamp_n))) {
        return false;
    }

    if (cur_in->type != GGML_TYPE_F32 || out_mul->type != GGML_TYPE_F32 ||
        (has_scale && scale_in->type != GGML_TYPE_F32)) {
        return false;
    }
    if (!ggml_is_contiguous(cur_in) || !ggml_is_contiguous(out_mul) ||
        !ggml_are_same_shape(cur_in, out_mul)) {
        return false;
    }
    if (cur_in->ne[0] <= 0 || cur_in->ne[0] > 256) {
        return false;
    }
    if (has_scale) {
        if (!ggml_is_contiguous(scale_in) || scale_in->ne[0] != cur_in->ne[0] ||
            (scale_in->ne[1] != 1 && scale_in->ne[1] != cur_in->ne[1]) ||
            scale_in->ne[2] != 1 || scale_in->ne[3] != 1) {
            return false;
        }
    }

    // Buffer overlap safety: out_mul may equal cur_in (each subgroup reads its full row before
    // writing), otherwise out_mul must not overlap cur_in or scale_in.
    auto disjoint = [](const ggml_tensor * a, const ggml_tensor * b) -> bool {
        const uintptr_t as = (uintptr_t) a->data;
        const uintptr_t ae = as + ggml_nbytes(a);
        const uintptr_t bs = (uintptr_t) b->data;
        const uintptr_t be = bs + ggml_nbytes(b);
        return ae <= bs || be <= as;
    };
    if (out_mul->data != cur_in->data && !disjoint(out_mul, cur_in)) {
        return false;
    }
    if (has_scale && !disjoint(out_mul, scale_in)) {
        return false;
    }

    return true;
}
