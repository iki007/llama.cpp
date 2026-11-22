#include "moe_glue.hpp"

#include <cstring>

#include "ggml-impl.h"

// Both kernels are element-wise over their output and repeat the graph's own arithmetic in the graph's order
// (x * scale, then the unary ops; the expert sum as the ADD chain adds the views), so they match the unfused
// kernels to rounding of the transcendental functions only. The outputs are allocated at the last node of their
// chain: when that memory overlaps an input (freed inside the chain), the kernel writes a pool buffer and copies.

struct moe_situ_args {
    const float *   x;      // MUL_MAT_ID output [2F, K, T]
    size_t          x_slot; // floats between slots
    size_t          x_tok;  // floats between tokens
    const float *   scale;  // per-expert scale [n_expert] or nullptr
    const int32_t * ids;    // [K, T]
    size_t          ids_tok;
    float *         out;    // [F, K, T] contiguous
    int             F, K, T;
    float           s1, s3, u1, u3;
};

static void moe_situ_launch(const moe_situ_args & a, float * out, dpct::queue_ptr stream) {
    const size_t        total = (size_t) a.F * a.K * a.T;
    const moe_situ_args k     = a;
    stream->parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
        const size_t i  = idx[0];
        const int    f  = (int) (i % k.F);
        const size_t kt = i / k.F;
        const int    s  = (int) (kt % k.K);
        const int    t  = (int) (kt / k.K);
        const float  sc = k.scale ? k.scale[k.ids[(size_t) t * k.ids_tok + s]] : 1.0f;
        const float * xr = k.x + (size_t) t * k.x_tok + (size_t) s * k.x_slot;
        const float g   = xr[f] * sc;
        const float u   = xr[k.F + f] * sc;
        const float act = (sycl::tanh(g * k.s1) * k.s3) * (1.0f / (1.0f + sycl::exp(-g)));
        const float up  = sycl::tanh(u * k.u1) * k.u3;
        out[i] = act * up;
    });
}

struct moe_combine_args {
    const float *   y;      // MUL_MAT_ID output [E, K, T]
    size_t          y_slot;
    size_t          y_tok;
    const float *   scale;  // per-expert scale or nullptr
    const int32_t * ids;
    size_t          ids_tok;
    const float *   w;      // router weights [1, K, T]
    size_t          w_slot;
    size_t          w_tok;
    int             E, K, T;
};

static void moe_combine_launch(const moe_combine_args & a, float * out, dpct::queue_ptr stream) {
    const size_t           total = (size_t) a.E * a.T;
    const moe_combine_args k     = a;
    stream->parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
        const size_t i = idx[0];
        const int    e = (int) (i % k.E);
        const int    t = (int) (i / k.E);
        float        acc = 0.0f;
        for (int s = 0; s < k.K; ++s) {
            const float sc = k.scale ? k.scale[k.ids[(size_t) t * k.ids_tok + s]] : 1.0f;
            const float v  = (k.y[(size_t) t * k.y_tok + (size_t) s * k.y_slot + e] * sc) * k.w[(size_t) t * k.w_tok + (size_t) s * k.w_slot];
            acc = s == 0 ? v : acc + v;
        }
        out[i] = acc;
    });
}

static bool moe_glue_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t as = (uintptr_t) a->data;
    const uintptr_t ae = as + ggml_nbytes(a);
    const uintptr_t bs = (uintptr_t) b->data;
    const uintptr_t be = bs + ggml_nbytes(b);
    return !(ae <= bs || be <= as);
}

static int moe_glue_index(const ggml_cgraph * g, const ggml_tensor * t, int from, int to) {
    for (int j = from; j < to && j < g->n_nodes; ++j) {
        if (g->nodes[j] == t) {
            return j;
        }
    }
    return -1;
}

static bool moe_glue_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE ||
           t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE;
}

// the next node from j that is not a view; -1 past the window
static int moe_glue_next(const ggml_cgraph * g, int j, int limit) {
    for (; j < g->n_nodes && j < limit; ++j) {
        if (!moe_glue_is_view(g->nodes[j])) {
            return j;
        }
    }
    return -1;
}

static bool moe_glue_unary_is(const ggml_tensor * t, ggml_unary_op op) {
    return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
}

static float moe_glue_scale_of(const ggml_tensor * t) {
    float s;
    memcpy(&s, t->op_params, sizeof(float));
    return s;
}

static bool moe_glue_scale_ok(const ggml_tensor * t) {
    float b;
    memcpy(&b, (const float *) t->op_params + 1, sizeof(float));
    return t->op == GGML_OP_SCALE && b == 0.0f;
}

static bool moe_glue_used_once(const ggml_cgraph * g, int idx, int n) {
    const ggml_tensor * t = g->nodes[idx];
    return ggml_node_get_use_count(g, idx) == n && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
}

// REPEAT(scale) -> GET_ROWS(ids) -> MUL(x, .): the per-expert scale of a MUL_MAT_ID output x
struct moe_scaled {
    const ggml_tensor * x;
    const ggml_tensor * scale; // the [1, n_expert] scale tensor (the REPEAT's source)
    const ggml_tensor * ids;
    const ggml_tensor * mul;
    int                 i_mul;
};

static bool moe_glue_match_scaled(const ggml_cgraph * g, int i, moe_scaled & m) {
    const ggml_tensor * rep = g->nodes[i];
    const int           ig  = moe_glue_next(g, i + 1, i + 4);
    if (ig < 0) {
        return false;
    }
    const ggml_tensor * gr = g->nodes[ig];
    const int           im = moe_glue_next(g, ig + 1, ig + 4);
    if (im < 0) {
        return false;
    }
    const ggml_tensor * mul = g->nodes[im];
    if (rep->op != GGML_OP_REPEAT || gr->op != GGML_OP_GET_ROWS || gr->src[0] != rep || mul->op != GGML_OP_MUL ||
        mul->src[1] != gr) {
        return false;
    }
    const ggml_tensor * x     = mul->src[0];
    const ggml_tensor * scale = rep->src[0];
    const ggml_tensor * ids   = gr->src[1];
    if (x->op != GGML_OP_MUL_MAT_ID || x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || x->ne[3] != 1 ||
        scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || scale->ne[0] != 1 ||
        ggml_nelements(scale) != scale->ne[1] || ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
        ids->ne[0] != x->ne[1] || ids->ne[1] != x->ne[2] || rep->ne[0] != 1 || rep->ne[1] != scale->ne[1] ||
        gr->ne[0] != 1 || gr->ne[1] != x->ne[1] || gr->ne[2] != x->ne[2]) {
        return false;
    }
    if (!moe_glue_used_once(g, i, 1) || !moe_glue_used_once(g, ig, 1)) {
        return false;
    }
    m = { x, scale, ids, mul, im };
    return true;
}

static int moe_glue_situ(ggml_backend_sycl_context & ctx, const ggml_cgraph * g, int i, const moe_scaled & m) {
    const ggml_tensor * gu = m.mul;
    const int64_t       F  = gu->ne[0] / 2;
    // [a1 a2 a3 sg act u1 u2 u3 situ], views in between
    int                 idx[9];
    int                 j = m.i_mul + 1;
    for (int n = 0; n < 9; ++n) {
        j = moe_glue_next(g, j, m.i_mul + 40);
        if (j < 0) {
            return 0;
        }
        idx[n] = j++;
    }
    const ggml_tensor * a1   = g->nodes[idx[0]];
    const ggml_tensor * a2   = g->nodes[idx[1]];
    const ggml_tensor * a3   = g->nodes[idx[2]];
    const ggml_tensor * sg   = g->nodes[idx[3]];
    const ggml_tensor * act  = g->nodes[idx[4]];
    const ggml_tensor * u1   = g->nodes[idx[5]];
    const ggml_tensor * u2   = g->nodes[idx[6]];
    const ggml_tensor * u3   = g->nodes[idx[7]];
    const ggml_tensor * situ = g->nodes[idx[8]];
    const ggml_tensor * gate = a1->src[0];
    const ggml_tensor * up   = u1->src[0];
    if (!moe_glue_scale_ok(a1) || !moe_glue_unary_is(a2, GGML_UNARY_OP_TANH) || a2->src[0] != a1 ||
        !moe_glue_scale_ok(a3) || a3->src[0] != a2 || !moe_glue_unary_is(sg, GGML_UNARY_OP_SIGMOID) ||
        sg->src[0] != gate || act->op != GGML_OP_MUL || act->src[0] != a3 || act->src[1] != sg ||
        !moe_glue_scale_ok(u1) || !moe_glue_unary_is(u2, GGML_UNARY_OP_TANH) || u2->src[0] != u1 ||
        !moe_glue_scale_ok(u3) || u3->src[0] != u2 || situ->op != GGML_OP_MUL || situ->src[0] != act ||
        situ->src[1] != u3) {
        return 0;
    }
    // gate / up: the halves of the scaled gate_up output
    if (gate->op != GGML_OP_VIEW || gate->view_src != gu || gate->view_offs != 0 || up->op != GGML_OP_VIEW ||
        up->view_src != gu || up->view_offs != (size_t) F * sizeof(float) || gate->ne[0] != F || up->ne[0] != F ||
        gu->type != GGML_TYPE_F32 || !ggml_is_contiguous(gu) || situ->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(situ) || situ->ne[0] != F || situ->ne[1] != gu->ne[1] || situ->ne[2] != gu->ne[2] ||
        situ->ne[3] != 1) {
        return 0;
    }
    // every node of the chain is read only inside it: gate_up by its two views, gate by a1 and sg
    if (!moe_glue_used_once(g, m.i_mul, 2)) {
        return 0;
    }
    const int ig = moe_glue_index(g, gate, m.i_mul + 1, idx[0]);
    const int iu = moe_glue_index(g, up, idx[4] + 1, idx[5]);
    if (ig < 0 || iu < 0 || !moe_glue_used_once(g, ig, 2) || !moe_glue_used_once(g, iu, 1)) {
        return 0;
    }
    for (int n = 0; n < 8; ++n) {
        if (!moe_glue_used_once(g, idx[n], 1)) {
            return 0;
        }
    }

    moe_situ_args a;
    a.x       = (const float *) m.x->data;
    a.x_slot  = m.x->nb[1] / sizeof(float);
    a.x_tok   = m.x->nb[2] / sizeof(float);
    a.scale   = (const float *) m.scale->data;
    a.ids     = (const int32_t *) m.ids->data;
    a.ids_tok = m.ids->nb[1] / sizeof(int32_t);
    a.out     = (float *) situ->data;
    a.F       = (int) F;
    a.K       = (int) gu->ne[1];
    a.T       = (int) gu->ne[2];
    a.s1      = moe_glue_scale_of(a1);
    a.s3      = moe_glue_scale_of(a3);
    a.u1      = moe_glue_scale_of(u1);
    a.u3      = moe_glue_scale_of(u3);

    scope_op_debug_print scope_dbg_print(__func__, situ, /*num_src=*/2);
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const bool alias = moe_glue_overlap(situ, m.x) || moe_glue_overlap(situ, m.ids);
    if (alias) {
        ggml_sycl_pool_alloc<float> tmp(ctx.pool(), ggml_nelements(situ));
        moe_situ_launch(a, tmp.get(), ctx.stream());
        ctx.stream()->memcpy(situ->data, tmp.get(), ggml_nbytes(situ));
    } else {
        moe_situ_launch(a, (float *) situ->data, ctx.stream());
    }
    return idx[8] - i;
}

static int moe_glue_combine(ggml_backend_sycl_context & ctx, const ggml_cgraph * g, int i, const moe_scaled & m) {
    const ggml_tensor * d  = m.mul;
    const int           iw = moe_glue_next(g, m.i_mul + 1, m.i_mul + 4);
    if (iw < 0) {
        return 0;
    }
    const ggml_tensor * wt = g->nodes[iw];
    const int64_t       E  = d->ne[0];
    const int64_t       K  = d->ne[1];
    const int64_t       T  = d->ne[2];
    if (wt->op != GGML_OP_MUL || wt->src[0] != d || K < 2) {
        return 0;
    }
    const ggml_tensor * w = wt->src[1];
    if (w->type != GGML_TYPE_F32 || w->ne[0] != 1 || w->ne[1] != K || w->ne[2] != T || w->ne[3] != 1 ||
        d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || wt->type != GGML_TYPE_F32 || !ggml_is_contiguous(wt)) {
        return 0;
    }
    // the ADD chain v0 + v1 + ... + v(K-1) over the K views of the weighted experts; the views (no-ops) may sit
    // before the adds (build_moe_ffn) or between them
    auto view_ok = [&](const ggml_tensor * v, int64_t k) {
        if (v->op != GGML_OP_VIEW || v->view_src != wt || v->view_offs != (size_t) k * wt->nb[1] || v->ne[0] != E ||
            v->ne[1] != T || v->nb[1] != wt->nb[2]) {
            return false;
        }
        const int iv = moe_glue_index(g, v, iw + 1, iw + 4 * (int) K + 4);
        return iv >= 0 && moe_glue_used_once(g, iv, 1);
    };
    const ggml_tensor * prev = nullptr;
    int                 ia   = iw;
    for (int64_t k = 1; k < K; ++k) {
        ia = moe_glue_next(g, ia + 1, iw + 4 * (int) K + 4);
        if (ia < 0) {
            return 0;
        }
        const ggml_tensor * add = g->nodes[ia];
        if (add->op != GGML_OP_ADD || !view_ok(add->src[1], k) ||
            (k == 1 ? !view_ok(add->src[0], 0) : add->src[0] != prev) || (k < K - 1 && !moe_glue_used_once(g, ia, 1))) {
            return 0;
        }
        prev = add;
    }
    const ggml_tensor * out = prev;
    if (out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || out->ne[0] != E || out->ne[1] != T ||
        !moe_glue_used_once(g, m.i_mul, 1) || !moe_glue_used_once(g, iw, (int) K)) {
        return 0;
    }

    moe_combine_args a;
    a.y       = (const float *) m.x->data;
    a.y_slot  = m.x->nb[1] / sizeof(float);
    a.y_tok   = m.x->nb[2] / sizeof(float);
    a.scale   = (const float *) m.scale->data;
    a.ids     = (const int32_t *) m.ids->data;
    a.ids_tok = m.ids->nb[1] / sizeof(int32_t);
    a.w       = (const float *) w->data;
    a.w_slot  = w->nb[1] / sizeof(float);
    a.w_tok   = w->nb[2] / sizeof(float);
    a.E       = (int) E;
    a.K       = (int) K;
    a.T       = (int) T;
    if (w->nb[0] != sizeof(float) || w->nb[1] % sizeof(float) || w->nb[2] % sizeof(float)) {
        return 0;
    }

    scope_op_debug_print scope_dbg_print(__func__, out, /*num_src=*/2);
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const bool alias = moe_glue_overlap(out, m.x) || moe_glue_overlap(out, m.ids) || moe_glue_overlap(out, w);
    if (alias) {
        ggml_sycl_pool_alloc<float> tmp(ctx.pool(), ggml_nelements(out));
        moe_combine_launch(a, tmp.get(), ctx.stream());
        ctx.stream()->memcpy(out->data, tmp.get(), ggml_nbytes(out));
    } else {
        moe_combine_launch(a, (float *) out->data, ctx.stream());
    }
    return ia - i;
}

int ggml_sycl_try_moe_glue_fusion(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool enabled = g_ggml_sycl_enable_fusion && ggml_sycl_get_env("GGML_SYCL_MOE_GLUE", 1) != 0;
    if (!enabled || cgraph->nodes[i]->op != GGML_OP_REPEAT) {
        return 0;
    }
    moe_scaled m;
    if (!moe_glue_match_scaled(cgraph, i, m)) {
        return 0;
    }
    // gate_up (twice the width of its views) or down
    const int n = moe_glue_situ(ctx, cgraph, i, m);
    return n > 0 ? n : moe_glue_combine(ctx, cgraph, i, m);
}
