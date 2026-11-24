#include "conv_glue.hpp"

#include <cstring>

#include "ggml-impl.h"

// out[t, c] = (cur[t, c] + sum_i x[c][t + i] * w[c][i]) * s + b, the taps summed in ascending order as the SSM_CONV
// kernel does, so the result matches the unfused kernels to rounding of the final add and scale only. The output is
// allocated at the SCALE: when it overlaps x or w (freed inside the chain) the kernel writes a pool buffer and copies.
struct conv_glue_args {
    const float * x;      // SSM_CONV source [n_conv - 1 + T, C]
    size_t        x_ch;   // floats between channels
    const float * w;      // [d_conv, C] contiguous
    const float * cur;    // residual [C, T]
    size_t        cur_tok;
    float *       out;    // [C, T] contiguous
    int           C, T, d_conv;
    float         s, b;
};

template <int DC> static void conv_glue_launch(const conv_glue_args & a, float * out, dpct::queue_ptr stream) {
    const conv_glue_args k = a;
    // a (token, channel) range: no 64-bit division of a flat index
    stream->parallel_for(sycl::range<2>((size_t) a.T, (size_t) a.C), [=](sycl::id<2> idx) {
        const int     t  = (int) idx[0];
        const int     c  = (int) idx[1];
        const float * xs = k.x + (size_t) c * k.x_ch + t;
        const float * wc = k.w + (size_t) c * (DC > 0 ? DC : k.d_conv);
        float         sumf = 0.0f;
        if constexpr (DC > 0) {
#pragma unroll
            for (int i = 0; i < DC; ++i) {
                sumf += xs[i] * wc[i];
            }
        } else {
            for (int i = 0; i < k.d_conv; ++i) {
                sumf += xs[i] * wc[i];
            }
        }
        out[(size_t) t * k.C + c] = (k.cur[(size_t) t * k.cur_tok + c] + sumf) * k.s + k.b;
    });
}

static bool conv_glue_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t as = (uintptr_t) a->data;
    const uintptr_t ae = as + ggml_nbytes(a);
    const uintptr_t bs = (uintptr_t) b->data;
    const uintptr_t be = bs + ggml_nbytes(b);
    return !(ae <= bs || be <= as);
}

static bool conv_glue_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE ||
           t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE;
}

static int conv_glue_index(const ggml_cgraph * g, const ggml_tensor * t, int from, int to) {
    for (int j = from; j < to && j < g->n_nodes; ++j) {
        if (g->nodes[j] == t) {
            return j;
        }
    }
    return -1;
}

static bool conv_glue_used_once(const ggml_cgraph * g, int idx) {
    return idx >= 0 && ggml_node_get_use_count(g, idx) == 1 && !(g->nodes[idx]->flags & GGML_TENSOR_FLAG_OUTPUT);
}

int ggml_sycl_try_conv_glue_fusion(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool enabled = g_ggml_sycl_enable_fusion && ggml_sycl_get_env("GGML_SYCL_CONV_GLUE", 1) != 0;
    const ggml_tensor * conv  = cgraph->nodes[i];
    if (!enabled || conv->op != GGML_OP_SSM_CONV) {
        return 0;
    }
    // the next non-view nodes: ADD(cur, view(conv)), then SCALE(add)
    int ia = i + 1;
    while (ia < cgraph->n_nodes && ia < i + 4 && conv_glue_is_view(cgraph->nodes[ia])) {
        ++ia;
    }
    if (ia + 1 >= cgraph->n_nodes) {
        return 0;
    }
    int is = ia + 1;
    while (is < cgraph->n_nodes && is < ia + 4 && conv_glue_is_view(cgraph->nodes[is])) {
        ++is;
    }
    if (is >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * add = cgraph->nodes[ia];
    const ggml_tensor * sc  = cgraph->nodes[is];
    if (add->op != GGML_OP_ADD || sc->op != GGML_OP_SCALE || sc->src[0] != add) {
        return 0;
    }
    const ggml_tensor * cv  = add->src[1];
    const ggml_tensor * cur = add->src[0];
    if (cv != conv && !(conv_glue_is_view(cv) && cv->view_src == conv && cv->view_offs == 0)) {
        return 0;
    }
    const ggml_tensor * x = conv->src[0];
    const ggml_tensor * w = conv->src[1];
    const int64_t       C = conv->ne[0];
    const int64_t       T = conv->ne[1];
    if (conv->ne[2] != 1 || conv->type != GGML_TYPE_F32 || !ggml_is_contiguous(conv) || x->type != GGML_TYPE_F32 ||
        x->nb[0] != sizeof(float) || x->ne[1] != C || x->ne[2] != 1 || w->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(w) || w->ne[1] != C || x->ne[0] != w->ne[0] - 1 + T || cur->type != GGML_TYPE_F32 ||
        cur->ne[0] != C || cur->ne[1] != T || cur->ne[2] != 1 || cur->ne[3] != 1 || cur->nb[0] != sizeof(float) ||
        sc->type != GGML_TYPE_F32 || !ggml_is_contiguous(sc) || sc->ne[0] != C || sc->ne[1] != T) {
        return 0;
    }
    // conv, its view and the ADD are read only inside the chain
    const int iv = cv == conv ? -1 : conv_glue_index(cgraph, cv, i + 1, ia);
    if (!conv_glue_used_once(cgraph, i) || (cv != conv && !conv_glue_used_once(cgraph, iv)) ||
        !conv_glue_used_once(cgraph, ia)) {
        return 0;
    }

    conv_glue_args a;
    a.x       = (const float *) x->data;
    a.x_ch    = x->nb[1] / sizeof(float);
    a.w       = (const float *) w->data;
    a.cur     = (const float *) cur->data;
    a.cur_tok = cur->nb[1] / sizeof(float);
    a.C       = (int) C;
    a.T       = (int) T;
    a.d_conv  = (int) w->ne[0];
    memcpy(&a.s, (const float *) sc->op_params + 0, sizeof(float));
    memcpy(&a.b, (const float *) sc->op_params + 1, sizeof(float));

    scope_op_debug_print scope_dbg_print(__func__, sc, /*num_src=*/2);
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    // the same element of cur is read before it is written by the same work-item; any other overlap goes via a copy
    const bool same_as_cur = sc->data == cur->data && (T == 1 || cur->nb[1] == sc->nb[1]);
    const bool alias = conv_glue_overlap(sc, x) || conv_glue_overlap(sc, w) || (!same_as_cur && conv_glue_overlap(sc, cur));
    float *    dst   = (float *) sc->data;
    ggml_sycl_pool_alloc<float> tmp(ctx.pool());
    if (alias) {
        dst = tmp.alloc(ggml_nelements(sc));
    }
    if (a.d_conv == 4) {
        conv_glue_launch<4>(a, dst, ctx.stream());
    } else {
        conv_glue_launch<0>(a, dst, ctx.stream());
    }
    if (alias) {
        ctx.stream()->memcpy(sc->data, dst, ggml_nbytes(sc));
    }
    return is - i;
}

// CONCAT(past, x) on dim 0 -> its state store, in one kernel: the window x_cat = [past | x] (read by the SSM_CONV
// later) and the last n_cols columns of it into the recurrent-state cache, either one CPY of the last columns
// (build_conv_state_store without rollback) or SET_ROWS of the rollback windows into their slots. Work-item per
// element (tau, c, s) of x_cat; the store writes the persistent cache, which follows the CONCAT in the graph.
struct conv_store_args {
    const char *    past;   // [n_cols, C, S]
    size_t          p0, p1, p2;
    const char *    xt;     // [T, C, S] (a transposed view)
    size_t          x0, x1, x2;
    float *         xcat;   // [n_cols + T, C, S] contiguous
    char *          dst;    // CPY: [n_cols * C, S]; SET_ROWS: [n_cols, K, C, S]
    size_t          d0, d1, d2, d3;
    const int32_t * slot;   // SET_ROWS: the slot of window w; nullptr for CPY
    int             n_cols, T, C, S, Kw, start;
};

static void conv_store_launch(const conv_store_args & a, dpct::queue_ptr stream) {
    const conv_store_args k = a;
    const int             W = a.n_cols + a.T;
    // a (sequence, channel, column) range: no 64-bit division of a flat index
    stream->parallel_for(sycl::range<3>((size_t) a.S, (size_t) a.C, (size_t) W), [=](sycl::id<3> id3) {
        const int    s   = (int) id3[0];
        const int    c   = (int) id3[1];
        const int    tau = (int) id3[2];
        const size_t idx = ((size_t) s * k.C + c) * W + tau;
        const float v = tau < k.n_cols ? *(const float *) (k.past + tau * k.p0 + c * k.p1 + s * k.p2)
                                       : *(const float *) (k.xt + (tau - k.n_cols) * k.x0 + c * k.x1 + s * k.x2);
        k.xcat[idx] = v;
        if (k.slot == nullptr) {
            const int j = tau - k.T; // the last n_cols columns, in the CPY's element order (j fastest, then c)
            if (j >= 0) {
                *(float *) (k.dst + ((size_t) c * k.n_cols + j) * k.d0 + s * k.d1) = v;
            }
        } else {
            for (int w = 0; w < k.Kw; ++w) {
                const int j = tau - k.start - w;
                if (j >= 0 && j < k.n_cols) {
                    *(float *) (k.dst + j * k.d0 + (size_t) k.slot[w] * k.d1 + c * k.d2 + s * k.d3) = v;
                }
            }
        }
    });
}

int ggml_sycl_try_conv_store_fusion(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool enabled = g_ggml_sycl_enable_fusion && ggml_sycl_get_env("GGML_SYCL_CONV_GLUE", 1) != 0;
    const ggml_tensor * cat   = cgraph->nodes[i];
    if (!enabled || cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0) {
        return 0;
    }
    // the store: the next non-view node
    int is = i + 1;
    while (is < cgraph->n_nodes && is < i + 6 && conv_glue_is_view(cgraph->nodes[is])) {
        ++is;
    }
    if (is >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * st   = cgraph->nodes[is];
    const ggml_tensor * past = cat->src[0];
    const ggml_tensor * xt   = cat->src[1];
    const ggml_tensor * win  = st->src[0];
    const int64_t       nc   = past->ne[0];
    const int64_t       T    = xt->ne[0];
    const int64_t       C    = cat->ne[1];
    const int64_t       S    = cat->ne[2];
    if (cat->type != GGML_TYPE_F32 || !ggml_is_contiguous(cat) || cat->ne[3] != 1 || past->type != GGML_TYPE_F32 ||
        xt->type != GGML_TYPE_F32 || past->ne[1] != C || xt->ne[1] != C || past->ne[2] != S || xt->ne[2] != S ||
        win == nullptr || !conv_glue_is_view(win) || win->view_src != cat || win->type != GGML_TYPE_F32 || nc < 1 ||
        nc > 8) {
        return 0;
    }
    conv_store_args a;
    a.past = (const char *) past->data;  a.p0 = past->nb[0]; a.p1 = past->nb[1]; a.p2 = past->nb[2];
    a.xt   = (const char *) xt->data;    a.x0 = xt->nb[0];   a.x1 = xt->nb[1];   a.x2 = xt->nb[2];
    a.xcat = (float *) cat->data;
    a.n_cols = (int) nc;
    a.T      = (int) T;
    a.C      = (int) C;
    a.S      = (int) S;
    const ggml_tensor * dstt = st->op == GGML_OP_CPY ? st->src[1] : st->src[0] == win ? st : nullptr;
    if (st->op == GGML_OP_CPY) {
        // the last nc columns of every channel, element order (j, c) into a [nc * C, S] cell view
        if (dstt == nullptr || dstt->type != GGML_TYPE_F32 || win->ne[0] != nc || win->ne[1] != C || win->ne[2] != S ||
            win->view_offs != (size_t) T * sizeof(float) || win->nb[1] != cat->nb[1] || win->nb[2] != cat->nb[2] ||
            dstt->ne[0] != nc * C || dstt->ne[1] != S || dstt->nb[0] != sizeof(float)) {
            return 0;
        }
        a.dst  = (char *) dstt->data;
        a.d0   = dstt->nb[0];
        a.d1   = dstt->nb[1];
        a.d2   = a.d3 = 0;
        a.slot = nullptr;
        a.Kw   = 0;
        a.start = 0;
    } else if (st->op == GGML_OP_SET_ROWS) {
        // windows [nc, Kw, C, S] one element apart from column start, into slots [nc, K, C, S] of the cache
        const ggml_tensor * idx = st->src[1];
        if (st->type != GGML_TYPE_F32 || idx->type != GGML_TYPE_I32 || idx->nb[0] != sizeof(int32_t) ||
            win->ne[0] != nc || win->ne[2] != C || win->ne[3] != S || win->nb[0] != sizeof(float) ||
            win->nb[1] != sizeof(float) || win->nb[2] != cat->nb[1] || win->nb[3] != cat->nb[2] ||
            win->view_offs % sizeof(float) != 0 || idx->ne[0] != win->ne[1] || st->ne[0] != nc || st->ne[2] != C ||
            st->ne[3] != S) {
            return 0;
        }
        a.dst   = (char *) st->data;
        a.d0    = st->nb[0];
        a.d1    = st->nb[1];
        a.d2    = st->nb[2];
        a.d3    = st->nb[3];
        a.slot  = (const int32_t *) idx->data;
        a.Kw    = (int) win->ne[1];
        a.start = (int) (win->view_offs / sizeof(float));
        if (a.start + a.Kw - 1 + nc > nc + T || a.Kw > 8) {
            return 0;
        }
    } else {
        return 0;
    }
    if (st->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return 0;
    }
    scope_op_debug_print scope_dbg_print(__func__, cat, /*num_src=*/2);
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    conv_store_launch(a, ctx.stream());
    return is - i;
}
