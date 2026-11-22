#include "loop_ctrl.hpp"

#include <cstring>

#include "ggml-impl.h"

// Kernel 1: the down projection as partial dot products. Work-group (r, p) reads quarter p of row r of W_down
// (parts 0 and 1 against cur, 2 and 3 against msg, unnormalized) for every token; the work-groups of row 0 also
// sum the squares of their quarter, so the norm scales are applied in kernel 2 and the CONCAT is never built.
// Kernel 2: one sub-group per output element i. Lane l holds gate entries l + 16 j (j < 4) of every token, rebuilt
// from the partials, and the same entries of rows i of W_forget and W_gain; two sub-group sums give the tanh
// arguments. The reduction order differs from the unfused kernels: the result matches them to rounding.
static constexpr int LC_SG     = 16;
static constexpr int LC_WG1    = 128;
static constexpr int LC_PARTS  = 4;
static constexpr int LC_WG2    = 256;
static constexpr int LC_ROWS2  = LC_WG2 / LC_SG;
static constexpr int LC_MAXT   = 8;
static constexpr int LC_MAXR   = 4 * LC_SG;

struct loop_ctrl_args {
    const float * cur;
    const float * msg;
    const char *  wd;
    const char *  wf;
    const char *  wg;
    float *       out;
    size_t        cur_tok;  // floats between the tokens of cur
    size_t        msg_tok;
    size_t        out_tok;
    size_t        wd_row;   // bytes between rows
    size_t        wf_row;
    size_t        wg_row;
    int           n_embd;
    int           rank;
    int           n_tokens;
    float         eps_c;
    float         eps_m;
};

static void loop_ctrl_launch(const loop_ctrl_args & a, float * part, dpct::queue_ptr stream) {
    // part: [n_tokens][LC_PARTS][rank] dot products, then [n_tokens][LC_PARTS] sums of squares
    float * ssq = part + (size_t) a.n_tokens * LC_PARTS * a.rank;
    {
        const loop_ctrl_args k = a;
        stream->parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t) a.rank, (size_t) LC_PARTS * LC_WG1), sycl::range<2>(1, LC_WG1)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(LC_SG)]] {
                const int r   = (int) it.get_group(0);
                const int p   = (int) it.get_group(1);
                const int tid = (int) it.get_local_id(1);

                const int     half = (k.n_embd + 1) / 2;
                const int     e0   = (p & 1) * half;
                const int     e1   = sycl::min(k.n_embd, e0 + half);
                const float * x    = p < 2 ? k.cur : k.msg;
                const size_t  xs   = p < 2 ? k.cur_tok : k.msg_tok;
                const float * w    = (const float *) (k.wd + (size_t) r * k.wd_row) + (p < 2 ? 0 : k.n_embd);
                const bool    sq   = r == 0;

                float acc[LC_MAXT];
                float ss[LC_MAXT];
#pragma unroll
                for (int t = 0; t < LC_MAXT; ++t) {
                    acc[t] = 0.0f;
                    ss[t]  = 0.0f;
                }
                for (int e = e0 + tid; e < e1; e += LC_WG1) {
                    const float we = w[e];
#pragma unroll
                    for (int t = 0; t < LC_MAXT; ++t) {
                        if (t < k.n_tokens) {
                            const float xe = x[t * xs + e];
                            acc[t] += we * xe;
                            ss[t] += xe * xe;
                        }
                    }
                }
#pragma unroll
                for (int t = 0; t < LC_MAXT; ++t) {
                    if (t < k.n_tokens) {
                        const float s = sycl::reduce_over_group(it.get_group(), acc[t], sycl::plus<float>());
                        if (tid == 0) {
                            part[((size_t) t * LC_PARTS + p) * k.rank + r] = s;
                        }
                        if (sq) {
                            const float q = sycl::reduce_over_group(it.get_group(), ss[t], sycl::plus<float>());
                            if (tid == 0) {
                                ssq[t * LC_PARTS + p] = q;
                            }
                        }
                    }
                }
            });
    }
    {
        const loop_ctrl_args k = a;
        const size_t n_wg = ((size_t) a.n_embd + LC_ROWS2 - 1) / LC_ROWS2;
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(n_wg * LC_WG2), sycl::range<1>(LC_WG2)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(LC_SG)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = (int) sg.get_local_linear_id();
                const int  i    = (int) it.get_group(0) * LC_ROWS2 + (int) sg.get_group_linear_id();
                if (i >= k.n_embd) {
                    return;
                }
                const float * wf = (const float *) (k.wf + (size_t) i * k.wf_row);
                const float * wg = (const float *) (k.wg + (size_t) i * k.wg_row);
                float         f[4];
                float         g[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = lane + j * LC_SG;
                    f[j] = c < k.rank ? wf[c] : 0.0f;
                    g[j] = c < k.rank ? wg[c] : 0.0f;
                }
                const float inv_n = 1.0f / (float) k.n_embd;
#pragma unroll
                for (int t = 0; t < LC_MAXT; ++t) {
                    if (t < k.n_tokens) {
                        const float * pt = part + (size_t) t * LC_PARTS * k.rank;
                        const float * st = ssq + t * LC_PARTS;
                        const float   sc = sycl::rsqrt((st[0] + st[1]) * inv_n + k.eps_c);
                        const float   sm = sycl::rsqrt((st[2] + st[3]) * inv_n + k.eps_m);
                        float         fs = 0.0f;
                        float         gs = 0.0f;
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int c = lane + j * LC_SG;
                            if (c < k.rank) {
                                const float d = sc * (pt[c] + pt[k.rank + c]) + sm * (pt[2 * k.rank + c] + pt[3 * k.rank + c]);
                                fs += f[j] * d;
                                gs += g[j] * d;
                            }
                        }
                        fs = sycl::reduce_over_group(sg, fs, sycl::plus<float>());
                        gs = sycl::reduce_over_group(sg, gs, sycl::plus<float>());
                        if (lane == 0) {
                            const float c  = k.cur[t * k.cur_tok + i];
                            const float m  = k.msg[t * k.msg_tok + i];
                            const float tf = sycl::tanh(fs);
                            const float tg = sycl::tanh(gs);
                            k.out[t * k.out_tok + i] = (c - tf * c) + tg * m;
                        }
                    }
                }
            });
    }
}

// f32 [n_embd, n_tokens] with dense elements, rows at any float stride
static bool loop_ctrl_vec_ok(const ggml_tensor * t, int64_t n_embd, int64_t n_tokens) {
    return t->type == GGML_TYPE_F32 && t->ne[0] == n_embd && t->ne[1] == n_tokens && t->ne[2] == 1 && t->ne[3] == 1 &&
           t->nb[0] == sizeof(float) && t->nb[1] % sizeof(float) == 0 && t->data != nullptr;
}

static bool loop_ctrl_weight_ok(const ggml_tensor * w, int64_t ne0, int64_t ne1) {
    return w->type == GGML_TYPE_F32 && w->ne[0] == ne0 && w->ne[1] == ne1 && w->ne[2] == 1 && w->ne[3] == 1 &&
           w->nb[0] == sizeof(float) && w->nb[1] % sizeof(float) == 0 && w->data != nullptr;
}

static bool loop_ctrl_unary_is(const ggml_tensor * t, ggml_unary_op op) {
    return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
}

// out may be cur or msg itself (each element is read before it is written, by the same work-item), never a part
static bool loop_ctrl_alias_ok(const ggml_tensor * out, const ggml_tensor * in) {
    const uintptr_t os = (uintptr_t) out->data;
    const uintptr_t oe = os + ggml_nbytes(out);
    const uintptr_t is = (uintptr_t) in->data;
    const uintptr_t ie = is + ggml_nbytes(in);
    if (oe <= is || ie <= os) {
        return true;
    }
    return os == is && (out->ne[1] == 1 || out->nb[1] == in->nb[1]);
}

// the chain from the CONCAT at index c; on success fills a and sets *rn_c_idx and *out_idx
static bool loop_ctrl_match(const ggml_cgraph * g, const int c, loop_ctrl_args & a, int * rn_c_idx, int * out_idx) {
    if (c < 1 || c + 9 >= g->n_nodes) {
        return false;
    }
    const ggml_tensor * cat  = g->nodes[c];
    const ggml_tensor * rn_m = g->nodes[c - 1];
    if (cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0 || cat->src[1] != rn_m ||
        rn_m->op != GGML_OP_RMS_NORM) {
        return false;
    }
    const ggml_tensor * rn_c = cat->src[0];
    if (rn_c == nullptr || rn_c->op != GGML_OP_RMS_NORM || rn_c == rn_m) {
        return false;
    }
    const ggml_tensor * d   = g->nodes[c + 1];
    const ggml_tensor * mf  = g->nodes[c + 2];
    const ggml_tensor * tf  = g->nodes[c + 3];
    const ggml_tensor * p1  = g->nodes[c + 4];
    const ggml_tensor * s   = g->nodes[c + 5];
    const ggml_tensor * mg  = g->nodes[c + 6];
    const ggml_tensor * tg  = g->nodes[c + 7];
    const ggml_tensor * p2  = g->nodes[c + 8];
    const ggml_tensor * out = g->nodes[c + 9];
    const ggml_tensor * cur = rn_c->src[0];
    const ggml_tensor * msg = rn_m->src[0];
    if (d->op != GGML_OP_MUL_MAT || d->src[1] != cat || mf->op != GGML_OP_MUL_MAT || mf->src[1] != d ||
        !loop_ctrl_unary_is(tf, GGML_UNARY_OP_TANH) || tf->src[0] != mf || p1->op != GGML_OP_MUL ||
        p1->src[0] != tf || p1->src[1] != cur || s->op != GGML_OP_SUB || s->src[0] != cur || s->src[1] != p1 ||
        mg->op != GGML_OP_MUL_MAT || mg->src[1] != d || !loop_ctrl_unary_is(tg, GGML_UNARY_OP_TANH) ||
        tg->src[0] != mg || p2->op != GGML_OP_MUL || p2->src[0] != tg || p2->src[1] != msg ||
        out->op != GGML_OP_ADD || out->src[0] != s || out->src[1] != p2) {
        return false;
    }
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const ggml_tensor * wd = d->src[0];
    const ggml_tensor * wf = mf->src[0];
    const ggml_tensor * wg = mg->src[0];
    const int64_t       rank = wd->ne[1];
    if (n_tokens < 1 || n_tokens > LC_MAXT || rank < 1 || rank > LC_MAXR || !loop_ctrl_vec_ok(cur, n_embd, n_tokens) ||
        !loop_ctrl_vec_ok(msg, n_embd, n_tokens) || !loop_ctrl_vec_ok(out, n_embd, n_tokens) ||
        !ggml_is_contiguous(out) || !loop_ctrl_weight_ok(wd, 2 * n_embd, rank) ||
        !loop_ctrl_weight_ok(wf, rank, n_embd) || !loop_ctrl_weight_ok(wg, rank, n_embd)) {
        return false;
    }
    // RMS_NORM(cur) somewhere before RMS_NORM(msg); every skipped node is read only inside the chain
    int rc = -1;
    for (int j = c - 2; j >= 0 && j >= c - 64; --j) {
        if (g->nodes[j] == rn_c) {
            rc = j;
            break;
        }
    }
    if (rc < 0) {
        return false;
    }
    const int uses_expected[11][2] = { { rc, 1 },    { c - 1, 1 }, { c, 1 },     { c + 1, 2 }, { c + 2, 1 }, { c + 3, 1 },
                                       { c + 4, 1 }, { c + 5, 1 }, { c + 6, 1 }, { c + 7, 1 }, { c + 8, 1 } };
    for (const auto & u : uses_expected) {
        const ggml_tensor * n = g->nodes[u[0]];
        if (ggml_node_get_use_count(g, u[0]) != u[1] || (n->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
    }
    if (!loop_ctrl_alias_ok(out, cur) || !loop_ctrl_alias_ok(out, msg)) {
        return false;
    }

    a.cur      = (const float *) cur->data;
    a.msg      = (const float *) msg->data;
    a.wd       = (const char *) wd->data;
    a.wf       = (const char *) wf->data;
    a.wg       = (const char *) wg->data;
    a.out      = (float *) out->data;
    a.cur_tok  = cur->nb[1] / sizeof(float);
    a.msg_tok  = msg->nb[1] / sizeof(float);
    a.out_tok  = out->nb[1] / sizeof(float);
    a.wd_row   = wd->nb[1];
    a.wf_row   = wf->nb[1];
    a.wg_row   = wg->nb[1];
    a.n_embd   = (int) n_embd;
    a.rank     = (int) rank;
    a.n_tokens = (int) n_tokens;
    memcpy(&a.eps_c, rn_c->op_params, sizeof(float));
    memcpy(&a.eps_m, rn_m->op_params, sizeof(float));
    *rn_c_idx = rc;
    *out_idx  = c + 9;
    return true;
}

int ggml_sycl_try_loop_ctrl_fusion(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool enabled = g_ggml_sycl_enable_fusion && ggml_sycl_get_env("GGML_SYCL_LOOP_CTRL", 1) != 0;
    const ggml_tensor * node = cgraph->nodes[i];
    if (!enabled || node->op != GGML_OP_RMS_NORM) {
        return 0;
    }
    loop_ctrl_args a;
    int            rc;
    int            oi;
    // RMS_NORM(msg): the chain starts at the next node
    if (i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->src[1] == node && loop_ctrl_match(cgraph, i + 1, a, &rc, &oi)) {
        scope_op_debug_print scope_dbg_print(__func__, cgraph->nodes[oi], /*num_src=*/2);
        SYCL_CHECK(ggml_sycl_set_device(ctx.device));
        ggml_sycl_pool_alloc<float> part(ctx.pool(), (size_t) a.n_tokens * LC_PARTS * (a.rank + 1));
        loop_ctrl_launch(a, part.get(), ctx.stream());
        return oi - i;
    }
    // RMS_NORM(cur): its only reader is the CONCAT of a chain that matches
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 64; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_CONCAT && n->src[0] == node) {
            return loop_ctrl_match(cgraph, j, a, &rc, &oi) && rc == i ? -1 : 0;
        }
    }
    return 0;
}
