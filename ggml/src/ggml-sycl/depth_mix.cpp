#include "depth_mix.hpp"

#include <cstring>
#include <type_traits>

#include "ggml-impl.h"

// One work-group per query column. Every work-item takes a strided slice of the n_embd elements and accumulates
// its part of all row scores (and of the sum of squares of cur, whose normalized row is the last state) at once,
// so the loads of all rows are in flight together; the sub-groups reduce their parts into local memory and a
// second step adds the sub-group parts. Then every work-item takes the softmax and writes its elements of the
// weighted sum. The rows are [stack rows..., row(cur)] as CONCAT(stack, row) lays them out; the reduction order
// differs from the unfused kernels, so the result matches them to rounding, not bit for bit.
static constexpr int DEPTH_MIX_WG       = 512;
static constexpr int DEPTH_MIX_SG       = 16;
static constexpr int DEPTH_MIX_NSG      = DEPTH_MIX_WG / DEPTH_MIX_SG;
static constexpr int DEPTH_MIX_MAX_ROWS = 24;
static constexpr int DEPTH_MIX_SLOTS    = DEPTH_MIX_MAX_ROWS + 1; // + the sum of squares of cur

static void depth_mix_launch(const float * cur, const float * stack, const size_t stack_row_stride, const int n_stack,
                             const char * q, const size_t q_col_stride, const int n_q, float * out, const int n_embd,
                             const float eps, const float sm_scale, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(DEPTH_MIX_NSG * DEPTH_MIX_SLOTS), cgh);
        sycl::local_accessor<float, 1> tot(sycl::range<1>(DEPTH_MIX_SLOTS), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) n_q * DEPTH_MIX_WG), sycl::range<1>(DEPTH_MIX_WG)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(DEPTH_MIX_SG)]] {
                const auto    sg     = it.get_sub_group();
                const int     sgid   = (int) sg.get_group_linear_id();
                const int     lane   = (int) sg.get_local_linear_id();
                const int     tid    = (int) it.get_local_id(0);
                const int     qi     = (int) it.get_group(0);
                const float * qv     = (const float *) (q + (size_t) qi * q_col_stride);
                const int     n_rows = n_stack + 1;

                float acc[DEPTH_MIX_SLOTS];
#pragma unroll
                for (int b = 0; b < DEPTH_MIX_SLOTS; ++b) {
                    acc[b] = 0.0f;
                }
                for (int e = tid; e < n_embd; e += DEPTH_MIX_WG) {
                    const float qe = qv[e];
                    const float x  = cur[e];
                    acc[DEPTH_MIX_MAX_ROWS] += x * x;
                    acc[n_stack] += x * qe;
#pragma unroll
                    for (int b = 0; b < DEPTH_MIX_MAX_ROWS; ++b) {
                        if (b < n_stack) {
                            acc[b] += stack[(size_t) b * stack_row_stride + e] * qe;
                        }
                    }
                }
#pragma unroll
                for (int b = 0; b < DEPTH_MIX_SLOTS; ++b) {
                    if (b < n_rows || b == DEPTH_MIX_MAX_ROWS) {
                        const float r = sycl::reduce_over_group(sg, acc[b], sycl::plus<float>());
                        if (lane == 0) {
                            part[sgid * DEPTH_MIX_SLOTS + b] = r;
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
                // row totals: sub-group r adds the NSG parts of slot r across its lanes
                static_assert(DEPTH_MIX_NSG == 2 * DEPTH_MIX_SG && DEPTH_MIX_SLOTS <= DEPTH_MIX_NSG, "tile");
                if (sgid < DEPTH_MIX_SLOTS) {
                    const float t = part[lane * DEPTH_MIX_SLOTS + sgid] + part[(lane + DEPTH_MIX_SG) * DEPTH_MIX_SLOTS + sgid];
                    const float r = sycl::reduce_over_group(sg, t, sycl::plus<float>());
                    if (lane == 0) {
                        tot[sgid] = r;
                    }
                }
                sycl::group_barrier(it.get_group());

                // softmax over the rows in sub-group 0: lane l holds rows l and l + SG; weights go to part[]
                const float scale = sycl::rsqrt(tot[DEPTH_MIX_MAX_ROWS] / n_embd + eps);
                if (sgid == 0) {
                    static_assert(DEPTH_MIX_MAX_ROWS <= 2 * DEPTH_MIX_SG, "two rows per lane");
                    const int   r0   = lane;
                    const int   r1   = lane + DEPTH_MIX_SG;
                    const float s0   = r0 < n_rows ? (r0 == n_stack ? tot[r0] * scale : tot[r0]) * sm_scale : -INFINITY;
                    const float s1   = r1 < n_rows ? (r1 == n_stack ? tot[r1] * scale : tot[r1]) * sm_scale : -INFINITY;
                    const float smax = sycl::reduce_over_group(sg, sycl::fmax(s0, s1), sycl::maximum<float>());
                    const float e0   = r0 < n_rows ? sycl::exp(s0 - smax) : 0.0f;
                    const float e1   = r1 < n_rows ? sycl::exp(s1 - smax) : 0.0f;
                    const float inv  = 1.0f / sycl::reduce_over_group(sg, e0 + e1, sycl::plus<float>());
                    if (r0 < n_rows) {
                        part[r0] = e0 * inv;
                    }
                    if (r1 < n_rows) {
                        part[r1] = e1 * inv;
                    }
                }
                sycl::group_barrier(it.get_group());

                const float w_cur = part[n_stack] * scale;
                for (int e = tid; e < n_embd; e += DEPTH_MIX_WG) {
                    float o = 0.0f;
                    for (int b = 0; b < n_stack; ++b) {
                        o += stack[(size_t) b * stack_row_stride + e] * part[b];
                    }
                    o += cur[e] * w_cur;
                    out[(size_t) qi * n_embd + e] = o;
                }
            });
    });
}

// Many tokens: one work-group per token. Work-item tid holds elements tid + k * wg (k < EPT) of all NR state rows
// in registers, so every row is read from memory once for the scores of all n_q queries and for the weighted
// sums. The sub-group parts of the scores go to local memory, the first work-items add them up, and every
// work-item then takes the softmax of each query and writes its elements of that query's output.
static constexpr int DEPTH_MIX_T_EPT  = 8;
static constexpr int DEPTH_MIX_T_MAXR = 8;
static constexpr int DEPTH_MIX_T_MAXQ = 4;

template <int NR>
static void depth_mix_tokens_launch(const float * cur, const size_t cur_tok, const float * stack, const size_t stack_row,
                                    const size_t stack_tok, const char * q, const size_t q_col, const size_t q_tok,
                                    const int n_q, float * out, const size_t out_col, const size_t out_tok,
                                    const int n_embd, const int n_tokens, const float eps, const float sm_scale,
                                    dpct::queue_ptr stream) {
    constexpr int EPT   = DEPTH_MIX_T_EPT;
    constexpr int NS    = NR - 1;                    // stack rows; row NS is cur, normalized
    constexpr int SLOTS = NR * DEPTH_MIX_T_MAXQ + 1; // the scores of every query + the sum of squares of cur
    constexpr int SSQ   = SLOTS - 1;
    const int     wg    = n_embd / EPT;
    const int     nsg   = wg / DEPTH_MIX_SG;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) nsg * SLOTS), cgh);
        sycl::local_accessor<float, 1> tot(sycl::range<1>(SLOTS), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) n_tokens * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(DEPTH_MIX_SG)]] {
                const auto    sg   = it.get_sub_group();
                const int     sgid = (int) sg.get_group_linear_id();
                const int     lane = (int) sg.get_local_linear_id();
                const int     tid  = (int) it.get_local_id(0);
                const size_t  t    = it.get_group(0);
                const float * ct   = cur + t * cur_tok;
                const float * st   = stack + t * stack_tok;
                const char *  qt   = q + t * q_tok;

                float x[NR][EPT];
#pragma unroll
                for (int k = 0; k < EPT; ++k) {
                    const int e = tid + k * wg;
#pragma unroll
                    for (int r = 0; r < NS; ++r) {
                        x[r][k] = st[(size_t) r * stack_row + e];
                    }
                    x[NS][k] = ct[e];
                }

                float acc[SLOTS];
#pragma unroll
                for (int s = 0; s < SLOTS; ++s) {
                    acc[s] = 0.0f;
                }
#pragma unroll
                for (int k = 0; k < EPT; ++k) {
                    acc[SSQ] += x[NS][k] * x[NS][k];
                }
#pragma unroll
                for (int qi = 0; qi < DEPTH_MIX_T_MAXQ; ++qi) {
                    if (qi < n_q) {
                        const float * qv = (const float *) (qt + (size_t) qi * q_col);
#pragma unroll
                        for (int k = 0; k < EPT; ++k) {
                            const float qe = qv[tid + k * wg];
#pragma unroll
                            for (int r = 0; r < NR; ++r) {
                                acc[qi * NR + r] += x[r][k] * qe;
                            }
                        }
                    }
                }
#pragma unroll
                for (int s = 0; s < SLOTS; ++s) {
                    if (s < n_q * NR || s == SSQ) {
                        const float r = sycl::reduce_over_group(sg, acc[s], sycl::plus<float>());
                        if (lane == 0) {
                            part[sgid * SLOTS + s] = r;
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
                for (int s = tid; s < SLOTS; s += wg) {  // a small work-group has fewer work-items than slots
                    if (s < n_q * NR || s == SSQ) {
                        float sum = 0.0f;
                        for (int g = 0; g < nsg; ++g) {
                            sum += part[g * SLOTS + s];
                        }
                        tot[s] = sum;
                    }
                }
                sycl::group_barrier(it.get_group());

                const float scale = sycl::rsqrt(tot[SSQ] / n_embd + eps);
#pragma unroll
                for (int qi = 0; qi < DEPTH_MIX_T_MAXQ; ++qi) {
                    if (qi < n_q) {
                        float w[NR];
                        float smax = -INFINITY;
#pragma unroll
                        for (int r = 0; r < NR; ++r) {
                            w[r] = (r == NS ? tot[qi * NR + r] * scale : tot[qi * NR + r]) * sm_scale;
                            smax = sycl::fmax(smax, w[r]);
                        }
                        float sum = 0.0f;
#pragma unroll
                        for (int r = 0; r < NR; ++r) {
                            w[r] = sycl::exp(w[r] - smax);
                            sum += w[r];
                        }
                        const float inv = 1.0f / sum;
#pragma unroll
                        for (int r = 0; r < NR; ++r) {
                            w[r] *= r == NS ? inv * scale : inv;
                        }
                        float * o = out + t * out_tok + (size_t) qi * out_col;
#pragma unroll
                        for (int k = 0; k < EPT; ++k) {
                            float v = 0.0f;
#pragma unroll
                            for (int r = 0; r < NR; ++r) {
                                v += x[r][k] * w[r];
                            }
                            o[tid + k * wg] = v;
                        }
                    }
                }
            });
    });
}

static void depth_mix_tokens(const int n_rows, const float * cur, const size_t cur_tok, const float * stack,
                             const size_t stack_row, const size_t stack_tok, const char * q, const size_t q_col,
                             const size_t q_tok, const int n_q, float * out, const size_t out_col, const size_t out_tok,
                             const int n_embd, const int n_tokens, const float eps, const float sm_scale,
                             dpct::queue_ptr stream) {
    auto run = [&](auto nr) {
        depth_mix_tokens_launch<decltype(nr)::value>(cur, cur_tok, stack, stack_row, stack_tok, q, q_col, q_tok, n_q,
                                                     out, out_col, out_tok, n_embd, n_tokens, eps, sm_scale, stream);
    };
    switch (n_rows) {
        case 2: run(std::integral_constant<int, 2>{}); break;
        case 3: run(std::integral_constant<int, 3>{}); break;
        case 4: run(std::integral_constant<int, 4>{}); break;
        case 5: run(std::integral_constant<int, 5>{}); break;
        case 6: run(std::integral_constant<int, 6>{}); break;
        case 7: run(std::integral_constant<int, 7>{}); break;
        case 8: run(std::integral_constant<int, 8>{}); break;
        default: GGML_ABORT("depth mix: %d rows", n_rows);
    }
    static_assert(DEPTH_MIX_T_MAXR == 8, "the cases above");
}

static bool depth_mix_disjoint(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t as = (uintptr_t) a->data;
    const uintptr_t ae = as + ggml_nbytes(a);
    const uintptr_t bs = (uintptr_t) b->data;
    const uintptr_t be = bs + ggml_nbytes(b);
    return ae <= bs || be <= as;
}

int ggml_sycl_try_depth_mix_fusion(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, const int i) {
    // 0 off, 1 one token, 2 (default) any number of tokens
    static const int mode = g_ggml_sycl_enable_fusion ? ggml_sycl_get_env("GGML_SYCL_DEPTH_MIX", 2) : 0;
    if (mode == 0 || i + 6 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * norm = cgraph->nodes[i];
    const ggml_tensor * cat  = cgraph->nodes[i + 1];
    const ggml_tensor * tr   = cgraph->nodes[i + 2];
    const ggml_tensor * ct   = cgraph->nodes[i + 3];
    if (norm->op != GGML_OP_RMS_NORM || cat->op != GGML_OP_CONCAT || tr->op != GGML_OP_TRANSPOSE ||
        ct->op != GGML_OP_CONT) {
        return 0;
    }
    // the query weight may come through a few view nodes (a row view of the loop queries, a reshape) and, for
    // several tokens, a REPEAT over the tokens
    int j   = i + 4;
    int rep = -1;
    while (j < cgraph->n_nodes && j < i + 8 &&
           (cgraph->nodes[j]->op == GGML_OP_VIEW || cgraph->nodes[j]->op == GGML_OP_RESHAPE ||
            (cgraph->nodes[j]->op == GGML_OP_REPEAT && rep < 0))) {
        if (cgraph->nodes[j]->op == GGML_OP_REPEAT) {
            rep = j;
        }
        ++j;
    }
    if (j + 2 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * mm  = cgraph->nodes[j];
    const ggml_tensor * sm  = cgraph->nodes[j + 1];
    const ggml_tensor * out = cgraph->nodes[j + 2];
    if (mm->op != GGML_OP_MUL_MAT || sm->op != GGML_OP_SOFT_MAX || out->op != GGML_OP_MUL_MAT) {
        return 0;
    }

    // the chain: cat = concat(stack, norm, dim 1), tr/ct = its transpose, mm = cat x q, sm = softmax(mm), out = ct x sm
    const ggml_tensor * cur   = norm->src[0];
    const ggml_tensor * stack = cat->src[0];
    const ggml_tensor * q     = mm->src[1];
    if (cat->src[1] != norm || ((const int32_t *) cat->op_params)[0] != 1 || tr->src[0] != cat || ct->src[0] != tr ||
        mm->src[0] != cat || sm->src[0] != mm || sm->src[1] != nullptr || sm->src[2] != nullptr ||
        out->src[0] != ct || out->src[1] != sm) {
        return 0;
    }
    // a REPEAT inside the window is skipped: the kernel reads its source, the same queries for every token
    if (rep >= 0 && cgraph->nodes[rep] != q) {
        return 0;
    }
    const ggml_tensor * qs       = rep >= 0 ? q->src[0] : q;
    const int64_t       n_embd   = norm->ne[0];
    const int64_t       n_tokens = norm->ne[2];
    const int64_t       n_stack  = stack->ne[1];
    const int64_t       n_q      = q->ne[1];
    if (n_tokens > 1 && mode < 2) {
        return 0;
    }
    // f32, dense rows; the queries per token or one set for all
    if (norm->ne[1] != 1 || norm->ne[3] != 1 || stack->ne[0] != n_embd || stack->ne[2] != n_tokens ||
        stack->ne[3] != 1 || q->ne[0] != n_embd || q->ne[2] != n_tokens || q->ne[3] != 1 || qs->ne[0] != n_embd ||
        qs->ne[1] != n_q || (qs->ne[2] != 1 && qs->ne[2] != n_tokens) || qs->ne[3] != 1 || out->ne[0] != n_embd ||
        out->ne[1] != n_q || out->ne[2] != n_tokens || out->ne[3] != 1) {
        return 0;
    }
    if (cur->type != GGML_TYPE_F32 || stack->type != GGML_TYPE_F32 || qs->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || !ggml_is_contiguous(cur) || stack->nb[0] != sizeof(float) ||
        stack->nb[1] % sizeof(float) != 0 || stack->nb[2] % sizeof(float) != 0 || qs->nb[0] != sizeof(float) ||
        !ggml_is_contiguous(out)) {
        return 0;
    }
    if (n_tokens == 1 ? (n_stack + 1 > DEPTH_MIX_MAX_ROWS || n_q < 1 || n_q > 64)
                      : (n_stack + 1 < 2 || n_stack + 1 > DEPTH_MIX_T_MAXR || n_q < 1 || n_q > DEPTH_MIX_T_MAXQ ||
                         n_embd % (DEPTH_MIX_T_EPT * DEPTH_MIX_SG) != 0 || n_embd / DEPTH_MIX_T_EPT > 1024)) {
        return 0;
    }
    float eps;
    float sm_scale;
    float max_bias;
    memcpy(&eps, norm->op_params, sizeof(float));
    memcpy(&sm_scale, (const float *) sm->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) sm->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) {
        return 0;
    }
    // every skipped node is used only inside the chain
    const int uses_expected[7][2] = { { i, 1 }, { i + 1, 2 }, { i + 2, 1 }, { i + 3, 1 }, { j, 1 }, { j + 1, 1 },
                                      { rep >= 0 ? rep : j, 1 } };
    for (const auto & u : uses_expected) {
        if (ggml_node_get_use_count(cgraph, u[0]) != u[1] || (cgraph->nodes[u[0]]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
    }
    // the kernel runs at norm and writes out there: out (allocated at the last node) must not overlap what it reads
    if (!depth_mix_disjoint(out, cur) || !depth_mix_disjoint(out, stack) || !depth_mix_disjoint(out, qs)) {
        return 0;
    }

    scope_op_debug_print scope_dbg_print(__func__, out, /*num_src=*/2);
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    if (n_tokens == 1) {
        depth_mix_launch((const float *) cur->data, (const float *) stack->data, stack->nb[1] / sizeof(float),
                         (int) n_stack, (const char *) qs->data, qs->nb[1], (int) n_q, (float *) out->data,
                         (int) n_embd, eps, sm_scale, ctx.stream());
    } else {
        depth_mix_tokens((int) n_stack + 1, (const float *) cur->data, (size_t) n_embd, (const float *) stack->data,
                         stack->nb[1] / sizeof(float), stack->nb[2] / sizeof(float), (const char *) qs->data, qs->nb[1],
                         qs->ne[2] == 1 ? 0 : qs->nb[2], (int) n_q, (float *) out->data, (size_t) n_embd,
                         (size_t) (n_embd * n_q), (int) n_embd, (int) n_tokens, eps, sm_scale, ctx.stream());
    }
    return j + 2 - i;
}
