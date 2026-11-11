#pragma once

#include "common.hpp"

// strides in floats, as in launch_gated_delta_net
struct ggml_sycl_gdn_esimd_args {
    int64_t H, n_tokens, n_seqs, neqk1, rq3;
    int64_t sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
    float   scale;
    int     K;            // snapshot slots: the states after the last K tokens
    int64_t slot_stride;  // between snapshot slots of state_out
};

// whether the ESIMD kernel covers this gated delta net: Xe2, state size 128, one gate per head, 16-byte aligned rows
// and buffers
bool ggml_sycl_gdn_esimd_supported(const ggml_backend_sycl_context & ctx, int64_t S_v, bool kda, int K,
                                   const float * q, const float * k, const float * v, const float * state_in,
                                   const float * attn, const float * state_out, const ggml_sycl_gdn_esimd_args & a);

void ggml_sycl_gdn_esimd(ggml_backend_sycl_context & ctx, const float * q, const float * k, const float * v,
                         const float * g, const float * beta, const float * state_in, float * attn, float * state_out,
                         const ggml_sycl_gdn_esimd_args & a);
