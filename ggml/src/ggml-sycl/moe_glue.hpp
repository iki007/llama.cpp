#pragma once

#include "common.hpp"

// The element-wise chains between and after the expert GEMMs of build_moe_ffn() in one kernel each, starting at the
// REPEAT of a per-expert weight scale (build_lora_mm_id: res * get_rows(repeat(scale), ids)):
//  - gate_up: MUL_MAT_ID output x expert scale, then the SiTU activation of its gate / up halves
//      situ(g, u) = s3 * tanh(s1 * g) * sigmoid(g) * u3 * tanh(u1 * u)   (Inkling: s1 = 1/beta, s3 = beta, ...)
//  - down:    MUL_MAT_ID output x expert scale x router weight, summed over the used experts (the ADD chain of
//      the expert views), in the order of the ADD chain
// Returns the number of nodes after node_idx that the fused launch covers (0 = not fused).
// GGML_SYCL_MOE_GLUE=0 turns it off.
int ggml_sycl_try_moe_glue_fusion(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx);
