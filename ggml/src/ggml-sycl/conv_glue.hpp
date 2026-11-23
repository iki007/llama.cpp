#pragma once

#include "common.hpp"

// Inkling's short convolution with a residual (convolve() in src/models/inkling2.cpp) after its SSM_CONV, in one
// kernel: SSM_CONV(x, w) -> [RESHAPE] -> ADD(cur, .) -> SCALE  =>  out = (cur + conv(x, w)) * s + b.
// Returns the number of nodes after node_idx that the fused launch covers (0 = not fused).
// GGML_SYCL_CONV_GLUE=0 turns it off.
int ggml_sycl_try_conv_glue_fusion(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// CONCAT(past, x) followed by its conv state store (CPY, or SET_ROWS into the rollback slots) in one kernel.
int ggml_sycl_try_conv_store_fusion(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx);
