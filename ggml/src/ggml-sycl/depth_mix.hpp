#pragma once

#include "common.hpp"

// Depth attention over earlier block outputs (Inkling's mix()) in one kernel, for one token:
//   RMS_NORM(cur) -> CONCAT(stack, row) -> TRANSPOSE -> CONT -> [VIEW/RESHAPE of the query weight]
//   -> MUL_MAT(states, q) -> SOFT_MAX -> MUL_MAT(states^T, p)
// Returns the number of nodes after node_idx that the fused launch covers (0 = not fused).
int ggml_sycl_try_depth_mix_fusion(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx);
