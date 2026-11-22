#pragma once

#include "common.hpp"

// Inkling's loop controller (control() in src/models/inkling2.cpp) in two kernels, for 1-8 tokens:
//   gate = W_down x CONCAT(RMS_NORM(cur), RMS_NORM(msg))
//   out  = (cur - TANH(W_forget x gate) * cur) + TANH(W_gain x gate) * msg
// The kernels run at RMS_NORM(msg), the node before the CONCAT; RMS_NORM(cur), which the graph may place earlier
// (before the message's own depth mix), is skipped there. Returns, for an RMS_NORM node:
//   > 0  the fused launch ran, skip this many nodes after node_idx
//   < 0  this node is the RMS_NORM(cur) of a chain fused later: nothing to compute here
//   0    not fused
// GGML_SYCL_LOOP_CTRL=0 turns it off.
int ggml_sycl_try_loop_ctrl_fusion(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx);
