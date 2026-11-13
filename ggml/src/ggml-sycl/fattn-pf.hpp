#pragma once
// Prompt-size flash attention on Xe2 XMX (fattn-pf.cpp): head 256, f16 or q8_0 K / V, all streams in one launch
#include "common.hpp"

// runs dst if the kernel takes its shape (GGML_SYCL_FA_PROMPT=0 turns it off); false leaves it to the other kernels
bool ggml_sycl_fa_prompt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
