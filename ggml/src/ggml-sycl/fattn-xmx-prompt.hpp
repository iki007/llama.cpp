#pragma once

#include "common.hpp"

// Prompt flash attention on XMX (ESIMD DPAS), head 128: one thread owns 8 rows (query x head of one KV head) with
// their whole output in registers; the threads of a work-group share the K / V tiles. An f16 cache is read in place, a
// quantized one in chunks converted to f16. An indexed bias (ggml_flash_attn_ext_add_indexed_bias) is applied per
// 64-token block class: dead, one bin for the whole block, or per token. GGML_SYCL_FA_XMX_PROMPT=0 turns it off;
// GGML_SYCL_FA_XMX_PROMPT_MIN_Q is the smallest batch it takes (32).
bool ggml_sycl_flash_attn_ext_xmx_prompt_supported(int device, const ggml_tensor * dst);
// the data pointers of a supported node: 64-byte aligned K / V head slices (checked at the launch, not in supported)
bool ggml_sycl_flash_attn_ext_xmx_prompt_data_ok(const ggml_tensor * dst);
void ggml_sycl_flash_attn_ext_xmx_prompt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
