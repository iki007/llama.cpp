#pragma once

#include "common.hpp"

// Decode flash attention on XMX over a q8_0 cache read in place (head 128 / 256, up to GGML_SYCL_FA_XMX_MAX_Q query
// rows, GQA, sinks, an indexed bias): the decode path of the fork's fattn-xmx.cpp. GGML_SYCL_FA_XMX_DEC=0 turns it off.
bool ggml_sycl_flash_attn_ext_xmx_dec_supported(int device, const ggml_tensor * dst);
// the data pointers of a supported node: 64-byte aligned q8_0 layouts, distinct K and V (checked at the launch)
bool ggml_sycl_flash_attn_ext_xmx_dec_data_ok(const ggml_tensor * dst);
void ggml_sycl_flash_attn_ext_xmx_dec(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
