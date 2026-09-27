#ifndef GGML_SYCL_FATTN_XMX_HPP
#define GGML_SYCL_FATTN_XMX_HPP

#include "common.hpp"

// Flash attention on XMX (DPAS) for short query batches (decode, draft verify) with grouped-query attention.
bool ggml_sycl_flash_attn_ext_xmx_supported(int device, const ggml_tensor * dst);

void ggml_sycl_flash_attn_ext_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_XMX_HPP
