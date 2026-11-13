#pragma once
// Prompt-size GEMM with the weights dequantized inside the kernel (pgemm.cpp): Q4_K, Q5_K, IQ4_XS on Xe2 XMX
#include "common.hpp"

// the device and type can take ggml_sycl_pgemm (GGML_SYCL_PGEMM=0 turns it off)
bool ggml_sycl_pgemm_supported(int device, ggml_type type);

// dst[t * ldd + r] = sum_k W[r][k] * Y[t][k] for nrows rows of W (plain blocks, or the reordered layout when reordered)
// and ntok f16 rows of Y. Returns false for shapes it does not take (K % 256, nrows % 128, ntok < 64, alignment).
bool ggml_sycl_pgemm(ggml_type type, const void * vx, bool reordered, const sycl::half * Y, float * dst, int K, int nrows,
                     int ntok, int64_t ldd, dpct::queue_ptr stream);
