//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//


#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-tile.hpp"
#include "fattn-vec.hpp"
#include "fattn.hpp"
#include "fattn-onednn.hpp"
#include "fattn-sparse.hpp"
#include "fattn-dpas.hpp"
#include "fattn-xmx-prompt.hpp"
#include "fattn-xmx-dec.hpp"
#include "convert.hpp"

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_sycl_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                    \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

static void ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_SYCL_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#endif // GGML_SYCL_FA_ALL_QUANTS

    GGML_ABORT("Not match KV type in vec");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_DPAS     = 120, // q4_0 / q8_0 K/V decode on Xe2 XMX
    BEST_FATTN_KERNEL_XMXD     = 125, // decode / verify rows over a q8_0 (read in place) or f16 cache on Xe2 XMX (fattn-xmx-dec.cpp)
    BEST_FATTN_KERNEL_XMXP     = 130, // prompt batches, head 128, f16 K/V on Xe2 XMX (fattn-xmx-prompt.cpp)
    BEST_FATTN_KERNEL_ONEDNN   = 150, // oneDNN SDPA: native F16 (PR #25222)
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_MKL      = 300,
};


static best_fattn_kernel ggml_sycl_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef SYCL_FLASH_ATTN
    GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// SYCL_FLASH_ATTN

    if(!g_ggml_sycl_enable_flash_attention) return BEST_FATTN_KERNEL_NONE;

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    // a mask with its own slice for each head (ne[2] > 1, head h reads slice h % ne[2]; e.g. Inkling 2's relative-position
    // table): only the VEC and TILE kernels index it per head, TILE one head a group
    const bool per_head_mask = mask && mask->ne[2] != 1;
    if (per_head_mask && Q->ne[2] % mask->ne[2] != 0) {
        return BEST_FATTN_KERNEL_NONE;
    }
    // an indexed bias (ggml_flash_attn_ext_add_indexed_bias) is applied by the tile kernel (one head a group), the MKL
    // path, the vector kernel (one token) and the DPAS kernel (q4_0 / q8_0 cache, <= 31 rows)
    const bool indexed_bias = ggml_flash_attn_ext_has_indexed_bias(dst);
    if (indexed_bias && (!mask || max_bias != 0.0f || logit_softcap != 0.0f ||
                         (K->ne[0] != 64 && K->ne[0] != 128) || V->ne[0] != K->ne[0])) {
        return BEST_FATTN_KERNEL_NONE;
    }
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && !per_head_mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // prompt batches over an f16 cache, head 128, with or without an indexed bias
    if (!per_head_mask && ggml_sycl_flash_attn_ext_xmx_prompt_supported(device, dst)) {
        return BEST_FATTN_KERNEL_XMXP;
    }

    // XMX-accelerated path: oneDNN SDPA (native F16 and dequant+non-F16).
    // ONEDNN requires min 32 query tokens — short-circuit decode to avoid
    // calling _supported() on every decode FA call.
    if (Q->ne[1] >= 32 && !per_head_mask && !indexed_bias && !ggml_sycl_flash_attn_ext_onednn_failed(device)
        && ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        return BEST_FATTN_KERNEL_ONEDNN;
    }

    // MKL path: XMX-accelerated GEMM for prompt processing (all KV cache types).
    // The MKL kernel converts non-F16 K/V to F16 via to_fp16_sycl before GEMM,
    // so quantized, F16, BF16, and F32 caches all benefit from XMX acceleration.
    // Activates automatically when flash-attn is enabled (--flash-attn on or -fa)
    // and n_kv >= 1024. Falls through to TILE/VEC for ALiBi, logit softcap,
    // and mismatched batch dimensions (unsupported by the MKL kernel).
    // Set GGML_SYCL_ENABLE_MKL_FA=0 to force TILE/VEC path for A/B testing.
    // Example: GGML_SYCL_ENABLE_MKL_FA=0 llama-cli -m model.gguf -fa -ngl 99 ...
    // Note: MKL GEMM calls are incompatible with SYCL graph capture replay.
    const bool standard_shape = Q->ne[0] >= 64 && Q->ne[0] <= 512 &&
        Q->ne[0] % 64 == 0 && Q->ne[0] == V->ne[0];
    // GLM-4.7 Flash's MLA shape is already expressible by the MKL pipeline:
    // KQ GEMM uses DKQ=576 while VKQ and output use DV=512. Keep this narrow
    // until other mismatched K/V shapes have independent correctness data.
    // gqa 20: GLM-4.7 Flash, gqa 40: Raon-1 mini (test-backend-ops covers both)
    const bool glm_mla_shape = Q->ne[0] == 576 && K->ne[0] == 576 &&
        V->ne[0] == 512 && (gqa_ratio == 20 || gqa_ratio == 40) &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;

    // GGML_SYCL_MKL_FA_MIN_KV: the fewest K/V cells the MKL path takes (default 1024)
    static const int mkl_fa_min_kv = ggml_sycl_get_env("GGML_SYCL_MKL_FA_MIN_KV", 1024);
    // MLA decode / verify batches with >= 32 heads on one KV head (Raon-1 mini: 40) go to MKL as one GEMM of the heads
    // once keys x tokens reach GGML_SYCL_MKL_FA_MLA_MIN_WORK (default 8192): the tile kernel re-reads the latent cache
    // once per 8 heads (64K keys, 1 token: 1.84 -> 0.68 ms), but MKL costs ~145 us at any size (crossover ~5K keys at
    // 1 token, ~2K at 4). GLM-4.7 Flash (gqa 20) keeps the tile kernel below 32 query rows.
    static const int64_t mkl_fa_mla_min_work = ggml_sycl_get_env("GGML_SYCL_MKL_FA_MLA_MIN_WORK", 8192);
    const bool mla_decode_mkl = glm_mla_shape && gqa_ratio >= 32 && K->ne[1] * Q->ne[1] >= mkl_fa_mla_min_work;
    // an indexed bias and its sinks are applied by the MKL softmax / normalization (fattn-mkl.cpp); sinks without an
    // indexed bias stay on the kernels below as before
    // the MKL path converts a non-F16 cache with ggml_get_to_fp16(_nc)_sycl (fattn-mkl.cpp, mkl_fa_dequant_chunk): a type
    // without that conversion (e.g. iq4_nl) would call a null function, so it stays on the kernels below
    const auto mkl_kv_ok = [](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F16 || ggml_get_to_fp16_nc_sycl(t->type) != nullptr;
    };
    if (g_ggml_sycl_enable_mkl_fa == 1 && mask && !per_head_mask && (!sinks || indexed_bias) && gqa_ratio >= 2 &&
        mkl_kv_ok(K) && mkl_kv_ok(V) &&
        (standard_shape || glm_mla_shape) &&
        (Q->ne[1] >= 32 || mla_decode_mkl) && K->ne[1] >= mkl_fa_min_kv &&
        max_bias == 0.0f && logit_softcap == 0.0f &&
        (Q->ne[3] == K->ne[3] || K->ne[3] == 1)) {
        // F16 K/V strides must be a multiple of ne[0]*2 (the natural row size
        // in bytes). This passes both dense (nb1 == ne0*2) and interleaved
        // (nb1 == H * ne0*2). Only pathological test strides like nb1=32 or
        // nb1=75 for ne0=40 fall through to TILE.
        bool kv_strides_ok = true;
        for (const ggml_tensor * t : {K, V}) {
            const bool glm_v_stride = glm_mla_shape && t == V &&
                V->view_src && V->nb[1] == K->nb[1];
            if (!glm_v_stride && t->type == GGML_TYPE_F16 &&
                    t->nb[1] % (t->ne[0] * 2) != 0) {
                kv_strides_ok = false;
                break;
            }
        }
        if (kv_strides_ok) {
            return BEST_FATTN_KERNEL_MKL;
        }
    }
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_SYCL_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_SYCL_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_SYCL_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_SYCL_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }


    // q4_0 / q8_0 K/V decode and small batches on Xe2: read the cache directly, no F16 copy of K and V
    // (the DPAS kernel applies an indexed bias at its mask read, fattn-dpas.cpp)
    if (!per_head_mask && ggml_sycl_flash_attn_ext_xmx_dec_supported(device, dst)) {
        return BEST_FATTN_KERNEL_XMXD;
    }
    if (!per_head_mask && ggml_sycl_flash_attn_ext_dpas_supported(device, dst)) {
        return BEST_FATTN_KERNEL_DPAS;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes.
    // BF16 is excluded: the VEC kernel has no BF16 template (it needs GGML_SYCL_FA_ALL_QUANTS for non-F16/Q4_0/Q8_0).
    const bool has_bf16 = (K->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_BF16);
    const bool can_use_vector_kernel = Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0
        && !has_bf16;

    // Fused-XMX path: oneDNN Graph SDPA (flash attention). Strictly
    // additive -- taken only when statically supported, otherwise falls through to VEC/TILE below.
    if (!per_head_mask && !indexed_bias && !ggml_sycl_flash_attn_ext_onednn_failed(device) && ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        return BEST_FATTN_KERNEL_ONEDNN;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                // with an indexed bias the mask is shared, which would allow the GQA tile; the vector kernel takes it
                if (!gqa_opt_applies || indexed_bias) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                // TILE is faster for quantized KV decode on Xe2 (BMG); keep VEC on untested archs
                const gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
                if (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31) {
                    return BEST_FATTN_KERNEL_TILE;
                }
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_set_device(ctx.device);

    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    if (K && Q && K->ne[0] != Q->ne[0] && Q->ne[0] > 0 && K->ne[0] % Q->ne[0] == 0) {
        const int64_t d = Q->ne[0];
        const int64_t n_heads = (K->ne[0] * K->ne[2]) / d;
        K->ne[0] = d;
        K->ne[2] = n_heads;
    }
    if (V && Q && V->ne[0] != Q->ne[0] && Q->ne[0] > 0 && V->ne[0] % Q->ne[0] == 0) {
        const int64_t d = Q->ne[0];
        const int64_t n_heads = (V->ne[0] * V->ne[2]) / d;
        V->ne[0] = d;
        V->ne[2] = n_heads;
    }

    // sparse nodes are gathered down to n_kv_max rows and re-dispatched here
    if (ggml_sycl_flash_attn_ext_sparse(ctx, dst)) {
        return;
    }

    // n_kv watchdog: log when n_kv differs from the last FA call with
    // the same D — helps detect cache-truncation issues.
    static int nkv_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    if (nkv_debug == 1) {
        const ggml_tensor * K_dbg = dst->src[1];
        const ggml_tensor * V_dbg = dst->src[2];
        static int64_t last_nkv_d256 = 0, last_nkv_d512 = 0;
        static int fa_call_seq = 0;
        fa_call_seq++;
        int64_t cur_nkv = K_dbg->ne[1];
        int Dk = (int)K_dbg->ne[0];
        const char * kname = "TILE";
        best_fattn_kernel k = ggml_sycl_get_best_fattn_kernel(ctx.device, dst);
        if (k == BEST_FATTN_KERNEL_MKL)  kname = "MKL";
        if (k == BEST_FATTN_KERNEL_ONEDNN)  kname = "ONEDNN";
        if (k == BEST_FATTN_KERNEL_VEC)  kname = "VEC";
        if (k == BEST_FATTN_KERNEL_DPAS) kname = "DPAS";
        if (k == BEST_FATTN_KERNEL_XMXP) kname = "XMXP";
        if (k == BEST_FATTN_KERNEL_XMXD) kname = "XMXD";
        int64_t delta = 0;
        if (Dk == 256) {
            delta = cur_nkv - last_nkv_d256;
            last_nkv_d256 = cur_nkv;
        } else if (Dk == 512) {
            delta = cur_nkv - last_nkv_d512;
            last_nkv_d512 = cur_nkv;
        }
        GGML_LOG_INFO("[FA-DISP] #%d %s D=%d n_kv=%lld delta=%lld "
                "V_ne1=%lld\n",
                fa_call_seq, kname, Dk,
                (long long)cur_nkv, (long long)delta,
                (long long)V_dbg->ne[1]);
    }

    const best_fattn_kernel fk = ggml_sycl_get_best_fattn_kernel(ggml_sycl_get_device(), dst);
    switch (fk) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("Not support Flash-Attention");
        case BEST_FATTN_KERNEL_ONEDNN:
            // guarded: ggml_sycl_flash_attn_ext_onednn() is only defined under GGML_SYCL_DNNL;
            // the reference must be compiled out here or the GGML_SYCL_DNNL=0 build fails to link.
#if GGML_SYCL_DNNL
            ggml_sycl_flash_attn_ext_onednn(ctx, dst);
#endif
            break;
        case BEST_FATTN_KERNEL_TILE:
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_DPAS:
            ggml_sycl_flash_attn_ext_dpas(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MKL:
            ggml_sycl_flash_attn_ext_mkl(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_XMXD:
            // the 2D loads read the cache (q8_0 in place, or f16) per head: another layout takes the DPAS kernel
            if (ggml_sycl_flash_attn_ext_xmx_dec_data_ok(dst)) {
                ggml_sycl_flash_attn_ext_xmx_dec(ctx, dst);
            } else {
                ggml_sycl_flash_attn_ext_dpas(ctx, dst);
            }
            break;
        case BEST_FATTN_KERNEL_XMXP:
            // the 2D block loads need 64-byte aligned head slices; another layout takes the tile kernel (f16: no staging)
            if (ggml_sycl_flash_attn_ext_xmx_prompt_data_ok(dst)) {
                ggml_sycl_flash_attn_ext_xmx_prompt(ctx, dst);
            } else {
                ggml_sycl_flash_attn_ext_tile(ctx, dst);
            }
            break;
    }

    // --- Output fingerprint (GGML_SYCL_MKL_FA_DIAG=1) ---
    // Copy first 64 float output values to host for fingerprinting.
    // Compare MKL vs TILE (GGML_SYCL_ENABLE_MKL_FA=0) to detect divergence.
    // Only fingerprints the first 6 FA calls with n_kv >= 1024.
    static int fa_diag = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DIAG", 0);
    static int fa_diag_count = 0;
    if (fa_diag == 1 && fa_diag_count < 6) {
        const ggml_tensor * K_diag = dst->src[1];
        const ggml_tensor * V_diag = dst->src[2];
        const ggml_tensor * Q_diag = dst->src[0];
        dpct::queue_ptr q = ctx.stream();
        if (K_diag->ne[1] >= 1024 && q->ext_oneapi_get_state() != sycl::ext::oneapi::experimental::queue_state::recording) {
            fa_diag_count++;
            float diag_buf[64];
            q->memcpy(diag_buf, dst->data, 64 * sizeof(float));
            q->wait();
            const char * kname = "???";
            best_fattn_kernel kb = ggml_sycl_get_best_fattn_kernel(ctx.device, dst);
            if (kb == BEST_FATTN_KERNEL_ONEDNN) kname = "ONEDNN";
            if (kb == BEST_FATTN_KERNEL_MKL) kname = "MKL";
            if (kb == BEST_FATTN_KERNEL_TILE) kname = "TILE";
            if (kb == BEST_FATTN_KERNEL_VEC) kname = "VEC";
            if (kb == BEST_FATTN_KERNEL_DPAS) kname = "DPAS";
            GGML_LOG_INFO("[FA-DIAG] #%d %s D=%d n_kv=%lld n_q=%lld "
                    "n_qh=%lld n_kvh=%lld K=%s V=%s "
                    "nb1=%zu nb2=%zu first 64 floats:\n",
                    fa_diag_count, kname,
                    (int)K_diag->ne[0], (long long)K_diag->ne[1],
                    (long long)Q_diag->ne[1],
                    (long long)Q_diag->ne[2], (long long)K_diag->ne[2],
                    ggml_type_name(K_diag->type),
                    ggml_type_name(V_diag->type),
                    K_diag->nb[1], K_diag->nb[2]);
            for (int i = 0; i < 64; i += 8) {
                GGML_LOG_INFO("  [%2d] %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        i,
                        *(unsigned *)&diag_buf[i+0], *(unsigned *)&diag_buf[i+1],
                        *(unsigned *)&diag_buf[i+2], *(unsigned *)&diag_buf[i+3],
                        *(unsigned *)&diag_buf[i+4], *(unsigned *)&diag_buf[i+5],
                        *(unsigned *)&diag_buf[i+6], *(unsigned *)&diag_buf[i+7]);
            }
        }
    }

}

bool ggml_sycl_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_sycl_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}

bool ggml_sycl_flash_attn_ext_needs_sync(int device, const ggml_tensor * dst) {
    return ggml_sycl_get_best_fattn_kernel(device, dst) == BEST_FATTN_KERNEL_MKL;
}

bool ggml_sycl_flash_attn_ext_uses_onednn(int device, const ggml_tensor * dst) {
    return ggml_sycl_get_best_fattn_kernel(device, dst) == BEST_FATTN_KERNEL_ONEDNN;
}

static uintptr_t ggml_sycl_fattn_reserve_halves(ggml_sycl_fattn_extra & extra, size_t n_halves) {
    if (n_halves == 0) {
        return 0;
    }
    extra.end = GGML_PAD(extra.end, SYCL_BUFFER_ALIGNMENT);
    const uintptr_t block = extra.end;
    extra.end += n_halves * sizeof(sycl::half);
    return block;
}

ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(const ggml_tensor * dst, const int device) {
    ggml_sycl_fattn_extra extra;

    extra.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    if (dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return extra;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (!Q || !K || !V) {
        return extra;
    }

    // MKL converts a quantized cache chunk by chunk in pool memory and DPAS reads it in place: neither reads the
    // staging blocks, which for a quantized cache are the F16 size of all of K and V (2 GiB a layer at 512K cells).
    // The choice is the dispatcher's: it depends on the shapes and the environment, and the only state it reads (a
    // failed oneDNN) can only rule oneDNN out. GGML_SYCL_FA_STAGING_ALL=1 reserves for every kernel, as before.
    static const bool staging_all = ggml_sycl_get_env("GGML_SYCL_FA_STAGING_ALL", 0) != 0;
    if (!staging_all) {
        const best_fattn_kernel k = ggml_sycl_get_best_fattn_kernel(device, dst);
        if (k == BEST_FATTN_KERNEL_MKL || k == BEST_FATTN_KERNEL_DPAS || k == BEST_FATTN_KERNEL_XMXP ||
            k == BEST_FATTN_KERNEL_XMXD) {
            return extra;
        }
    }

    const int64_t d = K->ne[0];
    const int64_t H = Q->ne[2];
    const int64_t q = Q->ne[1];

    // calculate the worst-case memory consumption across all kernels
    const bool onednn_supported = ggml_sycl_flash_attn_ext_onednn_supported(dst, /* use_shape_limit */ false);

    const bool tile_needs_K = K->type != GGML_TYPE_F16;
    const bool tile_needs_V = V->type != GGML_TYPE_F16;

    const bool V_is_K_view = V->view_src &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    size_t need_K = 0, need_V = 0, need_Q = 0, need_out = 0, need_scale = 0;
    if (onednn_supported) {
        need_Q     = (size_t) H * q * d;
        need_out   = (size_t) H * q * d;
        need_scale = 1;
        // an f16 cache is bound in place, so it needs no staging copy
        if (!ggml_sycl_fattn_onednn_binds_kv(K, V)) {
            need_K = (size_t) ggml_nelements(K);
            need_V = (size_t) ggml_nelements(V);
        }
    }
    if (tile_needs_K) {
        need_K = std::max(need_K, (size_t) ggml_nelements(K));
    }
    if (tile_needs_V) {
        need_V = std::max(need_V, (size_t) ggml_nelements(V));
    }

    extra.Q_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_Q);
    extra.K_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_K);
    extra.V_buffer_ptr = (V_is_K_view && !onednn_supported && need_V)
                       ? extra.K_buffer_ptr
                       : ggml_sycl_fattn_reserve_halves(extra, need_V);
    extra.scale_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_scale);
    extra.out_buffer_ptr   = ggml_sycl_fattn_reserve_halves(extra, need_out);

    return extra;
}

size_t ggml_sycl_flash_attn_ext_get_alloc_size(const ggml_tensor * dst, const int device) {
    const ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst, device);
    return (size_t) (extra.end - (uintptr_t) dst->data);
}
