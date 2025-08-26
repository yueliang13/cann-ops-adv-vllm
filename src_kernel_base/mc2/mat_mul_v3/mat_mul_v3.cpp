/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mat_mul_v3.cpp
 * \brief
 */
#include "mat_mul_v3_common.h"
#include "mat_mul_asw_kernel.h"
#include "mat_mul_base_kernel.h"
#include "mat_mul_deterministic_splitk_kernel.h"
#include "mat_mul_sc_splitk_kernel.h"
#include "mat_mul_unaligned_base_kernel.h"
#include "mat_mul_unaligned_deterministic_splitk_kernel.h"
#include "mat_mul_unaligned_sc_splitk_kernel.h"
#include "mat_mul_optimized_fixpipe_algorithm.h"
#include "mat_mul_l1_full_load.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "mat_mul_multi_core_splitk_kernel.h"
#endif
using namespace AscendC;
using namespace matmul;
#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#if defined(FORMAT_X1) && FORMAT_X1 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x1 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x1 = CubeFormat::ND;
#endif

#if defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x2 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x2 = CubeFormat::ND;
#endif

#if defined(FORMAT_Y) && FORMAT_Y == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_y = CubeFormat::NZ;
#else
constexpr CubeFormat format_y = CubeFormat::ND;
#endif

#define MMV3_IMPL(templateFunc, cFormat, ...)                                                                                 \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, cFormat, DTYPE_Y>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateFunc<aType, bType, cType, biasType, __VA_ARGS__>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateFunc<aType, bType, cType, biasType, __VA_ARGS__>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateFunc<aType, bType, cType, biasType, __VA_ARGS__>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateFunc<aType, bType, cType, biasType, __VA_ARGS__>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        }                                                                                                            \
    } while(0)

#define MMV3_IMPL_CLASS(templateClass, aFormat, ...)                                                                 \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        }                                                                                                            \
    } while (0)

// cFormat is for tempCGlobal Nz out, not cTensor out
#define MMV3_IMPL_C_CLASS(templateClass, aFormat, cFormat, ...)                                                      \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, cFormat, DTYPE_Y>;                                          \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        }                                                                                                            \
    } while (0)

#define MMV3_IMPL_CLASS_TRANS(transA, transB, templateClass, ...)                                                    \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>;                                   \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, transA>;                               \
        using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, transB>;                               \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                                \
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                         \
        op.Process();                                                                                                \
    } while (0)

extern "C" __global__ __aicore__ void mat_mul_v3(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
    GM_ADDR offsetWGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);
#if defined(__DAV_C310__)
    REGISTER_TILING_DEFAULT(MatMulV3TilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    // Adaptive Sliding Window Kernel
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (TILING_KEY_IS(10000900009000090000UL)) {
        MMV3_IMPL_CLASS_TRANS(false, false, MatmulV3Advanced::MatmulAswKernel, MatmulV3Advanced::MatmulAswBlock,
                              MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000090001UL)) {
        MMV3_IMPL_CLASS_TRANS(true, false, MatmulV3Advanced::MatmulAswKernel, MatmulV3Advanced::MatmulAswBlock,
                              MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000090002UL)) {
        MMV3_IMPL_CLASS_TRANS(false, true, MatmulV3Advanced::MatmulAswKernel, MatmulV3Advanced::MatmulAswBlock,
                              MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000090003UL)) {
        MMV3_IMPL_CLASS_TRANS(true, true, MatmulV3Advanced::MatmulAswKernel, MatmulV3Advanced::MatmulAswBlock,
                              MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000290000UL)) {
        KERNEL_TASK_TYPE(10000900009000290000UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(false, false, MatmulV3Advanced::MatmulStreamKKernel, MatmulV3Advanced::MatmulStreamKBlock,
                              MM_CFG_NO_PRELOAD, false);
    } else if (TILING_KEY_IS(10000900009000290001UL)) {
        KERNEL_TASK_TYPE(10000900009000290001UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(true, false, MatmulV3Advanced::MatmulStreamKKernel, MatmulV3Advanced::MatmulStreamKBlock,
                              MM_CFG_NO_PRELOAD, false);
    } else if (TILING_KEY_IS(10000900009000290002UL)) {
        KERNEL_TASK_TYPE(10000900009000290002UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(false, true, MatmulV3Advanced::MatmulStreamKKernel, MatmulV3Advanced::MatmulStreamKBlock,
                              MM_CFG_NO_PRELOAD, false);
    } else if (TILING_KEY_IS(10000900009000290003UL)) {
        KERNEL_TASK_TYPE(10000900009000290003UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(true, true, MatmulV3Advanced::MatmulStreamKKernel, MatmulV3Advanced::MatmulStreamKBlock,
                              MM_CFG_NO_PRELOAD, false);
    } else if (TILING_KEY_IS(10000902009000290000UL)) {
        KERNEL_TASK_TYPE(10000902009000290000UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(false, false, MatmulV3Advanced::MatmulStreamKKernel,
                                    MatmulV3Advanced::MatmulStreamKBlock, MM_CFG_NO_PRELOAD, true);
    } else if (TILING_KEY_IS(10000902009000290001UL)) {
        KERNEL_TASK_TYPE(10000902009000290001UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(true, false, MatmulV3Advanced::MatmulStreamKKernel,
                                    MatmulV3Advanced::MatmulStreamKBlock, MM_CFG_NO_PRELOAD, true);
    } else if (TILING_KEY_IS(10000902009000290002UL)) {
        KERNEL_TASK_TYPE(10000902009000290002UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(false, true, MatmulV3Advanced::MatmulStreamKKernel,
                                    MatmulV3Advanced::MatmulStreamKBlock, MM_CFG_NO_PRELOAD, true);
    } else if (TILING_KEY_IS(10000902009000290003UL)) {
        KERNEL_TASK_TYPE(10000902009000290003UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(true, true, MatmulV3Advanced::MatmulStreamKKernel,
                                    MatmulV3Advanced::MatmulStreamKBlock, MM_CFG_NO_PRELOAD, true);
    } else if (TILING_KEY_IS(10000900009001090000UL)) {
        MMV3_IMPL_CLASS_TRANS(false, false, MatmulV3Advanced::MatmulAswKernelAL1FullLoad,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001090001UL)) {
        MMV3_IMPL_CLASS_TRANS(true, false, MatmulV3Advanced::MatmulAswKernelAL1FullLoad,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001090002UL)) {
        MMV3_IMPL_CLASS_TRANS(false, true, MatmulV3Advanced::MatmulAswKernelAL1FullLoad,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001090003UL)) {
        MMV3_IMPL_CLASS_TRANS(true, true, MatmulV3Advanced::MatmulAswKernelAL1FullLoad,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002090000UL)) {
        MMV3_IMPL_CLASS_TRANS(false, false, MatmulV3Advanced::MatmulAswKernelBL1FullLoad,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002090001UL)) {
        MMV3_IMPL_CLASS_TRANS(true, false, MatmulV3Advanced::MatmulAswKernelBL1FullLoad,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002090002UL)) {
        MMV3_IMPL_CLASS_TRANS(false, true, MatmulV3Advanced::MatmulAswKernelBL1FullLoad,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002090003UL)) {
        MMV3_IMPL_CLASS_TRANS(true, true, MatmulV3Advanced::MatmulAswKernelBL1FullLoad,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000901009000090000UL)) {
        KERNEL_TASK_TYPE(10000901009000090000UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(false, false, MatmulV3Advanced::MatmulFixpipeOptiKernel,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000901009000090001UL)) {
        KERNEL_TASK_TYPE(10000901009000090001UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(true, false, MatmulV3Advanced::MatmulFixpipeOptiKernel,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000901009000090002UL)) {
        KERNEL_TASK_TYPE(10000901009000090002UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(false, true, MatmulV3Advanced::MatmulFixpipeOptiKernel,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000901009000090003UL)) {
        KERNEL_TASK_TYPE(10000901009000090003UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(true, true, MatmulV3Advanced::MatmulFixpipeOptiKernel,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000902009000090000UL)) {
        KERNEL_TASK_TYPE(10000902009000090000UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(false, false, MatmulV3Advanced::MatmulFixpipeOptiDualDstKernel,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000902009000090001UL)) {
        KERNEL_TASK_TYPE(10000902009000090001UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(true, false, MatmulV3Advanced::MatmulFixpipeOptiDualDstKernel,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000902009000090002UL)) {
        KERNEL_TASK_TYPE(10000902009000090002UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(false, true, MatmulV3Advanced::MatmulFixpipeOptiDualDstKernel,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000902009000090003UL)) {
        KERNEL_TASK_TYPE(10000902009000090003UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(true, true, MatmulV3Advanced::MatmulFixpipeOptiDualDstKernel,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009003090000UL)) {
        MMV3_IMPL_CLASS_TRANS(false, false, MatmulV3Advanced::MatmulAswKernelABL1FullLoad,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009003090001UL)) {
        MMV3_IMPL_CLASS_TRANS(true, false, MatmulV3Advanced::MatmulAswKernelABL1FullLoad,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009003090002UL)) {
        MMV3_IMPL_CLASS_TRANS(false, true, MatmulV3Advanced::MatmulAswKernelABL1FullLoad,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009003090003UL)) {
        MMV3_IMPL_CLASS_TRANS(true, true, MatmulV3Advanced::MatmulAswKernelABL1FullLoad,
            MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    }
#else
    GET_TILING_DATA(tilingData, tilingGM);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    // 第一个模板使用mix类型的，使得整个算子的coreType在dyn场景都为mix，静态则根据选择的tilingkey决定coreType
    if (TILING_KEY_IS(10000000000000000000UL)) {
        MMV3_IMPL_CLASS(MatmulBaseUnAlignedKernel, format_x1, MatmulBaseBlock, MM_CFG_VEC_ND2NZ);
    } else if (TILING_KEY_IS(10000000000000000001UL)) {
        MMV3_IMPL_CLASS(MatmulBaseKernel, format_x1, MatmulBaseBlock, MM_CFG_VEC_ND2NZ);
    }
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(10000000000000000000UL)) {
        MMV3_IMPL_CLASS(MatmulBaseUnAlignedKernel, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000021UL)) {
        MMV3_IMPL_CLASS(MatMulSingleCoreSplitKKernel, format_x1, MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_MK);
    } else if (TILING_KEY_IS(10000000000000000051UL)) {
        MMV3_IMPL_CLASS(MatMulSingleCoreSplitKKernel, format_x1, MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_MK, true);
    } else if (TILING_KEY_IS(10000000000000000020UL)) {
        MMV3_IMPL_CLASS(MatMulUnAlignedSingleCoreSplitKKernel, format_x1, MatmulSingleCoreSplitKBaseBlock,
                        MM_CFG_PRELOAD_MK);
    } else if (TILING_KEY_IS(10000000000000000031UL)) {
        MMV3_IMPL(MatMulKernelDeterministicSplitK, format_x1, FIXPIPE_OPT_SELECT::BASE);
    } else if (TILING_KEY_IS(10000000000000000030UL)) {
        MMV3_IMPL(MatMulUnAlignedKernelDeterministicSplitK, format_x1, FIXPIPE_OPT_SELECT::BASE);
    } else if (TILING_KEY_IS(10000000000000000041UL)) {
        KERNEL_TASK_TYPE(10000000000000000041UL, KERNEL_TYPE_MIX_AIC_1_0);
        MMV3_IMPL(MatMulMultiCoreSplitK, format_x1, FIXPIPE_OPT_SELECT::BASE);
    } else if (TILING_KEY_IS(10000000000000000001UL)) {
        KERNEL_TASK_TYPE(10000000000000000001UL, KERNEL_TYPE_AIC_ONLY);
        MMV3_IMPL_CLASS(MatmulBaseKernel, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000101UL)) {
        KERNEL_TASK_TYPE(10000000000000000101UL, KERNEL_TYPE_AIC_ONLY);
        MMV3_IMPL_CLASS(MatmulBaseKernelAL1FullLoad, format_x1, MatmulBaseBlock, MM_CFG_MDL);
    } else if (TILING_KEY_IS(10000000000000000201UL)) {
        KERNEL_TASK_TYPE(10000000000000000201UL, KERNEL_TYPE_AIC_ONLY);
        MMV3_IMPL_CLASS(MatmulBaseKernelBL1FullLoad, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
                        MatmulCallBackFunc<nullptr, nullptr, CopyBL1>);
    } else if (TILING_KEY_IS(10000000000000010201UL)) {
        MMV3_IMPL_CLASS(MatmulBaseUnalignedNKernel, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
                        MatmulCallBackFunc<nullptr, nullptr, CopyBL1>);
    } else if (TILING_KEY_IS(10000000000000000200UL)) {
        MMV3_IMPL_CLASS(MatmulBaseUnAlignedKernelBL1FullLoad, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
                        MatmulCallBackFunc<nullptr, nullptr, CopyBL1>);
    } else if (TILING_KEY_IS(10000000000000010200UL)) {
        MMV3_IMPL_CLASS(MatmulBaseAToNZWithBL1FixpipeKernel, CubeFormat::NZ, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
                        MatmulCallBackFunc<nullptr, nullptr, CopyBL1>);
    } else if (TILING_KEY_IS(10000000000000020201UL)) {
        MMV3_IMPL_C_CLASS(MatmulBaseUnalignedNKernel, format_x1, CubeFormat::NZ, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
                        MatmulCallBackFunc<nullptr, nullptr, CopyBL1>, FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT);
    } else if (TILING_KEY_IS(10000000000000020031UL)) {
        MMV3_IMPL(MatMulKernelDeterministicSplitK, CubeFormat::NZ, FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT);
    } else if (TILING_KEY_IS(10000000000000020030UL)) {
        MMV3_IMPL(MatMulUnAlignedKernelDeterministicSplitK, CubeFormat::NZ, FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT);
    }
#endif
#endif
}
