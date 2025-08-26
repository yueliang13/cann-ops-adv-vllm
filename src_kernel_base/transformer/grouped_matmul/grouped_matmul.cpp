/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul.cpp
 * \brief
 */
#include "grouped_matmul_utils.h"
#include "grouped_matmul_antiquant.h"
#include "grouped_matmul_vector.h"
#include "grouped_matmul.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220

#include "grouped_matmul_antiquant_a16w8_msd.h"
#include "grouped_matmul_antiquant_a8w4_msd_pre.h"
#include "grouped_matmul_antiquant_a8w4_msd.h"
#include "grouped_matmul_quant_mixcore.h"
#endif


using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MATMUL;

#ifndef FORMAT_FRACTAL_NZ
    #define FORMAT_FRACTAL_NZ
#endif

#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
constexpr CubeFormat wFormat = CubeFormat::NZ;
constexpr MatmulConfig matmulCFG = NZ_CFG_MDL;
#else
constexpr CubeFormat wFormat = CubeFormat::ND;
constexpr MatmulConfig matmulCFG = CFG_MDL;
#endif

#if defined(GMM_ANTI_QUANT_A8W4_MSD)
constexpr MatmulConfig A8W4_GMM_CFG_MDL = GetNormalConfig();
#endif

template <bool trans = false>
using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X, trans>;

template <bool trans = false>
using xTypeMSD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_WEIGHT, trans>;

template <bool trans = false>
using weightType = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_X, trans>;

template <bool trans = false>
using weightTypeMSD = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_WEIGHT, trans>;

using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, MM_DTYPE_Y>;

using yTypeMSD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;

using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;

#define GMM_IMP(computeClass, processClass, transA, transB, sync, cfg)                                             \
    do {                                                                                                           \
        using matmulType = MMType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                        \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, &mmTilingData_);                                       \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                         \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CUBE_IMP(transA, transB, sync, cfg)                                                                    \
    do {                                                                                                           \
        if ASCEND_IS_AIV {                                                                                         \
            return;                                                                                                \
        }                                                                                                          \
        using matmulType = MMImplType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                    \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        mm.SetSubBlockIdx(0);                                                                                      \
        mm.Init(&mmTilingData_, &tPipe);                                                                           \
        GMMCompute<matmulType, sync> computeOp(mm);                                                                \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1,  &gmmBaseParams_, &mmTilingData_, &tPipe);                                        \
        GMMProcess<decltype(computeOp)> op(computeOp);                                                             \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CV_SPLIT_IMP(computeClass, processClass, transA, transB, sync, cfg, aType, bType, cType)               \
    do {                                                                                                           \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, cfg>;                         \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                         \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CV_SPLIT_IMP_A8W4_MSD(computeClass, cfg)   \
    do { \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling); \
        if ASCEND_IS_AIV { \
            GMMA8W4PreProcess op1; \
            op1.Init(x, x, groupList, user1, gmmBaseParams_, &tPipe); \
            op1.Process(); \
            tPipe.Reset(); \
            tPipe.Destroy(); \
            tPipe.Init(); \
        }\
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X_DEV_A8W4MSD, false>; \
        using bT = MatmulType<TPosition::GM, CubeFormat::NZ, DTYPE_WEIGHT_DEV_A8W4MSD, false>; \
        using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>; \
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>; \
        using matmulType = MMImplType<aT, bT, cT, biasT, cfg>; \
        matmulType::MT mm; \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling); \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling); \
        if ASCEND_IS_AIC { \
            mm.SetSubBlockIdx(0); \
            mm.Init(&mmTilingData_, &tPipe); \
        } \
        computeClass<matmulType> op(mm); \
        op.Init(x, weight, bias, groupList, scale, perTokenScale, nullptr, nullptr, nullptr, \
                    y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe); \
        op.Process(); \
    } while (0)

extern "C" __global__ __aicore__ void grouped_matmul(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                                     GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                                     GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                                     GM_ADDR workspace, GM_ADDR tiling) {
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GM_ADDR user1 = GetUserWorkspace(workspace);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#if defined(GMM_ANTI_QUANT_A8W4_MSD)
    if (TILING_KEY_IS(8)) {  // antiquant msd
        KERNEL_TASK_TYPE(8, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP_A8W4_MSD(GMMA8W4MSDCompute, A8W4_GMM_CFG_MDL);
    }
#elif defined(GMM_ANTI_QUANT)
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_IMP(GMMAntiquantComputeNorm, GMMAntiquantProcess, false, false, false, matmulCFG);
    } else if (TILING_KEY_IS(2)) {  // weight tansposed
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_IMP(GMMAntiquantComputeNorm, GMMAntiquantProcess, false, true, false, matmulCFG);
    } else if (TILING_KEY_IS(3)) {  // antiquant performence
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_IMP(GMMAntiquantComputePerformance, GMMAntiquantProcess, false, false, false, matmulCFG);
    }
    #if defined(ORIG_DTYPE_WEIGHT) && defined(DT_INT8) && ORIG_DTYPE_WEIGHT == DT_INT8
        if (TILING_KEY_IS(6)) {  // antiquant msd
            KERNEL_TASK_TYPE(6, KERNEL_TYPE_MIX_AIC_1_1);
            GMM_CV_SPLIT_IMP(GMMA16W8MSDCompute, GMMA16W8MSDProcess, false, false, false, matmulCFG,
                            xTypeMSD, weightTypeMSD, yTypeMSD);
        } else if (TILING_KEY_IS(7)) {  // antiquant msd weight tansposed
            KERNEL_TASK_TYPE(7, KERNEL_TYPE_MIX_AIC_1_1);
            GMM_CV_SPLIT_IMP(GMMA16W8MSDCompute, GMMA16W8MSDProcess, false, true, false, matmulCFG,
                            xTypeMSD, weightTypeMSD, yTypeMSD);
        }
    #endif
#elif defined(GMM_QUANT_BF16) || defined(GMM_QUANT_FLOAT16)
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, false, false, matmulCFG, xType, weightType, yType);
    } else if (TILING_KEY_IS(2)) {  // weight tansposed
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, true, false, matmulCFG, xType, weightType, yType);
    } else if (TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, false, false, matmulCFG, xType, weightType, yType);
    } else if (TILING_KEY_IS(5)) {  // weight tansposed
        KERNEL_TASK_TYPE(5, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, true, false, matmulCFG, xType, weightType, yType);
    }
#else
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_0);
        GMM_CUBE_IMP(false, false, false, matmulCFGUnitFlag);
    } else if (TILING_KEY_IS(2)) {    // weight transposed
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_0);
        GMM_CUBE_IMP(false, true, false, matmulCFGUnitFlag);
    }
#endif

#if defined(GMM_FLOAT)
    if (TILING_KEY_IS(1)) {    // x transposed
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_1);
        if ASCEND_IS_AIV {
            GET_TILING_DATA(tilingData, tiling);
            EmptyTensorCompute<DTYPE_Y>(groupList, y, &tilingData);
        }
        if ASCEND_IS_AIC {
            GMM_CUBE_IMP(true, false, false, matmulCFG);
        }
    }
#endif
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
#if defined(GMM_FLOAT)
    if (TILING_KEY_IS(0)) {
        GMM_CUBE_IMP(false, false, false, matmulCFG);
    } else if (TILING_KEY_IS(1)) {    // x transposed
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_1);
        if ASCEND_IS_AIV {
            GET_TILING_DATA(tilingData, tiling);
            EmptyTensorCompute<DTYPE_Y>(groupList, y, &tilingData);
        }
        if ASCEND_IS_AIC {
            GMM_CUBE_IMP(true, false, false, matmulCFG);
        }
    } else if (TILING_KEY_IS(2)) {    // weight transposed
        GMM_CUBE_IMP(false, true, false, matmulCFG);
    }

#endif
#endif
}
