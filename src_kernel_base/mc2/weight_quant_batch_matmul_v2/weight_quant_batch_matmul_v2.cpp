/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file weight_quant_batch_matmul_v2.cpp
 * \brief
 */

#define K_MAX_SHAPE_DIM 0

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    #include "weight_quant_batch_matmul_v2_constant.h"
    #include "tool.h"
    #if (defined(ORIG_DTYPE_ANTIQUANT_SCALE) && \
         ((ORIG_DTYPE_ANTIQUANT_SCALE == DT_UINT64) || (ORIG_DTYPE_ANTIQUANT_SCALE == DT_INT64)))
        #include "fixpipe/weight_quant_batch_matmul_v2_fixpipe.h"
    #else
        #include "weight_quant_batch_matmul_v2_custom.h"
        #if (defined(ORIG_DTYPE_Y) && ORIG_DTYPE_Y != DT_INT8)
            #include "weight_quant_batch_matmul_v2_msd_multicore.h"
            #include "weight_quant_batch_matmul_v2_msd_group.h"
            #include "weight_quant_batch_matmul_v2_msd_split_k.h"
            #include "weight_quant_batch_matmul_v2_custom_mix_splitk.h"
            #if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT == FORMAT_FRACTAL_NZ))
                #include "weight_quant_batch_matmul_v2_custom_weight_nz.h"
                #include "weight_quant_batch_matmul_v2_custom_nz_splitk.h"
            #endif
            using WeightQuantBatchMatmulV2Msd::WeightQuantBatchMatmulV2MsdMultiCoreKernel;
        #endif
    #endif
#elif defined(__DAV_C310__)
    #define ENABLE_L2_CACHE
    #include "weight_quant_batch_matmul_v2_constant.h"
    #include "kernel_operator.h"
    #include "kernel_operator_intf.h"
    #include "lib/matmul_intf.h"
    #if defined(ORIG_DTYPE_WEIGHT)
        #if (ORIG_DTYPE_WEIGHT == DT_INT8 || ORIG_DTYPE_WEIGHT == DT_FLOAT8_E5M2 ||                                            \
             ORIG_DTYPE_WEIGHT == DT_FLOAT8_E4M3FN || ORIG_DTYPE_WEIGHT == DT_HIFLOAT8)
            #define WEIGHT_B8_BRANCH
        #endif
        #if (ORIG_DTYPE_WEIGHT == DT_FLOAT8_E5M2 || ORIG_DTYPE_WEIGHT == DT_FLOAT8_E4M3FN ||                                   \
             ORIG_DTYPE_WEIGHT == DT_HIFLOAT8)
            #define WEIGHT_F8_INPUT
        #endif
    #endif
    #if (defined(ORIG_DTYPE_Y) && (ORIG_DTYPE_Y != DT_INT8) && !defined(WEIGHT_F8_INPUT))
        #include "weight_quant_batch_matmul_v2_reg_base.h"
    #endif
    #if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
        #include "n_first/weight_quant_batch_matmul_v2_basic_block_controller.h"
    #endif
#else
    #include "weight_quant_batch_matmul_v2_weight_nz_performance.h"
#endif
using namespace WeightQuantBatchMatmulV2;

#if defined(__DAV_C310__)
    #if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
        static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_0 = {2, 512};
        static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_1 = {4, 512};
        static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_2 = {2, 1024};
        static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_3 = {4, 256};
    #endif
#endif

// if run with ttk without bias, can't get DTYPE_BIAS macro
#ifndef DTYPE_BIAS
    #if defined(ORIG_DTYPE_X) && defined(DT_FLOAT16) && ORIG_DTYPE_X == DT_FLOAT16
        #define DTYPE_BIAS DTYPE_X
    #else
        #define DTYPE_BIAS float
    #endif
#endif

#ifndef DTYPE_ANTIQUANT_OFFSET
    #if defined(ORIG_DTYPE_ANTIQUANT_SCALE) && defined(DT_UINT64) && \
        ORIG_DTYPE_ANTIQUANT_SCALE != DT_UINT64 && ORIG_DTYPE_ANTIQUANT_SCALE != DT_INT64
        #define DTYPE_ANTIQUANT_OFFSET DTYPE_ANTIQUANT_SCALE
    #else
       #define DTYPE_ANTIQUANT_OFFSET int32_t
    #endif
#endif

#if defined(ORIG_DTYPE_WEIGHT) && defined(DT_INT32) && ORIG_DTYPE_WEIGHT == DT_INT32
    #undef DTYPE_WEIGHT
    #define DTYPE_WEIGHT AscendC::int4b_t
#endif

#define INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(templateClass, ...)                                                          \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2TilingData, tilingDataIn, tiling);                       \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                   \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(templateClass, ...)                                           \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2ASTilingData, tilingDataIn, tiling);                     \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                   \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(templateClass, ...)                                                      \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdTilingData, tilingDataIn, tiling);                    \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                   \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(templateClass, ...)                                                  \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2FixpipeTilingData, tilingDataIn, tiling);                \
        templateClass<DTYPE_ANTIQUANT_OFFSET, DTYPE_BIAS, __VA_ARGS__> op;                                           \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_DTYPE(templateClass, ...)                                                    \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2TilingData, tilingDataIn, tiling);                       \
        templateClass<__VA_ARGS__> op;                                                                               \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(templateClass, ...)                                                       \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2NzTilingData, tilingDataIn, tiling);                     \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                   \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_CUSTOM_SPLITK_IMPL(templateClass, ...)                                            \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2CustomNzSplitKTilingData, tilingDataIn, tiling);         \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                   \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(templateClass, ...)                                                 \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2RegBaseTilingData, tilingDataIn, tiling);                \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                   \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

extern "C" __global__ __aicore__ void weight_quant_batch_matmul_v2(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale,
                                                                   GM_ADDR antiquantOffset, GM_ADDR quantScale,
                                                                   GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    AscendC::TPipe tPipe;
    #if defined(__DAV_C310__)
        #undef DTYPE_BIAS
        #define DTYPE_BIAS DTYPE_X
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        #if (defined(ORIG_DTYPE_Y) && (ORIG_DTYPE_Y != DT_INT8) & !defined(WEIGHT_F8_INPUT)) // 当前场景不支持c8和fp8输入
            if (TILING_KEY_IS(100300UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, QuantType::PER_GROUP);
            } else if (TILING_KEY_IS(100310UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, QuantType::PER_GROUP);
            } else if (TILING_KEY_IS(100311UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, true, true, false, QuantType::PER_GROUP);
            } else if (TILING_KEY_IS(100301UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, true, false, false, QuantType::PER_GROUP);
            } else if (TILING_KEY_IS(10100300UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, QuantType::PER_GROUP, true);
            } else if (TILING_KEY_IS(10100310UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, QuantType::PER_GROUP, true);
            } else if (TILING_KEY_IS(100200UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(100210UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(100211UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, true, true, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(100201UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, true, false, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(100100UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(100110UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(100111UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, true, true, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(100101UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, true, false, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(101300UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, QuantType::PER_GROUP);
            } else if (TILING_KEY_IS(101310UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, QuantType::PER_GROUP);
            } else if (TILING_KEY_IS(101311UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, true, true, true, QuantType::PER_GROUP);
            } else if (TILING_KEY_IS(101301UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, true, false, true, QuantType::PER_GROUP);
            } else if (TILING_KEY_IS(10101300UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, QuantType::PER_GROUP, true);
            } else if (TILING_KEY_IS(10101310UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, QuantType::PER_GROUP, true);
            } else if (TILING_KEY_IS(101200UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(101210UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(101211UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, true, true, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(101201UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, true, false, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(101100UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(101110UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(101111UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, true, true, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(101101UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                    WeightQuantBatchMatmulV2RegBaseKernel, true, false, true, QuantType::PER_TENSOR);
            }
        #endif
        #if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
            if (TILING_KEY_IS(2000030004000012100UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                #if defined(WEIGHT_B8_BRANCH)
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                #else
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                #endif
            } else if (TILING_KEY_IS(2000030004000032100UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    true, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                #if defined(WEIGHT_B8_BRANCH)
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                #else
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                #endif
            } else if (TILING_KEY_IS(2000030004000012120UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                #if defined(WEIGHT_B8_BRANCH)
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                #else
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                #endif
            } else if (TILING_KEY_IS(2000030004000032120UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    true, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                #if defined(WEIGHT_B8_BRANCH)
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                #else
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                #endif
            } else if (TILING_KEY_IS(2000030004000011100UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    false, true, QuantType::PER_TENSOR, false, QuantType::PER_TENSOR, CubeFormat::ND};
                #if defined(WEIGHT_B8_BRANCH)
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                #else
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
            #endif
            } else if (TILING_KEY_IS(2000030004000031100UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    true, true, QuantType::PER_TENSOR, false, QuantType::PER_TENSOR, CubeFormat::ND};
                #if defined(WEIGHT_B8_BRANCH)
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                #else
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                #endif
            } else if (TILING_KEY_IS(2000030004000011120UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    false, true, QuantType::PER_TENSOR, true, QuantType::PER_TENSOR, CubeFormat::ND};
                #if defined(WEIGHT_B8_BRANCH)
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                #else
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                #endif
            } else if (TILING_KEY_IS(2000030004000031120UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    true, true, QuantType::PER_TENSOR, true, QuantType::PER_TENSOR, CubeFormat::ND};
                #if defined(WEIGHT_B8_BRANCH)
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                #else
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                #endif
            } else if (TILING_KEY_IS(2000030003000002100UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    false, false, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                    WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
            } else if (TILING_KEY_IS(2000030003000022100UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    true, false, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                    WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
            } else if (TILING_KEY_IS(2000030003000002120UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    false, false, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                    WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
            } else if (TILING_KEY_IS(2000030003000022120UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    true, false, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                    WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
            } else if (TILING_KEY_IS(2000030003000001100UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    false, false, QuantType::PER_TENSOR, false, QuantType::PER_TENSOR, CubeFormat::ND};
                INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                    WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
            } else if (TILING_KEY_IS(2000030003000021100UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    true, false, QuantType::PER_TENSOR, false, QuantType::PER_TENSOR, CubeFormat::ND};
                INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                    WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
            } else if (TILING_KEY_IS(2000030003000001120UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    false, false, QuantType::PER_TENSOR, true, QuantType::PER_TENSOR, CubeFormat::ND};
                INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                    WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
            } else if (TILING_KEY_IS(2000030003000021120UL)) {
                static constexpr WqmmConfig wqmmCfg = {
                    true, false, QuantType::PER_TENSOR, true, QuantType::PER_TENSOR, CubeFormat::ND};
                INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                    WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
            }
            // 下列tiling key只在c8场景和非fp8输入下出现
            #if (defined(ORIG_DTYPE_Y) && (ORIG_DTYPE_Y == DT_INT8) && !defined(WEIGHT_F8_INPUT))
                else if (TILING_KEY_IS(2000030004000012200UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030004000032200UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, true, QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030004000012220UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030004000032220UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, true, QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030004000011200UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030004000031200UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, true, QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030004000011220UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030004000031220UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, true, QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030003000002200UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, false, QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                } else if (TILING_KEY_IS(2000030003000022200UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, false, QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                }  else if (TILING_KEY_IS(2000030003000002220UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, false, QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                } else if (TILING_KEY_IS(2000030003000022220UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, false, QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                } else if (TILING_KEY_IS(2000030003000001200UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, false, QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                } else if (TILING_KEY_IS(2000030003000021200UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, false, QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                } else if (TILING_KEY_IS(2000030003000001220UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, false, QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                } else if (TILING_KEY_IS(2000030003000021220UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, false, QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                }
            #else // 下列tiling key不在c8场景下出现
                else if (TILING_KEY_IS(2000020000000012100UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                } else if (TILING_KEY_IS(2000020001000012100UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                } else if (TILING_KEY_IS(2000020002000012100UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_2);
                } else if (TILING_KEY_IS(2000020003000012100UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                } else if (TILING_KEY_IS(2000020000000012120UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                } else if (TILING_KEY_IS(2000020001000012120UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                } else if (TILING_KEY_IS(2000020002000012120UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_2);
                } else if (TILING_KEY_IS(2000020003000012120UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                }
            #endif
        #endif
        #if defined(ORIG_DTYPE_X) && defined(DT_BF16) && ORIG_DTYPE_X == DT_BF16
            #undef DTYPE_BIAS
            #define DTYPE_BIAS float
            // 当前场景不支持c8和fp8输入
            #if (defined(ORIG_DTYPE_Y) && (ORIG_DTYPE_Y != DT_INT8) && !defined(WEIGHT_F8_INPUT))
                if (TILING_KEY_IS(1100300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, QuantType::PER_GROUP);
                } else if (TILING_KEY_IS(1100310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, QuantType::PER_GROUP);
                } else if (TILING_KEY_IS(1100311UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, true, true, false, QuantType::PER_GROUP);
                } else if (TILING_KEY_IS(1100301UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, true, false, false, QuantType::PER_GROUP);
                } else if (TILING_KEY_IS(11100300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, QuantType::PER_GROUP, true);
                } else if (TILING_KEY_IS(11100310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, QuantType::PER_GROUP, true);
                } else if (TILING_KEY_IS(1100200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(1100210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(1100211UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, true, true, false, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(1100201UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, true, false, false, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(1100100UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(1100110UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(1100111UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, true, true, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(1100101UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, true, false, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(1101300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, QuantType::PER_GROUP);
                } else if (TILING_KEY_IS(1101310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, QuantType::PER_GROUP);
                } else if (TILING_KEY_IS(1101311UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, true, true, true, QuantType::PER_GROUP);
                } else if (TILING_KEY_IS(1101301UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, true, false, true, QuantType::PER_GROUP);
                } else if (TILING_KEY_IS(11101300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, QuantType::PER_GROUP, true);
                } else if (TILING_KEY_IS(11101310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, QuantType::PER_GROUP, true);
                } else if (TILING_KEY_IS(1101200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(1101210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(1101211UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, true, true, true, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(1101201UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, true, false, true, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(1101100UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(1101110UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(1101111UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, true, true, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(1101101UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
                        WeightQuantBatchMatmulV2RegBaseKernel, true, false, true, QuantType::PER_TENSOR);
                }
            #endif
            #if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
                if (TILING_KEY_IS(2000030004000012140UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030004000032140UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030004000011140UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_TENSOR, false, QuantType::PER_TENSOR, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030004000031140UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, true, QuantType::PER_TENSOR, false, QuantType::PER_TENSOR, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030004000011160UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_TENSOR, true, QuantType::PER_TENSOR, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030004000031160UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, true, QuantType::PER_TENSOR, true, QuantType::PER_TENSOR, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030003000002140UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, false, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                } else if (TILING_KEY_IS(2000030003000022140UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, false, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                }  else if (TILING_KEY_IS(2000030004000012160UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                } else if (TILING_KEY_IS(2000030004000032160UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                    #if defined(WEIGHT_B8_BRANCH)
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                }  else if (TILING_KEY_IS(2000030003000002160UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, false, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                } else if (TILING_KEY_IS(2000030003000022160UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, false, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                }  else if (TILING_KEY_IS(2000030003000001140UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, false, QuantType::PER_TENSOR, false, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                } else if (TILING_KEY_IS(2000030003000021140UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, false, QuantType::PER_TENSOR, false, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                }  else if (TILING_KEY_IS(2000030003000001160UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        false, false, QuantType::PER_TENSOR, true, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                } else if (TILING_KEY_IS(2000030003000021160UL)) {
                    static constexpr WqmmConfig wqmmCfg = {
                        true, false, QuantType::PER_TENSOR, true, QuantType::PER_TENSOR, CubeFormat::ND};
                    INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                        WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                }
                // 下列tiling key只在c8场景和非fp8输入下出现
                #if (defined(ORIG_DTYPE_Y) && (ORIG_DTYPE_Y == DT_INT8) && !defined(WEIGHT_F8_INPUT))
                    else if (TILING_KEY_IS(2000030004000012240UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, true, QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                        #if defined(WEIGHT_B8_BRANCH)
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                        #else
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                        #endif
                    } else if (TILING_KEY_IS(2000030004000032240UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            true, true, QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                        #if defined(WEIGHT_B8_BRANCH)
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                        #else
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                        #endif
                    } else if (TILING_KEY_IS(2000030004000012260UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, true, QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                        #if defined(WEIGHT_B8_BRANCH)
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                        #else
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                        #endif
                    } else if (TILING_KEY_IS(2000030004000032260UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            true, true, QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                        #if defined(WEIGHT_B8_BRANCH)
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                        #else
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    #endif
                    } else if (TILING_KEY_IS(2000030004000011240UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, true, QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                        #if defined(WEIGHT_B8_BRANCH)
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                        #else
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                        #endif
                    } else if (TILING_KEY_IS(2000030004000031240UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            true, true, QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                        #if defined(WEIGHT_B8_BRANCH)
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    #else
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                        #endif
                    } else if (TILING_KEY_IS(2000030004000011260UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, true, QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                        #if defined(WEIGHT_B8_BRANCH)
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                        #else
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                        #endif
                    } else if (TILING_KEY_IS(2000030004000031260UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            true, true, QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                        #if defined(WEIGHT_B8_BRANCH)
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                        #else
                            INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                                WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                        #endif
                    } else if (TILING_KEY_IS(2000030003000002240UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, false, QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                    } else if (TILING_KEY_IS(2000030003000022240UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            true, false, QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                    } else if (TILING_KEY_IS(2000030003000002260UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, false, QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                    } else if (TILING_KEY_IS(2000030003000022260UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            true, false, QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                    } else if (TILING_KEY_IS(2000030003000001240UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, false, QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                    } else if (TILING_KEY_IS(2000030003000021240UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            true, false, QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                    } else if (TILING_KEY_IS(2000030003000001260UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, false, QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                    } else if (TILING_KEY_IS(2000030003000021260UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            true, false, QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                    }
                #else // 下列tiling key不在c8场景下出现
                    else if (TILING_KEY_IS(2000020000000012140UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    } else if (TILING_KEY_IS(2000020001000012140UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    } else if (TILING_KEY_IS(2000020002000012140UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_2);
                    } else if (TILING_KEY_IS(2000020003000012140UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                    } else if (TILING_KEY_IS(2000020000000012160UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
                    } else if (TILING_KEY_IS(2000020001000012160UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
                    } else if (TILING_KEY_IS(2000020002000012160UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_2);
                    } else if (TILING_KEY_IS(2000020003000012160UL)) {
                        static constexpr WqmmConfig wqmmCfg = {
                            false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
                        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
                            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
                    }
                #endif
            #endif
        #endif
    #elif (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220)
        #if (defined(ORIG_DTYPE_ANTIQUANT_SCALE) && \
             ((ORIG_DTYPE_ANTIQUANT_SCALE == DT_UINT64) || (ORIG_DTYPE_ANTIQUANT_SCALE == DT_INT64)))
            // fixp方案
            #if ((ORIG_DTYPE_X == DT_FLOAT16) && (ORIG_DTYPE_Y == DT_FLOAT16) && (ORIG_DTYPE_WEIGHT == DT_INT8))
                KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
                if ASCEND_IS_AIC {
                    // 模板参数为Trans bTrans antiquantType quantType hasAntiquantOffset hasBias weightFormat aFullLoad
                    if (TILING_KEY_IS(1000200000000012000UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, false, false, CubeFormat::ND, false);
                    } else if (TILING_KEY_IS(1000200001000012000UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, false, false, CubeFormat::ND, true);
                    } else if (TILING_KEY_IS(1000200000000012010UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, false, true, CubeFormat::ND, false);
                    } else if (TILING_KEY_IS(1000200001000012010UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, false, true, CubeFormat::ND, true);
                    } else if (TILING_KEY_IS(1000200000000012020UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, true, false, CubeFormat::ND, false);
                    } else if (TILING_KEY_IS(1000200001000012020UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, true, false, CubeFormat::ND, true);
                    } else if (TILING_KEY_IS(1000200000000012030UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, true, true, CubeFormat::ND, false);
                    } else if (TILING_KEY_IS(1000200001000012030UL)) {
                        INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(WeightQuantBatchMatmulV2FixpipeKernel, false, true,
                            QuantType::PER_CHANNEL, QuantType::NONE, true, true, CubeFormat::ND, true);
                    }
                }
            #endif
        #elif (ORIG_DTYPE_Y == DT_INT8)
            if (TILING_KEY_IS(310100UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311100UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310110UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311110UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310101UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311101UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310111UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311111UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310200UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311200UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310210UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311210UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310201UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311201UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310211UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311211UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310300UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310310UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310301UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(310311UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311300UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311310UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311301UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(311311UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
            } else if (TILING_KEY_IS(320300UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320310UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320301UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320311UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321300UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321310UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321301UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321311UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320100UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321100UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320110UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321110UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320101UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321101UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320111UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_TENSOR, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321111UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_TENSOR, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320200UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321200UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320210UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321210UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320201UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321201UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(320211UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_CHANNEL, false, QuantType::PER_CHANNEL);
            } else if (TILING_KEY_IS(321211UL)) {
                INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                QuantType::PER_CHANNEL, true, QuantType::PER_CHANNEL);
            }
        #else
            #if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
                if (TILING_KEY_IS(310100UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311100UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310110UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311110UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310101UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311101UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310111UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_TENSOR, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311111UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_TENSOR, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310201UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311201UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310211UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311211UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310301UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(310311UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311301UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(311311UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(320300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(320310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(320301UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(320311UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_GROUP, false, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(321300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, false,
                                                    QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(321310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, false, true,
                                                    QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(321301UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, false,
                                                    QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(321311UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomKernel, true, true,
                                                    QuantType::PER_GROUP, true, QuantType::PER_CHANNEL);
                } else if (TILING_KEY_IS(611200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, false,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(610200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, false,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(611210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(610210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(10611200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(10611300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_GROUP, CubeFormat::ND);
                } else if (TILING_KEY_IS(10610200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(10610300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_GROUP, CubeFormat::ND);
                } else if (TILING_KEY_IS(1000111000000003020UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_GROUP, CubeFormat::ND, HighPerformanceType);
                } else if (TILING_KEY_IS(1000111000000003000UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_GROUP, CubeFormat::ND, HighPerformanceType);
                } else if (TILING_KEY_IS(10611210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(10610210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND);
                } else if (TILING_KEY_IS(20611210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND,
                        PrecisionType::HIGH_PRECISION);
                } else if (TILING_KEY_IS(20610210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND,
                        PrecisionType::HIGH_PRECISION);
                } else if (TILING_KEY_IS(711300UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR, CubeFormat::ND, HighPreciseType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(710300UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_TENSOR, CubeFormat::ND, HighPreciseType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(1000101000000003020UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR, CubeFormat::ND, HighPerformanceType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(1000101000000003000UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_TENSOR, CubeFormat::ND, HighPerformanceType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(911300UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2TilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatmulV2MixSplitKKernel<bfloat16_t, int8_t, float, bfloat16_t, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                }
            #else
                if (TILING_KEY_IS(810200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, false,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(811200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, false,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(810210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(811210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(810201UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, true, false,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(811201UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, true, false,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(810211UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, true, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(811211UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, true, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(810300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(811300UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(810310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, true,
                        QuantType::PER_GROUP, false, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(811310UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(WeightQuantBatchMatmulV2CustomWeightNzKernel, false, true,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR);
                } else if (TILING_KEY_IS(8611200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, false,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(8610200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, false,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(8611210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(8610210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(1000111000000003021UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_GROUP, CubeFormat::NZ, HighPerformanceType);
                } else if (TILING_KEY_IS(1000110000000003021UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_GROUP, CubeFormat::NZ, HighPreciseType);
                } else if (TILING_KEY_IS(1000111000000003001UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_GROUP, CubeFormat::NZ, HighPerformanceType);
                } else if (TILING_KEY_IS(1000110000000003001UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_GROUP, CubeFormat::NZ, HighPreciseType);
                } else if (TILING_KEY_IS(28611210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::NZ,
                        PrecisionType::HIGH_PRECISION);
                } else if (TILING_KEY_IS(28610210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::NZ,
                        PrecisionType::HIGH_PRECISION);
                } else if (TILING_KEY_IS(18611200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(18610200UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, false,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(18611210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, true,
                        QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(18610210UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(WeightQuantBatchMatmulV2MsdSplitKKernel, false, true,
                        QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::NZ);
                } else if (TILING_KEY_IS(1000100000000003021UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR, CubeFormat::NZ, HighPreciseType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(1000100000000003001UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_TENSOR, CubeFormat::NZ, HighPreciseType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(1000101000000003021UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, true, QuantType::PER_TENSOR, CubeFormat::NZ, HighPerformanceType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(1000101000000003001UL)) {
                    GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
                    WeightQuantBatchMatMulV2MsdGroupKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false,
                        QuantType::PER_GROUP, false, QuantType::PER_TENSOR, CubeFormat::NZ, HighPerformanceType> op;
                    op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS,
                            &tilingDataIn, &tPipe);
                    op.Process();
                } else if (TILING_KEY_IS(1000010000000012001UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_CUSTOM_SPLITK_IMPL(WeightQuantBatchMatmulV2CustomNzSplitkKernel,
                        false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, false);
                } else if (TILING_KEY_IS(1000010000000012021UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_CUSTOM_SPLITK_IMPL(WeightQuantBatchMatmulV2CustomNzSplitkKernel,
                        false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, false);
                } else if (TILING_KEY_IS(1000010001000012001UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_CUSTOM_SPLITK_IMPL(WeightQuantBatchMatmulV2CustomNzSplitkKernel,
                        false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, true);
                } else if (TILING_KEY_IS(1000010001000012021UL)) {
                    INVOKE_WEIGHT_QUANT_BMM_OP_CUSTOM_SPLITK_IMPL(WeightQuantBatchMatmulV2CustomNzSplitkKernel,
                        false, true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, true);
                }
            #endif
        #endif
    #else
        if (TILING_KEY_IS(80010)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, false, true,
                QuantType::PER_TENSOR, false);
        } else if (TILING_KEY_IS(80011)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, true, true,
                QuantType::PER_TENSOR, false);
        } else if (TILING_KEY_IS(80020)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, false, true,
                QuantType::PER_CHANNEL, false);
        } else if (TILING_KEY_IS(80021)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, true, true,
                QuantType::PER_CHANNEL, false);
        } else if (TILING_KEY_IS(80110)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, false, true,
                QuantType::PER_TENSOR, true);
        } else if (TILING_KEY_IS(80111)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, true, true,
                QuantType::PER_TENSOR, true);
        } else if (TILING_KEY_IS(80120)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, false, true,
                QuantType::PER_CHANNEL, true);
        } else if (TILING_KEY_IS(80121)) {
            INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, true, true,
                QuantType::PER_CHANNEL, true);
        }
    #endif
}