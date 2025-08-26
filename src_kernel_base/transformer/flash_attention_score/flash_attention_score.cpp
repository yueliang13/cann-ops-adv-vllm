/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "flash_attention_score_empty_tensor.h"
#include "flash_attention_score_drop_mask_adapter.h"
#include "flash_attention_score_s1s2_bn2gs1.h"
#include "flash_attention_score_s1_bn2gs1.h"
#include "flash_attention_score_bn2gs1s2_b.h"
#include "flash_attention_var_len_score.h"

using namespace AscendC;

#ifdef __DAV_C220_CUBE__ // CUBE 实现

#define COPY_TILING_DATA(tiling)                                                                                       \
    GET_TILING_DATA_MEMBER(FlashAttentionScoreGeneralTilingData, bmm1TilingData, bmm1TilingDataVar, tiling);           \
    GET_TILING_DATA_MEMBER(FlashAttentionScoreGeneralTilingData, bmm2TilingData, bmm2TilingDataVar, tiling);           \
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData = nullptr;                                       \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar;

#define INVOKE_FA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                     \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,           \
                          bmm2tiling);                                                                                \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
    } while (0)

#else // VECTOR 实现

#define COPY_TILING_DATA(tiling)                                                                                       \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGeneralTilingData, tilingDataIn, tiling);                           \
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData = &tilingDataIn;                                 \
    const TCubeTiling *__restrict bmm1tiling = &(tilingData->bmm1TilingData);                                          \
    const TCubeTiling *__restrict bmm2tiling = &(tilingData->bmm2TilingData);

#define INVOKE_FA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Reset();                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                 \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,          \
                    softmaxOut, attentionOut, user, tilingData, &tPipe);                                               \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                 \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,          \
                    softmaxOut, attentionOut, user, tilingData, &tPipe);                                               \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Destroy();                                                                                           \
            TPipe tPipeOp;                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipeOp, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,     \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,          \
                    softmaxOut, attentionOut, user, tilingData, &tPipeOp);                                             \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,       \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,          \
                    softmaxOut, attentionOut, user, tilingData, &tPipe);                                               \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Destroy();                                                                                           \
            TPipe tPipeOp;                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipeOp, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,       \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,          \
                    softmaxOut, attentionOut, user, tilingData, &tPipeOp);                                             \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,       \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,          \
                    softmaxOut, attentionOut, user, tilingData, &tPipe);                                               \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    do {                                                                                                               \
        COPY_TILING_DATA(tiling);                                                                                      \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
        }                                                                                                              \
        tPipe.Reset();                                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                 \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
        op.UnpackInit(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, actualSeqLengths,              \
                      actualSeqLengthsKv, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe); \
        op.Process();                                                                                                  \
    } while (0)

#endif

extern "C" __global__ __aicore__ void
flash_attention_score(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                      __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                      __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                      __gm__ uint8_t *qStartIdx, __gm__ uint8_t *kvStartIdx, __gm__ uint8_t *softmaxMax,
                      __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut,
                      __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    set_mask_norm();

    if (TILING_KEY_IS(90) || TILING_KEY_IS(92) || TILING_KEY_IS(94)) {
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreTilingData, tiling_data_in, tiling);
        const FlashAttentionScoreTilingData *__restrict tiling_data = &tiling_data_in;
        if (TILING_KEY_IS(90)) {
            FlashAttentionScoreEmptyTensor<half> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
        } else if (TILING_KEY_IS(92)) {
            FlashAttentionScoreEmptyTensor<float> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
        } else if (TILING_KEY_IS(94)) {
            FlashAttentionScoreEmptyTensor<bfloat16_t> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
        }
        return;
    }

#if (ORIG_DTYPE_QUERY == DT_FLOAT16)             // 3
    if (TILING_KEY_IS(10000000000220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001000220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001000220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001000221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001000221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000002200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000002200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010002200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010002200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000002210130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002210132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000010002201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010002201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000002211130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002211132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000021130099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000021132099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, half, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000022430943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000022432943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001000220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001000220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001000221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001000221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000002200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000002200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010002200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010002200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000002210230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002210232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000010002201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010002201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000002211230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002211232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000021230099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000021232099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, half, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001000220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001000220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001000221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001000221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000002200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000002200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010002200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010002200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000002210330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002210332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000010002201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010002201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000002211330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002211332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000021330099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000021332099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000001220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001001220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001001220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001001221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001001221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000012200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000012200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010012200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010012200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000012210130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012210132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000010012201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010012201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000012211130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012211132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000121130099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000121132099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000122430943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000122432943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001001220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001001220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001001221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001001221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000012200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000012200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010012200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010012200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000012210230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012210232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000010012201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010012201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000012211230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012211232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000121230099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000121232099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001001220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001001220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001001221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001001221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000012200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000012200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010012200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010012200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000012210330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012210332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000010012201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010012201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000012211330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012211332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000121330099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000121332099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000010220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001010220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001010220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001010221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001010221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000102200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000102200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010102200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010102200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000102210130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102210132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000010102201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010102201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000102211130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102211132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001021130099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001021132099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001022430943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001022432943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001010220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001010220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001010221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001010221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000102200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000102200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010102200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010102200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000102210230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102210232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000010102201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010102201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000102211230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102211232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001021230099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001021232099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001010220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001010220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001010221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001010221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000102200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000102200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010102200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010102200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000102210330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102210332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000010102201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010102201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000102211330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102211332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001021330099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000001021332099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000011220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001011220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001011220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001011221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001011221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000112200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000112200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010112200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010112200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000112210130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112210132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000010112201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010112201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000112211130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112211132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001121130099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001121132099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001122430943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001122432943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001011220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001011220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001011221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001011221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000112200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000112200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010112200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010112200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000112210230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112210232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000010112201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010112201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000112211230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112211232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001121230099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001121232099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001011220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001011220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001011221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001011221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000112200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000112200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010112200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010112200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000112210330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112210332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000010112201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000010112201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000112211330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112211332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001121330099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000001121332099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000100220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000100221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001100220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001100220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001100221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001100221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001002200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000001002200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011002200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011002200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001002210130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002210132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000011002201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011002201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001002211130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002211132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010021130099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010021132099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010022430943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010022432943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000100221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001100220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001100220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001100221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001100221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001002200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000001002200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011002200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011002200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001002210230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002210232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000011002201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011002201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001002211230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002211232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010021230099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010021232099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000100220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000100221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001100220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001100220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001100221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001100221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001002200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000001002200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011002200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011002200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001002210330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002210332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000011002201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011002201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001002211330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002211332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010021330099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000010021332099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000101220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000101221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001101220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001101220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001101221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001101221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001012200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000001012200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011012200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011012200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001012210130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012210132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000011012201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011012201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001012211130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012211132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010121130099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010121132099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010122430943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010122432943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000101221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001101220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001101220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001101221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001101221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001012200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000001012200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011012200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011012200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001012210230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012210232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000011012201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011012201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001012211230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012211232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010121230099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010121232099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000101220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000101221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001101220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001101220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001101221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001101221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001012200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000001012200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011012200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011012200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001012210330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012210332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000011012201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011012201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001012211330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012211332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010121330099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000010121332099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000110220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000110221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001110220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001110220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001110221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001110221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001102200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000001102200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011102200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011102200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001102210130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102210132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000011102201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011102201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001102211130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102211132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011021130099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011021132099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011022430943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011022432943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000110221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001110220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001110220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001110221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001110221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001102200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000001102200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011102200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011102200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001102210230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102210232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000011102201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011102201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001102211230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102211232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011021230099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011021232099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000110220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000110221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001110220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001110220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001110221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001110221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001102200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000001102200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011102200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011102200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001102210330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102210332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000011102201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011102201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001102211330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102211332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011021330099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000011021332099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000111220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000111221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001111220130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001111220132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001111221130943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001111221132943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001112200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000001112200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011112200130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011112200132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001112210130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112210132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000011112201130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011112201132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001112211130953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112211132953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011121130099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011121132099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, half, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011122430943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011122432943UL)) { // VarLen SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000111221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001111220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001111220232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001111221230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001111221232943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001112200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000001112200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011112200230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011112200232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001112210230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112210232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000011112201230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011112201232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001112211230953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112211232953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011121230099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011121232099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, half, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000111220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000111221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001111220330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001111220332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001111221330943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001111221332943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001112200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000001112200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011112200330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011112200332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001112210330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112210332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(
                   10000011112201330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(
                   10000011112201332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001112211330953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112211332953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011121330099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000011121332099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, half, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) // 2
    // no pse, no attenmask, no dropout
    if (TILING_KEY_IS(10000000000220120943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000220122943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001000220120943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001000220122943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001000221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001000221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000002200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000002200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000010002200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010002200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000002210120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002210122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000010002201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010002201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000002211120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002211122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000021120099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000021122099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000022420943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000022422943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000220220943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000220222943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001000220220943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001000220222943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001000221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001000221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000002200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000002200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000010002200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010002200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000002210220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002210222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000010002201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010002201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000002211220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002211222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000021220099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000021222099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000220320943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000220322943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001000220320943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001000220322943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001000221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001000221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000002200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000002200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000010002200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010002200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000002210320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002210322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000010002201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010002201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000002211320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002211322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000021320099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000021322099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000001220120943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001220122943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001001220120943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001001220122943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001001221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001001221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000012200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000012200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000010012200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010012200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000012210120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012210122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000010012201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010012201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000012211120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012211122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000121120099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000121122099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000122420943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000122422943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001220220943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001220222943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001001220220943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001001220222943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001001221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001001221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000012200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000012200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000010012200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010012200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000012210220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012210222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000010012201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010012201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000012211220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012211222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000121220099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000121222099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001220320943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001220322943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001001220320943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001001220322943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001001221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001001221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000012200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000012200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000010012200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010012200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000012210320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012210322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000010012201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010012201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000012211320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012211322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000121320099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000121322099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000010220120943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010220122943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001010220120943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001010220122943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001010221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001010221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000102200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000102200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000010102200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010102200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000102210120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102210122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000010102201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010102201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000102211120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102211122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001021120099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001021122099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001022420943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001022422943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010220220943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010220222943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001010220220943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001010220222943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001010221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001010221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000102200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000102200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000010102200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010102200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000102210220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102210222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000010102201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010102201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000102211220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102211222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001021220099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001021222099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010220320943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010220322943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001010220320943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001010220322943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001010221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001010221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000102200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000102200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000010102200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010102200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000102210320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102210322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000010102201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010102201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000102211320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102211322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001021320099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000001021322099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000011220120943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011220122943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001011220120943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001011220122943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001011221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001011221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000112200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000112200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000010112200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010112200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000112210120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112210122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000010112201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010112201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000112211120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112211122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001121120099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001121122099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001122420943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001122422943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011220220943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011220222943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001011220220943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001011220222943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001011221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001011221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000112200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000112200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000010112200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010112200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000112210220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112210222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000010112201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010112201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000112211220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112211222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001121220099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001121222099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011220320943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011220322943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001011220320943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001011220322943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001011221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001011221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000000112200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000112200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000010112200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010112200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000112210320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112210322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000010112201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000010112201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000000112211320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112211322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001121320099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000001121322099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000100220120943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100220122943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000100221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001100220120943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001100220122943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001100221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001100221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;

    } else if (TILING_KEY_IS(10000001002200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000001002200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000011002200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011002200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001002210120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002210122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000011002201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011002201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001002211120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002211122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010021120099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010021122099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010022420943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010022422943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100220220943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100220222943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000100221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001100220220943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001100220222943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001100221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001100221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001002200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000001002200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000011002200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011002200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001002210220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002210222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000011002201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011002201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001002211220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002211222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010021220099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010021222099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000100220320943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100220322943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000100221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001100220320943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001100220322943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001100221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001100221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001002200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000001002200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000011002200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011002200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001002210320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002210322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000011002201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011002201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001002211320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002211322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010021320099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000010021322099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000101220120943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101220122943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000101221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001101220120943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001101220122943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001101221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001101221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001012200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000001012200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000011012200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011012200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001012210120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012210122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000011012201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011012201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001012211120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012211122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010121120099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010121122099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010122420943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010122422943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101220220943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101220222943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000101221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001101220220943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001101220222943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001101221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001101221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001012200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000001012200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000011012200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011012200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001012210220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012210222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000011012201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011012201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001012211220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012211222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010121220099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010121222099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000101220320943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101220322943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000101221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001101220320943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001101220322943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001101221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001101221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001012200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000001012200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000011012200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011012200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001012210320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012210322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000011012201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011012201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001012211320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012211322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010121320099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000010121322099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000110220120943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110220122943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000110221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001110220120943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001110220122943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001110221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001110221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001102200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000001102200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000011102200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011102200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001102210120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102210122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000011102201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011102201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001102211120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102211122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011021120099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011021122099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011022420943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011022422943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110220220943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110220222943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000110221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001110220220943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001110220222943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001110221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001110221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001102200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000001102200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000011102200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011102200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001102210220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102210222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000011102201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011102201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001102211220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102211222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011021220099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011021222099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000110220320943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110220322943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000110221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001110220320943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001110220322943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001110221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001110221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001102200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000001102200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000011102200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011102200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001102210320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102210322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000011102201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011102201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001102211320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102211322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011021320099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000011021322099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000111220120943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111220122943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000111221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001111220120943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001111220122943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001111221120943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001111221122943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001112200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000001112200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000011112200120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011112200122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001112210120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112210122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000011112201120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011112201122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001112211120953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112211122953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011121120099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011121122099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, bfloat16_t, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011122420943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011122422943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111220220943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111220222943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000111221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001111220220943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001111220222943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000001111221220943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001111221222943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001112200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000001112200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000011112200220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011112200222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true, CubeFormat::ND,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001112210220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112210222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true, CubeFormat::ND,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000011112201220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011112201222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true, CubeFormat::NZ,
                                  TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001112211220953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112211222953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true, CubeFormat::NZ,
                                  TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011121220099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011121222099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, bfloat16_t, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000111220320943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111220322943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000111221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001111220320943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001111220322943UL)) { // SplitS1S2HighPerf: BF16 s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001111221320943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001111221322943UL)) { // SplitS1S2HighPerf: BF16 Bmm1-NZ s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                         CubeFormat::NZ, true);
        return;
    } else if (TILING_KEY_IS(10000001112200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000001112200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000011112200320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011112200322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001112210320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112210322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-ND L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000011112201320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000011112201322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ new_L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::GM, CubeFormat::ND, true);
        return;
    } else if (TILING_KEY_IS(10000001112211320953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112211322953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16 Bmm1-NZ L1reuse
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011121320099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000011121322099UL)) { // SplitBbDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, bfloat16_t, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT) // 1
    if (TILING_KEY_IS(10000000000220110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000220112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000221110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000221112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000220210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000220212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000221210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000221212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000220310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000220312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000221310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000221312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001220110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001220112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001221110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001221112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, false, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001220210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001220212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001221210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001221212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, false, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001220310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001220312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001221310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000001221312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, false, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010220110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010220112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010221110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010221112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010220210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010220212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010221210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010221212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010220310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010220312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010221310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000010221312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011220110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011220112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011221110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011221112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, false, true, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011220210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011220212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011221210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011221212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, false, true, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011220310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011220312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011221310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000011221312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, false, true, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000100220110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100220112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100221110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000100221112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000100220210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100220212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100221210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000100221212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000100220310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100220312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000100221310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000100221312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000101220110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101220112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101221110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000101221112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, false, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000101220210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101220212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101221210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000101221212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, false, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000101220310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101220312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000101221310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000101221312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, false, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000110220110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110220112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110221110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000110221112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000110220210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110220212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110221210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000110221212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000110220310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110220312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000110221310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000110221312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, false, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000111220110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111220112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111221110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000111221112943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BSH, true, true, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000111220210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111220212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111221210943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000111221212943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_SBH, true, true, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000111220310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111220312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000111221310943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000111221312943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                         LayOutTypeEnum::LAYOUT_BNSD, true, true, true, float, float, true,
                                         CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002200110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
                                  false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000002200112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
                                  false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000002201110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, false, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, false, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002200210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000002200212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000002201210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, false, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, false, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002200310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000002200312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000002201310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, false, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000002201312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, false, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012200110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000012200112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000012201110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, false, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, false, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012200210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000012200212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000012201210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, false, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, false, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012200310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000012200312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000012201310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, false, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000012201312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, false, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102200110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000102200112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000102201110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, true, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, true, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102200210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000102200212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000102201210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, true, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, true, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102200310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000102200312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000102201310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, true, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000102201312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, true, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112200110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000112200112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000112201110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, true, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  false, true, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112200210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000112200212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000112201210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, true, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  false, true, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112200310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000112200312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000112201310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, true, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000112201312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  false, true, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002200110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001002200112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001002201110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, false, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, false, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002200210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001002200212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001002201210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, false, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, false, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002200310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001002200312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001002201310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, false, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001002201312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, false, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012200110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001012200112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001012201110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, false, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, false, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012200210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001012200212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001012201210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, false, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, false, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012200310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001012200312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
                                  true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001012201310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, false, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001012201312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, false, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102200110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001102200112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001102201110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, true, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, true, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102200210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001102200212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001102201210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, true, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, true, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102200310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001102200312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001102201310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, true, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001102201312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, true, false, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112200110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001112200112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001112201110953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, true, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201112953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BSH,
								  true, true, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112200210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001112200212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001112201210953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, true, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201212953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_SBH,
								  true, true, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112200310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001112200312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-ND
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000001112201310953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, true, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000001112201312953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT32_PRECISION Bmm1-NZ
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                                  ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION, LayOutTypeEnum::LAYOUT_BNSD,
								  true, true, true, float, float, true, CubeFormat::NZ);
        return;
    } else if (TILING_KEY_IS(10000000000021110099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, float, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000021112099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, float, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000021210099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, float, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000021212099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, false, float, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000021310099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000021312099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000121110099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, float, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000121112099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, true, float, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000121210099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, float, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000121212099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, false, true, float, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000121310099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000121312099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, true, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000001021110099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, float, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001021112099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, false, float, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001021210099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, float, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001021212099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, float, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001021310099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000001021312099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000001121110099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, float, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001121112099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, true, true, float, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001121210099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, float, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001121212099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, true, float, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001121310099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000001121312099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, true, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000010021110099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, float, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010021112099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, false, float, float, true,
                                  LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010021210099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, float, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010021212099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, false, float, float, true,
                                  LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010021310099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000010021312099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, false, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000010121110099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, float, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010121112099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, false, true, float, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010121210099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, float, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010121212099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, false, true, float, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010121310099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000010121312099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, false, true, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000011021110099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, float, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011021112099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, false, float, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011021210099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, float, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011021212099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, false, float, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011021310099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000011021312099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, false, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000011121110099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, float, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011121112099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, true, true, true, float, float, true, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011121210099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, float, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011121212099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, true, true, true, float, float, true, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011121310099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000011121312099UL)) { // SplitBbDBHighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, true, true, true, float, float, true,
                                  LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000022410943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000022412943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000122410943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000122412943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001022410943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001022412943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001122410943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000001122412943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, false, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010022410943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010022412943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, false, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010122410943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000010122412943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, false, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011022410943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011022412943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, true, false, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011122410943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, true, true, float, float, true);
        return;
    } else if (TILING_KEY_IS(10000000011122412943UL)) { // VarLen SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION,
                                          LayOutTypeEnum::LAYOUT_TND, true, true, true, float, float, true);
        return;
    }
#endif
}
