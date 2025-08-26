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
 * \file flash_attention_score_grad.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "flash_attention_score_grad_empty_tensor.h"
#include "flash_attention_score_grad_post.h"
#include "flash_attention_score_grad_s1s2_bn2gs1s2.h"
#include "flash_attention_score_grad_pre.h"
#include "flash_attention_score_grad_sfmg.h"
#include "flash_attention_score_grad_s1s2_bn2.h"
#include "flash_attention_score_grad_ngs1s2_bn.h"
#include "flash_attention_score_grad_bngs1s2_b.h"
#include "flash_attention_score_grad_s1s2_bn2gs1s2_sab.h"

constexpr MatmulConfig MM_CFG_EXCEED = GetNormalConfig(true);
constexpr MatmulConfig MM_CFG_NORMAL = GetNormalConfig(false);
constexpr CubeFormat MM_NZ_OUT_FORMAT = CubeFormat::NZ;
constexpr CubeFormat MM_ND_OUT_FORMAT = CubeFormat::ND_ALIGN;
constexpr CubeFormat MM_ND_OUT_NOALIGN = CubeFormat::ND;
constexpr uint64_t INPUT_NONE = 0;
constexpr uint64_t INPUT_EXIST = 1;
constexpr uint32_t INPUT_DISABLE = 0;
constexpr uint32_t INPUT_ENABLE = 1;

constexpr static uint32_t ND = 0;
constexpr static uint32_t NZ = 1;

constexpr static const uint32_t BNGSD = 0;
constexpr static const uint32_t SBNGD = 1;
constexpr static const uint32_t BSNGD = 2;
constexpr static const uint32_t TND = 3;

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT, \
                                              MM2_OUT_FORMAT)                                                          \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, tiling_data_in, tiling_data);       \
        const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict tilingData = &tiling_data_in;                  \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm2tiling = &(tilingData->mm2TilingData);                                       \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm3TilingData);                                       \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, true> opPre;      \
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
                                                                                                                       \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradS1s2Bn2gs1s2<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,          \
                                            INPUT_LAYOUT, MM2_OUT_FORMAT>                                              \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm3, bmm2tiling, op.mm4,             \
                          bmm3tiling);                                                                                 \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum,       \
                prefix, actual_seq_qlen, actual_seq_kvlen, dq, dk, dv, dpse, user, tilingData, &pipeBase);             \
        op.Process();                                                                                                  \
        op.SyncALLCores();                                                                                             \
        pipeBase.Destroy();                                                                                            \
        TPipe pipePost;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, true, INPUT_LAYOUT,     \
                                    input_format>                                                                      \
            opPost;                                                                                                    \
        opPost.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipePost);                       \
        opPost.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,        \
                                              INPUT_LAYOUT, MM2_OUT_FORMAT, IS_DTM)                                    \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, tiling_data_in, tiling_data); \
        const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *__restrict tilingData = &tiling_data_in;            \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm2tiling = &(tilingData->mm2TilingData);                                       \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm3TilingData);                                       \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, true> opPre;\
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        pipeIn.Destroy();                                                                                              \
        if ASCEND_IS_AIV {                                                                                             \
            TPipe pipeSfmg;                                                                                            \
            FlashAttentionScoreGradSfmg<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb,        \
                INPUT_LAYOUT> opSfmg;                                                                                  \
            opSfmg.Init(dy, attention_in, actual_seq_qlen, dq, dk, dv, drop_mask, user, tilingData, &pipeSfmg);        \
            opSfmg.Process();                                                                                          \
            pipeSfmg.Destroy();                                                                                        \
        }                                                                                                              \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,    \
                                            INPUT_LAYOUT, MM2_OUT_FORMAT, IS_DTM> op;                                  \
        REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm3, bmm2tiling, op.mm4,             \
                          bmm3tiling);                                                                                 \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum,       \
                prefix, actual_seq_qlen, actual_seq_kvlen, dq, dk, dv, dpse, user, tilingData);                        \
        op.ProcessFirstMM();                                                                                           \
        op.InitBuffer(&pipeBase);                                                                                      \
        op.Process();                                                                                                  \
        op.SyncALLCores();                                                                                             \
        pipeBase.Destroy();                                                                                            \
        TPipe pipePost;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, true, INPUT_LAYOUT,\
                                    input_format>                                                                      \
            opPost;                                                                                                    \
        opPost.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipePost);                       \
        opPost.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(INPUT_TYPE, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,     \
                                         INPUT_LAYOUT, MM2_OUT_FORMAT)                                                 \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2, tiling_data_in, tiling_data);            \
        const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData = &tiling_data_in;                       \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm31tiling = &(tilingData->mm31TilingData);                                     \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm4TilingData);                                       \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2, false> opPre;          \
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeOp;                                                                                                  \
        FlashAttentionScoreGradS1s2Bn2<INPUT_TYPE, float, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,             \
                                       DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT>                                      \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm4, bmm4tiling, op.mm3_1,             \
                          bmm31tiling);                                                                                \
        op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,       \
                prefix, softmax_in, actual_seq_qlen, actual_seq_kvlen, attention_in, dq, dk, dv, dpse, user,           \
                tilingData, &pipeOp);                                                                                  \
        op.Process();                                                                                                  \
        pipeOp.Destroy();                                                                                              \
        TPipe pipeCast;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2, true, INPUT_LAYOUT,          \
        input_format> opCast;                                                                                          \
        opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeCast);                       \
        opCast.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(INPUT_TYPE, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,       \
                                                    DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT)                         \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2, tiling_data_in, tiling_data);            \
        const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData = &tiling_data_in;                       \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm31tiling = &(tilingData->mm31TilingData);                                     \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm4TilingData);                                       \
        FlashAttentionScoreGradS1s2Bn2<INPUT_TYPE, float, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,             \
                                       DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT>                                      \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm4, bmm4tiling, op.mm3_1,             \
                          bmm31tiling);                                                                                \
        op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,       \
                prefix, softmax_in, actual_seq_qlen, actual_seq_kvlen, attention_in, dq, dk, dv, dpse, user,           \
                tilingData, &pipeIn);                                                                                  \
        op.Process();                                                                                                  \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeCast;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2, true, INPUT_LAYOUT,          \
        input_format> opCast;                                                                                          \
        opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeCast);                       \
        opCast.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(INPUT_TYPE, INPUT_LAYOUT, layout, MM_CONFIG, MM_OUT_FORMAT,             \
                                               MM2_OUT_FORMAT)                                                         \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradUbngs1s2BbTilingData, tiling_data_in, tiling_data);         \
        const FlashAttentionScoreGradUbngs1s2BbTilingData *__restrict tilingData = &tiling_data_in;                    \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradUbngs1s2BbTilingData, false> opPre;       \
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeOp;                                                                                                  \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1AndMm2TilingData);                                 \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm31TilingData);                                      \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm32AndMm4TilingData);                                \
        FlashAttentionScoreGradUngs1s2Bb<INPUT_TYPE, float, MM_CONFIG, INPUT_LAYOUT, MM_OUT_FORMAT, MM2_OUT_FORMAT> op;\
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm31, bmm3tiling,                      \
                          op.mm32, bmm4tiling, op.mm4, bmm4tiling);                                                    \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq,   \
                dk, dv, user, tilingData, &pipeOp);                                                                    \
        op.Process();                                                                                                  \
        pipeOp.Destroy();                                                                                              \
        TPipe pipeMuls;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradUbngs1s2BbTilingData, false,                    \
        layout, input_format> opMuls;                                                                                  \
        opMuls.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeMuls);                       \
        opMuls.Process();                                                                                              \
        pipeMuls.Destroy();                                                                                            \
    } while (0)

#define INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(INPUT_TYPE, INPUT_LAYOUT, layout, MM_CONFIG, MM_OUT_FORMAT, MM2_OUT_FORMAT)  \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataUngs1s2Bbn, tiling_data_in, tiling_data);         \
        const FlashAttentionScoreGradTilingDataUngs1s2Bbn *__restrict tilingData = &tiling_data_in;                    \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataUngs1s2Bbn, false> opPre;       \
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeOp;                                                                                                  \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1AndMm2TilingData);                                 \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm31TilingData);                                      \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm32AndMm4TilingData);                                \
        FlashAttentionScoreGradUngs1s2Bbn<INPUT_TYPE, float, MM_CONFIG, true, INPUT_LAYOUT, MM_OUT_FORMAT,             \
                                          MM2_OUT_FORMAT> op;                                                          \
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm31, bmm3tiling,                      \
                          op.mm32, bmm4tiling, op.mm4, bmm4tiling);                                                    \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq,   \
                dk, dv, user, tilingData, &pipeOp);                                                                    \
        op.Process();                                                                                                  \
        pipeOp.Destroy();                                                                                              \
        TPipe pipeMuls;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataUngs1s2Bbn, false,                    \
        layout, input_format> opMuls;                                                                                  \
        opMuls.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeMuls);                       \
        opMuls.Process();                                                                                              \
        pipeMuls.Destroy();                                                                                            \
    } while (0)

// implementation of kernel function
extern "C" __global__ __aicore__ void flash_attention_score_grad(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dy, __gm__ uint8_t *pse_shift,
    __gm__ uint8_t *drop_mask, __gm__ uint8_t *padding_mask, __gm__ uint8_t *atten_mask, __gm__ uint8_t *softmax_max,
    __gm__ uint8_t *softmax_sum, __gm__ uint8_t *softmax_in, __gm__ uint8_t *attention_in, __gm__ uint8_t *prefix,
    __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen, __gm__ uint8_t *q_start_idx,
    __gm__ uint8_t *kv_start_idx, __gm__ uint8_t *dq, __gm__ uint8_t *dk,
    __gm__ uint8_t *dv, __gm__ uint8_t *dpse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data) {
    TPipe pipeIn;
    set_mask_norm();
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

// --------------------------------------------float16 tilingkey------------------------------------------------------
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    // -----------------------SameAB start---------------------------------
    if (TILING_KEY_IS(10000001000111003434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000011003434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000101003434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000001003434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000110003434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000010003434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000100003434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000000003434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001111003434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001011003434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001101003434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001001003434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001110003434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001010003434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001100003434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001000003434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;

        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000001010111003434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010011003434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010101003434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010001003434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010110003434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010010003434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010100003434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010000003434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011111003434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011011003434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011101003434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011001003434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011110003434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011010003434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011100003434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011000003434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;

        // 1 格式为SBNGD
    } else if (TILING_KEY_IS(10000001000111013434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000011013434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000101013434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000001013434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000110013434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000010013434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000100013434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000000013434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001111013434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001011013434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001101013434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001001013434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001110013434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001010013434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001100013434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001000013434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;

        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000001010111013434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010011013434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010101013434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010001013434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010110013434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010010013434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010100013434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010000013434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011111013434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011011013434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011101013434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011001013434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011110013434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011010013434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011100013434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011000013434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;

        // 2 格式为BNGSD
    } else if (TILING_KEY_IS(10000001000111023434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000011023434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000101023434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000001023434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000110023434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000010023434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000100023434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000000023434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001111023434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001011023434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001101023434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001001023434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001110023434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001010023434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001100023434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001000023434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000001010111023434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010011023434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010101023434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010001023434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010110023434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010010023434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010100023434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010000023434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011111023434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011011023434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011101023434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011001023434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011110023434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011010023434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011100023434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011000023434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;

        // 3 格式为TND
    } else if (TILING_KEY_IS(10000001000111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000001010111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;

     // ---mm1 nz
    } else if (TILING_KEY_IS(10000001001111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
        // --- mm1 mm2 out Nz
    } else if (TILING_KEY_IS(10000001011111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;

        // 确定性计算
    } else if (TILING_KEY_IS(10000001100111003434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100011003434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100101003434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100001003434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100110003434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100010003434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100100003434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100000003434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101111003434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101011003434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101101003434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101001003434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101110003434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101010003434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101100003434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101000003434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;

        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000001110111003434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110011003434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110101003434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110001003434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110110003434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110010003434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110100003434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110000003434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111111003434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111011003434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111101003434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111001003434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111110003434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111010003434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111100003434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111000003434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;

        // 1 格式为SBNGD
    } else if (TILING_KEY_IS(10000001100111013434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100011013434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100101013434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100001013434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100110013434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100010013434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100100013434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100000013434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101111013434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101011013434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101101013434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101001013434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101110013434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101010013434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101100013434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101000013434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;

        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000001110111013434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110011013434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110101013434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110001013434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110110013434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110010013434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110100013434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110000013434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111111013434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111011013434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111101013434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111001013434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111110013434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111010013434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111100013434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111000013434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;

        // 2 格式为BNGSD
    } else if (TILING_KEY_IS(10000001100111023434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100011023434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100101023434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100001023434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100110023434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100010023434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100100023434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100000023434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101111023434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101011023434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101101023434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101001023434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101110023434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101010023434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101100023434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101000023434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000001110111023434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110011023434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110101023434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110001023434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110110023434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110010023434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110100023434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110000023434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111111023434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111011023434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111101023434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111001023434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111110023434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111010023434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111100023434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111000023434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;

        // 3 格式为TND
    } else if (TILING_KEY_IS(10000001100111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000001110111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;

    // --- mm1 out Nz mm2 ND
    } else if (TILING_KEY_IS(10000001101111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
        // --- mm1 and mm2 out Nz
    } else if (TILING_KEY_IS(10000001111111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    // -----------------------1.1 start---------------------------------
    // 格式为TND
    } else if (TILING_KEY_IS(10000000000111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000000010111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:Nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000000011001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000001000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:Nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
        // --- mm2 out Nd
    } else if (TILING_KEY_IS(10000000001001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
        // -----------------------1.1 end---------------------------------

        // -----------------------1.2 start---------------------------------
        // For BSNGD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // For SBNGD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // For BNGSD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // for TND
    } else if (TILING_KEY_IS(10000000000000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         TND, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         TND, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         TND, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         TND, MM_ND_OUT_NOALIGN);
        return;
        // For mm345 out
        // For BSNGD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000100010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // For SBNGD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000100010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // For BNGSD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000100010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // for TND
    } else if (TILING_KEY_IS(10000000100000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         TND, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         TND, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         TND, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         TND, MM_NZ_OUT_FORMAT);
        return;
        // -----------------------1.2 end---------------------------------

        // -----------------------3.1 start---------------------------------
    } else if (TILING_KEY_IS(10000000000000003199UL)) { // BSH BSNGD & FLOAT16_PRECISION
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000013199UL)) { // SBNGD & FLOAT16_PRECISION
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000023199UL)) { // BNGSD & FLOAT16_PRECISION
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001013199UL)) { // SBNGD & FLOAT16_PRECISION & speical MM tilingkey
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                          MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    // mm12 nzout
    } else if (TILING_KEY_IS(10000000000010003199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                          MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010013199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                          MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010023199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                          MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011013199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                          MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    // mm345 nzout
    } else if (TILING_KEY_IS(10000000000100003199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000100013199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000100023199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000101013199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                          MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    // mm all nzout
    } else if (TILING_KEY_IS(10000000000110003199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                          MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000110013199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                          MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000110023199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                          MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000111013199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                          MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;

        // -----------------------3.1 end---------------------------------

        // -----------------------4.1 start---------------------------------
    } else if (TILING_KEY_IS(10000000000000103099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000113099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000123099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000133099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001113099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    // mm12 nzout
    } else if (TILING_KEY_IS(10000000000010103099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010113099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010123099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010133099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011113099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                               MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    // mm345 nzout
    } else if (TILING_KEY_IS(10000000000100103099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000100113099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000100123099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000100133099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000101113099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                               MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    // mm all nzout
    } else if (TILING_KEY_IS(10000000000110103099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000110113099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000110123099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000110133099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000111113099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                               MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    }
        // -----------------------4.1 end---------------------------------
#endif
// --------------------------------------------------------------------------------------------------------------------

// --------------------------------------------bfloat16 tilingkey------------------------------------------------------
#if (ORIG_DTYPE_QUERY == DT_BF16)
    // -------------------SameAB----------------
    if (TILING_KEY_IS(10000001000111002434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000011002434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000101002434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000001002434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000110002434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000010002434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000100002434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000000002434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001111002434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001011002434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001101002434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001001002434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001110002434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001010002434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001100002434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001000002434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;

        // mm2 out nz
    } else if (TILING_KEY_IS(10000001010111002434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010011002434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010101002434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010001002434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010110002434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010010002434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010100002434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010000002434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011111002434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011011002434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011101002434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011001002434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011110002434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011010002434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011100002434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011000002434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
        // 1 for layout SBNGD
    } else if (TILING_KEY_IS(10000001000111012434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000011012434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000101012434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000001012434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000110012434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000010012434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000100012434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000000012434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001111012434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001011012434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001101012434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001001012434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001110012434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001010012434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001100012434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001000012434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;

        // mm2 out nz
    } else if (TILING_KEY_IS(10000001010111012434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010011012434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010101012434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010001012434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010110012434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010010012434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010100012434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010000012434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011111012434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011011012434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011101012434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011001012434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011110012434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011010012434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011100012434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011000012434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
        // 2 for layout BNGSD
    } else if (TILING_KEY_IS(10000001000111022434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000011022434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000101022434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000001022434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000110022434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000010022434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000100022434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000000022434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001111022434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001011022434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001101022434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001001022434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001110022434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001010022434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001100022434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001000022434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
        // mm2 out nz
    } else if (TILING_KEY_IS(10000001010111022434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010011022434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010101022434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010001022434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010110022434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010010022434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010100022434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010000022434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011111022434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011011022434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011101022434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011001022434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011110022434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011010022434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011100022434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011000022434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
        // 3 for TND layout
    } else if (TILING_KEY_IS(10000001000111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001000000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
        // mm2 out nz
    } else if (TILING_KEY_IS(10000001010111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001010000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;

    // mm1 out Nz mm2 Nd
    } else if (TILING_KEY_IS(10000001001111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001001000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_NZ_OUT_FORMAT, TND, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
        // mm1 and mm2 out nz
    } else if (TILING_KEY_IS(10000001011111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000001011000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_NZ_OUT_FORMAT, TND, MM_NZ_OUT_FORMAT, INPUT_DISABLE);
        return;

        // 确定性计算
    } else if (TILING_KEY_IS(10000001100111002434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100011002434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100101002434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100001002434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100110002434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100010002434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100100002434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100000002434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101111002434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101011002434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101101002434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101001002434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101110002434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101010002434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101100002434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101000002434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;

        // mm2 out nz
    } else if (TILING_KEY_IS(10000001110111002434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110011002434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110101002434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110001002434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110110002434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110010002434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110100002434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110000002434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111111002434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111011002434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111101002434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111001002434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111110002434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111010002434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111100002434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111000002434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
        // 1 for layout SBNGD
    } else if (TILING_KEY_IS(10000001100111012434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100011012434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100101012434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100001012434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100110012434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100010012434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100100012434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100000012434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101111012434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101011012434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101101012434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101001012434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101110012434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101010012434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101100012434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101000012434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;

        // mm2 out nz
    } else if (TILING_KEY_IS(10000001110111012434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110011012434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110101012434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110001012434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110110012434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110010012434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110100012434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110000012434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111111012434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111011012434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111101012434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111001012434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111110012434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111010012434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111100012434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111000012434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
        // 2 for layout BNGSD
    } else if (TILING_KEY_IS(10000001100111022434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100011022434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100101022434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100001022434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100110022434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100010022434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100100022434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100000022434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101111022434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101011022434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101101022434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101001022434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101110022434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101010022434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101100022434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101000022434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
        // mm2 out nz
    } else if (TILING_KEY_IS(10000001110111022434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110011022434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110101022434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110001022434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110110022434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110010022434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110100022434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110000022434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111111022434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111011022434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111101022434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111001022434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111110022434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111010022434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111100022434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111000022434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
        // 3 for TND layout
    } else if (TILING_KEY_IS(10000001100111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001100000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
        // mm2 out nz
    } else if (TILING_KEY_IS(10000001110111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001110000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001101000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_NZ_OUT_FORMAT, TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
        // mm2 out nz
    } else if (TILING_KEY_IS(10000001111111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;
    } else if (TILING_KEY_IS(10000001111000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_NZ_OUT_FORMAT, TND, MM_NZ_OUT_FORMAT, INPUT_ENABLE);
        return;

    // -----------------------1.1 start---------------------------------
    // for TND layout
    } else if (TILING_KEY_IS(10000000001000032434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:Nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
        // --- mm2 out Nd
    } else if (TILING_KEY_IS(10000000001001032434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010032434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011000032434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:Nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
        // --- mm2 out Nd
    } else if (TILING_KEY_IS(10000000011001032434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010032434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, TND, MM_ND_OUT_NOALIGN);
        return;
        // mm2 out nz
    } else if (TILING_KEY_IS(10000000010111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, TND, MM_NZ_OUT_FORMAT);
        return;
        // -----------------------1.1 end---------------------------------

        // -----------------------1.2 start-------------------------------
        // For BSNGD
    } else if (TILING_KEY_IS(10000000000010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // For SBNGD
    } else if (TILING_KEY_IS(10000000000010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000001011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000001111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // For BNGSD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // for TND
    } else if (TILING_KEY_IS(10000000000000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_ND_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND, MM_ND_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_ND_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_ND_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_ND_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_ND_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_ND_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_ND_OUT_FORMAT);
        return;
       // for mm345 NZ out
        // -----------------------1.2 start-------------------------------
        // For BSNGD
    } else if (TILING_KEY_IS(10000000100010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // For SBNGD
    } else if (TILING_KEY_IS(10000000100010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // For BNGSD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000100010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // for TND
    } else if (TILING_KEY_IS(10000000100000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_NZ_OUT_FORMAT);
        return;
        // -----------------------1.2 end---------------------------------
        // -----------------------3.1 start---------------------------------
    } else if (TILING_KEY_IS(10000000000000002199UL)) { // BSH BSNGD & BFLOAT16
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000012199UL)) { // SBNGD & BFLOAT16
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000022199UL)) { // BNGSD & BFLOAT16
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001012199UL)) { // SBNGD & BFLOAT16  & speical MM tilingkey
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                          MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    // mm12 nzout
    } else if (TILING_KEY_IS(10000000000010002199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                          MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010012199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                          MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010022199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                          MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011012199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                          MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    // mm345 nzout
    } else if (TILING_KEY_IS(10000000000100002199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000100012199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000100022199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000101012199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                          MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    // mm all nzout
    } else if (TILING_KEY_IS(10000000000110002199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                          MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000110012199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                          MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000110022199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                          MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000111012199UL)) {
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                          MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;

        // -----------------------3.1 end---------------------------------

        // -----------------------4.1 start---------------------------------
    } else if (TILING_KEY_IS(10000000000000102099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000112099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000122099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000132099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001112099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    // mm12 nzout
    } else if (TILING_KEY_IS(10000000000010102099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010112099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010122099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010132099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011112099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                               MM_NZ_OUT_FORMAT, MM_ND_OUT_NOALIGN);
        return;
    // mm345 nzout
    } else if (TILING_KEY_IS(10000000000100102099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000100112099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000100122099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000100132099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000101112099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                               MM_ND_OUT_NOALIGN, MM_NZ_OUT_FORMAT);
        return;
    // mm all nzout
    } else if (TILING_KEY_IS(10000000000110102099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000110112099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000110122099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000110132099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000111112099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                               MM_NZ_OUT_FORMAT, MM_NZ_OUT_FORMAT);
        return;
    }
        // -----------------------4.1 end---------------------------------
#endif
    // --------------------------------------------------------------------------------------------------------------------

// --------------------------------------------float32 tilingkey------------------------------------------------------
#if (ORIG_DTYPE_QUERY == DT_FLOAT)
    if (TILING_KEY_IS(10000001000000001434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    // -----------------------1.1 start---------------------------------
    } else if (TILING_KEY_IS(10000000000111001434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011001434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101001434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001001434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;

    } else if (TILING_KEY_IS(10000000000110001434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010001434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100001434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000001434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // 1
    } else if (TILING_KEY_IS(10000000000111011434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011011434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101011434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001011434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110011434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010011434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100011434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000011434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // 2
    } else if (TILING_KEY_IS(10000000000111021434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011021434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101021434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001021434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110021434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010021434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100021434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000021434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // 3
    } else if (TILING_KEY_IS(10000000000111031434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011031434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101031434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001031434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110031434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010031434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100031434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000031434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, TND, MM_ND_OUT_NOALIGN);
        return;
        // -----------------------1.1 end---------------------------------

        // -----------------------1.2 start-------------------------------
        // For BSNGD
    } else if (TILING_KEY_IS(10000000000000001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // For SBNGD
    } else if (TILING_KEY_IS(10000000000000011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_NOALIGN, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010001011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010101011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011001011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011101011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // For BNGSD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000000021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // for TND
    } else if (TILING_KEY_IS(10000000000000031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_ND_OUT_NOALIGN);
        return;
        // -----------------------1.2 end---------------------------------
    } else if (TILING_KEY_IS(10000000000000001199UL)) { // BSH BSNGD & FLOAT16_PRECISION
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(float, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000101099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(float, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    }
#endif

    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingData, tiling_data_in, tiling_data);
    const FlashAttentionScoreGradTilingData *__restrict tilingData = &tiling_data_in;

    if (TILING_KEY_IS(90)) {
        FlashAttentionScoreGradEmptyTensor<DTYPE_DQ> op;
        op.Init(dq, dk, dv, dpse, tilingData);
        op.Process();
    }
}
