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
 * \file flash_attention_score_grad_pre.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_PRE_KERNEL_H_
#define FLASH_ATTENTION_SCORE_GRAD_PRE_KERNEL_H_

#include "kernel_operator.h"

template <typename T1, typename T2, typename TILING_TYPE, const bool INIT_OUTPUT = true>
class FlashAttentionScoreGradPre {
public:
    __aicore__ inline FlashAttentionScoreGradPre(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *drop_mask,
                                __gm__ uint8_t *workspace, const TILING_TYPE *ordTilingData, TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> helpQue;
    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECIN, 1> castQue;
    TQue<QuePosition::VECOUT, 1> outQue;

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, dvGm;
    GlobalTensor<uint8_t> maskWorkSpaceGm;
    GlobalTensor<uint8_t> drop_maskGm;

    const TILING_TYPE *TilingData;
    constexpr static uint32_t HELP_LEN = 256;
    constexpr static uint32_t BIT8 = 8;
    constexpr static uint32_t NUMBER_8 = 8;
    constexpr static uint32_t B16_VECTOR_MASK = 128;

    uint32_t cBlockIdx;
    // query
    uint32_t ubBaseSize;
    uint32_t qPreBlockFactor;
    uint32_t qPreBlockTotal;
    uint32_t qPreBlockTail;
    uint32_t kvPreBlockFactor;
    uint32_t kvPreBlockTotal;
    uint32_t kvPreBlockTail;

    int64_t qSizeAlign;
    int64_t kvSizeAlign;

    int64_t initdqSize;
    int64_t dqOffset;
    int64_t initdkSize;
    int64_t dkvOffset;

    bool isDropBoolMode;
    uint32_t maskUsedCoreNum;
    uint32_t maskUBProcessNum;
    uint32_t maskTailUBProcessNum;
    uint32_t maskUBLoop;

    DataCopyParams copyParams;
    BinaryRepeatParams repParams;
    half padValue{1.0};
};

template <typename T1, typename T2, typename TILING_TYPE, const bool INIT_OUTPUT>
__aicore__ inline void FlashAttentionScoreGradPre<T1, T2, TILING_TYPE, INIT_OUTPUT>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *drop_mask, __gm__ uint8_t *workspace,
    const TILING_TYPE *orgTilingData, TPipe *pipe_in)
{
    cBlockIdx = GetBlockIdx();

    TilingData = orgTilingData;
    pipe = pipe_in;

    maskUsedCoreNum = TilingData->preTilingData.maskCoreNum;

    drop_maskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
    dvGm.SetGlobalBuffer((__gm__ float *)dv);

    if constexpr (INIT_OUTPUT) {
        // tiling_data
        qPreBlockFactor = TilingData->preTilingData.qPreBlockFactor;
        qPreBlockTotal = TilingData->preTilingData.qPreBlockTotal;
        qPreBlockTail = TilingData->preTilingData.qPreBlockTail;
        qSizeAlign = TilingData->postTilingData.qSizeAlign;
        kvPreBlockFactor = TilingData->preTilingData.kvPreBlockFactor;
        kvPreBlockTotal = TilingData->preTilingData.kvPreBlockTotal;
        kvPreBlockTail = TilingData->preTilingData.kvPreBlockTail;
        kvSizeAlign = TilingData->postTilingData.kvSizeAlign;

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  TilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  TilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  TilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));

        initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
        dqOffset = ((int64_t)cBlockIdx) * qPreBlockFactor;
        initdkSize = cBlockIdx == kvPreBlockTotal - 1 ? kvPreBlockTail : kvPreBlockFactor;
        dkvOffset = ((int64_t)cBlockIdx) * kvPreBlockFactor;
    }

    // dropMask params init
    isDropBoolMode = TilingData->preTilingData.dropoutIsDivisibleBy8 == 0 ? true : false;

    if (isDropBoolMode) {
        maskWorkSpaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace + TilingData->preTilingData.dropBeginAddr);

        pipe->InitBuffer(helpQue, 1, HELP_LEN);
        pipe->InitBuffer(inputQue, 1, TilingData->preTilingData.inputBufferLen);
        pipe->InitBuffer(castQue, 1, TilingData->preTilingData.castBufferLen);
        pipe->InitBuffer(outQue, 1, TilingData->preTilingData.outputBufferLen);

        // reset params
        repParams.src0BlkStride = 1;
        repParams.src0RepStride = 0;
        repParams.src1BlkStride = 0;
        repParams.src1RepStride = 0;
        repParams.dstBlkStride = 1;
        repParams.dstRepStride = NUMBER_8;

        copyParams.blockCount = 1;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
    }
}


template <typename T1, typename T2, typename TILING_TYPE, const bool INIT_OUTPUT>
__aicore__ inline void FlashAttentionScoreGradPre<T1, T2, TILING_TYPE, INIT_OUTPUT>::Process()
{
    // process clear dq dk dv workspace
    if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.qPreBlockTotal) {
        if constexpr (INIT_OUTPUT) {
            InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
        }
    }

    if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.kvPreBlockTotal) {
        if constexpr (INIT_OUTPUT) {
            InitOutput<float>(dkWorkSpaceGm[dkvOffset], initdkSize, 0);
            if constexpr (IsSameType<T1, float>::value) {
                InitOutput<float>(dvGm[dkvOffset], initdkSize, 0);
            } else {
                InitOutput<float>(dvWorkSpaceGm[dkvOffset], initdkSize, 0);
            }
        }
    }

    if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.maskCoreNum) {
        if (!isDropBoolMode) {
            return;
        }
        maskUBLoop = TilingData->preTilingData.maskSingleCoreLoop;
        maskTailUBProcessNum = TilingData->preTilingData.maskLastLoopNum;
        if (unlikely(cBlockIdx == maskUsedCoreNum - 1)) {
            maskUBLoop = TilingData->preTilingData.maskTailCoreLoop;
            maskTailUBProcessNum = TilingData->preTilingData.maskTailCoreLastLoopNum;
        }

        // malloc tensor filled by 1.0
        auto helpTensor = helpQue.AllocTensor<half>();
        Duplicate<half>(helpTensor, padValue, HELP_LEN / sizeof(half));
        pipe_barrier(PIPE_V);

        int64_t outputAddr = cBlockIdx * TilingData->preTilingData.maskSingleCoreNum;
        int64_t inputAddr = cBlockIdx * TilingData->preTilingData.maskSingleCoreNum / BIT8;

        // process
        for (int64_t idx = 0; idx < maskUBLoop; idx++) {
            maskUBProcessNum = TilingData->preTilingData.singleUBProcessNum;
            int64_t outputOffset = idx * maskUBProcessNum;
            int64_t inputOffset = idx * maskUBProcessNum / BIT8;
            if (unlikely(idx == maskUBLoop - 1)) {
                maskUBProcessNum = maskTailUBProcessNum;
            }

            // copyIn
            auto inputTensor = inputQue.AllocTensor<uint8_t>();
            copyParams.blockLen = maskUBProcessNum / BIT8;
            DataCopyPad(inputTensor, drop_maskGm[inputAddr + inputOffset], copyParams, {false, 0, 0, 0});
            inputQue.EnQue(inputTensor);
            inputQue.DeQue<uint8_t>();

            // select
            auto castTensor = castQue.AllocTensor<half>();
            uint8_t selectRepeat = (maskUBProcessNum + B16_VECTOR_MASK - 1) / B16_VECTOR_MASK;
            Select(castTensor, inputTensor, helpTensor, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, B16_VECTOR_MASK,
                   selectRepeat, repParams);
            pipe_barrier(PIPE_V);
            inputQue.FreeTensor(inputTensor);

            // cast
            auto outputTensor = outQue.AllocTensor<uint8_t>();
            Cast(outputTensor, castTensor, RoundMode::CAST_ROUND, maskUBProcessNum);
            castQue.FreeTensor(castTensor);

            // copyOut
            outQue.EnQue(outputTensor);
            outQue.DeQue<uint8_t>();
            copyParams.blockLen = maskUBProcessNum;
            DataCopyPad(maskWorkSpaceGm[outputAddr + outputOffset], outputTensor, copyParams);
            outQue.FreeTensor(outputTensor);
        }
        helpQue.FreeTensor(helpTensor);
    }
}

template <typename T1, typename T2, typename TILING_TYPE, const bool INIT_OUTPUT>
__aicore__ inline void FlashAttentionScoreGradPre<T1, T2, TILING_TYPE, INIT_OUTPUT>::SyncALLCores()
{
    SyncAll();
}

#endif // FLASH_ATTENTION_SCORE_GRAD_PRE_KERNEL_H_
