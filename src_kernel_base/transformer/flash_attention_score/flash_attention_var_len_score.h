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
 * \file flash_attention_var_len_score.h
 * \brief
 */

#ifndef FLASH_ATTENTION_VAR_LEN_SCORE_H
#define FLASH_ATTENTION_VAR_LEN_SCORE_H

#include "flash_attention_score_s1s2_bn2gs1.h"

// INPUT_T - means data type for input
// T       - means data type when calc
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T = INPUT_T, bool isBasicBlock = false, CubeFormat bmm1Format = CubeFormat::NZ>
class FlashAttentionVarLenScore
    : public FlashAttentionScoreS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                           bmm1Format> {
public:
    __aicore__ inline FlashAttentionVarLenScore(){};

    __aicore__ inline void
    UnpackInit(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
               __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
               __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *softmaxMax,
               __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut,
               __gm__ uint8_t *workspace, const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);

    __aicore__ inline void Process();

protected:
    __aicore__ inline void ComputeConstexpr();

    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);

    __aicore__ inline void ComputeBmm1Tail(SplitExtraInfo &extraInfo);

    __aicore__ inline void SetExtraInfo(SplitExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx);

    __aicore__ inline void CalS1OuterSize(int64_t offset);

    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);

    __aicore__ inline void GetS2LoopRange();
    // Unpack 用参数
    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqKvlenAddr;
    GM_ADDR prefixNAddr;
    constexpr static uint16_t FA_VARLEN_MAX_B = 2048;
    uint64_t s1OuterSizeAcc;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;
    uint64_t attenB1SSOffset;
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format>::UnpackInit(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->InitInput(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,
                    softmaxOut, attentionOut, workspace, tiling, tPipe); // gm设置
    this->ComputeConstexpr();
    this->InitBuffer();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    DropOutBitModeInit(apiTmpBuffer);
    if (this->blockIdx < this->tilingData->multiCoreParams.coreNum) {
        LocalTensor<half> pseHelpBuffer = this->stage1PingBuf.template Get<half>();
        PseInnerAlibiCreate<hasPse>(this->pseAlibiGm, pseHelpBuffer, this->pseInfo);
    }
    // 初始化unpack
    actualSeqQlenAddr = actualSeqLengths;
    actualSeqKvlenAddr = actualSeqLengthsKv;
    prefixNAddr = prefix;
    this->s2SizeSum = ((__gm__ uint64_t *)actualSeqLengthsKv)[this->tilingData->inputParams.bSize - 1];
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format>::ComputeConstexpr()
{
    this->gD = this->tilingData->inputParams.gSize * this->dSize;
    this->n2D = this->tilingData->inputParams.n2Size * this->dSize;
    this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->n2GD = this->tilingData->inputParams.n2Size * this->gD;

    // 计算切分轴的乘积
    this->s2BaseN2D = this->s2BaseSize * this->n2D;
    this->s2BaseNratioSize = this->s2BaseSize * this->tilingData->coreParams.nRatio;

    // layout(bs)ND
    this->s1BaseN2GD = this->s1BaseSize * this->n2GD;
    this->s2BaseNratioN2D = this->s2BaseN2D * this->tilingData->coreParams.nRatio;

    // layout(bs)ND
    this->mm1Ka = this->n2GD;
    this->mm1Kb = this->n2D;
    this->mm2Kb = this->n2D;
    if constexpr (hasPse == true) {
        this->pseInfo.gSize = this->tilingData->inputParams.gSize;
        this->pseInfo.pseShapeType = this->tilingData->inputParams.pseShapeType;
        this->pseInfo.n2G = this->n2G;
        this->pseInfo.pseBSize = this->tilingData->inputParams.pseBSize;
        this->pseInfo.s1BaseSize = this->s1BaseSize;
        this->pseInfo.pseS1Size = this->tilingData->inputParams.pseS1Size;
        this->pseInfo.pseS2Size = this->tilingData->inputParams.pseS2Size;
        this->pseInfo.s2BaseNratioSize = this->s2BaseNratioSize;
        this->pseInfo.pseEncodeType = (uint32_t)this->tilingData->inputParams.pseEncodeType;
        this->pseInfo.pseType = this->tilingData->inputParams.pseType;
        this->pseInfo.pseAlibiBaseS1 = this->tilingData->coreParams.pseAlibiBaseS1;
        this->pseInfo.pseAlibiBaseS2 = this->tilingData->coreParams.pseAlibiBaseS2;
        this->pseInfo.qStartIdx = this->tilingData->inputParams.qStartIdx;
        this->pseInfo.kvStartIdx = this->tilingData->inputParams.kvStartIdx;
    }

    if constexpr (hasDrop == true) {
        this->dropMaskInfo.gSize = this->tilingData->inputParams.gSize;
        this->dropMaskInfo.n2G = this->n2G;
        this->dropMaskInfo.s1BaseSize = this->s1BaseSize;
        this->dropMaskInfo.s2BaseNratioSize = this->s2BaseNratioSize;
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format>
__aicore__ inline void
FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                          bmm1Format>::GetSeqQlenKvlenByBoidx(int64_t boIdx,
                                                              int64_t &actualSeqQlen,
                                                              int64_t &actualSeqKvlen)
{
    if (unlikely(boIdx == 0)) {
        actualSeqQlen = ((__gm__ int64_t *)actualSeqQlenAddr)[0];
        actualSeqKvlen = ((__gm__ int64_t *)actualSeqKvlenAddr)[0];
    } else {
        actualSeqQlen = ((__gm__ int64_t *)actualSeqQlenAddr)[boIdx] - ((__gm__ int64_t *)actualSeqQlenAddr)[boIdx - 1];
        actualSeqKvlen =
            ((__gm__ int64_t *)actualSeqKvlenAddr)[boIdx] - ((__gm__ int64_t *)actualSeqKvlenAddr)[boIdx - 1];
    }
    return;
}

// 初始化s1方向上的累加值
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format>::CalS1OuterSize(int64_t offset)
{
    int64_t actualS1Outersize = 0;
    // 用于取actualS1Len下标
    this->boIdx = 0;
    this->s1OuterSizeAcc = 0;
    this->attenB1SSOffset = 0;
    this->s1SizeAcc = 0;
    this->s2SizeAcc = 0;

    int64_t actualS1Len;
    int64_t actualS2Len;
    for (int64_t i = 0; i < this->tilingData->inputParams.bSize; i++) {
        GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
        actualS1Outersize += (CeilDiv(actualS1Len, this->s1BaseSize) * this->n2G);
        if (offset >= actualS1Outersize) {
            this->s1OuterSizeAcc = actualS1Outersize;
            this->s1SizeAcc += actualS1Len;
            this->s2SizeAcc += actualS2Len;
            this->attenB1SSOffset += actualS1Len * actualS2Len;
            this->boIdx++;
            if (this->boIdx >= FA_VARLEN_MAX_B) {
                break;
            }
        } else {
            break;
        }
    }
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
    int64_t actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(actualS1Len, this->s1BaseSize) * this->n2G);
    while (multiCoreInnerIdx >= actualS1Outersize) {
        this->s1OuterSizeAcc = actualS1Outersize;
        this->s1SizeAcc += actualS1Len;
        this->s2SizeAcc += actualS2Len;
        this->attenB1SSOffset += actualS1Len * actualS2Len;
        this->boIdx++;
        if (this->boIdx >= FA_VARLEN_MAX_B) {
            break;
        }
        GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
        actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(actualS1Len, this->s1BaseSize) * this->n2G);
    }
    // 计算轴的idx
    int64_t tmpS1Outersize = CeilDiv(actualS1Len, this->s1BaseSize);
    actualS1Outersize = multiCoreInnerIdx - this->s1OuterSizeAcc;
    this->n2oIdx = actualS1Outersize / tmpS1Outersize / this->tilingData->inputParams.gSize;
    this->goIdx = actualS1Outersize / tmpS1Outersize % this->tilingData->inputParams.gSize;
    this->s1oIdx = actualS1Outersize % tmpS1Outersize;
    GetSeqQlenKvlenByBoidx(this->boIdx, this->s1Size, this->s2Size);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format>::GetS2LoopRange()
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
    if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) { // 下三角
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize, actualS2Len);
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL)) {
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize + actualS2Len - actualS1Len, actualS2Len);
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
        this->s2StartIdx = Max(
            this->s1oIdx * this->tilingData->coreParams.s1BaseSize - this->tilingData->coreParams.s1SparseValidSize, 0);
        this->s2EndIdx =
            Min((this->s1oIdx + 1) * this->s1BaseSize + this->tilingData->coreParams.s2SparseValidSize, actualS2Len);
        // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_COMPRESS)) {
        this->s2StartIdx = Max(this->s1oIdx * this->tilingData->coreParams.s1BaseSize - actualS1Len +
                                   Max(actualS2Len - this->tilingData->inputParams.preTokens, 0),
                               0);
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize + actualS2Len -
                                 Max(actualS1Len - this->tilingData->inputParams.nextTokens, 0),
                             actualS2Len);
        // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else if (this->tilingData->inputParams.sparseType ==
               static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL_BAND)) {
        if (this->boIdx == this->tilingData->inputParams.bandIndex) {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize + actualS2Len +
                                     this->tilingData->inputParams.nextTokens - actualS1Len,
                                 actualS2Len);
        } else {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize + actualS2Len - actualS1Len, actualS2Len);
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL)) {
        if (this->boIdx == this->tilingData->inputParams.bandIndex) {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize + actualS2Len -
                                     Max(actualS1Len - this->tilingData->inputParams.nextTokens, 0),
                                 actualS2Len);
        } else {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize, actualS2Len);
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
        this->s2StartIdx = 0;
        this->s2EndIdx =
            Max((this->s1oIdx + 1) * this->s1BaseSize - actualS1Len + actualS2Len,
                ((__gm__ int64_t *)prefixNAddr)[this->boIdx]);
        this->s2EndIdx = Min(this->s2EndIdx, this->s2Size);
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else { // 其它场景全计算
        this->s2StartIdx = 0;
        this->s2EndIdx = actualS2Len;
    }
    this->s2StartIdx = this->s2StartIdx / 8 * 8;
    this->s2EndIdx = Min(CeilDiv(this->s2EndIdx, 8) * 8, actualS2Len);
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }
    // 计算sparse场景下s1的循环范围
    this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);
    // 初始化AxisIdx
    this->CalS1OuterSize(multiCoreInnerOffset);

    SplitExtraInfo extraInfo[3];
    int64_t taskId = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
         ++multiCoreInnerIdx) {
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        this->ComputeAxisIdx(multiCoreInnerIdx);
        // s2轴循环计数, 支持sparse和非sparse场景
        this->GetS2LoopRange();
        int64_t s2LoopLimit = CeilDiv(this->s2EndIdx - this->s2StartIdx, this->s2BaseNratioSize) - 1;
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            if (taskId > 0) {
                // 对应extraInfo[(i+2)%3]
                this->WaitBmm1Result(extraInfo[(taskId + 2) % 3]);
            }
            this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);

            if (extraInfo[taskId % 3].needNz2Nd == 1) {
                this->IterateBmm1(extraInfo[taskId % 3], this->bmm1Nz);
            } else {
                this->IterateBmm1(extraInfo[taskId % 3], this->bmm1);
            }

            if (taskId > 0) {
                this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }

            if (taskId > 1) {
                // 对应extraInfo[(i+1)%3]
                this->WaitBmm2Result();
            }

            if (taskId > 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                this->IterateBmm2(extraInfo[(taskId + 2) % 3]);
            }

            if (taskId > 1) {
                this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
            }
            taskId++;
        }
    }
    if (taskId >= 1) {
        // 对应extraInfo[(i+2)%3]
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        this->WaitBmm1Result(extraInfo[(taskId + 2) % 3]);
        this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        if (taskId > 1) {
            // 对应extraInfo[(i+1)%3]
            this->WaitBmm2Result();
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        this->IterateBmm2(extraInfo[(taskId + 2) % 3]);
        if (taskId > 1) {
            this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
        }
    }
    taskId++;
    if (taskId >= 2) {
        // 对应extraInfo[(i+1)%3]
        this->WaitBmm2Result();
        this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
    }
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format>::SetExtraInfo(
    SplitExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    extraInfo.s2StartIdx = this->s2StartIdx;
    extraInfo.s2EndIdx = this->s2EndIdx;
    extraInfo.s2LoopCount = s2LoopCount;
    extraInfo.s1oIdx = this->s1oIdx;
    extraInfo.boIdx = this->boIdx;
    extraInfo.n2oIdx = this->n2oIdx;
    extraInfo.goIdx = this->goIdx;
    extraInfo.taskId = taskId;
    extraInfo.taskIdMod2 = taskId % 2;
    extraInfo.s2LoopLimit = s2LoopLimit;
    extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
    extraInfo.multiCoreInnerIdxMod2 = multiCoreInnerIdx % 2;
    extraInfo.s1SizeAcc = s1SizeAcc;
    extraInfo.s2SizeAcc = s2SizeAcc;
    extraInfo.attenB1SSOffset = attenB1SSOffset;
    extraInfo.attenMaskS2Size = extraInfo.s2Size;
    if (this->tilingData->inputParams.attenMaskShapeType == attenMaskS1S2) {
        extraInfo.attenMaskS2Size = this->tilingData->inputParams.s2Size;
    } else if (this->tilingData->inputParams.attenMaskShapeType == attenMaskTT) {
        extraInfo.attenMaskS2Size = this->s2SizeSum;
    }

    // band compress mode
    if (this->tilingData->inputParams.attenMaskCompressMode !=
        static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
        extraInfo.attenMaskS2Size = this->tilingData->inputParams.attenMaskS2Size;
    }

    GetSeqQlenKvlenByBoidx(extraInfo.boIdx, extraInfo.s1Size, extraInfo.s2Size);
    this->ComputeBmm1Tail(extraInfo);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format>::ComputeBmm1Tail(SplitExtraInfo &extraInfo)
{
    extraInfo.s1RealSize = this->s1BaseSize;
    if (extraInfo.s1Size < (extraInfo.s1oIdx + 1) * this->s1BaseSize) {
        extraInfo.s1RealSize = extraInfo.s1Size - extraInfo.s1oIdx * this->s1BaseSize;
    }
    extraInfo.s2RealSize = this->s2BaseNratioSize;
    extraInfo.s2AlignedSize = extraInfo.s2RealSize;
    if (extraInfo.s2StartIdx + (extraInfo.s2LoopCount + 1) * extraInfo.s2RealSize > extraInfo.s2EndIdx) {
        extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2LoopCount * extraInfo.s2RealSize - extraInfo.s2StartIdx;
        extraInfo.s2AlignedSize = Align(extraInfo.s2RealSize);
    }

    // In scenes where s2 is less than 8, when traversing s1basesize for computation, there is a memory issue.
    // Therefore, it's changed to compute once
    if (unlikely(extraInfo.s2RealSize < fp32BaseSize)) {
        extraInfo.vec1S1BaseSize = extraInfo.s1RealSize;
        extraInfo.realSplitN = 1;
    } else {
        extraInfo.vec1S1BaseSize = Min(1024 / extraInfo.s2AlignedSize * 8, extraInfo.s1RealSize); // Maximize the ub
        extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);
    }

    if (this->dSizeAlign16 > 64) {
        extraInfo.vec2S1BaseSize = 64 * 128 / this->dSizeAlign16;
    } else {
        extraInfo.vec2S1BaseSize = extraInfo.s1RealSize;
    }
    extraInfo.needNz2Nd = (extraInfo.s2RealSize % 64 == 0) ? 0 : 1;
    return;
}

#endif // FLASH_ATTENTION_VAR_LEN_SCORE_H
