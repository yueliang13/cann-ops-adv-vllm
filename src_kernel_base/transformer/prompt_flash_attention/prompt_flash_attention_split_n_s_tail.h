/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

/*!
 * \file prompt_flash_attention_split_n_s_tail.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_SCORE_SPLIT_N_S_TAIL_H
#define PROMPT_FLASH_ATTENTION_SCORE_SPLIT_N_S_TAIL_H

#include "prompt_flash_attention_base.h"

using namespace matmul;
template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M = Mode::HighPerformance>
class PromptFlashAttentionSplitNSTail : public PromptFlashAttentionBase<T, U, FORMAT, O, M>
{
public:
    // define datatype
    using mmInputType = typename PromptFlashAttentionTypeTraits<T, M>::mmInputType;
    using mmBiasType = typename PromptFlashAttentionTypeTraits<T, M>::mmBiasType;
    using mmOutputType = typename PromptFlashAttentionTypeTraits<T, M>::mmOutputType;
    using softmaxType = typename PromptFlashAttentionTypeTraits<T, M>::softmaxType;
    using pseShiftType = typename PromptFlashAttentionTypeTraits<T, M>::pseShiftType;
    using pseShiftCastType = typename PromptFlashAttentionTypeTraits<T, M>::pseShiftCastType;
    __aicore__ inline PromptFlashAttentionSplitNSTail() {};
    __aicore__ inline void Process();

protected:
    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t sinnerSize, uint32_t sInnerIdx);

    __aicore__ inline void PseShiftCopyIn(uint64_t offset, uint32_t sinnerSize, uint32_t sInnerLoopIdx);

    __aicore__ inline void PseShiftProcess(int64_t sInnerLoopIdx, uint32_t computeSize, LocalTensor<mmOutputType>& mmResUb);

    __aicore__ inline void Bmm1ResDoVecBmm2Compute(LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
                                            LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb,
                                            bool isLast, bool isSecond, event_t eventID, int64_t sInnerLoopIdx);

    __aicore__ inline void Bmm1ResDoVecBmm2ComputeFirst(LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
                                            LocalTensor<float>& softmaxSumUb, bool isLast, event_t eventID, int64_t sInnerLoopIdx);

    __aicore__ inline void ComputeEachCoreSInnerLoop(uint32_t startIndex, uint32_t endIndex);

    __aicore__ inline void SInnerLoopFunc(int32_t startIndex, int32_t endIndex);

    __aicore__ inline void ComputeEachCore(uint32_t coreIdx);
};

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSTail<T, U, FORMAT, O, M>::Process()
{
    ComputeEachCore(this->tmp_block_idx);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSTail<T, U, FORMAT, O, M>::PseShiftCopyIn(uint64_t offset, uint32_t sinnerSize,
                                                                           uint32_t sInnerLoopIdx) {
    if (!(this->usePseShift)) {
        return;
    }
    LocalTensor<pseShiftType> pseShiftUb = this->attenMaskQueue.template AllocTensor<pseShiftType>();
    pseShiftUb.SetSize(this->singleProcessSOuterSize * sinnerSize);

    DataCopyExtParams intriParams;
    intriParams.blockCount = this->singleProcessSOuterSize;
    intriParams.blockLen = sinnerSize * sizeof(pseShiftType);
    intriParams.srcStride = (this->pseShiftStride - sinnerSize) * sizeof(pseShiftType);

    if (sInnerLoopIdx == this->maxInnerLoopTimes - 1) {
        intriParams.blockLen = this->unalignSInner * sizeof(pseShiftType);
        intriParams.srcStride = (this->pseShiftStride - this->unalignSInner) * sizeof(pseShiftType);
    }

    intriParams.dstStride = 0;

    DataCopyPadExtParams<pseShiftType> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.paddingValue = 1;
    if (sInnerLoopIdx == this->maxInnerLoopTimes - 1) {
        padParams.rightPadding = this->pseShiftPadSize;
    } else {
        padParams.rightPadding = 0;
    }
    DataCopyPad(pseShiftUb, this->pseShiftGm[offset], intriParams, padParams);
    this->attenMaskQueue.EnQue(pseShiftUb);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSTail<T, U, FORMAT, O, M>::PseShiftProcess(int64_t sInnerLoopIdx,
    uint32_t computeSize, LocalTensor<mmOutputType>& mmResUb) {
    if (this->usePseShift) {
        this->PseShiftCopyIn(this->pseShiftOffset, this->pseShiftCopyInCol, sInnerLoopIdx);
        LocalTensor<pseShiftType> pseShiftUb = this->attenMaskQueue.template DeQue<pseShiftType>();
        if constexpr (AscendC::IsSameType<pseShiftCastType, float>::value) {
            LocalTensor<float> pseShiftCastTensor = this->pseShiftCastUb.template Get<float>(this->pseShiftUbSize);
            Cast(pseShiftCastTensor, pseShiftUb, RoundMode::CAST_NONE, computeSize);
            pipe_barrier(PIPE_V);
            Add(mmResUb, mmResUb, pseShiftCastTensor, computeSize);
        } else { //      api add pseShiftUb to mmResUb
            Add(mmResUb, mmResUb, pseShiftUb, computeSize);
        }

        pipe_barrier(PIPE_V);
        this->attenMaskQueue.FreeTensor(pseShiftUb);
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSTail<T, U, FORMAT, O, M>::AttenMaskCopyIn(uint64_t offset, uint32_t sinnerSize,
                                                                                         uint32_t sInnerLoopIdx)
{
    if (this->useMask == false) {
        return;
    }
    LocalTensor<U> attenMaskUb = this->attenMaskQueue.template AllocTensor<U>();
    attenMaskUb.SetSize(this->singleProcessSOuterSize * sinnerSize);

    DataCopyExtParams intriParams;
    intriParams.blockCount = this->singleProcessSOuterSize; // This should be non aligned.

    intriParams.blockLen = sinnerSize * sizeof(U);

    intriParams.srcStride = (this->attentionMaskStride - sinnerSize) *sizeof(U);
    if (sInnerLoopIdx == this->maxInnerLoopTimes - 1) {
        intriParams.blockLen = this->unalignSInner * sizeof(U);
        intriParams.srcStride = (this->attentionMaskStride -
                                this->unalignSInner) * sizeof(U);
    }

    intriParams.dstStride = 0;
    DataCopyPadExtParams<U> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.paddingValue = 1;
    if (sInnerLoopIdx == this->maxInnerLoopTimes - 1) {
        padParams.rightPadding = this->padSize;
    } else {
        padParams.rightPadding = 0;
    }
    DataCopyPad(attenMaskUb, this->attenMaskGm[offset], intriParams, padParams);
    this->attenMaskQueue.EnQue(attenMaskUb);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSTail<T, U, FORMAT, O, M>::Bmm1ResDoVecBmm2ComputeFirst(LocalTensor<mmOutputType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                            bool isLast, event_t eventID, int64_t sInnerLoopIdx) {
    WaitFlag<HardEvent::MTE3_MTE2>(eventID);
    this->mm.template GetTensorC<false>(mmResUb, false, false);

    uint32_t computeSize = this->singleProcessSInnerSizeNow * this->singleProcessSOuterSize;

    Muls(mmResUb, mmResUb, static_cast<mmOutputType>(this->tilingData->promptAttentionBaseParams.scaleValue), computeSize);
    pipe_barrier(PIPE_V);

    this->PseShiftProcess(sInnerLoopIdx, computeSize, mmResUb);

    this->AttenMaskCopyIn(this->attenMaskOffset, this->maskCopyInCol, sInnerLoopIdx);

    if(this->attentionMaskType == 4){ // 4:   band mode of sparseMode
        this->ElewiseCompute(mmResUb, computeSize, 0);

        this->AttenMaskCopyIn(this->attenMaskOffsetPre, this->maskCopyInCol, sInnerLoopIdx);
        this->ElewiseCompute(mmResUb, computeSize, 1);
    } else {
        this->ElewiseCompute(mmResUb, computeSize, 0);
    }

    pipe_barrier(PIPE_V); //  Vector pipeline    synchronization

    uint32_t alignSInner = (this->unalignSInner + this->typeByteNum -1) / this->typeByteNum * this->typeByteNum;
    SoftMaxShapeInfo shapeInfo = {this->singleProcessSOuterSize, alignSInner,
                                  this->singleProcessSOuterSize, this->unalignSInner};
    if (this->IsSoftmaxBasic()
        && this->singleProcessSInnerBmmTail == this->singleProcessSInnerSize
        && this->singleProcessSOuterSize % 8 == 0) {
        this->SoftmaxBasicComputeFirst(mmResUb, softmaxMaxUb, softmaxSumUb, shapeInfo);
    } else {
        this->SoftmaxComputeFirst(mmResUb, softmaxMaxUb, softmaxSumUb, shapeInfo);
    }

    this->Bmm2Compute(this->valueOffset, mmResUb);
    this->bmm2.template Iterate<false>();

    if (isLast) {
        LocalTensor<mmOutputType> bmm2ResUb = this->tempBmm2Queue.template AllocTensor<mmOutputType>();
        if (this->tilingData->promptAttentionBaseParams.headSize ==
            this->tilingData->promptAttentionBaseParams.alignedHeadSize) {
                this->bmm2.template GetTensorC<false>(bmm2ResUb, false, false);
        } else {
            // Adapt to D non aligned scenarios. When GetTensorC function copy something to bmm2ResUb, enable template parameters dopad, and transfer the original width and height. 
            this->bmm2.template GetTensorC<false, true>(bmm2ResUb, false, false,
                this->singleProcessSOuterSize, this->tilingData->promptAttentionBaseParams.headSize);
        }

        this->bmm2.End();
        this->DataCopyTransposeOut(bmm2ResUb);
        this->tempBmm2Queue.FreeTensor(bmm2ResUb);
    }

    SetFlag<HardEvent::MTE3_MTE2>(eventID);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSTail<T, U, FORMAT, O, M>::Bmm1ResDoVecBmm2Compute(LocalTensor<mmOutputType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                            LocalTensor<softmaxType>& softmaxExpUb, bool isLast,
                                            bool isSecond, event_t eventID, int64_t sInnerLoopIdx) {
    WaitFlag<HardEvent::MTE3_MTE2>(eventID);
    this->mm.template GetTensorC<false>(mmResUb, false, false);

    uint32_t computeSize = this->singleProcessSInnerSizeNow * this->singleProcessSOuterSize;

    Muls(mmResUb, mmResUb, static_cast<mmOutputType>(this->tilingData->promptAttentionBaseParams.scaleValue), computeSize);
    pipe_barrier(PIPE_V); // Vector  pipeline synchronization

    this->PseShiftProcess(sInnerLoopIdx, computeSize, mmResUb);

    this->AttenMaskCopyIn(this->attenMaskOffset, this->maskCopyInCol, sInnerLoopIdx);

    if(this->attentionMaskType == 4){ // 4:    band mode of sparseMode
        this->ElewiseCompute(mmResUb, computeSize, 0);

        this->AttenMaskCopyIn(this->attenMaskOffsetPre, this->maskCopyInCol, sInnerLoopIdx);
        this->ElewiseCompute(mmResUb, computeSize, 1);
    } else {
        this->ElewiseCompute(mmResUb, computeSize, 0);
    }

    pipe_barrier(PIPE_V); //  Vector pipeline     synchronization

    SoftMaxShapeInfo shapeInfo = {this->singleProcessSOuterSize, this->singleProcessSInnerSize,
                                  this->singleProcessSOuterSize, this->singleProcessSInnerSize};
    if (this->IsSoftmaxBasic()
        && this->singleProcessSInnerBmmTail == this->singleProcessSInnerSize
        && this->singleProcessSOuterSize % 8 == 0) {
        this->SoftmaxBasicCompute(mmResUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb, shapeInfo);
    } else {
        this->SoftmaxCompute(mmResUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb, shapeInfo);
    }

    LocalTensor<mmOutputType> bmm2ResUb;
    if (isSecond) {
        bmm2ResUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);
    } else {
        bmm2ResUb = this->tempBmm2Queue.template AllocTensor<mmOutputType>();
    }

    if (this->tilingData->promptAttentionBaseParams.headSize ==
        this->tilingData->promptAttentionBaseParams.alignedHeadSize) {
            this->bmm2.template GetTensorC<false>(bmm2ResUb, false, false);
    } else {
        // Adapt to D non aligned scenarios. When GetTensorC function copy something to bmm2ResUb, enable template parameters dopad, and transfer the original width and height.
        this->bmm2.template GetTensorC<false, true>(bmm2ResUb, false, false,
            this->singleProcessSOuterSize, this->tilingData->promptAttentionBaseParams.headSize);
    }
    this->bmm2.End();
    this->Bmm2Compute(this->valueOffset, mmResUb);
    this->bmm2.template Iterate<false>();

    if (!isSecond) {
        this->Bmm2UpdateAdd(bmm2ResUb);
        this->tempBmm2Queue.FreeTensor(bmm2ResUb);
    }

    this->UpdateVmul(softmaxExpUb);

    if (isLast) {
        bmm2ResUb = this->tempBmm2Queue.template AllocTensor<mmOutputType>();
        if (this->tilingData->promptAttentionBaseParams.headSize ==
            this->tilingData->promptAttentionBaseParams.alignedHeadSize) {
                this->bmm2.template GetTensorC<false>(bmm2ResUb, false, false);
        } else {
            // Adapt to D non aligned scenarios. When GetTensorC function copy something to bmm2ResUb, enable template parameters dopad, and transfer the original width and height.
            this->bmm2.template GetTensorC<false, true>(bmm2ResUb, false, false,
                this->singleProcessSOuterSize, this->tilingData->promptAttentionBaseParams.headSize);
        }
        this->bmm2.End();
        this->Bmm2UpdateAdd(bmm2ResUb);

        this->tempBmm2Queue.FreeTensor(bmm2ResUb);
        LocalTensor<mmOutputType> FinalResUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);

        this->DataCopyTransposeOut(FinalResUb);
    }

    SetFlag<HardEvent::MTE3_MTE2>(eventID);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSTail<T, U, FORMAT, O, M>::SInnerLoopFunc(int32_t startIndex, int32_t endIndex) {
    if (startIndex < 0) {
        startIndex = 0;
    }
    if (endIndex > this->maxInnerLoopTimes) {
        endIndex = this->maxInnerLoopTimes;
    }

    int sInnerLoopTimes = endIndex - startIndex;
    if (sInnerLoopTimes <= 0) {
        return; // If sInner loop times is not greater than 0, return early
    }
    this->tensorAOffset = this->tensorACoreOffset;
    this->tensorBOffset = this->tensorBCoreOffset + startIndex * this->singleProcessSInnerSize * this->MultiHeadQ;

    this->mm.SetTensorA(this->queryGm[this->tensorAOffset]);
    this->mm.SetTensorB(this->keyGm[this->tensorBOffset], true);
    // quant:
    if constexpr (IsSameType<mmInputType, int8_t>::value) {
        this->mm.SetQuantScalar(this->dequantScale1);
    }

    int curS;
    if (endIndex == this->maxInnerLoopTimes) {
        curS = this->singleProcessSInnerSize * (sInnerLoopTimes - 1) + this->singleProcessSInnerSizeTail;
    } else {
        curS = this->singleProcessSInnerSize * sInnerLoopTimes;
    }
    this->mm.SetTail(this->singleProcessSOuterSize, curS);
    this->mm.template Iterate<false>();

    ComputeEachCoreSInnerLoop(startIndex, endIndex);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSTail<T, U, FORMAT, O, M>::ComputeEachCoreSInnerLoop(uint32_t startIndex,
                                                                                     uint32_t endIndex) {
    bool isSecond = true;
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID);
    for (int64_t sInnerLoopIdx = startIndex; sInnerLoopIdx < endIndex; sInnerLoopIdx++) {
        LocalTensor<float> softmaxMaxUb = this->softmaxOutQueue.template AllocTensor<float>();
        LocalTensor<float> softmaxSumUb = softmaxMaxUb[this->softmaxMaxSize];
        this->ComputeOffset(sInnerLoopIdx);
        LocalTensor<mmOutputType> mmResUb = this->Bmm1Queue.template AllocTensor<mmOutputType>();

        if (sInnerLoopIdx == this->maxInnerLoopTimes - 1) {
            mmResUb.SetSize(this->singleProcessSOuterSize * this->singleProcessSInnerSizeTail);
            this->singleProcessSInnerSizeNow = this->singleProcessSInnerSizeTail;
            this->singleProcessSInnerBmmTail = this->unalignSInner;
            this->maskCopyInCol = this->maskInnerTailAlign;
            this->pseShiftCopyInCol = this->pseShiftInnerTailAlign;
        }
        else {
            this->singleProcessSInnerSizeNow = this->singleProcessSInnerSize;
            this->singleProcessSInnerBmmTail = this->singleProcessSInnerSize;
            this->maskCopyInCol = this->singleProcessSInnerSize;
            this->pseShiftCopyInCol = this->singleProcessSInnerSize;
        }
        bool isLast = sInnerLoopIdx == endIndex-1;

        if (sInnerLoopIdx == startIndex) {
            Bmm1ResDoVecBmm2ComputeFirst(mmResUb, softmaxMaxUb, softmaxSumUb, isLast, eventID, sInnerLoopIdx);
        }else {
            Bmm1ResDoVecBmm2Compute(mmResUb, softmaxMaxUb, softmaxSumUb, this->softmaxExpUb, isLast, isSecond, eventID, sInnerLoopIdx);
            isSecond = false;
        }
        this->Bmm1Queue.FreeTensor(mmResUb);
        this->softmaxOutQueue.FreeTensor(softmaxMaxUb);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(eventID);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSTail<T, U, FORMAT, O, M>::ComputeEachCore(uint32_t coreIdx) {
    this->spmTmpSize = this->tilingData->promptAttentionTensorSizeRect.spmTmpSize;
    this->mmResUbSize = this->tilingData->promptAttentionTensorSizeRect.mmResUbSize;
    this->bmm2ResUbSize = this->tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize;
    int reuseWorkspaceRatio = this->tilingData->promptAttentionSingleCoreParams.multiSmaxsInnerLoopTimes;
    this->mm.SetWorkspace((__gm__ uint8_t*)this->workspaceGm[GetBlockNum() * GetTaskRation() * this->spmTmpSize +
        coreIdx * this->mmResUbSize * reuseWorkspaceRatio].GetPhyAddr(), this->mmResUbSize * reuseWorkspaceRatio);

    uint32_t buff_offset = GetBlockNum() * GetTaskRation() * (this->spmTmpSize +
                           this->mmResUbSize * reuseWorkspaceRatio);
    this->bmm2.SetWorkspace((__gm__ uint8_t*)this->workspaceGm[buff_offset + coreIdx * this->bmm2ResUbSize].GetPhyAddr(), this->bmm2ResUbSize);
    uint32_t actualSeqLengthsIdx = 0; // idx of actual seq length

    for (int sIdx = 0; sIdx < this->tilingData->promptAttentionBaseParams.dimNumOfseq; sIdx++) {
		int curSeqCoreNum = this->tilingData->promptAttentionSeqParams.actualCoreNums[sIdx];
        int s1 = this->tilingData->promptAttentionSeqParams.actualS1[sIdx];
        uint32_t coreHeadNum = this->tilingData->promptAttentionSeqParams.singleCoreHeadNumSize[sIdx];
        int coreHeadNumTail = this->tilingData->promptAttentionSeqParams.CoreHeadNumTail[sIdx];
        actualSeqLengthsIdx = this->isActualLenDimsNull ? this->tilingData->promptAttentionBaseParams.seqSize : this->actualSeqLengthsGm.GetValue(sIdx);
        actualSeqLengthsIdx = (this->attentionMaskType == 0 && (int64_t)actualSeqLengthsIdx >
                               (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize +
                               (int64_t)this->tilingData->promptAttentionBaseParams.preTokens) ?
                            this->tilingData->promptAttentionBaseParams.seqInnerSize + this->tilingData->promptAttentionBaseParams.preTokens :
                            actualSeqLengthsIdx;
        int sOuterBlockNum = (actualSeqLengthsIdx + this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize - 1) /
                                this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
        this->multiSeqOffset = this->actualSeqOffsets[sIdx];
        if (g_coreType == AIV && coreIdx >= curSeqCoreNum) {
            continue;
        }

        int curSIdx = coreIdx % s1;
        int nGropOffeset;
        if (coreIdx / s1 < coreHeadNumTail) {
            coreHeadNum += 1;
            nGropOffeset = coreIdx / s1 * coreHeadNum;
        } else {
            nGropOffeset = coreIdx / s1 * coreHeadNum + coreHeadNumTail;
        }

        this->GetSingleCoreParam(sIdx);

        int formerNum = sOuterBlockNum % s1;

        for (uint32_t loopNIdx = 0; loopNIdx < coreHeadNum; loopNIdx++) {
            this->batchNOffset = nGropOffeset + loopNIdx;
            // When the number of cycles N is even, reverse curSidx.
            if (loopNIdx % 2 != 0) {
                curSIdx = s1 - curSIdx - 1;
            }
            int loopStart;
            if (curSIdx < formerNum) {
                this->loopSNum = sOuterBlockNum / s1 + 1;
                loopStart = this->loopSNum * curSIdx;
            } else {
                this->loopSNum = sOuterBlockNum / s1;
                loopStart = this->loopSNum * curSIdx + formerNum;
            }

            for (uint32_t sOuterLoopIdx = loopStart; sOuterLoopIdx < loopStart + this->loopSNum; sOuterLoopIdx++) {
                if (sOuterLoopIdx == sOuterBlockNum - 1) {
                    this->singleProcessSOuterSize = this->singleProcessSOuterSizeTail;
                } else {
                    this->singleProcessSOuterSize = this->singleProcessSOuterSizeWhole;
                }
                this->sOuterOffset = sOuterLoopIdx * this->singleProcessSOuterSizeWhole;
                int32_t preTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.preTokens);
                int32_t nextTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.nextTokens);
                this->GetSparseParam(&preTokens, &nextTokens);
                this->ComputeTokenOffset();
                if (nextTokens < 0 && this->sOuterOffset < ((nextTokens * (-1)) /
                    this->singleProcessSOuterSizeWhole * this->singleProcessSOuterSizeWhole)) {
                    continue;
                }
                int32_t start_idx = (this->sOuterOffset - preTokens) / (int32_t)(this->singleProcessSInnerSize);
                int32_t end_idx = (this->sOuterOffset + nextTokens +
                                  this->singleProcessSOuterSize + (int32_t)(this->singleProcessSInnerSize) - 1) /
                                  (int32_t)(this->singleProcessSInnerSize);
                this->LoopSOuterOffsetInit(this->actualSeqOffsets[sIdx], sIdx);
                SInnerLoopFunc(start_idx, end_idx);
            }
        }
    }
}

#endif  // PROMPT_FLASH_ATTENTION_SCORE_SPLIT_N_S_TAIL_H
