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
 * \file prompt_flash_attention_split_n_s_no_tail.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_SCORE_SPLIT_N_S_NO_TAIL_H
#define PROMPT_FLASH_ATTENTION_SCORE_SPLIT_N_S_NO_TAIL_H

#include "prompt_flash_attention_base.h"

using namespace matmul;
template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M = Mode::HighPerformance>
class PromptFlashAttentionSplitNSNoTail : public PromptFlashAttentionBase<T, U, FORMAT, O, M> {
public:
    // define datatype
    using mmInputType = typename PromptFlashAttentionTypeTraits<T, M>::mmInputType;
    using mmBiasType = typename PromptFlashAttentionTypeTraits<T, M>::mmBiasType;
    using mmOutputType = typename PromptFlashAttentionTypeTraits<T, M>::mmOutputType;
    using softmaxType = typename PromptFlashAttentionTypeTraits<T, M>::softmaxType;
    using pseShiftType = typename PromptFlashAttentionTypeTraits<T, M>::pseShiftType;
    using pseShiftCastType = typename PromptFlashAttentionTypeTraits<T, M>::pseShiftCastType;
    __aicore__ inline PromptFlashAttentionSplitNSNoTail() {};
    __aicore__ inline void Process();

protected:
    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t sinnerSize, uint32_t sInnerIdx);

    __aicore__ inline void PseShiftCopyIn(uint64_t offset, uint32_t sinnerSize, uint32_t sInnerLoopIdx);

    __aicore__ inline void PseShiftProcess(int64_t sInnerLoopIdx, uint32_t computeSize, LocalTensor<mmOutputType>& mmResUb);

    __aicore__ inline void Bmm1ResDoVecBmm2Compute(LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
                                            LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb,
                                            bool isLast, bool isSecond, event_t eventID, int64_t sInnerLoopIdx);

    __aicore__ inline void Bmm1ResDoVecBmm2ComputeFirst(LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
                                                        LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb,
                                                        bool isLast, event_t eventID, int64_t sInnerLoopIdx);

    __aicore__ inline void computeSingleS(uint32_t sOuterLoopIdx, int seqIdx);

    __aicore__ inline void ComputeEachCoreSInnerLoop(uint32_t startIndex, uint32_t endIndex);

    __aicore__ inline void ComputeEachCoreNSLoop(uint32_t coreHeadNum, int sOuterBlockNum,
                                             int nGropOffeset, int seqIdx, int s1, int curSIdx);
    __aicore__ inline void ComputeEachCoreSLoop(int sOuterBlockNum, int nGropOffeset, int seqIdx, int s1, int curSIdx);

    __aicore__ inline void SInnerLoopFunc(int32_t startIndex, int32_t endIndex);

    __aicore__ inline void ComputeEachCore(uint32_t coreIdx);
};

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSNoTail<T, U, FORMAT, O, M>::Process() {
    ComputeEachCore(this->tmp_block_idx);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSNoTail<T, U, FORMAT, O, M>::PseShiftCopyIn(uint64_t offset,
                                                                                     uint32_t sinnerSize,
                                                                                     uint32_t sInnerLoopIdx) {
    if (!(this->usePseShift)) {
        return;
    }
    LocalTensor<pseShiftType> pseShiftUb = this->attenMaskQueue.template AllocTensor<pseShiftType>();
    pseShiftUb.SetSize(this->singleProcessSOuterSize * sinnerSize);

    DataCopyParams intriParams;
    intriParams.blockCount = this->singleProcessSOuterSize;
    intriParams.blockLen = sinnerSize / this->pseShiftTypeByteNum;
    intriParams.srcStride = (this->pseShiftStride - sinnerSize) / this->pseShiftTypeByteNum;
    intriParams.dstStride = 0;

    DataCopy(pseShiftUb, this->pseShiftGm[offset], intriParams);
    this->attenMaskQueue.EnQue(pseShiftUb);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSNoTail<T, U, FORMAT, O, M>::PseShiftProcess(int64_t sInnerLoopIdx,
    uint32_t computeSize, LocalTensor<mmOutputType>& mmResUb) {
    if (this->usePseShift) {
        this->PseShiftCopyIn(this->pseShiftOffset, this->pseShiftCopyInCol, sInnerLoopIdx);
        LocalTensor<pseShiftType> pseShiftUb = this->attenMaskQueue.template DeQue<pseShiftType>();
        if constexpr (AscendC::IsSameType<pseShiftCastType, float>::value) {
            LocalTensor<float> pseShiftCastTensor = this->pseShiftCastUb.template Get<float>(this->pseShiftUbSize);
            Cast(pseShiftCastTensor, pseShiftUb, RoundMode::CAST_NONE, computeSize);
            pipe_barrier(PIPE_V);
            Add(mmResUb, mmResUb, pseShiftCastTensor, computeSize);
        } else { //     api add pseShiftUb to mmResUb
            Add(mmResUb, mmResUb, pseShiftUb, computeSize);
        }

        pipe_barrier(PIPE_V);
        this->attenMaskQueue.FreeTensor(pseShiftUb);
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSNoTail<T, U, FORMAT, O, M>::AttenMaskCopyIn(uint64_t offset, uint32_t sinnerSize,
                                                                             uint32_t sInnerLoopIdx) {
    if (this->useMask == false) {
        return;
    }
    LocalTensor<U> attenMaskUb = this->attenMaskQueue.template AllocTensor<U>();
    attenMaskUb.SetSize(this->singleProcessSOuterSize * sinnerSize);
    DataCopyParams intriParams;
    intriParams.blockCount = this->singleProcessSOuterSize;
    intriParams.blockLen = sinnerSize / this->maskTypeByteNum;
    intriParams.srcStride = (this->attentionMaskStride - sinnerSize) /
                            this->maskTypeByteNum;
    intriParams.dstStride = 0;

    DataCopy(attenMaskUb, this->attenMaskGm[offset], intriParams);
    this->attenMaskQueue.EnQue(attenMaskUb);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSNoTail<T, U, FORMAT, O, M>::Bmm1ResDoVecBmm2ComputeFirst(LocalTensor<mmOutputType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                            LocalTensor<softmaxType>& softmaxExpUb, bool isLast, event_t eventID, int64_t sInnerLoopIdx) {
    WaitFlag<HardEvent::MTE3_MTE2>(eventID);
    this->mm.template GetTensorC<false>(mmResUb, false, false);

    uint32_t computeSize = this->singleProcessSInnerSizeNow * this->singleProcessSOuterSize;

    Muls(mmResUb, mmResUb, static_cast<mmOutputType>(this->tilingData->promptAttentionBaseParams.scaleValue), computeSize);
    pipe_barrier(PIPE_V);

    this->PseShiftProcess(sInnerLoopIdx, computeSize, mmResUb);

    this->AttenMaskCopyIn(this->attenMaskOffset, this->maskCopyInCol, sInnerLoopIdx);

    if(this->attentionMaskType == 4){ // 4: band mode of sparseMode
        this->ElewiseCompute(mmResUb, computeSize, 0);

        this->AttenMaskCopyIn(this->attenMaskOffsetPre, this->maskCopyInCol, sInnerLoopIdx);
        this->ElewiseCompute(mmResUb, computeSize, 1);
    } else {
        this->ElewiseCompute(mmResUb, computeSize, 0);
    }
    pipe_barrier(PIPE_V); //  Vector pipeline  synchronization

    SoftMaxShapeInfo shapeInfo = {this->singleProcessSOuterSize, this->singleProcessSInnerSize,
                                  this->singleProcessSOuterSize, this->singleProcessSInnerSize};
    if (this->IsSoftmaxFlashBasic()) {
        this->SoftmaxBasicComputeFirstNoTail(mmResUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb, shapeInfo);
    } else {
        this->SoftmaxComputeFirstTail(mmResUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb, shapeInfo);
    }

    this->Bmm2Compute(this->valueOffset, mmResUb);
    this->bmm2.template Iterate<false>();

    if (isLast) {
        LocalTensor<mmOutputType> bmm2ResUb = this->tempBmm2Queue.template AllocTensor<mmOutputType>();
        this->bmm2.template GetTensorC<false>(bmm2ResUb, false, false);
        this->bmm2.End();

        this->Bmm2UpdateDivNoTail(bmm2ResUb, softmaxSumUb, softmaxExpUb);

        this->tempBmm2Queue.EnQue(bmm2ResUb);
        bmm2ResUb = this->tempBmm2Queue.template DeQue<mmOutputType>();

        this->DataCopyTransposeOut(bmm2ResUb);
        this->tempBmm2Queue.FreeTensor(bmm2ResUb);
    }

    SetFlag<HardEvent::MTE3_MTE2>(eventID);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSNoTail<T, U, FORMAT, O, M>::Bmm1ResDoVecBmm2Compute(LocalTensor<mmOutputType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                            LocalTensor<softmaxType>& softmaxExpUb, bool isLast,
                                            bool isSecond, event_t eventID, int64_t sInnerLoopIdx) {
    WaitFlag<HardEvent::MTE3_MTE2>(eventID);
    this->mm.template GetTensorC<false>(mmResUb, false, false);

    uint32_t computeSize = this->singleProcessSInnerSizeNow * this->singleProcessSOuterSize;

    Muls(mmResUb, mmResUb, static_cast<mmOutputType>(this->tilingData->promptAttentionBaseParams.scaleValue), computeSize);
    pipe_barrier(PIPE_V); // Vector pipeline synchronization

    this->PseShiftProcess(sInnerLoopIdx, computeSize, mmResUb);

    this->AttenMaskCopyIn(this->attenMaskOffset, this->maskCopyInCol, sInnerLoopIdx);

    if(this->attentionMaskType == 4){ // 4:  band mode of sparseMode
        this->ElewiseCompute(mmResUb, computeSize, 0);

        this->AttenMaskCopyIn(this->attenMaskOffsetPre, this->maskCopyInCol, sInnerLoopIdx);
        this->ElewiseCompute(mmResUb, computeSize, 1);
    } else {
        this->ElewiseCompute(mmResUb, computeSize, 0);
    }

    pipe_barrier(PIPE_V); //  Vector pipeline   synchronization

    SoftMaxShapeInfo shapeInfo = {this->singleProcessSOuterSize, this->singleProcessSInnerSize,
                                  this->singleProcessSOuterSize, this->singleProcessSInnerSize};
    if (this->IsSoftmaxFlashBasic()) {
        this->SoftmaxBasicComputeNoTail(mmResUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb, shapeInfo);
    } else {
        this->SoftmaxComputeTail(mmResUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb, shapeInfo);
    }

    LocalTensor<mmOutputType> bmm2ResUb;
    if (isSecond) {
        bmm2ResUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);
    } else {
        bmm2ResUb = this->tempBmm2Queue.template AllocTensor<mmOutputType>();
    }

    this->bmm2.template GetTensorC<false>(bmm2ResUb, false, false);
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
        this->bmm2.template GetTensorC<false>(bmm2ResUb, false, false);
        this->bmm2.End();
        this->Bmm2UpdateAdd(bmm2ResUb);

        pipe_barrier(PIPE_V);
        LocalTensor<mmOutputType> bmm2ResPreUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);
        this->Bmm2UpdateDivNoTail(bmm2ResPreUb, softmaxSumUb, softmaxExpUb);

        this->tempBmm2Queue.FreeTensor(bmm2ResUb);
        LocalTensor<mmOutputType> FinalResUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);

        this->DataCopyTransposeOut(FinalResUb);
    }

    SetFlag<HardEvent::MTE3_MTE2>(eventID);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSNoTail<T, U, FORMAT, O, M>::SInnerLoopFunc(int32_t startIndex, int32_t endIndex) {
    if (startIndex < 0) {
        startIndex = 0;
    }
    if (endIndex > this->maxInnerLoopTimes) {
        endIndex = this->maxInnerLoopTimes;
    }

    int sInnerLoopTimes = endIndex - startIndex;
    if (sInnerLoopTimes <= 0) {
        return;
    }
    this->tensorAOffset = this->tensorACoreOffset;
    this->tensorBOffset = this->tensorBCoreOffset + startIndex * this->singleProcessSInnerSize * this->MultiHeadQ;

    this->mm.SetTensorA(this->queryGm[this->tensorAOffset]);
    this->mm.SetTensorB(this->keyGm[this->tensorBOffset], true);
    // quant:
    if constexpr (IsSameType<mmInputType, int8_t>::value) {
        this->mm.SetQuantScalar(this->dequantScale1);
    }

    int curS = this->singleProcessSInnerSize * sInnerLoopTimes;
    this->mm.SetTail(this->singleProcessSOuterSize, curS);
    this->mm.template Iterate<false>();

    ComputeEachCoreSInnerLoop(startIndex, endIndex);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSNoTail<T, U, FORMAT, O, M>::ComputeEachCoreSInnerLoop(uint32_t startIndex,
                                                                                uint32_t endIndex) {
    bool isSecond = true;
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID);
    for (int64_t sInnerLoopIdx = startIndex; sInnerLoopIdx < endIndex; sInnerLoopIdx++) {
        LocalTensor<float> softmaxMaxUb = this->softmaxOutQueue.template AllocTensor<float>();
        LocalTensor<float> softmaxSumUb = softmaxMaxUb[this->softmaxMaxSize];
        this->ComputeOffset(sInnerLoopIdx);
        LocalTensor<mmOutputType> mmResUb = this->Bmm1Queue.template AllocTensor<mmOutputType>();

        this->singleProcessSInnerSizeNow = this->singleProcessSInnerSize;
        this->singleProcessSInnerBmmTail = this->singleProcessSInnerSize;
        this->maskCopyInCol = this->singleProcessSInnerSize;
        this->pseShiftCopyInCol = this->singleProcessSInnerSize;

        bool isLast = sInnerLoopIdx == endIndex-1;

        if (sInnerLoopIdx == startIndex) {
            Bmm1ResDoVecBmm2ComputeFirst(mmResUb, softmaxMaxUb, softmaxSumUb, this->softmaxExpUb, isLast, eventID, sInnerLoopIdx);
        } else {
            Bmm1ResDoVecBmm2Compute(mmResUb, softmaxMaxUb, softmaxSumUb, this->softmaxExpUb, isLast, isSecond, eventID, sInnerLoopIdx);
            isSecond = false;
        }
        this->Bmm1Queue.FreeTensor(mmResUb);
        this->softmaxOutQueue.FreeTensor(softmaxMaxUb);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(eventID);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSNoTail<T, U, FORMAT, O, M>::computeSingleS(uint32_t sOuterLoopIdx, int seqIdx) {
    this->sOuterOffset = sOuterLoopIdx * this->singleProcessSOuterSize;
    this->ComputeTokenOffset();

    int32_t preTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.preTokens);
    int32_t nextTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.nextTokens);
    this->GetSparseParam(&preTokens, &nextTokens);

    if (nextTokens < 0 && this->sOuterOffset < ((nextTokens * (-1)) /
        this->singleProcessSOuterSizeWhole * this->singleProcessSOuterSizeWhole)) {
        return;
    }

    int32_t start_idx = (this->sOuterOffset - preTokens) / (int32_t)(this->singleProcessSInnerSize);
    int32_t end_idx = (this->sOuterOffset + this->singleProcessSOuterSize + nextTokens +
                      (int32_t)(this->singleProcessSInnerSize) - 1) /
                      (int32_t)(this->singleProcessSInnerSize);
    this->LoopSOuterOffsetInit(this->actualSeqOffsets[seqIdx], seqIdx);
    SInnerLoopFunc(start_idx, end_idx);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSNoTail<T, U, FORMAT, O, M>::ComputeEachCoreNSLoop(uint32_t coreHeadNum,
                                                                                      int sOuterBlockNum,
                                                                                      int nGropOffeset, int seqIdx,
                                                                                      int s1, int curSIdx) {
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
            computeSingleS(sOuterLoopIdx, seqIdx);
      }
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSNoTail<T, U, FORMAT, O, M>::ComputeEachCoreSLoop(int sOuterBlockNum,
                                                                                     int nGropOffeset,
                                                                                     int seqIdx, int s1, int curSIdx) {
    this->batchNOffset = nGropOffeset;

    this->loopSNum = sOuterBlockNum / s1 / 2;  // For load balancing, Souter mirror computing is required.
    int loopStart = this->loopSNum * curSIdx;
    for (uint32_t sOuterLoopIdx = loopStart; sOuterLoopIdx < loopStart + this->loopSNum; sOuterLoopIdx++) {
        // Computing one Souter.
        computeSingleS(sOuterLoopIdx, seqIdx);
        // Mirror computing the next Souter.
        uint32_t mirroredSouterLoopIdex = sOuterBlockNum - sOuterLoopIdx - 1;
        computeSingleS(mirroredSouterLoopIdex, seqIdx);
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O, Mode M>
__aicore__ inline void PromptFlashAttentionSplitNSNoTail<T, U, FORMAT, O, M>::ComputeEachCore(uint32_t coreIdx) {
    this->spmTmpSize = this->tilingData->promptAttentionTensorSizeRect.spmTmpSize;
    this->mmResUbSize = this->tilingData->promptAttentionTensorSizeRect.mmResUbSize;
    this->bmm2ResUbSize = this->tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize;
    int reuseWorkspaceRatio = this->tilingData->promptAttentionSingleCoreParams.multiSmaxsInnerLoopTimes;
    this->mm.SetWorkspace((__gm__ uint8_t*)this->workspaceGm[GetBlockNum() * GetTaskRation() * this->spmTmpSize +
        coreIdx * this->mmResUbSize * reuseWorkspaceRatio].GetPhyAddr(), this->mmResUbSize * reuseWorkspaceRatio);

    uint32_t buff_offset = GetBlockNum() * GetTaskRation() * (this->spmTmpSize +
                           this->mmResUbSize * reuseWorkspaceRatio);
    this->bmm2.SetWorkspace((__gm__ uint8_t*)this->workspaceGm[buff_offset + coreIdx * this->bmm2ResUbSize].GetPhyAddr(), this->bmm2ResUbSize);

    uint32_t actualSeqLengthsIdx = 0;
    for (int sIdx = 0; sIdx <  this->tilingData->promptAttentionBaseParams.dimNumOfseq; sIdx++) {
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
        this->singleProcessSOuterSize = this->singleProcessSOuterSizeWhole;
        // n=1. In the s direction, each core processes an even number of Souter.
        if (coreHeadNum == 1 && s1 > 1 && sOuterBlockNum % s1 == 0 && sOuterBlockNum / s1 % 2 == 0) {
            ComputeEachCoreSLoop(sOuterBlockNum, nGropOffeset, sIdx, s1, curSIdx);
        }
        else {
            ComputeEachCoreNSLoop(coreHeadNum, sOuterBlockNum, nGropOffeset, sIdx, s1, curSIdx);
        }
    }
}

#endif  // PROMPT_FLASH_ATTENTION_SCORE_SPLIT_N_S_NO_TAIL_H
