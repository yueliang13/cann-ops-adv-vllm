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
 * \file prompt_flash_attention_bnstilling_n_s_no_tailWBNSD_KV_NZ.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_SCORE_BNSTILLING_N_S_WITHBNSD_NO_TAIL_KV_NZ_H
#define PROMPT_FLASH_ATTENTION_SCORE_BNSTILLING_N_S_WITHBNSD_NO_TAIL_KV_NZ_H

#include "prompt_flash_attention_s1s2_bns1_x310_base.h"
#include "prompt_flash_attention_nz_kv_base.h"

using namespace matmul;
template<typename T, typename U, CubeFormat FORMAT, typename O, ModeNZ M = ModeNZ::HighPerformanceNZ>
class PromptFlashAttentionBNSTillingNSWithBNSDNoTailKVNZ : public PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M> {
public:
    // define datatype
    using mmInputType = typename PromptFlashAttentionTypeTraitsNZ<T, M>::mmInputType;
    using mmBiasType = typename PromptFlashAttentionTypeTraitsNZ<T, M>::mmBiasType;
    using mmOutputType = typename PromptFlashAttentionTypeTraitsNZ<T, M>::mmOutputType;
    using softmaxType = typename PromptFlashAttentionTypeTraitsNZ<T, M>::softmaxType;
    __aicore__ inline PromptFlashAttentionBNSTillingNSWithBNSDNoTailKVNZ() {};
    __aicore__ inline void Process();

protected:
    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t sinnerSize, uint32_t sInnerIdx);

    __aicore__ inline void Bmm1ResDoVecBmm2Compute(LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
                                            LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb,
                                            bool isLast, bool isSecond, event_t eventID);

    __aicore__ inline void Bmm1ResDoVecBmm2ComputeFirst(LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
                                                        LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb,
                                                        bool isLast, event_t eventID);

    __aicore__ inline void ComputeEachCoreSInnerLoop(uint32_t outerLoopIndex);

    __aicore__ inline void ComputeEachCore(uint32_t coreIdx);
};

template<typename T, typename U, CubeFormat FORMAT, typename O, ModeNZ M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSWithBNSDNoTailKVNZ<T, U, FORMAT, O, M>::Process() {
    ComputeEachCore(this->tmp_block_idx);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, ModeNZ M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSWithBNSDNoTailKVNZ<T, U, FORMAT, O, M>::AttenMaskCopyIn(uint64_t offset,
                                                                                             uint32_t sinnerSize,
                                                                                             uint32_t sInnerLoopIdx) {
    LocalTensor<U> tmpUb = this->tmpSoftmaxFlashV2Ub_.template Get<U>(this->attenMaskUbSize);
    LocalTensor<mmOutputType> attenMaskUb = this->attenMaskUb_.template Get<mmOutputType>(this->attenMaskUbSize);
    LocalTensor<mmOutputType> tmpAttenMaskUb = this->Bmm1Queue.template AllocTensor<mmOutputType>();
    tmpUb.SetSize(this->singleProcessSOuterSize * sinnerSize);
    DataCopyParams intriParams;
    intriParams.blockCount = this->singleProcessSOuterSize;
    intriParams.blockLen = sinnerSize / this->maskTypeByteNum;
    intriParams.srcStride = (this->tilingData->promptAttentionBaseParams.seqInnerSize - sinnerSize) /
                            this->maskTypeByteNum;
    intriParams.dstStride = 0;

    DataCopy(tmpUb, this->attenMaskGm[offset], intriParams);
    this->AttenMaskTransND2NZ(attenMaskUb, tmpAttenMaskUb, this->singleProcessSInnerSize, this->singleProcessSOuterSize);
    this->Bmm1Queue.FreeTensor(tmpAttenMaskUb);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, ModeNZ M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSWithBNSDNoTailKVNZ<T, U, FORMAT, O, M>::ComputeEachCoreSInnerLoop(uint32_t outerLoopIndex) {
    bool isSecond = true;
    this->b1Local_ = this->b1Buf_.template Get<mmOutputType>();
    this->b2Local_ = this->b2Buf_.template Get<mmOutputType>();
    if (this->isOuterLoopStart_) {
        // L1 Residency SetTensorA to Obtain L1
        this->CopyND2NZOnTheFly(this->b1Local_, this->keyGm[this->tensorBCoreOffset], this->singleProcessSInnerSize, 
            this->tilingData->promptAttentionBaseParams.headSize, this->tilingData->promptAttentionBaseParams.headSize, true);      
    }

    this->Bmm1Compute(this->tensorACoreOffset, this->b1Local_);
    if (!this->isOuterLoopLast_) {
        this->tensorACoreOffset += this->singleProcessSOuterSize * this->tilingData->promptAttentionBaseParams.headSize;
    }

    LocalTensor<float> softmaxMaxUb = this->softmaxOutQueue.template AllocTensor<float>();
    LocalTensor<float> softmaxSumUb = softmaxMaxUb[this->softmaxMaxSize];
    this->attenMaskOffset = this->attenMaskCoreOffset;
    this->valueOffset = this->valueCoreOffset;
    LocalTensor<mmOutputType> mmResUb = this->Bmm1Queue.template AllocTensor<mmOutputType>();

    this->singleProcessSInnerSizeNow = this->singleProcessSInnerSize;
    this->singleProcessSInnerBmmTail = this->singleProcessSInnerSize;
    this->maskCopyInCol = this->singleProcessSInnerSize;
    this->needCalMask_ = this->useMask;

    this->mm.template GetTensorC<false>(mmResUb, false, true);
    this->mm.End();

    this->ElewiseCompute310P(mmResUb, this->singleProcessSInnerSizeNow, this->singleProcessSOuterSize);
    pipe_barrier(PIPE_V);
    LocalTensor<uint8_t> sharedTmpUb = this->tmpSoftmaxFlashV2Ub_.template Get<uint8_t>(this->softMaxV2Size_);
    // softmax tiling will change when next loop
    this->SoftmaxBasicComputeFirstNoTail(mmResUb, softmaxMaxUb, softmaxSumUb, this->softmaxExpUb, sharedTmpUb);

    LocalTensor<mmOutputType> bmm2ResUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);

    event_t evtIdV2MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evtIdV2MTE3);
    WaitFlag<HardEvent::V_MTE3>(evtIdV2MTE3);
    uint32_t valueIndex = this->tensorBCoreOffset + outerLoopIndex * this->singleProcessSOuterSize * this->tilingData->promptAttentionBaseParams.headSize;
    this->CopyND2NZOnTheFlyPerBlock(outerLoopIndex, this->b2Local_, this->valueGm[valueIndex], this->singleProcessSOuterSize, 
            this->tilingData->promptAttentionBaseParams.headSize, this->tilingData->promptAttentionBaseParams.headSize, true);
    
    for (int sInnerLoopIdx = 0; sInnerLoopIdx < outerLoopIndex + 1; sInnerLoopIdx++) {
        if (sInnerLoopIdx == 0) {
            bmm2ResUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);
        } else {
            bmm2ResUb = this->tempBmm2Queue.template AllocTensor<mmOutputType>();
        }
        this->Bmm2Compute(this->b2Local_[sInnerLoopIdx * this->singleProcessSOuterSize * this->tilingData->promptAttentionBaseParams.headSize],
                          mmResUb[sInnerLoopIdx * this->singleProcessSOuterSize * this->singleProcessSOuterSize]);
        LocalTensor<uint8_t> mm2TmpUb = this->tmpmm2Ub_.template Get<uint8_t>(this->mm2TmpUbSize_);
        this->bmm2.SetLocalWorkspace(mm2TmpUb);
        this->bmm2.template Iterate<false>();
        this->bmm2.template GetTensorC<false>(bmm2ResUb, false, true);
        if (sInnerLoopIdx != 0) {
            this->Bmm2UpdateAdd(bmm2ResUb);
            this->tempBmm2Queue.FreeTensor(bmm2ResUb);
        }
    }

    this->Bmm1Queue.FreeTensor(mmResUb);
    pipe_barrier(PIPE_V);
    LocalTensor<mmOutputType> bmm2ResPreUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);
    this->Bmm2UpdateDivNoTail310P(bmm2ResPreUb, softmaxSumUb, this->softmaxExpUb);
    this->softmaxOutQueue.FreeTensor(softmaxMaxUb);
    LocalTensor<mmOutputType> FinalResUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);
    this->DataCopyOutWithBNSD(FinalResUb);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, ModeNZ M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSWithBNSDNoTailKVNZ<T, U, FORMAT, O, M>::ComputeEachCore(uint32_t coreIdx) {
    this->spmTmpSize = this->tilingData->promptAttentionTensorSizeRect.spmTmpSize;
    this->mmResUbSize = this->tilingData->promptAttentionTensorSizeRect.mmResUbSize;
    this->bmm2ResUbSize = this->tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize;

    int actualCoreNums = this->tilingData->promptAttentionSingleCoreParams.actualCoreNums;
    if (g_coreType == AIV && coreIdx >= actualCoreNums) {
        return;
    }
    int sNum = this->tilingData->promptAttentionBaseParams.dimNumOfseq;

    // Temporary reuse
    // CoreHeadNumTail to coreNStart
    // actualS1 to coreNEnd
    // actualCoreNums to coreSidStart
    // singleCoreHeadNumSize to coreSidEnd
    int sIdStart = this->tilingData->promptAttentionSeqParams.actualCoreNums[coreIdx];
    int sIdEnd = this->tilingData->promptAttentionSeqParams.singleCoreHeadNumSize[coreIdx];
    int outerLoopStart = this->tilingData->promptAttentionSeqParams.coreSeqPosStart[coreIdx];
    int outerLoopEnd = this->tilingData->promptAttentionSeqParams.coreSeqPosEnd[coreIdx];
    uint32_t nLoopStart = this->tilingData->promptAttentionSeqParams.CoreHeadNumTail[coreIdx];
    uint32_t nLoopEnd = this->tilingData->promptAttentionSeqParams.actualS1[coreIdx];
    int32_t preTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.preTokens);
    int32_t nextTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.nextTokens);

    int tmpOuterLoopEnd;
    int tmpSLoopEnd;
    bool isLast = false;

    uint32_t actualSeqLengthsIdx = 0;

    for (uint32_t loopNIdx = nLoopStart; loopNIdx < nLoopEnd; loopNIdx++) {
        this->batchNOffset = loopNIdx;
        if (loopNIdx != nLoopEnd - 1) {
            tmpSLoopEnd = sNum;
        } else {
            tmpSLoopEnd = sIdEnd;
            isLast = true;
        }
        for (int sIdx = sIdStart; sIdx < tmpSLoopEnd; sIdx++) {
            this->GetSingleCoreParam(sIdx);
            actualSeqLengthsIdx = this->isActualLenDimsNull ? this->tilingData->promptAttentionBaseParams.seqSize : this->actualSeqLengthsGm.GetValue(sIdx);
            actualSeqLengthsIdx = (this->attentionMaskType == 0 && (int64_t)actualSeqLengthsIdx >
                               (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize +
                               (int64_t)this->tilingData->promptAttentionBaseParams.preTokens) ?
                               this->tilingData->promptAttentionBaseParams.seqInnerSize + this->tilingData->promptAttentionBaseParams.preTokens :
                               actualSeqLengthsIdx;
            int sOuterBlockNum = (actualSeqLengthsIdx + this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize - 1) /
                                  this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
            this->multiSeqOffset = this->actualSeqOffsets[sIdx];
            if (isLast && sIdx == tmpSLoopEnd - 1) {
                tmpOuterLoopEnd = outerLoopEnd;
            } else {
                tmpOuterLoopEnd = sOuterBlockNum;
            }
            this->singleProcessSOuterSize = this->singleProcessSOuterSizeWhole;
            for (uint32_t sOuterLoopIdx = outerLoopStart; sOuterLoopIdx < tmpOuterLoopEnd; sOuterLoopIdx++) {
                if ((preTokens <= this->tilingData->promptAttentionBaseParams.seqSize) || (nextTokens <= this->tilingData->promptAttentionBaseParams.seqInnerSize)) {
                    AttenMaskCopyIn(sOuterLoopIdx * this->singleProcessSOuterSize * this->singleProcessSInnerSize,
                                    this->singleProcessSInnerSize, 0);                
                }
                this->isOuterLoopLast_ = sOuterLoopIdx == tmpOuterLoopEnd - 1;
                this->isOuterLoopStart_ = sOuterLoopIdx == outerLoopStart;
                this->sOuterOffset = sOuterLoopIdx * this->singleProcessSOuterSizeWhole;
                this->LoopSOuterOffsetInitWithBNSD(this->actualSeqOffsets[sIdx], sIdx);
                ComputeEachCoreSInnerLoop(sOuterLoopIdx);
            }
            outerLoopStart = 0;
        }
        sIdStart = 0;
    }
}

#endif  // PROMPT_FLASH_ATTENTION_SCORE_BNSTILLING_N_S_WITHBNSD_NO_TAIL_KV_NZ_H