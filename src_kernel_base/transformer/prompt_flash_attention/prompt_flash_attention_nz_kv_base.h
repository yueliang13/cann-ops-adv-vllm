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
 * \file prompt_flash_attention_nz_kv_base.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_NZ_KV_BASE_H
#define PROMPT_FLASH_ATTENTION_NZ_KV_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_data_copy_transpose.h"
#include "prompt_flash_attention_s1s2_bns1_x310_base.h"

using namespace matmul;

template<typename T, typename U, CubeFormat FORMAT, typename O, ModeNZ M = ModeNZ::HighPerformanceNZ>
class PromptFlashAttentionNZKVBase {
public:
    __aicore__ inline PromptFlashAttentionNZKVBase() {};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
                                __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                const PromptFlashAttentionTilingData* __restrict tiling,
                                __gm__ uint8_t* gmTiling, TPipe* tPipe);
    __aicore__ inline void Process();
    // define datatype
    using mmInputType = typename PromptFlashAttentionTypeTraitsNZ<T, M>::mmInputType;
    using mmBiasType = typename PromptFlashAttentionTypeTraitsNZ<T, M>::mmBiasType;
    using mmOutputType = typename PromptFlashAttentionTypeTraitsNZ<T, M>::mmOutputType;
    using softmaxType = typename PromptFlashAttentionTypeTraitsNZ<T, M>::softmaxType;
    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, mmInputType, false>;
    using b1Type = MatmulType<TPosition::A1, CubeFormat::NZ, mmInputType, true>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
    using c1Type = MatmulType<TPosition::VECCALC, CubeFormat::NZ, mmOutputType>;
    Matmul<a1Type, b1Type, c1Type, bias1Type> mm;
    // define batchmatmul
    using a2Type = MatmulType<TPosition::VECCALC, CubeFormat::NZ, mmInputType, false>;
    using b2Type = MatmulType<TPosition::A1, CubeFormat::NZ, mmInputType, false>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
#if defined(__CCE_KT_TEST__)
    using c2Type = MatmulType<TPosition::VECCALC, CubeFormat::NZ, mmOutputType>; //  cpu
#else
    using c2Type = MatmulType<TPosition::VECCALC, CubeFormat::NZ, mmOutputType>; // npu
#endif
    Matmul<a2Type, b2Type, c2Type, bias2Type> bmm2;

protected:
    const PromptFlashAttentionTilingData* __restrict tilingData;
    TPipe* pipe;
    // define the que
    TBuf<> attenMaskUb_;
    TBuf<QuePosition::A1> b1Buf_;
    TBuf<QuePosition::A1> b2Buf_;
    TQue<QuePosition::VECIN, 1> eleWiseInQueue;
    TQue<QuePosition::VECOUT, 1> tempBmm2Queue;
    TQue<QuePosition::VECOUT, 1> Bmm1Queue;
    TQue<QuePosition::VECOUT, 1> softmaxOutQueue;
    TBuf<> selectSpaceUb;

    TBuf<> softmaxExpUb_;
    TBuf<> tempBmm2Ub;

    TBuf<> tmpSoftmaxFlashV2Ub_;
    TBuf<> tmpmm2Ub_;

    LocalTensor<softmaxType> softmaxExpUb;
    LocalTensor<mmOutputType> b1Local_;
    LocalTensor<mmOutputType> b2Local_;

    GlobalTensor<T> queryGm;
    GlobalTensor<T> keyGm;
    GlobalTensor<T> valueGm;
    GlobalTensor<U> attenMaskGm;
    GlobalTensor<O> attentionOutGm;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    GlobalTensor<softmaxType> workspaceGm;
    GlobalTensor<mmOutputType> workspaceGmProcessT;

    uint64_t dequantScale1; // quant: define quant variable
    float quantScale1;
    uint64_t dequantScale2;
    float quantScale2;
    float quantOffset2;

    bool isOuterLoopLast_ = false;
    bool isOuterLoopStart_ = false;
    bool useMask;
    bool needCalMask_ = false;
    uint32_t tmp_block_idx;
    uint32_t loopSNum;
    int32_t sOuterOffset;
    uint32_t batchNOffset;
    uint32_t maskOffset;
    uint32_t maskCoreOffset;
    uint64_t attenMaskCoreOffset;
    uint32_t valueOffset; // offset of value
    uint32_t valueCoreOffset;
    uint64_t attenMaskOffset;
    uint32_t tensorAOffset;
    uint32_t tensorBOffset;
    uint32_t tensorACoreOffset;
    uint32_t tensorBCoreOffset;
    uint32_t attentionOutOffset; //offset of out
    uint32_t offsetSS;
    uint32_t offsetSH;
    uint32_t offsetSTypeNum;
    uint32_t offsetNSTypeNum;
    uint32_t offsetNSS;
    uint32_t offsetNSH;
    uint32_t maskDataType;
    uint32_t attenMaskBatch;
    uint32_t maskCopyInCol;

    // tilingdata
    uint32_t singleProcessSOuterSizeWhole;
    uint32_t singleProcessSOuterSize;
    uint32_t singleProcessSOuterSizeTail;
    uint32_t singleProcessSInnerSize;
    uint32_t singleProcessSInnerSizeNow;
    uint32_t singleProcessSInnerSizeTail;
    uint32_t singleProcessSInnerBmmTail;
    uint32_t mmResUbSize;
    uint32_t attenMaskUbSize;
    uint32_t maskSize;
    uint32_t softmaxMaxSize;
    uint32_t softmaxSumSize;
    uint32_t softmaxExpSize; // size of softmax exp
    uint32_t spmTmpSize;
    uint32_t scmTmpSize;
    uint32_t bmm2ResUbSize;
    uint32_t tmpMMResBmm2PreUbSize;
    uint32_t tmpSoftmaxBmm2UbSize;
    uint32_t padSize;
    uint32_t unalignSInner;
    uint32_t typeByteNum;
    uint32_t outputTypeByteNum;
    uint32_t softmaxTypeByteNum;
    uint32_t headNumRatio;
    uint32_t maskTypeByteNum;
    uint32_t selectSpaceUbSize;
    uint32_t softMaxV2Size_;
    uint32_t mm2TmpUbSize_;
    uint32_t splitS2;
    uint32_t layoutType; // type of layout
    uint32_t maskInnerTailAlign;

    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxFlashTilingData;
    CopyTransposeTiling transposeTilingData;
    uint32_t bmm2_settail_m;
    uint32_t sOuterSizeTail_tmp;
    uint32_t tailCoresOuterSizeTail_tmp;
    uint32_t MultiHeadQ;
    uint32_t MultiHeadKV;
    uint32_t startInBlock;
    uint32_t maxInnerLoopTimes;
    uint32_t seqListOffset;
    int64_t multiSeqOffset;

    uint32_t attentionMaskStride;
    int32_t attentionMaskType;
    uint32_t attentionMaskMaxSize;

    bool isActualLenDimsNull; // true: actual seq length is null
    bool isActualLenDimsKVNull;
    int32_t actualSeqLengthPerBatch;
    int32_t actualSeqLengthKVPerBatch;
    uint32_t accumSOuterTilingNums[BATCH_NUM_MAX_NZ];
    uint32_t actualSeqOffsets[BATCH_NUM_MAX_NZ];

    __aicore__ inline void ElewiseCompute310P(LocalTensor<mmOutputType>& mmResUb, uint32_t SInnerSize, uint32_t SOuterSize);

    __aicore__ inline void AttenMaskTransND2NZ(LocalTensor<mmOutputType> &dstUb, LocalTensor<mmOutputType> &srcUb, uint32_t SInnerSize, uint32_t SOuterSize);

    __aicore__ inline void SoftmaxBasicComputeFirstNoTail(LocalTensor<mmOutputType>& mmResUb,
                                                          LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                          LocalTensor<softmaxType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb);

    __aicore__ inline void QuantCompute(LocalTensor<int8_t> quantResUb, LocalTensor<mmOutputType> mmResUb, float scale, float offset, uint32_t computeSize);

    __aicore__ inline void CopyND2NZOnTheFly(const LocalTensor<mmOutputType>& dst,  const GlobalTensor<mmOutputType>& src,
        const int height, const int width, const int gCol, const bool isA1);

    __aicore__ inline void CopyND2NZOnTheFlyPerBlock(uint32_t outerLoopIndex, const LocalTensor<mmOutputType>& dst,  const GlobalTensor<mmOutputType>& src,
        const int height, const int width, const int gCol, const bool isA1);

    __aicore__ inline void Bmm2UpdateDivNoTail310P(LocalTensor<mmOutputType>& bmm2ResPreUb,
                                               LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb);

    __aicore__ inline void Bmm2Compute(LocalTensor<mmOutputType> b2Local, LocalTensor<mmOutputType> bmm2ResL1);

    __aicore__ inline void UpdateVmul(LocalTensor<softmaxType>& softmaxExpUb);

    __aicore__ inline void Bmm2UpdateAdd(LocalTensor<mmOutputType>& bmm2ResUb);

    __aicore__ inline void DataCopyTransposeOut(LocalTensor<mmOutputType>& bmm2ResUb);

    __aicore__ inline void DataCopyOutWithBNSD(LocalTensor<mmOutputType>& bmm2ResUb);

    __aicore__ inline void ComputeAttenMaskOffset(int sInnerOffsetDataSize);

    __aicore__ inline void ComputeOffset(uint32_t sInnerLoopIdx);

    __aicore__ inline void ComputeOffsetWithBNSD(uint32_t sInnerLoopIdx);

    __aicore__ inline void LoopSOuterOffsetInitWithBNSD(uint32_t seqListOffsetSize, int sIdx);

    __aicore__ inline void LoopSOuterOffsetInit(uint32_t seqListOffsetSize, int sIdx);

    __aicore__ inline void initOffset();

    __aicore__ inline void InitTensorSize(const PromptAttentionSingleCoreTensorSize* tensorSizeTiling);

    __aicore__ inline void GetSingleCoreParam(int sIdx);

    __aicore__ inline void InitOutputSingleCore();

    __aicore__ inline void Bmm1Compute(uint32_t offset, LocalTensor<mmOutputType>& b1Local);
};

template<typename T, typename U, CubeFormat FORMAT, typename O, ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::Init(__gm__ uint8_t* query, __gm__ uint8_t* key,
                                        __gm__ uint8_t* value, __gm__ uint8_t* pseShift,
                                        __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengths,
                                        __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
                                        __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                        __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                        const PromptFlashAttentionTilingData* __restrict tiling,
                                        __gm__ uint8_t* gmTiling, TPipe* tPipe) {
    tmp_block_idx = GetBlockIdx();
    // init global buffer
    tilingData = tiling;
    queryGm.SetGlobalBuffer((__gm__ T*)query);
    keyGm.SetGlobalBuffer((__gm__ T*)key);
    valueGm.SetGlobalBuffer((__gm__ T*)value);
    attentionOutGm.SetGlobalBuffer((__gm__ O*)attentionOut);
    workspaceGm.SetGlobalBuffer((__gm__ softmaxType*)workspace);

    pipe = tPipe;
    typeByteNum = tilingData->promptAttentionBaseParams.typeByteNum; // byte num of type
    outputTypeByteNum = tilingData->promptAttentionBaseParams.outputTypeByteNum;
    softmaxTypeByteNum = tilingData->promptAttentionBaseParams.softmaxTypeByteNum;
    headNumRatio = tilingData->promptAttentionBaseParams.headNumRatio;
    maskDataType = tilingData->promptAttentionBaseParams.attenMaskElemType;
    maskTypeByteNum = tilingData->promptAttentionBaseParams.maskTypeByteNum;
    attenMaskBatch = tilingData->promptAttentionSingleCoreParams.attenMaskBatch;
    layoutType = tilingData->promptAttentionBaseParams.layoutType;
    initOffset();

    isActualLenDimsNull = true;
    if (!tilingData->promptAttentionBaseParams.isActualSeqLengthsNull) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, tilingData->promptAttentionBaseParams.batchSize);
        isActualLenDimsNull = false;
    }

    isActualLenDimsKVNull = true;
    if (!tilingData->promptAttentionBaseParams.isActualSeqLengthsKVNull) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, tilingData->promptAttentionBaseParams.batchSize);
        isActualLenDimsKVNull = false;
    }

    uint32_t preAccumSOuter = 0;
    uint32_t h = tilingData->promptAttentionBaseParams.headNumSize * tilingData->promptAttentionBaseParams.headSize;
    uint32_t s = tilingData->promptAttentionBaseParams.seqSize;
    uint32_t middle_actualSeqLengths = 0;
    uint32_t actualSeqLengthsIdx = 0;
    for (int i = 0; i < tilingData->promptAttentionBaseParams.batchSize; i++) {
        actualSeqLengthsIdx = isActualLenDimsNull ? tilingData->promptAttentionBaseParams.seqSize : actualSeqLengthsGm.GetValue(i);
        if (tilingData->promptAttentionBaseParams.isActualSeqLengthsNull) {
            actualSeqOffsets[i] = i * s * h;
        } else { // actual seq length is not null
            if (tilingData->promptAttentionBaseParams.isLayoutSH) {
                actualSeqOffsets[i] = middle_actualSeqLengths * h;
                middle_actualSeqLengths += actualSeqLengthsIdx;
            } else {
                actualSeqOffsets[i] = i * s * h;
            }
        }

        actualSeqLengthsIdx = ((int64_t)actualSeqLengthsIdx >
                               (int64_t)tilingData->promptAttentionBaseParams.seqInnerSize +
                               (int64_t)tilingData->promptAttentionBaseParams.preTokens) ?
                               tilingData->promptAttentionBaseParams.seqInnerSize + tilingData->promptAttentionBaseParams.preTokens :
                               actualSeqLengthsIdx;
        accumSOuterTilingNums[i] = (((actualSeqLengthsIdx + tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize - 1) /
                            tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize) *
                            tilingData->promptAttentionBaseParams.headNumSize) +
                            preAccumSOuter;
        preAccumSOuter = accumSOuterTilingNums[i];
    }
    accumSOuterTilingNums[0] = (headNumRatio != 1 ||
                                tilingData->promptAttentionInitOutputParams.needInit ||
                                tilingData->promptAttentionBaseParams.batchSize != 1) ?
                                0 : accumSOuterTilingNums[0];

    pipe->InitBuffer(b1Buf_, tilingData->promptAttentionTensorSizeRect.scmTmpSize * sizeof(mmOutputType));
    pipe->InitBuffer(b2Buf_, tilingData->promptAttentionTensorSizeRect.scmTmpSize * sizeof(mmOutputType));
    pipe->InitBuffer(attenMaskUb_, 2 * (tilingData->promptAttentionTensorSizeRect.attenMaskUbSize) * sizeof(U));
    pipe->InitBuffer(softmaxOutQueue, 1, 2 * tilingData->promptAttentionTensorSizeRect.softmaxMaxSize * sizeof(float));
    pipe->InitBuffer(softmaxExpUb_, tilingData->promptAttentionTensorSizeRect.softmaxExpSize * sizeof(softmaxType));
    softmaxExpUb = softmaxExpUb_.Get<softmaxType>(tilingData->promptAttentionTensorSizeRect.softmaxExpSize);

    pipe->InitBuffer(tempBmm2Ub, tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize * sizeof(mmOutputType));
    pipe->InitBuffer(tempBmm2Queue, 1, tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize * sizeof(mmOutputType));

    pipe->InitBuffer(Bmm1Queue, 1, tilingData->promptAttentionTensorSizeRect.mmResUbSize * sizeof(mmOutputType));
    if ((tilingData->promptAttentionTensorSizeRect.tmpSoftMaxV2Size) != 0 && (tilingData->promptAttentionTensorSizeRect.mm2TmpUbSize != 0)) {
        pipe->InitBuffer(tmpSoftmaxFlashV2Ub_, tilingData->promptAttentionTensorSizeRect.tmpSoftMaxV2Size / UB_ALIGN_NZ * UB_ALIGN_NZ * sizeof(uint8_t));
        pipe->InitBuffer(tmpmm2Ub_, tilingData->promptAttentionTensorSizeRect.mm2TmpUbSize * sizeof(uint8_t));
    }

    useMask = false;
    attentionMaskType = tilingData->promptAttentionBaseParams.sparseMode;
    if ((attenMask != NULL) && (tilingData->promptAttentionBaseParams.useMask == 1)) {
        useMask = true;
        attenMaskGm.SetGlobalBuffer((__gm__ U*)attenMask);
        attentionMaskStride = tilingData->promptAttentionBaseParams.maskKVsSize;
    }

    if (tilingData->promptAttentionInitOutputParams.needInit == 1) {
        InitOutputSingleCore();
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::InitOutputSingleCore()
{
    auto &initParams = tilingData->promptAttentionInitOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - tmp_block_idx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    InitOutput<O>(attentionOutGm[tmp_block_idx * initParams.singleCoreSize], singleInitOutputSize, 0);
    SyncAll();
}

template<>
__aicore__ inline void PromptFlashAttentionNZKVBase<int8_t, bool, CubeFormat::ND, int8_t>::InitOutputSingleCore() {}

template<>
__aicore__ inline void PromptFlashAttentionNZKVBase<int8_t, half, CubeFormat::ND, int8_t>::InitOutputSingleCore() {}

template<>
__aicore__ inline void PromptFlashAttentionNZKVBase<int8_t, float, CubeFormat::ND, int8_t>::InitOutputSingleCore() {}

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::initOffset() {
    offsetSS = tilingData->promptAttentionBaseParams.seqSize * tilingData->promptAttentionBaseParams.seqSize;
    offsetSH = tilingData->promptAttentionBaseParams.seqSize * tilingData->promptAttentionBaseParams.headSize;
    offsetSTypeNum = tilingData->promptAttentionBaseParams.seqSize * typeByteNum;
    offsetNSTypeNum = tilingData->promptAttentionBaseParams.headNumSize * offsetSTypeNum;
    offsetNSS = tilingData->promptAttentionBaseParams.headNumSize * offsetSS;
    offsetNSH = tilingData->promptAttentionBaseParams.headNumSize * offsetSH;
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::InitTensorSize(
                const PromptAttentionSingleCoreTensorSize* tensorSizeTiling) {
    mmResUbSize = tensorSizeTiling->mmResUbSize;
    attenMaskUbSize = tensorSizeTiling->attenMaskUbSize;
    maskSize = tensorSizeTiling->maskSize;
    softmaxMaxSize = tensorSizeTiling->softmaxMaxSize;     // Softmax maximum values UB size
    softmaxSumSize = tensorSizeTiling->softmaxSumSize;
    softmaxExpSize = tensorSizeTiling->softmaxExpSize;
    spmTmpSize = tensorSizeTiling->spmTmpSize;             // Temp UB size for sparse matrices
    scmTmpSize = tensorSizeTiling->scmTmpSize;
    bmm2ResUbSize = tensorSizeTiling->bmm2ResUbSize;
    tmpMMResBmm2PreUbSize = tensorSizeTiling->tmpMMResBmm2PreUbSize;
    tmpSoftmaxBmm2UbSize = tensorSizeTiling->tmpSoftmaxBmm2UbSize;
    selectSpaceUbSize = tensorSizeTiling->selectSpaceUbSize;
    softMaxV2Size_ = tensorSizeTiling->tmpSoftMaxV2Size / UB_ALIGN_NZ * UB_ALIGN_NZ;
    mm2TmpUbSize_ = tensorSizeTiling->mm2TmpUbSize;
}

template <typename T, typename U, CubeFormat FORMAT, typename O, ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::AttenMaskTransND2NZ(LocalTensor<mmOutputType> &dstUb, LocalTensor<mmOutputType> &srcUb,
                                                                                      uint32_t SInnerSize, uint32_t SOuterSize)
{
    struct DataCopyParams dataCopyParams;
    LocalTensor<int8_t> tmpUb = this->tmpSoftmaxFlashV2Ub_.template Get<int8_t>(this->attenMaskUbSize);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(srcUb, tmpUb, RoundMode::CAST_NONE, srcUb.GetSize());
    int32_t calHigh = (SOuterSize + BLOCK_CUBE - 1) / BLOCK_CUBE;
    int32_t calWidth = SInnerSize / BLOCK_CUBE;
    dataCopyParams.blockCount = SOuterSize;
    dataCopyParams.blockLen = 1;
    dataCopyParams.srcStride = SInnerSize / BLOCK_CUBE - 1;
    dataCopyParams.dstStride = 0;
    for(int i = 0; i < calWidth; i++) {
        DataCopy(dstUb[i * BLOCK_CUBE * singleProcessSOuterSize],srcUb[i * BLOCK_CUBE], dataCopyParams);
    }
    pipe_barrier(PIPE_V);
    Muls(dstUb, dstUb, static_cast<mmOutputType>(-10000.0), SOuterSize * SInnerSize);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::ElewiseCompute310P(LocalTensor<mmOutputType>& mmResUb, uint32_t SInnerSize, uint32_t SOuterSize) {
    uint32_t computeSize = SInnerSize * SOuterSize;
    Muls(mmResUb, mmResUb, static_cast<mmOutputType>(tilingData->promptAttentionBaseParams.scaleValue), computeSize);
    pipe_barrier(PIPE_V);
    if (needCalMask_) {
        LocalTensor<mmOutputType> tmpUb = this->attenMaskUb_.template Get<mmOutputType>(this->attenMaskUbSize);
        Add(mmResUb, mmResUb, tmpUb, computeSize);
        pipe_barrier(PIPE_V);
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::SoftmaxBasicComputeFirstNoTail(LocalTensor<mmOutputType>& mmResUb,
                                                        LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                        LocalTensor<softmaxType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb) {
    SoftmaxFlashV2<softmaxType, false, true, true, true>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                         mmResUb, softmaxExpUb, softmaxSumUb, softmaxMaxUb, sharedTmpUb, softmaxFlashTilingData);
                                                        }

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::Bmm2UpdateDivNoTail310P(LocalTensor<mmOutputType>& bmm2ResPreUb,
                                            LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb) {
    int32_t headLoop = tilingData->promptAttentionBaseParams.headSize / softmaxTypeByteNum;
    constexpr int32_t REPEAT_DATA_NUM = 256 / sizeof(softmaxType);

    BinaryRepeatParams repeatParams;
    // Default value for continuous calculation
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = 8;
    repeatParams.src1BlkStride = 1;
    repeatParams.src1RepStride = 8;
    repeatParams.dstBlkStride = 1;
    repeatParams.dstRepStride = 8;

    int32_t loop = tilingData->promptAttentionBaseParams.headSize / BLOCK_CUBE;
    int32_t remain = tilingData->promptAttentionBaseParams.headSize % BLOCK_CUBE;
    int32_t repeat = 16 * singleProcessSOuterSize * sizeof(mmOutputType) / 256;
    if constexpr (IsSameType<softmaxType, half>::value) {
        constexpr int32_t FP32_BLOCK_NUM = 8;
        int32_t calcSize = singleProcessSOuterSize * FP32_BLOCK_NUM;
        LocalTensor<float> tmpBuffer = tmpmm2Ub_.template Get<float>();

        // softmaxsumub s 128*8, need to convert to 128*16 then cast, cause type block need 32byte
        DataCopy(tmpBuffer, softmaxSumUb, {static_cast<uint16_t>(singleProcessSOuterSize), 1, 0, 1});
        DataCopy(tmpBuffer[FP32_BLOCK_NUM], softmaxSumUb, {static_cast<uint16_t>(singleProcessSOuterSize), 1, 0, 1});
        pipe_barrier(PIPE_V);
        Cast(softmaxExpUb, tmpBuffer, RoundMode::CAST_ODD, calcSize * 2);
        pipe_barrier(PIPE_V);

        for (int i = 0; i < loop; i++) {
            pipe_barrier(PIPE_V);
            Div(bmm2ResPreUb[i * BLOCK_CUBE * singleProcessSOuterSize], bmm2ResPreUb[i * BLOCK_CUBE * singleProcessSOuterSize], softmaxExpUb,
                REPEAT_DATA_NUM, repeat, repeatParams);
            pipe_barrier(PIPE_V);
        }
        if (remain) {
            Div(bmm2ResPreUb[loop * REPEAT_DATA_NUM], bmm2ResPreUb[loop * REPEAT_DATA_NUM], softmaxExpUb,
                remain, singleProcessSOuterSize, repeatParams);
        }
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O, ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::Bmm2Compute(LocalTensor<mmOutputType> b2Local, LocalTensor<mmOutputType> bmm2ResL1) {
    bmm2.SetTensorA(bmm2ResL1);
    bmm2.SetTensorB(b2Local);
    bmm2.SetTail(-1, -1, singleProcessSOuterSize);
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::UpdateVmul(LocalTensor<softmaxType>& softmaxExpUb) {
    LocalTensor<mmOutputType> bmm2ResPreUb = tempBmm2Ub.Get<mmOutputType>(bmm2ResUbSize);

    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 8;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1RepStride = 8;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 8;
    int32_t repeat = 16 * singleProcessSOuterSize * sizeof(mmOutputType) / 256;
    constexpr int32_t REPEAT_DATA_NUM = 256 / sizeof(softmaxType);
    // only support singleProcessSOuterSize <=255, headsize 32B align
    int32_t numOneRep = 256 / sizeof(softmaxType);
    int32_t loop = tilingData->promptAttentionBaseParams.headSize / BLOCK_CUBE;
    int32_t remain =  tilingData->promptAttentionBaseParams.headSize % BLOCK_CUBE;

    for (int i = 0; i < loop; i++) {
        pipe_barrier(PIPE_V);
        Mul(bmm2ResPreUb[i * BLOCK_CUBE * singleProcessSOuterSize], softmaxExpUb, bmm2ResPreUb[i * BLOCK_CUBE * singleProcessSOuterSize],
            REPEAT_DATA_NUM, repeat, repeatParams);
        pipe_barrier(PIPE_V);
    }
    if (remain) {
        Mul(bmm2ResPreUb[loop * tilingData->promptAttentionBaseParams.headSize], softmaxExpUb,
            bmm2ResPreUb[loop * tilingData->promptAttentionBaseParams.headSize], remain, singleProcessSOuterSize, repeatParams);
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::ComputeAttenMaskOffset(int sInnerOffsetDataSize) {
    attenMaskOffset = attenMaskCoreOffset + (uint64_t)sInnerOffsetDataSize;
}
template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::ComputeOffset(uint32_t sInnerLoopIdx) {
    int sInnerOffsetDataSize = sInnerLoopIdx * singleProcessSInnerSize;
    ComputeAttenMaskOffset(sInnerOffsetDataSize);
    // tensorBOffset cannot be updated here, as it will erase the previously set values
    valueOffset = valueCoreOffset + sInnerOffsetDataSize * MultiHeadKV;
    tensorAOffset = tensorACoreOffset;
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::ComputeOffsetWithBNSD(uint32_t sInnerLoopIdx) {
    int sInnerOffsetDataSize = sInnerLoopIdx * singleProcessSInnerSize;
    ComputeAttenMaskOffset(sInnerOffsetDataSize);
    valueOffset = valueCoreOffset + sInnerOffsetDataSize * tilingData->promptAttentionBaseParams.headSize;
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::DataCopyTransposeOut(LocalTensor<mmOutputType>& bmm2ResUb) {
    // Copying here needs to consider the conversion from BNSD to BSH and the conversion from NZ to ND.
    struct DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = singleProcessSOuterSize;    // Set block count
    dataCopyParams.blockLen = 1;                            // Set block length
    dataCopyParams.srcStride = 0;                           // Set source data stride
    dataCopyParams.dstStride = MultiHeadQ / BLOCK_CUBE - 1; // Set destination data stride, conversion from NA to ND
    int32_t loop = tilingData->promptAttentionBaseParams.headSize / BLOCK_CUBE;
    event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(enQueEvtID);
    WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
    int64_t startAddr = this->multiSeqOffset +
        batchNOffset * tilingData->promptAttentionBaseParams.headSize +
        sOuterOffset * MultiHeadQ;
    for(int i = 0; i < loop; i++) {
        DataCopy(attentionOutGm[startAddr + i * BLOCK_CUBE], bmm2ResUb[i * BLOCK_CUBE * singleProcessSOuterSize], dataCopyParams);
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::DataCopyOutWithBNSD(LocalTensor<mmOutputType>& bmm2ResUb) {
    uint32_t copySize = this->singleProcessSOuterSize * tilingData->promptAttentionBaseParams.headSize;
    struct DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = singleProcessSOuterSize;
    dataCopyParams.blockLen = 1;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = tilingData->promptAttentionBaseParams.headSize / BLOCK_CUBE - 1;
    int32_t loop = tilingData->promptAttentionBaseParams.headSize / BLOCK_CUBE;
    event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(enQueEvtID);
    WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
    for(int i = 0; i < loop; i++) {
        DataCopy(attentionOutGm[attentionOutOffset + i * BLOCK_CUBE], bmm2ResUb[i * BLOCK_CUBE * singleProcessSOuterSize], dataCopyParams);
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::Bmm1Compute(uint32_t offset,
    LocalTensor<mmOutputType>& b1Local) {
    mm.SetTensorA(queryGm[offset]);
    mm.SetTensorB(b1Local, true);
    mm.SetTail(singleProcessSOuterSize, singleProcessSInnerSize);
    LocalTensor<uint8_t> tmpSoftmaxFlashV2Ub = tmpSoftmaxFlashV2Ub_.template Get<uint8_t>(); // The size of workspace
    mm.SetLocalWorkspace(tmpSoftmaxFlashV2Ub); // Avoidance plan, additional application for localworkspace is required
    mm.template Iterate<false>();
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::LoopSOuterOffsetInit(uint32_t seqListOffsetSize, int sIdx) {
    uint64_t attenMaskBatchOffset = 0;
    if (attenMaskBatch != 1) {
        attenMaskBatchOffset = (uint64_t)sIdx * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize *
                               (uint64_t)tilingData->promptAttentionBaseParams.maskQsSize;
    }

    attenMaskCoreOffset = (uint64_t)sOuterOffset * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize + attenMaskBatchOffset; // mask offset of core

    tensorACoreOffset = seqListOffsetSize +
                        sOuterOffset * MultiHeadQ +
                        batchNOffset * tilingData->promptAttentionBaseParams.headSize; // core offset of tensor A

    uint32_t seqInnerOffsetSize =
        tilingData->promptAttentionBaseParams.seqSize == tilingData->promptAttentionBaseParams.seqInnerSize ?
        seqListOffsetSize / headNumRatio : sIdx * tilingData->promptAttentionBaseParams.seqInnerSize * MultiHeadKV;
    tensorBCoreOffset = seqInnerOffsetSize +
                        batchNOffset / headNumRatio * tilingData->promptAttentionBaseParams.headSize;

    valueCoreOffset = tensorBCoreOffset;
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::LoopSOuterOffsetInitWithBNSD(uint32_t seqListOffsetSize,
                                                                                    int sIdx) {
    uint64_t attenMaskBatchOffset = 0;
    uint32_t head_stride_q = tilingData->promptAttentionBaseParams.headSize *
                             tilingData->promptAttentionBaseParams.seqSize;
    uint32_t head_stride_kv = tilingData->promptAttentionBaseParams.headSize *
                              tilingData->promptAttentionBaseParams.seqInnerSize;
    uint32_t seq_stride = tilingData->promptAttentionBaseParams.headSize;

    if (attenMaskBatch != 1) {
        attenMaskBatchOffset = (uint64_t)sIdx * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize *
                               (uint64_t)tilingData->promptAttentionBaseParams.maskQsSize;
    }
    attenMaskCoreOffset = (uint64_t)sOuterOffset * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize + attenMaskBatchOffset;

    tensorACoreOffset = seqListOffsetSize + batchNOffset * head_stride_q + sOuterOffset*seq_stride;

    uint32_t seqInnerOffsetSize =
        tilingData->promptAttentionBaseParams.seqSize == tilingData->promptAttentionBaseParams.seqInnerSize ?
        seqListOffsetSize / headNumRatio : sIdx * head_stride_kv *
        tilingData->promptAttentionBaseParams.headNumSize / headNumRatio; // calculate the sequence inner offset size based on whether the sequence size equals the inner sequence size.

    tensorBCoreOffset = seqInnerOffsetSize + batchNOffset / headNumRatio * head_stride_kv;

    valueCoreOffset = tensorBCoreOffset;

    attentionOutOffset = seqListOffsetSize + batchNOffset * head_stride_q + sOuterOffset * seq_stride;
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::Bmm2UpdateAdd(LocalTensor<mmOutputType>& bmm2ResUb) {
    LocalTensor<mmOutputType> bmm2ResPreUb = tempBmm2Ub.Get<mmOutputType>(bmm2ResUbSize);
    Add(bmm2ResPreUb, bmm2ResUb, bmm2ResPreUb, bmm2ResUbSize);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::GetSingleCoreParam(int sIdx) {
    singleProcessSInnerSize = tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
    singleProcessSOuterSizeWhole = tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
    MultiHeadQ = tilingData->promptAttentionBaseParams.headSize * tilingData->promptAttentionBaseParams.headNumSize;
    MultiHeadKV = MultiHeadQ / headNumRatio;

    actualSeqLengthPerBatch = isActualLenDimsNull ? tilingData->promptAttentionBaseParams.seqSize :
                              actualSeqLengthsGm.GetValue(sIdx);
    actualSeqLengthPerBatch = ((int64_t)actualSeqLengthPerBatch >
                               (int64_t)tilingData->promptAttentionBaseParams.seqInnerSize +
                               (int64_t)tilingData->promptAttentionBaseParams.preTokens) ?
                               tilingData->promptAttentionBaseParams.seqInnerSize +
                               tilingData->promptAttentionBaseParams.preTokens :
                               actualSeqLengthPerBatch;
    actualSeqLengthKVPerBatch = isActualLenDimsKVNull ? tilingData->promptAttentionBaseParams.seqInnerSize : actualSeqLengthsKVGm.GetValue(sIdx);
    singleProcessSOuterSizeTail = (actualSeqLengthPerBatch % singleProcessSOuterSizeWhole != 0) ?
                                   actualSeqLengthPerBatch % singleProcessSOuterSizeWhole : singleProcessSOuterSizeWhole;
    unalignSInner = (actualSeqLengthKVPerBatch % singleProcessSInnerSize != 0) ?
                     actualSeqLengthKVPerBatch % singleProcessSInnerSize : singleProcessSInnerSize;
    maxInnerLoopTimes = (actualSeqLengthKVPerBatch + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
    singleProcessSInnerSizeTail = (unalignSInner + typeByteNum - 1) / typeByteNum * typeByteNum;
    maskInnerTailAlign = (unalignSInner + maskTypeByteNum - 1) / maskTypeByteNum * maskTypeByteNum;
    padSize = maskInnerTailAlign - unalignSInner;

    InitTensorSize(&tilingData->promptAttentionTensorSizeRect);
    transposeTilingData = tilingData->transposeTilingDataRect;
    softmaxTilingData = tilingData->softmaxTilingDataRect;
    softmaxFlashTilingData = tilingData->softmaxFlashTilingDataRect;
}

template<typename T, typename U, CubeFormat FORMAT, typename O, ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::CopyND2NZOnTheFly(
    const LocalTensor<mmOutputType>& dst,  const GlobalTensor<mmOutputType>& src, const int height,
    const int width, const int gCol, const bool isA1) {
    ASSERT(gCol >= width && "Copy ND block gm->ub width larger than origin matrix width.");
    int calcWidth = width / BLOCK_CUBE; // cube block numbers that do not need to be pad zero
    int dstOffset = 0;
    int srcOffset = 0;
    int calcHeight = (height + BLOCK_CUBE - 1) / BLOCK_CUBE;

    // gCol unaligned ,can not use dma copy repeat stride
    int src_gap = gCol * sizeof(mmOutputType) / UB_ALIGN_NZ - 1;
    for (int i = 0; i < calcWidth; i++) {
        dstOffset = i * calcHeight * CUBE_MAX_SIZE;
        srcOffset = i * BLOCK_CUBE;
        DataCopy(dst[dstOffset], src[srcOffset],
                 { static_cast<uint16_t>(height), 1, static_cast<uint16_t>(src_gap), 0});
    }
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, ModeNZ M>
__aicore__ inline void PromptFlashAttentionNZKVBase<T, U, FORMAT, O, M>::CopyND2NZOnTheFlyPerBlock(
    uint32_t outerLoopIndex, const LocalTensor<mmOutputType>& dst,  const GlobalTensor<mmOutputType>& src, const int height,
    const int width, const int gCol, const bool isA1) {
    ASSERT(gCol >= width && "Copy ND block gm->ub width larger than origin matrix width.");
    int calcWidth = width / BLOCK_CUBE; // cube block numbers that do not need to be pad zero
    int dstOffset = 0;
    int srcOffset = 0;
    int calcHeight = (height + BLOCK_CUBE - 1) / BLOCK_CUBE;

    // gCol unaligned ,can not use dma copy repeat stride
    int src_gap = gCol * sizeof(mmOutputType) / UB_ALIGN_NZ - 1;
    for (int i = 0; i < calcWidth; i++) {
        dstOffset = i * calcHeight * CUBE_MAX_SIZE + outerLoopIndex * singleProcessSOuterSize * tilingData->promptAttentionBaseParams.headSize;
        srcOffset = i * BLOCK_CUBE;
        DataCopy(dst[dstOffset], src[srcOffset],
                 { static_cast<uint16_t>(height), 1, static_cast<uint16_t>(src_gap), 0});
    }
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
}
#endif  // PROMPT_FLASH_ATTENTION_NZ_BASE_NO_FLASH_H