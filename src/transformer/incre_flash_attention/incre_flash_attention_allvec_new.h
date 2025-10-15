/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file incre_flash_attention_allvec_new.h
 * \brief
 */
#ifndef INCRE_FLASH_ATTENTION_ALLVEC_NEW
#define INCRE_FLASH_ATTENTION_ALLVEC_NEW

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "ifa_public_define.h"

using namespace AscendC;
using AscendC::MulAddDst;

// 默认srcGm上有效数据是连续的,srcStride为0
template <typename DATA_T>
__aicore__ inline void CopyIn(LocalTensor<DATA_T> &dstLocal, GlobalTensor<DATA_T> srcGm, uint64_t offset,
                              uint32_t rowCnt, uint32_t columnCntAlign, uint32_t actualColumnCnt)
{
    uint32_t typeElementSize = ONE_BLK_SIZE / sizeof(DATA_T);
    if ((actualColumnCnt % typeElementSize) == 0) {
        DataCopyParams intriParams;
        intriParams.blockCount = rowCnt;
        intriParams.dstStride = (columnCntAlign - actualColumnCnt) / typeElementSize;
        intriParams.blockLen = actualColumnCnt / typeElementSize;
        intriParams.srcStride = 0;
        DataCopy(dstLocal, srcGm[offset], intriParams);
    } else {
        // actualColumnCnt不按32B对齐的拷贝,310P当前不支持
#if (__CCE_AICORE__ > 200)
        DataCopyExtParams intriParams;
        intriParams.blockCount = rowCnt;
        intriParams.dstStride = (columnCntAlign - actualColumnCnt) / typeElementSize;
        intriParams.blockLen = actualColumnCnt * sizeof(DATA_T);
        intriParams.srcStride = 0;

        DataCopyPadExtParams<DATA_T> padParams;
        padParams.leftPadding = 0;
        padParams.rightPadding = (columnCntAlign - actualColumnCnt) % typeElementSize;
        padParams.paddingValue = 0;
        padParams.isPad = (padParams.rightPadding != 0);

        DataCopyPad(dstLocal, srcGm[offset], intriParams, padParams);
#endif
    }
}

template <typename IFAT> class IncreFlashAttentionAttenAllVecNew {
public:
    __aicore__ inline IncreFlashAttentionAttenAllVecNew(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                const IncreFlashAttentionTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2,
                                     __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
                                     __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                     __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

    // 中间计算数据类型为float，高精度模式
    using T = float;

    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using OUT_T = typename IFAT::outputType;
    using ORIGIN_T = typename IFAT::orginalType;
    static constexpr bool PAGE_ATTENTION = IFAT::pageAttention;
    static constexpr bool FLASH_DECODE = IFAT::flashDecode;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;
    static constexpr uint8_t PER_CHANNEL_MODE = 0; // 伪量化 K V per-channel
    static constexpr uint8_t PER_TOKEN_MODE = 1; // 伪量化 K V per-token
    static constexpr uint8_t ANTIQUANT_MODE = IFAT::antiquantMode;

    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    static constexpr bool ANTIQUANT_PER_TOKEN = (ANTIQUANT && (ANTIQUANT_MODE == PER_TOKEN_MODE));
    static constexpr bool ANTIQUANT_PER_CHANNEL = (ANTIQUANT && (ANTIQUANT_MODE == PER_CHANNEL_MODE));
    using ANTIQ_PARAMS_T = typename AscendC::Conditional<ANTIQUANT_PER_TOKEN, T, Q_T>::type;
    using MM_OUT_T = typename AscendC::Conditional<(ANTIQUANT || QUANT), int32_t, T>::type;

    // define pse datetype
    using pseShiftType = typename AscendC::Conditional<AscendC::IsSameType<Q_T, int8_t>::value, half, Q_T>::type;

protected:
    const IncreFlashAttentionTilingData *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<int32_t> blockTableGm;
    // atten mask
    GlobalTensor<bool> attenMaskBoolGm;
    GlobalTensor<half> attenMaskHalfGm;

    // PSE
    GlobalTensor<pseShiftType> pseShiftGm;

    // antiquant
    GlobalTensor<ANTIQ_PARAMS_T> antiqOffsetGm;
    GlobalTensor<ANTIQ_PARAMS_T> antiqScaleGm;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    // workspace
    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> logSumExpGm;
#if (__CCE_AICORE__ == 200)
    GlobalTensor<int32_t> syncGlobal;
#endif
    // kv_left_padding
    GlobalTensor<int64_t> kvPaddingSizeGm;

    // queue
    TQue<QuePosition::VECIN, 1> inputQue1;   // 1K, inque
    TQue<QuePosition::VECIN, 1> inputQue2;   // 32K, inque
    TQue<QuePosition::VECOUT, 1> outputQue1; // 1K, outque

    // 临时tbuf
    TBuf<> tmpBuff1; // 32K
    TBuf<> tmpBuff2; // 32K

    // 常驻tbuf
    TBuf<> brcbBuff;        // 8K
    TBuf<> queryBuff;       // 1K
    TBuf<> bmm1ResBuff;     // 32K
    TBuf<> bmm2ResBuff;     // 2K
    TBuf<> antiqScaleBuff;  // 2K
    TBuf<> antiqOffsetBuff; // 2K
    TBuf<> softmaxMaxBuff;  // 32B
    TBuf<> softmaxExpBuff;  // 32B
    TBuf<> softmaxSumBuff;  // 32B

    LocalTensor<Q_T> queryUb;
    LocalTensor<T> bmm1ResUb;
    LocalTensor<T> bmm2ResUb;
    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    // antiquant msd
    LocalTensor<T> antiqScaleUb;
    LocalTensor<T> antiqOffsetUb;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr uint32_t REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(T);
    static constexpr uint32_t ADDRESS_ALIGN_NUM = 512 / sizeof(KV_T);
    static constexpr uint32_t ADDRESS_ALIGN_NUM_THRESHLOD = 128 / sizeof(KV_T);
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    bool antiqOffsetExistFlag = false;
    uint32_t antiquantPerTensorFlag = 0U;
    uint64_t sUnitSize = 0;

    // copy kv param
    uint32_t copyKeySplitS = 0;
    uint32_t copyKeyActSplitS = 0;
    uint32_t copyKeyLoopCount = 0;
    uint32_t copyKeyTailStart = 0;
    uint32_t copyKeyTailSplitSize = 0;
    uint32_t copyValueSplitS = 0;
    uint32_t copyValueActSplitS = 0;
    uint32_t copyValueLoopCount = 0;
    uint32_t copyValueTailStart = 0;
    uint32_t copyValueTailSplitSize = 0;

    // kv_left_padding
    uint32_t kvPaddingFlag = 0;
    uint64_t kvPaddingBeginOffset = 0;
    uint64_t kvPaddingSPadDataNum = 0;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;
    uint32_t tmpBlockIdx = 0U;

    // tilingdata
    uint64_t singleProcessSInnerSize = 0ULL;
    uint32_t sInnerLoopTimes = 0U;
    uint64_t singleProcessSInnerSizeTail = 0ULL;
    uint32_t blockSplitBn2Range = 0U;
    uint32_t tailBlockSplitBn2Range = 0U;
    uint32_t formerCoreNum = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;

    uint32_t batchContinuous = 0U;

    uint64_t batchSize = 0U;
    uint64_t qHeadNum = 0U;
    uint64_t kvHeadNum = 0U;
    uint64_t gSize = 1U;
    uint64_t kvSeqSize = 0U;
    uint64_t headDim = 0U;
    uint64_t headDimAlign = 0U;
    bool useDataCopyPad = false;
    bool padForBankConflict = false;

    // attention mask
    bool attenMaskFlag = false;
    uint32_t selectWithByteMaskTmpMinSize = 0U;
    uint32_t attenMaskSizeAlign = 0U;
    // pse mask
    bool pseShiftFlag = false;
    uint32_t pseShiftB = 0U;
    uint32_t pseShiftS = 0U;
    uint64_t pseShiftOffset = 0U;
    uint64_t pseShiftCoreOffset = 0ULL;
    uint32_t pseMaskSizeAlign = 0U;
    // offset
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t valueOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;
    uint64_t antiqKeyParamOffset = 0ULL;
    uint64_t antiqValueParamOffset = 0ULL;
    uint64_t attenMaskOffset = 0ULL;
    uint64_t attenMaskCoreOffset = 0ULL;
    uint64_t antiqKeyParamCoreOffsetPerToken = 0ULL;
    uint64_t antiqValueParamCoreOffsetPerToken = 0ULL;
    uint64_t attentMaskSize = 0ULL;

    // splitKV
    uint32_t splitKVNum = 0U;
    uint32_t s2Idx = 0U;
    uint64_t sInnerLoopSize = 0ULL;
    uint32_t actualCombineLoopSize = 0U;
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;

    uint64_t curActualSeqLen = 0ULL;
    uint64_t actualSingleProcessSInnerSize = 0ULL;
    uint64_t actualSingleProcessSInnerSizeAlign = 0ULL;
    uint32_t beforeBlockSplitBn2Nums = 0U;
    uint32_t bn2LoopTimes = 0U;

    const uint32_t *coreStartIdx = nullptr;
    uint32_t actualLenDims = 0U;
    // out quant
    bool isPerChnU8Out = false;
    bool isOutQuantTypeBf16 = false;
    float quantScale2Value = 0;
    float quantOffset2Value = 0;
    bool isQuantOffset2Exist = false;

    bool curActSeqLenIsZero = false;

    // pageAttention
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint64_t s2BatchOffset = 0;
    uint64_t s2BatchBaseOffset = 0;
    uint64_t blockTableBaseOffset = 0;

    // softmaxlse
    bool softmaxLseFlag = false;

    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParams();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void GetActualSeqLen();
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void CalculateSUnitSize();
    __aicore__ inline void CalculatekvPaddingSPadDataNum(int64_t &startPosition);
    __aicore__ inline bool ComputeKVPaddingBeginOffset();
    __aicore__ inline void CalculateCopyKvSplitS();
    __aicore__ inline void UpdateCopyKvParam();

    __aicore__ inline void GetBN2id(const uint32_t bn2Idx);
    __aicore__ inline void CalcBN2Offset();
    __aicore__ inline void CalcBN2Params();

    __aicore__ inline void CalcSInnerOffsetAndParams(const uint32_t sInnerLoopIdx);

    __aicore__ inline void UpdateIsUseDataCopyPad();
    __aicore__ inline void InitDataCopyExtParams(DataCopyExtParams &intriParams, DataCopyPadExtParams<KV_T> &padParams,
                                                 uint32_t rowCnt);
    __aicore__ inline void InitDataCopyParams(DataCopyParams &intriParams, uint32_t rowCnt);
    __aicore__ inline void CopyKV(LocalTensor<KV_T> dstLocal, GlobalTensor<KV_T> &srcGm, uint64_t offset,
                                  uint32_t rowCnt);

    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t dealRowCount, uint32_t actualColumnCount);

    __aicore__ inline void QueryPreProcess();
    __aicore__ inline void FlashDecodeCompute();
    __aicore__ inline void ComputeMm1ByVmla(LocalTensor<T> &mmResUb, LocalTensor<Q_T> &aUb, LocalTensor<Q_T> &bUb,
                                            uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ComputeSingleMm1(uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void Bmm1Compute();
    __aicore__ inline void ComputeMm2ByVmla(LocalTensor<T> &vmlaResUb, LocalTensor<Q_T> &aUb, LocalTensor<Q_T> &bUb,
                                            uint32_t dealRowCount, uint32_t singleDealRowCnt, uint32_t columnCount,
                                            uint32_t actualColumnCount);
    __aicore__ inline void AccumulateByRow(LocalTensor<T> &srcUb, uint32_t rowCount, uint32_t columnCount);
    __aicore__ inline void CopyValueToUb(uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void ComputeSingleMm2(LocalTensor<T> vmlaResUb, uint32_t startRow, uint32_t dealRowCount,
                                            uint32_t singleComputeRowCnt);
    __aicore__ inline void Bmm2Compute();

    __aicore__ inline void ColumnSum(LocalTensor<T> srcUb, uint32_t dealRowCount, uint32_t columnCount);
    __aicore__ inline void ProcessVec1(const uint32_t sInnerLoopIdx);
    __aicore__ inline void ProcessVec2(const uint32_t sInnerLoopIdx);
    __aicore__ inline void SInnerLoopFunc(const uint32_t sInnerLoopIdx);

    __aicore__ inline void SoftmaxFlashV2Compute(LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void PseShiftCopyIn(uint32_t startRow, uint32_t rowCount, uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void Bmm2DataCopyOut(LocalTensor<OUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                              uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void CombineSplitKVRes();
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseMaxUb, LocalTensor<T> &lseSumUb,
                                             LocalTensor<T> &lseExpUb, LocalTensor<T> &lseLocal);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(LocalTensor<T> &softmaxMaxUb, LocalTensor<T> &softmaxSumUb);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<T> &softmaxMaxUb, LocalTensor<T> &softmaxSumUb);
    __aicore__ inline void Bmm2FDDataCopyOut(LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                             uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx);
    __aicore__ inline uint64_t SeqLenFromTensorList(uint32_t bIdx);
};

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::InitTilingData()
{
    singleProcessSInnerSize = tilingData->increFlashAttentionSingleCoreParams.singleProcessSInnerSize;
    sInnerLoopTimes = tilingData->increFlashAttentionSingleCoreParams.sInnerLoopTimes;
    singleProcessSInnerSizeTail = tilingData->increFlashAttentionSingleCoreParams.singleProcessSInnerSizeTail;
    usedCoreNum = tilingData->increFlashAttentionSingleCoreParams.usedCoreNum;
    formerCoreNum = tilingData->increFlashAttentionSingleCoreParams.formerCoreNum;
    blockSplitBn2Range = tilingData->increFlashAttentionSingleCoreParams.blockSplitBn2Range;
    tailBlockSplitBn2Range = tilingData->increFlashAttentionSingleCoreParams.tailSplitedBatchRange;
    splitKVNum = tilingData->splitKVParams.s2;
    sInnerLoopSize = tilingData->splitKVParams.sInnerLoopSize;

    batchSize = tilingData->baseParams.batchSize;
    kvHeadNum = tilingData->baseParams.kvHeadNum;
    qHeadNum = tilingData->baseParams.qHeadNum;
    gSize = tilingData->baseParams.nNumOfQInOneGroup;
    kvSeqSize = tilingData->baseParams.seqSize;
    headDim = tilingData->baseParams.headSize;
    batchContinuous = tilingData->baseParams.batchContinuousFlag;
    antiquantPerTensorFlag = tilingData->baseParams.antiquantPerTensorFlag;

    headDimAlign = Align(headDim, BYTE_BLOCK);

    UpdateIsUseDataCopyPad();

    padForBankConflict = (!ANTIQUANT && headDim == 256);

    attenMaskFlag = (tilingData->baseParams.attenMaskFlag != 0) ? true : false;
    attentMaskSize = tilingData->baseParams.attenMaskSize;
    selectWithByteMaskTmpMinSize = tilingData->baseParams.selectWithByteMaskTmpMinSize;

    pseShiftFlag = (tilingData->baseParams.pseShiftFlag == 1) ? true : false;
    if (pseShiftFlag) {
        pseShiftB = tilingData->baseParams.pseShiftB;
        pseShiftS = tilingData->baseParams.pseShiftS;
    }

    kvPaddingFlag = tilingData->baseParams.kvPaddingFlag;

    // 是否输出lse
    softmaxLseFlag = tilingData->baseParams.softmaxLseFlag;

    maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    kvCacheBlockSize = tilingData->baseParams.blockSize;

    coreStartIdx = tilingData->increFlashAttentionCoreParams.coreSidxEnd;
    // out quant
    isPerChnU8Out = tilingData->outputParams.isPerChnOut == 0 ? false : true;
    isOutQuantTypeBf16 = tilingData->outputParams.isOutQuantTypeBf16 == 0 ? false : true;
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::InitBuffers()
{
    // 0-32K
    pipe->InitBuffer(bmm1ResBuff, BUFFER_SIZE_BYTE_32K);

    // 32-96K
    // queue
    if (padForBankConflict) {
        pipe->InitBuffer(inputQue2, 2, BUFFER_SIZE_BYTE_32K + BUFFER_SIZE_BYTE_2K);
    } else {
        pipe->InitBuffer(inputQue2, 2, BUFFER_SIZE_BYTE_32K);
    }

    // 96-128K
    pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_32K);

    // 128-160K
    // tmpBuff
    pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);

    // 160-192K
    pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_1K);
    pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_1K);
    // brcb专用buffer
#if (__CCE_AICORE__ > 200)
    pipe->InitBuffer(brcbBuff, BUFFER_SIZE_BYTE_8K);
#else
    pipe->InitBuffer(brcbBuff, BUFFER_SIZE_BYTE_32K);
#endif

    // 常驻buffer
    pipe->InitBuffer(queryBuff, BUFFER_SIZE_BYTE_1K);
    pipe->InitBuffer(bmm2ResBuff, BUFFER_SIZE_BYTE_2K);
    pipe->InitBuffer(antiqScaleBuff, BUFFER_SIZE_BYTE_2K);
    pipe->InitBuffer(antiqOffsetBuff, BUFFER_SIZE_BYTE_2K);
    pipe->InitBuffer(softmaxMaxBuff, BUFFER_SIZE_BYTE_32B);
    pipe->InitBuffer(softmaxSumBuff, BUFFER_SIZE_BYTE_32B);
    pipe->InitBuffer(softmaxExpBuff, BUFFER_SIZE_BYTE_32B);

    queryUb = queryBuff.Get<Q_T>();
    bmm1ResUb = bmm1ResBuff.Get<T>();
    bmm2ResUb = bmm2ResBuff.Get<T>();
    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::CalculateCopyKvSplitS()
{
    // 1、非量化场景：KV_T=Q_T
    // 2、伪量化场景：需要CAST到Q_T，queue的大小与CAST后的buff均为32K
    uint32_t splitS = BUFFER_SIZE_BYTE_32K / sizeof(Q_T) / headDimAlign;
    copyKeySplitS = (splitS / BLOCK_ELEMENT_NUM) * BLOCK_ELEMENT_NUM; // SplitS作为bmm1ResUb的偏移，需要32Byte对齐

    constexpr uint32_t brcbMaxCnt = BUFFER_SIZE_BYTE_8K / ONE_BLK_SIZE;
    if (splitS > brcbMaxCnt) {
        splitS = brcbMaxCnt;
    }
    copyValueSplitS = (splitS / BLOCK_ELEMENT_NUM) * BLOCK_ELEMENT_NUM; // SplitS作为bmm1ResUb的偏移，需要32Byte对齐
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::UpdateCopyKvParam()
{
    // copy key param
    copyKeyActSplitS = copyKeySplitS;
    if (copyKeyActSplitS > actualSingleProcessSInnerSize) {
        copyKeyActSplitS = actualSingleProcessSInnerSize;
    }
    copyKeyLoopCount = (actualSingleProcessSInnerSize + copyKeyActSplitS - 1) / copyKeyActSplitS - 1;
    copyKeyTailStart = copyKeyLoopCount * copyKeyActSplitS;
    copyKeyTailSplitSize = actualSingleProcessSInnerSize - copyKeyTailStart;

    // copy value param
    copyValueActSplitS = copyValueSplitS;
    if (copyValueActSplitS > actualSingleProcessSInnerSize) {
        copyValueActSplitS = actualSingleProcessSInnerSize;
    }
    copyValueLoopCount = (actualSingleProcessSInnerSize + copyValueActSplitS - 1) / copyValueActSplitS - 1;
    copyValueTailStart = copyValueLoopCount * copyValueActSplitS;
    copyValueTailSplitSize = actualSingleProcessSInnerSize - copyValueTailStart;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengths)
{
    actualLenDims = tilingData->baseParams.actualLenDims;
    if (actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, actualLenDims);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::InitAllZeroOutput(uint32_t bIdx)
{
    uint32_t copySize = gSize * headDim;
    matmul::InitOutput<OUT_T>(attentionOutGm[(bIdx * kvHeadNum + n2Idx) * copySize], copySize, 0);

#if (__CCE_AICORE__ > 200)
    if (softmaxLseFlag) {
        LocalTensor<T> softmaxlseOut = outputQue1.template AllocTensor<T>();
        float minf = -3.40E+38;
        Duplicate(softmaxlseOut, minf, gSize);
        outputQue1.EnQue(softmaxlseOut);
        outputQue1.DeQue();
        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float) * gSize;
        intriParams1.blockCount = 1;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(softmaxLseGm[(bIdx * kvHeadNum + n2Idx) * gSize], softmaxlseOut, intriParams1);
        outputQue1.FreeTensor(softmaxlseOut);
    }
#endif
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::GetActualSeqLen()
{
    if (actualLenDims == 0) {
        curActualSeqLen = kvSeqSize;
        if (!batchContinuous) {
            curActualSeqLen = SeqLenFromTensorList(bIdx);
        }
    } else if (actualLenDims == 1) {
        curActualSeqLen = actualSeqLengthsGm.GetValue(0);
    } else {
        curActualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::GetBN2id(const uint32_t bn2Idx)
{
    if constexpr (FLASH_DECODE) {
        bIdx = tmpBlockIdx / (kvHeadNum * splitKVNum);
        n2Idx = (tmpBlockIdx / splitKVNum) % kvHeadNum;
        s2Idx = tmpBlockIdx % splitKVNum;
    } else {
        bIdx = (beforeBlockSplitBn2Nums + bn2Idx) / kvHeadNum;
        n2Idx = (beforeBlockSplitBn2Nums + bn2Idx) % kvHeadNum;
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::UpdateInnerLoopCond()
{
    if (curActualSeqLen == 0) {
        InitAllZeroOutput(bIdx);
        curActSeqLenIsZero = true;
        return;
    } else {
        curActSeqLenIsZero = false;
    }

    int32_t remainSinnerSize = curActualSeqLen + kvPaddingSPadDataNum;
    int32_t computeSinnerSize = curActualSeqLen + kvPaddingSPadDataNum;
    if constexpr (FLASH_DECODE) {
        remainSinnerSize = (int32_t)curActualSeqLen + kvPaddingSPadDataNum - sInnerLoopSize * s2Idx;
        computeSinnerSize = remainSinnerSize >= sInnerLoopSize ? sInnerLoopSize : remainSinnerSize;
        if (tmpBlockIdx >= batchSize * kvHeadNum * splitKVNum) {
            remainSinnerSize = 0;
        }
    }
    if (remainSinnerSize > 0) {
        if (computeSinnerSize <= singleProcessSInnerSize) {
            singleProcessSInnerSizeTail = computeSinnerSize;
            sInnerLoopTimes = 1;
        } else {
            sInnerLoopTimes = (computeSinnerSize + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
            singleProcessSInnerSizeTail = computeSinnerSize - (sInnerLoopTimes - 1) * singleProcessSInnerSize;
        }
    } else {
        sInnerLoopTimes = 0;
    }
}

template <typename IFAT>
__aicore__ inline uint64_t IncreFlashAttentionAttenAllVecNew<IFAT>::SeqLenFromTensorList(uint32_t bIndex)
{
    uint64_t dimInfo[4]; // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        return keyTensorDesc.GetShape(1); // BSH, idx of s is 1
    } else {
        return keyTensorDesc.GetShape(2); // BNSD, idx of s is 2
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::CalculateSUnitSize()
{
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        sUnitSize = headDim * kvHeadNum;
    } else {
        sUnitSize = headDim;
    }
    return;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::CalculatekvPaddingSPadDataNum(int64_t &startPosition)
{
    if (!attenMaskFlag || startPosition <= 0) {
        return;
    }

    uint64_t kvPaddingBeginOffsetBit = startPosition * sUnitSize;
    // already align, early quit
    if (kvPaddingBeginOffsetBit % ADDRESS_ALIGN_NUM == 0) {
        return;
    }

    // pad bit need fill to 512B
    uint64_t kvPaddingSPadDataBit = kvPaddingBeginOffsetBit % ADDRESS_ALIGN_NUM;

    // reduce the size of padding data for GM address alignment.
    int64_t addrAlignNum = ADDRESS_ALIGN_NUM;
    // condition 1: the size of padding data less than 50% of the original data.
    // condition 2: the size of padding data can not exceeds 0 offset of this batch.
    while ((kvPaddingSPadDataBit < curActualSeqLen * sizeof(KV_T) >> 1) &&
           (kvPaddingBeginOffsetBit > kvPaddingSPadDataBit)) {
        // if the conditions are not met, do not fill in data because there is no peak rate of GM/L2->L1 data
        // transmission.
        if (addrAlignNum < ADDRESS_ALIGN_NUM_THRESHLOD) {
            kvPaddingSPadDataBit = 0;
            break;
        }
        // kvPaddingUnit RemainSize should be a factor of sUnitSize, because remainSinnerSize and SinnerLoopTimes
        // will be recalculated.
        if (kvPaddingSPadDataBit / sizeof(KV_T) % sUnitSize == 0) {
            break;
        }
        addrAlignNum /= 2;
        kvPaddingSPadDataBit = kvPaddingBeginOffsetBit % addrAlignNum;
    }

    kvPaddingSPadDataNum = kvPaddingSPadDataBit / sizeof(KV_T) / sUnitSize;
    return;
}

template <typename IFAT> __aicore__ inline bool IncreFlashAttentionAttenAllVecNew<IFAT>::ComputeKVPaddingBeginOffset()
{
    if (kvPaddingFlag != 1) {
        return true;
    }
    int64_t paddingSize = kvPaddingSizeGm.GetValue(0);
    if (paddingSize < 0) {
        paddingSize = 0;
    }

    int64_t startPosition = kvSeqSize - paddingSize - curActualSeqLen;

    // Calculate GM address align dataNum
    CalculatekvPaddingSPadDataNum(startPosition);

    if (startPosition < 0) {
        InitAllZeroOutput(bIdx);
        return false;
    }

    kvPaddingBeginOffset = static_cast<uint64_t>(startPosition) - kvPaddingSPadDataNum;
    return true;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const IncreFlashAttentionTilingData *__restrict tiling, TPipe *tPipe)
{
    tmpBlockIdx = GetBlockIdx();

    pipe = tPipe;

    // init tiling data
    tilingData = tiling;
    InitTilingData();

    keyPtr = key;
    valuePtr = value;

    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    ListTensorDesc valueListTensorDesc((__gm__ void *)valuePtr);
    __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
    __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);

    // init global buffer
    keyGm.SetGlobalBuffer((__gm__ KV_T *)key_);
    valueGm.SetGlobalBuffer((__gm__ KV_T *)value_);
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    if (softmaxLseFlag) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    }
    if (tilingData->baseParams.l2CacheOffFlag) {
        // 关闭K、V的L2 Cache
#ifndef ASCENDC_OOM
        keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        valueGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
    }

    // GM for pse
    if (pseShiftFlag) {
        pseShiftGm.SetGlobalBuffer((__gm__ pseShiftType *)pseShift);
    }

    if (attenMaskFlag) {
        attenMaskBoolGm.SetGlobalBuffer((__gm__ bool *)attenMask);
    }

    InitActualSeqLen(actualSeqLengths);

    if (kvPaddingFlag == 1) {
        kvPaddingSizeGm.SetGlobalBuffer((__gm__ int64_t *)kvPaddingSize);
    }

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }

    InitBuffers();

    uint64_t offset = 0;
    if constexpr (FLASH_DECODE) {
        accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset = offset + tilingData->splitKVParams.accumOutSize * sizeof(float);
        logSumExpGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset = offset + tilingData->splitKVParams.logSumExpSize * sizeof(float);
#if (__CCE_AICORE__ == 200)
        syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace + offset),
                                   GetBlockNum() * BYTE_BLOCK / sizeof(int32_t));
        offset = offset + GetBlockNum() * BYTE_BLOCK;
#endif
    }

    CalculateCopyKvSplitS();
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenAllVecNew<IFAT>::InitQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1,
                                                   __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
                                                   __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale,
                                                   __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *workspace)
{
    if constexpr (ANTIQUANT) {
        antiqScaleGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)antiquantScale);
        antiqOffsetExistFlag = (antiquantOffset != nullptr);
        if (antiqOffsetExistFlag) {
            antiqOffsetGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)antiquantOffset);
        }

        if constexpr (ANTIQUANT_PER_CHANNEL) {
            antiqScaleUb = antiqScaleBuff.Get<T>();
            antiqOffsetUb = antiqOffsetBuff.Get<T>();
        }
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::InitCalcParams()
{
    bn2LoopTimes = blockSplitBn2Range;
    beforeBlockSplitBn2Nums = tmpBlockIdx * blockSplitBn2Range;
    // tail core
    if (tmpBlockIdx >= formerCoreNum) {
        bn2LoopTimes = tailBlockSplitBn2Range;
        beforeBlockSplitBn2Nums =
            formerCoreNum * blockSplitBn2Range + (tmpBlockIdx - formerCoreNum) * tailBlockSplitBn2Range;
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::InitCalcParamsEach()
{
    bn2LoopTimes = coreStartIdx[tmpBlockIdx + 1] - coreStartIdx[tmpBlockIdx];
    beforeBlockSplitBn2Nums = coreStartIdx[tmpBlockIdx];
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::CalcBN2Offset()
{
    if constexpr (LAYOUT_T == LAYOUT::BSND || LAYOUT_T == LAYOUT::BSH) {
        // B,1,N2,G,D
        tensorACoreOffset = bIdx * qHeadNum * headDim + n2Idx * gSize * headDim;
        // B,S2,N2,D
        tensorBCoreOffset =
            bIdx * kvSeqSize * kvHeadNum * headDim + n2Idx * headDim + kvPaddingBeginOffset * kvHeadNum * headDim;

        if (!batchContinuous) {
            tensorBCoreOffset = n2Idx * headDim;
        }

        if constexpr (FLASH_DECODE) {
            tensorBCoreOffset += s2Idx * sInnerLoopSize * headDim * kvHeadNum;
        }
    } else {
        // B,N2,G,1,D
        tensorACoreOffset = bIdx * qHeadNum * headDim + n2Idx * gSize * headDim;
        // B,N2,S2,D
        tensorBCoreOffset =
            bIdx * kvHeadNum * kvSeqSize * headDim + n2Idx * kvSeqSize * headDim + kvPaddingBeginOffset * headDim;

        if (!batchContinuous) {
            uint64_t seqSize = SeqLenFromTensorList(bIdx);
            tensorBCoreOffset = n2Idx * seqSize * headDim;
        }

        if constexpr (FLASH_DECODE) {
            tensorBCoreOffset += s2Idx * sInnerLoopSize * headDim;
        }
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::CalcBN2Params()
{
    blockTableBaseOffset = bIdx * maxBlockNumPerBatch;
    s2BatchBaseOffset = kvPaddingBeginOffset;
    attenMaskCoreOffset = bIdx * attentMaskSize + kvPaddingBeginOffset;
    if constexpr (FLASH_DECODE) {
        attenMaskCoreOffset += s2Idx * sInnerLoopSize;
        s2BatchBaseOffset += s2Idx * sInnerLoopSize;
    }
    // antiquant的offset和scale参数数据排列是先key后value
    if (antiquantPerTensorFlag == 1) {
        antiqKeyParamOffset = 0;
        antiqValueParamOffset = 1;
    } else {
        antiqKeyParamOffset = n2Idx * headDim;
        antiqValueParamOffset = kvHeadNum * headDim + n2Idx * headDim;
    }
    antiqKeyParamCoreOffsetPerToken = bIdx * kvSeqSize + kvPaddingBeginOffset;
    antiqValueParamCoreOffsetPerToken = batchSize * kvSeqSize + bIdx * kvSeqSize + kvPaddingBeginOffset;
    if constexpr (FLASH_DECODE) {
        antiqKeyParamCoreOffsetPerToken += s2Idx * sInnerLoopSize;
        antiqValueParamCoreOffsetPerToken += s2Idx * sInnerLoopSize;
    }
    if (!batchContinuous) {
        ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
        ListTensorDesc valueListTensorDesc((__gm__ void *)valuePtr);
        __gm__ uint8_t *key = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        __gm__ uint8_t *value = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        keyGm.SetGlobalBuffer((__gm__ KV_T *)key);
        valueGm.SetGlobalBuffer((__gm__ KV_T *)value);
        if (tilingData->baseParams.l2CacheOffFlag) {
            // 关闭K、V的L2 Cache
#ifndef ASCENDC_OOM
            valueGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
        }
    }
    // 更新actualSingleProcessSInnerSize，防止尾块值，影响第二次loop
    actualSingleProcessSInnerSize = singleProcessSInnerSize;
    actualSingleProcessSInnerSizeAlign = Align(singleProcessSInnerSize, BYTE_BLOCK);
    // key和value的Copy参数随actualSingleProcessSInnerSize更新
    UpdateCopyKvParam();

    if (pseShiftFlag) {
        pseShiftCoreOffset = (pseShiftB == 1) ? (n2Idx * pseShiftS * gSize) :
            (bIdx * qHeadNum * pseShiftS + n2Idx * gSize * pseShiftS);

        if constexpr (FLASH_DECODE) {
            pseShiftCoreOffset += s2Idx * sInnerLoopSize;
        }
        pseShiftCoreOffset += kvPaddingBeginOffset; // kv_padding_size
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::CalcSInnerOffsetAndParams(const uint32_t sInnerLoopIdx)
{
    uint64_t sInnerOffsetDataSize = sInnerLoopIdx * singleProcessSInnerSize;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        // B,Si,N2,D
        tensorBOffset = tensorBCoreOffset + sInnerOffsetDataSize * kvHeadNum * headDim;
    } else {
        tensorBOffset = tensorBCoreOffset + sInnerOffsetDataSize * headDim;
    }
    valueOffset = tensorBOffset;
    attenOutOffset = tensorACoreOffset;

    attenMaskOffset = attenMaskCoreOffset + sInnerOffsetDataSize;
    s2BatchOffset = s2BatchBaseOffset + sInnerOffsetDataSize;

    // Calc Params
    if (sInnerLoopIdx == sInnerLoopTimes - 1) {
        actualSingleProcessSInnerSize = singleProcessSInnerSizeTail;
        actualSingleProcessSInnerSizeAlign = Align(singleProcessSInnerSizeTail, BYTE_BLOCK);
        UpdateCopyKvParam();
    }

    // pse offset
    if (pseShiftFlag) {
        pseShiftOffset = pseShiftCoreOffset + sInnerOffsetDataSize;
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::UpdateIsUseDataCopyPad()
{
    uint32_t blockSize = ONE_BLK_SIZE / sizeof(KV_T);
    bool isHeadDimAlign = (headDim % blockSize == 0);
    useDataCopyPad = (!isHeadDimAlign);
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        if (isHeadDimAlign) {
            uint32_t blockLen = headDim / blockSize;
            if (blockLen * (kvHeadNum - 1) > MAX_UINT16) {
                useDataCopyPad = true;
            }
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::InitDataCopyExtParams(DataCopyExtParams &intriParams,
                                                                                      DataCopyPadExtParams<KV_T> &padParams,
                                                                                      uint32_t rowCnt)
{
    uint32_t typeElementSize = ONE_BLK_SIZE / sizeof(KV_T);
    intriParams.blockCount = rowCnt; // 需要copy的数据块个数，repeat次数
    intriParams.dstStride = (headDimAlign - headDim) / typeElementSize;
    if (padForBankConflict) {
        intriParams.dstStride = 1;
    }
    intriParams.blockLen = headDim * sizeof(KV_T); // 每次连续传输的数据块含有的Byte数量
    // 源操作数相邻数据块尾与头的间隔Byte数, pageAttention场景,均按照BSH处理
    intriParams.srcStride = intriParams.blockLen * (kvHeadNum - 1);
    if constexpr (!PAGE_ATTENTION && LAYOUT_T == LAYOUT::BNSD) {
        intriParams.srcStride = 0;
    }

    padParams.leftPadding = 0;
    padParams.rightPadding = (headDimAlign - headDim) % typeElementSize;
    padParams.paddingValue = 0;
    padParams.isPad = (padParams.rightPadding != 0);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::InitDataCopyParams(DataCopyParams &intriParams,
                                                                                   uint32_t rowCnt)
{
    uint32_t typeElementSize = ONE_BLK_SIZE / sizeof(KV_T);
    intriParams.blockCount = rowCnt; // 需要copy的数据块个数，repeat次数
    intriParams.dstStride = (headDimAlign - headDim) / typeElementSize;
    if (padForBankConflict) {
        intriParams.dstStride = 1;
    }
    intriParams.blockLen = headDim / typeElementSize; // 每次连续传输的数据块含有的Byte数量
    // 源操作数相邻数据块尾与头的间隔Byte数, pageAttention场景,均按照BSH处理
    intriParams.srcStride = intriParams.blockLen * (kvHeadNum - 1);
    if constexpr (!PAGE_ATTENTION && LAYOUT_T == LAYOUT::BNSD) {
        intriParams.srcStride = 0;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::CopyKV(LocalTensor<KV_T> dstLocal,
                                                                       GlobalTensor<KV_T> &srcGm, uint64_t offset,
                                                                       uint32_t rowCnt)
{
    if (useDataCopyPad) {
#if (__CCE_AICORE__ > 200)
        DataCopyExtParams intriParams;
        DataCopyPadExtParams<KV_T> padParams;
        InitDataCopyExtParams(intriParams, padParams, rowCnt);
        DataCopyPad(dstLocal, srcGm[offset], intriParams, padParams);
#endif
    } else {
        DataCopyParams intriParams;
        InitDataCopyParams(intriParams, rowCnt);
        DataCopy(dstLocal, srcGm[offset], intriParams);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::AttenMaskCopyIn(uint64_t offset, uint32_t dealRowCount,
                                                                                uint32_t actualColumnCount)
{
    LocalTensor<bool> maskUb = inputQue2.AllocTensor<bool>();
    attenMaskSizeAlign = Align(actualColumnCount, 32U);
    maskUb.SetSize(dealRowCount * attenMaskSizeAlign);
#if (__CCE_AICORE__ > 200)
    if (actualColumnCount % 32 == 0) {
        DataCopy(maskUb, attenMaskBoolGm[offset], attenMaskSizeAlign);
    } else {
        uint32_t typeElementSize = BYTE_BLOCK / sizeof(bool);
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = actualColumnCount * sizeof(bool);
        intriParams.dstStride = (attenMaskSizeAlign - actualColumnCount) / typeElementSize;
        intriParams.srcStride = 0;
        DataCopyPadExtParams<bool> padParams;
        padParams.leftPadding = 0;
        padParams.rightPadding = (attenMaskSizeAlign - actualColumnCount) % typeElementSize;
        padParams.paddingValue = 0;
        padParams.isPad = true;
        DataCopyPad(maskUb, attenMaskBoolGm[offset], intriParams, padParams);
    }
#else
    DataCopy(maskUb, attenMaskBoolGm[offset], attenMaskSizeAlign);
#endif
    inputQue2.template EnQue(maskUb);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::QueryPreProcess()
{
    LocalTensor<Q_T> inputUb = inputQue1.template AllocTensor<Q_T>();
    CopyIn<Q_T>(inputUb, queryGm, tensorACoreOffset, 1, headDimAlign, headDim);
    inputQue1.template EnQue(inputUb);
    inputQue1.DeQue<Q_T>();
    DataCopy(queryUb, inputUb, headDimAlign);
    inputQue1.FreeTensor(inputUb);

    if constexpr (ANTIQUANT) {
        pipe_barrier(PIPE_V);
        LocalTensor<Q_T> inputUb = inputQue1.template AllocTensor<Q_T>();
        CopyIn<Q_T>(inputUb, antiqScaleGm, antiqKeyParamOffset, 1, headDimAlign, headDim);
        inputQue1.template EnQue(inputUb);
        inputQue1.DeQue<Q_T>();
        Mul(queryUb, queryUb, inputUb, headDimAlign);
        inputQue1.FreeTensor(inputUb);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenAllVecNew<IFAT>::ComputeScaleValue(LocalTensor<T> &lseMaxUb, LocalTensor<T> &lseSumUb,
                                                           LocalTensor<T> &lseExpUb, LocalTensor<T> &lseLocal)
{
    Duplicate(lseMaxUb, -FLOAT_MAX, gSize * FP32_ONE_BLOCK_SIZE);
    Duplicate(lseSumUb, FLOAT_ZERO, gSize * FP32_ONE_BLOCK_SIZE);
    pipe_barrier(PIPE_V);
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Max(lseMaxUb, lseMaxUb, lseLocal[i * gSize * FP32_ONE_BLOCK_SIZE], gSize * FP32_ONE_BLOCK_SIZE);
        pipe_barrier(PIPE_V);
    }
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Sub(lseExpUb[i * gSize * FP32_ONE_BLOCK_SIZE], lseLocal[i * gSize * FP32_ONE_BLOCK_SIZE], lseMaxUb,
            gSize * FP32_ONE_BLOCK_SIZE);
        pipe_barrier(PIPE_V);
    }
    Exp(lseExpUb, lseExpUb, actualCombineLoopSize * gSize * FP32_ONE_BLOCK_SIZE);
    pipe_barrier(PIPE_V);
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Add(lseSumUb, lseSumUb, lseExpUb[i * gSize * FP32_ONE_BLOCK_SIZE], gSize * FP32_ONE_BLOCK_SIZE);
        pipe_barrier(PIPE_V);
    }
    Log(lseSumUb, lseSumUb, gSize * FP32_ONE_BLOCK_SIZE);
    pipe_barrier(PIPE_V);
    Add(lseSumUb, lseSumUb, lseMaxUb, gSize * FP32_ONE_BLOCK_SIZE);
    pipe_barrier(PIPE_V);
    if (softmaxLseFlag) {
#if (__CCE_AICORE__ > 200)
        LocalTensor<float> softmaxlseUb = outputQue1.template AllocTensor<float>();
        DataCopy(softmaxlseUb, lseSumUb, gSize * FP32_ONE_BLOCK_SIZE);
        outputQue1.EnQue(softmaxlseUb);
        outputQue1.DeQue<float>();

        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float);
        intriParams1.blockCount = gSize;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(softmaxLseGm[bIdx * kvHeadNum * gSize + n2Idx * gSize], softmaxlseUb, intriParams1);
        outputQue1.FreeTensor(softmaxlseUb);
#endif
    }
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Sub(lseExpUb[i * gSize * FP32_ONE_BLOCK_SIZE], lseLocal[i * gSize * FP32_ONE_BLOCK_SIZE], lseSumUb,
            gSize * FP32_ONE_BLOCK_SIZE);
        pipe_barrier(PIPE_V);
    }
    Exp(lseExpUb, lseExpUb, actualCombineLoopSize * gSize * FP32_ONE_BLOCK_SIZE);
    pipe_barrier(PIPE_V);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::CombineSplitKVRes()
{
    if (curActualSeqLen == 0) {
        // 待补充，置0操作
    } else {
        pipe_barrier(PIPE_V);
        LocalTensor<T> lseMaxUb = softmaxMaxUb;
        LocalTensor<T> lseSumUb = softmaxSumUb;
        LocalTensor<T> lseExpUb = tmpBuff1.Get<T>();
        LocalTensor<T> lseLocal = inputQue2.template AllocTensor<T>();
        DataCopy(lseLocal, logSumExpGm[combineLseOffset], actualCombineLoopSize * BLOCK_ELEMENT_NUM);
        inputQue2.template EnQue(lseLocal);
        inputQue2.DeQue<T>();
        ComputeScaleValue(lseMaxUb, lseSumUb, lseExpUb, lseLocal);
        inputQue2.FreeTensor(lseLocal);

        Duplicate(bmm2ResUb, FLOAT_ZERO, headDimAlign);
        pipe_barrier(PIPE_V);

        uint32_t singleCombineCnt = BUFFER_SIZE_BYTE_32K / sizeof(T) / headDimAlign;
        if (singleCombineCnt > actualCombineLoopSize) {
            singleCombineCnt = actualCombineLoopSize;
        }
        for (uint32_t i = 0, actDealCnt = singleCombineCnt; i < actualCombineLoopSize; i += singleCombineCnt) {
            if (i + singleCombineCnt > actualCombineLoopSize) {
                actDealCnt = actualCombineLoopSize - i;
            }
            LocalTensor<T> accumOutLocal = inputQue2.template AllocTensor<T>();
            CopyIn<T>(accumOutLocal, accumOutGm, combineAccumOutOffset + i * headDim, actDealCnt, headDimAlign,
                      headDim);
            inputQue2.template EnQue(accumOutLocal);
            inputQue2.DeQue<T>();
            RowMuls(accumOutLocal, accumOutLocal, lseExpUb[i * FP32_ONE_BLOCK_SIZE], actDealCnt, headDimAlign, headDim);
            pipe_barrier(PIPE_V);
            ColumnSum(accumOutLocal, actDealCnt, headDimAlign);
            Add(bmm2ResUb, bmm2ResUb, accumOutLocal, headDim);
            inputQue2.FreeTensor(accumOutLocal);
        }
        pipe_barrier(PIPE_V);

        LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
#if (__CCE_AICORE__ > 200)
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, headDim);
#else
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_NONE, headDim);
#endif
        outputQue1.EnQue(tmpBmm2ResCastTensor);
        outputQue1.DeQue<OUT_T>();
        Bmm2DataCopyOut(tmpBmm2ResCastTensor, 0, 1, headDimAlign, headDim);
        outputQue1.FreeTensor(tmpBmm2ResCastTensor);
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::FlashDecodeCompute()
{
    bIdx = tmpBlockIdx / kvHeadNum;
    n2Idx = tmpBlockIdx % kvHeadNum;
    attenOutOffset = bIdx * kvHeadNum * gSize * headDim + n2Idx * gSize * headDim;
    if (tmpBlockIdx >= batchSize * kvHeadNum) {
        return;
    }

    if (actualLenDims == 0) {
        curActualSeqLen = kvSeqSize;
        if (!batchContinuous) {
            curActualSeqLen = SeqLenFromTensorList(bIdx);
        }
    } else if (actualLenDims == 1) {
        curActualSeqLen = actualSeqLengthsGm.GetValue(0);
    } else {
        curActualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    }

    actualCombineLoopSize = (curActualSeqLen + sInnerLoopSize - 1) / sInnerLoopSize;
    combineLseOffset = (bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum) * gSize * FP32_ONE_BLOCK_SIZE;
    combineAccumOutOffset = (bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum) * gSize * headDim;
    CombineSplitKVRes();
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::SoftmaxLseCopyOut(LocalTensor<T> &softmaxSumUb,
                                                                                  LocalTensor<T> &softmaxMaxUb)
{
#if (__CCE_AICORE__ > 200)
    // 非GQA走全V,gSize为1
    LocalTensor<T> lseUb = outputQue1.AllocTensor<T>();
    Log(lseUb, softmaxSumUb, gSize * BLOCK_ELEMENT_NUM);
    pipe_barrier(PIPE_V);
    Add(lseUb, lseUb, softmaxMaxUb, gSize * BLOCK_ELEMENT_NUM);
    outputQue1.EnQue(lseUb);
    outputQue1.DeQue<T>();
    DataCopyExtParams intriParams1;
    intriParams1.blockLen = sizeof(float);
    intriParams1.blockCount = gSize;
    intriParams1.srcStride = 0;
    intriParams1.dstStride = 0;
    DataCopyPad(softmaxLseGm[bIdx * kvHeadNum * gSize + n2Idx * gSize], lseUb, intriParams1);
    outputQue1.FreeTensor(lseUb);
#endif
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenAllVecNew<IFAT>::ComputeLogSumExpAndCopyToGm(LocalTensor<T> &softmaxSumUb,
                                                                     LocalTensor<T> &softmaxMaxUb)
{
    // 非GQA走全V,gSize为1
    LocalTensor<T> lseUb = outputQue1.AllocTensor<T>();
    Log(lseUb, softmaxSumUb, gSize * BLOCK_ELEMENT_NUM);
    pipe_barrier(PIPE_V);
    Add(lseUb, lseUb, softmaxMaxUb, gSize * BLOCK_ELEMENT_NUM);
    outputQue1.EnQue(lseUb);
    outputQue1.DeQue<T>();
    DataCopy(logSumExpGm[bIdx * kvHeadNum * splitKVNum * gSize * BLOCK_ELEMENT_NUM +
                         n2Idx * splitKVNum * gSize * BLOCK_ELEMENT_NUM + s2Idx * gSize * BLOCK_ELEMENT_NUM],
             lseUb, gSize * BLOCK_ELEMENT_NUM);
    outputQue1.FreeTensor(lseUb);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenAllVecNew<IFAT>::ComputeMm1ByVmla(LocalTensor<T> &mmResUb, LocalTensor<Q_T> &aUb,
                                                          LocalTensor<Q_T> &bUb, uint32_t dealRowCount,
                                                          uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> vmlaResUb = tmpBuff1.Get<T>();
    uint32_t elementSize = vmlaResUb.GetSize();
    uint32_t maxDealRowCount = elementSize / (ONE_BLK_SIZE / sizeof(Q_T)); // 8192/16

    uint32_t singleDealRowCnt = maxDealRowCount; // 512
    if (maxDealRowCount > dealRowCount) {
        singleDealRowCnt = dealRowCount;
    }

    uint32_t rowLoopCnt = (dealRowCount + singleDealRowCnt - 1) / singleDealRowCnt;
    uint32_t rowTail = dealRowCount - (rowLoopCnt - 1) * singleDealRowCnt;
    uint32_t columnLoopCnt = headDim / FP16_ONE_BLOCK_SIZE;
    uint32_t remainColumnCnt = headDim % FP16_ONE_BLOCK_SIZE;
    uint32_t rowElementCnt = headDimAlign;
    if (padForBankConflict) {
        rowElementCnt += FP16_ONE_BLOCK_SIZE;
    }
    for (uint32_t i = 0, curDealRowCnt = singleDealRowCnt; i < rowLoopCnt; i++) {
        uint32_t rowStart = i * singleDealRowCnt;
        if (i + 1 == rowLoopCnt) {
            curDealRowCnt = rowTail;
        }
        BinaryRepeatParams repeatParams;
        uint32_t repeat_times = Align(curDealRowCnt, (uint32_t)VMLA_ONE_REPEATE_ROW_COUNT) / VMLA_ONE_REPEATE_ROW_COUNT;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 0;
        repeatParams.src1BlkStride = rowElementCnt / FP16_ONE_BLOCK_SIZE;
        repeatParams.dstRepStride = 2 * VMLA_ONE_REPEATE_ROW_COUNT;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = VMLA_ONE_REPEATE_ROW_COUNT * rowElementCnt / FP16_ONE_BLOCK_SIZE;

        Duplicate(vmlaResUb, FLOAT_ZERO, repeat_times * VMLA_ONE_REPEATE_ROW_COUNT * 16);
        pipe_barrier(PIPE_V);
        for (uint32_t j = 0; j < columnLoopCnt; j++) {
            MulAddDst(vmlaResUb, aUb[j * FP16_ONE_BLOCK_SIZE], bUb[rowStart * rowElementCnt + j * FP16_ONE_BLOCK_SIZE],
                      64, repeat_times, repeatParams);
            pipe_barrier(PIPE_V);
        }

        repeat_times = IFA_MAX_REPEAT_TIMES - 1;
        for (uint32_t j = 0; j < curDealRowCnt; j += repeat_times) {
            if (j + repeat_times > curDealRowCnt) {
                repeat_times = curDealRowCnt - j;
            }
            if (remainColumnCnt != 0) {
                repeatParams.dstBlkStride = 1;
                repeatParams.src0BlkStride = 0;
                repeatParams.src1BlkStride = 1;
                repeatParams.dstRepStride = 2;
                repeatParams.src0RepStride = 0;
                repeatParams.src1RepStride = rowElementCnt / FP16_ONE_BLOCK_SIZE;
                MulAddDst(vmlaResUb[j * FP16_ONE_BLOCK_SIZE], aUb[columnLoopCnt * VMLA_ONE_REPEATE_COLUMN_COUNT],
                          bUb[(rowStart + j) * rowElementCnt + columnLoopCnt * VMLA_ONE_REPEATE_COLUMN_COUNT],
                          remainColumnCnt, repeat_times, repeatParams);
                pipe_barrier(PIPE_V);
            }

            BinaryRepeatParams addRepeatParamsForBmm1;
            addRepeatParamsForBmm1.src0BlkStride = 1;
            addRepeatParamsForBmm1.src1BlkStride = 1;
            addRepeatParamsForBmm1.dstBlkStride = 1;
            addRepeatParamsForBmm1.src0RepStride = 2;
            addRepeatParamsForBmm1.src1RepStride = 2;
            addRepeatParamsForBmm1.dstRepStride = 2;
            Add(vmlaResUb[j * FP16_ONE_BLOCK_SIZE], vmlaResUb[j * FP16_ONE_BLOCK_SIZE],
                vmlaResUb[j * FP16_ONE_BLOCK_SIZE + 8], 8, repeat_times, addRepeatParamsForBmm1);
        }
        pipe_barrier(PIPE_V);
        // 要保证rowStart偏移是32Byte对齐，即singleDealRowCnt是8个元素对齐
        BlockReduceSum(mmResUb[rowStart], vmlaResUb[0], Align(curDealRowCnt, (uint32_t)8) / 8, 64, 1, 2, 16);
        pipe_barrier(PIPE_V);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::ComputeSingleMm1(uint32_t startRow,
                                                                                 uint32_t dealRowCount)
{
    uint64_t step;
    if constexpr (!PAGE_ATTENTION) {
        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            step = headDim;
        } else {
            step = kvHeadNum * headDim;
        }
    } else {
        step = kvHeadNum * headDim;
    }
    LocalTensor<KV_T> keyUb = inputQue2.template AllocTensor<KV_T>();
    if constexpr (PAGE_ATTENTION) {
        uint32_t curSeqIdx = s2BatchOffset + startRow;
        uint32_t copyFinishRowCnt = 0;
        while (copyFinishRowCnt < dealRowCount) {
            uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize; // 获取blcok table上的索引
            uint64_t reaminRowCnt = curSeqIdx % kvCacheBlockSize;  // 获取在单个块上超出的行数
            uint64_t idInBlockTable =
                blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
            // 计算可以拷贝行数
            uint32_t copyRowCnt = kvCacheBlockSize - reaminRowCnt;
            if (copyFinishRowCnt + copyRowCnt > dealRowCount) {
                copyRowCnt = dealRowCount - copyFinishRowCnt;
            }
            uint64_t keyOffset =
                (idInBlockTable * kvCacheBlockSize + reaminRowCnt) * step + (uint64_t)(n2Idx * headDim);
            CopyKV(keyUb[copyFinishRowCnt * headDimAlign], keyGm, keyOffset, copyRowCnt);

            // 更新循环变量
            copyFinishRowCnt += copyRowCnt;
            curSeqIdx += copyRowCnt;
        }
    } else {
        uint64_t keyOffset = tensorBOffset + (uint64_t)startRow * step;
        CopyKV(keyUb, keyGm, keyOffset, dealRowCount);
    }
    inputQue2.template EnQue(keyUb);
    inputQue2.DeQue<KV_T>();
    LocalTensor<T> mmResUb = bmm1ResUb[startRow];
    if constexpr (ANTIQUANT) {
        LocalTensor<Q_T> keyCastUb = tmpBuff2.Get<Q_T>();
        // dealRowCount是根据Q_T计算的，可以保证UB足够
        Cast(keyCastUb, keyUb, AscendC::RoundMode::CAST_NONE, dealRowCount * headDimAlign);
        inputQue2.FreeTensor(keyUb);
        ComputeMm1ByVmla(mmResUb, queryUb, keyCastUb, dealRowCount, headDimAlign, headDim);
    } else {
        ComputeMm1ByVmla(mmResUb, queryUb, keyUb, dealRowCount, headDimAlign, headDim);
        inputQue2.FreeTensor(keyUb);
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::Bmm1Compute()
{
    for (uint32_t i = 0; i < copyKeyLoopCount; i++) {
        ComputeSingleMm1(i * copyKeyActSplitS, copyKeyActSplitS);
    }
    ComputeSingleMm1(copyKeyTailStart, copyKeyTailSplitSize);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::ComputeMm2ByVmla(
    LocalTensor<T> &vmlaResUb, LocalTensor<Q_T> &aUb, LocalTensor<Q_T> &bUb, uint32_t dealRowCount,
    uint32_t singleDealRowCnt, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t repeateCount = singleDealRowCnt;
    uint32_t rowElementCnt = columnCount;
    if (padForBankConflict) {
        rowElementCnt += FP16_ONE_BLOCK_SIZE;
    }
    for (uint32_t i = 0; i < dealRowCount; i += singleDealRowCnt) {
        if (i + singleDealRowCnt > dealRowCount) {
            repeateCount = dealRowCount - i;
        }
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 0;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 1;
        repeatParams.src1RepStride = rowElementCnt / FP16_ONE_BLOCK_SIZE;

        pipe_barrier(PIPE_V);
        uint32_t mask = REPEAT_ELEMENT_NUM;
        for (uint32_t j = 0; j < columnCount; j += REPEAT_ELEMENT_NUM) {
            if (j + REPEAT_ELEMENT_NUM > actualColumnCount) {
                mask = actualColumnCount - j;
            }
            MulAddDst(vmlaResUb[j], aUb[i * FP16_ONE_BLOCK_SIZE], bUb[i * rowElementCnt + j], mask, repeateCount,
                      repeatParams);
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::AccumulateByRow(LocalTensor<T> &srcUb,
                                                                                uint32_t rowCount, uint32_t columnCount)
{
    for (uint32_t i = rowCount; i > 1;) {
        i >>= 1;
        pipe_barrier(PIPE_V);
        Add(srcUb, srcUb, srcUb[i * columnCount], i * columnCount);
    }
    pipe_barrier(PIPE_V);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::CopyValueToUb(uint32_t startRow, uint32_t dealRowCount)
{
    uint64_t step;
    if constexpr (!PAGE_ATTENTION) {
        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            step = headDim;
        } else {
            step = kvHeadNum * headDim;
        }
    } else {
        step = kvHeadNum * headDim;
    }

    LocalTensor<KV_T> valueUb = inputQue2.template AllocTensor<KV_T>();
    if constexpr (PAGE_ATTENTION) {
        uint32_t curSeqIdx = s2BatchOffset + startRow;
        uint32_t copyFinishRowCnt = 0;
        while (copyFinishRowCnt < dealRowCount) {
            uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize; // 获取blcok table上的索引
            uint64_t reaminRowCnt = curSeqIdx % kvCacheBlockSize;  // 获取在单个块上超出的行数
            uint64_t idInBlockTable =
                blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
            // 计算可以拷贝行数
            uint32_t copyRowCnt = kvCacheBlockSize - reaminRowCnt;
            if (copyFinishRowCnt + copyRowCnt > dealRowCount) {
                copyRowCnt = dealRowCount - copyFinishRowCnt;
            }
            uint64_t curOffset =
                (idInBlockTable * kvCacheBlockSize + reaminRowCnt) * step + (uint64_t)(n2Idx * headDim);
            CopyKV(valueUb[copyFinishRowCnt * headDimAlign], valueGm, curOffset, copyRowCnt);

            // 更新循环变量
            copyFinishRowCnt += copyRowCnt;
            curSeqIdx += copyRowCnt;
        }
    } else {
        uint64_t curOffset = valueOffset + (uint64_t)startRow * step;
        CopyKV(valueUb, valueGm, curOffset, dealRowCount);
    }
    inputQue2.template EnQue(valueUb);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenAllVecNew<IFAT>::ComputeSingleMm2(LocalTensor<T> vmlaResUb, uint32_t startRow,
                                                          uint32_t dealRowCount, uint32_t singleComputeRowCnt)
{
    LocalTensor<T> mmResUb = bmm1ResUb[startRow];
    LocalTensor<Q_T> tmpMmResUb = mmResUb.template ReinterpretCast<Q_T>();
    tmpMmResUb.SetSize(mmResUb.GetSize());
    Cast(tmpMmResUb, mmResUb, AscendC::RoundMode::CAST_NONE, dealRowCount);
    pipe_barrier(PIPE_V);

#if (__CCE_AICORE__ > 200)
    LocalTensor<Q_T> brcbResUb = brcbBuff.Get<Q_T>();
    Brcb(brcbResUb, tmpMmResUb, (dealRowCount + 7) / 8, {1, 8});
#else
    uint32_t queryElementSize = ONE_BLK_SIZE / sizeof(Q_T);
    uint32_t queryElementSizeAlign = Align(queryElementSize, 16U);
    LocalTensor<Q_T> brcbResUb = brcbBuff.Get<Q_T>();
    LocalTensor<Q_T> addsResUb = brcbResUb[BUFFER_SIZE_BYTE_32K / sizeof(Q_T) / 2];
    uint32_t lenAlign = Align(dealRowCount, 16U); // 长和宽的元素数需要是16的倍数
    for (int i = 0; i < queryElementSize; i++) {
        DataCopy(addsResUb[i * lenAlign], tmpMmResUb, lenAlign);
    }
    pipe_barrier(PIPE_V);
    ConfusionTransposeTiling transposeTiling;
    int64_t numR = lenAlign;
    int64_t numC = queryElementSizeAlign; // 长和宽的元素数需要是16的倍数
    GetConfusionTransposeTiling(numR, numC, numR * numC, sizeof(Q_T), transposeTiling);
    ConfusionTranspose(brcbResUb, addsResUb, TransposeType::TRANSPOSE_ND2ND_ONLY, transposeTiling);
#endif

    if (startRow != 0) {
        CopyValueToUb(startRow, dealRowCount);
    }
    LocalTensor<KV_T> valueUb = inputQue2.DeQue<KV_T>();
    if constexpr (ANTIQUANT) {
        LocalTensor<Q_T> valueCastUb = tmpBuff2.Get<Q_T>();
        // dealRowCount是根据Q_T计算的，可以保证UB足够
        Cast(valueCastUb, valueUb, AscendC::RoundMode::CAST_NONE, dealRowCount * headDimAlign);
        inputQue2.FreeTensor(valueUb);
        ComputeMm2ByVmla(vmlaResUb, brcbResUb, valueCastUb, dealRowCount, singleComputeRowCnt, headDimAlign, headDim);
    } else {
        ComputeMm2ByVmla(vmlaResUb, brcbResUb, valueUb, dealRowCount, singleComputeRowCnt, headDimAlign, headDim);
        inputQue2.FreeTensor(valueUb);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::GetConfusionTransposeTiling(
    int64_t numR, int64_t numC, const uint32_t stackBufferSize, const uint32_t typeSize,
    ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = numC;
    uint32_t width = numR;
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::Bmm2Compute()
{
    // 单次处理行数
    LocalTensor<T> vmlaResUb = tmpBuff1.Get<T>();
    uint32_t elementSize = vmlaResUb.GetSize();
    uint32_t maxDealRowCount = elementSize / headDimAlign; // 16(8192/512)~256(8192/32)
    if (maxDealRowCount >= (IFA_MAX_REPEAT_TIMES - 1)) {   // repeat次数最大255
        maxDealRowCount = IFA_MAX_REPEAT_TIMES - 1;
    }

    uint32_t singleComputeRowCnt = maxDealRowCount; // 16~255
    if (maxDealRowCount > copyValueActSplitS) {
        singleComputeRowCnt = copyValueActSplitS;
    }

    Duplicate(vmlaResUb, FLOAT_ZERO, singleComputeRowCnt * headDimAlign);

    for (uint32_t i = 0; i < copyValueLoopCount; i++) {
        ComputeSingleMm2(vmlaResUb, i * copyValueActSplitS, copyValueActSplitS, singleComputeRowCnt);
    }
    ComputeSingleMm2(vmlaResUb, copyValueTailStart, copyValueTailSplitSize, singleComputeRowCnt);
    ColumnSum(vmlaResUb, singleComputeRowCnt, headDimAlign);
    Add(bmm2ResUb, bmm2ResUb, vmlaResUb, headDim);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::ElewiseCompute(LocalTensor<T> &mmResUb, TBuf<> &tmpBuf,
                                                                               uint32_t startRow, uint32_t dealRowCount,
                                                                               uint32_t columnCount,
                                                                               uint32_t actualColumnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    pipe_barrier(PIPE_V);

    // pse shift mask
    if (pseShiftFlag) {
        PseShiftCopyIn(startRow, dealRowCount, actualColumnCount);
        LocalTensor<pseShiftType> pseShiftUb = inputQue2.DeQue<pseShiftType>();
        LocalTensor<float> pseShiftUbFloat = tmpBuf.Get<float>();
        for (uint32_t i = 0; i < dealRowCount; ++i) {
            Cast(pseShiftUbFloat[i * columnCount], pseShiftUb[i * pseMaskSizeAlign], AscendC::RoundMode::CAST_NONE,
                 pseMaskSizeAlign);
        }

        inputQue2.FreeTensor(pseShiftUb);
        pipe_barrier(PIPE_V);
        Add(mmResUb, mmResUb, pseShiftUbFloat, dealRowCount * columnCount);
        pipe_barrier(PIPE_V);
    }

    // attenMask
    if (attenMaskFlag) {
        AttenMaskCopyIn(attenMaskOffset, dealRowCount, actualColumnCount);
        LocalTensor<bool> attenMaskUb = inputQue2.DeQue<bool>();
        for (int i = 1; i < dealRowCount; i++) {
            DataCopy(attenMaskUb[i * attenMaskSizeAlign], attenMaskUb, attenMaskSizeAlign);
        }
        pipe_barrier(PIPE_V);

        LocalTensor<uint8_t> ubWorkSpace = tmpBuf.Get<uint8_t>(selectWithByteMaskTmpMinSize);
        SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
        selectWithBytesMaskShapeInfo.firstAxis = dealRowCount;
        selectWithBytesMaskShapeInfo.srcLastAxis = columnCount;
        selectWithBytesMaskShapeInfo.maskLastAxis = attenMaskSizeAlign;
        attenMaskUb.SetSize(dealRowCount * attenMaskSizeAlign);
        mmResUb.SetSize(dealRowCount * columnCount);
        SelectWithBytesMask(mmResUb, mmResUb, BOOL_ATTEN_MASK_SCALAR_VALUE, attenMaskUb, ubWorkSpace,
                            selectWithBytesMaskShapeInfo);
        inputQue2.FreeTensor(attenMaskUb);
        pipe_barrier(PIPE_V);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::SoftmaxFlashV2Compute(
    LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    SoftMaxShapeInfo srcShape = {dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
    SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG>(
        mmResUb, softmaxSumUb[baseOffset], softmaxMaxUb[baseOffset], mmResUb, softmaxExpUb[baseOffset],
        softmaxSumUb[baseOffset], softmaxMaxUb[baseOffset], softmaxTmpUb, newTiling, srcShape);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenAllVecNew<IFAT>::Bmm2FDDataCopyOut(LocalTensor<T> &attenOutUb, uint32_t startRow,
                                                           uint32_t dealRowCount, uint32_t columnCount,
                                                           uint32_t actualColumnCount)
{
    if (actualColumnCount % BLOCK_ELEMENT_NUM == 0) {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount / BLOCK_ELEMENT_NUM;
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopy(accumOutGm[tensorACoreOffset * splitKVNum + s2Idx * gSize * actualColumnCount +
                            startRow * actualColumnCount],
                 attenOutUb, dataCopyParams);
    } else {
#if (__CCE_AICORE__ > 200)
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm[tensorACoreOffset * splitKVNum + s2Idx * gSize * actualColumnCount +
                               startRow * actualColumnCount],
                    attenOutUb, dataCopyParams);
#endif
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenAllVecNew<IFAT>::Bmm2DataCopyOut(LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                                         uint32_t dealRowCount, uint32_t columnCount,
                                                         uint32_t actualColumnCount)
{
    uint32_t typeElementSize = ONE_BLK_SIZE / sizeof(OUT_T);
    if (actualColumnCount % typeElementSize == 0) {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount / typeElementSize;
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / typeElementSize;
        dataCopyParams.dstStride = 0;
        DataCopy(attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
    } else {
#if (__CCE_AICORE__ > 200)
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / typeElementSize;
        dataCopyParams.dstStride = 0;
        DataCopyPad(attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
#endif
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenAllVecNew<IFAT>::Bmm2CastAndCopyOut(LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                            uint32_t dealRowCount, uint32_t columnCount,
                                                            uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        LocalTensor<T> accumOutUb = outputQue1.AllocTensor<T>();
        DataCopy(accumOutUb, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(accumOutUb);
        outputQue1.DeQue<T>();
        Bmm2FDDataCopyOut(accumOutUb, startRow, dealRowCount, columnCount, actualColumnCount);
        outputQue1.FreeTensor(accumOutUb);
    } else {
        LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
#if (__CCE_AICORE__ > 200)
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
#else
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_NONE, dealRowCount * columnCount);
#endif
        outputQue1.EnQue(tmpBmm2ResCastTensor);
        outputQue1.DeQue<OUT_T>();
        Bmm2DataCopyOut(tmpBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
        outputQue1.FreeTensor(tmpBmm2ResCastTensor);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::PseShiftCopyIn(uint32_t startRow, uint32_t rowCount,
                                                                               uint32_t actualColumnCount)
{
    LocalTensor<pseShiftType> pseShiftUb = inputQue2.AllocTensor<pseShiftType>();
    pseMaskSizeAlign = Align(actualColumnCount, 16U); // 16: align to 32bytes
    uint32_t computeSize = rowCount * pseMaskSizeAlign;
    pseShiftUb.SetSize(computeSize);
#if (__CCE_AICORE__ > 200)
    if (actualColumnCount % 16 == 0) {
        for (uint32_t i = 0; i < rowCount; ++i) {
            DataCopy(pseShiftUb[i * pseMaskSizeAlign],
                     pseShiftGm[pseShiftOffset + startRow * pseShiftS + i * pseShiftS], pseMaskSizeAlign);
        }
    } else {
        uint32_t typeElementSize = BYTE_BLOCK / sizeof(pseShiftType);
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = actualColumnCount * sizeof(pseShiftType);
        intriParams.dstStride = (pseMaskSizeAlign - actualColumnCount) / typeElementSize;
        intriParams.srcStride = 0;
        DataCopyPadExtParams<pseShiftType> padParams;
        padParams.leftPadding = 0;
        padParams.rightPadding = (pseMaskSizeAlign - actualColumnCount) % typeElementSize;
        padParams.paddingValue = 0;
        padParams.isPad = true;
        for (uint32_t i = 0; i < rowCount; ++i) {
            DataCopyPad(pseShiftUb[i * pseMaskSizeAlign],
                        pseShiftGm[pseShiftOffset + startRow * pseShiftS + i * pseShiftS], intriParams, padParams);
        }
    }
#else
    for (uint32_t i = 0; i < rowCount; ++i) {
        DataCopy(pseShiftUb[i * pseMaskSizeAlign], pseShiftGm[pseShiftOffset + startRow * pseShiftS + i * pseShiftS],
                 pseMaskSizeAlign);
    }
#endif
    inputQue2.EnQue(pseShiftUb);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::ColumnSum(LocalTensor<T> srcUb, uint32_t dealRowCount,
                                                                          uint32_t columnCount)
{
    // 将srcUb的dealRowCount行累加到第一行,每行columnCount各元素
    for (uint32_t mask = IFA_MAX_REPEAT_TIMES / 2; mask > 0; mask = mask / 2) {
        if (dealRowCount & mask) {
            if (dealRowCount > mask) {
                pipe_barrier(PIPE_V);
                Add(srcUb, srcUb, srcUb[mask * columnCount], (dealRowCount - mask) * columnCount);
            }
            AccumulateByRow(srcUb, mask, columnCount);
            break;
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::ProcessVec1(const uint32_t sInnerLoopIdx)
{
    pipe_barrier(PIPE_V);
    ElewiseCompute(bmm1ResUb, tmpBuff2, 0, 1, actualSingleProcessSInnerSizeAlign, actualSingleProcessSInnerSize);

    if (copyValueLoopCount != 0) {
        CopyValueToUb(0, copyValueActSplitS);
    } else {
        CopyValueToUb(0, copyValueTailSplitSize);
    }

    LocalTensor<uint8_t> softmaxTmpUb = tmpBuff2.Get<uint8_t>();
    SoftmaxFlashV2Compute(bmm1ResUb, softmaxTmpUb, 0, 1, actualSingleProcessSInnerSizeAlign,
                          actualSingleProcessSInnerSize);
    pipe_barrier(PIPE_V);

    // 除第一个循环外，均需要更新中间计算结果
    if (sInnerLoopIdx > 0) {
        RowMuls(bmm2ResUb, bmm2ResUb, softmaxExpUb, 1, headDimAlign, headDim);
    }

    // copy lse
    if constexpr (FLASH_DECODE) {
        bool isLast = false;
        if (sInnerLoopIdx == sInnerLoopTimes - 1)
            isLast = true;
        if (isLast) {
            ComputeLogSumExpAndCopyToGm(softmaxSumUb, softmaxMaxUb);
        }
    } else if (softmaxLseFlag) {
        SoftmaxLseCopyOut(softmaxSumUb, softmaxMaxUb);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::ProcessVec2(const uint32_t sInnerLoopIdx)
{
    // 最后一次输出计算结果
    if (sInnerLoopIdx + 1 == sInnerLoopTimes) {
        pipe_barrier(PIPE_V);
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumUb, 1, headDimAlign, headDim);
        pipe_barrier(PIPE_V);
        if constexpr (ANTIQUANT) {
            LocalTensor<T> tmpAntiqParamUb = tmpBuff2.Get<T>();
            if (antiqOffsetExistFlag) {
                // bmm2Res + offsetV
                LocalTensor<Q_T> antiqOffsetUb = inputQue2.template AllocTensor<Q_T>();
                CopyIn<Q_T>(antiqOffsetUb, antiqOffsetGm, antiqValueParamOffset, 1, headDimAlign, headDim);
                inputQue2.template EnQue(antiqOffsetUb);
                inputQue2.DeQue<Q_T>();
                Cast(tmpAntiqParamUb, antiqOffsetUb, AscendC::RoundMode::CAST_NONE, headDim);
                inputQue2.FreeTensor(antiqOffsetUb);
                pipe_barrier(PIPE_V);
                Add(bmm2ResUb, bmm2ResUb, tmpAntiqParamUb, headDim);
                pipe_barrier(PIPE_V);
            }

            // ScaleV * bmm2Res
            LocalTensor<Q_T> antiqScaleUb = inputQue2.template AllocTensor<Q_T>();
            CopyIn<Q_T>(antiqScaleUb, antiqScaleGm, antiqValueParamOffset, 1, headDimAlign, headDim);
            inputQue2.template EnQue(antiqScaleUb);
            inputQue2.DeQue<Q_T>();
            Cast(tmpAntiqParamUb, antiqScaleUb, AscendC::RoundMode::CAST_NONE, headDim);
            inputQue2.FreeTensor(antiqScaleUb);
            pipe_barrier(PIPE_V);
            Mul(bmm2ResUb, bmm2ResUb, tmpAntiqParamUb, headDim);
            pipe_barrier(PIPE_V);
        }
        Bmm2CastAndCopyOut(bmm2ResUb, 0, 1, headDimAlign, headDim);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::SInnerLoopFunc(const uint32_t sInnerLoopIdx)
{
    // mm1
    Bmm1Compute();

    // v1
    ProcessVec1(sInnerLoopIdx);

    // mm2
    Bmm2Compute();

    // v2
    ProcessVec2(sInnerLoopIdx);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenAllVecNew<IFAT>::Process()
{
    if (g_coreType == AIV && tmpBlockIdx >= usedCoreNum) {
        // skip cores
    } else {
        // 初始化计算参数
        if constexpr (FLASH_DECODE) {
            InitCalcParams();
        } else {
            InitCalcParamsEach();
        }

        for (uint32_t bn2Idx = 0; bn2Idx < bn2LoopTimes; bn2Idx++) {
            GetBN2id(bn2Idx);
            GetActualSeqLen();
            CalculateSUnitSize();
            // ComputeKVPaddingBeginOffset return false means this loop skip calculation
            if (!ComputeKVPaddingBeginOffset()) {
                continue;
            }

            // 计算BN2方向的offset
            CalcBN2Offset();
            CalcBN2Params();
            // 根据当前块实际长度, 重配flashattention循环条件
            UpdateInnerLoopCond();
            pipe_barrier(PIPE_V);
            if (curActSeqLenIsZero) {
                continue;
            }

            // softmax不区分首次
            Duplicate(softmaxMaxUb, SOFTMAX_MIN_NUM, BUFFER_SIZE_BYTE_32B / sizeof(T));
            Duplicate(softmaxSumUb, FLOAT_ZERO, BUFFER_SIZE_BYTE_32B / sizeof(T));
            Duplicate(bmm2ResUb, FLOAT_ZERO, headDimAlign);

            QueryPreProcess();

            for (uint32_t sInnerLoopIdx = 0; sInnerLoopIdx < sInnerLoopTimes; sInnerLoopIdx++) {
                // 计算s2方向的offset
                CalcSInnerOffsetAndParams(sInnerLoopIdx);

                SInnerLoopFunc(sInnerLoopIdx);
            }
        }
    }

    // printf("usedCoreNum: %d\n", usedCoreNum);

    if constexpr (FLASH_DECODE) {
        // 多核同步
#if (__CCE_AICORE__ > 200)
        SyncAll();
#else
        LocalTensor<int32_t> workLocal = inputQue2.AllocTensor<int32_t>();
        SyncAll(syncGlobal, workLocal);
        inputQue2.FreeTensor(workLocal);
#endif
        FlashDecodeCompute();
    }
}
#endif // INCRE_FLASH_ATTENTION_ALLVEC_NEW
