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
 * \file flash_attention_score_grad_s1s2_bn2gs1s2.h
 * \brief
 */

#ifndef _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2GS1S2_H_
#define _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2GS1S2_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "pse.h"
#include "dropmask.h"

using namespace matmul;

constexpr static MatmulConfig NORM_DISABLE_INIT = {true,  false, false, 0,     0,     0,     false, false,
                                                   false, false, 0,     0,     0,     0,     0,     0,
                                                   0,     0,     true,  false, false, false, false, false};

__aicore__ inline void DataCopyOut(const __gm__ void *gm, const LocalTensor<int8_t> &co1Local,
                                   const void *dataCopyOutParams, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    const DataCopyOutParams *param = reinterpret_cast<const DataCopyOutParams *>(dataCopyOutParams);
    uint64_t dstStride = dataPtr * 16 / 8 - param->burstLen;
    FixpipeParams<float> fixpipeParams(param->cBurstNum, param->burstLen, param->srcStride,
                                       static_cast<uint32_t>(dstStride));

    if (param->enUnitFlag) {
        fixpipeParams.unitFlag = 3;
    }
    LocalTensor<float> tmpLocal = co1Local.template ReinterpretCast<float>();
    GlobalTensor<float> tmpGm;
    tmpGm.SetGlobalBuffer((__gm__ float *)(gm));
    Fixpipe(tmpGm, tmpLocal, fixpipeParams);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK = 0, const uint32_t IS_PSE = 1,
          const uint32_t IS_DROP = 1, const CubeFormat MM_OUT_FORMAT = CubeFormat::ND, const uint32_t INPUT_LAYOUT = 0,
          const CubeFormat MM2_OUT_FORMAT = CubeFormat::NZ>
class FlashAttentionScoreGradS1s2Bn2gs1s2 {
public:
    __aicore__ inline FlashAttentionScoreGradS1s2Bn2gs1s2(){};

    __aicore__ inline void Init(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query,
                                __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask,
                                __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
                                __gm__ uint8_t *prefixN, __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                                __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
                                __gm__ uint8_t *workspace,
                                const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict ordTilingData,
                                TPipe *pipe_in);
    __aicore__ inline void CopyInSoftMaxGrad(LocalTensor<T1> &dstTensor, LocalTensor<T1> &dstTensor2,
                                             int64_t softmaxGradOffset, uint32_t s1Extend, uint32_t dExtend,
                                             uint32_t dExtendAlign);
    __aicore__ inline void CalcSoftMaxGrad(LocalTensor<float> &sfmgClc3, int64_t aTensorOffset, uint32_t s1Extend);
    __aicore__ inline void CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend, uint32_t softMaxOffset);
    __aicore__ inline void CalcSoftMax(LocalTensor<T2> &dstTensor, LocalTensor<float> &srcTensor, uint32_t s1Extend,
                                       uint32_t s2Extend, uint32_t s2ExtendAlign, const SoftMaxTiling &tiling);
    __aicore__ inline void CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor, int64_t attenMaskOffset,
                                               uint32_t s1Extend, uint32_t s2Extend);
    __aicore__ inline void CalcAttenMaskBool(LocalTensor<T2> &dstTensor, LocalTensor<uint8_t> srcTensor,
                                             uint32_t s1Extend, uint32_t s2Extend, uint8_t maskType = 0);
    __aicore__ inline void CalcAttenMaskOffset(int64_t &attenMaskOffset, const int64_t delta, uint32_t s1VSize,
                                               uint32_t s2VSize);
    __aicore__ inline void CalcAttenBandMode(int64_t compressMode, int64_t causal_delta);
    __aicore__ inline void CalcAttenMaskOffsetForPrefixCompressMode(int64_t &attenMaskOffset, int64_t &attenMaskOffse2,
                                                                    const int64_t delta, uint32_t s1VSize,
                                                                    uint32_t s2VSize, uint32_t s2VBegin,
                                                                    bool &canSimplify);
    __aicore__ inline void CalcAttenMaskOffsetWithSparseMode(int64_t &attenMaskOffset, int64_t &attenMaskOffset2,
                                                             uint32_t s1VSize, uint32_t s2VSize, int64_t curS1Idx,
                                                             uint32_t s2VBegin, bool &canSimplify);
    __aicore__ inline void CalcAttenMaskOffsetWithSparseModeForUnpad(int64_t &attenMaskOffset,
                                                                     int64_t &attenMaskOffset2, uint32_t s1VSize,
                                                                     uint32_t s2VSize, int64_t curS1Idx,
                                                                     uint32_t s2VBegin, bool unpadUseBand,
                                                                     bool &canSimplify);
    __aicore__ inline void DropOutCopy(LocalTensor<uint8_t> &vecInDropBuffer, int64_t curS1Idx, int64_t s2VBegin);

    __aicore__ inline void Process();
    __aicore__ inline void UpdateToken(int64_t bIdx);
    __aicore__ inline bool IsCubeBlockNeedCompute(int64_t baseIndex);
    __aicore__ inline void InitIndex(int64_t index);
    __aicore__ inline void SubGrapA(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx);
    __aicore__ inline void SubGrapB(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx);
    __aicore__ inline void Compute(int64_t preIndex, int64_t nextIndex);
    __aicore__ inline void SyncALLCores();
    __aicore__ inline bool CheckIsValidBlock(int64_t baseIdx, int64_t s1oDimIdx, int64_t s2oCvDimIdx, int64_t curBIdx);
    __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);

    using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
    using cType1 = MatmulType<TPosition::GM, MM_OUT_FORMAT, T2>;
    using biasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using aType2 = MatmulType<TPosition::GM, MM_OUT_FORMAT, T1, true>;
    using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using cType2 = MatmulType<TPosition::GM, MM2_OUT_FORMAT, float>;
    using biasType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    Matmul<aType1, bType1, cType1, biasType1, NORM_DISABLE_INIT> mm1;
    using modeTypeMm = typename AscendC::Conditional<
        (MM2_OUT_FORMAT == CubeFormat::NZ),
        Matmul<aType2, bType2, cType2, biasType2, NORM_DISABLE_INIT, MatmulCallBackFunc<DataCopyOut>>,
        Matmul<aType2, bType2, cType2, biasType2, NORM_DISABLE_INIT>>::type;
    modeTypeMm mm3;

    using modeTypeMm4 = typename AscendC::Conditional<
        (MM2_OUT_FORMAT == CubeFormat::NZ),
        Matmul<aType2, bType2, cType2, biasType2, NORM_DISABLE_INIT, MatmulCallBackFunc<DataCopyOut>>,
        Matmul<aType2, bType2, cType2, biasType2, NORM_DISABLE_INIT>>::type;

    modeTypeMm4 mm4;

    __aicore__ inline void NZCopyIn(int64_t mmAddr, GlobalTensor<T2> &mmWspGm, LocalTensor<T2> &mmTensorCurr,
                                    uint32_t s1VecSize, uint32_t s2VecSize);
    __aicore__ inline void NZ2ND(LocalTensor<T2> &mmTensorCurr, LocalTensor<T2> &tmpTensor, uint32_t s1VecSize,
                                 uint32_t s2VecSize);
    __aicore__ inline void ND2NZ(LocalTensor<T1> &mmTensorCurr, LocalTensor<T1> &tmpTensor, uint32_t s1VecSize,
                                 uint32_t s2VecSize);

protected:
    TPipe *pipe;
    TBuf<> ubBuffer;
    TBuf<> tmpBuffer;
    TBuf<> vecClc3;

    uint32_t coreNum;
    int64_t cBlockIdx;
    const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict TilingData;

    // input
    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, forwardResGm, pseGm;
    GlobalTensor<uint8_t> maskWorkSpaceGm, attenMaskU8Gm, dropMaskGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm;

    // output
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    GlobalTensor<T1> dropWorkSpaceGm, mulWorkSpaceGm;

    // workspace
    GlobalTensor<T2> mm1WorkspaceGm;
    GlobalTensor<T2> mm2WorkspaceGm;

    __gm__ uint8_t *prefixN_addr;
    __gm__ uint8_t *actual_seq_qlen_addr;
    __gm__ uint8_t *actual_seq_kvlen_addr;

    // AscendC
    GlobalTensor<int32_t> syncGlobal;

    // matmal1/matmal2 result buffer
    GlobalTensor<float> matmalResultBuffer1;
    GlobalTensor<float> matmalResultBuffer2;

    GlobalTensor<half> pseAlibiGm;
    GlobalTensor<float> dvGm;
    __gm__ uint8_t *pseSlope;

    PseInfo pseInfo = {0};

    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t TND = 3;
    constexpr static uint32_t ENABLE = 1;

    T2 mulsValue = -10000.0;

    // optional control
    float keepProb;
    int64_t s1Token;
    int64_t s2Token;
    int64_t actualCalcS1Token;
    int64_t actualCalcS2Token;
    uint32_t sparseMode;
    bool isSparse;

    // org shape info
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t attenMaskDimS2;

    uint32_t baseMN;
    uint32_t cubeBaseMN;

    // split info
    int64_t s1Outer;
    uint32_t s1CvRatio;
    uint32_t s1CvInner;
    uint32_t s1CvTail;
    uint32_t s1CvExtend{0};

    int64_t s2Outer;
    uint32_t s2CvRatio;
    uint32_t s2Inner;
    uint32_t sfmgdOuter;
    uint32_t sfmgdInner;
    uint32_t sfmgdTail;
    uint32_t sfmgdTailAlign;
    bool dropBitMode;

    int64_t blockOuter;

    // sparse block info
    const int64_t *blockStarts;
    const int64_t *blockEnds;

    // buferinfo
    uint32_t matmalWorkspaceSize;

    // base info
    int64_t n2gs1os2o;
    int64_t gs1os2o;
    int64_t s1os2o;

    int64_t baseIdx{0};
    int64_t bDimIdx{0};
    int64_t n2DimIdx{0};
    int64_t gDimIdx{0};
    int64_t s1oDimIdx{0};
    int64_t s2oCvDimIdx{0};

    uint32_t preS2CvBegin{0};
    uint32_t preS2CvEnd{0};
    uint32_t nextS2CvBegin{0};
    uint32_t nextS2CvEnd{0};

    int32_t isStart = 1;
    uint32_t pingpongIdx = 1;

    // db
    int64_t s2CvExtend = 0;
    int64_t s2CvExtendAlign = 0;
    int64_t s1CvExtendAlign = 0;
    uint32_t s1VecLoop = 0;
    uint32_t s1VecSize = 0;
    uint32_t s1ExtendSubGraph = 0;
    uint32_t s2Extend = 0;
    uint32_t s2ExtendAlign = 0;
    uint32_t s2VecLoop = 0;
    uint32_t s2VecSize = 0;
    uint32_t s2VecSizeAlign = 0;

    // unpack
    int64_t bDimIdxTmp = 0;
    int64_t n2DimIdxTmp = 0;
    int64_t gDimIdxTmp = 0;
    int64_t s1oDimIdxTmp = 0;
    int64_t bandIdx = 0;

    DropMaskInfo dropMaskInfo = {0};
    // db buffer
    constexpr static uint32_t T2Begin = 0;
    constexpr static uint32_t T1Begin = 33 * 1024;
    constexpr static uint32_t BoolBegin = 50 * 1024;
    constexpr static uint32_t T2BlockBegin = 58 * 1024;
    constexpr static uint32_t U8Begin = 66 * 1024;
    constexpr static uint32_t DbBegin = 74 * 1024;
    constexpr static uint32_t hufTmpBuffBegin = 16 * 1024;

    // calDtype
    constexpr static uint32_t calDtypeBytes = 4;

    // other const
    constexpr static uint32_t cal_block_num = 32 / sizeof(T2);
    constexpr static uint32_t cal_repeat_num = 256 / sizeof(T2);
    constexpr static uint32_t input_block_num = 32 / sizeof(T1);
    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t VEC_REPEAT = 8;
    constexpr static uint32_t MAX_BASIC_BLOCK_SIZE = 1024;
    constexpr static uint32_t PSE_PERFORMANCE_MODE = 0x12;
    constexpr static uint32_t PREFIX_COMPRESS_CAUSAL_S_SIZE = 2048;
    constexpr static uint32_t PREFIX_COMPRESS_ALL_MASK_S1_SIZE = 1024;

    constexpr static uint32_t VEC_S2_LEN = 256;
    enum class AttenMaskCompress {
        Empty = 0,
        PreOnly = 1,
        NextOnly = 2,
        All = 3
    };
    AttenMaskCompress AttenBandMode = AttenMaskCompress::All;
};

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<
    T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
    MM2_OUT_FORMAT>::Init(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query,
                          __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask,
                          __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
                          __gm__ uint8_t *prefixN, __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                          __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
                          __gm__ uint8_t *workspace,
                          const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict ordTilingData,
                          TPipe *pipe_in)
{
    keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    dxGm.SetGlobalBuffer((__gm__ T1 *)dx);
    queryGm.SetGlobalBuffer((__gm__ T1 *)query);
    forwardResGm.SetGlobalBuffer((__gm__ T1 *)forward_res);
    pseGm.SetGlobalBuffer((__gm__ T1 *)pse_shift);
    pseSlope = pse_shift;

    dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
    attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    dvGm.SetGlobalBuffer((__gm__ float *)dv);

    // init current core tilingInfo
    cBlockIdx = GetBlockIdx();
    TilingData = ordTilingData;
    pipe = pipe_in;
    coreNum = TilingData->s1s2BNGS1S2BaseParams.coreNum;

    // shape info
    b = TilingData->s1s2BNGS1S2BaseParams.b;
    n2 = TilingData->s1s2BNGS1S2BaseParams.n2;
    g = TilingData->s1s2BNGS1S2BaseParams.g;
    s1 = TilingData->s1s2BNGS1S2BaseParams.s1;
    s2 = TilingData->s1s2BNGS1S2BaseParams.s2;
    d = TilingData->s1s2BNGS1S2BaseParams.d;
    attenMaskDimS2 = TilingData->s1s2BNGS1S2BaseParams.attenMaskS2Size;

    s1Token = TilingData->s1s2BNGS1S2BaseParams.s1Token;
    s2Token = TilingData->s1s2BNGS1S2BaseParams.s2Token;
    actualCalcS1Token = s1Token;
    actualCalcS2Token = s2Token;
    sparseMode = TilingData->s1s2BNGS1S2BaseParams.sparseMode;
    isSparse = false;
    if (TilingData->s1s2BNGS1S2BaseParams.isSparse == 1) {
        isSparse = true;
    }
    bandIdx = TilingData->s1s2BNGS1S2SplitCoreParams.bandIdx;

    // split info
    s1Outer = TilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
    s1CvRatio = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvRatio;
    s1CvInner = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvInner;
    s1CvTail = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvTail;
    s2Outer = TilingData->s1s2BNGS1S2SplitCoreParams.s2Outer;
    s2CvRatio = TilingData->s1s2BNGS1S2SplitCoreParams.s2CvRatio;
    s2Inner = TilingData->s1s2BNGS1S2SplitCoreParams.s2Inner;

    sfmgdOuter = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdOuter;
    sfmgdInner = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdFactor;
    sfmgdTail = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdTail;
    sfmgdTailAlign = (sfmgdTail + input_block_num - 1) / input_block_num * input_block_num;

    // no sparse blockouter ceil to even
    blockOuter = (TilingData->s1s2BNGS1S2SplitCoreParams.blockOuter + 1) / 2 * 2;
    baseMN = TilingData->s1s2BNGS1S2SplitCoreParams.baseMN;
    cubeBaseMN = s1CvRatio * s2CvRatio * baseMN;

    blockStarts = TilingData->s1s2BNGS1S2BlockNumList.blockStarts;
    blockEnds = TilingData->s1s2BNGS1S2BlockNumList.blockEnds;

    prefixN_addr = prefixN;
    actual_seq_qlen_addr = actual_seq_qlen;
    actual_seq_kvlen_addr = actual_seq_kvlen;

    if constexpr (IS_PSE == ENABLE) {
        uint32_t pseShapeType = TilingData->s1s2BNGS1S2BaseParams.pseShapeType;
        pseInfo.s2Size = s2;
        pseInfo.s1Size = s1;
        pseInfo.gSize = g;
        pseInfo.n2G = n2 * g;
        pseInfo.pseType = TilingData->s1s2BNGS1S2BaseParams.pseType;
        pseInfo.pseShapeType = pseShapeType;
        if (pseShapeType == 2 || pseShapeType == 3 || pseShapeType == 4) {
            pseInfo.pseShapeType = 0;
        } else if (pseShapeType == 5) {
            pseInfo.pseShapeType = 2;
        } else if (pseShapeType == 6) {
            pseInfo.pseShapeType = 3;
        }

        pseInfo.pseAlibiBaseS1 = TilingData->s1s2BNGS1S2BaseParams.pseAlibiBaseS1;
        pseInfo.pseAlibiBaseS2 = TilingData->s1s2BNGS1S2BaseParams.pseAlibiBaseS2;
        pseInfo.qStartIdx = TilingData->s1s2BNGS1S2BaseParams.qStartIdx;
        pseInfo.kvStartIdx = TilingData->s1s2BNGS1S2BaseParams.kvStartIdx;
        pseInfo.pseEncodeType = (pseShapeType == 3 || pseShapeType == 4) ? 0x11 : 0;
        pseInfo.pseBSize = (pseShapeType == 2 || pseShapeType == 4) ? 1 : b;
        pseInfo.pseS1Size = 1024;
        pseInfo.pseS2Size = s2;
        pseInfo.needCast = false;
    }

    dropBitMode = s2 % 8 == 0;
    keepProb = TilingData->s1s2BNGS1S2BaseParams.keepProb;
    if constexpr (INPUT_LAYOUT == TND) {
        int64_t seqS2Len = 0;
        seqS2Len = ((__gm__ int64_t *)actual_seq_kvlen)[0];
        dropBitMode = (seqS2Len % 8 == 0);
        for (int64_t i = 0; i + 1 < b; i++) {
            seqS2Len = ((__gm__ int64_t *)actual_seq_kvlen)[i + 1] - ((__gm__ int64_t *)actual_seq_kvlen)[i];
            dropBitMode = (dropBitMode && (seqS2Len % 8 == 0));
        }
    }

    // idx info
    n2gs1os2o = n2 * g * s1Outer * s2Outer;
    gs1os2o = g * s1Outer * s2Outer;
    s1os2o = s1Outer * s2Outer;

    int64_t maskPreBlockTotal = TilingData->preTilingData.maskPreBlockTotal;
    int64_t qPostBlockTotal = TilingData->postTilingData.qSizeAlign;
    int64_t kvPostBlockTotal = TilingData->postTilingData.kvSizeAlign;

    // init workspace address
    syncGlobal.SetGlobalBuffer((__gm__ int32_t *)workspace);
    int64_t workspaceOffsets = SYNC_GLOBAL_WORKSPACE_SIZE;
    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + qPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + kvPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + kvPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

    if constexpr (IS_DROP == ENABLE) {
        if (!dropBitMode) {
            maskWorkSpaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace + workspaceOffsets);
            workspaceOffsets =
                (workspaceOffsets + maskPreBlockTotal + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
        }
    }

    int64_t pseInnerAlibiSize = TilingData->s1s2BNGS1S2BaseParams.pseAlibiBaseS1 *
                                this->TilingData->s1s2BNGS1S2BaseParams.pseAlibiBaseS2 * sizeof(half);
    int64_t pseAlibiOffset =  CeilDiv(pseInnerAlibiSize, 512) * 512;

    // matmal1 and matmal2 workspace size
    matmalWorkspaceSize = cubeBaseMN * sizeof(float);
    mm1WorkspaceGm.SetGlobalBuffer((__gm__ T2 *)(workspace + workspaceOffsets + cBlockIdx * matmalWorkspaceSize));
    mm2WorkspaceGm.SetGlobalBuffer(
        (__gm__ T2 *)(workspace + workspaceOffsets + coreNum * matmalWorkspaceSize + cBlockIdx * matmalWorkspaceSize));

    // drop workspace offset
    workspaceOffsets = (workspaceOffsets + coreNum * cubeBaseMN * sizeof(float) * INPUT_NUMS + ADDR_ALIGN_SIZE) /
                       ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dropWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + workspaceOffsets / sizeof(T1));

    // mul workspace offset
    workspaceOffsets = (workspaceOffsets + coreNum * cubeBaseMN * sizeof(T1) * 2 + ADDR_ALIGN_SIZE) /
                       ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    mulWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + workspaceOffsets / sizeof(T1));

    uint64_t pseAlibiAddr = (workspaceOffsets + coreNum * cubeBaseMN * sizeof(T1) * 2 + ADDR_ALIGN_SIZE) /
                       ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    this->pseAlibiGm.SetGlobalBuffer((__gm__ half*)(workspace + pseAlibiAddr + cBlockIdx * pseAlibiOffset));

    InitOutput<int32_t>(syncGlobal[GetBlockIdx() * 8], 8, 0);

    pipe->InitBuffer(ubBuffer, 148 * 1024);
    pipe->InitBuffer(tmpBuffer, 33 * 1024);
    pipe->InitBuffer(vecClc3, 8 * 1024);

    if constexpr (IS_DROP == ENABLE) {
        if constexpr (INPUT_LAYOUT != TND) {
            // for compute dropout mask offset
            dropMaskInfo.s1Size = s1;
            dropMaskInfo.s2Size = s2;
        }

        // for compute dropout mask offset
        dropMaskInfo.n2G = n2 * g;
        dropMaskInfo.gSize = g;
        dropMaskInfo.s2Idx = 1;
        dropMaskInfo.s1BaseSize = s1CvInner;

        // for copy and compute in dropout mask
        dropMaskInfo.boolMode = dropBitMode ? false : true;
        dropMaskInfo.keepProb = keepProb;
    }

    if constexpr (IS_PSE == ENABLE) {
        if (cBlockIdx < coreNum &&
            (TilingData->s1s2BNGS1S2BaseParams.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            TilingData->s1s2BNGS1S2BaseParams.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            LocalTensor<half> pseHelpBuffer = ubBuffer.GetWithOffset<half>(16 * 1024 / sizeof(half), T1Begin);
            PseInnerAlibiCreate<true>(this->pseAlibiGm, pseHelpBuffer, pseInfo);
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::GetSeqQlenKvlenByBidx(int64_t bIdx,
                                                                            int64_t &actualSeqQlen,
                                                                            int64_t &actualSeqKvlen)
{
    if (unlikely(bIdx == 0)) {
        actualSeqQlen = ((__gm__ int64_t *)actual_seq_qlen_addr)[0];
        actualSeqKvlen = ((__gm__ int64_t *)actual_seq_kvlen_addr)[0];
    } else {
        actualSeqQlen =
            ((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx] - ((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx - 1];
        actualSeqKvlen =
            ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIdx] - ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIdx - 1];
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::CopyInSoftMaxGrad(LocalTensor<T1> &dstTensor,
                                                                       LocalTensor<T1> &dstTensor2,
                                                                       int64_t softmaxGradFrontOffset,
                                                                       uint32_t s1Extend, uint32_t dExtend,
                                                                       uint32_t dExtendAlign)
{
    int64_t transpse_stride = 0;
    if constexpr (INPUT_LAYOUT == BNGSD) {
        transpse_stride = (d - dExtend) * sizeof(T1);
    } else if constexpr (INPUT_LAYOUT == SBNGD) {
        transpse_stride = (static_cast<int64_t>(b) * n2 * g * d - dExtend) * sizeof(T1);
    } else if constexpr (INPUT_LAYOUT == BSNGD) {
        transpse_stride = (n2 * g * d - dExtend) * sizeof(T1);
    } else if constexpr (INPUT_LAYOUT == TND) {
        transpse_stride = (n2 * g * d - dExtend) * sizeof(T1);
    }

    if (transpse_stride == 0) {
        DataCopy(dstTensor, forwardResGm[softmaxGradFrontOffset], s1Extend * dExtend);
        DataCopy(dstTensor2, dxGm[softmaxGradFrontOffset], s1Extend * dExtend);
    } else {
        DataCopyPad(dstTensor, forwardResGm[softmaxGradFrontOffset],
                    {static_cast<uint16_t>(s1Extend), static_cast<uint32_t>(dExtend * sizeof(T1)),
                     static_cast<uint32_t>(transpse_stride), 0, 0},
                    {true, 0, static_cast<uint8_t>((dExtendAlign - dExtend)), 0});
        DataCopyPad(dstTensor2, dxGm[softmaxGradFrontOffset],
                    {static_cast<uint16_t>(s1Extend), static_cast<uint32_t>(dExtend * sizeof(T1)),
                     static_cast<uint32_t>(transpse_stride), 0, 0},
                    {true, 0, static_cast<uint8_t>((dExtendAlign - dExtend)), 0});
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::CalcSoftMaxGrad(LocalTensor<float> &sfmgClc3,
                                                                     int64_t aTensorOffset, uint32_t s1Extend)
{
    LocalTensor<float> sfmgClc1 = ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(T2), 0);
    LocalTensor<float> sfmgClc2 = ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(T2), 32 * 1024);
    Duplicate<float>(sfmgClc3, 0.0, s1Extend * 32 / sizeof(float));

    event_t curEventId = EVENT_ID7;
    for (uint32_t sfmgdIdx = 0; sfmgdIdx < sfmgdOuter; sfmgdIdx++) {
        LocalTensor<T1> vecInBuffer;
        LocalTensor<T1> vecInBuffer2;
        if constexpr (!IsSameType<T1, float>::value) {
            vecInBuffer = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), 64 * 1024);
            vecInBuffer2 = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), 80 * 1024);
        }
        LocalTensor<uint8_t> vecOutBuffer = ubBuffer.GetWithOffset<uint8_t>(32 * 1024 / sizeof(uint8_t), 96 * 1024);
        LocalTensor<T2> softmaxGradTmp = ubBuffer.GetWithOffset<T2>(8 * 1024 / sizeof(T2), 128 * 1024);
        int64_t softmaxGradFrontOffset = aTensorOffset + sfmgdIdx * sfmgdInner;
        uint32_t dExtend = (sfmgdIdx == sfmgdOuter - 1) ? sfmgdTail : sfmgdInner;
        uint32_t dExtendAlign = (sfmgdIdx == sfmgdOuter - 1) ? sfmgdTailAlign : sfmgdInner;
        bool isBasicBlock = (s1Extend % 8 == 0) && (dExtend % 64 == 0);

        if (sfmgdIdx > 0) {
            wait_flag(PIPE_V, PIPE_MTE2, curEventId);
        }
        if constexpr (!IsSameType<T1, float>::value) {
            CopyInSoftMaxGrad(vecInBuffer, vecInBuffer2, softmaxGradFrontOffset, s1Extend, dExtend, dExtendAlign);
        } else {
            CopyInSoftMaxGrad(sfmgClc1, sfmgClc2, softmaxGradFrontOffset, s1Extend, dExtend, dExtendAlign);
        }
        set_flag(PIPE_MTE2, PIPE_V, curEventId);
        wait_flag(PIPE_MTE2, PIPE_V, curEventId);
        if constexpr (!IsSameType<T1, float>::value) {
            Cast(sfmgClc1, vecInBuffer, RoundMode::CAST_NONE, s1Extend * dExtendAlign);
            Cast(sfmgClc2, vecInBuffer2, RoundMode::CAST_NONE, s1Extend * dExtendAlign);
            pipe_barrier(PIPE_V);
        }
        uint32_t shapeArray[2];
        shapeArray[0] = s1Extend;
        shapeArray[1] = dExtendAlign;
        sfmgClc1.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        sfmgClc2.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        uint32_t shapeArray1[2];
        shapeArray1[0] = s1Extend;
        shapeArray1[1] = 32 / sizeof(float);
        softmaxGradTmp.SetShapeInfo(ShapeInfo(2, shapeArray1, DataFormat::ND));

        if (isBasicBlock) {
            SoftmaxGradFront<float, true>(softmaxGradTmp, sfmgClc1, sfmgClc2, vecOutBuffer,
                                          TilingData->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<float, false>(softmaxGradTmp, sfmgClc1, sfmgClc2, vecOutBuffer,
                                           TilingData->softmaxGradTilingData);
        }
        pipe_barrier(PIPE_V);
        Add(sfmgClc3, softmaxGradTmp, sfmgClc3, s1Extend * 32 / sizeof(float));
        if (sfmgdIdx < (sfmgdOuter - 1)) {
            set_flag(PIPE_V, PIPE_MTE2, curEventId);
        }
    }
    pipe_barrier(PIPE_ALL);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend,
                                                                   uint32_t softMaxOffset)
{
    DataCopyPad(dstTensor, softmaxSumGm[softMaxOffset], {1, static_cast<uint16_t>(s1Extend * 32), 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(dstTensor[s1Extend * 32 / sizeof(float)], softmaxMaxGm[softMaxOffset],
                {1, static_cast<uint16_t>(s1Extend * 32), 0, 0}, {false, 0, 0, 0});
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::CalcSoftMax(LocalTensor<T2> &dstTensor,
                                                                 LocalTensor<float> &srcTensor, uint32_t s1Extend,
                                                                 uint32_t s2Extend, uint32_t s2ExtendAlign,
                                                                 const SoftMaxTiling &tiling)
{
    bool isBasicBlock = (s1Extend % 8 == 0) && (s2Extend % 64 == 0);

    if (isBasicBlock) {
        LocalTensor<uint8_t> vecOutBuffer = tmpBuffer.Get<uint8_t>();
        uint32_t shapeArray1[2];
        shapeArray1[0] = s1Extend;
        shapeArray1[1] = s2Extend;
        dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray1, DataFormat::ND));
        SimpleSoftMax<T2, true, true>(dstTensor, srcTensor, srcTensor[s1Extend * 32 / sizeof(float)], dstTensor,
                                      vecOutBuffer, tiling);
    } else {
        LocalTensor<T2> vecOutBuffer = tmpBuffer.Get<T2>();
        uint32_t sub_block_count = (s2Extend + cal_repeat_num - 1) / cal_repeat_num;

        for (uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
            uint32_t subMaskCount =
                (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * cal_repeat_num) : cal_repeat_num;
            Sub(dstTensor[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num], srcTensor[s1Extend * 8],
                subMaskCount, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8), 1});
            pipe_barrier(PIPE_V);
            Exp(vecOutBuffer[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num], subMaskCount, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8)});
            pipe_barrier(PIPE_V);
            Div(dstTensor[subIdx * cal_repeat_num], vecOutBuffer[subIdx * cal_repeat_num], srcTensor, subMaskCount,
                s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8), 1});
            pipe_barrier(PIPE_V);
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor,
                                                                         int64_t attenMaskOffset, uint32_t s1Extend,
                                                                         uint32_t s2Extend)
{
    DataCopyPad(dstTensor, attenMaskU8Gm[attenMaskOffset],
                {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2Extend * sizeof(uint8_t)),
                 static_cast<uint16_t>((attenMaskDimS2 - s2Extend) * sizeof(uint8_t)), 0},
                {false, 0, 0, 0});
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::CalcAttenMaskBool(LocalTensor<T2> &dstTensor,
                                                                       LocalTensor<uint8_t> srcTensor,
                                                                       uint32_t s1Extend, uint32_t s2Extend,
                                                                       uint8_t maskType)
{
    LocalTensor<uint8_t> tmpUbBuffer = tmpBuffer.Get<uint8_t>();

    T2 scalar;
    if constexpr (IsSameType<T2, float>::value) {
        uint32_t tmp = 0xFF7FFFFF;
        scalar = *((float *)&tmp);
    } else {
        uint16_t tmp = 0xFBFF;
        scalar = *((half *)&tmp);
    }

    SelectWithBytesMaskShapeInfo info;
    info.firstAxis = s1Extend;
    info.srcLastAxis = s2Extend;
    info.maskLastAxis = (s2Extend * sizeof(uint8_t) + 31) / 32 * 32 / sizeof(uint8_t);
    dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
    srcTensor.SetSize(info.firstAxis * info.maskLastAxis);
    if (maskType == 0) {
        SelectWithBytesMask(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
    } else {
        SelectWithBytesMask(dstTensor, scalar, dstTensor, srcTensor, tmpUbBuffer, info);
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT>::Process()
{
    if (isSparse) {
        int64_t preIndex = blockStarts[cBlockIdx];
        if (blockEnds[cBlockIdx] == 0) {
            return;
        }

        int64_t blockEndsTemp = blockEnds[cBlockIdx] + 1;
        for (int64_t blockInnerIdx = blockStarts[cBlockIdx] + 1; blockInnerIdx < blockEndsTemp; blockInnerIdx++) {
            if (isStart == 1) {
                if (!IsCubeBlockNeedCompute(preIndex)) {
                    preIndex = blockInnerIdx;
                    continue;
                }
            }
            preS2CvBegin = nextS2CvBegin;
            preS2CvEnd = nextS2CvEnd;
            if (IsCubeBlockNeedCompute(blockInnerIdx) || (blockInnerIdx >= blockEnds[cBlockIdx])) {
                int64_t nextIndex = blockInnerIdx;
                if (blockInnerIdx >= blockEnds[cBlockIdx]) {
                    nextIndex = 0u;
                }
                Compute(preIndex, nextIndex);
                preIndex = nextIndex;
            }
        }
    } else {
        int64_t preIndex = cBlockIdx;
        int64_t total = static_cast<int64_t>(b) * n2 * g * s1Outer * s2Outer;
        if (cBlockIdx >= total) {
            return;
        }

        IsCubeBlockNeedCompute(preIndex);
        preS2CvBegin = nextS2CvBegin;
        preS2CvEnd = nextS2CvEnd;

        int64_t totalTemp = total + blockOuter;
        for (int64_t blockInnerIdx = cBlockIdx + blockOuter; blockInnerIdx < totalTemp; blockInnerIdx += blockOuter) {
            if (IsCubeBlockNeedCompute(blockInnerIdx) || (blockInnerIdx >= total)) {
                int64_t nextIndex = blockInnerIdx;
                if (blockInnerIdx >= total) {
                    nextIndex = 0u;
                }
                Compute(preIndex, nextIndex);
                preIndex = nextIndex;
                preS2CvBegin = nextS2CvBegin;
                preS2CvEnd = nextS2CvEnd;
            }
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline bool
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::CheckIsValidBlock(int64_t baseIdx, int64_t s1oDimIdx,
                                                                       int64_t s2oCvDimIdx, int64_t curBIdx)
{
    int64_t S1 = static_cast<int64_t>(s1);
    int64_t S2 = static_cast<int64_t>(s2);

    if constexpr (INPUT_LAYOUT == TND) {
        GetSeqQlenKvlenByBidx(curBIdx, S1, S2);
    }
    int64_t cvS2Inner = static_cast<int64_t>(s2CvRatio) * s2Inner;
    int64_t s2IgnoredEndLen = S1 - static_cast<int64_t>(s1CvInner * (s1oDimIdx + 1));
    int64_t s2EndLen = 0;
    if (S2 > s2IgnoredEndLen) {
        s2EndLen = S2 - s2IgnoredEndLen;
    } else {
        s2EndLen = 0;
    }

    if (sparseMode == 5 || sparseMode == 6) {
        s2EndLen = s2EndLen > ((__gm__ int64_t *)prefixN_addr)[curBIdx]
                            ? s2EndLen
                            : ((__gm__ int64_t *)prefixN_addr)[curBIdx];
        s2EndLen = s2EndLen < S2 ? s2EndLen : S2;
    }

    uint32_t s2IdxLeft = s2oCvDimIdx * s2Inner * s2CvRatio;
    uint32_t s2IdxRight = (s2oCvDimIdx + 1) * cvS2Inner;
    bool doSparse = s2IdxLeft < s2EndLen;
    if (doSparse) {
        nextS2CvBegin = s2IdxLeft;
        nextS2CvEnd = s2IdxRight > s2EndLen ? s2EndLen : s2IdxRight;
    }
    return doSparse;
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT>::UpdateToken(int64_t bIdx)
{
    // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
    if constexpr (IS_ATTEN_MASK != ENABLE) {
        return;
    }

    int64_t actualS1Len;
    int64_t actualS2Len;
    if (sparseMode == 7 && bIdx != bandIdx) {
        GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
        actualCalcS1Token = static_cast<int64_t>(INT32_MAX) + actualS1Len - actualS2Len;
        actualCalcS2Token = static_cast<int64_t>(0) - actualS1Len + actualS2Len;
    } else if (sparseMode == 8 && bIdx != bandIdx) {
        actualCalcS1Token = INT32_MAX;
        actualCalcS2Token = 0;
    } else if (sparseMode == 3 || sparseMode == 4 || (sparseMode == 7 && bIdx == bandIdx) ||
               (sparseMode == 8 && bIdx == bandIdx)) {
        GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
        actualCalcS1Token = s1Token + actualS1Len - actualS2Len;
        actualCalcS2Token = s2Token - actualS1Len + actualS2Len;
    }
}


template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline bool
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::IsCubeBlockNeedCompute(int64_t baseIdx)
{
    if constexpr (INPUT_LAYOUT == TND) {
        // 安全防护，baseIdx不可小于0，防止gDimTail、s2oCvDimIdx等出现除0
        int64_t resbaseIdx = baseIdx < 0 ? 0 : baseIdx;
        int64_t actualS1Len;
        int64_t actualS2Len;
        for (int64_t bIdx = 0; bIdx < b; bIdx++) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            int32_t s1OuterTmp = (actualS1Len + s1CvInner - 1) / s1CvInner;
            int32_t s2OuterTmp = (actualS2Len + s2Inner * s2CvRatio - 1) / (s2Inner * s2CvRatio);
            int64_t totalBaseIdx = static_cast<int64_t>(n2) * g * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                int32_t gDimTail = resbaseIdx % (s1OuterTmp * s2OuterTmp);
                int32_t s2oCvDimIdx = gDimTail / s1OuterTmp;
                int32_t s1oDimIdx = gDimTail % s1OuterTmp;
                int32_t s2IdxLeft = s2oCvDimIdx * s2Inner * s2CvRatio;
                int32_t s2IdxRight = (s2oCvDimIdx + 1) * s2Inner * s2CvRatio
                                        < actualS2Len
                                        ? (s2oCvDimIdx + 1) * s2Inner * s2CvRatio
                                        : actualS2Len;

                // 6: prefix压缩，unpad只支持prefix压缩，不支持prefix
                if (sparseMode == 6) {
                    return CheckIsValidBlock(baseIdx, s1oDimIdx, s2oCvDimIdx, bIdx);
                }

                UpdateToken(bIdx);

                int32_t s2SparseLeft = int64_t(s1CvInner * s1oDimIdx) - actualCalcS1Token < 0 ?
                                           0 :
                                           s1CvInner * s1oDimIdx - actualCalcS1Token;
                s2SparseLeft = s2SparseLeft / 64 * 64;
                int32_t s2SparseRight = (s1CvInner * (s1oDimIdx + 1) + actualCalcS2Token + 63) / 64 * 64 < 0 ?
                                            0 :
                                            (s1CvInner * (s1oDimIdx + 1) + actualCalcS2Token + 63) / 64 * 64;
                s2SparseRight = static_cast<uint32_t>(s2SparseRight)
                                    < actualS2Len ? s2SparseRight : actualS2Len;
                if (s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft) {
                    nextS2CvBegin = s2IdxLeft < s2SparseLeft ? s2SparseLeft : s2IdxLeft;
                    nextS2CvEnd = s2IdxRight > s2SparseRight ? s2SparseRight : s2IdxRight;
                    return true;
                } else {
                    return false;
                }
            } else {
                resbaseIdx -= totalBaseIdx;
            }
        }
        return false;
    } else {
        uint32_t gDimTail = baseIdx % s1os2o;
        uint32_t s2oCvDimIdx = gDimTail / s1Outer;
        uint32_t s1oDimIdx = gDimTail % s1Outer;
        uint32_t s2IdxLeft = s2oCvDimIdx * s2Inner * s2CvRatio;
        uint32_t s2IdxRight =
            (s2oCvDimIdx + 1) * s2Inner * s2CvRatio < s2 ? (s2oCvDimIdx + 1) * s2Inner * s2CvRatio : s2;
        if (!isSparse) {
            nextS2CvBegin = s2IdxLeft;
            nextS2CvEnd = s2IdxRight;
            return true;
        }

        if (sparseMode == 5 || sparseMode == 6) {
            uint32_t curBIdx = baseIdx / n2gs1os2o;
            return CheckIsValidBlock(baseIdx, s1oDimIdx, s2oCvDimIdx, curBIdx);
        } else {
            uint32_t s2SparseLeft =
                int64_t(s1CvInner * s1oDimIdx) - actualCalcS1Token < 0 ? 0 : s1CvInner * s1oDimIdx - actualCalcS1Token;
            s2SparseLeft = s2SparseLeft / 64 * 64;
            uint32_t s2SparseRight = (s1CvInner * (s1oDimIdx + 1) + actualCalcS2Token + 63) / 64 * 64 < 0 ?
                                         0 :
                                         (s1CvInner * (s1oDimIdx + 1) + actualCalcS2Token + 63) / 64 * 64;
            s2SparseRight = s2SparseRight < s2 ? s2SparseRight : s2;
            if (s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft) {
                nextS2CvBegin = s2IdxLeft < s2SparseLeft ? s2SparseLeft : s2IdxLeft;
                nextS2CvEnd = s2IdxRight > s2SparseRight ? s2SparseRight : s2IdxRight;
                return true;
            } else {
                return false;
            }
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::CalcAttenMaskOffset(int64_t &attenMaskOffset, const int64_t delta,
                                                                         uint32_t s1VSize, uint32_t s2VSize)
{
    if (delta == 0) {
        attenMaskOffset = 0;
    } else if (delta < 0) {
        if (-delta > s1VSize) {
            attenMaskOffset = s1VSize;
        } else {
            attenMaskOffset = -delta;
        }
    } else {
        if (delta > s2VSize) {
            attenMaskOffset = s2VSize * attenMaskDimS2;
        } else {
            attenMaskOffset = delta * attenMaskDimS2;
        }
    }
}


template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::CalcAttenMaskOffsetForPrefixCompressMode(int64_t &attenMaskOffset,
                                                                                              int64_t &attenMaskOffset2,
                                                                                              const int64_t delta,
                                                                                              uint32_t s1VSize,
                                                                                              uint32_t s2VSize,
                                                                                              uint32_t s2VBegin,
                                                                                              bool &canSimplify)
{
    /*
      prefix压缩attenmask形状:
      ||
      ||||
      ||||||            Causal
      ||||||||
      ||||              All Mask
      ||||

      s1 + N <= S2，等效于RightDownCausal
      S1 + N > S2 场景
      先推出映射在压缩Prefix下三角部分的Mask(Mask1)的偏移
      再推出映射在压缩Prefix矩形部分的Mask(Mask2)的偏移
      如果整个vector基本块在N范围内，则直接使用Mask2
    */

    canSimplify = false;

    int64_t S1 = static_cast<int64_t>(s1);
    int64_t S2 = static_cast<int64_t>(s2);
    uint32_t curBatchDimIdx = bDimIdx;
    if constexpr (INPUT_LAYOUT == TND) {
        curBatchDimIdx = bDimIdxTmp;
        GetSeqQlenKvlenByBidx(curBatchDimIdx, S1, S2);
    }

    int64_t N = ((__gm__ int64_t *)prefixN_addr)[curBatchDimIdx];

    // s1 + N <= s2, equivalent to RightDownCausal
    if (S1 + N <= S2) {
        canSimplify = true;
        int64_t causal_delta = delta - S1 + S2;
        CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
        return;
    }

    int64_t delta1 = delta - S1 + S2;
    int64_t delta2 = N + 1 - static_cast<int64_t>(s2VBegin);

    // Y + n <= N, return mask2 offset directly
    if (delta2 > static_cast<int64_t>(s2VSize)) {
        canSimplify = true;
        attenMaskOffset = PREFIX_COMPRESS_CAUSAL_S_SIZE * attenMaskDimS2;
        return;
    }

    // other, mask = mask1 & mask2, need calculate two mask offsets
    // mask1 part
    if (delta1 >= 0) {
        attenMaskOffset = (delta1 <= s2VSize) ? delta1 * static_cast<int64_t>(attenMaskDimS2) :
                                                s2VSize * static_cast<int64_t>(attenMaskDimS2);
    } else {
        attenMaskOffset = (-delta1 <= s1VSize) ? -delta1 : s1VSize;
    }

    // mask2 part
    int64_t offsetStartPos =
        (int64_t)PREFIX_COMPRESS_CAUSAL_S_SIZE * (int64_t)attenMaskDimS2 + (int64_t)PREFIX_COMPRESS_ALL_MASK_S1_SIZE;
    attenMaskOffset2 = (delta2 > 0) ? (offsetStartPos - delta2 + 1) : offsetStartPos;
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<
    T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
    MM2_OUT_FORMAT>::CalcAttenMaskOffsetWithSparseModeForUnpad(int64_t &attenMaskOffset, int64_t &attenMaskOffset2,
                                                               uint32_t s1VSize, uint32_t s2VSize, int64_t curS1Idx,
                                                               uint32_t s2VBegin, bool unpadUseBand, bool &canSimplify)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    uint64_t compressMode = TilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode;
    int64_t causal_delta =
        static_cast<int64_t>(s1oDimIdxTmp * s1CvInner + curS1Idx * s1VecSize) - static_cast<int64_t>(s2VBegin);
    CalcAttenBandMode(compressMode, causal_delta);
    if (compressMode == 1 || (sparseMode == 8 && bDimIdxTmp != bandIdx)) { // causal s1==s2
        CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
        return;
    }

    if (compressMode == 2 || (sparseMode == 7 && bDimIdxTmp != bandIdx)) { // causal s1!=s2
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        causal_delta = causal_delta - actualS1Len + actualS2Len;
        CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
        return;
    }

    if (compressMode == 3 || unpadUseBand) { // band
        int64_t next_delta = causal_delta + actualCalcS2Token;
        CalcAttenMaskOffset(attenMaskOffset, next_delta, s1VSize, s2VSize);
        int64_t pre_delta = causal_delta - actualCalcS1Token - 1;
        CalcAttenMaskOffset(attenMaskOffset2, pre_delta, s1VSize, s2VSize);
        return;
    }

    if (compressMode == 4) { // 4: prefix compress
        CalcAttenMaskOffsetForPrefixCompressMode(attenMaskOffset, attenMaskOffset2, causal_delta, s1VSize, s2VSize,
                                                 s2VBegin, canSimplify);
        return;
    }

    if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 0) {
        attenMaskDimS2 = (uint32_t)s2;
        attenMaskOffset += (static_cast<int64_t>(s1oDimIdxTmp) * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
    } else if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 1) {
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        attenMaskDimS2 = (uint32_t)actualS2Len;
        for (uint32_t bidx = 0; bidx < bDimIdxTmp; bidx++) {
            GetSeqQlenKvlenByBidx(bidx, actualS1Len, actualS2Len);
            attenMaskOffset += actualS1Len * actualS2Len;
        }
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        attenMaskOffset += (static_cast<int64_t>(s1oDimIdxTmp) * s1CvInner + curS1Idx * s1VecSize) *
                           actualS2Len + s2VBegin;
    } else {
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        attenMaskDimS2 = (uint32_t)actualS2Len;
        for (uint32_t bidx = 0; bidx < bDimIdxTmp; bidx++) {
            GetSeqQlenKvlenByBidx(bidx, actualS1Len, actualS2Len);
            attenMaskOffset += static_cast<int64_t>(n2) * g * actualS1Len * actualS2Len;
        }
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        attenMaskOffset += ((static_cast<int64_t>(n2DimIdxTmp) * g + gDimIdxTmp) * actualS1Len +
                           s1oDimIdxTmp * s1CvInner + curS1Idx * s1VecSize) * actualS2Len + s2VBegin;
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::CalcAttenMaskOffsetWithSparseMode(int64_t &attenMaskOffset,
                                                                                       int64_t &attenMaskOffset2,
                                                                                       uint32_t s1VSize,
                                                                                       uint32_t s2VSize,
                                                                                       int64_t curS1Idx,
                                                                                       uint32_t s2VBegin,
                                                                                       bool &canSimplify)
{
    uint64_t compressMode = TilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode;
    int64_t causal_delta =
        static_cast<int64_t>(s1oDimIdx * s1CvInner + curS1Idx * s1VecSize) - static_cast<int64_t>(s2VBegin);
    CalcAttenBandMode(compressMode, causal_delta);
    if (compressMode == 1) { // 1: LeftUpCausal
        // causal s1==s2
        CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
        return;
    }

    if (compressMode == 2) { // 2: RightDownCausal
        // causal s1!=s2
        causal_delta = causal_delta - s1 + s2;
        CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
        return;
    }

    if (compressMode == 3) { // 3: band
        int64_t pre_delta = causal_delta - actualCalcS1Token - 1;
        CalcAttenMaskOffset(attenMaskOffset2, pre_delta, s1VSize, s2VSize);
        int64_t next_delta = causal_delta + actualCalcS2Token;
        CalcAttenMaskOffset(attenMaskOffset, next_delta, s1VSize, s2VSize);
        return;
    }

    if (compressMode == 4) { // 4: prefix compress
        CalcAttenMaskOffsetForPrefixCompressMode(attenMaskOffset, attenMaskOffset2, causal_delta, s1VSize, s2VSize,
                                                 s2VBegin, canSimplify);
        return;
    }

    if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 0) {
        attenMaskOffset = (static_cast<int64_t>(s1oDimIdx) * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
    } else if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 1) {
        attenMaskOffset =
            (static_cast<int64_t>(bDimIdx) * s1 + s1oDimIdx * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
    } else {
        attenMaskOffset = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * s1 +
                           s1oDimIdx * s1CvInner + curS1Idx * s1VecSize) *
                              s2 +
                          s2VBegin;
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::NZCopyIn(int64_t mmAddr, GlobalTensor<T2> &mmWspGm,
                                                              LocalTensor<T2> &mmTensorCurr, uint32_t s1VecSize,
                                                              uint32_t s2VecSize)
{
    /*
    Func:
    MM输出NZ数据，数据搬运进UB
    */
    DataCopyParams intriParams;
    intriParams.blockCount = s2VecSize / C0_SIZE;
    intriParams.blockLen = s1VecSize * C0_SIZE / cal_block_num;
    intriParams.srcStride = s1CvExtend * C0_SIZE / cal_block_num - intriParams.blockLen;
    intriParams.dstStride = 1;
    DataCopy(mmTensorCurr, mmWspGm[mmAddr], intriParams);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::NZ2ND(LocalTensor<T2> &mmTensorCurr, LocalTensor<T2> &tmpTensor,
                                                           uint32_t s1VecSize, uint32_t s2VecSize)
{
    /*
    Func:
    将NZ转为ND
    */
    CopyRepeatParams nz2ndParams;
    nz2ndParams.srcStride = s1VecSize * C0_SIZE / cal_block_num + 1;
    nz2ndParams.dstStride = C0_SIZE / cal_block_num;
    nz2ndParams.srcRepeatSize = C0_SIZE / cal_block_num;
    nz2ndParams.dstRepeatSize = s2VecSize / cal_block_num;

    uint16_t c0_repeat = C0_SIZE / cal_block_num;
    uint16_t c1_repeat = s2VecSize / C0_SIZE / VEC_REPEAT;
    uint16_t c1_remain = s2VecSize / C0_SIZE % VEC_REPEAT;
    uint16_t n_repeat = s1VecSize;
    for (uint16_t i = 0; i < c0_repeat; ++i) {
        for (uint16_t j = 0; j < c1_repeat; ++j) {
            Copy(mmTensorCurr[i * cal_block_num + j * VEC_REPEAT * C0_SIZE],
                 tmpTensor[i * cal_block_num + j * VEC_REPEAT * (s1VecSize * C0_SIZE + cal_block_num)],
                 VEC_REPEAT * cal_block_num, n_repeat, nz2ndParams);
        }
        if (c1_remain > 0) {
            Copy(mmTensorCurr[i * cal_block_num + c1_repeat * VEC_REPEAT * C0_SIZE],
                 tmpTensor[i * cal_block_num + c1_repeat * VEC_REPEAT * (s1VecSize * C0_SIZE + cal_block_num)],
                 VEC_REPEAT * c1_remain, n_repeat, nz2ndParams);
        }
    }
    pipe_barrier(PIPE_V);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::ND2NZ(LocalTensor<T1> &mmTensorCurr, LocalTensor<T1> &tmpTensor,
                                                           uint32_t s1VecSize, uint32_t s2VecSize)
{
    /*
    Func:
    将ND转为NZ
    */

    CopyRepeatParams nd2nzParams;
    nd2nzParams.dstStride = s1VecSize * C0_SIZE / input_block_num + 1;
    nd2nzParams.srcStride = C0_SIZE / input_block_num;
    nd2nzParams.dstRepeatSize = C0_SIZE / input_block_num;
    nd2nzParams.srcRepeatSize = s2VecSize / input_block_num;

    uint16_t c1_repeat = s2VecSize / C0_SIZE / VEC_REPEAT;
    uint16_t c1_remain = s2VecSize / C0_SIZE % VEC_REPEAT;

    auto mmTensorCurrTmp = mmTensorCurr.template ReinterpretCast<half>();
    auto tmpTensorTmp = tmpTensor.template ReinterpretCast<half>();

    for (uint16_t j = 0; j < c1_repeat; ++j) {
        Copy(mmTensorCurrTmp[j * 8 * (s1VecSize + 1) * C0_SIZE], tmpTensorTmp[j * 128], VEC_REPEAT * input_block_num,
             s1VecSize, nd2nzParams);
    }

    if (c1_remain > 0) {
        Copy(mmTensorCurrTmp[c1_repeat * 8 * (s1VecSize + 1) * C0_SIZE], tmpTensorTmp[c1_repeat * 128],
             input_block_num * c1_remain, s1VecSize, nd2nzParams);
    }
}


template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT>::InitIndex(int64_t index)
{
    if constexpr (INPUT_LAYOUT == TND) {
        // 安全防护，index不可小于0，防止n2DimIdx、gDimIdx等出现除0
        int64_t resbaseIdx = index < 0 ? 0 : index;
        int64_t actualS1Len;
        int64_t actualS2Len;
        for (uint32_t bIdx = 0; bIdx < b; bIdx++) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            uint32_t s1OuterTmp = (actualS1Len + s1CvInner - 1) / s1CvInner;
            uint32_t s2OuterTmp = (actualS2Len + s2Inner * s2CvRatio - 1)
                / (s2Inner * s2CvRatio);
            int64_t totalBaseIdx = static_cast<int64_t>(n2) * g * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                uint32_t s1CvTailTmp = actualS1Len - (s1OuterTmp - 1) * s1CvInner;
                bDimIdx = bIdx;
                uint32_t bDimTail = resbaseIdx;
                n2DimIdx = bDimTail / (g * s1OuterTmp * s2OuterTmp);
                uint32_t n2DimTail = bDimTail % (g * s1OuterTmp * s2OuterTmp);
                gDimIdx = n2DimTail / (s1OuterTmp * s2OuterTmp);
                uint32_t gDimTail = n2DimTail % (s1OuterTmp * s2OuterTmp);
                s2oCvDimIdx = gDimTail / s1OuterTmp;
                s1oDimIdx = gDimTail % s1OuterTmp;
                s1CvExtend = (s1oDimIdx == s1OuterTmp - 1) ? s1CvTailTmp : s1CvInner;
                break;
            } else {
                resbaseIdx -= totalBaseIdx;
            }
        }
    } else {
        baseIdx = index;
        bDimIdx = baseIdx / n2gs1os2o;
        uint32_t bDimTail = baseIdx % n2gs1os2o;
        n2DimIdx = bDimTail / gs1os2o;
        uint32_t n2DimTail = bDimTail % gs1os2o;
        gDimIdx = n2DimTail / s1os2o;
        uint32_t gDimTail = n2DimTail % s1os2o;
        s2oCvDimIdx = gDimTail / s1Outer;
        s1oDimIdx = gDimTail % s1Outer;
        s1CvExtend = (s1oDimIdx == s1Outer - 1) ? s1CvTail : s1CvInner;
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::CalcAttenBandMode(int64_t compressMode, int64_t causal_delta)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    if (compressMode == 1 || compressMode == 2 || compressMode == 3 || sparseMode == 7 || sparseMode == 8) { // compress
        int64_t next_delta = causal_delta;
        int64_t pre_delta = causal_delta - INT32_MAX - 1;
        if (compressMode == 1 || (sparseMode == 8 && bDimIdxTmp != bandIdx)) {
        } else if (compressMode == 2) {
            if constexpr (INPUT_LAYOUT == TND) {
                GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
                next_delta = causal_delta - actualS1Len + actualS2Len;
            } else {
                next_delta = causal_delta - s1 + s2;
            }
        } else if (sparseMode == 7 && bDimIdxTmp != bandIdx) {
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
            next_delta = causal_delta - actualS1Len + actualS2Len;
        } else {
            next_delta = causal_delta + actualCalcS2Token;
            pre_delta = causal_delta - actualCalcS1Token - 1;
        }

        bool NoNext = (next_delta - s2Extend >= 0);
        bool NoPre = (pre_delta + 1 + s1ExtendSubGraph <= 0);

        if (NoNext && NoPre) {
            AttenBandMode = AttenMaskCompress::Empty;
        } else if (NoNext && !NoPre) {
            AttenBandMode = AttenMaskCompress::PreOnly;
        } else if (!NoNext && NoPre) {
            AttenBandMode = AttenMaskCompress::NextOnly;
        } else {
            AttenBandMode = AttenMaskCompress::All;
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::DropOutCopy(LocalTensor<uint8_t> &vecInDropBuffer,
                                                                 int64_t curS1Idx, int64_t s2VBegin)
{
    // for compute dropout mask offset
    dropMaskInfo.s2StartIdx = s2VBegin;
    // for copy in dropout mask
    dropMaskInfo.s2CopySize = s2Extend;

    CopyInDropMask<true>(vecInDropBuffer, maskWorkSpaceGm, dropMaskGm, this->dropMaskInfo);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::SubGrapA(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    s2Extend = (curS2Idx == s2VecLoop - 1) ? (s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
    s2ExtendAlign = (s2Extend + 15) / 16 * 16;
    uint32_t s2VBegin = preS2CvBegin + curS2Idx * s2VecSize;

    event_t curEventId = ((curIdx % 2) == 0) ? EVENT_ID7 : EVENT_ID6;
    uint32_t ubBufferOffset = ((curIdx % 2) == 0) ? 0 : DbBegin;

    if (curIdx > 1) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, curEventId);
    }

    LocalTensor<float> vecInBuffer3 =
        ubBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), ubBufferOffset + T2BlockBegin);
    int64_t softMaxOffset = 0;
    if constexpr (INPUT_LAYOUT == TND) {
        if (bDimIdxTmp > 0) {
            softMaxOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[bDimIdxTmp - 1] * n2 * g * 32 / sizeof(float);
        }
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        softMaxOffset += ((n2DimIdxTmp * g + gDimIdxTmp) * actualS1Len +
                         s1oDimIdxTmp * s1CvInner + curS1Idx * s1VecSize) * 32 / sizeof(float);
    } else {
        softMaxOffset = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1CvInner +
                         curS1Idx * s1VecSize) * 32 / sizeof(float);
    }
    CopyInSoftMax(vecInBuffer3, s1ExtendSubGraph, softMaxOffset);

    LocalTensor<T1> pseUbT1 = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
    LocalTensor<half> pseUb = pseUbT1.template ReinterpretCast<half>();
    if constexpr (IS_PSE == ENABLE) {
        pseInfo.bSSOffset = bDimIdx * s1 * s2;
        pseInfo.s2SizeAcc = bDimIdx * s2;
        pseInfo.boIdx = bDimIdx;
        pseInfo.n2oIdx = n2DimIdx;
        pseInfo.goIdx = gDimIdx;
        pseInfo.s1oIdx = s1oDimIdx;
        pseInfo.loopIdx = curS1Idx;
        pseInfo.vec1S1BaseSize = s1VecSize;
        pseInfo.vec1S1RealSize = s1ExtendSubGraph;
        pseInfo.s1BaseSize = s1CvInner;
        pseInfo.s2RealSize = s2Extend;
        pseInfo.s2AlignedSize = s2ExtendAlign;
        pseInfo.s2StartIdx = s2VBegin;
        LocalTensor<T2> noCastedPseUb = ubBuffer.GetWithOffset<T2>(0 / sizeof(T2), 0);
        bool innerAlibiFlag = false; // alibi核内生成相关配置，仅在LAYOUT=TND，SparseMode=8时生效
        if constexpr (INPUT_LAYOUT == TND) {
            if (sparseMode == 8 && pseInfo.boIdx != 0) {
                innerAlibiFlag = true;
            }
        }

        if (pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            if (innerAlibiFlag) {
                pseInfo.kvStartIdx = 0;
                pseInfo.qStartIdx = 0;
            }
            PseSlopeCopyIn<T2, true>(noCastedPseUb, pseUb, pseSlope, this->pseAlibiGm, pseInfo);
        } else {
            if constexpr (!IsSameType<T1, float>::value) {
                if constexpr (INPUT_LAYOUT == TND) {
                    PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_TND, true>(noCastedPseUb, pseUbT1, this->pseGm, pseInfo);
                } else {
                    PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_BNSD, true>(noCastedPseUb, pseUbT1, this->pseGm, pseInfo);
                }
            }
        }
    }

    LocalTensor<uint8_t> attenMaskUbuint8 =
        ubBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + BoolBegin);
    bool unpadUseBand = (sparseMode == 7 && bDimIdxTmp == bandIdx) || (sparseMode == 8 && bDimIdxTmp == bandIdx);
    int64_t attenMaskOffsetPre = 0;
    bool prefixCompressCanSimplify = false;
    if constexpr (IS_ATTEN_MASK == ENABLE) {
        int64_t attenMaskOffset = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            CalcAttenMaskOffsetWithSparseModeForUnpad(attenMaskOffset, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend,
                                                      curS1Idx, s2VBegin, unpadUseBand, prefixCompressCanSimplify);
        } else {
            CalcAttenMaskOffsetWithSparseMode(attenMaskOffset, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend, curS1Idx,
                                              s2VBegin, prefixCompressCanSimplify);
        }
        // uint8_t
        if (AttenBandMode == AttenMaskCompress::All || AttenBandMode == AttenMaskCompress::NextOnly) {
            CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffset, s1ExtendSubGraph, s2Extend);
        } else if (AttenBandMode == AttenMaskCompress::PreOnly) {
            CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);
        }
    }

    LocalTensor<uint8_t> vecInDropBuffer =
        ubBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + U8Begin);
    if constexpr (IS_DROP == ENABLE) {
        if constexpr (IsSameType<T1, float>::value) {
            pipe_barrier(PIPE_ALL);
        }
        DropOutCopy(vecInDropBuffer, curS1Idx, s2VBegin);
    }

    LocalTensor<float> vecClc2Buffer =
        ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);
    if constexpr (MM_OUT_FORMAT == CubeFormat::ND) {
        if (s2VecLoop == 1) {
            DataCopy(vecClc2Buffer, mm2WorkspaceGm[curS1Idx * s1VecSize * s2ExtendAlign],
                     s1ExtendSubGraph * s2ExtendAlign);
        } else {
            DataCopyPad(vecClc2Buffer, mm2WorkspaceGm[curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                        {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                         static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                        {false, 0, 0, 0});
        }
        set_flag(PIPE_MTE2, PIPE_V, curEventId);
        wait_flag(PIPE_MTE2, PIPE_V, curEventId);
    } else {
        int64_t mmAddr = curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtend * s2VecSizeAlign;
        NZCopyIn(mmAddr, mm2WorkspaceGm, vecClc2Buffer, s1VecSize, s2ExtendAlign);
        set_flag(PIPE_MTE2, PIPE_V, curEventId);
        wait_flag(PIPE_MTE2, PIPE_V, curEventId);
        auto tmpTensor = tmpBuffer.Get<T2>();
        DataCopy(tmpTensor, vecClc2Buffer, s1VecSize * s2ExtendAlign + s2ExtendAlign / C0_SIZE * VEC_REPEAT);
        pipe_barrier(PIPE_V);
        NZ2ND(vecClc2Buffer, tmpTensor, s1VecSize, s2ExtendAlign);
    }

    ///////////////////////////////////////////////////////////////
    // pse + muls
    ///////////////////////////////////////////////////////////////
    // pse shape  0--BN2G1S2    1--BN2GS1S2
    if constexpr (IS_PSE == ENABLE) {
        if (TilingData->s1s2BNGS1S2BaseParams.pseType != (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
        pipe_barrier(PIPE_V);
        Muls(vecClc2Buffer, vecClc2Buffer, (T2)(TilingData->s1s2BNGS1S2BaseParams.scaleValue),
            s1ExtendSubGraph * s2ExtendAlign);
        }
        uint16_t repeatTimes = static_cast<uint16_t>(s1ExtendSubGraph);
        if (TilingData->s1s2BNGS1S2BaseParams.pseShapeType == 1) {
            repeatTimes = 1;
        }
        LocalTensor<T2> castTensor = tmpBuffer.Get<T2>();
        if (!(pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE)) {

            if constexpr (!IsSameType<T1, float>::value) {
                uint32_t calculateRowsAlign = (s2Extend + input_block_num - 1) / input_block_num * input_block_num;
                Cast(castTensor, pseUbT1, RoundMode::CAST_NONE, repeatTimes * calculateRowsAlign);
                pipe_barrier(PIPE_V);
            } else {
                set_flag(PIPE_V, PIPE_MTE2, curEventId);
                wait_flag(PIPE_V, PIPE_MTE2, curEventId);
                if constexpr (INPUT_LAYOUT == TND) {
                    PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_TND, true>(castTensor, castTensor, this->pseGm, pseInfo);
                } else {
                    PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_BNSD, true>(castTensor, castTensor, this->pseGm, pseInfo);
                }
                set_flag(PIPE_MTE2, PIPE_V, curEventId);
                wait_flag(PIPE_MTE2, PIPE_V, curEventId);
            }
        } else {
            PseSlopeCast<T2, true>(castTensor, pseUb, pseSlope, pseInfo);
        }
        pipe_barrier(PIPE_V);
        PseCompute<T2, true>(vecClc2Buffer, castTensor, pseInfo);
        pipe_barrier(PIPE_V);
    }
    if (TilingData->s1s2BNGS1S2BaseParams.pseType == (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
        pipe_barrier(PIPE_V);
        Muls(vecClc2Buffer, vecClc2Buffer, (T2)(TilingData->s1s2BNGS1S2BaseParams.scaleValue),
            s1ExtendSubGraph * s2ExtendAlign);
    }
    ///////////////////////////////////////////////////////////////
    // attenMask
    ///////////////////////////////////////////////////////////////
    // attenMaskOffset     attenMaskShapeType  0--111S1S2        1--B11S1S2         2--BN2GS1S2
    if constexpr (IS_ATTEN_MASK == ENABLE) {
        int64_t compressMode = TilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode;
        pipe_barrier(PIPE_V);

        if (compressMode == 4) { // 4: prefix compress
            if (prefixCompressCanSimplify == false) {
                uint32_t ubTmpBufferOffset = ((curIdx % 2) == 0) ? 0 : hufTmpBuffBegin;
                LocalTensor<uint8_t> attenMaskUbPreuint8 =
                    tmpBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubTmpBufferOffset);
                uint32_t s2ExtendPadAlign = (s2Extend + 31) / 32 * 32; // attenmask做pad时会32对齐，故加31/32做ceil
                int32_t maskNum = s1ExtendSubGraph * s2ExtendPadAlign / 2; // 除2数据量按照uint16类型折半

                set_flag(PIPE_V, PIPE_MTE2, curEventId);
                wait_flag(PIPE_V, PIPE_MTE2, curEventId);
                CopyInAttenMaskBool(attenMaskUbPreuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);

                set_flag(PIPE_MTE2, PIPE_V, curEventId);
                wait_flag(PIPE_MTE2, PIPE_V, curEventId);
                auto attenMaskUbuint8Tmp = attenMaskUbuint8.ReinterpretCast<uint16_t>();
                auto attenMaskUbPreuint8Tmp = attenMaskUbPreuint8.ReinterpretCast<uint16_t>();
                And(attenMaskUbuint8Tmp, attenMaskUbPreuint8Tmp, attenMaskUbuint8Tmp, maskNum);
                pipe_barrier(PIPE_V);
                attenMaskUbuint8 = attenMaskUbuint8Tmp.ReinterpretCast<uint8_t>();
            }
        }

        // uint8_t
        if (AttenBandMode == AttenMaskCompress::All || AttenBandMode == AttenMaskCompress::NextOnly) {
            CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign);
        } else if (AttenBandMode == AttenMaskCompress::PreOnly) {
            CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign, 1);
        }

        if ((compressMode == 3 || unpadUseBand) && AttenBandMode == AttenMaskCompress::All) { // 3: band
            set_flag(PIPE_V, PIPE_MTE2, curEventId);
            wait_flag(PIPE_V, PIPE_MTE2, curEventId);
            CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);
            set_flag(PIPE_MTE2, PIPE_V, curEventId);
            wait_flag(PIPE_MTE2, PIPE_V, curEventId);
            CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign, 1);
        }
    }

    ///////////////////////////////////////////////////////////////
    // simpleSoftMax
    ///////////////////////////////////////////////////////////////
    pipe_barrier(PIPE_V);
    CalcSoftMax(vecClc2Buffer, vecInBuffer3, s1ExtendSubGraph, s2Extend, s2ExtendAlign, TilingData->softmaxTilingData);

    ///////////////////////////////////////////////////////////////
    // dropout
    ///////////////////////////////////////////////////////////////
    LocalTensor<T2> vecDropBuffer = vecClc2Buffer;
    if constexpr (IS_DROP == ENABLE) {
        vecDropBuffer = tmpBuffer.GetWithOffset<T2>(32 * 1024 / sizeof(T2), 0);
        pipe_barrier(PIPE_V);
        LocalTensor<uint8_t> tmpDropBuffer =
            ubBuffer.GetWithOffset<uint8_t>(32 * 1024 / sizeof(uint8_t), ubBufferOffset + T1Begin);

        // for compute dropout mask
        dropMaskInfo.lstAxis = s2ExtendAlign;
        dropMaskInfo.maskLstAxis = s2ExtendAlign;
        ComputeDropMask<T2, true>(vecDropBuffer, vecClc2Buffer, vecInDropBuffer, tmpDropBuffer, this->dropMaskInfo);
        if constexpr (IsSameType<T1, float>::value) {
            pipe_barrier(PIPE_ALL);
        }
    }

    ///////////////////////////////////////////////////////////////
    // cast fp322bf16
    ///////////////////////////////////////////////////////////////
    LocalTensor<T1> vecOut1Buffer1;
    if constexpr (!IsSameType<T1, float>::value) {
        vecOut1Buffer1 = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
        pipe_barrier(PIPE_V);
        Cast(vecOut1Buffer1, vecDropBuffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
    }
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        pipe_barrier(PIPE_V);
        LocalTensor<T1> tmpTensor = tmpBuffer.Get<T1>();
        if constexpr (!IsSameType<T1, float>::value) {
            DataCopy(tmpTensor, vecOut1Buffer1, s1ExtendSubGraph * s2ExtendAlign);
            pipe_barrier(PIPE_V);
            ND2NZ(vecOut1Buffer1, tmpTensor, s1ExtendSubGraph, s2ExtendAlign);

            set_flag(PIPE_V, PIPE_MTE3, curEventId);
            wait_flag(PIPE_V, PIPE_MTE3, curEventId);
            DataCopyPad(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                        curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtendAlign * s2VecSize],
                        vecOut1Buffer1,
                        {static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                        static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)), 1,
                        static_cast<uint16_t>((s1CvExtendAlign - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))});
        } else {
            DataCopy(tmpTensor, vecDropBuffer, s1ExtendSubGraph * s2ExtendAlign);
            pipe_barrier(PIPE_V);
            ND2NZ(vecDropBuffer, tmpTensor, s1ExtendSubGraph, s2ExtendAlign);

            set_flag(PIPE_V, PIPE_MTE3, curEventId);
            wait_flag(PIPE_V, PIPE_MTE3, curEventId);
            DataCopyPad(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                        curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtendAlign * s2VecSize],
                        vecDropBuffer,
                        {static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                        static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)), 1,
                        static_cast<uint16_t>((s1CvExtendAlign - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))});
        }
    } else {
        set_flag(PIPE_V, PIPE_MTE3, curEventId);
        wait_flag(PIPE_V, PIPE_MTE3, curEventId);
        if constexpr (!IsSameType<T1, float>::value) {
                DataCopyPad(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                    curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    vecOut1Buffer1,
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
        } else {
                DataCopyPad(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                    curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    vecDropBuffer,
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
        }
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, curEventId);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT>::SubGrapB(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx)
{
    event_t curEventId = ((curIdx % 2) == 0) ? EVENT_ID7 : EVENT_ID6;
    uint32_t ubBufferOffset = ((curIdx % 2) == 0) ? 0 : DbBegin;
    s2Extend = (curS2Idx == s2VecLoop - 1) ? (s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
    s2ExtendAlign = (s2Extend + 15) / 16 * 16;

    wait_flag(PIPE_MTE3, PIPE_MTE2, curEventId);

    LocalTensor<uint8_t> vecInDropBuffer =
        ubBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + U8Begin);
    if constexpr (IS_DROP == ENABLE) {
        int64_t s2VBegin = preS2CvBegin + curS2Idx * s2VecSize;
        DropOutCopy(vecInDropBuffer, curS1Idx, s2VBegin);
        if constexpr (IsSameType<T1, float>::value) {
            pipe_barrier(PIPE_ALL);
        }
    }

    LocalTensor<T2> vecClc1Buffer = ubBuffer.GetWithOffset<T2>(32 * 1024 / sizeof(T2), ubBufferOffset + T1Begin);
    if constexpr (MM_OUT_FORMAT == CubeFormat::ND) {
        if (s2VecLoop == 1) {
            DataCopy(vecClc1Buffer, mm1WorkspaceGm[curS1Idx * s1VecSize * s2ExtendAlign],
                     s1ExtendSubGraph * s2ExtendAlign);
        } else {
            DataCopyPad(vecClc1Buffer, mm1WorkspaceGm[curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                        {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                         static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                        {false, 0, 0, 0});
        }
        set_flag(PIPE_MTE2, PIPE_V, curEventId);
        wait_flag(PIPE_MTE2, PIPE_V, curEventId);
    } else {
        int64_t mmAddr = curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtend * s2VecSizeAlign;
        NZCopyIn(mmAddr, mm1WorkspaceGm, vecClc1Buffer, s1VecSize, s2ExtendAlign);
        set_flag(PIPE_MTE2, PIPE_V, curEventId);
        wait_flag(PIPE_MTE2, PIPE_V, curEventId);
        auto tmpTensor = tmpBuffer.Get<T2>();
        DataCopy(tmpTensor, vecClc1Buffer, s1VecSize * s2ExtendAlign + s2ExtendAlign / C0_SIZE * VEC_REPEAT);
        pipe_barrier(PIPE_V);
        NZ2ND(vecClc1Buffer, tmpTensor, s1VecSize, s2ExtendAlign);
    }

    ///////////////////////////////////////////////////////////////
    // ss
    ///////////////////////////////////////////////////////////////
    if constexpr (IS_DROP == ENABLE) {
        LocalTensor<uint8_t> tmpDropBuffer = tmpBuffer.GetWithOffset<uint8_t>(32 * 1024 / sizeof(uint8_t), 0);

        // for compute dropout mask
        dropMaskInfo.lstAxis = s2ExtendAlign;
        dropMaskInfo.maskLstAxis = s2ExtendAlign;
        ComputeDropMask<T2, true>(vecClc1Buffer, vecClc1Buffer, vecInDropBuffer, tmpDropBuffer, this->dropMaskInfo);
    }

    ///////////////////////////////////////////////////////////////
    // sub to improve
    ///////////////////////////////////////////////////////////////
    uint32_t sub_block_cout = (s2ExtendAlign + cal_repeat_num - 1) / cal_repeat_num;
    uint32_t sfmgStartIndex = s1CvRatio > 1 ? 0 : curS1Idx * s1VecSize * 8;
    LocalTensor<float> sfmgClc3 = vecClc3.Get<float>();
    pipe_barrier(PIPE_V);
    for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
        uint32_t subMaskCout =
            (subIdx == sub_block_cout - 1) ? (s2ExtendAlign - subIdx * cal_repeat_num) : cal_repeat_num;
        Sub(vecClc1Buffer[subIdx * cal_repeat_num], vecClc1Buffer[subIdx * cal_repeat_num], sfmgClc3[sfmgStartIndex],
            subMaskCout, s1ExtendSubGraph,
            {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
             static_cast<uint8_t>(s2ExtendAlign / 8), 1});
    }

    ///////////////////////////////////////////////////////////////
    // mul
    ///////////////////////////////////////////////////////////////
    pipe_barrier(PIPE_V);
    LocalTensor<float> vecClc2Buffer =
        ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);
    Mul(vecClc1Buffer, vecClc1Buffer, vecClc2Buffer, s1ExtendSubGraph * s2ExtendAlign);
    LocalTensor<T1> vecOutBuffer;
    if constexpr (!IsSameType<T1, float>::value) {
        vecOutBuffer = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
        pipe_barrier(PIPE_V);
        Cast(vecOutBuffer, vecClc1Buffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
    }
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        pipe_barrier(PIPE_V);
        auto tmpTensor1 = tmpBuffer.Get<T1>();
        if constexpr (IsSameType<T1, float>::value) {
            DataCopy(tmpTensor1, vecClc1Buffer, s1ExtendSubGraph * s2ExtendAlign);
            pipe_barrier(PIPE_V);
            ND2NZ(vecClc1Buffer, tmpTensor1, s1ExtendSubGraph, s2ExtendAlign);
        } else {
            DataCopy(tmpTensor1, vecOutBuffer, s1ExtendSubGraph * s2ExtendAlign);
            pipe_barrier(PIPE_V);
            ND2NZ(vecOutBuffer, tmpTensor1, s1ExtendSubGraph, s2ExtendAlign);
        }

        set_flag(PIPE_V, PIPE_MTE3, curEventId);
        wait_flag(PIPE_V, PIPE_MTE3, curEventId);

        if constexpr(IsSameType<T1, float>::value){
            DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                   curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtendAlign * s2VecSize],
            vecClc1Buffer,
            {static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)), 1,
            static_cast<uint16_t>((s1CvExtendAlign - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))});
        } else {
            DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                   curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtendAlign * s2VecSize],
            vecOutBuffer,
            {static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)), 1,
            static_cast<uint16_t>((s1CvExtendAlign - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))});
        }
    } else {
        set_flag(PIPE_V, PIPE_MTE3, curEventId);
        wait_flag(PIPE_V, PIPE_MTE3, curEventId);

        if constexpr(IsSameType<T1, float>::value) {
               DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                   curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    vecClc1Buffer,
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
        } else {
                  DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                   curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    vecOutBuffer,
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
        }
    }

    if ((s1VecLoop * s2VecLoop > 2) && (curIdx < (s1VecLoop * s2VecLoop - 2))) {
        set_flag(PIPE_MTE3, PIPE_MTE2, curEventId);
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT>::Compute(int64_t preIndex,
                                                                                                  int64_t nextIndex)
{
    pingpongIdx = 1 - pingpongIdx;
    if (isStart == 1) {
        InitIndex(preIndex);
    }
    bDimIdxTmp = bDimIdx;
    n2DimIdxTmp = n2DimIdx;
    gDimIdxTmp = gDimIdx;
    s1oDimIdxTmp = s1oDimIdx;
    int64_t mm1aTensorOffsetCv = 0;
    int64_t mm2aTensorOffsetCv = 0;
    int64_t bTensorOffsetCv = 0;
    s2CvExtend = preS2CvEnd - preS2CvBegin;
    s2CvExtendAlign = (s2CvExtend + 15) / 16 * 16;
    s1CvExtendAlign = (s1CvExtend + 15) / 16 * 16;
    int64_t dqOffset = 0;
    int64_t dkvOffset = 0;
    int64_t s1StrideSize = 0;
    int64_t s2StrideSize = 0;
    int64_t dAlign = (d + 15) / 16 * 16;
    int64_t actualS1Len;
    int64_t actualS2Len;
    if constexpr (INPUT_LAYOUT == TND) {
        UpdateToken(bDimIdxTmp);
        if (bDimIdxTmp > 0) {
            mm1aTensorOffsetCv = ((__gm__ int64_t *)actual_seq_qlen_addr)[bDimIdxTmp - 1] * n2 * g * d;
            bTensorOffsetCv = ((__gm__ int64_t *)actual_seq_kvlen_addr)[bDimIdxTmp - 1] * n2 * d;
        }
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
            dqOffset = mm1aTensorOffsetCv / d * dAlign;
            dkvOffset = bTensorOffsetCv / d * dAlign;
            dqOffset += ((n2DimIdxTmp * g + gDimIdxTmp) * actualS1Len) * dAlign + s1oDimIdxTmp * s1CvInner * C0_SIZE;
            dkvOffset += (n2DimIdxTmp * actualS2Len) * dAlign + preS2CvBegin * C0_SIZE;
        }
        mm1aTensorOffsetCv +=
            ((static_cast<int64_t>(s1oDimIdxTmp) * s1CvInner * n2 + n2DimIdxTmp) * g + gDimIdxTmp) * d;
        bTensorOffsetCv += (preS2CvBegin * n2 + n2DimIdxTmp) * d;
        mm2aTensorOffsetCv = mm1aTensorOffsetCv;
        s1StrideSize = n2 * g * d;
        s2StrideSize = n2 * d;
    } else {
        if constexpr (INPUT_LAYOUT == BNGSD) {
            mm1aTensorOffsetCv =
                (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1CvInner) * d;
            mm2aTensorOffsetCv = mm1aTensorOffsetCv;
            bTensorOffsetCv = ((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * s2 + preS2CvBegin) * d;
            s1StrideSize = d;
            s2StrideSize = d;
        } else if constexpr (INPUT_LAYOUT == SBNGD) {
            mm1aTensorOffsetCv =
                ((((static_cast<int64_t>(s1oDimIdx) * s1CvInner) * b + bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * d;
            mm2aTensorOffsetCv = mm1aTensorOffsetCv;
            bTensorOffsetCv = ((static_cast<int64_t>(preS2CvBegin) * b + bDimIdx) * n2 + n2DimIdx) * d;
            s1StrideSize = static_cast<int64_t>(b) * n2 * g * d;
            s2StrideSize = static_cast<int64_t>(b) * n2 * d;
        } else if constexpr (INPUT_LAYOUT == BSNGD) {
            mm1aTensorOffsetCv =
                (((static_cast<int64_t>(bDimIdx) * s1 + s1oDimIdx * s1CvInner) * n2 + n2DimIdx) * g + gDimIdx) * d;
            mm2aTensorOffsetCv = mm1aTensorOffsetCv;
            bTensorOffsetCv = ((static_cast<int64_t>(bDimIdx) * s2 + preS2CvBegin) * n2 + n2DimIdx) * d;
            s1StrideSize = n2 * g * d;
            s2StrideSize = n2 * d;
        }
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            dqOffset = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * s1) * dAlign + s1oDimIdx * s1CvInner * C0_SIZE;
            dkvOffset = ((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * s2) * dAlign + preS2CvBegin * C0_SIZE;
        }
    }
    if constexpr (MM2_OUT_FORMAT == CubeFormat::ND) {
        dqOffset = mm1aTensorOffsetCv;
        dkvOffset = bTensorOffsetCv;
    }

    if (isStart == 1) {
        if constexpr (INPUT_LAYOUT == TND) {
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
            if (MM_OUT_FORMAT == CubeFormat::NZ) {
                mm1.SetOrgShape(s1CvExtend, s2CvExtend, s1StrideSize, s2StrideSize, s2CvExtendAlign);
            } else {
                mm1.SetOrgShape(actualS1Len, actualS2Len, s1StrideSize, s2StrideSize, s2CvExtendAlign);
            }
        } else {
            if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                mm1.SetOrgShape(s1CvExtend, s2, s1StrideSize, s2StrideSize, s2CvExtendAlign);
            } else {
                mm1.SetOrgShape(s1, s2, s1StrideSize, s2StrideSize, s2CvExtendAlign);
            }
        }
        mm1.SetTail(s1CvExtend, s2CvExtend, d); // M N K
        mm1.SetTensorA(dxGm[mm1aTensorOffsetCv]);
        mm1.SetTensorB(valueGm[bTensorOffsetCv], true);
        mm1.template IterateAll<false>(mm1WorkspaceGm, false, false, true);

        mm1.SetTail(s1CvExtend, s2CvExtend, d); // M N K
        mm1.SetTensorA(queryGm[mm2aTensorOffsetCv]);
        mm1.SetTensorB(keyGm[bTensorOffsetCv], true);
        mm1.template IterateAll<false>(mm2WorkspaceGm, false, false, true);

        isStart = 0;
    }

    s2VecSize = s2CvExtend > VEC_S2_LEN ? VEC_S2_LEN : s2CvExtend;
    s2VecSize = s2VecSize == 0 ? cal_block_num : s2VecSize;
    s2VecLoop = (s2CvExtend + s2VecSize - 1) / s2VecSize;

    if constexpr (IS_DROP == ENABLE) {
        // dropout last dim 32B align
        s2VecSizeAlign = (s2VecSize + 31) / 32 * 32;
    } else if constexpr (IS_ATTEN_MASK == ENABLE) {
        // attenmask last dim 32B align
        s2VecSizeAlign = (s2VecSize + 31) / 32 * 32;
    } else {
        s2VecSizeAlign = (s2VecSize + 15) / 16 * 16;
    }

    s1VecSize = baseMN / s2VecSizeAlign;
    s1VecSize = s1VecSize < s1CvExtend ? s1VecSize : s1CvExtend;
    s1VecSize = s1VecSize > 128 ? 128 : s1VecSize;
    s1VecLoop = (s1CvExtend + s1VecSize - 1) / s1VecSize;

    dropMaskInfo.splitS1BaseSize = s1VecSize;
    if constexpr (INPUT_LAYOUT == TND) {
        GetSeqQlenKvlenByBidx(bDimIdx, actualS1Len, actualS2Len);
        dropMaskInfo.s2TotalSize = actualS2Len;
        int64_t bSSOffset = 0;
        int64_t s2Accu = 0;
        for (int64_t bidx = 0; bidx < bDimIdxTmp; bidx++) {
            GetSeqQlenKvlenByBidx(bidx, actualS1Len, actualS2Len);
            bSSOffset += actualS1Len * actualS2Len;
            s2Accu += actualS2Len;
        }
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        dropMaskInfo.bSSOffset = bSSOffset;
        dropMaskInfo.s1Size = actualS1Len;
        dropMaskInfo.s2Size = actualS2Len;
        pseInfo.bSSOffset = bSSOffset;
        pseInfo.s2SizeAcc = s2Accu;
        pseInfo.s1Size = dropMaskInfo.s1Size;
        pseInfo.s2Size = dropMaskInfo.s2Size;
    } else {
        dropMaskInfo.s2TotalSize = s2;
        dropMaskInfo.bSSOffset = bDimIdx * s1 * s2;
        pseInfo.s2SizeAcc = bDimIdx * s2;
        pseInfo.bSSOffset = dropMaskInfo.bSSOffset;
    }
    // for compute dropout mask offset
    dropMaskInfo.gOutIdx = gDimIdx;
    dropMaskInfo.n2OutIdx = n2DimIdx;
    dropMaskInfo.s1OutIdx = s1oDimIdx;

    ///////////////////////////////////////////////////////////////
    // SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    LocalTensor<float> sfmgClc3 = vecClc3.Get<float>();
    if (s1CvRatio <= 1) {
        CalcSoftMaxGrad(sfmgClc3, mm1aTensorOffsetCv, s1CvExtend);
    }

    mm1.WaitIterateAll();
    mm1.WaitIterateAll();

    uint32_t curIdxPing = 0;
    uint32_t curIdxPong = 0;
    uint32_t curS2IdxPing = 0;
    uint32_t curS2IdxPong = 0;
    for (uint32_t curS1Idx = 0; curS1Idx < s1VecLoop; curS1Idx++) {
        s1ExtendSubGraph = (curS1Idx == s1VecLoop - 1) ? (s1CvExtend - (s1VecLoop - 1) * s1VecSize) : s1VecSize;
        dropMaskInfo.s1CopySize = s1ExtendSubGraph;
        if (s1CvRatio > 1) {
            pipe_barrier(PIPE_ALL);
            int64_t sfmgOffset = 0;
            if constexpr (INPUT_LAYOUT == TND) {
                if (bDimIdxTmp > 0) {
                    sfmgOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[bDimIdxTmp - 1] * n2 * g * d;;
                }
                sfmgOffset +=
                    (((static_cast<int64_t>(s1oDimIdxTmp) * s1CvInner + curS1Idx * s1VecSize) * n2 + n2DimIdxTmp) * g +
                     gDimIdxTmp) *
                    d;
            } else {
                if constexpr (INPUT_LAYOUT == BNGSD) {
                    sfmgOffset = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * s1 +
                                  s1oDimIdx * s1CvInner + curS1Idx * s1VecSize) *
                                 d;
                } else if constexpr (INPUT_LAYOUT == SBNGD) {
                    sfmgOffset =
                        ((((static_cast<int64_t>(s1oDimIdx) * s1CvInner + curS1Idx * s1VecSize) * b + bDimIdx) * n2 +
                          n2DimIdx) *
                             g +
                         gDimIdx) *
                        d;
                } else if constexpr (INPUT_LAYOUT == BSNGD) {
                    sfmgOffset =
                        (((static_cast<int64_t>(bDimIdx) * s1 + s1oDimIdx * s1CvInner + curS1Idx * s1VecSize) * n2 +
                          n2DimIdx) *
                             g +
                         gDimIdx) *
                        d;
                }
            }
            LocalTensor<float> sfmgClc3 = vecClc3.Get<float>();
            CalcSoftMaxGrad(sfmgClc3, sfmgOffset, s1ExtendSubGraph);
        }

        // for compute dropout mask offset
        dropMaskInfo.s1InnerIdx = curS1Idx;
        // for compute dropout mask
        dropMaskInfo.firstAxis = s1ExtendSubGraph;

        for (uint32_t curS2Idx = 0; curS2Idx < s2VecLoop; curS2Idx = curS2Idx + 2) {
            curS2IdxPing = curS2Idx;
            curS2IdxPong = curS2Idx + 1;
            curIdxPing = curS1Idx * s2VecLoop + curS2IdxPing;
            curIdxPong = curS1Idx * s2VecLoop + curS2IdxPong;
            SubGrapA(curIdxPing, curS1Idx, curS2IdxPing);
            if (curS2IdxPong < s2VecLoop) {
                SubGrapA(curIdxPong, curS1Idx, curS2IdxPong);
            }

            SubGrapB(curIdxPing, curS1Idx, curS2IdxPing);
            if (curS2IdxPong < s2VecLoop) {
                SubGrapB(curIdxPong, curS1Idx, curS2IdxPong);
            }
        }
    }

    uint32_t preS1Extend = s1CvExtend;
    if (nextIndex != 0) {
        InitIndex(nextIndex);
        int64_t nextS2CvExtend = nextS2CvEnd - nextS2CvBegin;
        int64_t nextS2CvExtendAlign = (nextS2CvExtend + 15) / 16 * 16;
        int64_t mm1aTensorOffsetCv1 = 0;
        int64_t mm2aTensorOffsetCv1 = 0;
        int64_t bTensorOffsetCv1 = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            int64_t bDimIdxTmp = bDimIdx;
            int64_t n2DimIdxTmp = n2DimIdx;
            int64_t gDimIdxTmp = gDimIdx;
            int64_t s1oDimIdxTmp = s1oDimIdx;
            if (bDimIdxTmp > 0) {
                mm1aTensorOffsetCv1 = ((__gm__ int64_t *)actual_seq_qlen_addr)[bDimIdxTmp - 1] * n2 * g * d;
                bTensorOffsetCv1 = ((__gm__ int64_t *)actual_seq_kvlen_addr)[bDimIdxTmp - 1] * n2 * d;
            }
            mm1aTensorOffsetCv1 +=
                ((static_cast<int64_t>(s1oDimIdxTmp) * s1CvInner * n2 + n2DimIdxTmp) * g + gDimIdxTmp) * d;
            mm2aTensorOffsetCv1 = mm1aTensorOffsetCv1;
            bTensorOffsetCv1 += (nextS2CvBegin * n2 + n2DimIdxTmp) * d;
        } else {
            if constexpr (INPUT_LAYOUT == BNGSD) {
                mm1aTensorOffsetCv1 =
                    (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1CvInner) * d;
                mm2aTensorOffsetCv1 = mm1aTensorOffsetCv1;
                bTensorOffsetCv1 = ((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * s2 + nextS2CvBegin) * d;
            } else if constexpr (INPUT_LAYOUT == SBNGD) {
                mm1aTensorOffsetCv1 =
                    ((((static_cast<int64_t>(s1oDimIdx) * s1CvInner) * b + bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * d;
                mm2aTensorOffsetCv1 = mm1aTensorOffsetCv1;
                bTensorOffsetCv1 = ((static_cast<int64_t>(nextS2CvBegin) * b + bDimIdx) * n2 + n2DimIdx) * d;
            } else if constexpr (INPUT_LAYOUT == BSNGD) {
                mm1aTensorOffsetCv1 =
                    (((static_cast<int64_t>(bDimIdx) * s1 + s1oDimIdx * s1CvInner) * n2 + n2DimIdx) * g + gDimIdx) * d;
                mm2aTensorOffsetCv1 = mm1aTensorOffsetCv1;
                bTensorOffsetCv1 = ((static_cast<int64_t>(bDimIdx) * s2 + nextS2CvBegin) * n2 + n2DimIdx) * d;
            }
        }
        if constexpr (INPUT_LAYOUT == TND) {
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
            if (MM_OUT_FORMAT == CubeFormat::NZ) {
                mm1.SetOrgShape(s1CvExtend, s2CvExtend, s1StrideSize, s2StrideSize, nextS2CvExtendAlign);
            } else {
                mm1.SetOrgShape(actualS1Len, actualS2Len, s1StrideSize, s2StrideSize, nextS2CvExtendAlign);
            }
        } else {
            if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                mm1.SetOrgShape(s1CvExtend, s2, s1StrideSize, s2StrideSize, nextS2CvExtendAlign);
            } else {
                mm1.SetOrgShape(s1, s2, s1StrideSize, s2StrideSize, nextS2CvExtendAlign);
            }
        }
        mm1.SetTail(s1CvExtend, nextS2CvExtend, d);
        mm1.SetTensorA(dxGm[mm1aTensorOffsetCv1]);
        mm1.SetTensorB(valueGm[bTensorOffsetCv1], true);
        mm1.template IterateAll<false>(mm1WorkspaceGm, false, false, true);

        mm1.SetTail(s1CvExtend, nextS2CvExtend, d);
        mm1.SetTensorA(queryGm[mm2aTensorOffsetCv1]);
        mm1.SetTensorB(keyGm[bTensorOffsetCv1], true);
        mm1.template IterateAll<false>(mm2WorkspaceGm, false, false, true);
    }

    int64_t s1_size = s1;
    int64_t s1TndSize = actualS1Len;
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        s1_size = s1CvExtendAlign;
        s1TndSize = s1CvExtendAlign;
    }

    ///////////////////////////////////////////////////////////////
    // Matmal4 dq
    ///////////////////////////////////////////////////////////////
    // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
    if constexpr (INPUT_LAYOUT == BNGSD) {
        if (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm4.SetOrgShape(s1_size, d, s2CvExtendAlign);
            mm4.SetSelfDefineData(s1);
        } else {
            mm4.SetOrgShape(s1_size, d, s2CvExtendAlign);
        }
    } else if constexpr (INPUT_LAYOUT == SBNGD) {
        if (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm4.SetOrgShape(s1_size, static_cast<int64_t>(b) * n2 * d, s2CvExtendAlign, s2, d);
            mm4.SetSelfDefineData(s1);
        } else {
            mm4.SetOrgShape(s1_size, static_cast<int64_t>(b) * n2 * d, s2CvExtendAlign, s2,
                        static_cast<int64_t>(b) * n2 * g * d);
        }
    } else if constexpr (INPUT_LAYOUT == BSNGD) {
        if (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm4.SetOrgShape(s1_size, n2 * d, s2CvExtendAlign, s2, d);
            mm4.SetSelfDefineData(s1);
        } else {
            mm4.SetOrgShape(s1_size, n2 * d, s2CvExtendAlign, s2, n2 * g * d);
        }
    } else if constexpr (INPUT_LAYOUT == TND) {
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm4.SetOrgShape(s1TndSize, n2 * d, s2CvExtendAlign, actualS2Len, d);
            mm4.SetSelfDefineData(actualS1Len);
        } else {
            mm4.SetOrgShape(s1TndSize, n2 * d, s2CvExtendAlign, actualS2Len, n2 * g * d);
        }
    }
    mm4.SetTail(preS1Extend, -1, s2CvExtend); // M N K
    mm4.SetTensorA(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN]);
    mm4.SetTensorB(keyGm[bTensorOffsetCv]);
    mm4.template IterateAll<false>(dqWorkSpaceGm[dqOffset], true);
    mm4.End();

    ///////////////////////////////////////////////////////////////
    // Matmal4 dk
    ///////////////////////////////////////////////////////////////
    // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
    if constexpr (INPUT_LAYOUT == BNGSD) {
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s2CvExtendAlign, d, s1_size);
            mm3.SetSelfDefineData(s2);
        } else {
            mm3.SetOrgShape(s2CvExtendAlign, d, s1_size);
        }
    } else if constexpr (INPUT_LAYOUT == SBNGD) {
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s2CvExtendAlign, static_cast<int64_t>(b) * n2 * g * d, s1_size, s1, d);
            mm3.SetSelfDefineData(s2);
        } else {
            mm3.SetOrgShape(s2CvExtendAlign, static_cast<int64_t>(b) * n2 * g * d, s1_size, s1,
                        static_cast<int64_t>(b) * n2 * d);
        }
    } else if constexpr (INPUT_LAYOUT == BSNGD) {
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s2CvExtendAlign, n2 * g * d, s1_size, s1, d);
            mm3.SetSelfDefineData(s2);
        } else {
            mm3.SetOrgShape(s2CvExtendAlign, n2 * g * d, s1_size, s1, n2 * d);
        }
    } else if constexpr (INPUT_LAYOUT == TND) {
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s2CvExtendAlign, n2 * g * d, s1TndSize, actualS1Len, d);
            mm3.SetSelfDefineData(actualS2Len);
        } else {
            mm3.SetOrgShape(s2CvExtendAlign, n2 * g * d, s1TndSize, actualS1Len, n2 * d);
        }
    }
    mm3.SetTail(s2CvExtend, -1, preS1Extend);
    mm3.SetTensorA(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN], true);
    mm3.SetTensorB(queryGm[mm2aTensorOffsetCv]);
    mm3.template IterateAll<false>(dkWorkSpaceGm[dkvOffset], true);
    mm3.End();

    ///////////////////////////////////////////////////////////////
    // Matmal5 dv
    ///////////////////////////////////////////////////////////////
    // left [B, N2, G, S1, S2] right [B, N2, G, S1, D] output [B, N2, 1, S2, D]
    mm3.SetTail(s2CvExtend, -1, preS1Extend);
    mm3.SetTensorA(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN], true);
    mm3.SetTensorB(dxGm[mm1aTensorOffsetCv]);
    if constexpr (IsSameType<T1, float>::value) {
        if (nextIndex == 0) {
            mm3.template IterateAll<true>(dvGm[dkvOffset], true);
        } else {
            mm3.template IterateAll<false>(dvGm[dkvOffset], true);
        }
    } else {
        if (nextIndex == 0) {
            mm3.template IterateAll<true>(dvWorkSpaceGm[dkvOffset], true);
        } else {
            mm3.template IterateAll<false>(dvWorkSpaceGm[dkvOffset], true);
        }
    }

    mm3.End();
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT>::SyncALLCores()
{
    SyncAll();
}

#endif // _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2GS1S2_H_
