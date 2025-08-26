/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_quant_mixcore.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_QUANT_MIXCORE_H
#define ASCENDC_GROUPED_MATMUL_QUANT_MIXCORE_H

#include "grouped_matmul_utils.h"
#include "grouped_matmul.h"

#if defined(GMM_QUANT_BF16) || defined(GMM_QUANT_FLOAT16)
namespace GROUPED_MATMUL {
/*@brief store variables for core split configuration
*/
constexpr int32_t PIPELINE_NUM = 4;
constexpr uint32_t BROADCAST_DIM = 2;

/** @brief intenal computation class
*/
template <class mmType, bool sync = false>
class GMMQuantMixCoreCompute : public GMMCompute<mmType, sync> {
 public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using B = typename mmType::BT;
    using CT = typename mmType::CT::T;
    using BiasT = typename mmType::BiasT::T;
    using WT = DTYPE_WEIGHT;
    constexpr static bool transposeX = mmType::AT::isTrans;
    constexpr static bool transposeW = mmType::BT::isTrans;

    /** @brief constructor */
    __aicore__ inline GMMQuantMixCoreCompute(typename mmType::MT& mm_) : GMMCompute<mmType, sync>(mm_) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
                                GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR group_list,
                                GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                const GMMBaseParams* __restrict gmmBaseParams,
                                const TCubeTiling* __restrict mmTilingData, TPipe* tPipe);

    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig, uint32_t coreIdx);

    __aicore__ inline void VectorCompute(MNConfig& mnConfig);

    __aicore__ inline void PostCompute();

 private:
    __aicore__ inline void Dequant(MNConfig& mnConfig);

    __aicore__ inline void SetPerTokenQuantStaticBuffer(const GMMBaseParams* __restrict gmmBaseParams,
                                                        const TCubeTiling* __restrict mmTilingData, GM_ADDR workspace);

    __aicore__ inline void DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);

    __aicore__ inline void DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig, uint32_t curBaseM, uint32_t alignBaseN,
                                                        uint32_t offsetM);

    __aicore__ inline void SetPerTokenQuantRefreshedBuffer(const MNConfig mnConfig);

    __aicore__ inline void ActivationCompute(uint32_t computeSize, LocalTensor<float> preResUb,
                                             LocalTensor<uint8_t> actTmpLocal);

    __aicore__ inline void ComputeDequantAndActivate(MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, 
                                                     uint32_t offsetM);

    __aicore__ inline void DataCopyOut(MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t curVecBaseN,
                                       uint32_t alignBaseN, uint64_t outOffset);

    __aicore__ inline void VectorTilingCalc(MNConfig& mnConfig, uint32_t& curCubeSingleN, uint32_t& curCubeSingleM, 
                                            uint32_t& vecBaseN, uint32_t& vecBaseM);

    GM_ADDR scaleTensorPtr;
    GM_ADDR perTokenScaleTensorPtr;
    GlobalTensor<DTYPE_SCALE> scaleGm;
    GlobalTensor<float> perTokenScaleGm;
    GlobalTensor<CT> mmOutGm;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> scaleInQueue;
    TQue<QuePosition::VECIN, 1> perTokenScaleInQueue;
    TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<CT> mmOutInUb;
    LocalTensor<DTYPE_SCALE> scaleInUb;
    LocalTensor<float> perTokenScaleInUb;
    LocalTensor<float> dequantMiddleResult;
    LocalTensor<uint8_t> sharedTmpLocal;
    LocalTensor<float> mulsResultLocal;
    LocalTensor<float> pertokenBrcbLocal;
    LocalTensor<float> actResultLocal;
    bool sequentialWrite = true;
    bool isPerTokenQuant;
    uint32_t cubeNum;  // Matmul completions on the kernel
};

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantMixCoreCompute<mmType, sync>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                                                  GM_ADDR scale, GM_ADDR offset, GM_ADDR antiquantScale,
                                                                  GM_ADDR antiquantOffset, GM_ADDR groupList,
                                                                  GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                                                  const GMMBaseParams* __restrict gmmBaseParams,
                                                                  const TCubeTiling* __restrict mmTilingData,
                                                                  TPipe* tPipe) {
    this->GMMCompute<mmType, sync>::Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList,
        perTokenScale, y, workspace, gmmBaseParams, mmTilingData, tPipe);
    isPerTokenQuant = gmmBaseParams->quantParam == 1;
    sequentialWrite = gmmBaseParams->singleN == 0;
    scaleTensorPtr = scale;
    perTokenScaleTensorPtr = perTokenScale;
    cubeNum = 0;
    SetPerTokenQuantStaticBuffer(gmmBaseParams, mmTilingData, workspace);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantMixCoreCompute<mmType, sync>::PostCompute() {
    if ASCEND_IS_AIC {
        for (int32_t idx = 0; idx < Min<int32_t>(cubeNum, PIPELINE_NUM); ++idx) {
            CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);
        }
    }
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantMixCoreCompute<mmType, sync>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig,
                                                                       uint32_t coreIdx) {
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM
                                                                 : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    if constexpr (transposeX) {
        xOffset = mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    // init global buffer
    if (this->singleX == 0) {
        this->xGm.SetGlobalBuffer(GetTensorAddr<AT>(groupIdx, this->xTensorPtr));
    } else {
        this->xGm.SetGlobalBuffer(GetTensorAddr<AT>(0, this->xTensorPtr) + mnConfig.xBaseOffset);
    }
    GlobalTensor<BT> weightGm = this->SetGlobalBufferW(groupIdx, tailN, mnConfig);
    if (sequentialWrite) {
        mnConfig.workSpaceOffset = mnConfig.baseN * mnConfig.baseM * \
                                   (coreIdx + (cubeNum % PIPELINE_NUM) * this->coreNum);
    } else {
        mnConfig.workSpaceOffset = outOffset + mnConfig.yBaseOffset;
    }
    if ASCEND_IS_AIC {
        if (this->cubeNum >= PIPELINE_NUM) {
            CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);
        }
        this->mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
        this->mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
        this->mm.SetTensorA(this->xGm[xOffset], transposeX);
        this->mm.SetTensorB(weightGm, transposeW);
        this->SetGlobalBufferBias(groupIdx, tailN, mnConfig);
        while (this->mm.Iterate()) {
            this->mm.GetTensorC(mmOutGm[mnConfig.workSpaceOffset], 0, sequentialWrite);
        }
        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
    }
    cubeNum++;
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantMixCoreCompute<mmType, sync>::VectorCompute(MNConfig& mnConfig) {
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
        SetPerTokenQuantRefreshedBuffer(mnConfig);
        Dequant(mnConfig);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG);
    }
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantMixCoreCompute<mmType, sync>::ComputeDequantAndActivate(MNConfig& mnConfig, 
    uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM) {
    DataCopyPerTokenScaleAndBrcb(mnConfig, curVecBaseM, alignBaseN, offsetM);
    mmOutInUb = vecInQueue.DeQue<CT>();
    LocalTensor<DTYPE_Y> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_Y>();

    #if defined(GMM_QUANT_BF16)
    if (!isPerTokenQuant && this->activeType == 0) {  // BF16 static quantization without activation.
        AscendDequant(yLocalInUb, mmOutInUb, scaleInUb, sharedTmpLocal, {curVecBaseM, alignBaseN, curVecBaseN});
        vecInQueue.FreeTensor(mmOutInUb);
        vecOutQueue.EnQue(yLocalInUb);
        return;
    }
    #endif
    AscendDequant(dequantMiddleResult, mmOutInUb, scaleInUb, sharedTmpLocal, {curVecBaseM, alignBaseN, curVecBaseN});
    PipeBarrier<PIPE_V>();
    LocalTensor<float> preResUb = dequantMiddleResult;
    LocalTensor<float> yFP32LocalInUb = dequantMiddleResult;
    LocalTensor<uint8_t> actTmpLocal = sharedTmpLocal;
    // pertoken antiquant
    if (isPerTokenQuant) {
        Mul(mulsResultLocal, dequantMiddleResult, pertokenBrcbLocal, curVecBaseM * alignBaseN);
        PipeBarrier<PIPE_V>();
        preResUb = mulsResultLocal;
        yFP32LocalInUb = mulsResultLocal;
        actTmpLocal = tmpBuff.GetWithOffset<uint8_t>(2 * this->ubCalSize * sizeof(float), 0);
    }
    // activation function
    if (this->activeType != 0) {
        uint32_t computeSize = curVecBaseM * alignBaseN;
        ActivationCompute(computeSize, preResUb, actTmpLocal);
        yFP32LocalInUb = actResultLocal;
    }
    // get final output after Cast
    #if defined(GMM_QUANT_BF16)
        Cast(yLocalInUb, yFP32LocalInUb, RoundMode::CAST_RINT, curVecBaseM * alignBaseN);
    #elif defined(GMM_QUANT_FLOAT16)
        Cast(yLocalInUb, yFP32LocalInUb, RoundMode::CAST_NONE, curVecBaseM * alignBaseN);
    #endif
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);
    vecOutQueue.EnQue(yLocalInUb);
    return;
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantMixCoreCompute<mmType, sync>::VectorTilingCalc(
    MNConfig& mnConfig, uint32_t& curCubeSingleN, uint32_t& curCubeSingleM, uint32_t& vecBaseN,
    uint32_t& vecBaseM) {
    curCubeSingleN = mnConfig.nIdx == mnConfig.blockDimN - 1 ?
                              mnConfig.n - mnConfig.nIdx * mnConfig.singleN : mnConfig.singleN;
    curCubeSingleM = mnConfig.mIdx == mnConfig.blockDimM - 1 ?
                              mnConfig.m - mnConfig.mIdx * mnConfig.singleM : mnConfig.singleM;
    vecBaseN = sequentialWrite ? curCubeSingleN : mnConfig.baseN;
    vecBaseM = this->ubCalSize / AlignUp(vecBaseN, static_cast<uint32_t>(UB_BLOCK_DOUBLE_UNIT_SIZE / sizeof(int32_t)));
    vecBaseM = Min(vecBaseM, curCubeSingleM);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantMixCoreCompute<mmType, sync>::Dequant(MNConfig& mnConfig) {
    uint32_t curCubeSingleN;
    uint32_t curCubeSingleM;
    uint32_t vecBaseN;
    uint32_t vecBaseM;
    VectorTilingCalc(mnConfig, curCubeSingleN, curCubeSingleM, vecBaseN, vecBaseM);
    uint32_t curVecBaseN = vecBaseN;
    uint32_t curVecBaseM;
    uint32_t vecCount = 0;
    uint32_t rowLength = sequentialWrite ? curCubeSingleN : mnConfig.n;
    uint32_t taskRation = GetTaskRation();
    for (uint32_t offsetN = 0; offsetN < curCubeSingleN; offsetN += vecBaseN) {
        if (unlikely(offsetN + vecBaseN >= curCubeSingleN)) { curVecBaseN = curCubeSingleN - offsetN; }
        uint32_t alignBaseN = AlignUp(curVecBaseN, static_cast<uint32_t>(UB_BLOCK_DOUBLE_UNIT_SIZE / sizeof(int32_t)));
        uint64_t scaleOffset = mnConfig.nIdx * mnConfig.singleN + offsetN;
        DataCopyScale(curVecBaseN, alignBaseN, scaleOffset);
        curVecBaseM = vecBaseM;
        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM) {
            vecCount++;
            if (vecCount % taskRation != this->subBlockIdx) {
                continue; 
            }
            if (unlikely(offsetM + vecBaseM >= curCubeSingleM)) { 
                curVecBaseM = curCubeSingleM - offsetM; 
            }
            // use AscendDequant interface to do perchannel dequant
            uint64_t mmOutOffset = mnConfig.workSpaceOffset + offsetM * static_cast<uint64_t>(rowLength) + offsetN;
            LocalTensor<CT> mmOutLocal = vecInQueue.AllocTensor<CT>();
            DataCopyPad2D(mmOutLocal, mmOutGm[mmOutOffset], curVecBaseM, curVecBaseN, rowLength);
            vecInQueue.EnQue(mmOutLocal);
            ComputeDequantAndActivate(mnConfig, curVecBaseM, alignBaseN, curVecBaseN, offsetM);
            uint64_t outOffset = (mnConfig.mIdx * mnConfig.singleM + offsetM) * mnConfig.n + \
                                  mnConfig.nIdx * mnConfig.singleN + offsetN;
            DataCopyOut(mnConfig, curVecBaseM, curVecBaseN, alignBaseN, outOffset);
        }
        scaleInQueue.FreeTensor(scaleInUb);
    }
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantMixCoreCompute<mmType, sync>::DataCopyOut(MNConfig& mnConfig, uint32_t curVecBaseM,
                                                                         uint32_t curVecBaseN, uint32_t alignBaseN,
                                                                         uint64_t outOffset) {
    // Copy the result of vector to yGm.
    LocalTensor<DTYPE_Y> yLocal = vecOutQueue.DeQue<DTYPE_Y>();
    DataCopyPad2D(this->yGm[outOffset], yLocal, curVecBaseM, curVecBaseN, alignBaseN, mnConfig.n);
    vecOutQueue.FreeTensor(yLocal);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantMixCoreCompute<mmType, sync>::ActivationCompute(uint32_t computeSize,
                                                                               LocalTensor<float> preResUb,
                                                                               LocalTensor<uint8_t> actTmpLocal) {
    ActiveType active = ActiveType(this->activeType);
    if (active == ActiveType::FASTGELU) {
        FasterGelu(actResultLocal, preResUb, actTmpLocal, computeSize);
    } else if (active == ActiveType::RELU) {
        Relu(actResultLocal, preResUb, computeSize);
    } else if (active == ActiveType::SILU) {
        Silu(actResultLocal, preResUb, computeSize);
    } else if (active == ActiveType::GELU_TANH) {
        Gelu(actResultLocal, preResUb, actTmpLocal, computeSize);
    }
    PipeBarrier<PIPE_V>();
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantMixCoreCompute<mmType, sync>::SetPerTokenQuantStaticBuffer(
    const GMMBaseParams* __restrict gmmBaseParams, const TCubeTiling* __restrict mmTilingData, GM_ADDR workspace) {
    // Initialize ub and gm memories that do not need to be reinitialized due to changes in groupidx.
    if ASCEND_IS_AIV {
        // 2: enabling double buffer, occupying two buffer.
        this->pipe->InitBuffer(scaleInQueue, 2, mmTilingData->baseN * sizeof(DTYPE_SCALE));
        if (isPerTokenQuant) {
            // 2: enabling double buffer, occupying two buffer.
            this->pipe->InitBuffer(perTokenScaleInQueue, 2, mmTilingData->baseM * sizeof(float));
        }
        // 2: enabling double buffer, occupying two buffer.
        this->pipe->InitBuffer(vecInQueue, 2, this->ubCalSize * sizeof(CT));
        // 2: enabling double buffer, occupying two buffer.
        this->pipe->InitBuffer(vecOutQueue, 2, this->ubCalSize * sizeof(DTYPE_Y));
        this->pipe->InitBuffer(tmpBuff, gmmBaseParams->ubRestBytes);
        dequantMiddleResult = tmpBuff.GetWithOffset<float>(this->ubCalSize, 0);
        #if defined(GMM_QUANT_FLOAT16) 
        uint32_t factor = 1;
        #else 
        uint32_t factor = 0;
        #endif
        // 2: Indicates the first two blocks of ub are already occupied.
        factor = !isPerTokenQuant && this->activeType == 0 ? factor : 2;
        uint32_t ubCalSizeFloat = this->ubCalSize * sizeof(float);
        uint32_t offset = factor * ubCalSizeFloat;
        // 2: Indicates a temporary space twice the size is needed.
        sharedTmpLocal = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, offset);
        if (isPerTokenQuant) {
            // 2: Indicates the first two blocks of ub are already occupied.
            mulsResultLocal = tmpBuff.GetWithOffset<float>(this->ubCalSize, 2 * ubCalSizeFloat);
            pertokenBrcbLocal = tmpBuff.GetWithOffset<float>(this->ubCalSize, ubCalSizeFloat);
        }
        if (this->activeType != 0) {
            // 2: Indicates the first three blocks of ub are already occupied.
            uint32_t offsetAct = !isPerTokenQuant ? ubCalSizeFloat : 3 * ubCalSizeFloat;
            actResultLocal = tmpBuff.GetWithOffset<float>(this->ubCalSize, offsetAct);
        }
    }
    mmOutGm.SetGlobalBuffer((__gm__ MM_DTYPE_Y *)workspace);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantMixCoreCompute<mmType, sync>::SetPerTokenQuantRefreshedBuffer(const MNConfig mnConfig) {
    // Initialize gm memories that need to be reinitialized due to changes in groupidx.
    // Currently, pertoken quant only supports single-tensor mode, 
    // hence set according to x and weight single-tensor mode.
    // Add an if branch if multi-tensor mode for weght is required.
    scaleGm.SetGlobalBuffer(GetTensorAddr<DTYPE_SCALE>(0, scaleTensorPtr) + mnConfig.nAxisBaseOffset);
    if (isPerTokenQuant) {
        perTokenScaleGm.SetGlobalBuffer((__gm__ float *)perTokenScaleTensorPtr + mnConfig.mAxisBaseOffset);
    }
    // Add an if branch if multi-tensor mode for y is required.
    this->yGm.SetGlobalBuffer(GetTensorAddr<DTYPE_Y>(0, this->yTensorPtr) + mnConfig.yBaseOffset);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantMixCoreCompute<mmType, sync>::DataCopyScale(uint32_t curBaseN,
                                                                           uint32_t alignBaseN,
                                                                           uint64_t scaleOffset)
{
    // GM copy scale
    DataCopyPadExtParams<DTYPE_SCALE> padParams;
    DataCopyExtParams scaleParams;
    scaleParams.blockLen = curBaseN * sizeof(DTYPE_SCALE);
    scaleParams.blockCount = 1;
    scaleParams.srcStride = 0;
    scaleParams.dstStride = 0;
    LocalTensor<DTYPE_SCALE> scaleLocal = scaleInQueue.AllocTensor<DTYPE_SCALE>();
    DataCopyPad(scaleLocal, scaleGm[scaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);

    scaleInUb = scaleInQueue.DeQue<DTYPE_SCALE>();
    scaleInUb.SetSize(alignBaseN);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantMixCoreCompute<mmType, sync>::DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig,
                                                                                          uint32_t curBaseM,
                                                                                          uint32_t alignBaseN,
                                                                                          uint32_t offsetM)
{
    if (!isPerTokenQuant) {
        return;
    }
    uint64_t perTokenScaleOffset = mnConfig.mIdx * mnConfig.singleM + offsetM;
    // GM copy per token scale
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams perTokenScaleParams;
    perTokenScaleParams.blockLen = curBaseM * sizeof(float);
    perTokenScaleParams.blockCount = 1;
    perTokenScaleParams.srcStride = 0;
    perTokenScaleParams.dstStride = 0;
    LocalTensor<float> perTokenScaleLocal = perTokenScaleInQueue.AllocTensor<float>();
    DataCopyPad(perTokenScaleLocal, perTokenScaleGm[perTokenScaleOffset], perTokenScaleParams, padParams);
    perTokenScaleInQueue.EnQue(perTokenScaleLocal);

    perTokenScaleInUb = perTokenScaleInQueue.DeQue<float>();
    const uint32_t broadCastDst[BROADCAST_DIM] = {curBaseM, alignBaseN};
    const uint32_t broadCastSrc[BROADCAST_DIM] = {curBaseM, 1};
    BroadCast<float, BROADCAST_DIM, 1>(pertokenBrcbLocal, perTokenScaleInUb, broadCastDst, broadCastSrc,
                                       sharedTmpLocal);
    perTokenScaleInQueue.FreeTensor(perTokenScaleInUb);
}

}  // namespace GROUPED_MATMUL

#endif
#endif  // ASCENDC_GROUPED_MATMUL_QUANT_MIXCORE_H
