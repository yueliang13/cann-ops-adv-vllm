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
 * \file grouped_matmul_antiquant_a8w4_msd.h
 * \brief
 */

#ifndef ASCENDC_GROUPED_MATMUL_ANTIQUANT_A8W4_MSD_H
#define ASCENDC_GROUPED_MATMUL_ANTIQUANT_A8W4_MSD_H

#include "grouped_matmul_utils.h"
#include "grouped_matmul.h"

#ifdef GMM_ANTI_QUANT_A8W4_MSD
namespace GROUPED_MATMUL{
using namespace matmul;
using namespace AscendC;

#define DTYPE_PERTOKEN_SCALE_A8W4MSD float
#define DTYPE_BIAS_A8W4MSD float

#ifdef GMM_ANTI_QUANT_A8W4_MSD_OUT_BF16
    #define DTYPE_Y_A8W4MSD bfloat16_t
#else
    #define DTYPE_Y_A8W4MSD half
#endif

#define DTYPE_X_DEV_A8W4MSD int4b_t
#define DTYPE_WEIGHT_DEV_A8W4MSD int4b_t
#define DTYPE_SCALE_DEV_A8W4MSD uint64_t

constexpr uint64_t SYNC_AIV_TO_AIC = 3;
constexpr uint64_t SYNC_AIC_TO_AIV = 5;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t MM_BASE_BLOCK_OFFSET = 32768; // baseM * baseN = 128 * 256

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcDim0) {
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    // 32: int32 -> float16, 为防止跨行数据进入同一32B block，提前每行按偶数block对齐
    params.dstStride = Ceil(dim0 * sizeof(T), 32) % 2;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4(const GlobalTensor<T> dst, const LocalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcDim0, uint32_t dstDim0) {
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    // 32: ub访问粒度为32B
    params.srcStride = (srcDim0 - dim0) * sizeof(T) / 32;
    params.dstStride = (dstDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

template <class mmType>
class GMMA8W4MSDCompute {
public:
    using aT = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X_DEV_A8W4MSD>;
    using bT = MatmulType<TPosition::GM, CubeFormat::NZ, DTYPE_WEIGHT_DEV_A8W4MSD>;
    using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using cT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using DTYPE_OUT = DTYPE_Y_A8W4MSD;

public:
    __aicore__ inline GMMA8W4MSDCompute(typename mmType::MT &matmul) : mm(matmul) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
        GM_ADDR group_tokens, GM_ADDR scale, GM_ADDR pertoken_scale, GM_ADDR logits, GM_ADDR token_ranks, GM_ADDR residual,
        GM_ADDR y, GM_ADDR workspace, GMMBaseParams* tilingData, TCubeTiling* mmTilingData, TPipe *tPipeIn);
    __aicore__ inline void Process();
private:
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void VectorCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void ComputeDequantAndActivate(MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t alignBaseN,
                                                     uint32_t curVecBaseN, uint32_t offsetM);
    __aicore__ inline void DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig, uint32_t curBaseM, uint32_t alignBaseN,
                                                        uint32_t offsetM);

private:
    typename mmType::MT& mm;
    GlobalTensor<DTYPE_X_DEV_A8W4MSD> xGm;
    GlobalTensor<DTYPE_WEIGHT_DEV_A8W4MSD> weightGm;
    GlobalTensor<DTYPE_BIAS_A8W4MSD> biasGm;  // for 8 * weight
    GlobalTensor<cT::T> mmOutGm;
    GlobalTensor<DTYPE_SCALE_DEV_A8W4MSD> scaleGm;
    GlobalTensor<DTYPE_PERTOKEN_SCALE_A8W4MSD> perTokenScaleGm;
    GlobalTensor<int64_t> groupTokensGm;
    GlobalTensor<float> logitsGm;
    GlobalTensor<bfloat16_t> residualGm;
    GlobalTensor<int32_t> tokenRanksGm;
    GlobalTensor<DTYPE_OUT> yGm;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> scaleInQueue;
    TQue<QuePosition::VECIN, 1> perTokenScaleInQueue;
    TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<DTYPE_BIAS_A8W4MSD> scaleInUb;
    LocalTensor<float> buffer1;
    LocalTensor<float> buffer2;
    LocalTensor<float> buffer3;
    LocalTensor<float> buffer4;
    LocalTensor<float> buffer5;
    LocalTensor<uint8_t> buffer6;
    LocalTensor<uint8_t> buffer7;
    uint32_t subBlockIdx;
    uint32_t coreIdx;
    uint32_t quantGroupSize;
    uint32_t cubeCount = 0;
    uint32_t vecCount = 0;
    TPipe *pipe;
    GMMBaseParams *tiling;
    TCubeTiling* mmTilingData;
};

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
        GM_ADDR group_tokens, GM_ADDR scale, GM_ADDR pertoken_scale, GM_ADDR logits, GM_ADDR token_ranks, GM_ADDR residual,
        GM_ADDR y, GM_ADDR workspace, GMMBaseParams* tilingData, TCubeTiling* mmTilingData, TPipe *tPipeIn)
{
    xGm.SetGlobalBuffer(GetTensorAddr<DTYPE_X_DEV_A8W4MSD>(0, x));
    weightGm.SetGlobalBuffer(GetTensorAddr<DTYPE_WEIGHT_DEV_A8W4MSD>(0, weight));
    biasGm.SetGlobalBuffer(GetTensorAddr<DTYPE_BIAS_A8W4MSD>(0, bias));
    mmOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ cT::T *>(workspace));
    scaleGm.SetGlobalBuffer(GetTensorAddr<DTYPE_SCALE_DEV_A8W4MSD>(0, scale));
    perTokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_PERTOKEN_SCALE_A8W4MSD *>(pertoken_scale));
    groupTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(group_tokens));
    yGm.SetGlobalBuffer(GetTensorAddr<DTYPE_OUT>(0, y));

    tiling = tilingData;
    this->mmTilingData = mmTilingData;
    quantGroupSize = tiling->k / tiling->quantGroupNum;  // 约束为整除关系
    subBlockIdx = GetSubBlockIdx();
    coreIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        if (GetTaskRation() != 0) {
            coreIdx /= GetTaskRation();
        }
    }
    pipe = tPipeIn;
    InitUbBuffer();
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::InitUbBuffer()
{
    if ASCEND_IS_AIC {
        return;
    }
    pipe->InitBuffer(scaleInQueue, BUFFER_NUM, mmTilingData->baseN * sizeof(DTYPE_BIAS_A8W4MSD)); // bias queue
    pipe->InitBuffer(perTokenScaleInQueue, BUFFER_NUM, Ceil(tiling->vBaseM * sizeof(float), 32) * 32);
    pipe->InitBuffer(vecInQueue, BUFFER_NUM, tiling->ubCalSize * 2 * sizeof(cT::T));
    pipe->InitBuffer(vecOutQueue, BUFFER_NUM, tiling->ubCalSize * sizeof(DTYPE_OUT));
    pipe->InitBuffer(tmpBuff, tiling->ubRestBytes);
    uint32_t ubCalSizeFloat = tiling->ubCalSize * sizeof(float);
    // ub分配，依次划分中间结果，划分方式参考设计文档
    buffer1 = tmpBuff.GetWithOffset<float>(tiling->ubCalSize * 2, 0);
    buffer7 = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, 0);
    uint32_t offset = ubCalSizeFloat * 2;
    buffer2 = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, offset);
    buffer6 = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, offset);
    offset += ubCalSizeFloat;
    buffer3 = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, offset);
    buffer4 = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 0);
    buffer5 = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, ubCalSizeFloat);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::Process()
{
    MNConfig mnConfig;
    mnConfig.baseM = mmTilingData->baseM;
    mnConfig.baseN = mmTilingData->baseN;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;
    mnConfig.blockDimN = Ceil(tiling->n, mnConfig.singleN);
    mnConfig.singleN = Ceil(tiling->n, mnConfig.blockDimN);
    if ASCEND_IS_AIC {
        SyncAll<false>();
    }
    for (uint32_t groupIdx = 0, preCount = 0; groupIdx < tiling->groupNum; ++groupIdx) {
        int32_t m = static_cast<int32_t>(groupTokensGm.GetValue(groupIdx));
        if (m <= 0) {
            continue;
         }
        mnConfig.m = static_cast<uint32_t>(m) * 2;      // 2: int8 has been split in 2 int4
        mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
        mm.SetOrgShape(mnConfig.m, tiling->n, tiling->k);
        uint32_t curCount = preCount + mnConfig.blockDimN * mnConfig.blockDimM;
        uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + tiling->coreNum;

        while (curBlock < curCount) {
            mnConfig.mIdx = (curBlock - preCount) / mnConfig.blockDimN;
            mnConfig.nIdx = (curBlock - preCount) % mnConfig.blockDimN;
            MMCompute(groupIdx, mnConfig);
            if ASCEND_IS_AIV {
                VectorCompute(groupIdx, mnConfig);
            }
            curBlock += tiling->coreNum;
        }
        preCount = curCount % tiling->coreNum;
        mnConfig.offsetM += mnConfig.m;
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    mnConfig.workSpaceOffset = MM_BASE_BLOCK_OFFSET * \
                                   (coreIdx + (cubeCount % tiling->parallNum) * tiling->coreNum);
    if ASCEND_IS_AIC {
        uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
        uint32_t curSingleN = mnConfig.singleN;
        if (unlikely(mnConfig.nIdx == mnConfig.blockDimN - 1)) {
            curSingleN = tiling->n - tailN;
        }
        uint32_t curSingleM = mnConfig.singleM;
        if (unlikely(mnConfig.mIdx == mnConfig.blockDimM - 1)) {
            curSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
        }

        uint64_t xOffset = (mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM) * tiling->k;
        uint64_t weightOffset = groupIdx * tiling->n * tiling->k + tailN * tiling->k;  // for no transpose nz weight
        if (cubeCount >= tiling->parallNum) {
            CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
        }
        mm.SetSingleShape(curSingleM, curSingleN, quantGroupSize); // 8, 256, 512 --> 514us
        for (uint32_t loopK = 0; loopK < tiling->quantGroupNum; loopK++) {
            mm.SetTensorA(xGm[xOffset + loopK * quantGroupSize]);
            auto weightSlice = weightGm[weightOffset + loopK * quantGroupSize * 64];  // 64: 64 int4 in 32B
            if (mnConfig.blockDimM == 1) {
                weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
            mm.SetTensorB(weightSlice);
            mm.SetQuantVector(scaleGm[groupIdx * tiling->n * tiling->quantGroupNum + loopK * tiling->n + tailN]);
            uint64_t worskspaceOffset = mnConfig.workSpaceOffset;
            mm.Iterate();
            mm.GetTensorC(mmOutGm[worskspaceOffset], loopK == 0 ? 0 : 1, true);
            worskspaceOffset += MM_BASE_BLOCK_OFFSET;
        }
        CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);  // 2: mode为2, group内同步
    }
    cubeCount++;
}
template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::VectorCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    uint32_t curCubeSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) { curCubeSingleN = tiling->n - mnConfig.nIdx * mnConfig.singleN; }
    uint32_t curCubeSingleM = mnConfig.singleM / 2; // 2: 2 lines int4 to 1 line int8
    uint32_t mGlobalOffset = mnConfig.offsetM / 2 + mnConfig.mIdx * curCubeSingleM; // 2: 2 lines int4 to 1 line int8
    uint64_t outOffset = mGlobalOffset * tiling->n + mnConfig.nIdx * mnConfig.singleN;
     // 2: 2 lines int4 to 1 line int8
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) { curCubeSingleM = mnConfig.m / 2 - mnConfig.mIdx * curCubeSingleM; }
    uint32_t vecBaseM = tiling->ubCalSize / (Ceil(mnConfig.baseN, uint32_t(8)) * 8);  //  8: num int32_t in 32B ub block  32*256/256
    vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;
    uint32_t curVecBaseN = mnConfig.baseN;
    uint64_t scaleOffset = groupIdx * tiling->n + mnConfig.nIdx * mnConfig.singleN;
    uint32_t taskRation = GetTaskRation();
    CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
    for (uint32_t offsetN = 0; offsetN < curCubeSingleN; offsetN += mnConfig.baseN) {
        if (unlikely(offsetN + mnConfig.baseN >= curCubeSingleN)) curVecBaseN = curCubeSingleN - offsetN;
        uint32_t alignBaseN = Ceil(curVecBaseN, uint32_t(8)) * 8;  //  8: num int32_t in 32B ub block
        DataCopyScale(curVecBaseN, alignBaseN, scaleOffset + offsetN);
        uint32_t curVecBaseM = vecBaseM;
        uint64_t mmOutOffset = mnConfig.workSpaceOffset + offsetN * mnConfig.baseM;
        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM) {
            vecCount++;
            if (taskRation != 0 && vecCount % taskRation != subBlockIdx) { continue; }
            if (unlikely(offsetM + vecBaseM >= curCubeSingleM)) { curVecBaseM = curCubeSingleM - offsetM; }
            LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
            DataCopyPad2DA8W4(mmOutLocal, mmOutGm[mmOutOffset + offsetM * 2 * curVecBaseN],
                          curVecBaseM, curVecBaseN, curVecBaseN * 2);   // 2: 2 lines int4 to 1 line int8
            DataCopyPad2DA8W4(mmOutLocal[curVecBaseM * alignBaseN],
                          mmOutGm[mmOutOffset + (offsetM * 2 + 1) * curVecBaseN],
                          curVecBaseM, curVecBaseN, curVecBaseN * 2);    // 2: 2 lines int4 to 1 line int8
            vecInQueue.EnQue(mmOutLocal);
            ComputeDequantAndActivate(mnConfig, curVecBaseM, alignBaseN, curVecBaseN, offsetM);
            LocalTensor<DTYPE_OUT> yLocal = vecOutQueue.DeQue<DTYPE_OUT>();
            DataCopyPad2DA8W4(yGm[outOffset + offsetM * tiling->n + offsetN], yLocal,
                        curVecBaseM, curVecBaseN, alignBaseN, tiling->n);
            vecOutQueue.FreeTensor(yLocal);
        }
        scaleInQueue.FreeTensor(scaleInUb);
    }
    CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);  // 2: mode为2, group内同步
}
template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::ComputeDequantAndActivate(MNConfig& mnConfig, 
    uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM)
{
    uint32_t computeSize = curVecBaseM * alignBaseN;
    LocalTensor<cT::T> mmOutInUb = vecInQueue.DeQue<cT::T>();

    Cast(buffer1, mmOutInUb, RoundMode::CAST_NONE, computeSize * 2);  // 2: in pre process, x has been splited to [2m,n]
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);
    const float RIGHT_MOVE = 16.0f;         // right move int4 to int8
    Muls(buffer2, buffer1, RIGHT_MOVE, computeSize);
    PipeBarrier<PIPE_V>();
    Add(buffer3, buffer1[computeSize], buffer2, computeSize);
    PipeBarrier<PIPE_V>();
    uint32_t loop = alignBaseN / 64; // 256B为64个float，alignBaseN需约束为64倍数
    uint8_t blkStride = static_cast<uint8_t>(alignBaseN * sizeof(float) / 32);  //32: 单位32B
    BinaryRepeatParams param(1, 1, 1, blkStride, blkStride, 0);
    uint64_t mask = 64;     // float is 32 bit, mask continous mode [1,64], set all params involved in the computation
    for (uint32_t i = 0; i < loop; i++) {  // 待优化为broadcast
        uint32_t offset = i * 64; // 每次64个元素
        Add(buffer2[offset], buffer3[offset], scaleInUb[offset], mask, curVecBaseM, param);
    }
    PipeBarrier<PIPE_V>();

    DataCopyPerTokenScaleAndBrcb(mnConfig, curVecBaseM, alignBaseN, offsetM);

    Mul(buffer4, buffer2, buffer3, computeSize);
    PipeBarrier<PIPE_V>();
    auto out = buffer4;

    LocalTensor<DTYPE_OUT> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_OUT>();
    // Cast后获得最终输出
    Cast(yLocalInUb, out, RoundMode::CAST_RINT, computeSize);
    PipeBarrier<PIPE_V>();
    vecOutQueue.EnQue(yLocalInUb);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset)
{
    // GM拷贝scale
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams scaleParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    LocalTensor<float> scaleLocal = scaleInQueue.AllocTensor<DTYPE_BIAS_A8W4MSD>();
    DataCopyPad(scaleLocal, biasGm[scaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);
    scaleInUb = scaleInQueue.DeQue<float>();
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig,
        uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM)
{
    uint64_t perTokenScaleOffset = mnConfig.offsetM / 2 + mnConfig.mIdx * mnConfig.singleM / 2 + offsetM; //2: m方向两行合并为1行
    uint32_t alignBaseM = Ceil(curBaseM, uint32_t(8)) * 8;  //  8: num int32_t in 32B ub block
    // GM拷贝per token scale
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams perTokenScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    LocalTensor<float> perTokenScaleLocal = perTokenScaleInQueue.AllocTensor<float>();
    DataCopyPad(perTokenScaleLocal, perTokenScaleGm[perTokenScaleOffset], perTokenScaleParams, padParams);

    perTokenScaleInQueue.EnQue(perTokenScaleLocal);

    perTokenScaleLocal = perTokenScaleInQueue.DeQue<float>();
    auto scaleTmp = perTokenScaleLocal;

    const uint32_t broadCastDst[2] = {curBaseM, alignBaseN};
    const uint32_t broadCastSrc[2] = {curBaseM, 1};
    BroadCast<float, 2, 1>(buffer3, scaleTmp, broadCastDst, broadCastSrc, buffer7);
    perTokenScaleInQueue.FreeTensor(perTokenScaleLocal);
}
} // namespace GROUPED_MATMUL
#endif
#endif