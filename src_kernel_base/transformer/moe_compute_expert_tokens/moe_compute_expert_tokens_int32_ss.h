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
 * \file moe_compute_expert_tokens_int32_ss.h
 * \brief
 */
#ifndef MOE_COMPUTE_EXPERT_TOKENS_INT32_SS
#define MOE_COMPUTE_EXPERT_TOKENS_INT32_SS

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeCompute {

using namespace AscendC;

template <typename T>
class MoeComputeExpertTokensInt32SS {
public:
    __aicore__ inline MoeComputeExpertTokensInt32SS(){};
    __aicore__ inline void Init(GM_ADDR sortExperts,
                                GM_ADDR out,
                                GM_ADDR workspace,
                                const MoeComputeExpertTokensTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();
    __aicore__ inline int64_t Int32AlignmentProcess(int64_t param);
    __aicore__ inline int64_t Int256AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcessInt32(int64_t param);
    __aicore__ inline void ParseTilingData(const MoeComputeExpertTokensTilingData *tilingData);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> tmpOutQueue_;
    TBuf<QuePosition::VECCALC> maskBuf1_;
    TBuf<QuePosition::VECCALC> totalBuf_;
    TQue<QuePosition::VECIN, 1> workQueue_;

    // syncall before
    GlobalTensor<T> gmInput_;
    GlobalTensor<T> gmWorkspace_;
    GlobalTensor<T> gmOutput_;

    GM_ADDR workspace_;

    // multi-core sync
    GlobalTensor<int32_t> syncGlobal_;
    TQue<QuePosition::VECIN, 1> syncWorkQueue_;

    // [syncall before base param]
    int64_t sortedExpertNum_{0};
    int64_t handleNumPerCoreBefore_{0};
    int64_t handleNumTailCoreBefore_{0};
    int64_t workLocalNeedSize_{0};
    int64_t handleNum_{0};
    int64_t usedCoreNumBefore_{0};

    // input number
    int64_t totalCoreNum_{0};
    int64_t numOfExpert_{0};

    bool isPaddingBefore_{false};
    int64_t rightPaddingBefore_{0};

    const int64_t ONCE_ALGN_NUM_INT32{8};
    const int64_t ONCE_ALGN_NUM_INT256{64};
};

template <typename T>
__aicore__ inline int64_t MoeComputeExpertTokensInt32SS<T>::PadProcessInt32(int64_t param)
{
    return  ONCE_ALGN_NUM_INT32 - param % ONCE_ALGN_NUM_INT32;
}

template <typename T>
__aicore__ inline int64_t MoeComputeExpertTokensInt32SS<T>::Int32AlignmentProcess(int64_t param)
{
    return (param + ONCE_ALGN_NUM_INT32 - 1) / ONCE_ALGN_NUM_INT32 * ONCE_ALGN_NUM_INT32;
}

template <typename T>
__aicore__ inline int64_t MoeComputeExpertTokensInt32SS<T>::Int256AlignmentProcess(int64_t param)
{
    return (param + ONCE_ALGN_NUM_INT256 - 1) / ONCE_ALGN_NUM_INT256 * ONCE_ALGN_NUM_INT256;
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32SS<T>::ParseTilingData(
    const MoeComputeExpertTokensTilingData *tilingData)
{
    // 使用核数
    usedCoreNumBefore_ = tilingData->usedCoreNumBefore;

    // 输入专家个数
    sortedExpertNum_ = tilingData->sortedExpertNum;
    numOfExpert_ = tilingData->numOfExpert;

    // SyncAll前，尾核 & 非尾核 
    handleNumPerCoreBefore_ = tilingData->normalCoreHandleNumBefore;
    handleNumTailCoreBefore_ = tilingData->tailCoreHandleNumBefore;
    workLocalNeedSize_ = tilingData->workLocalNeedSize;

    // 使用核数信息
    totalCoreNum_ = tilingData->totalCoreNum;
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32SS<T>::Init(GM_ADDR sortExperts,
                                                              GM_ADDR out,
                                                              GM_ADDR workspace,
                                                              const MoeComputeExpertTokensTilingData* tilingData)
{
    // init tiling data
    ParseTilingData(tilingData);
    workspace_ = workspace;

    // SetGlobalBuffer
    gmInput_.SetGlobalBuffer((__gm__ T*)sortExperts + GetBlockIdx() * handleNumPerCoreBefore_);
    gmWorkspace_.SetGlobalBuffer((__gm__ T*)workspace);
    gmOutput_.SetGlobalBuffer((__gm__ T *)out);
    
    // InitBuffer
    handleNum_ = (GetBlockIdx() != usedCoreNumBefore_ - 1) ? handleNumPerCoreBefore_ : handleNumTailCoreBefore_;
    pipe_.InitBuffer(inputQueue_, 1, Int32AlignmentProcess(handleNum_) * sizeof(T));
    pipe_.InitBuffer(tmpOutQueue_, 1, Int32AlignmentProcess(numOfExpert_) * sizeof(T));
    pipe_.InitBuffer(maskBuf1_, Int32AlignmentProcess(handleNum_ * sizeof(uint8_t) / sizeof(float)) * sizeof(float));
    pipe_.InitBuffer(totalBuf_, 6 * Int256AlignmentProcess(handleNum_) * sizeof(float));
    pipe_.InitBuffer(workQueue_, 1, workLocalNeedSize_ * sizeof(float));
   
    // clear output
    if (GetBlockIdx() == 0) {
        InitOutput<int32_t>(gmOutput_, numOfExpert_, 0);
    }
    #ifndef __CCE_KT_TEST__
        AscendC::SyncAll();
    #endif

}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32SS<T>::CopyIn()
{
    LocalTensor<T> ubInput = inputQueue_.AllocTensor<T>();
    if (handleNum_ * sizeof(T) % 32) {
        isPaddingBefore_ = true;
        rightPaddingBefore_ = PadProcessInt32(handleNum_);
    }
    DataCopyParams copyParams {
        (uint16_t)1,
        (uint16_t)(handleNum_ * sizeof(T)),
        (uint16_t)0,
        (uint16_t)0
    };
    DataCopyPadParams padParams {
        isPaddingBefore_,
        (uint8_t)0,
        (uint8_t)rightPaddingBefore_,
        (uint8_t)0
    };
    DataCopyPad(ubInput, gmInput_, copyParams, padParams);
    inputQueue_.EnQue(ubInput);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32SS<T>::Compute()
{
    LocalTensor<T> input = inputQueue_.DeQue<T>();
    LocalTensor<T> output = tmpOutQueue_.AllocTensor<T>();
    LocalTensor<uint8_t> mask1 = maskBuf1_.Get<uint8_t>();
    LocalTensor<float> workLocal = workQueue_.AllocTensor<float>();

    LocalTensor<float> totalBuf = totalBuf_.Get<float>();
    LocalTensor<float> targetBuf = totalBuf[Int256AlignmentProcess(handleNum_) * 0];
    LocalTensor<float> oneBuf = totalBuf[Int256AlignmentProcess(handleNum_) * 1];
    LocalTensor<float> reduceMaxAnsBuf = totalBuf[Int256AlignmentProcess(handleNum_) * 2];
    LocalTensor<float> reduceSumAnsBuf = totalBuf[Int256AlignmentProcess(handleNum_) * 3];
    LocalTensor<float> inputCast = totalBuf[Int256AlignmentProcess(handleNum_) * 4];
    LocalTensor<float> resultBuf = totalBuf[Int256AlignmentProcess(handleNum_) * 5];

    uint64_t mask = 256 / sizeof(float);
    int32_t startIdx = 0;
    int32_t endIdx = startIdx + handleNum_ - 1;
    Duplicate(oneBuf, static_cast<float>(1), 8);
    Cast(inputCast, input, RoundMode::CAST_NONE, handleNum_);
    Duplicate(output, 0, numOfExpert_);
    
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    float startTarget = inputCast.GetValue(startIdx);
    float endTarget = inputCast.GetValue(endIdx);

    int32_t startOffset = handleNumPerCoreBefore_ * GetBlockIdx();
    int32_t targetLocation = 0;
    int32_t lastIdx = 0;
    int32_t lastVal = 0;
    
    int32_t repeat = (handleNum_ + mask - 1) / mask;
    BinaryRepeatParams repeatParamsCompare = {1, 1, 0, 8, 8, 0};
    BinaryRepeatParams repeatParamsSelect = {1, 0, 0, 8, 0, 0};

    for (float target = startTarget; target <= endTarget; target++)
    {
        Duplicate(targetBuf, target, 8);

        pipe_barrier(PIPE_V);
        Compare(mask1, inputCast, targetBuf, CMPMODE::EQ, mask, repeat, repeatParamsCompare);
        pipe_barrier(PIPE_V);
        Select<float>(resultBuf, mask1, oneBuf, 0, SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, repeat, repeatParamsSelect);
        pipe_barrier(PIPE_V);
        ReduceMax<float>(reduceMaxAnsBuf, resultBuf, workLocal, handleNum_, true);

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        bool isFind = static_cast<int32_t>(reduceMaxAnsBuf.GetValue(0)) != 0;
        if (!isFind) {
            output.SetValue(static_cast<int32_t>(target), lastVal);
            continue;
        }
        ReduceSum<float>(reduceSumAnsBuf, resultBuf, workLocal, handleNum_);      

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float index = reduceMaxAnsBuf.GetValue(1);
        float sumNum = reduceSumAnsBuf.GetValue(0);
        targetLocation = (reinterpret_cast<int32_t&>(index) + sumNum);
        lastVal = startOffset + targetLocation;
        lastIdx = target;
        output.SetValue(static_cast<int32_t>(target), lastVal);
    }
    for (int k = lastIdx + 1; k < numOfExpert_; k++) {
        output.SetValue(k, lastVal);
    }
    
    tmpOutQueue_.EnQue<T>(output);
    inputQueue_.FreeTensor(input);
    workQueue_.FreeTensor(workLocal);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32SS<T>::CopyOut()
{
    LocalTensor<T> output = tmpOutQueue_.DeQue<T>();
    uint16_t blockCount = 1;
    uint16_t blockLen = numOfExpert_ * sizeof(T);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
    set_atomic_s32();
    set_atomic_max();
    DataCopyPad(gmOutput_, output, dataCopyParams);
    SetAtomicNone();
    tmpOutQueue_.FreeTensor(output);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32SS<T>::Process()
{
    if (GetBlockIdx() < usedCoreNumBefore_) {
        CopyIn();
        Compute();
        CopyOut();
    }
}

}  // namespace Moe
#endif  // MOE_COMPUTE_EXPERT_TOKENS_INT32_SS