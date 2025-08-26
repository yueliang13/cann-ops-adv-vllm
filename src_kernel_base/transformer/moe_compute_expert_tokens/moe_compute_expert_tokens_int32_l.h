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
 * \file moe_compute_expert_tokens_int32_l.h
 * \brief
 */
#ifndef MOE_COMPUTE_EXPERT_TOKENS_INT32_L
#define MOE_COMPUTE_EXPERT_TOKENS_INT32_L

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeCompute {

using namespace AscendC;

template <typename T>
class MoeComputeExpertTokensInt32L {
public:
    __aicore__ inline MoeComputeExpertTokensInt32L(){};
    __aicore__ inline void Init(GM_ADDR sortExperts,
                                GM_ADDR out,
                                GM_ADDR workspace,
                                const MoeComputeExpertTokensTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessBefore();
    __aicore__ inline void ProcessAfter();
    __aicore__ inline void CopyInBefore(int64_t loop1Idx, int64_t loop2Idx);
    __aicore__ inline void CopyInTailCoreBefore(int64_t loop1Idx, int64_t loop2Idx);
    __aicore__ inline void CopyInTailCoreLastBefore(int64_t loop1Idx, int64_t loop2Idx);
    __aicore__ inline void ComputeBefore(int64_t loop1Idx, int64_t loop2Idx, LocalTensor<T> &output, int32_t indexOffset, int32_t handleNum);
    __aicore__ inline void ComputeTailCoreBefore(int64_t loop1Idx, int64_t loop2Idx, LocalTensor<T> &output, int32_t indexOffset, int32_t handleNum, int32_t startOffset);
    __aicore__ inline void CopyOutBefore(int64_t loop1Idx, int32_t handleNum);

    __aicore__ inline void CopyInAfter(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void ComputeAfter(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void CopyOutAfter(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline int64_t Int32AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcessInt32(int64_t param);
    __aicore__ inline void ParseTilingData(const MoeComputeExpertTokensTilingData *tilingData);
    __aicore__ inline void SyncAllCore();

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> tmpOutQueue_;
    TBuf<QuePosition::VECCALC> tmpOutputBuf_;

    TQue<QuePosition::VECIN, 1> workspaceQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;

    // syncall before
    GlobalTensor<T> gmInput_;

    LocalTensor<T> tbuf;

    // syncall before
    GlobalTensor<T> gmWorkspace_;
    GlobalTensor<T> gmOutput_;

    TBuf<QuePosition::VECCALC> inputCastTmpBuf_;
    TBuf<QuePosition::VECCALC> outputCastTmpBuf_;
    GM_ADDR workspace_;

    // multi-core sync
    GlobalTensor<int32_t> syncGlobal_;
    TQue<QuePosition::VECIN, 1> syncAllQueue_;

    // syncall before base param
    int64_t handleNumPerCoreBefore_{0};
    int64_t handleNumTailCoreBefore_{0};
    int64_t usedCoreNumBefore3_{0};

    int64_t loopCountBefore_{0};                  // 非尾核, 每个核处理sorted_expert时的loop次数
    int64_t loopCountTailCoreBefore_{0};          // 尾核, 处理sorted_expert时的loop次数
    int64_t handleNumPerLoopBefore_{0};          // 非尾核，每次loop处理的sorted_expert数量
    int64_t handleNumTailCorePerLoopBefore_{0};   // 尾核，每次loop处理的sorted_expert数量
    int64_t handleExpertNumLoopCount_{0};      // 切E需要的loop次数
    int64_t handleExpertNumMainCorePerLoop_{0};  // 每次loop，非最后一次切分处理的E的个数
    int64_t handleExpertNumTailCorePerLoop_{0};  // 每次loop，最后一次切分处理的E个数

    int64_t loopCountTailCoreMainLoop_{0};
    int64_t handleNumTailCoreMainLoop_{0};
    int64_t loopCountTailCoreTailLoop_{0};
    int64_t handleNumTailCoreTailLoop_{0};

    // syncall after base param
    // normal core
    int64_t normalCoreHandleNum_{0};
    int64_t normalCoreLoopNum_{0};
    int64_t normalCoreHandleNumPerLoop_{0};
    int64_t normalCoreHandleNumTailLoop_{0};

    // tail core
    int64_t tailCoreHandleNum_{0};
    int64_t tailCoreLoopNum_{0};
    int64_t tailCoreHandleNumPerLoop_{0};
    int64_t tailCoreHandleNumTailLoop_{0};

    int64_t curCoreHandleNumPerLoop_{0};
    int64_t curCoreHandleNumTailLoop_{0};
    int64_t curCoreHandleNum_{0};
    int64_t loopCount_{0};

    int64_t usedCoreNumAfter_{0};

    // input number
    int64_t totalCoreNum_{0};
    int64_t numOfExpert_{0};
    int64_t inputIndex_{0};
    int64_t outputIndex_{0};

    bool isPadding_{false};
    int64_t rightPadding_{0};
    bool isTailCore_{false};

    int64_t currVal_{0};
    int64_t prevVal_{1};
    int64_t lastVal_{0};

    const int64_t INT32_BYTES{4};
    const int64_t ONCE_ALGN_NUM_INT32{8};
    const int64_t PLACEHOLDER_NUM{7};
};

template <typename T>
__aicore__ inline int64_t MoeComputeExpertTokensInt32L<T>::PadProcessInt32(int64_t param)
{
    return  ONCE_ALGN_NUM_INT32 - param % ONCE_ALGN_NUM_INT32;
}

template <typename T>
__aicore__ inline int64_t MoeComputeExpertTokensInt32L<T>::Int32AlignmentProcess(int64_t param)
{
    return (param + ONCE_ALGN_NUM_INT32 - 1) / ONCE_ALGN_NUM_INT32 * ONCE_ALGN_NUM_INT32;
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::ParseTilingData(
    const MoeComputeExpertTokensTilingData *tilingData)
{
    numOfExpert_ = tilingData->numOfExpert;
    handleNumPerCoreBefore_ = tilingData->handleNumPerCoreBefore;
    handleNumTailCoreBefore_ = tilingData->handleNumTailCoreBefore;
    loopCountBefore_ = tilingData->loopCountBefore;                                // 非尾核, 每个核处理sorted_expert时的loop次数
    handleNumPerLoopBefore_ = tilingData->handleNumPerLoopBefore;                  // 非尾核，每次loop处理的sorted_expert数量
    usedCoreNumBefore3_ = tilingData->usedCoreNumBefore3;    
    handleExpertNumLoopCount_ = tilingData->handleExpertNumLoopCount;              // 切E需要的loop次数
    handleExpertNumMainCorePerLoop_ = tilingData->handleExpertNumMainCorePerLoop;  // 每次loop，非最后一次切分处理的E的个数
    handleExpertNumTailCorePerLoop_ = tilingData->handleExpertNumTailCorePerLoop;  // 每次loop，最后一次切分处理的E个数

    loopCountTailCoreMainLoop_ = tilingData->loopCountTailCoreMainLoop;
    handleNumTailCoreMainLoop_ = tilingData->handleNumTailCoreMainLoop;
    loopCountTailCoreTailLoop_ = tilingData->loopCountTailCoreTailLoop;
    handleNumTailCoreTailLoop_ = tilingData->handleNumTailCoreTailLoop;

    // 使用核数
    usedCoreNumAfter_ = tilingData->usedCoreNumAfter;

    // SyncAll后，非尾核
    normalCoreHandleNum_ = tilingData->normalCoreHandleNumAfter;
    normalCoreLoopNum_ = tilingData->normalCoreLoopNumAfter;
    normalCoreHandleNumPerLoop_ = tilingData->normalCoreHandleNumPerLoopAfter;
    normalCoreHandleNumTailLoop_ = tilingData->normalCoreHandleNumTailLoopAfter;

    // SyncAll后，尾核
    tailCoreHandleNum_ = tilingData->tailCoreHandleNumAfter;
    tailCoreLoopNum_ = tilingData->tailCoreLoopNumAfter;
    tailCoreHandleNumPerLoop_ = tilingData->tailCoreHandleNumPerLoopAfter;
    tailCoreHandleNumTailLoop_ = tilingData->tailCoreHandleNumTailLoopAfter;

    // 使用核数信息
    totalCoreNum_ = tilingData->totalCoreNum;
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::Init(GM_ADDR sortExperts,
                                                            GM_ADDR out,
                                                            GM_ADDR workspace,
                                                            const MoeComputeExpertTokensTilingData* tilingData)
{
    // init tiling data
    ParseTilingData(tilingData);
    workspace_ = workspace;

    // syncall before
    isTailCore_ = GetBlockIdx() == usedCoreNumBefore3_ - 1;

    gmInput_.SetGlobalBuffer((__gm__ T*)sortExperts + GetBlockIdx() * handleNumPerCoreBefore_);
    gmWorkspace_.SetGlobalBuffer((__gm__ T*)workspace);

    // gmWorkspace_清零
    int64_t n = numOfExpert_ * totalCoreNum_;
    int32_t initValue = 0;
    if ((GetBlockIdx() + 1) < GetBlockNum()) {
        InitOutput<int32_t>(gmWorkspace_[n / GetBlockNum() * GetBlockIdx()], n / GetBlockNum(), initValue);
    }
    if ((GetBlockIdx() + 1) == GetBlockNum()) {
        InitOutput<int32_t>(gmWorkspace_[n / GetBlockNum() * GetBlockIdx()], n - n / GetBlockNum() * (GetBlockNum() - 1), initValue);
    }
#ifndef __CCE_KT_TEST__
  AscendC::SyncAll();
#endif

    // 内存初始化
    int64_t handleNum = isTailCore_ ? handleNumTailCoreMainLoop_ : handleNumPerLoopBefore_;
    pipe_.InitBuffer(inputQueue_, 1, (handleNum * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    pipe_.InitBuffer(tmpOutQueue_, 1, ((PLACEHOLDER_NUM * (handleExpertNumMainCorePerLoop_ - 1) + handleExpertNumMainCorePerLoop_) * sizeof(T)  + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    pipe_.InitBuffer(tmpOutputBuf_, ((PLACEHOLDER_NUM * (handleExpertNumMainCorePerLoop_ - 1) + handleExpertNumMainCorePerLoop_) * sizeof(T)  + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);

    tbuf = tmpOutputBuf_.Get<T>();

    // syncall before
    if (GetBlockIdx() + 1 == usedCoreNumAfter_) {
        curCoreHandleNumPerLoop_ = tailCoreHandleNumPerLoop_;
        curCoreHandleNumTailLoop_ = tailCoreHandleNumTailLoop_;
        curCoreHandleNum_ = tailCoreHandleNum_;
        loopCount_ = tailCoreLoopNum_;
    } else {
        curCoreHandleNumPerLoop_ = normalCoreHandleNumPerLoop_;
        curCoreHandleNumTailLoop_ = normalCoreHandleNumTailLoop_;
        curCoreHandleNum_ = normalCoreHandleNum_;
        loopCount_ = normalCoreLoopNum_;
    }

    // output 初始化
    outputIndex_ = GetBlockIdx() * normalCoreHandleNum_;
    gmOutput_.SetGlobalBuffer((__gm__ T *)out + outputIndex_, curCoreHandleNum_);

    // 申请buffer空间
    pipe_.InitBuffer(workspaceQueue_, 1, curCoreHandleNumPerLoop_ * Int32AlignmentProcess(usedCoreNumBefore3_) * sizeof(int32_t));
    pipe_.InitBuffer(outputQueue_, 1, curCoreHandleNumPerLoop_ * sizeof(int32_t));

    pipe_.InitBuffer(inputCastTmpBuf_, curCoreHandleNumPerLoop_ * Int32AlignmentProcess(usedCoreNumBefore3_) * sizeof(float));
    pipe_.InitBuffer(outputCastTmpBuf_, curCoreHandleNumPerLoop_ * sizeof(float));
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::ComputeTailCoreBefore(int64_t loop1Idx, int64_t loop2Idx, LocalTensor<T> &output, int32_t indexOffset, int32_t handleNum, int32_t startOffset)
{
    LocalTensor<T> input = inputQueue_.DeQue<T>();
    int32_t startIdx = 0;
    int32_t endIdx = startIdx + indexOffset - 1;

    int32_t startTarget = loop1Idx * handleExpertNumMainCorePerLoop_;
    int32_t endTarget = startTarget + handleNum - 1;

    int32_t lastIdx = 0; // 最后一个可以找到的专家索引
    int32_t lastVal = lastVal_; // 最后一个可以找到的专家号

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    currVal_ = input.GetValue(startIdx);

    if (currVal_ > endTarget) {
        inputQueue_.FreeTensor(input);
        return;
    }

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (int32_t target = startTarget; target <= endTarget; target++) {
        int32_t low = startIdx;
        int32_t high = endIdx - startIdx;
        int32_t targetLocation = 0;
        int32_t mid = 0;
        while (low <= high) {
            mid = (low + high) / 2;
            if (input.GetValue(mid) > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
                targetLocation = mid;
            }
        }
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        // 可以找到
        if (input.GetValue(targetLocation) == target) {
            Duplicate(output[(target - handleExpertNumMainCorePerLoop_ * loop1Idx) * ONCE_ALGN_NUM_INT32], startOffset + targetLocation + 1, 1);
            lastIdx = target;
            lastVal = startOffset + targetLocation + 1;
            lastVal_ = lastVal;
        } else {
            // target找不到，该位置数置为0
            Duplicate(output[(target - handleExpertNumMainCorePerLoop_ * loop1Idx) * ONCE_ALGN_NUM_INT32], lastVal, 1);
        }
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    }
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    if (currVal_ >= 0 && currVal_ >= prevVal_ && currVal_ >= startTarget && currVal_ - handleExpertNumMainCorePerLoop_ * loop1Idx > 0) {
        Muls(output, tbuf, 1, (currVal_ - handleExpertNumMainCorePerLoop_ * loop1Idx) * ONCE_ALGN_NUM_INT32);
        pipe_barrier(PIPE_V);
    }

    prevVal_ = input.GetValue(endIdx);
    Muls(tbuf, output, 1, output.GetSize());

    inputQueue_.FreeTensor(input);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::ComputeBefore(int64_t loop1Idx, int64_t loop2Idx, LocalTensor<T> &output, int32_t indexOffset, int32_t handleNum)
{
    LocalTensor<T> input = inputQueue_.DeQue<T>();

    int32_t startIdx = 0;
    int32_t endIdx = startIdx + indexOffset - 1;

    int32_t startTarget = loop1Idx * handleExpertNumMainCorePerLoop_;
    int32_t endTarget = startTarget + handleNum - 1;

    int32_t lastIdx = 0; // 最后一个可以找到的专家索引
    int32_t lastVal = lastVal_; // 最后一个可以找到的专家号

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    currVal_ = input.GetValue(startIdx);

    if (currVal_ > endTarget) {
        inputQueue_.FreeTensor(input);
        return;
    }

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (int32_t target = startTarget; target <= endTarget; target++) {
        int32_t low = startIdx;
        int32_t high = endIdx - startIdx;
        int32_t targetLocation = 0;
        int32_t mid = 0;
        while (low <= high) {
            mid = (low + high) / 2;
            if (input.GetValue(mid) > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
                targetLocation = mid;
            }
        }
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        // 可以找到
        if (input.GetValue(targetLocation) == target) {
            int32_t startOffset = handleNumPerCoreBefore_ * GetBlockIdx() + loop2Idx * handleNumPerLoopBefore_;
            Duplicate(output[(target - handleExpertNumMainCorePerLoop_ * loop1Idx) * ONCE_ALGN_NUM_INT32], startOffset + targetLocation + 1, 1);
            lastIdx = target;
            lastVal = startOffset + targetLocation + 1;
            lastVal_ = lastVal;
        } else {
            // target找不到，该位置数置为0
            Duplicate(output[(target - handleExpertNumMainCorePerLoop_ * loop1Idx) * ONCE_ALGN_NUM_INT32], lastVal, 1);
        }
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    }
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    if (currVal_ >= 0 && currVal_ >= prevVal_ && currVal_ >= startTarget && currVal_ - handleExpertNumMainCorePerLoop_ * loop1Idx > 0) {
        Muls(output, tbuf, 1, (currVal_ - handleExpertNumMainCorePerLoop_ * loop1Idx) * ONCE_ALGN_NUM_INT32);
        pipe_barrier(PIPE_V);
    }

    prevVal_ = input.GetValue(endIdx);
    Muls(tbuf, output, 1, output.GetSize());

    inputQueue_.FreeTensor(input);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::CopyOutBefore(int64_t loop1Idx, int32_t handleNum)
{
    LocalTensor<T> output = tmpOutQueue_.DeQue<T>();
    uint16_t blockCount = handleNum;
    uint16_t blockLen = sizeof(T);
    uint16_t srcStride = 0;
    uint16_t dstStride = (usedCoreNumBefore3_ - 1) * sizeof(T);
    DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
    DataCopyPad(gmWorkspace_[GetBlockIdx() + (handleExpertNumMainCorePerLoop_ * loop1Idx) * usedCoreNumBefore3_], output, dataCopyParams);
    tmpOutQueue_.FreeTensor(output);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::SyncAllCore()
{
    // set workspace as 0, each core handle workspace 32bytes
    constexpr int32_t each_core_handle_num = ONE_BLK_SIZE / sizeof(int32_t);
    syncGlobal_.SetGlobalBuffer((__gm__ int32_t*)workspace_+ (numOfExpert_ * usedCoreNumBefore3_));
    InitOutput<int32_t>(syncGlobal_[each_core_handle_num * GetBlockIdx()], each_core_handle_num, static_cast<int32_t>(0));
    #ifndef __CCE_KT_TEST__
        AscendC::SyncAll();
    #endif
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::CopyInBefore(int64_t loop1Idx, int64_t loop2Idx)
{
    bool isPadding = false;
    int64_t rightPadding = 0;
    LocalTensor<T> ubInput = inputQueue_.AllocTensor<T>();
    if (handleNumPerLoopBefore_ * sizeof(T) % 32) {
        isPadding = true;
        rightPadding = PadProcessInt32(handleNumPerLoopBefore_);
    }

    DataCopyParams copyParams {
        (uint16_t)(1),
        (uint16_t)(handleNumPerLoopBefore_ * sizeof(T)),
        (uint16_t)0,
        (uint16_t)0
    };
    DataCopyPadParams padParams {
        isPadding,
        (uint8_t)0,
        (uint8_t)rightPadding,
        (uint8_t)0
    };
    DataCopyPad(ubInput, gmInput_[handleNumPerLoopBefore_ * loop2Idx], copyParams, padParams);
    inputQueue_.EnQue(ubInput);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::CopyInTailCoreBefore(int64_t loop1Idx, int64_t loop2Idx)
{
    bool isPadding = false;
    int64_t rightPadding = 0;
    LocalTensor<T> ubInput = inputQueue_.AllocTensor<T>();
    if (handleNumTailCoreMainLoop_ * sizeof(T) % 32) {
        isPadding = true;
        rightPadding = PadProcessInt32(handleNumTailCoreMainLoop_);
    }

    DataCopyParams copyParams {
        (uint16_t)(1),
        (uint16_t)(handleNumTailCoreMainLoop_ * sizeof(T)),
        (uint16_t)0,
        (uint16_t)0
    };
    DataCopyPadParams padParams {
        isPadding,
        (uint8_t)0,
        (uint8_t)rightPadding,
        (uint8_t)0
    };
    DataCopyPad(ubInput, gmInput_[handleNumTailCoreMainLoop_ * loop2Idx], copyParams, padParams);
    inputQueue_.EnQue(ubInput);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::CopyInTailCoreLastBefore(int64_t loop1Idx, int64_t loop2Idx)
{
    bool isPadding = false;
    int64_t rightPadding = 0;
    LocalTensor<T> ubInput = inputQueue_.AllocTensor<T>();
    if (handleNumTailCoreTailLoop_ * sizeof(T) % 32) {
        isPadding = true;
        rightPadding = PadProcessInt32(handleNumTailCoreTailLoop_);
    }

    DataCopyParams copyParams {
        (uint16_t)(1),
        (uint16_t)(handleNumTailCoreTailLoop_ * sizeof(T)),
        (uint16_t)0,
        (uint16_t)0
    };
    DataCopyPadParams padParams {
        isPadding,
        (uint8_t)0,
        (uint8_t)rightPadding,
        (uint8_t)0
    };
    int64_t offset = handleNumTailCoreMainLoop_ * loopCountTailCoreMainLoop_;
    DataCopyPad(ubInput, gmInput_[offset + handleNumTailCoreTailLoop_ * loop2Idx], copyParams, padParams);
    inputQueue_.EnQue(ubInput);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::ProcessBefore()
{
    if (GetBlockIdx() >= usedCoreNumBefore3_) {
        return;
    }

    if (isTailCore_) {
        int64_t loops1 = handleExpertNumLoopCount_;
        int64_t loops2 = loopCountTailCoreMainLoop_;
        int32_t startOffset = 0;
        int32_t idxOffset = handleNumTailCoreMainLoop_;
        int32_t handleExpertNum = 0;
        for (int64_t i = 0; i < loops1; i++) {
            LocalTensor<T> output = tmpOutQueue_.AllocTensor<T>();
            handleExpertNum = (i != loops1 - 1) ? handleExpertNumMainCorePerLoop_ : handleExpertNumTailCorePerLoop_;
            Duplicate(tbuf, 0, (PLACEHOLDER_NUM * (handleExpertNumMainCorePerLoop_ - 1) + handleExpertNumMainCorePerLoop_));
            Duplicate(output, 0, (PLACEHOLDER_NUM * (handleExpertNumMainCorePerLoop_ - 1) + handleExpertNumMainCorePerLoop_));
            idxOffset = handleNumTailCoreMainLoop_;
            // 尾核，主loop
            for (int64_t j = 0; j < loops2; j++) {
                startOffset = handleNumPerCoreBefore_ * GetBlockIdx() + j * handleNumTailCoreMainLoop_;
                CopyInTailCoreBefore(i, j);
                ComputeTailCoreBefore(i, j, output, idxOffset, handleExpertNum, startOffset);
            }
            // 尾核，尾loop
            int64_t tailLoop = loopCountTailCoreTailLoop_;
            idxOffset = handleNumTailCoreTailLoop_;
            for (int64_t k = 0; k < tailLoop; k++) {
                startOffset = handleNumPerCoreBefore_ * GetBlockIdx() + loopCountTailCoreMainLoop_ * handleNumTailCoreMainLoop_;
                CopyInTailCoreLastBefore(i, k);
                ComputeTailCoreBefore(i, k, output, idxOffset, handleExpertNum, startOffset);
            }
            tmpOutQueue_.EnQue<T>(output);
            CopyOutBefore(i, handleExpertNum);
        }
    } else {
        int64_t loops1 = handleExpertNumLoopCount_;
        int64_t loops2 = loopCountBefore_;
        int32_t idxOffset = handleNumPerLoopBefore_;
        int32_t handleExpertNum = 0;
        for (int64_t i = 0; i < loops1; i++) {
            LocalTensor<T> output = tmpOutQueue_.AllocTensor<T>();
            handleExpertNum = (i != loops1 - 1) ? handleExpertNumMainCorePerLoop_ : handleExpertNumTailCorePerLoop_;
            Duplicate(tbuf, 0, (PLACEHOLDER_NUM * (handleExpertNumMainCorePerLoop_ - 1) + handleExpertNumMainCorePerLoop_));
            Duplicate(output, 0, (PLACEHOLDER_NUM * (handleExpertNumMainCorePerLoop_ - 1) + handleExpertNumMainCorePerLoop_));
            for (int64_t j = 0; j < loops2; j++) {
                CopyInBefore(i, j);
                ComputeBefore(i, j, output, idxOffset, handleExpertNum);
            }
            tmpOutQueue_.EnQue<T>(output);
            CopyOutBefore(i, handleExpertNum);
        }
    }
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::CopyInAfter(int64_t nLoopIdx, int64_t numOfLoop)
{
    LocalTensor<T> inputLocal = workspaceQueue_.AllocTensor<T>();
    if (usedCoreNumBefore3_ * sizeof(T) % 32) {
        isPadding_ = true;
        rightPadding_ = PadProcessInt32(usedCoreNumBefore3_);
    }

    DataCopyParams copyParams {
        (uint16_t)(numOfLoop),
        (uint16_t)(usedCoreNumBefore3_ * sizeof(T)),
        (uint16_t)0,
        (uint16_t)0
    };
    DataCopyPadParams padParams {
        isPadding_,
        (uint8_t)0,
        (uint8_t)rightPadding_,
        (uint8_t)0
    };
    
    DataCopyPad(inputLocal, gmWorkspace_[nLoopIdx * curCoreHandleNumPerLoop_ * usedCoreNumBefore3_], copyParams, padParams);
    workspaceQueue_.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::ComputeAfter(int64_t nLoopIdx, int64_t numOfLoop)
{
    LocalTensor<T> inputLocal = workspaceQueue_.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();

    LocalTensor<float> inputCastTmpUb = inputCastTmpBuf_.Get<float>();
    LocalTensor<float> outputCastTmpUb = outputCastTmpBuf_.Get<float>();

    Cast(inputCastTmpUb, inputLocal, RoundMode::CAST_NONE, numOfLoop * Int32AlignmentProcess(usedCoreNumBefore3_));
    pipe_barrier(PIPE_V);
    uint64_t mask = usedCoreNumBefore3_;
    int32_t repeatTimes = numOfLoop;
    int32_t dstRepStride = 1;
    int32_t srcBlkStride = 1;
    int32_t srcRepStride = Int32AlignmentProcess(usedCoreNumBefore3_) * sizeof(float) / 32;
    WholeReduceMax<float>(outputCastTmpUb, inputCastTmpUb, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride,
                          ReduceOrder::ORDER_ONLY_VALUE);

    pipe_barrier(PIPE_V);
    Cast(outputLocal, outputCastTmpUb, RoundMode::CAST_ROUND, Int32AlignmentProcess(numOfLoop));
    outputQueue_.EnQue(outputLocal);
    workspaceQueue_.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::CopyOutAfter(int64_t nLoopIdx, int64_t numOfLoop)
{
    LocalTensor<T> outLocal = outputQueue_.DeQue<T>();
    DataCopyParams copyParamsOut{
        (uint16_t)1,
        (uint16_t)(numOfLoop * sizeof(T)),
        (uint16_t)0,
        (uint16_t)0
    };
    DataCopyPad(gmOutput_[nLoopIdx * curCoreHandleNumPerLoop_], outLocal, copyParamsOut);
    outputQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::ProcessAfter()
{
    if (GetBlockIdx() >= usedCoreNumAfter_) {
        return;
    }

    int64_t inputIdx = GetBlockIdx() * normalCoreHandleNum_ * usedCoreNumBefore3_;
    gmWorkspace_.SetGlobalBuffer((__gm__ T *)workspace_ + inputIdx, curCoreHandleNum_ * Int32AlignmentProcess(usedCoreNumBefore3_));

    auto numOfLoop = curCoreHandleNumPerLoop_;
    for (int64_t n = 0; n < loopCount_; n++) {
        if (n == (loopCount_ - 1)) {
            numOfLoop = curCoreHandleNumTailLoop_;
        }
        CopyInAfter(n, numOfLoop);
        ComputeAfter(n, numOfLoop);
        CopyOutAfter(n, numOfLoop);
    }
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32L<T>::Process()
{
    ProcessBefore();
    #ifndef __CCE_KT_TEST__
        AscendC::SyncAll();
    #endif
    ProcessAfter();
}

}  // namespace Moe
#endif  // MOE_COMPUTE_EXPERT_TOKENS_INT32_L
