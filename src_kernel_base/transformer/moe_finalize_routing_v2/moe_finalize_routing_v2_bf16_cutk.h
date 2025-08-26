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
 * \file moe_finalize_routing_v2_bf16_cutk.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_BF16_CUTK
#define MOE_FINALIZE_ROUTING_V2_BF16_CUTK

#include "moe_finalize_routing_v2_common.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeFinalizeRoutingV2 {

using namespace AscendC;

template <typename T, const bool ISBIASEXIST>
class MoeFinalizeRoutingV2Bf16CutK {
public:
    __aicore__ inline MoeFinalizeRoutingV2Bf16CutK(
        const MoeFinalizeRoutingV2TilingData &tilingData,
        TPipe &pipe) : pipe_(pipe), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, 
        GM_ADDR skip2, GM_ADDR bias,GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CkechColAlignment();
    __aicore__ inline void CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void Compute(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline int64_t AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcessT(int64_t param);

private:
    TPipe &pipe_;
    const MoeFinalizeRoutingV2TilingData &tilingData_;
    TQue<QuePosition::VECIN, 1> skip1Queue_;
    TQue<QuePosition::VECIN, 1> skip2Queue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    TBuf<QuePosition::VECCALC> expertForSourceRowBuf_;
    TBuf<QuePosition::VECCALC> scalesBuf_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb_;
    TBuf<QuePosition::VECCALC> biasBufDb_;

    TBuf<QuePosition::VECCALC> skip1CastBuf_;
    TBuf<QuePosition::VECCALC> skip2CastBuf_;
    TBuf<QuePosition::VECCALC> biasCastBuf_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsCastBuf_;

    GlobalTensor<T> gmExpandedPermutedRows_;
    GlobalTensor<T> gmSkip1_;
    GlobalTensor<T> gmSkip2_;
    GlobalTensor<T> gmBias_;
    GlobalTensor<T> gmScales_;
    GlobalTensor<int32_t> gmExpandedSrcToDstRow_;
    GlobalTensor<int32_t> gmExpertForSourceRow_;
    GlobalTensor<T> gmOut_;

    int64_t workspace_{0};

    // input number
    int64_t inputSkipIdx_{0};
    int64_t inputScalesAndExpertIdx_{0};
    int64_t outputIdx_{0};
    int64_t curCoreHandleNumPerLoop_{0};
    int64_t biasInCore_{0};

    const int64_t onceAlgnNum_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T))};

    bool isPadNormalH_ = false;
    bool isPadUnnormalH_ = false;
    bool isPadNormalK_ = false;
    bool isPadUnnormalK_ = false;
    bool isPadNormalKInt32_ = false;
    bool isPadUnnormalKInt32_ = false;
    bool initFlag_ = true;
    bool skipsAllNone_ = false;
    uint16_t rightPaddingNormalH_ = 0;
    uint16_t rightPaddingUnnormalH_ = 0;
    uint16_t rightPaddingNormalK_ = 0;
    uint16_t rightPaddingUnnormalK_ = 0;
    uint16_t rightPaddingNormalKInt32_ = 0;
    uint16_t rightPaddingUnnormalKInt32_ = 0;
    uint16_t rightPaddingKInt32_ = 0;
};

template <typename T, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2Bf16CutK<T, ISBIASEXIST>::AlignmentProcess(int64_t param)
{
    return (param + onceAlgnNum_ - 1) / onceAlgnNum_ * onceAlgnNum_;
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2Bf16CutK<T, ISBIASEXIST>::PadProcessT(int64_t param)
{
    return  onceAlgnNum_ - param % onceAlgnNum_;
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CutK<T, ISBIASEXIST>::CkechColAlignment()
{
    if (tilingData_.normalH * sizeof(T) % ONE_BLK_SIZE) {
        isPadNormalH_ = true;
        rightPaddingNormalH_ = PadProcessT(tilingData_.normalH);
    }

    if (tilingData_.unnormalH * sizeof(T) % ONE_BLK_SIZE) {
        isPadUnnormalH_ = true;
        rightPaddingUnnormalH_ = PadProcessT(tilingData_.unnormalH);
    }

    if (tilingData_.normalK * sizeof(T) % ONE_BLK_SIZE) {
        isPadNormalK_ = true;
        rightPaddingNormalK_ = PadProcessT(tilingData_.normalK);
    }

    if (tilingData_.unnormalK * sizeof(T) % ONE_BLK_SIZE) {
        isPadUnnormalK_ = true;
        rightPaddingUnnormalK_ = PadProcessT(tilingData_.unnormalK);
    }

    if (tilingData_.normalK * INT32_BYTES % ONE_BLK_SIZE) {
        isPadNormalKInt32_ = true;
        rightPaddingNormalKInt32_ = PadProcessInt32(tilingData_.normalK);
    }

    if (tilingData_.unnormalK * INT32_BYTES % ONE_BLK_SIZE) {
        isPadUnnormalKInt32_ = true;
        rightPaddingUnnormalKInt32_ = PadProcessInt32(tilingData_.unnormalK);
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CutK<T, ISBIASEXIST>::Init(GM_ADDR expandedPermutedRows, 
    GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, GM_ADDR skip2,
    GM_ADDR bias, GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out,
    GM_ADDR workspace)
{
    if (GetBlockIdx() + 1 == tilingData_.usedCoreNum) {
        curCoreHandleNumPerLoop_ = tilingData_.tailCoreHandleNumPerLoop;
    } else {
        curCoreHandleNumPerLoop_ = tilingData_.normalCoreHandleNumPerLoop;
    }
    biasInCore_ = GetBlockIdx() * tilingData_.normalCoreHandleNum;

    // 检查要处理的列数是否对齐以及应该如何对齐
    CkechColAlignment();

    // gmInput分核 && 输入偏移量初始化
    inputSkipIdx_ = biasInCore_ * tilingData_.H;
    if (tilingData_.skip1IsNull == 0) {
        gmSkip1_.SetGlobalBuffer((__gm__ T *)skip1 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);
    }
    pipe_.InitBuffer(skip1Queue_, 1, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));
    if (tilingData_.skip2IsNull == 0) {
        gmSkip2_.SetGlobalBuffer((__gm__ T *)skip2 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);
        pipe_.InitBuffer(skip2Queue_, 1, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));
        pipe_.InitBuffer(skip2CastBuf_, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(float));
    }

    inputScalesAndExpertIdx_ = biasInCore_ * tilingData_.K;
    if (tilingData_.scalesIsNull == 0) {
        gmScales_.SetGlobalBuffer((__gm__ T *)scales + inputScalesAndExpertIdx_, tilingData_.normalCoreHandleNum * tilingData_.K);
        pipe_.InitBuffer(scalesBuf_, AlignmentProcess(tilingData_.normalK) * sizeof(T));
    }

    gmExpandedSrcToDstRow_.SetGlobalBuffer((__gm__ int32_t *)expandedSrcToDstRow, tilingData_.totalRowNum * tilingData_.K);

    outputIdx_ = biasInCore_ * tilingData_.H;
    gmOut_.SetGlobalBuffer((__gm__ T *)out + outputIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);

    gmExpandedPermutedRows_.SetGlobalBuffer((__gm__ T *)expandedPermutedRows, tilingData_.totalRowNum * tilingData_.K * tilingData_.H);
    if constexpr (ISBIASEXIST) {
        gmBias_.SetGlobalBuffer((__gm__ T *)bias, tilingData_.biasRowNum * tilingData_.H);
        pipe_.InitBuffer(biasBufDb_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
        pipe_.InitBuffer(biasCastBuf_, AlignmentProcess(tilingData_.normalH) * sizeof(float));
        gmExpertForSourceRow_.SetGlobalBuffer((__gm__ int32_t *)expertForSourceRow + inputScalesAndExpertIdx_,
                                        tilingData_.normalCoreHandleNum * tilingData_.K);
        pipe_.InitBuffer(expertForSourceRowBuf_, Int32AlignmentProcess(tilingData_.normalK) * sizeof(int32_t));
    }

    // 申请 buffer 空间
    pipe_.InitBuffer(outQueue_, BUFFER_NUM, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb_, AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(skip1CastBuf_, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(float));

    pipe_.InitBuffer(expandedPermutedRowsCastBuf_, AlignmentProcess(tilingData_.normalH) * sizeof(float));
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CutK<T, ISBIASEXIST>::CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    bool isNormalH = (nLoopIdx + 1) % (tilingData_.hSliceNum + 1) != 0;
    int64_t bias = isNormalH ? (nLoopIdx % tilingData_.hSliceNum) * tilingData_.normalH : tilingData_.hSliceNum * tilingData_.normalH;
    int64_t dataLen = isNormalH ? tilingData_.normalH : tilingData_.unnormalH;
    int64_t rightPaddingH = isNormalH ? rightPaddingNormalH_ : rightPaddingUnnormalH_;
    bool isPadH = isNormalH ? isPadNormalH_ : isPadUnnormalH_;
    

    DataCopyParams copyParamsSkip{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(dataLen * sizeof(T)),
                                  0, 0};
    DataCopyPadParams padParamsSkip{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};

    if (tilingData_.hSliceNum == 0) {
        bias = 0;
    }
    LocalTensor<T> skip1Local;
    if (tilingData_.skip1IsNull == 0) {
        skip1Local = skip1Queue_.AllocTensor<T>();
#ifndef __CCE_KT_TEST__
        DataCopyPad(skip1Local, gmSkip1_[nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_ * tilingData_.H + bias],
            copyParamsSkip, padParamsSkip);
#endif

        skip1Queue_.EnQue(skip1Local);
    }
    
    LocalTensor<T> skip2Local;
    if (tilingData_.skip2IsNull == 0) {
        skip2Local = skip2Queue_.AllocTensor<T>();
#ifndef __CCE_KT_TEST__
        DataCopyPad(skip2Local, gmSkip2_[nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_ * tilingData_.H + bias],
            copyParamsSkip, padParamsSkip);
#endif
        skip2Queue_.EnQue(skip2Local);
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CutK<T, ISBIASEXIST>::Compute(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    bool isNormalH = (nLoopIdx + 1) % (tilingData_.hSliceNum + 1) != 0;
    int64_t bias = isNormalH ? (nLoopIdx % tilingData_.hSliceNum) * tilingData_.normalH : tilingData_.hSliceNum * tilingData_.normalH;
    if (tilingData_.hSliceNum == 0) {
        bias = 0;
    }
    int64_t dataLen = isNormalH ? tilingData_.normalH : tilingData_.unnormalH;
    int64_t rightPaddingH = isNormalH ? rightPaddingNormalH_ : rightPaddingUnnormalH_;
    bool isPadH = isNormalH ? isPadNormalH_ : isPadUnnormalH_;
    int64_t biasInRow =  nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_;
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
    
    LocalTensor<float> skip1CastUb = skip1CastBuf_.Get<float>();
    LocalTensor<T> skip1Local;
    if (tilingData_.skip1IsNull == 0) {
        skip1Local = skip1Queue_.DeQue<T>();
        Cast(skip1CastUb, skip1Local, RoundMode::CAST_NONE, curRepeatTimes * AlignmentProcess(dataLen));
        pipe_barrier(PIPE_V);
    }

    LocalTensor<T> skip2Local;
    LocalTensor<float> skip2CastUb;
    if (tilingData_.skip2IsNull == 0) {
        skip2Local = skip2Queue_.DeQue<T>();
        skip2CastUb = skip2CastBuf_.Get<float>();
        Cast(skip2CastUb, skip2Local, RoundMode::CAST_NONE, curRepeatTimes * AlignmentProcess(dataLen));
        pipe_barrier(PIPE_V);
    }
    
    if ((tilingData_.skip1IsNull == 0) && (tilingData_.skip2IsNull == 0)) {
        Add(skip1CastUb, skip1CastUb, skip2CastUb, curRepeatTimes * AlignmentProcess(dataLen));
    } else if ((tilingData_.skip1IsNull == 0) && (tilingData_.skip2IsNull == 1)) {
        Adds(skip1CastUb, skip1CastUb, (float)0.0, curRepeatTimes * AlignmentProcess(dataLen));
    } else if ((tilingData_.skip1IsNull == 1) && (tilingData_.skip2IsNull == 0)) {
        Adds(skip1CastUb, skip2CastUb, (float)0.0, curRepeatTimes * AlignmentProcess(dataLen));
    } else {
        initFlag_ = false;
        skipsAllNone_ = true;
    }

    LocalTensor<T> expandedPermutedTmpUb = expandedPermutedRowsBufDb_.Get<T>();

    LocalTensor<T> biasTmpUb;
    LocalTensor<float> biasCastUb;
    if constexpr (ISBIASEXIST) {
        biasTmpUb = biasBufDb_.Get<T>();
        biasCastUb = biasCastBuf_.Get<float>();
    }
    
    LocalTensor<float> expandedPermutedRowsCastUb = expandedPermutedRowsCastBuf_.Get<float>();
    

    DataCopyParams copyParams{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};
    for (int64_t i = 0; i < curRepeatTimes; i++) {
        int64_t outRowIndex = i * AlignmentProcess(dataLen);
        int64_t len = tilingData_.normalK;
        bool isPadK = isPadNormalK_;
        bool isPadKInt32 = isPadNormalKInt32_;
        uint16_t rightPaddingK = rightPaddingNormalK_;
        uint16_t rightPaddingKInt32 = rightPaddingNormalKInt32_;
        if (skipsAllNone_) {
            initFlag_ = false;
        }
        for (int64_t n = 0; n < tilingData_.kSliceNum; n++) {
            if (n == tilingData_.kSliceNum - 1) {
                len = tilingData_.unnormalK;
                isPadK = isPadUnnormalK_;
                isPadKInt32 = isPadUnnormalKInt32_;
                rightPaddingK = rightPaddingUnnormalK_;
                rightPaddingKInt32 = rightPaddingUnnormalKInt32_;
            }
            int64_t biasOfK= n * tilingData_.normalK;
            // ---------------------------- [Scales] -------------------------------
            LocalTensor<T> scalesLocal;
            if (tilingData_.scalesIsNull == 0) {
                scalesLocal = scalesBuf_.Get<T>();
                DataCopyParams copyParamsScales {1, static_cast<uint16_t>(len * sizeof(T)), 0, 0};
                DataCopyPadParams padParamsScales {isPadK, 0, static_cast<uint8_t>(rightPaddingK), 0};
#ifndef __CCE_KT_TEST__
                DataCopyPad(scalesLocal, gmScales_[biasInRow * tilingData_.K + i * tilingData_.K + biasOfK], copyParamsScales, padParamsScales);
#endif
            }
            // ---------------------------- [Expert] -------------------------------
            LocalTensor<int32_t> expertForSourceRowLocal;
            if constexpr (ISBIASEXIST) {
                expertForSourceRowLocal = expertForSourceRowBuf_.Get<int32_t>();
                DataCopyParams copyParamsExpert {1, static_cast<uint16_t>(len * sizeof(int32_t)), 0, 0};
                DataCopyPadParams padParamsExpert {isPadKInt32, 0, static_cast<uint8_t>(rightPaddingKInt32), 0};
#ifndef __CCE_KT_TEST__
                DataCopyPad(expertForSourceRowLocal, gmExpertForSourceRow_[biasInRow * tilingData_.K + i * tilingData_.K + biasOfK],
                    copyParamsExpert, padParamsExpert);
#endif
            }
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            set_flag(PIPE_V, PIPE_S, EVENT_ID1);
            for (int64_t j = 0; j < len; j++) {
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                int64_t expandedSrcToDstRowIndex = 0;
                if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
                    expandedSrcToDstRowIndex = biasInRow + i + (j + biasOfK) * tilingData_.totalRowNum + biasInCore_;
                } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
                    expandedSrcToDstRowIndex = biasInRow * tilingData_.K + i * tilingData_.K + j + biasOfK + 
                                                        GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
                }
                int64_t expandedPermutedRowsIndex = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndex);
                set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

                wait_flag(PIPE_V, PIPE_S, EVENT_ID1);

                int64_t biasIndex = 0;
                if constexpr (ISBIASEXIST) {
                    biasIndex = expertForSourceRowLocal.GetValue(j);
                }
                float scalesVal = 1.0;
                if (tilingData_.scalesIsNull == 0) {
                    scalesVal = ToFloat(scalesLocal.GetValue(j));
                }
                set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);

                wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
                if (expandedPermutedRowsIndex != INVALID_ROW_INDEX) {
#ifndef __CCE_KT_TEST__
                    DataCopyPad(expandedPermutedTmpUb, gmExpandedPermutedRows_[expandedPermutedRowsIndex * tilingData_.H + bias],
                        copyParams, padParams);
#endif

                }
                if constexpr (ISBIASEXIST) {
#ifndef __CCE_KT_TEST__
                    DataCopyPad(biasTmpUb, gmBias_[biasIndex * tilingData_.H + bias], copyParams, padParams);
#endif
                }
                
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if (expandedPermutedRowsIndex != INVALID_ROW_INDEX) {
                    Cast(expandedPermutedRowsCastUb, expandedPermutedTmpUb, RoundMode::CAST_NONE, dataLen);
                }
                if constexpr (ISBIASEXIST) {
                    Cast(biasCastUb, biasTmpUb, RoundMode::CAST_NONE, dataLen);
                }
                pipe_barrier(PIPE_V);
                if (expandedPermutedRowsIndex == INVALID_ROW_INDEX && 
                    (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                    Duplicate(expandedPermutedRowsCastUb, (float)0.0, dataLen);
                    pipe_barrier(PIPE_V);
                }
                if constexpr (ISBIASEXIST) {
                    Add(expandedPermutedRowsCastUb, expandedPermutedRowsCastUb, biasCastUb, dataLen);
                } else {
                    Adds(expandedPermutedRowsCastUb, expandedPermutedRowsCastUb, (float)0.0, dataLen);
                }
                pipe_barrier(PIPE_V);
                if (!initFlag_) {
                    Muls(skip1CastUb[outRowIndex], expandedPermutedRowsCastUb, scalesVal, dataLen);
                    initFlag_ = true;
                    set_flag(PIPE_V, PIPE_S, EVENT_ID1);
                } else {
                    Muls(expandedPermutedRowsCastUb, expandedPermutedRowsCastUb, scalesVal, dataLen);
                    set_flag(PIPE_V, PIPE_S, EVENT_ID1);
                    pipe_barrier(PIPE_V);
                    Add(skip1CastUb[outRowIndex], skip1CastUb[outRowIndex], expandedPermutedRowsCastUb, dataLen);
                }
            }
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
        }
    }
    pipe_barrier(PIPE_V);
    Cast(outLocal, skip1CastUb, RoundMode::CAST_ROUND, curRepeatTimes * AlignmentProcess(dataLen));
    outQueue_.EnQue(outLocal);
    
    if (tilingData_.skip1IsNull == 0) {
        skip1Queue_.FreeTensor(skip1Local);
    }
    if (tilingData_.skip2IsNull == 0) {
        skip2Queue_.FreeTensor(skip2Local);
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CutK<T, ISBIASEXIST>::CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    bool isNormalH = (nLoopIdx + 1) % (tilingData_.hSliceNum + 1) != 0;
    int64_t bias = isNormalH ? (nLoopIdx % tilingData_.hSliceNum) * tilingData_.normalH : tilingData_.hSliceNum * tilingData_.normalH;
    if (tilingData_.hSliceNum == 0) {
        bias = 0;
    }
    int64_t dataLen = isNormalH ? tilingData_.normalH : tilingData_.unnormalH;
    LocalTensor<T> outLocal = outQueue_.DeQue<T>();
    DataCopyParams copyParams{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
#ifndef __CCE_KT_TEST__
    DataCopyPad(gmOut_[nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_ * tilingData_.H + bias], outLocal, copyParams);
#endif
    outQueue_.FreeTensor(outLocal);
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CutK<T, ISBIASEXIST>::Process()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNum) {
        return;
    }
    int64_t loopCount = tilingData_.normalCoreLoopNum;
    int64_t tailLoopBlock = tilingData_.normalCoreHandleNumTailLoop;
    if ((GetBlockIdx() + 1) == tilingData_.usedCoreNum) {
        loopCount = tilingData_.tailCoreLoopNum;
        tailLoopBlock = tilingData_.tailCoreHandleNumTailLoop;
    }

    for (int64_t n = 0; n < loopCount - 1; n++) {
        CopyIn(n, curCoreHandleNumPerLoop_);
        Compute(n, curCoreHandleNumPerLoop_);
        CopyOut(n, curCoreHandleNumPerLoop_);
    }
    // tail loop
    {
        CopyIn(loopCount - 1, tailLoopBlock);
        Compute(loopCount - 1, tailLoopBlock);
        CopyOut(loopCount - 1, tailLoopBlock);
    }
}

}  // namespace MoeFinalizeRoutingV2
#endif  // MOE_FINALIZE_ROUTING_V2_BF16_CUTK
