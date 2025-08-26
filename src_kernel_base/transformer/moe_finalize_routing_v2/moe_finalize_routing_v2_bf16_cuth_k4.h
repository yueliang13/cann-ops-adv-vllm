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
 * \file moe_finalize_routing_v2_bf16_cuth_k4.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_BF16_CUTH_K_FOUR
#define MOE_FINALIZE_ROUTING_V2_BF16_CUTH_K_FOUR

#include "moe_finalize_routing_v2_common.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeFinalizeRoutingV2 {

using namespace AscendC;

template <typename T, const bool ISBIASEXIST>
class MoeFinalizeRoutingV2Bf16CuthK4 {
public:
    __aicore__ inline MoeFinalizeRoutingV2Bf16CuthK4(
        const MoeFinalizeRoutingV2TilingData &tilingData,
        TPipe &pipe) : pipe_(pipe), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, 
        GM_ADDR skip2, GM_ADDR bias, GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CkechColAlignment();
    __aicore__ inline void CopyIn(int64_t nLoopIdx, int64_t bias, int64_t dataLen, bool isPadH, int64_t rightPaddingH);
    __aicore__ inline void Compute(int64_t nLoopIdx, int64_t bias, int64_t dataLen,
        bool isPadH, int64_t rightPaddingH);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t bias, int64_t dataLen);
    __aicore__ inline int64_t AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcessT(int64_t param);

private:
    TPipe &pipe_;
    const MoeFinalizeRoutingV2TilingData &tilingData_;
    TQue<QuePosition::VECIN, 1> skip1Queue_;
    TQue<QuePosition::VECIN, 1> skip2Queue_;
    TQue<QuePosition::VECIN, 1> scalesQueue_;
    TQue<QuePosition::VECIN, 1> expertForSourceRowQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb0_;
    TBuf<QuePosition::VECCALC> biasBufDb0_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb1_;
    TBuf<QuePosition::VECCALC> biasBufDb1_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb2_;
    TBuf<QuePosition::VECCALC> biasBufDb2_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb3_;
    TBuf<QuePosition::VECCALC> biasBufDb3_;

    TBuf<QuePosition::VECCALC> skip1CastBuf_;
    TBuf<QuePosition::VECCALC> skip2CastBuf_;
    TBuf<QuePosition::VECCALC> biasCastBuf0_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsCastBuf0_;
    TBuf<QuePosition::VECCALC> biasCastBuf1_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsCastBuf1_;
    TBuf<QuePosition::VECCALC> biasCastBuf2_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsCastBuf2_;
    TBuf<QuePosition::VECCALC> biasCastBuf3_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsCastBuf3_;

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

    const int64_t onceAlgnNum_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T))};

    bool isPadNormalH_ = false;
    bool isPadUnnormalH_ = false;
    bool isPadK_ = false;
    bool isPadKInt32_ = false;
    bool initFlag_ = true;
    uint16_t rightPaddingNormalH_ = 0;
    uint16_t rightPaddingUnnormalH_ = 0;
    uint16_t rightPaddingK_ = 0;
    uint16_t rightPaddingKInt32_ = 0;
};

template <typename T, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2Bf16CuthK4<T, ISBIASEXIST>::AlignmentProcess(int64_t param)
{
    return (param + onceAlgnNum_ - 1) / onceAlgnNum_ * onceAlgnNum_;
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2Bf16CuthK4<T, ISBIASEXIST>::PadProcessT(int64_t param)
{
    return  onceAlgnNum_ - param % onceAlgnNum_;
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CuthK4<T, ISBIASEXIST>::CkechColAlignment()
{
    if (tilingData_.normalH * sizeof(T) % ONE_BLK_SIZE) {
        isPadNormalH_ = true;
        rightPaddingNormalH_ = PadProcessT(tilingData_.normalH);
    }

    if (tilingData_.unnormalH * sizeof(T) % ONE_BLK_SIZE) {
        isPadUnnormalH_ = true;
        rightPaddingUnnormalH_ = PadProcessT(tilingData_.unnormalH);
    }

    if (tilingData_.K * sizeof(T) % ONE_BLK_SIZE) {
        isPadK_ = true;
        rightPaddingK_ = PadProcessT(tilingData_.K);
    }

   if (tilingData_.K * INT32_BYTES % ONE_BLK_SIZE) {
        isPadKInt32_ = true;
        rightPaddingKInt32_ = PadProcessInt32(tilingData_.K);
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CuthK4<T, ISBIASEXIST>::Init(GM_ADDR expandedPermutedRows,
    GM_ADDR expandedSrcToDstRow, GM_ADDR skip1,GM_ADDR skip2,
    GM_ADDR bias, GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out,
    GM_ADDR workspace)
{
    if (GetBlockIdx() + 1 == tilingData_.usedCoreNum) {
        curCoreHandleNumPerLoop_ = tilingData_.tailCoreHandleNumPerLoop;
    } else {
        curCoreHandleNumPerLoop_ = tilingData_.normalCoreHandleNumPerLoop;
    }

    CkechColAlignment();

    // gmInput分核 && 输入偏移量初始化
    inputSkipIdx_ = GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.H;
    if (tilingData_.skip1IsNull == 0) {
        gmSkip1_.SetGlobalBuffer((__gm__ T *)skip1 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);
    }
    pipe_.InitBuffer(skip1Queue_, 1, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));
    if (tilingData_.skip2IsNull == 0) {
        gmSkip2_.SetGlobalBuffer((__gm__ T *)skip2 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);
        pipe_.InitBuffer(skip2Queue_, 1, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));
        pipe_.InitBuffer(skip2CastBuf_, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(float));
    }

    inputScalesAndExpertIdx_ = GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
    if (tilingData_.scalesIsNull == 0) {
        gmScales_.SetGlobalBuffer((__gm__ T *)scales + inputScalesAndExpertIdx_, tilingData_.normalCoreHandleNum * tilingData_.K);
        pipe_.InitBuffer(scalesQueue_, BUFFER_NUM, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.K) * sizeof(T));
    }

    gmExpandedSrcToDstRow_.SetGlobalBuffer((__gm__ int32_t *)expandedSrcToDstRow, tilingData_.totalRowNum * tilingData_.K);

    outputIdx_ = GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.H;
    gmOut_.SetGlobalBuffer((__gm__ T *)out + outputIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);

    gmExpandedPermutedRows_.SetGlobalBuffer((__gm__ T *)expandedPermutedRows, tilingData_.totalRowNum * tilingData_.K * tilingData_.H);

    if constexpr (ISBIASEXIST) {
        gmExpertForSourceRow_.SetGlobalBuffer((__gm__ int32_t *)expertForSourceRow + inputScalesAndExpertIdx_,
                                        tilingData_.normalCoreHandleNum * tilingData_.K);
        pipe_.InitBuffer(expertForSourceRowQueue_, BUFFER_NUM,
                        tilingData_.normalCoreHandleNumPerLoop * Int32AlignmentProcess(tilingData_.K) * sizeof(int32_t));
        gmBias_.SetGlobalBuffer((__gm__ T *)bias, tilingData_.biasRowNum * tilingData_.H);
        pipe_.InitBuffer(biasBufDb0_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
        pipe_.InitBuffer(biasBufDb1_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
        pipe_.InitBuffer(biasBufDb2_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
        pipe_.InitBuffer(biasBufDb3_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
        pipe_.InitBuffer(biasCastBuf0_, AlignmentProcess(tilingData_.normalH) * sizeof(float));
        pipe_.InitBuffer(biasCastBuf1_, AlignmentProcess(tilingData_.normalH) * sizeof(float));
        pipe_.InitBuffer(biasCastBuf2_, AlignmentProcess(tilingData_.normalH) * sizeof(float));
        pipe_.InitBuffer(biasCastBuf3_, AlignmentProcess(tilingData_.normalH) * sizeof(float));
    }

    // 申请 buffer 空间
    pipe_.InitBuffer(outQueue_, BUFFER_NUM, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb0_, AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb1_, AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb2_, AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb3_, AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(skip1CastBuf_, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(float));
    pipe_.InitBuffer(expandedPermutedRowsCastBuf0_, AlignmentProcess(tilingData_.normalH) * sizeof(float));

    pipe_.InitBuffer(expandedPermutedRowsCastBuf1_, AlignmentProcess(tilingData_.normalH) * sizeof(float));

    pipe_.InitBuffer(expandedPermutedRowsCastBuf2_, AlignmentProcess(tilingData_.normalH) * sizeof(float));

    pipe_.InitBuffer(expandedPermutedRowsCastBuf3_, AlignmentProcess(tilingData_.normalH) * sizeof(float));
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CuthK4<T, ISBIASEXIST>::CopyIn(int64_t nLoopIdx, int64_t bias, int64_t dataLen,
    bool isPadH, int64_t rightPaddingH)
{
    LocalTensor<T> scalesLocal;
    if (tilingData_.scalesIsNull == 0) {
        scalesLocal = scalesQueue_.AllocTensor<T>();
    }
    LocalTensor<int32_t> expertForSourceRowLocal;
    if constexpr (ISBIASEXIST) {
        expertForSourceRowLocal = expertForSourceRowQueue_.AllocTensor<int32_t>();
    }
    LocalTensor<T> skip1Local;
    if (tilingData_.skip1IsNull == 0) {
        skip1Local = skip1Queue_.AllocTensor<T>();
    }

    LocalTensor<T> skip2Local;
    if (tilingData_.skip2IsNull == 0) {
        skip2Local = skip2Queue_.AllocTensor<T>();
    }

    // ---------------------------- [Skip] -------------------------------
    DataCopyParams copyParamsSkip{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadParams padParamsSkip{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};
    if (tilingData_.skip1IsNull == 0) {
#ifndef __CCE_KT_TEST__
        DataCopyPad(skip1Local, gmSkip1_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.H + bias], copyParamsSkip, padParamsSkip);
#endif
    }
    if (tilingData_.skip2IsNull == 0) {
#ifndef __CCE_KT_TEST__
        DataCopyPad(skip2Local, gmSkip2_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.H + bias], copyParamsSkip, padParamsSkip);
#endif
    }

    // ---------------------------- [Scales] -------------------------------
    if (tilingData_.scalesIsNull == 0) {
        DataCopyParams copyParamsScales{1, static_cast<uint16_t>(tilingData_.K * sizeof(T)), 0, 0};
        DataCopyPadParams padParamsScales{isPadK_, 0, static_cast<uint8_t>(rightPaddingK_), 0};
#ifndef __CCE_KT_TEST__
        DataCopyPad(scalesLocal, gmScales_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K], copyParamsScales, padParamsScales);
#endif
    }
    // ---------------------------- [Expert] -------------------------------
    if constexpr (ISBIASEXIST) {
        DataCopyParams copyParamsExpert{1, static_cast<uint16_t>(tilingData_.K * sizeof(int32_t)), 0, 0};
        DataCopyPadParams padParamsExpert{isPadKInt32_, 0, static_cast<uint8_t>(rightPaddingKInt32_), 0};
#ifndef __CCE_KT_TEST__
        DataCopyPad(expertForSourceRowLocal, gmExpertForSourceRow_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K],
            copyParamsExpert, padParamsExpert);
#endif

    }
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    if (tilingData_.skip2IsNull == 0) {
        skip2Queue_.EnQue(skip2Local);
    }
    if (tilingData_.skip1IsNull == 0) {
        skip1Queue_.EnQue(skip1Local);
    }
    if (tilingData_.scalesIsNull == 0) {
        scalesQueue_.EnQue(scalesLocal);
    }
    if constexpr (ISBIASEXIST) {
        expertForSourceRowQueue_.EnQue(expertForSourceRowLocal);
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CuthK4<T, ISBIASEXIST>::Compute(int64_t nLoopIdx, int64_t bias, int64_t dataLen,
    bool isPadH, int64_t rightPaddingH)
{
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

    LocalTensor<T> scalesLocal;
    if (tilingData_.scalesIsNull == 0) {
        scalesLocal = scalesQueue_.DeQue<T>();
    }
    LocalTensor<float> skip1CastUb = skip1CastBuf_.Get<float>();
    LocalTensor<T> skip1Local;
    if (tilingData_.skip1IsNull == 0) {
        skip1Local = skip1Queue_.DeQue<T>();
        Cast(skip1CastUb, skip1Local, RoundMode::CAST_NONE, AlignmentProcess(dataLen));
        pipe_barrier(PIPE_V);
    }

    LocalTensor<T> skip2Local;
    LocalTensor<float> skip2CastUb;
    if (tilingData_.skip2IsNull == 0) {
        skip2Local = skip2Queue_.DeQue<T>();
        skip2CastUb = skip2CastBuf_.Get<float>();
        Cast(skip2CastUb, skip2Local, RoundMode::CAST_NONE, AlignmentProcess(dataLen));
        pipe_barrier(PIPE_V);
    }

    if ((tilingData_.skip1IsNull == 0) && (tilingData_.skip2IsNull == 0)) {
        Add(skip1CastUb, skip1CastUb, skip2CastUb, AlignmentProcess(dataLen));
    } else if ((tilingData_.skip1IsNull == 0) && (tilingData_.skip2IsNull == 1)) {
        Adds(skip1CastUb, skip1CastUb, float(0.0), AlignmentProcess(dataLen));
    } else if ((tilingData_.skip1IsNull == 1) && (tilingData_.skip2IsNull == 0)) {
        Adds(skip1CastUb, skip2CastUb, float(0.0), AlignmentProcess(dataLen));
    } else {
        initFlag_ = false;
    }

    LocalTensor<T> expandedPermutedTmpUbDb0 = expandedPermutedRowsBufDb0_.Get<T>();
    LocalTensor<T> expandedPermutedTmpUbDb1 = expandedPermutedRowsBufDb1_.Get<T>();
    LocalTensor<T> expandedPermutedTmpUbDb2 = expandedPermutedRowsBufDb2_.Get<T>();
    LocalTensor<T> expandedPermutedTmpUbDb3 = expandedPermutedRowsBufDb3_.Get<T>();
    LocalTensor<float> expandedPermutedRowsCastUb0 = expandedPermutedRowsCastBuf0_.Get<float>();
    LocalTensor<float> expandedPermutedRowsCastUb1 = expandedPermutedRowsCastBuf1_.Get<float>();
    LocalTensor<float> expandedPermutedRowsCastUb2 = expandedPermutedRowsCastBuf2_.Get<float>();
    LocalTensor<float> expandedPermutedRowsCastUb3 = expandedPermutedRowsCastBuf3_.Get<float>();
    LocalTensor<T> biasTmpUbDb0;
    LocalTensor<T> biasTmpUbDb1;
    LocalTensor<T> biasTmpUbDb2;
    LocalTensor<T> biasTmpUbDb3;

    LocalTensor<int32_t> expertForSourceRowLocal;
    LocalTensor<float> biasCastUb0;
    LocalTensor<float> biasCastUb1;
    LocalTensor<float> biasCastUb2;
    LocalTensor<float> biasCastUb3;
    if constexpr (ISBIASEXIST) {
        biasTmpUbDb0 = biasBufDb0_.Get<T>();
        biasTmpUbDb1 = biasBufDb1_.Get<T>();
        biasTmpUbDb2 = biasBufDb2_.Get<T>();
        biasTmpUbDb3 = biasBufDb3_.Get<T>();
        biasCastUb0 = biasCastBuf0_.Get<float>();
        biasCastUb1 = biasCastBuf1_.Get<float>();
        biasCastUb2 = biasCastBuf2_.Get<float>();
        biasCastUb3 = biasCastBuf3_.Get<float>();
        expertForSourceRowLocal = expertForSourceRowQueue_.DeQue<int32_t>();
    }
    /*******************************乒***********************************************/
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    int64_t biasIndexDb0 = 0;
    int64_t biasIndexDb1 = 0;
    if constexpr (ISBIASEXIST) {
        biasIndexDb0 = expertForSourceRowLocal.GetValue(0);
        biasIndexDb1 = expertForSourceRowLocal.GetValue(1);
    }
    float scalesValDb0 = 1.0;
    float scalesValDb1 = 1.0;
    if (tilingData_.scalesIsNull == 0) {
        scalesValDb0 = ToFloat(scalesLocal.GetValue(0));
        scalesValDb1 = ToFloat(scalesLocal.GetValue(1));
    }
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

    int64_t expandedSrcToDstRowIndexDb0 = 0;
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
            expandedSrcToDstRowIndexDb0 = nLoopIdx / (tilingData_.hSliceNum + 1) + 0 * tilingData_.totalRowNum + GetBlockIdx() * tilingData_.normalCoreHandleNum;
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        expandedSrcToDstRowIndexDb0 = nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K + 0 +
                                    GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
    }
    int64_t expandedPermutedRowsIndexDb0 = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb0);
    int64_t expandedSrcToDstRowIndexDb1 = 0;
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
        expandedSrcToDstRowIndexDb1 = nLoopIdx / (tilingData_.hSliceNum + 1) + 1 * tilingData_.totalRowNum + GetBlockIdx() * tilingData_.normalCoreHandleNum;
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        expandedSrcToDstRowIndexDb1 = nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K + 1 +
                                    GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
    }
    int64_t expandedPermutedRowsIndexDb1 = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb1);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);

    /*******************************乓***********************************************/

    int64_t biasIndexDb2 = 0;
    int64_t biasIndexDb3 = 0;
    if constexpr (ISBIASEXIST) {
        biasIndexDb2 = expertForSourceRowLocal.GetValue(2);
        biasIndexDb3 = expertForSourceRowLocal.GetValue(3);
    }
    float scalesValDb2 = 1.0;
    float scalesValDb3 = 1.0;
    if (tilingData_.scalesIsNull == 0) {
        scalesValDb2 = ToFloat(scalesLocal.GetValue(2));
        scalesValDb3 = ToFloat(scalesLocal.GetValue(3));
    }
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);

    int64_t expandedSrcToDstRowIndexDb2 = 0; 
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
        expandedSrcToDstRowIndexDb2 = nLoopIdx / (tilingData_.hSliceNum + 1) + 2 * tilingData_.totalRowNum + GetBlockIdx() * tilingData_.normalCoreHandleNum;
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        expandedSrcToDstRowIndexDb2 = nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K + 2 +
                                    GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
    }
    int64_t expandedPermutedRowsIndexDb2 = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb2);
    int64_t expandedSrcToDstRowIndexDb3 = 0;
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
        expandedSrcToDstRowIndexDb3 = nLoopIdx / (tilingData_.hSliceNum + 1) + 3 * tilingData_.totalRowNum + GetBlockIdx() * tilingData_.normalCoreHandleNum;
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        expandedSrcToDstRowIndexDb3 = nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K + 3 +
                                    GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
    }
    int64_t expandedPermutedRowsIndexDb3 = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb3);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID3);

    DataCopyParams copyParams{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};

    /*******************************乒***********************************************/
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    if constexpr (ISBIASEXIST) {
#ifndef __CCE_KT_TEST__
        DataCopyPad(biasTmpUbDb0, gmBias_[biasIndexDb0 * tilingData_.H + bias], copyParams, padParams);
#endif
#ifndef __CCE_KT_TEST__
        DataCopyPad(biasTmpUbDb1, gmBias_[biasIndexDb1 * tilingData_.H + bias], copyParams, padParams);
#endif
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);
    if (expandedPermutedRowsIndexDb0 != INVALID_ROW_INDEX) {
#ifndef __CCE_KT_TEST__
        DataCopyPad(expandedPermutedTmpUbDb0, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb0 * tilingData_.H + bias],
            copyParams, padParams);
#endif

    }
    if (expandedPermutedRowsIndexDb1 != INVALID_ROW_INDEX) {
#ifndef __CCE_KT_TEST__
        DataCopyPad(expandedPermutedTmpUbDb1, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb1 * tilingData_.H + bias],
            copyParams, padParams);
#endif

    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

    /*******************************乓***********************************************/
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
    if constexpr (ISBIASEXIST) {
#ifndef __CCE_KT_TEST__
        DataCopyPad(biasTmpUbDb2, gmBias_[biasIndexDb2 * tilingData_.H + bias], copyParams, padParams);
#endif
#ifndef __CCE_KT_TEST__
        DataCopyPad(biasTmpUbDb3, gmBias_[biasIndexDb3 * tilingData_.H + bias], copyParams, padParams);
#endif
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID3);
    if (expandedPermutedRowsIndexDb2 != INVALID_ROW_INDEX) {
#ifndef __CCE_KT_TEST__
        DataCopyPad(expandedPermutedTmpUbDb2, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb2 * tilingData_.H + bias],
            copyParams, padParams);
#endif
    }
    if (expandedPermutedRowsIndexDb3 != INVALID_ROW_INDEX) {
#ifndef __CCE_KT_TEST__
        DataCopyPad(expandedPermutedTmpUbDb3, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb3 * tilingData_.H + bias],
            copyParams, padParams);
#endif
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

    /*******************************乒***********************************************/
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    if (expandedPermutedRowsIndexDb0 != INVALID_ROW_INDEX) {
        Cast(expandedPermutedRowsCastUb0, expandedPermutedTmpUbDb0, RoundMode::CAST_NONE, dataLen);
    }
    if (expandedPermutedRowsIndexDb1 != INVALID_ROW_INDEX) {
        Cast(expandedPermutedRowsCastUb1, expandedPermutedTmpUbDb1, RoundMode::CAST_NONE, dataLen);
    }
    if constexpr (ISBIASEXIST) {
        Cast(biasCastUb0, biasTmpUbDb0, RoundMode::CAST_NONE, dataLen);
        Cast(biasCastUb1, biasTmpUbDb1, RoundMode::CAST_NONE, dataLen);
    }
    pipe_barrier(PIPE_V);
    if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX && 
                    (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
        Duplicate(expandedPermutedRowsCastUb0, (float)0.0, dataLen);
        pipe_barrier(PIPE_V);
    }
    if (expandedPermutedRowsIndexDb1 == INVALID_ROW_INDEX && 
                    (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
        Duplicate(expandedPermutedRowsCastUb1, (float)0.0, dataLen);
        pipe_barrier(PIPE_V);
    }
    if constexpr (ISBIASEXIST) {
        Add(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, biasCastUb0, dataLen);
        Add(expandedPermutedRowsCastUb1, expandedPermutedRowsCastUb1, biasCastUb1, dataLen);
    } else {
        Adds(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, (float)0.0, dataLen);
        Adds(expandedPermutedRowsCastUb1, expandedPermutedRowsCastUb1, (float)0.0, dataLen);
    }
    pipe_barrier(PIPE_V);
    if (!initFlag_) {
        Muls(skip1CastUb[0], expandedPermutedRowsCastUb0, scalesValDb0, dataLen);
        Muls(expandedPermutedRowsCastUb1, expandedPermutedRowsCastUb1, scalesValDb1, dataLen);
        initFlag_ = true;
    } else {
        Muls(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, scalesValDb0, dataLen);
        Muls(expandedPermutedRowsCastUb1, expandedPermutedRowsCastUb1, scalesValDb1, dataLen);
        pipe_barrier(PIPE_V);
        Add(skip1CastUb[0], skip1CastUb[0], expandedPermutedRowsCastUb0, dataLen);
    }
    pipe_barrier(PIPE_V);
    Add(skip1CastUb[0], skip1CastUb[0], expandedPermutedRowsCastUb1, dataLen);

    /*******************************乓***********************************************/
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    if (expandedPermutedRowsIndexDb2 != INVALID_ROW_INDEX) {
        Cast(expandedPermutedRowsCastUb2, expandedPermutedTmpUbDb2, RoundMode::CAST_NONE, dataLen);
    }
    if (expandedPermutedRowsIndexDb3 != INVALID_ROW_INDEX) {
        Cast(expandedPermutedRowsCastUb3, expandedPermutedTmpUbDb3, RoundMode::CAST_NONE, dataLen);
    }
    if constexpr (ISBIASEXIST) {
        Cast(biasCastUb2, biasTmpUbDb2, RoundMode::CAST_NONE, dataLen);
        Cast(biasCastUb3, biasTmpUbDb3, RoundMode::CAST_NONE, dataLen);
    }
    pipe_barrier(PIPE_V);
    if (expandedPermutedRowsIndexDb2 == INVALID_ROW_INDEX && 
                    (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
        Duplicate(expandedPermutedRowsCastUb2, (float)0.0, dataLen);
        pipe_barrier(PIPE_V);
    }
    if (expandedPermutedRowsIndexDb3 == INVALID_ROW_INDEX && 
                    (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
        Duplicate(expandedPermutedRowsCastUb3, (float)0.0, dataLen);
        pipe_barrier(PIPE_V);
    }
    if constexpr (ISBIASEXIST) {
        Add(expandedPermutedRowsCastUb2, expandedPermutedRowsCastUb2, biasCastUb2, dataLen);
        Add(expandedPermutedRowsCastUb3, expandedPermutedRowsCastUb3, biasCastUb3, dataLen);
    } else {
        Adds(expandedPermutedRowsCastUb2, expandedPermutedRowsCastUb2, (float)0.0, dataLen);
        Adds(expandedPermutedRowsCastUb3, expandedPermutedRowsCastUb3, (float)0.0, dataLen);
    }
    pipe_barrier(PIPE_V);
    Muls(expandedPermutedRowsCastUb2, expandedPermutedRowsCastUb2, scalesValDb2, dataLen);
    Muls(expandedPermutedRowsCastUb3, expandedPermutedRowsCastUb3, scalesValDb3, dataLen);
    pipe_barrier(PIPE_V);
    Add(skip1CastUb[0], skip1CastUb[0], expandedPermutedRowsCastUb2, dataLen);
    pipe_barrier(PIPE_V);
    Add(skip1CastUb[0], skip1CastUb[0], expandedPermutedRowsCastUb3, dataLen);
    pipe_barrier(PIPE_V);
    Cast(outLocal, skip1CastUb, RoundMode::CAST_ROUND, dataLen);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    outQueue_.EnQue(outLocal);

    if constexpr (ISBIASEXIST) {
        expertForSourceRowQueue_.FreeTensor(expertForSourceRowLocal);
    }
    if (tilingData_.scalesIsNull == 0) {
        scalesQueue_.FreeTensor(scalesLocal);
    }
    if (tilingData_.skip1IsNull == 0) {
        skip1Queue_.FreeTensor(skip1Local);
    }
    if (tilingData_.skip2IsNull == 0) {
        skip2Queue_.FreeTensor(skip2Local);
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CuthK4<T, ISBIASEXIST>::CopyOut(int64_t nLoopIdx, int64_t bias, int64_t dataLen)
{
    LocalTensor<T> outLocal = outQueue_.DeQue<T>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#ifndef __CCE_KT_TEST__
    DataCopyPad(gmOut_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.H + bias], outLocal, copyParams);
#endif
    outQueue_.FreeTensor(outLocal);
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CuthK4<T, ISBIASEXIST>::Process()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNum) {
        return;
    }
    int64_t loopCount = tilingData_.normalCoreLoopNum;
    if ((GetBlockIdx() + 1) == tilingData_.usedCoreNum) {
        loopCount = tilingData_.tailCoreLoopNum;
    }

    for (int64_t n = 0; n < loopCount; n++) {
        bool isNormalH = (n + 1) % (tilingData_.hSliceNum + 1) != 0;
        int64_t bias = isNormalH ? (n % tilingData_.hSliceNum) * tilingData_.normalH : tilingData_.hSliceNum * tilingData_.normalH;
        int64_t dataLen = isNormalH ? tilingData_.normalH : tilingData_.unnormalH;
        int64_t rightPaddingH = isNormalH ? rightPaddingNormalH_ : rightPaddingUnnormalH_;
        int64_t isPadH = isNormalH ? isPadNormalH_ : isPadUnnormalH_;
        CopyIn(n, bias, dataLen, isPadH, rightPaddingH);
        Compute(n, bias, dataLen, isPadH, rightPaddingH);
        CopyOut(n, bias, dataLen);
    }
}

}  // namespace MoeFinalizeRoutingV2
#endif  // MOE_FINALIZE_ROUTING_V2_BF16_CUTH_K_FOUR
