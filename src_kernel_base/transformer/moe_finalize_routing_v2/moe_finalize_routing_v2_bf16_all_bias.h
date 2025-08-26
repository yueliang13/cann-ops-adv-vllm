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
 * \file moe_finalize_routing_v2_bf16_all_bias.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_BF16_ALL_BIAS
#define MOE_FINALIZE_ROUTING_V2_BF16_ALL_BIAS

#include "moe_finalize_routing_v2_common.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeFinalizeRoutingV2 {

using namespace AscendC;

template <typename T, const bool ISBIASEXIST>
class MoeFinalizeRoutingV2BF16AllBias {
public:
    __aicore__ inline MoeFinalizeRoutingV2BF16AllBias(
        const MoeFinalizeRoutingV2TilingData &tilingData,
        TPipe &pipe) : pipe_(pipe), tilingData_(tilingData){};
    __aicore__ inline void Init(
        GM_ADDR expandedPermutedRows,
        GM_ADDR expandedSrcToDstRow,
        GM_ADDR skip1,
        GM_ADDR skip2,
        GM_ADDR bias,
        GM_ADDR scales,
        GM_ADDR expertForSourceRow,
        GM_ADDR out,
        GM_ADDR workspace
    );
    __aicore__ inline void Process();

private:
    __aicore__ inline void CkechColAlignment();
    __aicore__ inline void CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void Compute(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline int64_t AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcess(int64_t param);
    __aicore__ inline void PrepareData();

private:
    TPipe &pipe_;
    const MoeFinalizeRoutingV2TilingData &tilingData_;
    TQue<QuePosition::VECIN, 1> skip1Queue_;
    TQue<QuePosition::VECIN, 1> skip2Queue_;
    TQue<QuePosition::VECIN, 1> scalesQueue_;
    TQue<QuePosition::VECIN, 1> expertForSourceRowQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    TBuf<QuePosition::VECCALC> expandedPermutedRowsBuf0_;
    TBuf<QuePosition::VECCALC> biasBuf0_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBuf1_;
    TBuf<QuePosition::VECCALC> biasBuf1_;

    TBuf<QuePosition::VECCALC> expandedSrcToDstRowBuff_;

    TBuf<QuePosition::VECCALC> skip1CastBuf_;
    TBuf<QuePosition::VECCALC> skip2CastBuf_;
    TBuf<QuePosition::VECCALC> biasCastBuf0_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsCastBuf0_;
    TBuf<QuePosition::VECCALC> biasCastBuf1_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsCastBuf1_;

    GlobalTensor<T> gmExpandedPermutedRows_;
    GlobalTensor<T> gmSkip1_;
    GlobalTensor<T> gmSkip2_;
    GlobalTensor<T> gmBias_;
    GlobalTensor<T> gmScales_;
    GlobalTensor<int32_t> gmExpandedSrcToDstRow_;
    GlobalTensor<int32_t> gmExpertForSourceRow_;
    GlobalTensor<T> gmOut_;

    LocalTensor<int32_t> expandedSrcToDstRow_;

    int64_t workspace_{0};

    // input number
    int64_t inputSkipIdx_{0};
    int64_t inputScalesAndExpertIdx_{0};
    int64_t outputIdx_{0};
    int64_t curCoreHandleNumPerLoop_{0};
    int64_t biasInCore_{0};

    // tiling params
    const int64_t onceAlgnNum_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T))};

    bool isPadH = false;
    bool isPadK = false;
    bool isPadKInt32 = false;
    bool initFlag_ = true;
    bool skipsAllNone_ = false;
    uint16_t rightPaddingH = 0;
    uint16_t rightPaddingK = 0;
    uint16_t rightPaddingKInt32 = 0;

    bool isPadSourceToDstRow_ = false;
    uint16_t rightPaddingSourceToDstRow_ = 0;
    int64_t curCoreHandleNum_ = 0;
};

template <typename T, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2BF16AllBias<T, ISBIASEXIST>::AlignmentProcess(int64_t param)
{
    return (param + onceAlgnNum_ - 1) / onceAlgnNum_ * onceAlgnNum_;
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2BF16AllBias<T, ISBIASEXIST>::PadProcess(int64_t param)
{
    return  onceAlgnNum_ - param % onceAlgnNum_;
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2BF16AllBias<T, ISBIASEXIST>::CkechColAlignment()
{
    if (tilingData_.H * sizeof(T) % ONE_BLK_SIZE) {
        isPadH = true;
        rightPaddingH = PadProcess(tilingData_.H);
    }

    if (tilingData_.K * sizeof(T) % ONE_BLK_SIZE) {
        isPadK = true;
        rightPaddingK = PadProcess(tilingData_.K);
    }

   if (tilingData_.K * INT32_BYTES % ONE_BLK_SIZE) {
        isPadKInt32 = true;
        rightPaddingKInt32 = PadProcessInt32(tilingData_.K);
    }
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
        if (curCoreHandleNum_ * INT32_BYTES % ONE_BLK_SIZE) {
            isPadSourceToDstRow_ = true;
            rightPaddingSourceToDstRow_ = PadProcessInt32(curCoreHandleNum_);
        }
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) { 
        if (curCoreHandleNum_ * tilingData_.K * INT32_BYTES % ONE_BLK_SIZE) {
            isPadSourceToDstRow_ = true;
            rightPaddingSourceToDstRow_ = PadProcessInt32(curCoreHandleNum_ * tilingData_.K);
        }
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2BF16AllBias<T, ISBIASEXIST>::Init(
    GM_ADDR expandedPermutedRows,
    GM_ADDR expandedSrcToDstRow,
    GM_ADDR skip1,
    GM_ADDR skip2,
    GM_ADDR bias,
    GM_ADDR scales,
    GM_ADDR expertForSourceRow,
    GM_ADDR out,
    GM_ADDR workspace
)
{
    if (GetBlockIdx() + 1 == tilingData_.usedCoreNum) {
        curCoreHandleNumPerLoop_ = tilingData_.tailCoreHandleNumPerLoop;
        curCoreHandleNum_ = tilingData_.tailCoreHandleNum;
    } else {
        curCoreHandleNumPerLoop_ = tilingData_.normalCoreHandleNumPerLoop;
        curCoreHandleNum_ = tilingData_.normalCoreHandleNum;
    }
    biasInCore_ = GetBlockIdx() * tilingData_.normalCoreHandleNum;

    CkechColAlignment();

    // gmInput分核 && 输入偏移量初始化
    inputSkipIdx_ = biasInCore_ * tilingData_.H;
    if (tilingData_.skip1IsNull == 0) {
        gmSkip1_.SetGlobalBuffer((__gm__ T *)skip1 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);
    }
    pipe_.InitBuffer(skip1Queue_, BUFFER_NUM, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.H) * sizeof(T));
    if (tilingData_.skip2IsNull == 0) {
        gmSkip2_.SetGlobalBuffer((__gm__ T *)skip2 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);
        pipe_.InitBuffer(skip2Queue_, BUFFER_NUM, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.H) * sizeof(T));
        pipe_.InitBuffer(skip2CastBuf_, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.H) * sizeof(float));
    }

    inputScalesAndExpertIdx_ = biasInCore_ * tilingData_.K;
    if (tilingData_.scalesIsNull == 0) {
        gmScales_.SetGlobalBuffer((__gm__ T *)scales + inputScalesAndExpertIdx_, tilingData_.normalCoreHandleNum * tilingData_.K);
        pipe_.InitBuffer(scalesQueue_, BUFFER_NUM, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.K) * sizeof(T));
    }

    gmExpandedSrcToDstRow_.SetGlobalBuffer((__gm__ int32_t *)expandedSrcToDstRow, tilingData_.totalRowNum * tilingData_.K);

    outputIdx_ = biasInCore_ * tilingData_.H;
    gmOut_.SetGlobalBuffer((__gm__ T *)out + outputIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);

    gmExpandedPermutedRows_.SetGlobalBuffer((__gm__ T *)expandedPermutedRows, tilingData_.totalRowNum * tilingData_.K * tilingData_.H);
    if constexpr (ISBIASEXIST) {
        gmExpertForSourceRow_.SetGlobalBuffer((__gm__ int32_t *)expertForSourceRow + inputScalesAndExpertIdx_,
                                          tilingData_.normalCoreHandleNum * tilingData_.K);

        pipe_.InitBuffer(expertForSourceRowQueue_, BUFFER_NUM,
                        tilingData_.normalCoreHandleNumPerLoop * Int32AlignmentProcess(tilingData_.K) * sizeof(int32_t));
        gmBias_.SetGlobalBuffer((__gm__ T *)bias, tilingData_.biasRowNum * tilingData_.H);
        pipe_.InitBuffer(biasBuf0_, AlignmentProcess(tilingData_.H) * sizeof(T));
        pipe_.InitBuffer(biasBuf1_, AlignmentProcess(tilingData_.H) * sizeof(T));
        pipe_.InitBuffer(biasCastBuf0_, AlignmentProcess(tilingData_.H) * sizeof(float));
        pipe_.InitBuffer(biasCastBuf1_, AlignmentProcess(tilingData_.H) * sizeof(float));
    }

    // 申请 buffer 空间
    pipe_.InitBuffer(outQueue_, BUFFER_NUM, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.H) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBuf0_, AlignmentProcess(tilingData_.H) * sizeof(T));
    pipe_.InitBuffer(expandedPermutedRowsBuf1_, AlignmentProcess(tilingData_.H) * sizeof(T));

    pipe_.InitBuffer(skip1CastBuf_, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.H) * sizeof(float));
    pipe_.InitBuffer(expandedPermutedRowsCastBuf0_, AlignmentProcess(tilingData_.H) * sizeof(float));
    pipe_.InitBuffer(expandedPermutedRowsCastBuf1_, AlignmentProcess(tilingData_.H) * sizeof(float));

    pipe_.InitBuffer(expandedSrcToDstRowBuff_, Int32AlignmentProcess(curCoreHandleNum_) * tilingData_.K * sizeof(int32_t));
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2BF16AllBias<T, ISBIASEXIST>::PrepareData()
{
    expandedSrcToDstRow_ = expandedSrcToDstRowBuff_.Get<int32_t>();
    DataCopyPadExtParams<int32_t> padParamsSrcToDstRow{isPadSourceToDstRow_, 0,
                                          static_cast<uint8_t>(rightPaddingSourceToDstRow_), 0};
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {                                          
        DataCopyExtParams copyParamsSrcToDstRow{static_cast<uint16_t>(tilingData_.K), static_cast<uint32_t>(curCoreHandleNum_ *
                                            sizeof(int32_t)), static_cast<uint32_t>((tilingData_.totalRowNum - curCoreHandleNum_) *
                                            sizeof(int32_t)), 0, 0};
#ifndef __CCE_KT_TEST__
        DataCopyPad(expandedSrcToDstRow_, gmExpandedSrcToDstRow_[GetBlockIdx() * tilingData_.normalCoreHandleNum],
            copyParamsSrcToDstRow, padParamsSrcToDstRow);
#endif

    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        DataCopyExtParams copyParamsSrcToDstRow{static_cast<uint16_t>(1),
                                            static_cast<uint32_t>(curCoreHandleNum_ * tilingData_.K * sizeof(int32_t)),
                                            0, 0, 0};
#ifndef __CCE_KT_TEST__
        DataCopyPad(expandedSrcToDstRow_, gmExpandedSrcToDstRow_[GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K], 
            copyParamsSrcToDstRow, padParamsSrcToDstRow);
#endif
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2BF16AllBias<T, ISBIASEXIST>::CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes)
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
    DataCopyParams copyParamsSkip{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(tilingData_.H * sizeof(T)), 0, 0};
    DataCopyPadParams padParamsSkip{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};
    if (tilingData_.skip1IsNull == 0) {
#ifndef __CCE_KT_TEST__
        DataCopyPad(skip1Local, gmSkip1_[nLoopIdx * curCoreHandleNumPerLoop_ * tilingData_.H], copyParamsSkip, padParamsSkip);
#endif
    }
    if (tilingData_.skip2IsNull == 0) {
#ifndef __CCE_KT_TEST__
        DataCopyPad(skip2Local, gmSkip2_[nLoopIdx * curCoreHandleNumPerLoop_ * tilingData_.H], copyParamsSkip, padParamsSkip);
#endif
    }

    // ---------------------------- [Scales] -------------------------------
    if (tilingData_.scalesIsNull == 0) {
        DataCopyParams copyParamsScales{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(tilingData_.K * sizeof(T)),
                                        0, 0};
        DataCopyPadParams padParamsScales{isPadK, 0, static_cast<uint8_t>(rightPaddingK), 0};
#ifndef __CCE_KT_TEST__
        DataCopyPad(scalesLocal, gmScales_[nLoopIdx * curCoreHandleNumPerLoop_ * tilingData_.K], copyParamsScales, padParamsScales);
#endif
    }
    // ---------------------------- [Expert] -------------------------------
    if constexpr (ISBIASEXIST) {
        DataCopyParams copyParamsExpert{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(tilingData_.K * sizeof(int32_t)),
                                        0, 0};
        DataCopyPadParams padParamsExpert{isPadKInt32, 0, static_cast<uint8_t>(rightPaddingKInt32), 0};
#ifndef __CCE_KT_TEST__
        DataCopyPad(expertForSourceRowLocal, gmExpertForSourceRow_[nLoopIdx * curCoreHandleNumPerLoop_ * tilingData_.K],
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
__aicore__ inline void MoeFinalizeRoutingV2BF16AllBias<T, ISBIASEXIST>::Compute(int64_t nLoopIdx, int64_t curRepeatTimes)
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
        Cast(skip1CastUb, skip1Local, RoundMode::CAST_NONE, curRepeatTimes * AlignmentProcess(tilingData_.H));
        pipe_barrier(PIPE_V);
    }

    LocalTensor<T> skip2Local;
    LocalTensor<float> skip2CastUb;
    if (tilingData_.skip2IsNull == 0) {
        skip2Local = skip2Queue_.DeQue<T>();
        skip2CastUb = skip2CastBuf_.Get<float>();
        Cast(skip2CastUb, skip2Local, RoundMode::CAST_NONE, curRepeatTimes * AlignmentProcess(tilingData_.H));
        pipe_barrier(PIPE_V);
    }

    if ((tilingData_.skip1IsNull == 0) && (tilingData_.skip2IsNull == 0)) {
        Add(skip1CastUb, skip1CastUb, skip2CastUb, curRepeatTimes * AlignmentProcess(tilingData_.H));
    } else if ((tilingData_.skip1IsNull == 0) && (tilingData_.skip2IsNull == 1)) {
        Adds(skip1CastUb, skip1CastUb, float(0.0), curRepeatTimes * AlignmentProcess(tilingData_.H));
    } else if ((tilingData_.skip1IsNull == 1) && (tilingData_.skip2IsNull == 0)) {
        Adds(skip1CastUb, skip2CastUb, float(0.0), curRepeatTimes * AlignmentProcess(tilingData_.H));
    } else {
        skipsAllNone_ = true;
        initFlag_ = false;
    }
    
    LocalTensor<T> expandedPermutedTmpUbDb0 = expandedPermutedRowsBuf0_.Get<T>();
    LocalTensor<T> expandedPermutedTmpUbDb1 = expandedPermutedRowsBuf1_.Get<T>();

    LocalTensor<float> expandedPermutedRowsCastUb0 = expandedPermutedRowsCastBuf0_.Get<float>();

    LocalTensor<float> expandedPermutedRowsCastUb1 = expandedPermutedRowsCastBuf1_.Get<float>();
    
    LocalTensor<int32_t> expertForSourceRowLocal;
    LocalTensor<T> biasTmpUbDb0;
    LocalTensor<T> biasTmpUbDb1;
    LocalTensor<float> biasCastUb0;
    LocalTensor<float> biasCastUb1;
    if constexpr (ISBIASEXIST) {
        biasTmpUbDb0 = biasBuf0_.Get<T>();
        biasTmpUbDb1 = biasBuf1_.Get<T>();
        biasCastUb0 = biasCastBuf0_.Get<float>();
        biasCastUb1 = biasCastBuf1_.Get<float>();
        expertForSourceRowLocal = expertForSourceRowQueue_.DeQue<int32_t>();
    }

    DataCopyParams copyParams{1, static_cast<uint16_t>(tilingData_.H * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};

    int64_t biasInLoop = nLoopIdx * curCoreHandleNumPerLoop_;

    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
    set_flag(PIPE_V, PIPE_S, EVENT_ID2);
    set_flag(PIPE_V, PIPE_S, EVENT_ID3);
    for (int64_t i = 0; i < curRepeatTimes; i++) {
        if (skipsAllNone_) {
            initFlag_ = false;
        }
        int64_t outRowIndex = i * AlignmentProcess(tilingData_.H);
        for (int64_t j = 0; j < tilingData_.K / PARALLEL_NUM; j++) {
            /*******************************乒***********************************************/
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            int64_t expandedSrcToDstRowIndexDb0 = 0; 
            if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
                expandedSrcToDstRowIndexDb0 = biasInLoop + i + PARALLEL_NUM * j * Int32AlignmentProcess(curCoreHandleNum_);
            } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
                expandedSrcToDstRowIndexDb0 = biasInLoop * tilingData_.K + i * tilingData_.K + PARALLEL_NUM * j;
            }
            int64_t expandedPermutedRowsIndexDb0 = expandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb0);
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

            wait_flag(PIPE_V, PIPE_S, EVENT_ID2);
            
            int64_t biasIndexDb0 = 0;
            if constexpr (ISBIASEXIST) {
                biasIndexDb0 = expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(tilingData_.K) + PARALLEL_NUM * j);
            }
            float scalesValDb0 = 1.0;
            if (tilingData_.scalesIsNull == 0) {
                scalesValDb0 = ToFloat(scalesLocal.GetValue(i * AlignmentProcess(tilingData_.K) + PARALLEL_NUM * j));
            }
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);

            /*******************************乒***********************************************/
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);
            if (expandedPermutedRowsIndexDb0 != INVALID_ROW_INDEX) {
#ifndef __CCE_KT_TEST__
                DataCopyPad(expandedPermutedTmpUbDb0, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb0 * tilingData_.H],
                    copyParams, padParams);
#endif

            }
            if constexpr (ISBIASEXIST) {
#ifndef __CCE_KT_TEST__
                DataCopyPad(biasTmpUbDb0, gmBias_[biasIndexDb0 * tilingData_.H], copyParams, padParams);
#endif
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);


            /*******************************乓***********************************************/
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
            int64_t expandedSrcToDstRowIndexDb1 = 0; 
            if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
                expandedSrcToDstRowIndexDb1 = biasInLoop + i + (PARALLEL_NUM * j + 1) * Int32AlignmentProcess(curCoreHandleNum_);
            } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
                expandedSrcToDstRowIndexDb1 = biasInLoop * tilingData_.K + i * tilingData_.K + PARALLEL_NUM * j + 1;
            }
            int64_t expandedPermutedRowsIndexDb1 = expandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb1);
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);

            wait_flag(PIPE_V, PIPE_S, EVENT_ID3);
            int64_t biasIndexDb1 = 0;
            if constexpr (ISBIASEXIST) {
                biasIndexDb1 = expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(tilingData_.K) + PARALLEL_NUM * j + 1);
            }
            float scalesValDb1 = 1.0;
            if (tilingData_.scalesIsNull == 0) {
                scalesValDb1 = ToFloat(scalesLocal.GetValue(i * AlignmentProcess(tilingData_.K) + PARALLEL_NUM * j + 1));
            }
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID3);

            /*******************************乓***********************************************/
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID3);
            if (expandedPermutedRowsIndexDb1 != INVALID_ROW_INDEX) {
#ifndef __CCE_KT_TEST__
                DataCopyPad(expandedPermutedTmpUbDb1, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb1 * tilingData_.H],
                    copyParams, padParams);
#endif

            }
            if constexpr (ISBIASEXIST) {
#ifndef __CCE_KT_TEST__
                DataCopyPad(biasTmpUbDb1, gmBias_[biasIndexDb1 * tilingData_.H], copyParams, padParams);
#endif
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);

            /*******************************乒***********************************************/
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            if (expandedPermutedRowsIndexDb0 != INVALID_ROW_INDEX) {
                Cast(expandedPermutedRowsCastUb0, expandedPermutedTmpUbDb0, RoundMode::CAST_NONE, tilingData_.H);
            }
            if constexpr (ISBIASEXIST) {
                Cast(biasCastUb0, biasTmpUbDb0, RoundMode::CAST_NONE, tilingData_.H);
            }
            pipe_barrier(PIPE_V);
            if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX && (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                Duplicate(expandedPermutedRowsCastUb0, (float)0.0, tilingData_.H);
                pipe_barrier(PIPE_V);
            }
            if constexpr (ISBIASEXIST) {
                Add(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, biasCastUb0, tilingData_.H);
            } else {
                Adds(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, (float)0.0, tilingData_.H);
            }
            pipe_barrier(PIPE_V);
            if (!initFlag_) {
                Muls(skip1CastUb[outRowIndex], expandedPermutedRowsCastUb0, scalesValDb0, tilingData_.H);
                initFlag_ = true;
                set_flag(PIPE_V, PIPE_S, EVENT_ID2);
            } else {
                Muls(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, scalesValDb0, tilingData_.H);
                set_flag(PIPE_V, PIPE_S, EVENT_ID2);
                pipe_barrier(PIPE_V);
                Add(skip1CastUb[outRowIndex], skip1CastUb[outRowIndex], expandedPermutedRowsCastUb0, tilingData_.H);
            }

            /*******************************乓***********************************************/
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            if (expandedPermutedRowsIndexDb1 != INVALID_ROW_INDEX) {
                Cast(expandedPermutedRowsCastUb1, expandedPermutedTmpUbDb1, RoundMode::CAST_NONE, tilingData_.H);
            }
            if constexpr (ISBIASEXIST) {
                Cast(biasCastUb1, biasTmpUbDb1, RoundMode::CAST_NONE, tilingData_.H);
            }
            pipe_barrier(PIPE_V);
            if (expandedPermutedRowsIndexDb1 == INVALID_ROW_INDEX && (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                Duplicate(expandedPermutedRowsCastUb1, (float)0.0, tilingData_.H);
                pipe_barrier(PIPE_V);
            }
            if constexpr (ISBIASEXIST) {
                Add(expandedPermutedRowsCastUb1, expandedPermutedRowsCastUb1, biasCastUb1, tilingData_.H);
            } else {
                Adds(expandedPermutedRowsCastUb1, expandedPermutedRowsCastUb1, (float)0.0, tilingData_.H);
            }
            pipe_barrier(PIPE_V);
            Muls(expandedPermutedRowsCastUb1, expandedPermutedRowsCastUb1, scalesValDb1, tilingData_.H);
            set_flag(PIPE_V, PIPE_S, EVENT_ID3);
            pipe_barrier(PIPE_V);
            Add(skip1CastUb[outRowIndex], skip1CastUb[outRowIndex], expandedPermutedRowsCastUb1, tilingData_.H);
        }
        if (tilingData_.K % PARALLEL_NUM != 0) {
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            int64_t expandedSrcToDstRowIndexDb0 = 0;
            if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
                expandedSrcToDstRowIndexDb0 = biasInLoop + i + (tilingData_.K - 1) * Int32AlignmentProcess(curCoreHandleNum_);
            } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
                expandedSrcToDstRowIndexDb0 = biasInLoop * tilingData_.K + i * tilingData_.K + tilingData_.K - 1;
            }
            int64_t expandedPermutedRowsIndexDb0 = expandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb0);
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

            wait_flag(PIPE_V, PIPE_S, EVENT_ID2);
            int64_t biasIndexDb0 = 0;
            if constexpr (ISBIASEXIST) {
                biasIndexDb0 = expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(tilingData_.K) + tilingData_.K - 1);
            }
            float scalesValDb0 = 1.0;
            if (tilingData_.scalesIsNull == 0) {
                scalesValDb0 = ToFloat(scalesLocal.GetValue(i * AlignmentProcess(tilingData_.K) + tilingData_.K - 1));
            }
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);

            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);
            if (expandedPermutedRowsIndexDb0 != INVALID_ROW_INDEX) {
#ifndef __CCE_KT_TEST__
                DataCopyPad(expandedPermutedTmpUbDb0, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb0 * tilingData_.H],
                    copyParams, padParams);
#endif

            }
            if constexpr (ISBIASEXIST) {
#ifndef __CCE_KT_TEST__
                DataCopyPad(biasTmpUbDb0, gmBias_[biasIndexDb0 * tilingData_.H], copyParams, padParams);
#endif
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            if (expandedPermutedRowsIndexDb0 != INVALID_ROW_INDEX) {
                Cast(expandedPermutedRowsCastUb0, expandedPermutedTmpUbDb0, RoundMode::CAST_NONE, tilingData_.H);
            }
            if constexpr (ISBIASEXIST) {
                Cast(biasCastUb0, biasTmpUbDb0, RoundMode::CAST_NONE, tilingData_.H);
            }
            pipe_barrier(PIPE_V);
            if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX && (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                Duplicate(expandedPermutedRowsCastUb0, (float)0.0, tilingData_.H);
                pipe_barrier(PIPE_V);
            }
            if constexpr (ISBIASEXIST) {
                Add(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, biasCastUb0, tilingData_.H);
            } else {
                Adds(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, (float)0.0, tilingData_.H);
            }
            pipe_barrier(PIPE_V);
            if (!initFlag_) {
                Muls(skip1CastUb[outRowIndex], expandedPermutedRowsCastUb0, scalesValDb0, tilingData_.H);
                initFlag_ = true;
                set_flag(PIPE_V, PIPE_S, EVENT_ID2);
            } else {
                Muls(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, scalesValDb0, tilingData_.H);
                set_flag(PIPE_V, PIPE_S, EVENT_ID2);
                pipe_barrier(PIPE_V);
                Add(skip1CastUb[outRowIndex], skip1CastUb[outRowIndex], expandedPermutedRowsCastUb0, tilingData_.H);
            }
        }
    }
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID3);
    pipe_barrier(PIPE_V);
    Cast(outLocal, skip1CastUb, RoundMode::CAST_ROUND, curRepeatTimes * AlignmentProcess(tilingData_.H));
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
__aicore__ inline void MoeFinalizeRoutingV2BF16AllBias<T, ISBIASEXIST>::CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    LocalTensor<T> outLocal = outQueue_.DeQue<T>();
    DataCopyParams copyParams{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(tilingData_.H * sizeof(T)), 0, 0};
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#ifndef __CCE_KT_TEST__
    DataCopyPad(gmOut_[nLoopIdx * tilingData_.H * curCoreHandleNumPerLoop_], outLocal, copyParams);
#endif
    outQueue_.FreeTensor(outLocal);
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2BF16AllBias<T, ISBIASEXIST>::Process()
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
    PrepareData();
    for (int64_t n = 0; n < loopCount - 1; n++) {
        CopyIn(n, curCoreHandleNumPerLoop_);
        Compute(n, curCoreHandleNumPerLoop_);
        CopyOut(n, curCoreHandleNumPerLoop_);
    }

    CopyIn(loopCount - 1, tailLoopBlock);
    Compute(loopCount - 1, tailLoopBlock);
    CopyOut(loopCount - 1, tailLoopBlock);
}

}  // namespace MoeFinalizeRoutingV2
#endif  // MOE_FINALIZE_ROUTING_V2_BF16_ALL_BIAS
