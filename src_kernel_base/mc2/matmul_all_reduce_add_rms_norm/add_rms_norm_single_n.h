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
 * \file add_rms_norm_single_n.h
 * \brief
 */
#ifndef MC2_ADD_RMS_NORM_SINGLE_N_H
#define MC2_ADD_RMS_NORM_SINGLE_N_H
#include "rms_norm_base.h"

using namespace AscendC;

template <typename T> class KernelAddRmsNormSingleN {
public:
    __aicore__ inline KernelAddRmsNormSingleN()
    {
    }
    __aicore__ inline void Init(GM_ADDR gammaGM, AddRMSNormTilingData &tiling, TPipe *pipe, uint32_t blockDim)
    {
        ASSERT(blockDim != 0 && "Block dim can not be zero!");
        this->numCol_ = tiling.num_col;
        this->ubFactor_ = tiling.ub_factor;
        this->epsilon_ = tiling.epsilon;
        this->avgFactor_ = (numCol_ != 0) ? (float)1.0 / numCol_ : 0;
        // get start index for current core, core parallel
        gamma_.SetGlobalBuffer((__gm__ T *)gammaGM, numCol_);
        pipe->InitBuffer(unitBuf_, 195584); // (192 - 1) * 1024 byte
    }

    __aicore__ inline void Process()
    {
        if constexpr (IsSameType<T, half>::value) {
            ProcessFp16();
        } else {
            ProcessBf16();
        }
    }

    __aicore__ inline void ComputeProcess(GM_ADDR normOutGM, GM_ADDR residualGM, GM_ADDR yGM,
                                          AddRMSNormTilingData &tilingData, uint32_t addRmsNormCount, uint32_t rcvCnt)
    {
        uint64_t cOffset = CalcShapeOffset(sizeof(T), tilingData.num_row, tilingData.num_col); // 偏移*size
        for (; addRmsNormCount <= rcvCnt; ++addRmsNormCount) {
            normOut_.SetGlobalBuffer((__gm__ T *)normOutGM + GetBlockIdx() * numCol_, numCol_);
            residual_.SetGlobalBuffer((__gm__ T *)residualGM + GetBlockIdx() * numCol_, numCol_);
            y_.SetGlobalBuffer((__gm__ T *)yGM + GetBlockIdx() * numCol_, numCol_);
            Process();
            normOutGM += cOffset;
            residualGM += cOffset;
            yGM += cOffset;
        }
    }

private:
    __aicore__ inline void ProcessFp16()
    {
        LocalTensor<float> ubLocal = unitBuf_.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor_];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor_];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor_ * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor_ * 3];

        DataCopyCustom<T>(x1Local, normOut_, numCol_);
        event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V1);
        DataCopyCustom<T>(x2Local, residual_, numCol_);
        event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V1);
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);
        Add(x1Local, x1Local, x2Local, numCol_);
        PipeBarrier<PIPE_V>();

        // copy gamma
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        set_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
        wait_flag(PIPE_V, PIPE_MTE2, eventVMTE2);

        DataCopyCustom<T>(x2Local, gamma_, numCol_); // gammaLocal use x2Local
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);

        // copy x out
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        DataCopyCustom<T>(y_, x1Local, numCol_);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        set_flag(PIPE_MTE3, PIPE_V, eventMTE3V);

        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol_);
        PipeBarrier<PIPE_V>();
        Mul(sqxLocal, xFp32Local, xFp32Local, numCol_);
        PipeBarrier<PIPE_V>();
        Muls(sqxLocal, sqxLocal, avgFactor_, numCol_);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol_);
        PipeBarrier<PIPE_V>();
        Adds(sqxLocal, sqxLocal, epsilon_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(tmpLocal, (float)1.0, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, tmpLocal, sqxLocal, 1);
        PipeBarrier<PIPE_V>();

        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        set_flag(PIPE_V, PIPE_S, eventVS);
        wait_flag(PIPE_V, PIPE_S, eventVS);
        float rstdValue = sqxLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        set_flag(PIPE_S, PIPE_V, eventSV);
        wait_flag(PIPE_S, PIPE_V, eventSV);

        Muls(xFp32Local, xFp32Local, rstdValue, numCol_);
        PipeBarrier<PIPE_V>();
        wait_flag(PIPE_MTE3, PIPE_V, eventMTE3V);
        Cast(x1Local, xFp32Local, RoundMode::CAST_NONE, numCol_);
        PipeBarrier<PIPE_V>();
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);
        Mul(x1Local, x1Local, x2Local, numCol_);
        set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        DataCopyCustom<T>(normOut_, x1Local, numCol_);
    }

    __aicore__ inline void ProcessBf16()
    {
        LocalTensor<float> ubLocal = unitBuf_.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor_];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor_];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor_ * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor_ * 3];

        DataCopyCustom<T>(x1Local, normOut_, numCol_);
        event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V1);
        DataCopyCustom<T>(x2Local, residual_, numCol_);
        event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V1);
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol_);
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);
        Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol_);
        PipeBarrier<PIPE_V>();
        Add(xFp32Local, xFp32Local, sqxLocal, numCol_);
        PipeBarrier<PIPE_V>();
        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol_);
        PipeBarrier<PIPE_V>();
        // copy gamma
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        set_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
        wait_flag(PIPE_V, PIPE_MTE2, eventVMTE2);

        DataCopyCustom<T>(x2Local, gamma_, numCol_); // gammaLocal use x2Local
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);

        // copy x out
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        DataCopyCustom<T>(y_, x1Local, numCol_);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        set_flag(PIPE_MTE3, PIPE_V, eventMTE3V);

        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol_);
        PipeBarrier<PIPE_V>();
        Mul(sqxLocal, xFp32Local, xFp32Local, numCol_);
        PipeBarrier<PIPE_V>();
        Muls(sqxLocal, sqxLocal, avgFactor_, numCol_);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol_);
        PipeBarrier<PIPE_V>();
        Adds(sqxLocal, sqxLocal, epsilon_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(tmpLocal, (float)1.0, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, tmpLocal, sqxLocal, 1);
        PipeBarrier<PIPE_V>();

        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        set_flag(PIPE_V, PIPE_S, eventVS);
        wait_flag(PIPE_V, PIPE_S, eventVS);
        float rstdValue = sqxLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        set_flag(PIPE_S, PIPE_V, eventSV);
        wait_flag(PIPE_S, PIPE_V, eventSV);
        Muls(xFp32Local, xFp32Local, rstdValue, numCol_);
        PipeBarrier<PIPE_V>();
        wait_flag(PIPE_MTE3, PIPE_V, eventMTE3V);

        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol_);
        PipeBarrier<PIPE_V>();
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol_);
        PipeBarrier<PIPE_V>();
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);

        Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol_);
        PipeBarrier<PIPE_V>();
        Mul(xFp32Local, xFp32Local, sqxLocal, numCol_);
        PipeBarrier<PIPE_V>();
        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol_);
        set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        DataCopyCustom<T>(normOut_, x1Local, numCol_);
    }

private:
    TBuf<TPosition::VECCALC> unitBuf_;
    GlobalTensor<T> normOut_;
    GlobalTensor<T> residual_;
    GlobalTensor<T> gamma_;
    GlobalTensor<T> y_;

    uint32_t numCol_;
    uint32_t ubFactor_;
    float epsilon_;
    float avgFactor_;
};
#endif // MC2_ADD_RMS_NORM_SINGLE_N_H