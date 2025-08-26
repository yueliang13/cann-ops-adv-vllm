/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file moe_finalize_routing_v2_grad_with_scale_cut_h.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_GRAD_WITH_SCALE_CUT_H_H
#define MOE_FINALIZE_ROUTING_V2_GRAD_WITH_SCALE_CUT_H_H

#include "moe_finalize_routing_v2_grad_base.h"

namespace MoeFinalizeRoutingV2Grad {
using namespace AscendC;

template <typename T1, typename T2> class MoeFinalizeRoutingV2GradWithScaleCutH : MoeFinalizeRoutingV2GradBase<T1, T2> {
public:
    __aicore__ inline MoeFinalizeRoutingV2GradWithScaleCutH(){};
    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR scales,
        GM_ADDR expertIdx, GM_ADDR bias, GM_ADDR gradExpandedX, GM_ADDR gradScales, GM_ADDR workspace,
        const MoeFinalizeRoutingV2GradTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SubProcessWithBias();
    __aicore__ inline void SubProcessWithOutBias();
    __aicore__ inline void SubProcess();
};

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradWithScaleCutH<T1, T2>::Init(GM_ADDR gradY, GM_ADDR expandedRowIdx,
    GM_ADDR expandedX, GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR bias, GM_ADDR gradExpandedX, GM_ADDR gradScales,
    GM_ADDR workspace, const MoeFinalizeRoutingV2GradTilingData *tilingData, TPipe *pipe)
{
    this->SubInit(gradY, expandedRowIdx, gradExpandedX, tilingData, pipe);
    this->expandedXGm_.SetGlobalBuffer((__gm__ T1 *)expandedX);
    this->scalesGm_.SetGlobalBuffer((__gm__ T1 *)scales);
    this->expertIdxGm_.SetGlobalBuffer((__gm__ T2 *)expertIdx);
    this->biasGm_.SetGlobalBuffer((__gm__ T1 *)bias);
    this->gradScalesGm_.SetGlobalBuffer((__gm__ T1 *)gradScales);
    this->pipe_->InitBuffer(this->gradYInQueue_, 1, this->tilingData_->hiddenPrePart * sizeof(float));
    this->pipe_->InitBuffer(this->expandedRowIdxInQueue_, 1, BYTE_BLOCK);
    this->pipe_->InitBuffer(this->expandedXInQueue_, 1, this->tilingData_->hiddenPrePart * sizeof(float));
    this->pipe_->InitBuffer(this->scalesInQueue_, 1, BYTE_BLOCK);

    if (bias != nullptr) {
        this->hasBias = true;
        this->pipe_->InitBuffer(this->expertIdxInQueue_, 1, BYTE_BLOCK);
        this->pipe_->InitBuffer(this->biasInQueue_, 1, this->tilingData_->hiddenPrePart * sizeof(float));
    }

    if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
        this->orgF16OBf16UbOffset_ = this->tilingData_->hiddenPrePart;
    }

    this->pipe_->InitBuffer(this->gradExpandedXOutQueue_, 1, this->tilingData_->hiddenPrePart * sizeof(T1));
    this->pipe_->InitBuffer(this->gradScalesOutQueue_, 1, BYTE_BLOCK);
}

template <typename T1, typename T2> __aicore__ inline void MoeFinalizeRoutingV2GradWithScaleCutH<T1, T2>::Process()
{
    if (this->tilingData_->dropPadMode == 1) {
        this->InitOutCutH();
        SyncAll();
    }
    SubProcess();
}

template <typename T1, typename T2> __aicore__ inline void MoeFinalizeRoutingV2GradWithScaleCutH<T1, T2>::SubProcess()
{
    this->GetSubProcessBatches();
    if (this->hasBias) {
        SubProcessWithBias();
    } else {
        SubProcessWithOutBias();
    }
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradWithScaleCutH<T1, T2>::SubProcessWithBias()
{
    int64_t innerOffset = 0;
    for (int64_t batchIdx = this->startBatchIdx_; batchIdx < this->endBatchIdx_; batchIdx++) {
        for (int64_t innerLoopIdx = 0; innerLoopIdx < this->tilingData_->hiddenInnerLoops; innerLoopIdx++) {
            innerOffset = innerLoopIdx * this->tilingData_->hiddenPrePart;
            this->CopyInWithBias(batchIdx, innerOffset, this->tilingData_->hiddenPrePart);
            this->ComputeWithBias(innerOffset, this->tilingData_->hiddenPrePart);
        }
        if (this->tilingData_->hiddenLastPart != 0) {
            innerOffset = this->tilingData_->hiddenInnerLoops * this->tilingData_->hiddenPrePart;
            this->CopyInWithBias(batchIdx, innerOffset, this->tilingData_->hiddenLastPart);
            this->ComputeWithBias(innerOffset, this->tilingData_->hiddenLastPart);
        }

        this->CopyOutGradScales(batchIdx);
    }
}

template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradWithScaleCutH<T1, T2>::SubProcessWithOutBias()
{
    int64_t innerOffset = 0;
    for (int64_t batchIdx = this->startBatchIdx_; batchIdx < this->endBatchIdx_; batchIdx++) {
        for (int64_t innerLoopIdx = 0; innerLoopIdx < this->tilingData_->hiddenInnerLoops; innerLoopIdx++) {
            innerOffset = innerLoopIdx * this->tilingData_->hiddenPrePart;
            this->CopyInWithOutBias(batchIdx, innerOffset, this->tilingData_->hiddenPrePart);
            this->ComputeWithOutBias(innerOffset, this->tilingData_->hiddenPrePart);
        }
        if (this->tilingData_->hiddenLastPart != 0) {
            innerOffset = this->tilingData_->hiddenInnerLoops * this->tilingData_->hiddenPrePart;
            this->CopyInWithOutBias(batchIdx, innerOffset, this->tilingData_->hiddenLastPart);
            this->ComputeWithOutBias(innerOffset, this->tilingData_->hiddenLastPart);
        }

        this->CopyOutGradScales(batchIdx);
    }
}
} // namespace MoeFinalizeRoutingV2Grad

#endif // MOE_FINALIZE_ROUTING_V2_GRAD_WITH_SCALE_CUT_H_H
