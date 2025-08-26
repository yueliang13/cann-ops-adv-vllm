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
 * \file scaled_masked_softmax_grad_v2.cpp
 * \brief
 */
#include "scaled_masked_softmax_grad_v2_norm_headdim.h"
#include "scaled_masked_softmax_grad_v2_large_headdim.h"

using namespace ScaledMaskedSoftmaxGradV2;

extern "C" __global__ __aicore__ void scaled_masked_softmax_grad_v2(
    const GM_ADDR yGrad,
    const GM_ADDR y,
    const GM_ADDR mask,
    const GM_ADDR xGrad,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipeIn;

    if (TILING_KEY_IS(1000)) {
        ScaledMaskedSoftmaxGradV2NormHeadDim<bfloat16_t> op;
        op.Init(yGrad, y, mask, xGrad, tilingData, &pipeIn);
        op.Process();
    } else if (TILING_KEY_IS(1001)) {
        ScaledMaskedSoftmaxGradV2NormHeadDim<half> op;
        op.Init(yGrad, y, mask, xGrad, tilingData, &pipeIn);
        op.Process();
    } else if (TILING_KEY_IS(1002)) {
        ScaledMaskedSoftmaxGradV2NormHeadDim<float> op;
        op.Init(yGrad, y, mask, xGrad, tilingData, &pipeIn);
        op.Process();
    } else if (TILING_KEY_IS(2000)) {
        ScaledMaskedSoftmaxGradV2LargeHeadDim<bfloat16_t> op;
        op.Init(yGrad, y, mask, xGrad, tilingData, &pipeIn);
        op.Process();
    } else if (TILING_KEY_IS(2001)) {
        ScaledMaskedSoftmaxGradV2LargeHeadDim<half> op;
        op.Init(yGrad, y, mask, xGrad, tilingData, &pipeIn);
        op.Process();
    } else if (TILING_KEY_IS(2002)) {
        ScaledMaskedSoftmaxGradV2LargeHeadDim<float> op;
        op.Init(yGrad, y, mask, xGrad, tilingData, &pipeIn);
        op.Process();
    }
}