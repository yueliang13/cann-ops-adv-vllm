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
 * \file scaled_masked_softmax_v2.cpp
 * \brief
 */

#include "scaled_masked_softmax_v2.h"

extern "C" __global__ __aicore__ void scaled_masked_softmax_v2(GM_ADDR x, GM_ADDR mask, GM_ADDR y,
                                                            GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        AscendC::ScaledMaskedSoftmaxV2<float> op;
        op.Init(x, mask, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        AscendC::ScaledMaskedSoftmaxV2<half> op;
        op.Init(x, mask, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        AscendC::ScaledMaskedSoftmaxV2<bfloat16_t> op;
        op.Init(x, mask, y, tilingData);
        op.Process();
    }
}