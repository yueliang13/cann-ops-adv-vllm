/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_token_permute_grad.cpp
 * \brief
 */

#include "moe_token_permute_grad.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void moe_token_permute_grad(GM_ADDR permutedTokens, GM_ADDR sortedIndices,
                                                             GM_ADDR permutedTokensGrad, GM_ADDR workspace,
                                                             GM_ADDR tiling)
{
    GET_TILING_DATA(tilingDataIn, tiling);
    const MoeTokenPermuteGradTilingData *__restrict tilingData = &tilingDataIn;

    // 调用unpermute计算， probs始终false
    if (TILING_KEY_IS(0)) {
        KernelMoeTokenPermuteGrad<bfloat16_t, int32_t, bfloat16_t, false> op;
        op.Init(permutedTokens, sortedIndices, nullptr, permutedTokensGrad, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelMoeTokenPermuteGrad<half, int32_t, half, false> op;
        op.Init(permutedTokens, sortedIndices, nullptr, permutedTokensGrad, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelMoeTokenPermuteGrad<float, int32_t, float, false> op;
        op.Init(permutedTokens, sortedIndices, nullptr, permutedTokensGrad, tilingData);
        op.Process();
    }
}
