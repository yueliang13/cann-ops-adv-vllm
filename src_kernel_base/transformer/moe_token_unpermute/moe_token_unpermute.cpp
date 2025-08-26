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
 * \file moe_token_unpermute.cpp
 * \brief
 */

#include "moe_token_unpermute.h"
#include "kernel_operator.h"

#define MOE_TOKEN_UNPERMUTE_IMPL(T1, T2, T3, haveProbs)                                                                \
    do {                                                                                                               \
        GET_TILING_DATA(tiling_data_in, tiling);                                                                       \
        const MoeTokenUnpermuteTilingData *__restrict tiling_data = &tiling_data_in;                                   \
        KernelMoeTokenUnpermute<T1, T2, T3, haveProbs> op;                                                             \
        op.Init(permuted_tokens, sorted_indices, probs, unpermuted_tokens, tiling_data);                               \
        op.Process();                                                                                                  \
    } while (0)

extern "C" __global__ __aicore__ void moe_token_unpermute(GM_ADDR permuted_tokens, GM_ADDR sorted_indices,
                                                          GM_ADDR probs, GM_ADDR unpermuted_tokens, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    //==================================BF16==================================
    if (TILING_KEY_IS(0)) {
        MOE_TOKEN_UNPERMUTE_IMPL(bfloat16_t, int32_t, bfloat16_t, false);
    } else if (TILING_KEY_IS(1)) {
        MOE_TOKEN_UNPERMUTE_IMPL(bfloat16_t, int32_t, bfloat16_t, true);
    } else if (TILING_KEY_IS(17)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(bfloat16_t, int32_t, half, true);
    } else if (TILING_KEY_IS(25)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(bfloat16_t, int32_t, float, true);
    }
    //==================================FP16==================================
    else if (TILING_KEY_IS(2)) {
        MOE_TOKEN_UNPERMUTE_IMPL(half, int32_t, half, false);
    } else if (TILING_KEY_IS(3)) {
        MOE_TOKEN_UNPERMUTE_IMPL(half, int32_t, half, true);
    } else if (TILING_KEY_IS(11)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(half, int32_t, bfloat16_t, true);
    } else if (TILING_KEY_IS(27)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(half, int32_t, float, true);
    }
    //==================================FP32==================================
    else if (TILING_KEY_IS(4)) {
        MOE_TOKEN_UNPERMUTE_IMPL(float, int32_t, float, false);
    } else if (TILING_KEY_IS(5)) {
        MOE_TOKEN_UNPERMUTE_IMPL(float, int32_t, float, true);
    } else if (TILING_KEY_IS(13)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(float, int32_t, bfloat16_t, true);
    } else if (TILING_KEY_IS(21)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(float, int32_t, half, true);
    }
}
