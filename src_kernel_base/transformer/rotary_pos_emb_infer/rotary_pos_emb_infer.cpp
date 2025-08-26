/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotary_pos_emb_infer.cpp
 * \brief
 */
#include "rotary_pos_emb_fp16.h"
#include "rotary_pos_emb_fp16_large_ntokens.h"
#include "rotary_pos_emb_fp32.h"
#include "rotary_pos_emb_bf16.h"
#include "rotary_pos_emb_bf16_align.h"
using namespace RopeInfer;

extern "C" __global__ __aicore__ void rotary_pos_emb_infer(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin,
                                                           GM_ADDR seqLen, GM_ADDR outQ, GM_ADDR outK,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR sync = workspace;
    workspace = sync + tiling_data.realCore * BLK_SIZE;
    if (AscendC::GetBlockIdx() >= tiling_data.realCore){
        return ;
    }

    if (TILING_KEY_IS(31)) {
        RopeFp32<half, float, true> ropeFp32(&tiling_data, &pipe);
        ropeFp32.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp32.Process(workspace);
    } else if (TILING_KEY_IS(32)) {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        RopeBf16<bfloat16_t, bfloat16_t, true> ropeBf16(&tiling_data, &pipe);
        ropeBf16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeBf16.Process(workspace);
#endif
    } else if (TILING_KEY_IS(33)) {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        RopeBf16Align<bfloat16_t, bfloat16_t, true> ropeBf16(&tiling_data, &pipe);
        ropeBf16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeBf16.Process(workspace);
#endif
    } else if (TILING_KEY_IS(20)) {
        RopeFp16<half, half, false> ropeFp16(&tiling_data, &pipe);
        ropeFp16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp16.Process(workspace, sync);
    } else if (TILING_KEY_IS(21)) {
        RopeFp32<half, float, false> ropeFp32(&tiling_data, &pipe);
        ropeFp32.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp32.Process(workspace);
    } else if (TILING_KEY_IS(22)) {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        RopeBf16<bfloat16_t, bfloat16_t, false> ropeBf16(&tiling_data, &pipe);
        ropeBf16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeBf16.Process(workspace);
#endif
    } else if (TILING_KEY_IS(23)) {
        RopeFp16LargeNtokens<half, half, false> ropeFp16LargeNtokens(&tiling_data, &pipe);
        ropeFp16LargeNtokens.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp16LargeNtokens.Process(workspace, sync);
    }
}