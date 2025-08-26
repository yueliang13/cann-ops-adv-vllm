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
 * \file moe_distribute_dispatch.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "moe_distribute_dispatch_tiling.h"
#include "moe_distribute_dispatch_a2.h"
#include "moe_distribute_dispatch_a2_layered.h"

using namespace AscendC;
using namespace MoeDistributeDispatchA2Impl;
extern "C" __global__ __aicore__ void moe_distribute_dispatch(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchA2TilingData);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(2000001000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
        op.Init(x,expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                workspaceGM, &pipe, tilingGM);
        op.Process();
    } else if (TILING_KEY_IS(2100001000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
        op.Init(x,expertIds, scales, expertScales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
                epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if (TILING_KEY_IS(2000001002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
        op.Init(x,expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                workspaceGM, &pipe, tilingGM);
        op.Process();
    } else if (TILING_KEY_IS(2000001012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
        op.Init(x,expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                workspaceGM, &pipe, tilingGM);
        op.Process();
    } else if (TILING_KEY_IS(2100001002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
        op.Init(x,expertIds, scales, expertScales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
                epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    } else if (TILING_KEY_IS(2100001012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
        op.Init(x,expertIds, scales, expertScales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
                epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    }
#endif
}