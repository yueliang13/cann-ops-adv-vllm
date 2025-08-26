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
 * \file moe_distribute_combine.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "moe_distribute_combine_tiling.h"
#include "moe_distribute_combine_a2.h"
#include "moe_distribute_combine_a2_layered.h"

using namespace AscendC;
using namespace MoeDistributeCombineA2Impl;
extern "C" __global__ __aicore__ void moe_distribute_combine(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                                             GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount,
                                                             GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale,
                                                             GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR XOut, 
                                                             GM_ADDR workspaceGM, GM_ADDR tilingGM)

{
  REGISTER_TILING_DEFAULT(MoeDistributeCombineA2TilingData);
  TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
  if (TILING_KEY_IS(2000)) {
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
    auto tiling = (__gm__ MoeDistributeCombineA2TilingData*)tilingGM;
    __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));
    __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));
    MoeDistributeCombineA2<DTYPE_EXPAND_X, int32_t> op;
    op.Init(expandX, expertIds, expandIdx, epSendCount, scales, XOut, workspaceGM, &pipe, &tilingData,
      mc2InitTiling, mc2CcTiling);
    op.Process();
  } else if (TILING_KEY_IS(3000)) {
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
    auto tiling = (__gm__ MoeDistributeCombineA2TilingData*)tilingGM;
    __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));
    __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));
    MoeDistributeCombineA2Layered<DTYPE_EXPAND_X, int32_t> op;
    op.Init(expandX, expertIds, expandIdx, epSendCount, expandScales, XOut, workspaceGM, &pipe, &tilingData,
      mc2InitTiling, mc2CcTiling);
    op.Process();
  }
#endif
}