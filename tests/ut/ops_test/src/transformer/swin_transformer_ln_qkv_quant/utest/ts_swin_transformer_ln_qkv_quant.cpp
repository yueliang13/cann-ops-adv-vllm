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
 * \file ts_moe_gating_top_k_softmax_v2.cpp
 * \brief MoeGatingTopKSoftmaxV2用例.
 */

#include "ts_swin_transformer_ln_qkv_quant.h"

SwinTransformerLnQkvQuantCase InitNormalCase(int64_t B, int64_t S, int64_t H, int64_t ori_height, int64_t ori_weight, 
 int64_t headNum,  int64_t hWin, int64_t wWin, int64_t sizePerhead, bool bTrans, float epslion, ge::graphStatus result, int64_t tilingKey)
{
    SwinTransformerLnQkvQuantCase cs;

    cs.mParam = SwinTransformerLnQkvQuantCase::Param(B, S, H, ori_height, ori_weight, headNum, hWin, wWin, sizePerhead, bTrans, epslion);
    if (result == ge::GRAPH_SUCCESS) {
        cs.mOpInfo.mExp.mTilingKey = tilingKey;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = false;
    } else {
        cs.mOpInfo.mExp.mSuccess = false;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = false;
    }
    cs.Init();
    return cs;
}

void InitAndRunNormalCase(int64_t B, int64_t S, int64_t H, int64_t ori_height, int64_t ori_weight, int64_t headNum,  int64_t hWin,\
    int64_t wWin, int64_t sizePerhead, bool bTrans, float epslion, ge::graphStatus result, int64_t tilingKey)
{
    SwinTransformerLnQkvQuantCase cs = InitNormalCase(B, S, H, ori_height, ori_weight, headNum, hWin, wWin, sizePerhead, bTrans, epslion, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}


TEST_F(Ts_SwinTransformerLnQkvQuant_Ascend310P3, swin_transformer_ln_qkv_quant_100000)
{
    InitAndRunNormalCase(1, 49, 32, 7, 7, 1, 7, 7, 32, 1, 0.0001f, ge::GRAPH_SUCCESS, 100000);
}