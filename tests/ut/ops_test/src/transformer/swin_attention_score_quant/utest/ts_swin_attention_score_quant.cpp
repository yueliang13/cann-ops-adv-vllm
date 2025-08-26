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

#include "ts_swin_attention_score_quant.h"

SwinAttentionScoreQuantCase InitNormalCase(int64_t pB, int64_t pN, int64_t pS, int64_t pH,
                        bool qTranspose, bool kTranspose, bool vTranspose, int pSoftmaxAxes, ge::graphStatus result, int64_t tilingKey)
{
    SwinAttentionScoreQuantCase cs;

    cs.mParam = SwinAttentionScoreQuantCase::Param(pB, pN, pS, pH, qTranspose, kTranspose, vTranspose, pSoftmaxAxes);
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

void InitAndRunNormalCase(int64_t pB, int64_t pN, int64_t pS, int64_t pH, 
                            bool qTranspose, bool kTranspose, bool vTranspose, int pSoftmaxAxes, ge::graphStatus result, int64_t tilingKey)
{
    SwinAttentionScoreQuantCase cs = InitNormalCase(pB, pN, pS, pH, qTranspose, kTranspose, vTranspose, pSoftmaxAxes, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}


TEST_F(Ts_SwinAttentionScoreQuant_Ascend310P3, swin_attention_score_quant_1)
{
    InitAndRunNormalCase(1288, 3, 49, 32, 0, 0, 0, -1, ge::GRAPH_SUCCESS, 1);
}