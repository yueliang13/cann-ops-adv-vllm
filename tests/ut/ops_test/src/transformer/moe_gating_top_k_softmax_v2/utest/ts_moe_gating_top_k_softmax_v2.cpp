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

#include "ts_moe_gating_top_k_softmax_v2.h"

MoeGatingTopKSoftmaxV2Case InitNormalCase(int64_t N, int64_t H, int64_t K, int64_t renorm, ge::DataType xDtype,
ge::graphStatus result, int64_t tilingKey)
{
MoeGatingTopKSoftmaxV2Case cs;

cs.mParam = {N, H, K, renorm, xDtype};
if (result == ge::GRAPH_SUCCESS) {
    cs.mOpInfo.mExp.mTilingKey = tilingKey;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
} else {
    cs.mOpInfo.mExp.mSuccess = false;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
}
cs.Init();
return cs;
}

void InitAndRunNormalCase(int64_t N, int64_t H, int64_t K, int64_t renorm, ge::DataType xDtype, ge::graphStatus result,
                          int64_t tilingKey)
{
    MoeGatingTopKSoftmaxV2Case cs = InitNormalCase(N, H, K, renorm, xDtype, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}


TEST_F(Ts_MoeGatingTopKSoftmaxV2, moe_gating_top_k_softmax_v2_101011)
{
    InitAndRunNormalCase(20, 20, 2, 0, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 101011);
}

TEST_F(Ts_MoeGatingTopKSoftmaxV2, moe_gating_top_k_softmax_v2_102011)
{
    InitAndRunNormalCase(20, 8161, 2, 0, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 102011);
}

TEST_F(Ts_MoeGatingTopKSoftmaxV2, moe_gating_top_k_softmax_v2_102111)
{
    InitAndRunNormalCase(20, 8161, 2, 1, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 102111);
}

TEST_F(Ts_MoeGatingTopKSoftmaxV2, moe_gating_top_k_softmax_v2_103011)
{
    InitAndRunNormalCase(20, 6, 2, 0, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 103011);
}