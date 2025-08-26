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
 * \file ts_moe_gating_top_k_softmax.cpp
 * \brief MoeGatingTopKSoftmax用例.
 */

#include "ts_moe_gating_top_k_softmax.h"

MoeGatingTopKSoftmaxCase InitMGTKCase(int64_t N, int64_t H, int64_t K, ge::DataType xDtype,
ge::graphStatus result, int64_t tilingKey)
{
MoeGatingTopKSoftmaxCase cs;

cs.mParam = {N, H, K, xDtype};
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

void InitAndRunMGTKCase(int64_t N, int64_t H, int64_t K, ge::DataType xDtype, ge::graphStatus result,
                          int64_t tilingKey)
{
    MoeGatingTopKSoftmaxCase cs = InitMGTKCase(N, H, K, xDtype, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}


TEST_F(Ts_MoeGatingTopKSoftmax, moe_gating_top_k_softmax_1)
{
    InitAndRunMGTKCase(20, 20, 2, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1);
}

TEST_F(Ts_MoeGatingTopKSoftmax, moe_gating_top_k_softmax_6)
{
    InitAndRunMGTKCase(20, 8161, 2, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 6);
}

TEST_F(Ts_MoeGatingTopKSoftmax, moe_gating_top_k_softmax_15)
{
    InitAndRunMGTKCase(20, 80, 2, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 15);
}

TEST_F(Ts_MoeGatingTopKSoftmax, moe_gating_top_k_softmax_9)
{
    InitAndRunMGTKCase(20, 6, 2, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 9);
}