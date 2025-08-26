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
 * \file ts_moe_compute_expert_tokens.cpp
 * \brief MoeComputeExpertTokens用例.
 */

#include "ts_moe_compute_expert_tokens.h"

MoeComputeExpertTokensCase InitNormalCase(int64_t N, int64_t E, int64_t numExperts, ge::DataType dataType, ge::graphStatus result, int64_t tilingKey)
{
    MoeComputeExpertTokensCase cs;
    cs.mParam = {N, E, numExperts, dataType};
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

void InitAndRunNormalCase(int64_t N, int64_t E, int64_t numExperts, ge::DataType dataType, ge::graphStatus result,
                          int64_t tilingKey)
{
    MoeComputeExpertTokensCase cs = InitNormalCase(N, E, numExperts, dataType, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_MoeComputeExpertTokens, moe_compute_expert_tokens_1)
{
    InitAndRunNormalCase(20000, 100, 100,  ge::DT_INT32, ge::GRAPH_SUCCESS, 1002);
}
