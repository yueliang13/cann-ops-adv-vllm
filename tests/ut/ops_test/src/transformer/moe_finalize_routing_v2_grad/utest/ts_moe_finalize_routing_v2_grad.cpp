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
 * \file ts_moe_finalize_routing_v2_grad_tc.cpp
 * \brief MoeFinalizeRoutingV2Grad用例.
 */

#include "ts_moe_finalize_routing_v2_grad.h"

MoeFinalizeRoutingV2GradCase InitMFRGCase(int64_t E, int64_t C,int64_t numRows, int64_t H, int64_t K, int64_t dropPadMode, ge::DataType dataType, ge::graphStatus result, int64_t tilingKey)
{
    MoeFinalizeRoutingV2GradCase cs;
    cs.mParam = {E, C, numRows, H, K, dropPadMode, dataType};
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

void InitAndRunMFRGCase(int64_t E, int64_t C,int64_t numRows, int64_t H, int64_t K, int64_t dropPadMode, ge::DataType dataType, ge::graphStatus result,
                          int64_t tilingKey)
{
    MoeFinalizeRoutingV2GradCase cs = InitMFRGCase(E, C, numRows, H, K, dropPadMode, dataType, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_MoeFinalizeRoutingV2Grad, moe_finalize_routing_v2_grad_0)
{
    InitAndRunMFRGCase(1, 0, 5, 8, 1, 0, ge::DataType::DT_FLOAT16, ge::GRAPH_SUCCESS, 20001);
}
