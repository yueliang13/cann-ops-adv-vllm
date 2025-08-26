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
 * \file ts_moe_init_routing_tc.cpp
 * \brief MoeInitRouting用例.
 */

#include "ts_moe_init_routing.h"

MoeInitRoutingCase InitMIRCase(int64_t N, int64_t H, int64_t K, int64_t activeNum, ge::DataType xDtype, ge::DataType yDtype, ge::graphStatus result, int64_t tilingKey)
{
    MoeInitRoutingCase cs;
    cs.mParam = {N, H, K, activeNum, xDtype, yDtype};
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

void InitAndRunMIRCase(int64_t N, int64_t H, int64_t K, int64_t activeNum, ge::DataType xDtype, ge::DataType yDtype, ge::graphStatus result,
                        int64_t tilingKey)
{
    MoeInitRoutingCase cs = InitMIRCase(N, H, K, activeNum, xDtype, yDtype, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_MoeInitRouting, moe_init_routing_test0)
{
    InitAndRunMIRCase(2, 5, 3, 6, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 0);
}
