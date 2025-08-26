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
 * \file ts_arope_tc.cpp
 * \brief ApplyRotaryPosEmb 用例.
 */

#include "ts_arope.h"

ARopeCase InitARopeCase(int64_t pb, int64_t ps, int64_t pqn, int64_t pkn, int64_t pd, ge::DataType pDataType,
                        ge::graphStatus result, int64_t tilingKey)
{
    ARopeCase cs;
    cs.mParam = {pb, ps, pqn, pkn, pd, pDataType};
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

void InitAndRunARopeCase(int64_t pb, int64_t ps, int64_t pqn, int64_t pkn, int64_t pd, ge::DataType pDataType,
                         ge::graphStatus result, int64_t tilingKey)
{
    ARopeCase cs = InitARopeCase(pb, ps, pqn, pkn, pd, pDataType, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_ARope_Ascend910B1, case_case0)
{
    InitAndRunARopeCase(2, 4, 2, 1, 128, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 1);
}

TEST_F(Ts_ARope_Ascend910B1, case_case1)
{
    InitAndRunARopeCase(16, 50, 4, 1, 128, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 4);
}

TEST_F(Ts_ARope_Ascend910B1, case_case2)
{
    InitAndRunARopeCase(2, 4, 1, 1, 128, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1);
}

TEST_F(Ts_ARope_Ascend910B1, case_case3)
{
    InitAndRunARopeCase(4, 100, 16, 1, 128, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 4);
}

TEST_F(Ts_ARope_Ascend910B1, case_case4)
{
    InitAndRunARopeCase(2, 8, 1, 1, 128, ge::DT_BF16, ge::GRAPH_SUCCESS, 1);
}

TEST_F(Ts_ARope_Ascend910B1, case_case5)
{
    InitAndRunARopeCase(4, 256, 16, 1, 128, ge::DT_BF16, ge::GRAPH_SUCCESS, 4);
}