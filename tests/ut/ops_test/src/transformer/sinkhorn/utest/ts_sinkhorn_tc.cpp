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
 * \file ts_sinkhorn_tc.cpp
 * \brief Sinkhorn用例.
 */

#include "ts_sinkhorn.h"

SinkhornCase InitSinkhornCase(int64_t pTokens, int64_t pExperts, float pTol, ge::DataType pDataType, ge::graphStatus result, int64_t tilingKey)
{
    SinkhornCase cs;
    cs.mParam = {pTokens, pExperts, pTol, pDataType};
    if (result == ge::GRAPH_SUCCESS) {
        cs.mOpInfo.mExp.mTilingKey = tilingKey;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = true;
    }  else {
        cs.mOpInfo.mExp.mSuccess = false;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = false;
    }
    cs.Init();
    return cs;
}

void InitAndRunSinkhornCase(int64_t pTokens, int64_t pExperts, float pTol, ge::DataType pDataType, ge::graphStatus result, int64_t tilingKey)
{
    SinkhornCase cs = InitSinkhornCase(pTokens, pExperts, pTol, pDataType, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_Sinkhorn, test_sinkhorn_case0)
{
    InitAndRunSinkhornCase(200, 16, 0.0001f, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 0);
}

TEST_F(Ts_Sinkhorn, test_sinkhorn_case1)
{
    InitAndRunSinkhornCase(128, 32, 0.0001f, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 1);
}

TEST_F(Ts_Sinkhorn, test_sinkhorn_case2)
{
    InitAndRunSinkhornCase(128, 80, 0.0001f, ge::DT_BF16, ge::GRAPH_SUCCESS, 27);
}