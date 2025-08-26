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
 * \file ts_rope_tc.cpp
 * \brief RotaryPositionEmbedding用例.
 */

#include "ts_rope_infer.h"

RopeInferCase InitRopeInferCase(int64_t pBatch, int64_t pHiddensizeQ, int64_t pHiddensizeK, int64_t pHeadDim, bool pLargeDim, int64_t pRotaryCoefficiency,
    std::string pLayout, ge::DataType pDataType, ge::DataType pDataTypeCos, ge::graphStatus result, int64_t tilingKey)
{
    RopeInferCase cs;
    cs.mParam = {pBatch, pHiddensizeQ, pHiddensizeK, pHeadDim, pLargeDim, pRotaryCoefficiency, pLayout, pDataType, pDataTypeCos};
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

void InitAndRunRopeInferCase(int64_t pBatch, int64_t pHiddensizeQ, int64_t pHiddensizeK, int64_t pHeadDim, bool pLargeDim, int64_t pRotaryCoefficiency,
    std::string pLayout, ge::DataType pDataType, ge::DataType pDataTypeCos, ge::graphStatus result, int64_t tilingKey)
{
    RopeInferCase cs = InitRopeInferCase(pBatch, pHiddensizeQ, pHiddensizeK, pHeadDim, pLargeDim, pRotaryCoefficiency,
        pLayout, pDataType, pDataTypeCos, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_Rope, case_ND_case0)
{
    InitAndRunRopeInferCase(16, 16640, 9216, 256, false, 2, "ND", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 20);
}

TEST_F(Ts_Rope, case_ND_case1)
{
    InitAndRunRopeInferCase(320, 16640, 9216, 256, false, 2, "ND", ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 21);
}

TEST_F(Ts_Rope, case_ND_case2)
{
    InitAndRunRopeInferCase(320, 16640, 9216, 256, false, 2, "ND", ge::DT_BF16, ge::DT_BF16, ge::GRAPH_SUCCESS, 22);
}

TEST_F(Ts_Rope, case_ND_case3)
{
    InitAndRunRopeInferCase(320, 16640, 9216, 256, false, 2, "ND", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 23);
}

TEST_F(Ts_Rope, case_ND_case4)
{
    InitAndRunRopeInferCase(320, 16640, 9216, 256, true, 2, "ND", ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 31);
}

TEST_F(Ts_Rope, case_ND_case5)
{
    InitAndRunRopeInferCase(320, 16640, 9216, 256, true, 256, "ND", ge::DT_BF16, ge::DT_BF16, ge::GRAPH_SUCCESS, 32);
}

TEST_F(Ts_Rope, case_ND_case6)
{
    InitAndRunRopeInferCase(320, 16640, 9216, 256, true, 2, "ND", ge::DT_BF16, ge::DT_BF16, ge::GRAPH_SUCCESS, 33);
}

TEST_F(Ts_Rope, case_ND_small_case0)
{
    InitAndRunRopeInferCase(16, 1024, 1024, 256, false, 2, "ND", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 20);
}

TEST_F(Ts_Rope, case_ND_small_case1)
{
    InitAndRunRopeInferCase(320, 1024, 1024, 256, false, 2, "ND", ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 21);
}

TEST_F(Ts_Rope, case_ND_small_case2)
{
    InitAndRunRopeInferCase(320, 1024, 1024, 256, false, 2, "ND", ge::DT_BF16, ge::DT_BF16, ge::GRAPH_SUCCESS, 22);
}

TEST_F(Ts_Rope, case_ND_small_case3)
{
    InitAndRunRopeInferCase(320, 1024, 1024, 256, false, 2, "ND", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 23);
}

TEST_F(Ts_Rope, case_ND_small_case4)
{
    InitAndRunRopeInferCase(320, 1024, 1024, 256, true, 2, "ND", ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 31);
}

TEST_F(Ts_Rope, case_ND_small_case5)
{
    InitAndRunRopeInferCase(320, 1024, 1024, 256, true, 256, "ND", ge::DT_BF16, ge::DT_BF16, ge::GRAPH_SUCCESS, 32);
}

TEST_F(Ts_Rope, case_ND_small_case6)
{
    InitAndRunRopeInferCase(320, 1024, 1024, 256, true, 2, "ND", ge::DT_BF16, ge::DT_BF16, ge::GRAPH_SUCCESS, 33);
}

TEST_F(Ts_Rope, case_ND_rt1_case0)
{
InitAndRunRopeInferCase(320, 1024, 1024, 256, true, 256, "ND", ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 31);
}

TEST_F(Ts_Rope, case_ND_rt1_case1)
{
InitAndRunRopeInferCase(16, 1024, 1024, 256, false, 256, "ND", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 20);
}
TEST_F(Ts_Rope, case_ND_rt1_case2)
{
InitAndRunRopeInferCase(320, 1024, 1024, 256, false, 256, "ND", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 23);
}