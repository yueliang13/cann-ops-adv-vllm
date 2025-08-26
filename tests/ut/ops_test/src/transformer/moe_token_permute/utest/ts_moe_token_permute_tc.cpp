/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file ts_moe_token_permute_tc.cpp
 * \brief MoeTokenPermute用例.
 */

#include "ts_moe_token_permute.h"

MoeTokenPermuteCase InitNormalCase(int64_t N, int64_t H, int64_t K, int64_t activeNum, int64_t C, int64_t E,
                                    bool dropPadMode,
                                    ge::DataType indexDtype, ge::DataType xDtype,
                                    ge::graphStatus result, int64_t tilingKey)
{
    MoeTokenPermuteCase cs;

    cs.mParam = {N, H, K, activeNum, C, E, dropPadMode, indexDtype, xDtype};
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

void InitAndRunNormalCase(int64_t N, int64_t H, int64_t K, int64_t activeNum, int64_t C, int64_t E, bool dropPadMode,
                          ge::DataType indexDtype, ge::DataType xDtype, ge::graphStatus result,
                          int64_t tilingKey)
{
    MoeTokenPermuteCase cs = InitNormalCase(N, H, K, activeNum, C, E, dropPadMode, indexDtype, xDtype,
                                            result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_MoeTokenPermute, moe_token_permute_multi_core_dropless)
{
    InitAndRunNormalCase(3200, 3000, 56, 1000, 0, 64,false, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 6);
}

TEST_F(Ts_MoeTokenPermute, moe_token_permute_perf_template)
{
    InitAndRunNormalCase(8, 30, 6, 32, 0, 8,  false, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 5);
}

TEST_F(Ts_MoeTokenPermute, moe_token_permute_single_core_drop)
{
    InitAndRunNormalCase(8, 30, 6, 0, 6, 8, true, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1);
}

// x_shape dim != 2
TEST_F(Ts_MoeTokenPermute, moe_token_permute_x_dim_check)
{
    MoeTokenPermuteCase cs = InitNormalCase(320, 60, 56, 1000, 0, 64, false, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_FAILED, 0);
    cs.x = Tensor("x", {cs.mParam.n}, "1", cs.mParam.xDtype, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}

// expert_tokens_before_capacity_shape_flag =1 but E = 0
TEST_F(Ts_MoeTokenPermute, moe_token_permute_e_check)
{
    InitAndRunNormalCase(320, 3000, 56, 1000, 0, 0, false, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 6);
}

// dropPadMode = 1 but C = 0
TEST_F(Ts_MoeTokenPermute, moe_token_permute_c_check)
{
    InitAndRunNormalCase(320, 3000, 56, 1000, 0, 64, false, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 6);
}

// expanded_x_shape shape != (E, C, H)
TEST_F(Ts_MoeTokenPermute, moe_token_permute_droppad_expanded_x_shape_check)
{
    MoeTokenPermuteCase cs = InitNormalCase(320, 3000, 56, 0, 30, 64, true, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_FAILED, 0);

    // expanded_x_shape second dim != C
    cs.expandedX = Tensor("expandedX", {cs.mParam.e, cs.mParam.c * 10, cs.mParam.h}, "3", cs.mParam.xDtype,
                          ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape first dim != E
    cs.expandedX = Tensor("expandedX", {cs.mParam.e * 2, cs.mParam.c * 10, cs.mParam.h}, "3",
                          cs.mParam.xDtype, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape third dim != H
    cs.expandedX = Tensor("expandedX", {cs.mParam.e * 2, cs.mParam.c * 10, cs.mParam.h + 1}, "3",
                          cs.mParam.xDtype, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}

TEST_F(Ts_MoeTokenPermute, moe_token_permute_tiling_activenum_check)
{
    MoeTokenPermuteCase cs = InitNormalCase(320, 3000, 56, 0, 30, 64, false, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_FAILED, 0);
    // expanded_x_shape shape != (N * k, H)
    cs.expandedX = Tensor("expandedX", {64, 30, 3000}, "3", cs.mParam.xDtype, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape shape != (activeNum, H)
    cs.expandedX = Tensor("expandedX", {2939, 3000}, "2", cs.mParam.xDtype, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape shape != (activeNum, H)
    cs.expandedX = Tensor("expandedX", {300, 3001}, "2", cs.mParam.xDtype, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}

