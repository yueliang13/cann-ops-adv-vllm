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
 * \file ts_moe_init_routing_v2_tc.cpp
 * \brief MoeInitRoutingV2用例.
 */

#include "ts_moe_init_routing_v2.h"

namespace {
    constexpr int64_t LARGE_TOKEN_NUM = 320;
    constexpr int64_t LARGE_HiDDEN = 3000;
    constexpr int64_t ACTIVATE_NUM = 1000;
    constexpr int64_t ZERO_NUM = 0;
    constexpr int64_t LARGE_TOPK = 56;
    constexpr int64_t PER_TILINGKEY = 20000;
    constexpr int64_t SINGLE_CORE_TILINGKEY = 10011;
    constexpr int64_t ERROR_TILINGKEY = 10011;

}

MoeInitRoutingV2Case InitNormalCase(int64_t N, int64_t H, int64_t K, int64_t activeNum, int64_t C, int64_t E,
                                    int64_t dropPadMode, int64_t countFlag, bool tokenFlag,
                                    ge::DataType optionalOutputDt, ge::graphStatus result, int64_t tilingKey)
{
    MoeInitRoutingV2Case cs;
    cs.mParam = {N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag, optionalOutputDt};
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

void InitAndRunNormalCase(int64_t N, int64_t H, int64_t K, int64_t activeNum, int64_t C, int64_t E, int64_t dropPadMode,
                          int64_t countFlag, bool tokenFlag, ge::DataType optionalOutputDt, ge::graphStatus result,
                          int64_t tilingKey)
{
    MoeInitRoutingV2Case cs = InitNormalCase(N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag,
                                             optionalOutputDt, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_perf_template)
{
    InitAndRunNormalCase(8, 30, 6, 32, ZERO_NUM, 8, ZERO_NUM, 1, false, ge::DT_INT32, ge::GRAPH_SUCCESS, PER_TILINGKEY);
}

TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_single_core_drop)
{
    InitAndRunNormalCase(8, 30, 6, ZERO_NUM, 6, 8, 1, ZERO_NUM, true, ge::DT_INT32, ge::GRAPH_SUCCESS, SINGLE_CORE_TILINGKEY);
}

// x_shape dim != 2
TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_x_dim_check)
{
    MoeInitRoutingV2Case cs = InitNormalCase(LARGE_TOKEN_NUM, 60, LARGE_TOPK, ACTIVATE_NUM, ZERO_NUM, 64, ZERO_NUM, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, ERROR_TILINGKEY);
    cs.x = Tensor("x", {cs.mParam.n}, "1", cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}

// expert_tokens_before_capacity_shape_flag =1 but E = 0
TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_e_check)
{
    InitAndRunNormalCase(LARGE_TOKEN_NUM, LARGE_HiDDEN, LARGE_TOPK, ACTIVATE_NUM, ZERO_NUM, ZERO_NUM, ZERO_NUM, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, ERROR_TILINGKEY);
}

// dropPadMode = 1 but C = 0
TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_c_check)
{
    InitAndRunNormalCase(LARGE_TOKEN_NUM, LARGE_HiDDEN, LARGE_TOPK, ACTIVATE_NUM, ZERO_NUM, 64, 1, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, ERROR_TILINGKEY);
}

// expanded_x_shape shape != (E, C, H)
TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_droppad_expanded_x_shape_check)
{
    MoeInitRoutingV2Case cs = InitNormalCase(LARGE_TOKEN_NUM, LARGE_HiDDEN, LARGE_TOPK, ZERO_NUM, 30, 64, 1, ZERO_NUM, true, ge::DT_INT32, ge::GRAPH_FAILED, ERROR_TILINGKEY);
    cs.expandedX =
        Tensor("expandedX", {cs.mParam.k * cs.mParam.n, cs.mParam.h}, "2", cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape second dim != C
    cs.expandedX = Tensor("expandedX", {cs.mParam.e, cs.mParam.c * 10, cs.mParam.h}, "3", cs.mParam.optionalOutputDt,
                          ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape first dim != E
    cs.expandedX = Tensor("expandedX", {cs.mParam.e * 2, cs.mParam.c * 10, cs.mParam.h}, "3",
                          cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape third dim != H
    cs.expandedX = Tensor("expandedX", {cs.mParam.e * 2, cs.mParam.c * 10, cs.mParam.h + 1}, "3",
                          cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}

// expert_tokens_before_capacity_shape dirst dim != E
TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_tiling_expert_tokens_before_capacity_shape_check)
{
    MoeInitRoutingV2Case cs = InitNormalCase(LARGE_TOKEN_NUM, LARGE_HiDDEN, LARGE_TOPK, ZERO_NUM, 30, 64, 1, ZERO_NUM, true, ge::DT_INT32, ge::GRAPH_FAILED, ERROR_TILINGKEY);
    cs.expertTokensBeforeCapacity =
        Tensor("expertTokensBeforeCapacity", {cs.mParam.e * 2}, "1", ge::DT_INT32, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}

TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_tiling_activenum_check)
{
    MoeInitRoutingV2Case cs = InitNormalCase(LARGE_TOKEN_NUM, LARGE_HiDDEN, LARGE_TOPK, ZERO_NUM, 30, 64, ZERO_NUM, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, ERROR_TILINGKEY);
    // expanded_x_shape shape != (N * k, H)
    cs.expandedX = Tensor("expandedX", {64, 30, LARGE_HiDDEN}, "3", cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape shape != (activeNum, H)
    cs.expandedX = Tensor("expandedX", {2939, LARGE_HiDDEN}, "2", cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());

    // expanded_x_shape shape != (activeNum, H)
    cs.expandedX = Tensor("expandedX", {300, LARGE_HiDDEN}, "2", cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}

TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_negative_check)
{
    // activeNum < 0
    InitAndRunNormalCase(LARGE_TOKEN_NUM, LARGE_HiDDEN, LARGE_TOPK, -3, ZERO_NUM, 64, ZERO_NUM, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, ERROR_TILINGKEY);
    // C < 0
    InitAndRunNormalCase(LARGE_TOKEN_NUM, LARGE_HiDDEN, LARGE_TOPK, ZERO_NUM, -5, 64, ZERO_NUM, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, ERROR_TILINGKEY);
    // E < 0
    InitAndRunNormalCase(LARGE_TOKEN_NUM, LARGE_HiDDEN, LARGE_TOPK, ZERO_NUM, ZERO_NUM, -4, ZERO_NUM, 1, false, ge::DT_INT32, ge::GRAPH_FAILED, ERROR_TILINGKEY);
}

// N < capacity
TEST_F(Ts_MoeInitRoutingV2, moe_init_routing_v2_capacity_check)
{
    InitAndRunNormalCase(160, 30000, 12, ZERO_NUM, ACTIVATE_NUM, 32, 1, ZERO_NUM, false, ge::DT_INT32, ge::GRAPH_FAILED, ERROR_TILINGKEY);
}