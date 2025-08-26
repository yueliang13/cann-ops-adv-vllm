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
 * \file ts_grouped_bias_add_grad_tc.cpp
 * \brief GroupedBiasAddGrad用例.
 */

#include "ts_grouped_bias_add_grad.h"

GroupedBiasAddGradCase InitNormalCaseGBAG(int64_t N, int64_t H, int64_t K, bool inputOpt,
                                    ge::DataType indexDtype, ge::DataType xDtype,
                                    ge::graphStatus result, int64_t tilingKey)
{
    GroupedBiasAddGradCase cs;

    cs.mParam = {N, H, K, inputOpt, indexDtype, xDtype};
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

void InitAndRunNormalCaseGBAG(int64_t N, int64_t H, int64_t K, bool inputOpt,
                          ge::DataType indexDtype, ge::DataType xDtype, ge::graphStatus result,
                          int64_t tilingKey)
{
    GroupedBiasAddGradCase cs = InitNormalCaseGBAG(N, H, K, inputOpt, indexDtype, xDtype,
                                            result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_GroupedBiasAddGrad, grouped_bias_add_grad_no_group_idx)
{
    InitAndRunNormalCaseGBAG(1, 1, 1, false, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1000101);
}

TEST_F(Ts_GroupedBiasAddGrad, grouped_bias_add_grad_no_group_idx_2)
{
    InitAndRunNormalCaseGBAG(1, 1024, 1, false, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1000101);
}

TEST_F(Ts_GroupedBiasAddGrad, grouped_bias_add_grad_no_group_idx_float16)
{
    InitAndRunNormalCaseGBAG(1, 1024, 1, false, ge::DT_INT32, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 1000100);
}

TEST_F(Ts_GroupedBiasAddGrad, grouped_bias_add_grad_group_idx)
{
    InitAndRunNormalCaseGBAG(1, 1, 1, true, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1000111);
}

TEST_F(Ts_GroupedBiasAddGrad, grouped_bias_add_grad_group_idx_f16)
{
    InitAndRunNormalCaseGBAG(1, 1, 1, true, ge::DT_INT32, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 1000110);
}

TEST_F(Ts_GroupedBiasAddGrad, grouped_bias_add_grad_group_idx_bf16)
{
    InitAndRunNormalCaseGBAG(1, 1, 1, true, ge::DT_INT32, ge::DT_BF16, ge::GRAPH_SUCCESS, 1000112);
}

TEST_F(Ts_GroupedBiasAddGrad, grouped_bias_add_grad_group_idx_64)
{
    InitAndRunNormalCaseGBAG(1, 1, 1, true, ge::DT_INT64, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1010111);
}