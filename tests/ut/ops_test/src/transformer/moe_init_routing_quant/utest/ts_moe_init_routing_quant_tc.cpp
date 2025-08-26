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
 * \file ts_moe_init_routing_quant_tc.cpp
 * \brief MoeInitRoutingQuant用例.
 */

#include "ts_moe_init_routing_quant.h"

MoeInitRoutingQuantCase InitNormalCaseMIRQ(int64_t N, int64_t H, int64_t K, int64_t activeNum, int64_t C, int64_t E,
                                    int64_t dropPadMode, int64_t countFlag, bool tokenFlag,
                                    ge::DataType optionalOutputDt, int64_t quantMode, int64_t smoothtype, ge::graphStatus result, int64_t tilingKey)
{
    MoeInitRoutingQuantCase cs;
    cs.mParam = {N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag, optionalOutputDt, quantMode, smoothtype};
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

void InitAndRunNormalCaseMIRQ(int64_t N, int64_t H, int64_t K, int64_t activeNum, int64_t C, int64_t E, int64_t dropPadMode,
                          int64_t countFlag, bool tokenFlag, ge::DataType optionalOutputDt, int64_t quantMode, int64_t smoothtype, ge::graphStatus result,
                          int64_t tilingKey)
{
    MoeInitRoutingQuantCase cs = InitNormalCaseMIRQ(N, H, K, activeNum, C, E, dropPadMode, countFlag, tokenFlag,
                                                  optionalOutputDt, quantMode, smoothtype, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_MoeInitRoutingQuant, moe_init_routing_quant_multi_core_dropless)
{
    InitAndRunNormalCaseMIRQ(320, 320, 32, 100, 0, 64, 0, 1, false, ge::DT_INT32, 0, 0, ge::GRAPH_SUCCESS, 4);
}

TEST_F(Ts_MoeInitRoutingQuant, moe_init_routing_quant_perf_template)
{
    InitAndRunNormalCaseMIRQ(8, 32, 6, 100, 0, 8, 0, 1, false, ge::DT_INT32, 1, 0, ge::GRAPH_SUCCESS, 1);
}

TEST_F(Ts_MoeInitRoutingQuant, moe_init_routing_quant_x_dim_check)
{
    MoeInitRoutingQuantCase cs = InitNormalCaseMIRQ(320, 60, 56, 1000, 0, 64, 0, 1, false, ge::DT_INT32, 1, 2, ge::GRAPH_FAILED, 0);
    cs.x = Tensor("x", {cs.mParam.n}, "1", cs.mParam.optionalOutputDt, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}
