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
 * \file ts_moe_token_unpermute.cpp
 * \brief MoeTokenUnpermute用例.
 */

#include "ts_moe_token_unpermute.h"

MoeTokenUnpermuteCase InitNormalCase(int64_t N, int64_t H, int64_t K, bool paddedMode, ge::DataType permutedTokensDtype,
                                     ge::DataType sortedIndicesDtype, ge::DataType probsDtype, ge::graphStatus result,
                                     int64_t tilingKey)
{
    MoeTokenUnpermuteCase cs;

    cs.mParam = {N, H, K, paddedMode, permutedTokensDtype, sortedIndicesDtype, probsDtype};
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

void InitAndRunNormalCase(int64_t N, int64_t H, int64_t K, bool paddedMode, ge::DataType permutedTokensDtype,
                          ge::DataType sortedIndicesDtype, ge::DataType probsDtype, ge::graphStatus result,
                          int64_t tilingKey)
{
    MoeTokenUnpermuteCase cs =
        InitNormalCase(N, H, K, paddedMode, permutedTokensDtype, sortedIndicesDtype, probsDtype, result, tilingKey);

    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_MoeTokenUnpermute, moe_token_unpermute_0)
{
    InitAndRunNormalCase(32, 4096, 8, false, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 3);
}

TEST_F(Ts_MoeTokenUnpermute, moe_token_unpermute_1)
{
    InitAndRunNormalCase(32, 4096, 1, false, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 5);
}