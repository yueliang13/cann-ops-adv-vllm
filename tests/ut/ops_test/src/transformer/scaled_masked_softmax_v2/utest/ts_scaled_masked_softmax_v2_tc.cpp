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
 * \file ts_scaled_masked_softmax_v2_tc.cpp
 * \brief
 */

#include "ts_scaled_masked_softmax_v2.h"

ScaledMaskedSoftmaxV2Case InitNormalCase(int64_t b, int64_t n, int64_t s1, int64_t s2, float scale, bool genMask,
                                        ScaledMaskedSoftmaxV2Case::MaskType maskType, ge::DataType xDtype,
                                        ge::graphStatus result, int64_t tilingKey)
{
    ScaledMaskedSoftmaxV2Case cs;

    cs.mParam = {b, n, s1, s2, scale, genMask, maskType, xDtype};
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

void InitAndRunNormalCase(int64_t b, int64_t n, int64_t s1, int64_t s2, float scale, bool genMask,
                        ScaledMaskedSoftmaxV2Case::MaskType maskType, ge::DataType xDtype,
                        ge::graphStatus result, int64_t tilingKey)
{
    ScaledMaskedSoftmaxV2Case cs = InitNormalCase(b, n, s1, s2, scale, genMask, maskType, xDtype,
                                                result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_ScaledMaskedSoftmaxV2, scaled_masked_softmax_v2_sameBN)
{
    InitAndRunNormalCase(8, 16, 128, 128, 1.0, false, ScaledMaskedSoftmaxV2Case::MaskType::SameShape, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 0);
}

TEST_F(Ts_ScaledMaskedSoftmaxV2, scaled_masked_softmax_v2_broadcastB)
{
    InitAndRunNormalCase(8, 16, 128, 128, 1.0, false, ScaledMaskedSoftmaxV2Case::MaskType::BroadCastB, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 1);
}

TEST_F(Ts_ScaledMaskedSoftmaxV2, scaled_masked_softmax_v2_broadcastN)
{
    InitAndRunNormalCase(8, 16, 128, 128, 1.0, false, ScaledMaskedSoftmaxV2Case::MaskType::BroadCastB, ge::DT_BF16, ge::GRAPH_SUCCESS, 2);
}