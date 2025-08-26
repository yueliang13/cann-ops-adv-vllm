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
 * \file ts_scaled_masked_softmax_grad_v2_tc.cpp
 * \brief ScaledMaskedSoftmaxGradV2用例.
 */

#include "ts_scaled_masked_softmax_grad_v2.h"

ScaledMaskedSoftmaxGradV2Case InitScaledMaskedSoftmaxGradV2Case(int64_t pb, int64_t pn, int64_t ps, int64_t pd,
    int64_t pmaskB, int64_t pmaskN, float pscaleValue, bool pfixedTriuMask, ge::DataType pDataType,
    ge::DataType pmaskDataType, ge::graphStatus result, int64_t tilingKey)
{
    ScaledMaskedSoftmaxGradV2Case cs;
    cs.mParam = {pb, pn, ps, pd, pmaskB, pmaskN, pscaleValue, pfixedTriuMask, pDataType, pmaskDataType};
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

void InitAndRunScaledMaskedSoftmaxGradV2Case(int64_t pb, int64_t pn, int64_t ps, int64_t pd, int64_t pmaskB,
    int64_t pmaskN, float pscaleValue, bool pfixedTriuMask, ge::DataType pDataType, ge::DataType pmaskDataType,
    ge::graphStatus result, int64_t tilingKey)
{
    ScaledMaskedSoftmaxGradV2Case cs = InitScaledMaskedSoftmaxGradV2Case(pb, pn, ps, pd, pmaskB, pmaskN, pscaleValue, pfixedTriuMask, pDataType, pmaskDataType, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode0_norm_headdim_case0)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 128, 4, 4,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_BF16,         // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 1000);  // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode0_norm_headdim_case1)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 128, 4, 4,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT16,      // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 1001);  // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode0_norm_headdim_case2)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 128, 4, 4,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT,        // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 1002);  // mask dtype, status, tilingKey
}

TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode1_norm_headdim_case0)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 128, 1, 4,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_BF16,         // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 1000);  // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode1_norm_headdim_case1)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 128, 1, 4,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT16,      // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 1001);  // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode1_norm_headdim_case2)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 128, 1, 4,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT,        // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 1002);  // mask dtype, status, tilingKey
}

TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode2_norm_headdim_case0)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 128, 4, 1,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_BF16,         // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 1000);  // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode2_norm_headdim_case1)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 128, 4, 1,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT16,      // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 1001);  // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode2_norm_headdim_case2)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 128, 4, 1,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT,        // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 1002);  // mask dtype, status, tilingKey
}

TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode3_norm_headdim_case0)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 128, 1, 1,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_BF16,         // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 1000);  // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode3_norm_headdim_case1)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 128, 1, 1,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT16,      // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 1001);  // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode3_norm_headdim_case2)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 128, 1, 1,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT,        // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 1002);  // mask dtype, status, tilingKey
}

TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode0_large_headdim_case0)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 1536, 4, 4,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_BF16,          // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 2000);   // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode0_large_headdim_case1)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 1088, 4, 4,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT16,       // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 2001);   // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode0_large_headdim_case2)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 1088, 4, 4,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT,         // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 2002);   // mask dtype, status, tilingKey
}

TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode1_large_headdim_case0)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 1088, 1, 4,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_BF16,          // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 2000);   // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode1_large_headdim_case1)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 1088, 1, 4,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT16,       // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 2001);   // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode1_large_headdim_case2)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 1088, 1, 4,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT,         // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 2002);   // mask dtype, status, tilingKey
}

TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode2_large_headdim_case0)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 1088, 4, 1,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_BF16,          // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 2000);   // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode2_large_headdim_case1)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 1088, 4, 1,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT16,       // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 2001);   // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode2_large_headdim_case2)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 1088, 4, 1,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT,         // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 2002);   // mask dtype, status, tilingKey
}

TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode3_large_headdim_case0)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 1088, 1, 1,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_BF16,          // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 2000);   // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode3_large_headdim_case1)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 1088, 1, 1,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT16,       // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 2001);   // mask dtype, status, tilingKey
}
TEST_F(Ts_ScaledMaskedSoftmaxGradV2, case_mask_mode3_large_headdim_case2)
{
    InitAndRunScaledMaskedSoftmaxGradV2Case(4, 4, 64, 1088, 1, 1,                  // B, N, S1, S2, maskB, maskN
                                            float(0.9), false, ge::DT_FLOAT,         // scale, fixTriuMask, yGrad dtype
                                            ge::DT_BOOL, ge::GRAPH_SUCCESS, 2002);   // mask dtype, status, tilingKey
}