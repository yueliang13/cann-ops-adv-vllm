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
 * \file ts_ring_attention_update_tc.cpp
 * \brief RingAttentionUpdate UTest
 */

#include "ts_ring_attention_update.h"

RingAttentionUpdateCase InitNormalCase(int64_t S, int64_t B, int64_t H, int64_t N, int64_t T, int64_t D, std::string Layout,
                                    ge::DataType attnDataType, ge::DataType softmaxDataType, ge::DataType seqLenDataType, 
                                    ge::graphStatus result, int64_t tilingKey)
{
    RingAttentionUpdateCase cs;

    cs.mParam = {S, B, H, N, T, D, Layout, attnDataType, softmaxDataType, seqLenDataType};
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

void InitAndRunNormalCase(int64_t S, int64_t B, int64_t H, int64_t N, int64_t T, int64_t D, std::string Layout,
                        ge::DataType attnDataType, ge::DataType softmaxDataType, ge::DataType seqLenDataType, 
                        ge::graphStatus result, int64_t tilingKey)
{
    RingAttentionUpdateCase cs = InitNormalCase(S, B, H, N, T, D, Layout, attnDataType, softmaxDataType, seqLenDataType,
                                            result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_RingAttentionUpdate, ring_attention_update_bfloat16)
{
    InitAndRunNormalCase(1, 1, 64, 64, 64, 64, "SBH", ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT64, ge::GRAPH_SUCCESS, 1);
}

TEST_F(Ts_RingAttentionUpdate, ring_attention_update_float16)
{
    InitAndRunNormalCase(1, 1, 64, 64, 64, 64, "SBH", ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT64, ge::GRAPH_SUCCESS, 0);
}

TEST_F(Ts_RingAttentionUpdate, ring_attention_update_float32)
{
    InitAndRunNormalCase(1, 1, 64, 64, 64, 64, "SBH", ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::GRAPH_SUCCESS, 2);
}

TEST_F(Ts_RingAttentionUpdate, ring_attention_update_TND_bfloat16)
{
    InitAndRunNormalCase(1, 1, 64, 64, 1, 64, "TND", ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT64, ge::GRAPH_SUCCESS, 11);
}

TEST_F(Ts_RingAttentionUpdate, ring_attention_update_TND_float16)
{
    InitAndRunNormalCase(1, 1, 64, 64, 1, 64, "TND", ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT64, ge::GRAPH_SUCCESS, 10);
}

TEST_F(Ts_RingAttentionUpdate, ring_attention_update_TND_float32)
{
    InitAndRunNormalCase(1, 1, 64, 64, 1, 64, "TND", ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::GRAPH_SUCCESS, 12);
}