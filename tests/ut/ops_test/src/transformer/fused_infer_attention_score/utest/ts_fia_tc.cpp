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
 * \file ts_fia_tc.cpp
 * \brief FusedInferAttentionScore用例.
 */

#include "ts_fia.h"

TEST_F(Ts_Fia_Ascend910B1, case_001)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Fia_Ascend910B1, case_value_antiquant_scale)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_key_antiquant_scale_1)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_value_antiquant_offset)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_key_antiquant_scale_2)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_key_antiquant_scale_3)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_compare_key_value_offset)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 4}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_compare_key_value_scale)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 4}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_compare_key_antiquant_mode)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 3;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_1)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd2) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd3) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd4) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd5) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 17, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd6) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 17, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd7) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 32;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 32, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd8) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 256;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 256, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 256, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 256, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd9) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd1) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd2) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd3) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd4) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 17, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd5) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 17, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd6) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 32;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 32, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd7) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 256;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 256, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 256, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 256, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_2)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.antiquant_mode = 1;
    cs.mParam.softmax_lse_flag = 1;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 2, 2}, "3", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 2, 2}, "3", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_3)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.antiquant_mode = 1;
    cs.mParam.softmax_lse_flag = 1;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 4, 2048}, "3", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 4, 2048}, "3", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_actual_share_prefix_bsh)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 4, 2048}, "3", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 4, 2048}, "3", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_actual_share_prefix_bnsd)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 1, 10}, "BNSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 1, 2048, 10}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 1, 2048, 10}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 1, 10}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 10, 1, 10}, "4", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 10, 1, 10}, "4", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 10, 0, 10}, "4", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 10, 0, 10}, "4", ge::DT_INT8, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {1, 10, 0, 10}, "4", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_actual_share_pre_fixlen)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.antiquant_mode = 1;
    cs.mParam.softmax_lse_flag = 1;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 4, 2048}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 4, 2048}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {1}, "1", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_fia_1)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd_kvSep) // for msd IFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd_kvSep_0) // for msd IFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd_kvSep_1) // for msd IFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd_kvSep_2) // for msd IFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 4, 16}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 4, 16}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd_kvSep_3) // for msd IFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 4, 16}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}