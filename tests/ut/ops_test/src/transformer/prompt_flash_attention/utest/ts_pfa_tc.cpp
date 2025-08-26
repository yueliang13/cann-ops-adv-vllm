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
 * \file ts_pfa_tc.cpp
 * \brief PromptFlashAttention用例.
 */

#include "ts_pfa.h"
class Ts_Pfa_Ascend910B2_Case : public Ts_Pfa_WithParam_Ascend910B2 {};
class Ts_Pfa_Ascend310P3_Case : public Ts_Pfa_WithParam_Ascend310P3 {};
TEST_P(Ts_Pfa_Ascend910B2_Case, general_case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}
TEST_P(Ts_Pfa_Ascend310P3_Case, general_case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}

const auto Tc_Pfa_General_Case =
    ::testing::Values(PfaCase("case_001", true, "",
                              OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey,
                                                                          ExpectInfo::kInvalidTilingBlockDim)),
                              PfaCase::Param(1, 4, 1024, 128, "BSH", 4, 4, 1.0f, 0, 1, 0, 524288, 0)));

INSTANTIATE_TEST_SUITE_P(Pfa, Ts_Pfa_Ascend910B2_Case, Tc_Pfa_General_Case);
INSTANTIATE_TEST_SUITE_P(Pfa, Ts_Pfa_Ascend310P3_Case, Tc_Pfa_General_Case);

TEST_F(Ts_Pfa_Ascend910B2, case_empty_query)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "BNSD", cs.mParam.qDataType, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Pfa_Ascend910B2, case_empty_query_bf16)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "BNSD", cs.mParam.qDataType, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Pfa_Ascend910B2, prompt_flash_attention_tiling_int8_in_fp16_out)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", cs.mParam.qDataType,
    ge::FORMAT_ND); ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, prompt_flash_attention_BNSD_attenMask_sparse_mode4_case0)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 8;
    cs.mParam.s = 2048;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 8;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", cs.mParam.qDataType,
    ge::FORMAT_ND); ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, prompt_flash_attention_BNSD_attenMask_sparse_mode4_case1)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 12;
    cs.mParam.s = 16384;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 12;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", cs.mParam.qDataType,
    ge::FORMAT_ND); ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, prompt_flash_attention_BNSD_attenMask_sparse_mode4_case2)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 8;
    cs.mParam.s = 8193;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 8;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", cs.mParam.qDataType,
    ge::FORMAT_ND); ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, prompt_flash_attention_tiling_bf16)
{
    PfaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 1024;
    cs.mParam.d = 5120;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.s, cs.mParam.n * cs.mParam.d}, "BSH", cs.mParam.qDataType,
    ge::FORMAT_ND); ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, prompt_flash_attention_tiling_3)
{
    PfaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 2;
    cs.mParam.s = 1024;
    cs.mParam.d = 5120;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.s, cs.mParam.n * cs.mParam.d}, "BSH", cs.mParam.qDataType,
    ge::FORMAT_ND); ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, prompt_flash_attention_tiling_int8_in_int8_out)
{
    PfaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", cs.mParam.qDataType,
    ge::FORMAT_ND); ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, prompt_flash_attention_tiling_4)
{
    PfaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 2;
    cs.mParam.s = 1000;
    cs.mParam.d = 5120;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.s, cs.mParam.n * cs.mParam.d}, "BSH", cs.mParam.qDataType,
    ge::FORMAT_ND); ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, prompt_flash_attention_tiling_5)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", cs.mParam.qDataType,
    ge::FORMAT_ND); ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, prompt_flash_attention_tiling_6)
{
    PfaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 40;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "NSD";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.n, cs.mParam.s, cs.mParam.d}, "NSD", cs.mParam.qDataType, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, prompt_flash_attention_tiling_7)
{
    PfaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 40;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "SH";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {256, 5120}, "SH", cs.mParam.qDataType, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Pfa_Ascend910B2, case_empty_key)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 100;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.key = Tensor("key", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "BNSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "BNSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_invalid_quant_1)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.qDataType = ge::DT_INT8;
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.quantType = QuantShapeType::ALL_1;
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_invalid_atten_mask_1)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.actualSeqLength = {1000};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.attenMask = Tensor("attenMask", {2, 40, 1, 1000}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_invalid_hd)
{
    PfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.n = 11;
    cs.mParam.s = 2014;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 11;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Pfa_Ascend910B2, case_atten_mask_2)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_BN_greater_than_core_number)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 49;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.numHeads = 49;
    cs.mParam.kvNumHeads = 1;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}
TEST_F(Ts_Pfa_Ascend910B2, case_quant_1)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.outDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.quantType = QuantShapeType::POST_1;
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}
TEST_F(Ts_Pfa_Ascend910B2, case_kvAntiQuant_1)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.actualSeqLength = {1000};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}
TEST_F(Ts_Pfa_Ascend310P3, case_kvAntiQuant_1)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.actualSeqLength = {1000};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}
TEST_F(Ts_Pfa_Ascend310P3, case_kvAntiQuant_1_01)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "NSD";
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.actualSeqLength = {1000};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "NSD", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "NSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "NSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut =
        Tensor("attentionOut", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "NSD", cs.mParam.outDataType, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}
TEST_F(Ts_Pfa_Ascend310P3, case_kvAntiQuant_1_02)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "SH";
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.actualSeqLength = {1000};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "SH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "SH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "SH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut =
        Tensor("attentionOut", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "SH", cs.mParam.outDataType, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_qunat_kvAntiQuant_1)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.quantType = QuantShapeType::POST_1;
    cs.mParam.actualSeqLength = {1000};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_kvAntiQuant_bf16)
{
    PfaCase cs;
    cs.mParam.b = 5;
    cs.mParam.n = 40;
    cs.mParam.s = 16000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.quantType = QuantShapeType::POST_1;
    cs.mParam.actualSeqLength = {1000,1000,1000,1000,1000};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_kvAntiQuant_bf16_01)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.quantType = QuantShapeType::POST_1;
    cs.mParam.actualSeqLength = {1000,1000,1000,1000,1000};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Pfa_Ascend910B2, case_kvAntiQuant_unflash_splitB_largeS)
{
    PfaCase cs;
    cs.mParam.b = 96;
    cs.mParam.n = 11;
    cs.mParam.s = 8192;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType =ge::DT_BF16;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    cs.mParam.numHeads = 11;
    cs.mParam.kvNumHeads=1;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_empty_kvPadding)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads=2;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.actualSeqLength = {1};
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Pfa_Ascend310P3, case_empty_kvPadding)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads=2;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.actualSeqLength = {1};
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Pfa_Ascend910B2, case_kvPadding)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.actualSeqLength = {1};
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Pfa_Ascend310P3, case_kvPadding)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.actualSeqLength = {1};
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Pfa_Ascend910B2, case_kvPadding_no_act_sqe_len)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Pfa_Ascend910B2, case_kvAntiquant_bf16_quant_scale2_type_3)
{
    PfaCase cs;
    cs.mParam.b = 5;
    cs.mParam.n = 40;
    cs.mParam.s = 16000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 40;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_INT8;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.quantType = QuantShapeType::POST_1;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}
TEST_F(Ts_Pfa_Ascend910B2, case_kvAntiquant_workspace_opt_unflashSplitB_largeS)
{
    PfaCase cs;
    cs.mParam.b = 96;
    cs.mParam.n = 11;
    cs.mParam.s = 8192;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 11;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_BF16;
    cs.mParam.actualSeqLength = {1};
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_kvAntiquant_workspace_opt_unflashSplitB)
{
    PfaCase cs;
    cs.mParam.b = 96;
    cs.mParam.n = 11;
    cs.mParam.s = 4096;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 11;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_BF16;
    cs.mParam.actualSeqLength = {1};
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_invalid_hn_bsh)
{
    PfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 11;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_attenmask_fp16)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 4;
    cs.mParam.s = 8192;
    cs.mParam.d = 256;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 4;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.pseShiftType = PseShiftShapeType::B_1_N_S;
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mOpInfo.mExp.mSuccess = false;
    cs.pseShift = Tensor("pseShift", {cs.mParam.b, cs.mParam.n, 1, cs.mParam.s}, "B_1_N_S", cs.mParam.qDataType, ge::FORMAT_ND);
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_pfa_1)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 4;
    cs.mParam.s = 8192;
    cs.mParam.d = 256;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 4;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.pseShiftType = PseShiftShapeType::B_1_N_S;
    cs.mParam.qDataType = ge::DT_BF16;
    cs.pseShift = Tensor("pseShift", {cs.mParam.b, cs.mParam.n, 1, cs.mParam.s}, "B_1_N_S", cs.mParam.qDataType, ge::FORMAT_ND);
    ASSERT_TRUE(cs.Init());
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend310P3, case_pfa_2)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.actualSeqLength = {1};
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Pfa_Ascend310P3, case_pfa_3)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "SH";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.actualSeqLength = {1};
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4096, 320}, "SH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4096, 32}, "SH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4096, 32}, "SH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4096, 320}, "SH", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend310P3, case_pfa_4)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "NSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.actualSeqLength = {1};
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {20, 4096, 16}, "NSD", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {2, 4096, 16}, "NSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {2, 4096, 16}, "NSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {20, 4096, 16}, "NSD", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_pfa_5)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.actualSeqLength = {1};
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Pfa_Ascend910B2, case_pfa_6)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "SH";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.actualSeqLength = {1};
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4096, 320}, "SH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4096, 32}, "SH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4096, 32}, "SH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4096, 320}, "SH", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_pfa_7)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "NSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.actualSeqLength = {1};
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {20, 4096, 16}, "NSD", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {2, 4096, 16}, "NSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {2, 4096, 16}, "NSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {20, 4096, 16}, "NSD", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend310P3, prompt_flash_attention_tiling_8)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "SH";
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.actualSeqLength = {1000};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "SH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "SH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "SH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut =
        Tensor("attentionOut", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "SH", cs.mParam.outDataType, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, prompt_flash_attention_tiling_9)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.actualSeqLength = {1000};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.attenMask = Tensor("attenMask", {2, 40, 1, 1000}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, prompt_flash_attention_tiling_10)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 32; // 32: D: [32, 128)对齐范围的临界值
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", cs.mParam.qDataType,
    ge::FORMAT_ND); ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_attenmask_fp16_wrong)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 4;
    cs.mParam.s = 8192;
    cs.mParam.d = 256;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 4;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.pseShiftType = PseShiftShapeType::B_1_N_S;
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mOpInfo.mExp.mSuccess = false;
    cs.pseShift = Tensor("pseShift", {cs.mParam.b, cs.mParam.n, 1, cs.mParam.s}, "B_1_N_S", ge::DT_FLOAT16,
    ge::FORMAT_ND);
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Pfa_Ascend910B2, case_invalid_bf16_post_quant)
{
    PfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1;
    cs.mParam.d = 127;
    cs.mParam.layout = "BSH";
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mParam.kvDataType = ge::DT_BF16;
    cs.mParam.outDataType = ge::DT_INT8;
    cs.mParam.quantType = QuantShapeType::ALL_1;
    cs.mParam.numHeads = 40;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}