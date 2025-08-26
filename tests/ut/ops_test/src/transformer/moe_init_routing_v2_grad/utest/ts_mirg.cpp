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
 * \file ts_mirg.cpp
 * \brief MoeInitRoutingV2Grad UTest 相关基类定义.
 */
#include "ts_mirg.h"
class Ts_mirg_Ascend910B2_Case : public Ts_mirg_WithParam_Ascend910B2 {};
class Ts_mirg_Ascend310P3_Case : public Ts_mirg_WithParam_Ascend310P3 {};
TEST_P(Ts_mirg_Ascend910B2_Case, general_case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}
TEST_P(Ts_mirg_Ascend310P3_Case, general_case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}

const auto Tc_Mirg_General_Case = ::testing::Values(
    MoeInitRoutingV2GradCase("case_001", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            MoeInitRoutingV2GradCase::Param(8, 2, 5120, 0, 0, 0, 0, ge::DT_FLOAT16)),
    MoeInitRoutingV2GradCase("case_002", true, "dbginfo",
                OpInfo(ControlInfo(true, false),
                       ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
                MoeInitRoutingV2GradCase::Param(8, 2, 5120, 0, 0, 0, 0, ge::DT_FLOAT16))
        );

INSTANTIATE_TEST_SUITE_P(Mirg, Ts_mirg_Ascend910B2_Case, Tc_Mirg_General_Case);
INSTANTIATE_TEST_SUITE_P(Mirg, Ts_mirg_Ascend310P3_Case, Tc_Mirg_General_Case);

TEST_F(Ts_mirg_Ascend910B2, case_dropless_fp32)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;
    int64_t num_rows = 8;
    int64_t k = 2;
    int64_t hidden_size = 5120;
    int64_t e = 0;
    int64_t c = 0;
    int64_t drop_pad_mode = 0;
    int64_t active_num = 0;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT);
    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_dropless_fp16)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;
    int64_t num_rows = 4096;
    int64_t k = 40;
    int64_t hidden_size = 8;
    int64_t e = 0;
    int64_t c = 0;
    int64_t drop_pad_mode = 0;
    int64_t active_num = 0;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT16);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_dropless_bf16)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;
    int64_t num_rows = 4096;
    int64_t k = 40;
    int64_t hidden_size = 8;
    int64_t e = 0;
    int64_t c = 0;
    int64_t drop_pad_mode = 0;
    int64_t active_num = 0;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_BF16);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_act_fp16)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 10;
    int64_t k = 64;
    int64_t hidden_size = 512;
    int64_t e = 0;
    int64_t c = 0;
    int64_t drop_pad_mode = 0;
    int64_t active_num = 40;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT16);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_act_bf16)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 10;
    int64_t k = 64;
    int64_t hidden_size = 512;
    int64_t e = 0;
    int64_t c = 0;
    int64_t drop_pad_mode = 0;
    int64_t active_num = 40;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_BF16);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_act_fp32)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 10;
    int64_t k = 64;
    int64_t hidden_size = 512;
    int64_t e = 0;
    int64_t c = 0;
    int64_t drop_pad_mode = 0;
    int64_t active_num = 40;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_drop_pad_bf16)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 80;
    int64_t k = 8;
    int64_t hidden_size = 5120;
    int64_t e = 10;
    int64_t c = 8;
    int64_t drop_pad_mode = 1;
    int64_t active_num = 40;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_BF16);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_drop_pad_fp16)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 80;
    int64_t k = 8;
    int64_t hidden_size = 5120;
    int64_t e = 10;
    int64_t c = 8;
    int64_t drop_pad_mode = 1;
    int64_t active_num = 40;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT16);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_drop_pad_fp32)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 80;
    int64_t k = 8;
    int64_t hidden_size = 5120;
    int64_t e = 10;
    int64_t c = 8;
    int64_t drop_pad_mode = 1;
    int64_t active_num = 40;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_drop_pad_noact_bf16)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 80;
    int64_t k = 8;
    int64_t hidden_size = 5120;
    int64_t e = 10;
    int64_t c = 8;
    int64_t drop_pad_mode = 1;
    int64_t active_num = 0;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_BF16);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_drop_pad_noact_fp16)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 80;
    int64_t k = 8;
    int64_t hidden_size = 5120;
    int64_t e = 10;
    int64_t c = 8;
    int64_t drop_pad_mode = 1;
    int64_t active_num = 0;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT16);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_drop_pad_noact_fp32)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 80;
    int64_t k = 8;
    int64_t hidden_size = 5120;
    int64_t e = 10;
    int64_t c = 8;
    int64_t drop_pad_mode = 1;
    int64_t active_num = 0;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mExp.mSuccess = true;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_fail1_fp32)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 80;
    int64_t k = 8;
    int64_t hidden_size = 5120;
    int64_t e = 10;
    int64_t c = 8;
    int64_t drop_pad_mode = 2;
    int64_t active_num = 0;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mExp.mSuccess = false;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_FALSE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_fail2_fp32)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 80;
    int64_t k = 8;
    int64_t hidden_size = 5120;
    int64_t e = 10;
    int64_t c = 8;
    int64_t drop_pad_mode = 1;
    int64_t active_num = -10;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mExp.mSuccess = false;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_FALSE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_fail3_fp32)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 80;
    int64_t k = -8;
    int64_t hidden_size = 5120;
    int64_t e = 10;
    int64_t c = 8;
    int64_t drop_pad_mode = 1;
    int64_t active_num = 0;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mExp.mSuccess = false;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_FALSE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_fail4_fp32)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 80;
    int64_t k = 8;
    int64_t hidden_size = 5120;
    int64_t e = 10;
    int64_t c = 8;
    int64_t drop_pad_mode = 0;
    int64_t active_num = 0;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mExp.mSuccess = false;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    cs.gradExpandedX = ops::adv::tests::utils::Tensor("gradExpandX", {1,2,3,4,5}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_fail5_fp32)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 80;
    int64_t k = 8;
    int64_t hidden_size = 5120;
    int64_t e = 10;
    int64_t c = 8;
    int64_t drop_pad_mode = 1;
    int64_t active_num = 0;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mExp.mSuccess = false;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    cs.gradExpandedX = ops::adv::tests::utils::Tensor("gradExpandX", {2,3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}

TEST_F(Ts_mirg_Ascend910B2, case_fail6_fp32)
{
    MoeInitRoutingV2GradCase cs;
    uint32_t blockDim = 48;

    int64_t num_rows = 80;
    int64_t k = 8;
    int64_t hidden_size = 5120;
    int64_t e = 10;
    int64_t c = 8;
    int64_t drop_pad_mode = 1;
    int64_t active_num = 0;
    cs.mParam = MoeInitRoutingV2GradCase::Param(num_rows, k, hidden_size, e, c, drop_pad_mode, active_num, ge::DT_FLOAT);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mExp.mSuccess = false;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    cs.gradExpandedX = ops::adv::tests::utils::Tensor("gradExpandX", {2,3,3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    ASSERT_FALSE(cs.Run());
}