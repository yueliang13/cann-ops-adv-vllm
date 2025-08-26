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
 * \file ts_gmmadd.cpp
 * \brief MoeFinalizeRoutingV2 测试用例.
 */
#include "ts_gmm_add.h"
class Ts_gmmadd_Ascend910B2_Case : public Ts_gmmadd_WithParam_Ascend910B2 {};
class Ts_gmmadd_Ascend310P3_Case : public Ts_gmmadd_WithParam_Ascend310P3 {};
TEST_P(Ts_gmmadd_Ascend910B2_Case, general_case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}
TEST_P(Ts_gmmadd_Ascend310P3_Case, general_case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}

const auto Tc_GmmAdd_General_Case = ::testing::Values(
    //e(E), c(C), h(H), numRows(NUM_ROWS), k(K), dx(dx_), dropPadMode(dropPadMode_)
    GmmAddCase("case_001", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            GmmAddCase::Param(128, 256, 1024, 2, ge::DT_FLOAT16)),
    GmmAddCase("case_002", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            GmmAddCase::Param(128, 256, 1024, 2, ge::DT_BF16)));


INSTANTIATE_TEST_SUITE_P(GmmAdd, Ts_gmmadd_Ascend910B2_Case, Tc_GmmAdd_General_Case);
INSTANTIATE_TEST_SUITE_P(GmmAdd, Ts_gmmadd_Ascend310P3_Case, Tc_GmmAdd_General_Case);


TEST_F(Ts_gmmadd_Ascend910B2, case_bf16)
{
    GmmAddCase cs;
    uint32_t M = 12;
    uint32_t N = 7;
    uint32_t K = 2;
    uint32_t GroupNum = 3;
    cs.mParam = GmmAddCase::Param(M, N, K, GroupNum, ge::DT_BF16);
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_gmmadd_Ascend910B2, case_fp16)
{
    GmmAddCase cs;
    uint32_t M = 12;
    uint32_t N = 7;
    uint32_t K = 2;
    uint32_t GroupNum = 3;
    cs.mParam = GmmAddCase::Param(M, N, K, GroupNum, ge::DT_FLOAT16);
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
