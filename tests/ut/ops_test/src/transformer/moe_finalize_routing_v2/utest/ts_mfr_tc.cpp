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
 * \file ts_mfr.cpp
 * \brief MoeFinalizeRoutingV2 测试用例.
 */
#include "ts_mfr.h"
class Ts_mfr_Ascend910B2_Case : public Ts_mfr_WithParam_Ascend910B2 {};
class Ts_mfr_Ascend310P3_Case : public Ts_mfr_WithParam_Ascend310P3 {};
TEST_P(Ts_mfr_Ascend910B2_Case, general_case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}
TEST_P(Ts_mfr_Ascend310P3_Case, general_case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}

const auto Tc_Mfr_General_Case = ::testing::Values(
    //e(E), c(C), h(H), numRows(NUM_ROWS), k(K), dx(dx_), dropPadMode(dropPadMode_)
    MfrCase("case_001", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            MfrCase::Param(340, 1, 2560, 1024, 258, ge::DT_FLOAT, 0)),
    MfrCase("case_002", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            MfrCase::Param(340, 1, 2560, 1024, 258, ge::DT_FLOAT16, 0)),
    MfrCase("case_003", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            MfrCase::Param(340, 1, 2560, 1024, 258, ge::DT_BF16, 0)));


INSTANTIATE_TEST_SUITE_P(Mfr, Ts_mfr_Ascend910B2_Case, Tc_Mfr_General_Case);
INSTANTIATE_TEST_SUITE_P(Mfr, Ts_mfr_Ascend310P3_Case, Tc_Mfr_General_Case);


TEST_F(Ts_mfr_Ascend910B2, case_bf16)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 0);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 0);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 0);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_bf16_1)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 1, true, false, true);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_1)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 1, true, false, true);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16_1)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 1, true, false, true);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_bf16_2)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 2, true, false, false);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_2)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 2, true, false, false);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16_2)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 2, true, false, false);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_bf16_3)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 3, true, true, false);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_3)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 3, true, true, false);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16_3)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 2;
    uint32_t E = 3;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 3, true, true, false);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_bf16_4)
{
    MfrCase cs;
    uint32_t BS = 1024;
    uint32_t H = 30;
    uint32_t K = 3;
    uint32_t E = 10;
    uint32_t blockDim = 47;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 3, true, true, true);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_4)
{
    MfrCase cs;
    uint32_t BS = 1024;
    uint32_t H = 30;
    uint32_t K = 3;
    uint32_t E = 10;
    uint32_t blockDim = 47;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 3, true, true, true);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16_4)
{
    MfrCase cs;
    uint32_t BS = 1024;
    uint32_t H = 30;
    uint32_t K = 3;
    uint32_t E = 10;
    uint32_t blockDim = 47;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 3, true, true, true);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_bf16_5)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 3, true, true, false);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_5)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 3, true, true, false);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16_5)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 3, true, true, false);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_bf16_6)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 3, true, true, true);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_6)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 3, true, true, true);

    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16_6)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 7;
    uint32_t K = 257;
    uint32_t E = 300;
    uint32_t blockDim = 12;
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 3, true, true, true);
    cs.mOpInfo.mExp.mTilingBlockDim = blockDim;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}


TEST_F(Ts_mfr_Ascend910B2, case_bf16_7)
{
    MfrCase cs;
    uint32_t BS = 933;
    uint32_t H = 30;
    uint32_t K = 4;
    uint32_t E = 471;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 3, true, true, false);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_7)
{
    MfrCase cs;
    uint32_t BS = 933;
    uint32_t H = 30;
    uint32_t K = 4;
    uint32_t E = 471;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 3, true, true, false);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16_7)
{
    MfrCase cs;
    uint32_t BS = 933;
    uint32_t H = 30;
    uint32_t K = 4;
    uint32_t E = 471;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 3, true, true, false);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_bf16_8)
{
    MfrCase cs;
    uint32_t BS = 933;
    uint32_t H = 30;
    uint32_t K = 4;
    uint32_t E = 471;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_8)
{
    MfrCase cs;
    uint32_t BS = 933;
    uint32_t H = 30;
    uint32_t K = 4;
    uint32_t E = 471;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16_8)
{
    MfrCase cs;
    uint32_t BS = 933;
    uint32_t H = 30;
    uint32_t K = 4;
    uint32_t E = 471;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_bf16_9)
{
    MfrCase cs;
    uint32_t BS = 10;
    uint32_t H = 30;
    uint32_t K = 2;
    uint32_t E = 10;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_9)
{
    MfrCase cs;
    uint32_t BS = 10;
    uint32_t H = 30;
    uint32_t K = 2;
    uint32_t E = 10;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16_9)
{
    MfrCase cs;
    uint32_t BS = 10;
    uint32_t H = 30;
    uint32_t K = 2;
    uint32_t E = 10;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_bf16_10)
{
    MfrCase cs;
    uint32_t BS = 10;
    uint32_t H = 30;
    uint32_t K = 4;
    uint32_t E = 10;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_10)
{
    MfrCase cs;
    uint32_t BS = 10;
    uint32_t H = 30;
    uint32_t K = 4;
    uint32_t E = 10;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16_10)
{
    MfrCase cs;
    uint32_t BS = 10;
    uint32_t H = 30;
    uint32_t K = 4;
    uint32_t E = 10;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_bf16_11)
{
    MfrCase cs;
    uint32_t BS = 105;
    uint32_t H = 30;
    uint32_t K = 6;
    uint32_t E = 592;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_11)
{
    MfrCase cs;
    uint32_t BS = 105;
    uint32_t H = 30;
    uint32_t K = 6;
    uint32_t E = 592;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16_11)
{
    MfrCase cs;
    uint32_t BS = 105;
    uint32_t H = 30;
    uint32_t K = 6;
    uint32_t E = 592;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_bf16_12)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 20;
    uint32_t K = 4;
    uint32_t E = 16;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_12)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 20;
    uint32_t K = 4;
    uint32_t E = 16;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16_12)
{
    MfrCase cs;
    uint32_t BS = 12;
    uint32_t H = 20;
    uint32_t K = 4;
    uint32_t E = 16;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_bf16_13)
{
    MfrCase cs;
    uint32_t BS = 340;
    uint32_t H = 30;
    uint32_t K = 5;
    uint32_t E = 702;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_13)
{
    MfrCase cs;
    uint32_t BS = 340;
    uint32_t H = 30;
    uint32_t K = 5;
    uint32_t E = 702;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_fp16_13)
{
    MfrCase cs;
    uint32_t BS = 340;
    uint32_t H = 30;
    uint32_t K = 5;
    uint32_t E = 702;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp16_14)
{
    MfrCase cs;
    uint32_t BS = 34;
    uint32_t H = 102400;
    uint32_t K = 2;
    uint32_t E = 51;
    
    cs.mParam = MfrCase::Param(E, 19, H, BS, K, ge::DT_FLOAT16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_bf16_14)
{
    MfrCase cs;
    uint32_t BS = 34;
    uint32_t H = 102400;
    uint32_t K = 2;
    uint32_t E = 51;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_14)
{
    MfrCase cs;
    uint32_t BS = 34;
    uint32_t H = 102400;
    uint32_t K = 2;
    uint32_t E = 51;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp16_15)
{
    MfrCase cs;
    uint32_t BS = 34;
    uint32_t H = 102400;
    uint32_t K = 4;
    uint32_t E = 51;
    
    cs.mParam = MfrCase::Param(E, 19, H, BS, K, ge::DT_FLOAT16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_mfr_Ascend910B2, case_bf16_15)
{
    MfrCase cs;
    uint32_t BS = 34;
    uint32_t H = 102400;
    uint32_t K = 4;
    uint32_t E = 51;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_BF16, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_mfr_Ascend910B2, case_fp32_15)
{
    MfrCase cs;
    uint32_t BS = 34;
    uint32_t H = 102400;
    uint32_t K = 4;
    uint32_t E = 51;
    
    cs.mParam = MfrCase::Param(E, 1, H, BS, K, ge::DT_FLOAT, 3, true, true, true);
    
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}