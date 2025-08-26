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
 * \file ts_rope_tc.cpp
 * \brief RotaryPositionEmbedding用例.
 */

#include "ts_rope.h"

RopeCase InitRopeCase(int64_t pb, int64_t pn, int64_t ps, int64_t pd, int64_t pTriB, int64_t pTriN, int64_t pMode,
                      std::string pLayout, ge::DataType pDataType, ge::graphStatus result, int64_t tilingKey)
{
    RopeCase cs;
    cs.mParam = {pb, pn, ps, pd, pTriB, pTriN, pMode, pLayout, pDataType};
    if (result == ge::GRAPH_SUCCESS) {
        cs.mOpInfo.mExp.mTilingKey = tilingKey;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = true;
    }  else {
        cs.mOpInfo.mExp.mSuccess = false;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = false;
    }
    cs.Init();
    return cs;
}

void InitAndRunRopeCase(int64_t pb, int64_t pn, int64_t ps, int64_t pd, int64_t pTriB, int64_t pTriN, int64_t pMode,
                        std::string pLayout, ge::DataType pDataType, ge::graphStatus result, int64_t tilingKey)
{
    RopeCase cs = InitRopeCase(pb, pn, ps, pd, pTriB, pTriN, pMode, pLayout, pDataType, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_Rope, case_mode0_BNSD_case0)
{
    InitAndRunRopeCase(2, 2, 1024, 128, 1, 1, 0, "BNSD", ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 1012);
}

TEST_F(Ts_Rope, case_mode0_BSND_case1)
{
    InitAndRunRopeCase(2, 2, 512, 64, 1, 1, 0, "BSND", ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1021);
}

TEST_F(Ts_Rope, case_mode0_SBND_case2)
{
    InitAndRunRopeCase(2, 2, 1024, 64, 1, 1, 0, "SBND", ge::DT_BF16, ge::GRAPH_SUCCESS, 1033);
}

TEST_F(Ts_Rope, case_mode0_r_B1SD_case3)
{
    InitAndRunRopeCase(2, 2, 1024, 192, 2, 1, 0, "BNSD", ge::DT_BF16, ge::GRAPH_SUCCESS, 1063);
}

TEST_F(Ts_Rope, case_mode0_r_BNSD_case4)
{
    InitAndRunRopeCase(2, 2, 2048, 192, 2, 2, 0, "BNSD", ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1041);
}

TEST_F(Ts_Rope, case_mode0_BNSD_D_unaligned_case5)
{
    InitAndRunRopeCase(2, 2, 1024, 120, 1, 1, 0, "BNSD", ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 1012);
}

TEST_F(Ts_Rope, case_mode0_BSND_D_unaligned_case6)
{
    InitAndRunRopeCase(2, 1, 1024, 80, 1, 1, 0, "BSND", ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1021);
}

TEST_F(Ts_Rope, case_mode0_SBND_D_unaligned_case7)
{
    InitAndRunRopeCase(2, 2, 1024, 90, 1, 1, 0, "SBND", ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1031);
}

TEST_F(Ts_Rope, case_mode1_BNSD_case0)
{
    InitAndRunRopeCase(2, 3, 1024, 128, 1, 1, 1, "BNSD", ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 2000);
}

TEST_F(Ts_Rope, case_mode1_BSND_case1)
{
    InitAndRunRopeCase(1, 1, 2048, 64, 1, 1, 1, "BSND", ge::DT_FLOAT, ge::GRAPH_SUCCESS, 2020);
}

TEST_F(Ts_Rope, case_mode1_SBND_case2)
{
    InitAndRunRopeCase(4, 1, 1024, 64, 1, 1, 1, "SBND", ge::DT_BF16, ge::GRAPH_SUCCESS, 2010);
}

TEST_F(Ts_Rope, case_mode1_BNSD_D_unaligned_case3)
{
    InitAndRunRopeCase(2, 1, 1024, 120, 1, 1, 1, "BNSD", ge::DT_FLOAT, ge::GRAPH_SUCCESS, 2020);
}

TEST_F(Ts_Rope, case_mode1_BSND_D_unaligned_case4)
{
    InitAndRunRopeCase(2, 2, 1024, 120, 1, 1, 1, "BSND", ge::DT_BF16, ge::GRAPH_SUCCESS, 2011);
}

TEST_F(Ts_Rope, case_mode1_SBND_D_unaligned_case5)
{
    InitAndRunRopeCase(2, 2, 2048, 80, 1, 1, 1, "SBND", ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 2000);
}

TEST_F(Ts_Rope, case_mode1_BNSD_case6)
{
    InitAndRunRopeCase(1, 1, 2048, 48, 1, 1, 1, "BNSD", ge::DT_FLOAT, ge::GRAPH_SUCCESS, 2020);
}

TEST_F(Ts_Rope, case_mode1_BSND_case7)
{
    InitAndRunRopeCase(8, 12, 128, 128, 1, 1, 1, "BSND", ge::DT_BF16, ge::GRAPH_SUCCESS, 2110);
}

TEST_F(Ts_Rope, case_mode1_SBND_case8)
{
    InitAndRunRopeCase(4, 4, 256, 64, 1, 1, 1, "SBND", ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 2000);
}