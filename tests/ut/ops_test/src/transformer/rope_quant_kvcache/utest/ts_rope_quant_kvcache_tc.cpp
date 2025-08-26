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
 * \file ts_rope_quant_kvcache_tc.cpp
 * \brief RotaryPositionEmbedding用例.
 */

#include "ts_rope_quant_kvcache.h"

RopeQuantKvcacheCase InitRopeQuantKvcacheCase(ge::graphStatus result, int64_t tilingKey)
{
    RopeQuantKvcacheCase cs;
    cs.mParam = {};
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

void InitAndRunRopeQuantKvcacheCase(ge::graphStatus result, int64_t tilingKey)
{
    RopeQuantKvcacheCase cs = InitRopeQuantKvcacheCase(result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_Rope_Quant_Kvcache, case0)
{
    InitAndRunRopeQuantKvcacheCase(ge::GRAPH_SUCCESS, 0);
}

