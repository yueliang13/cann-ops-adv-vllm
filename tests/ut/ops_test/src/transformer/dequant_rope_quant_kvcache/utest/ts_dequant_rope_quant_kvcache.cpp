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
 * \file ts_dequant_rope_quant_kvcache.cpp
 * \brief DequantRopeQuantKvcache用例.
 */

#include "ts_dequant_rope_quant_kvcache.h"

DequantRopeQuantKvcacheCase InitDRQKCase(int64_t pB, int64_t pS, int64_t pNkv, int64_t pNq, int64_t pD, int64_t pC1, int64_t pC2, 
                                            bool pOutOptional, std::string pCacheOptional,ge::DataType xDtypeIn, 
                                            ge::DataType sinDtypeIn, ge::DataType biasDtypeIn, ge::graphStatus result, int64_t tilingKey)
{
DequantRopeQuantKvcacheCase cs;

cs.mParam = {pB, pS, pNkv, pNq, pD, pC1, pC2, pOutOptional, pCacheOptional, xDtypeIn, sinDtypeIn, biasDtypeIn};
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

void InitAndRunDRQKCase(int64_t pB, int64_t pS, int64_t pNkv, int64_t pNq, int64_t pD, int64_t pC1, int64_t pC2, 
                                            bool pOutOptional, std::string pCacheOptional,ge::DataType xDtypeIn, 
                                            ge::DataType sinDtypeIn, ge::DataType biasDtypeIn, ge::graphStatus result, int64_t tilingKey)
{
    DequantRopeQuantKvcacheCase cs = InitDRQKCase(pB, pS, pNkv, pNq, pD, pC1, pC2, pOutOptional, pCacheOptional, xDtypeIn, 
                                                  sinDtypeIn, biasDtypeIn, result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}


TEST_F(Ts_DequantRopeQuantKvcache, dequant_rope_quant_kvcache_1)
{
    InitAndRunDRQKCase(320, 1, 1, 8, 128, 320, 1280, true, "contigunous", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 1);
}

TEST_F(Ts_DequantRopeQuantKvcache, dequant_rope_quant_kvcache_0)
{
    InitAndRunDRQKCase(1, 1024, 1, 8, 128, 1, 1280, true, "contigunous", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 0);
}

TEST_F(Ts_DequantRopeQuantKvcache, dequant_rope_quant_kvcache_2)
{
    InitAndRunDRQKCase(1, 1024, 1, 8, 128, 1, 1280, true, "contigunous", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::GRAPH_SUCCESS, 2);
}

TEST_F(Ts_DequantRopeQuantKvcache, dequant_rope_quant_kvcache_3)
{
    InitAndRunDRQKCase(1, 1024, 1, 8, 128, 1, 1280, true, "contigunous", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::GRAPH_SUCCESS, 3);
}
