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
 * \file ts_moe_finalize_routing.cpp
 * \brief moe_finalize_routing 用例.
 */

#include "ts_moe_finalize_routing.h"

MoeFinalizeRoutingCase InitNormalCase(int expert_num, int token_len, int top_k, int num_rows, std::string layout, ge::DataType dType, ge::graphStatus result, int64_t tilingKey)
{
  MoeFinalizeRoutingCase cs;
  cs.mParam = {expert_num, token_len, top_k, num_rows, layout, dType};
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

void InitAndRunNormalCase(int expert_num, int token_len, int top_k, int num_rows, std::string layout, ge::DataType dType, ge::graphStatus result, int64_t tilingKey)
{
  MoeFinalizeRoutingCase cs = InitNormalCase(expert_num, token_len, top_k, num_rows, layout, dType, result, tilingKey);
  if (result == ge::GRAPH_SUCCESS) {
    ASSERT_TRUE(cs.Run());
  } else {
    ASSERT_FALSE(cs.Run());
}
}

TEST_F(Ts_MoeFinalizeRouting, moe_finalize_routing_0)
{
InitAndRunNormalCase(16, 836, 258, 128, "BSH", ge::DT_FLOAT, ge::GRAPH_SUCCESS, 20000);
}
