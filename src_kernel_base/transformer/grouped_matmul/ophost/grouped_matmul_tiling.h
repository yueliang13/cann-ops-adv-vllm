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
 * \file grouped_matmul_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GMMBaseParams)
  TILING_DATA_FIELD_DEF(uint32_t, groupNum);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint32_t, activeType);
  TILING_DATA_FIELD_DEF(uint32_t, ubBaseK);
  TILING_DATA_FIELD_DEF(uint32_t, ubBaseN);
  TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
  TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
  TILING_DATA_FIELD_DEF(uint32_t, singleWeight);
  TILING_DATA_FIELD_DEF(uint32_t, singleX);
  TILING_DATA_FIELD_DEF(uint32_t, singleY);
  TILING_DATA_FIELD_DEF(int32_t, groupType);
  TILING_DATA_FIELD_DEF(uint32_t, singleN);  // If sequential write， the value should be zero!
  TILING_DATA_FIELD_DEF(uint32_t, quantParam);  // in quant case, PerToken: 1; in antiquant case, the value represents PerGroupSize
  TILING_DATA_FIELD_DEF(uint32_t, groupListType);
  TILING_DATA_FIELD_DEF(uint32_t, m);
  TILING_DATA_FIELD_DEF(uint32_t, hasBias);
  TILING_DATA_FIELD_DEF(uint64_t, workspaceSize);
  TILING_DATA_FIELD_DEF(uint64_t, totalInGroup);  // for A8W4 MSD
  TILING_DATA_FIELD_DEF(uint64_t, k);  // for A8W4 MSD
  TILING_DATA_FIELD_DEF(uint64_t, n);  // for A8W4 MSD
  TILING_DATA_FIELD_DEF(uint64_t, vBaseM);  // for A8W4 MSD
  TILING_DATA_FIELD_DEF(uint64_t, parallNum);  // for A8W4 MSD 
  TILING_DATA_FIELD_DEF(uint64_t, quantGroupNum);  // for A8W4 MSD
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMBaseParamsOp, GMMBaseParams)

BEGIN_TILING_DATA_DEF(GMMArray)
  TILING_DATA_FIELD_DEF_ARR(int32_t, 128, mList);  // 128 ：MAX_TENSOR_CONT
  TILING_DATA_FIELD_DEF_ARR(int32_t, 128, kList);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 128, nList);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMArrayOp, GMMArray)

BEGIN_TILING_DATA_DEF(GMMTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(GMMBaseParams, gmmBaseParams);
  TILING_DATA_FIELD_DEF_STRUCT(GMMArray, gmmArray);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedMatmul, GMMTilingData)
}

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_H