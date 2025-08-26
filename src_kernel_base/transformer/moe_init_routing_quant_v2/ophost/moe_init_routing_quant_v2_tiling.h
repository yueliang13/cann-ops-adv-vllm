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
 * \file moe_init_routing_quant_v2_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_QUANT_V2_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_QUANT_V2_H
#include <cmath>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "error/ops_error.h"
#include "log/ops_log.h"
#include "moe_init_routing_v2_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeInitRoutingQuantV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, expertCapacity);
TILING_DATA_FIELD_DEF(int64_t, expertNum);
TILING_DATA_FIELD_DEF(int64_t, dropPadMode);
TILING_DATA_FIELD_DEF(int64_t, expertTokensCountOrCumsumFlag);
TILING_DATA_FIELD_DEF(int64_t, expertTokensBeforeCapacityFlag);
TILING_DATA_FIELD_DEF(int64_t, smoothType);
TILING_DATA_FIELD_DEF_STRUCT(InnerMoeV2VBSComputeTilingData, vbsComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(InnerMoeV2VMSMiddleComputeTilingData, vmsMiddleComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(InnerMoeV2SortOutComputeTilingData, sortOutComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(InnerMoeV2GatherOutComputeTilingData, srcToDstComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(InnerMoeV2GatherOutComputeTilingData, srcToDstCapacityComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(InnerMoeV2GatherOutComputeTilingData, gatherOutComputeParamsOp);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeInitRoutingQuantV2, MoeInitRoutingQuantV2TilingData)
struct MoeInitRoutingQuantV2CompileInfo {};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_QUANT_V2_H