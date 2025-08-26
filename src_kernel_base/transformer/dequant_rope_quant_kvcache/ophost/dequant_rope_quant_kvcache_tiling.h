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
 * \file dequant_rope_quant_kvcache.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_DEQUANT_ROPE_QUANT_KVCACHE_H
#define OPS_BUILD_IN_OP_TILING_DEQUANT_ROPE_QUANT_KVCACHE_H
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "log/ops_log.h"

namespace optiling {
struct DequantRopeQuantKvcacheCompileInfo {};

BEGIN_TILING_DATA_DEF(DequantRopeQuantKvcacheTilingData)
TILING_DATA_FIELD_DEF(int64_t, qHeadNum);
TILING_DATA_FIELD_DEF(int64_t, kvHeadNum);
TILING_DATA_FIELD_DEF(int64_t, hiddenSize);
TILING_DATA_FIELD_DEF(int64_t, hiddenSizeFp32Align);
TILING_DATA_FIELD_DEF(int64_t, hiddenSizeFp16Align);
TILING_DATA_FIELD_DEF(int64_t, hiddenSizeInt8Align);
TILING_DATA_FIELD_DEF(int64_t, OnceUBMaxS);
TILING_DATA_FIELD_DEF(int64_t, cacheSeqlen);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, qHiddenSize);
TILING_DATA_FIELD_DEF(int64_t, kHiddenSize);
TILING_DATA_FIELD_DEF(int64_t, vHiddenSize);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, frontCoreNum);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, hasQuantOffset);
TILING_DATA_FIELD_DEF(int64_t, ifKVout);
TILING_DATA_FIELD_DEF(int64_t, isPA);
TILING_DATA_FIELD_DEF(int64_t, hasBias);
TILING_DATA_FIELD_DEF(int64_t, hasAS);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DequantRopeQuantKvcache, DequantRopeQuantKvcacheTilingData)
}
#endif  // OPS_BUILD_IN_OP_TILING_DEQUANT_ROPE_QUANT_KVCACHE_H