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
 * \file apply_rotary_pos_emb_tiling.h
 * \brief
 */
#ifndef AIR_RUNTIME_V2_OP_IMPL_APPLY_ROTARY_POS_EMB_TILING_H
#define AIR_RUNTIME_V2_OP_IMPL_APPLY_ROTARY_POS_EMB_TILING_H
#ifdef ASCENDC_OP_TEST
#define AROPE_EXTERN_C extern "C"
#else
#define AROPE_EXTERN_C
#endif

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ApplyRotaryPosEmbTilingData)
TILING_DATA_FIELD_DEF(int64_t, useCoreNum);
TILING_DATA_FIELD_DEF(int64_t, lastDim);
TILING_DATA_FIELD_DEF(int64_t, halfNum);
TILING_DATA_FIELD_DEF(int64_t, preCBatchB);
TILING_DATA_FIELD_DEF(int64_t, preCBatchL);
TILING_DATA_FIELD_DEF(int64_t, lastCBatchL);
TILING_DATA_FIELD_DEF(int64_t, comBatchBB);
TILING_DATA_FIELD_DEF(int64_t, comBatchBBL);
TILING_DATA_FIELD_DEF(int64_t, comBatchBLL);
TILING_DATA_FIELD_DEF(int64_t, comBatchLLL);
TILING_DATA_FIELD_DEF(int64_t, qPart1Ub);
TILING_DATA_FIELD_DEF(int64_t, q2q1Part1Ub);
TILING_DATA_FIELD_DEF(int64_t, cosPart1Ub);
TILING_DATA_FIELD_DEF(int64_t, sin1UbSize);
TILING_DATA_FIELD_DEF(int64_t, preCLTimes);
TILING_DATA_FIELD_DEF(int64_t, lastCLTimes);
TILING_DATA_FIELD_DEF(int64_t, preCBBTimes);
TILING_DATA_FIELD_DEF(int64_t, preCBLTimes);
TILING_DATA_FIELD_DEF(int64_t, preCLLTimes);
TILING_DATA_FIELD_DEF(int64_t, qCoreOffset);
TILING_DATA_FIELD_DEF(int64_t, kCoreOffset);
TILING_DATA_FIELD_DEF(int64_t, cosCoreOffset);
TILING_DATA_FIELD_DEF(int64_t, qcdNum);
TILING_DATA_FIELD_DEF(int64_t, kcdNum);
TILING_DATA_FIELD_DEF(int64_t, coscdNum);
TILING_DATA_FIELD_DEF(int64_t, qkcNum);
TILING_DATA_FIELD_DEF(int64_t, mulNum);
TILING_DATA_FIELD_DEF(int64_t, qcdHalfNum);
TILING_DATA_FIELD_DEF(int64_t, dstRepSBr);
TILING_DATA_FIELD_DEF(int64_t, blockLenQ);
TILING_DATA_FIELD_DEF(int64_t, srcStrideK);
TILING_DATA_FIELD_DEF(int64_t, blockLenq2q1);
TILING_DATA_FIELD_DEF(int64_t, mask);
END_TILING_DATA_DEF;
    
REGISTER_TILING_DATA_CLASS(ApplyRotaryPosEmb, ApplyRotaryPosEmbTilingData)

struct ApplyRotaryPosEmbCompileInfo {
    int64_t totalCoreNum = 0;
    int64_t totalUbSize = 0;
    int64_t sysWorkspaceSize = 0;
};

enum class ApplyRotaryPosEmbTilingKey : int64_t {
    TILINGKEY_SMALL = 1,
    TILINGKEY_2 = 2,
    TILINGKEY_AB = 3,
    TILINGKEY_AB_CAST = 4,
};

}  // namespace optiling
#endif  // AIR_RUNTIME_V2_OP_IMPL_APPLY_ROTARY_POS_EMB_TILING_H
