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
 * \file ring_attention_update_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_RING_ATTENTION_UPDATE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_RING_ATTENTION_UPDATE_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RingAttentionUpdateTilingData)
    TILING_DATA_FIELD_DEF(int64_t, batchSize);
    TILING_DATA_FIELD_DEF(int64_t, headNum);
    TILING_DATA_FIELD_DEF(int64_t, seqNum);
    TILING_DATA_FIELD_DEF(int64_t, headDim);
    TILING_DATA_FIELD_DEF(int64_t, softmaxTailSize);

    TILING_DATA_FIELD_DEF(int64_t, coreNum);

    TILING_DATA_FIELD_DEF(int64_t, coreNumGroup);
    TILING_DATA_FIELD_DEF(int64_t, bnNumGroup);
    TILING_DATA_FIELD_DEF(int64_t, seqNumCoreEach);
    TILING_DATA_FIELD_DEF(int64_t, seqNumCoreTail);
    TILING_DATA_FIELD_DEF(int64_t, seqNumLoopEach);
    TILING_DATA_FIELD_DEF(int64_t, headNumLoopEach);
    TILING_DATA_FIELD_DEF(int64_t, headDimLoopEach);

    TILING_DATA_FIELD_DEF(int64_t, batchSizeCoreEach);
    TILING_DATA_FIELD_DEF(int64_t, batchSizeCoreTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RingAttentionUpdate, RingAttentionUpdateTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_RING_ATTENTION_UPDATE_H_