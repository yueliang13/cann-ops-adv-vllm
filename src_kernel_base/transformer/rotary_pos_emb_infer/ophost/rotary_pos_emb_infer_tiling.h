/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotary_pos_emb_infer_tiling.h
 * \brief
 */
#ifndef ASCEND_OPS_ROTARY_POS_EMB_INFER_TILING_H
#define ASCEND_OPS_ROTARY_POS_EMB_INFER_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "register/op_def_registry.h"
#include "log/ops_log.h"
#include "error/ops_error.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RotaryPosEmbInferTilingData)
TILING_DATA_FIELD_DEF(uint32_t, hiddenSizeQ);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSizeK);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, headNumQ);
TILING_DATA_FIELD_DEF(uint32_t, headNumK);
TILING_DATA_FIELD_DEF(uint32_t, rotaryCoeff);
TILING_DATA_FIELD_DEF(uint32_t, ntokens);
TILING_DATA_FIELD_DEF(uint32_t, realCore);
TILING_DATA_FIELD_DEF(uint32_t, cosFormat);
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, maxUbSize);
TILING_DATA_FIELD_DEF(uint32_t, multiple);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RotaryPosEmbInfer, RotaryPosEmbInferTilingData)

struct Tiling4RotaryPosEmbInferCompileInfo {
    uint32_t coreNum;
    uint64_t ubSizePlatForm;
    uint32_t sysWorkspaceSize;
};

}
#endif  // ASCEND_OPS_ROTARY_POS_EMB_INFER_TILING_H