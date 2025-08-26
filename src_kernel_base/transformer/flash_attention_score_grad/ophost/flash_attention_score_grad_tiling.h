/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_grad_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "tiling/data_copy_transpose_tiling_def.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(EmptyTensorTilingData)
TILING_DATA_FIELD_DEF(uint32_t, formerDqNum);
TILING_DATA_FIELD_DEF(uint32_t, formerDkNum);
TILING_DATA_FIELD_DEF(uint32_t, formerDpseNum);
TILING_DATA_FIELD_DEF(uint32_t, res);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreDqNum);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreDqNum);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreDkNum);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreDkNum);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreDpseNum);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreDpseNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(EmptyTensorTilingDataOp, EmptyTensorTilingData)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradTilingData)
TILING_DATA_FIELD_DEF_STRUCT(EmptyTensorTilingData, emptyTensorTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_90, FlashAttentionScoreGradTilingData)
} // namespace optiling
