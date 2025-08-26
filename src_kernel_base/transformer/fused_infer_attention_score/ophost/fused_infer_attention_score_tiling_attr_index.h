/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling_attr_index.h
 * \brief
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_ATTR_INDEX_H
#define FUSED_INFER_ATTENTION_SCORE_TILING_ATTR_INDEX_H
#include "prompt_flash_attention_tiling.h"
#include "incre_flash_attention_tiling.h"
#include "register/tilingdata_base.h"

namespace optiling {
// Attributes Index
constexpr uint32_t ATTR_N_INDEX = 0;
constexpr uint32_t ATTR_SCALE_INDEX = 1;
constexpr uint32_t ATTR_PRE_TOKEN_INDEX = 2;
constexpr uint32_t ATTR_NEXT_TOKEN_INDEX = 3;
constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 4;
constexpr uint32_t ATTR_NUM_KV_HEADS_INDEX = 5;
constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 6;
constexpr uint32_t ATTR_INNER_PRECISE_INDEX = 7;
constexpr uint32_t ATTR_BLOCK_SIZE_INDEX = 8;
constexpr uint32_t ANTIQUANT_MODE_INDEX = 9;
constexpr uint32_t SOFTMAX_LSE_FLAG_INDEX = 10;
constexpr uint32_t KEY_ANTIQUANT_MODE_INDEX = 11;
constexpr uint32_t VALUE_ANTIQUANT_MODE_INDEX = 12;
} // namespace optiling

#endif // FUSED_INFER_ATTENTION_SCORE_TILING_ATTR_INDEX_H