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
 * \file fused_infer_attention_score_tiling_input_index.h
 * \brief
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_INPUT_INDEX_H
#define FUSED_INFER_ATTENTION_SCORE_TILING_INPUT_INDEX_H
#include "prompt_flash_attention_tiling.h"
#include "incre_flash_attention_tiling.h"
#include "register/tilingdata_base.h"

namespace optiling {
// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t PSE_SHIFT_INDEX = 3;
constexpr uint32_t ATTEN_MASK_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX = 6;
constexpr uint32_t DEQUANT_SCALE1_INDEX = 7;
constexpr uint32_t QUANT_SCALE1_INDEX = 8;
constexpr uint32_t DEQUANT_SCALE2_INDEX = 9;
constexpr uint32_t QUANT_SCALE2_INDEX = 10;
constexpr uint32_t QUANT_OFFSET2_INDEX = 11;
constexpr uint32_t ANTIQUANT_SCALE_INDEX = 12;
constexpr uint32_t ANTIQUANT_OFFSET_INDEX = 13;
constexpr uint32_t BLOCK_TABLE_INDEX = 14;
constexpr uint32_t QUERY_PADDING_SIZE_INDEX = 15;
constexpr uint32_t KV_PADDING_SIZE_INDEX = 16;
constexpr uint32_t KEY_ANTIQUANT_SCALE_INDEX = 17;
constexpr uint32_t KEY_ANTIQUANT_OFFSET_INDEX = 18;
constexpr uint32_t VALUE_ANTIQUANT_SCALE_INDEX = 19;
constexpr uint32_t VALUE_ANTIQUANT_OFFSET_INDEX = 20;
constexpr uint32_t KEY_SHARED_PREFIX_INDEX = 21;
constexpr uint32_t VALUE_SHARED_PREFIX_INDEX = 22;
constexpr uint32_t ACTUAL_SHARED_PREFIX_LEN_INDEX = 23;
constexpr uint32_t QUERY_ROPE_INDEX = 24;
constexpr uint32_t KEY_ROPE_INDEX = 25;
constexpr uint32_t KEY_ROPE_ANTIQUANT_SCALE_INDEX = 26;
} // namespace optiling

#endif // FUSED_INFER_ATTENTION_SCORE_TILING_INPUT_INDEX_H
