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
 * \file fused_infer_attention_score_tiling_output_index.h
 * \brief
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_OUTPUT_INDEX_H
#define FUSED_INFER_ATTENTION_SCORE_TILING_OUTPUT_INDEX_H
#include "prompt_flash_attention_tiling.h"
#include "incre_flash_attention_tiling.h"
#include "register/tilingdata_base.h"

namespace optiling {
// Output Index
constexpr uint32_t ATTENTION_OUT_INDEX = 0;
constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

constexpr uint32_t FROM_FUSED_FLAG = 71;
constexpr uint32_t FROM_TILING_SINK = 27;

constexpr int64_t D_ALIGN_16 = 16;
constexpr int64_t D_ALIGN_32 = 32;
} // namespace optiling

#endif // FUSED_INFER_ATTENTION_SCORE_TILING_OUTPUT_INDEX_H