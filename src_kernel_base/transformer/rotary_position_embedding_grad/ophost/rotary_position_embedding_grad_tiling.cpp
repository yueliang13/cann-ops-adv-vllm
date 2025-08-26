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
 * \file rotary_position_embedding_grad_tiling.cpp
 * \brief
 */
#include "rotary_position_embedding_grad_tiling.h"
#include "rope_half_grad_tiling.h"
#include "rope_interleaved_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/ops_log.h"

namespace optiling {
constexpr uint32_t MODE_ATTR_IDX = 0;
constexpr uint32_t MODE_ROTATE_HALF = 0;
constexpr uint32_t MODE_ROTATE_INTERLEAVED = 1;

ROPE_GRAD_EXTERN_C ge::graphStatus TilingRotaryPositionEmbeddingGrad(gert::TilingContext* context)
{
  auto attrs = context->GetAttrs();
  if (attrs == nullptr) {
    OPS_LOG_E(context, "attrs is null.");
    return ge::GRAPH_FAILED;
  }
  const char* inputMode = attrs->GetAttrPointer<char>(MODE_ATTR_IDX);
  OPS_LOG_E_IF_NULL(context, inputMode, return ge::GRAPH_FAILED);
  OPS_LOG_I(context, "[mode]: %s", inputMode);

  if (*inputMode == MODE_ROTATE_INTERLEAVED) {
    return Tiling4RopeInterleavedGrad(context);
  }

  return Tiling4RopeHalfGrad(context);
}

ROPE_GRAD_EXTERN_C ge::graphStatus TilingPrepareForRotaryPositionEmbeddingGrad(gert::TilingParseContext *context)
{
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RotaryPositionEmbeddingGrad)
  .Tiling(TilingRotaryPositionEmbeddingGrad)
  .TilingParse<RotaryPositionEmbeddingGradCompileInfo>(TilingPrepareForRotaryPositionEmbeddingGrad);
}  // namespace optiling
