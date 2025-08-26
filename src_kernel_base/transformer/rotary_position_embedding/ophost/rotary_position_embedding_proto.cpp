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
 * \file rotary_position_embedding_proto.cpp
 * \brief
 */
 #include <graph/utils/type_utils.h>
 #include <register/op_impl_registry.h>
 #include "log/ops_log.h"

using namespace ge;
static constexpr size_t INPUT_X_INDEX = 0;
static constexpr size_t INPUT_COS_INDEX = 1;
static constexpr size_t INPUT_SIN_INDEX = 2;
static constexpr size_t OUTPUT_Y_INDEX = 0;

namespace ops {
static ge::graphStatus InferShapeForRotaryPositionEmbedding(gert::InferShapeContext* context) {
  OPS_LOG_D(context->GetNodeName(), "Begin to do InferShapeForRotaryPositionEmbedding.");
  const gert::Shape* xShape = context->GetInputShape(INPUT_X_INDEX);
  OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED)
  const gert::Shape* cosShape = context->GetInputShape(INPUT_COS_INDEX);
  OPS_LOG_E_IF_NULL(context, cosShape, return ge::GRAPH_FAILED)
  const gert::Shape* sinShape = context->GetInputShape(INPUT_SIN_INDEX);
  OPS_LOG_E_IF_NULL(context, sinShape, return ge::GRAPH_FAILED)
  gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
  OPS_LOG_E_IF_NULL(context, yShape, return ge::GRAPH_FAILED)

  *yShape = *xShape;

  OPS_LOG_D(context->GetNodeName(), "End to do InferShapeForRotaryPositionEmbedding.");
  return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForRotaryPositionEmbedding(gert::InferDataTypeContext *context) {
  context->SetOutputDataType(OUTPUT_Y_INDEX, context->GetInputDataType(INPUT_X_INDEX));
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotaryPositionEmbedding).InferShape(InferShapeForRotaryPositionEmbedding)
                                .InferDataType(InferDataTypeForRotaryPositionEmbedding);
}  // namespace ops
