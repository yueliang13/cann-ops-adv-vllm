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
 * \file rotary_position_embedding_grad_proto.cpp
 * \brief
 */
 #include <graph/utils/type_utils.h>
 #include <register/op_impl_registry.h>
 #include "log/ops_log.h"

using namespace ge;
static constexpr size_t INPUT_GRAD_INDEX = 0;
static constexpr size_t INPUT_COS_INDEX = 1;
static constexpr size_t INPUT_SIN_INDEX = 2;
static constexpr size_t OUTPUT_DX_INDEX = 0;
static constexpr size_t OUTPUT_DCOS_INDEX = 1;
static constexpr size_t OUTPUT_DSIN_INDEX = 2;

namespace ops {
static ge::graphStatus InferShapeForRotaryPositionEmbeddingGrad(gert::InferShapeContext* context) {
  OPS_LOG_D(context, "Begin to do InferShapeForRotaryPositionEmbeddingGrad.");
  const gert::Shape* gradShape = context->GetInputShape(INPUT_GRAD_INDEX);
  OPS_LOG_E_IF_NULL(context, gradShape, return ge::GRAPH_FAILED)
  const gert::Shape* cosShape = context->GetInputShape(INPUT_COS_INDEX);
  OPS_LOG_E_IF_NULL(context, cosShape, return ge::GRAPH_FAILED)
  const gert::Shape* sinShape = context->GetInputShape(INPUT_SIN_INDEX);
  OPS_LOG_E_IF_NULL(context, sinShape, return ge::GRAPH_FAILED)
  gert::Shape* dxShape = context->GetOutputShape(OUTPUT_DX_INDEX);
  OPS_LOG_E_IF_NULL(context, dxShape, return ge::GRAPH_FAILED)
  gert::Shape* dcosShape = context->GetOutputShape(OUTPUT_DCOS_INDEX);
  OPS_LOG_E_IF_NULL(context, dcosShape, return ge::GRAPH_FAILED)
  gert::Shape* dsinShape = context->GetOutputShape(OUTPUT_DSIN_INDEX);
  OPS_LOG_E_IF_NULL(context, dsinShape, return ge::GRAPH_FAILED)
  
  *dxShape = *gradShape;
  *dcosShape = *cosShape;
  *dsinShape = *sinShape;

  OPS_LOG_D(context, "End to do InferShapeForRotaryPositionEmbeddingGrad.");
  return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForRotaryPositionEmbeddingGrad(gert::InferDataTypeContext *context) {
  context->SetOutputDataType(OUTPUT_DX_INDEX, context->GetInputDataType(INPUT_GRAD_INDEX));
  context->SetOutputDataType(OUTPUT_DCOS_INDEX, context->GetInputDataType(INPUT_COS_INDEX));
  context->SetOutputDataType(OUTPUT_DSIN_INDEX, context->GetInputDataType(INPUT_SIN_INDEX));
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotaryPositionEmbeddingGrad).InferShape(InferShapeForRotaryPositionEmbeddingGrad)
                                .InferDataType(InferDataTypeForRotaryPositionEmbeddingGrad);
}  // namespace ops