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
 * \file scaled_masked_softmax_grad_v2_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_Y_GRAD_IDX = 0;
static constexpr size_t OUTPUT_X_GRAD_IDX = 0;

static ge::graphStatus ScaledMaskedSoftmaxGradV2InferShape(gert::InferShapeContext* context) {
  OPS_LOG_D(context->GetNodeName(), "Begin to do ScaledMaskedSoftmaxGradV2InferShape");
  // get input shapes
  const gert::Shape* yGradShape = context->GetInputShape(INPUT_Y_GRAD_IDX);
  OPS_LOG_E_IF_NULL(context, yGradShape, return ge::GRAPH_FAILED);
  // get output shapes
  gert::Shape* xGradShape = context->GetOutputShape(OUTPUT_X_GRAD_IDX);
  OPS_LOG_E_IF_NULL(context, xGradShape, return ge::GRAPH_FAILED);

  *xGradShape = *yGradShape;

  OPS_LOG_D(context->GetNodeName(), "End to do ScaledMaskedSoftmaxGradV2InferShape");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ScaledMaskedSoftmaxGradV2InferDtype (gert::InferDataTypeContext* context) {
  OPS_LOG_D(context->GetNodeName(), "ScaledMaskedSoftmaxGradV2InferDtype enter");
  // Get input tout
  auto inputDtype = context->GetInputDataType(INPUT_Y_GRAD_IDX);
  context->SetOutputDataType(OUTPUT_X_GRAD_IDX, inputDtype);

  OPS_LOG_D(context->GetNodeName(), "ScaledMaskedSoftmaxGradV2InferDtype end");

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScaledMaskedSoftmaxGradV2)
  .InferShape(ScaledMaskedSoftmaxGradV2InferShape)
  .InferDataType(ScaledMaskedSoftmaxGradV2InferDtype);
}  // namespace ops