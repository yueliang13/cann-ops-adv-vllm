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
 * \file scaled_masked_softmax_v2_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;
namespace ops {
static constexpr size_t IDX_IN_X = 0;
static constexpr size_t IDX_IN_MASK = 1;
static constexpr size_t IDX_OUT_Y = 0;

static ge::graphStatus ScaledMaskedSoftmaxV2InferShape(gert::InferShapeContext* context) {
  OPS_LOG_D(context->GetNodeName(), "Begin to do ScaledMaskedSoftmaxV2InferShape");

  // get input shapes
  const gert::Shape* xShape = context->GetInputShape(IDX_IN_X);
  OPS_LOG_E_IF_NULL(context, xShape, return false);
  const gert::Shape* maskShape = context->GetInputShape(IDX_IN_MASK);
  OPS_LOG_E_IF_NULL(context, maskShape, return false);
  // get output shapes
  gert::Shape* yShape = context->GetOutputShape(IDX_OUT_Y);
  OPS_LOG_E_IF_NULL(context, yShape, return false);

  *yShape = *xShape;

  OPS_LOG_D(context->GetNodeName(), "End to do ScaledMaskedSoftmaxV2InferShape");
  return ge::GRAPH_SUCCESS;
}

static graphStatus ScaledMaskedSoftmaxV2InferDtype (gert::InferDataTypeContext* context) {
  OPS_LOG_D(context->GetNodeName(), "ScaledMaskedSoftmaxV2InferDtype enter");
  // Get input tout
  auto attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return false);
  auto inputDtype = context->GetInputDataType(IDX_IN_X);
  context->SetOutputDataType(IDX_OUT_Y, inputDtype);

  OPS_LOG_D(context->GetNodeName(), "ScaledMaskedSoftmaxV2InferDtype end");

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScaledMaskedSoftmaxV2)
  .InferShape(ScaledMaskedSoftmaxV2InferShape)
  .InferDataType(ScaledMaskedSoftmaxV2InferDtype);
}  // namespace ops
