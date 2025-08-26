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
 * \file ring_attention_update_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;

namespace ops {

static constexpr size_t INPUT_ATTN = 0;
static constexpr size_t INPUT_SOFTMAX_MAX = 1;
static constexpr size_t INPUT_SOFTMAX_SUM = 2;
static constexpr size_t OUTPUT_ATTN = 0;
static constexpr size_t OUTPUT_SOFTMAX_MAX = 1;
static constexpr size_t OUTPUT_SOFTMAX_SUM = 2;

static graphStatus InferShape4RingAttentionUpdate(gert::InferShapeContext* context) {
  OPS_LOG_D(context->GetNodeName(), "Begin to do InferShape4RingAttentionUpdate");
  // get input shape
  const gert::Shape* inputAttnShape = context->GetInputShape(INPUT_ATTN);
  OPS_LOG_E_IF_NULL(context, inputAttnShape, return false);
  const gert::Shape* inputSoftmaxMax = context->GetInputShape(INPUT_SOFTMAX_MAX);
  OPS_LOG_E_IF_NULL(context, inputSoftmaxMax, return false);
  const gert::Shape* inputSoftmaxSum = context->GetInputShape(INPUT_SOFTMAX_SUM);
  OPS_LOG_E_IF_NULL(context, inputSoftmaxSum, return false);
  //get output shape
  gert::Shape* outputAttnShape = context->GetOutputShape(OUTPUT_ATTN);
  OPS_LOG_E_IF_NULL(context, outputAttnShape, return false);
  gert::Shape* outputSoftmaxMax = context->GetOutputShape(OUTPUT_SOFTMAX_MAX);
  OPS_LOG_E_IF_NULL(context, outputSoftmaxMax, return false);
  gert::Shape* outputSoftmaxSum = context->GetOutputShape(OUTPUT_SOFTMAX_SUM);
  OPS_LOG_E_IF_NULL(context, outputSoftmaxSum, return false);
  // infer shape
  *outputAttnShape = *inputAttnShape;
  *outputSoftmaxMax = *inputSoftmaxMax;
  *outputSoftmaxSum = *inputSoftmaxSum;

  OPS_LOG_D(context->GetNodeName(), "End to do InferShape4RingAttentionUpdate");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4RingAttentionUpdate(gert::InferDataTypeContext* context) {
  OPS_LOG_D(context->GetNodeName(), "Begin to do InferDataType4RingAttentionUpdate");
  // infer shape
  context->SetOutputDataType(OUTPUT_ATTN, context->GetInputDataType(INPUT_ATTN));
  context->SetOutputDataType(OUTPUT_SOFTMAX_MAX, context->GetInputDataType(INPUT_SOFTMAX_MAX));
  context->SetOutputDataType(OUTPUT_SOFTMAX_SUM, context->GetInputDataType(INPUT_SOFTMAX_SUM));

  OPS_LOG_D(context->GetNodeName(), "End to do InferDataType4RingAttentionUpdate");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RingAttentionUpdate)
    .InferShape(InferShape4RingAttentionUpdate)
    .InferDataType(InferDataType4RingAttentionUpdate);
}  // namespace ops
