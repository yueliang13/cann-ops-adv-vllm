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
 * \file prompt_flash_attention_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;
namespace {
const uint32_t ATTR_INPUT_LAYOUT_INDEX = 4;
}  // namespace
namespace ops {
static ge::graphStatus InferShapePromptFlashAttention(gert::InferShapeContext* context) {
  OPS_LOG_I(context->GetNodeName(), "Enter PromptFlashAttention infershape impl.");
  // query shape : (B, S, H)
  const gert::Shape* query_shape = context->GetInputShape(0);
  const gert::Shape* value_shape = context->GetInputShape(2);
  OPS_LOG_E_IF_NULL(context, query_shape, return ge::GRAPH_FAILED)
  OPS_LOG_E_IF_NULL(context, value_shape, return ge::GRAPH_FAILED)

  // attentionOut: (B, S, H)
  gert::Shape* attentionOutShape = context->GetOutputShape(0);
  OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)

  // Get attr
  auto attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED)
  std::string inputLayout = std::string(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));

  // Set output shape
  if (inputLayout == "BNSD_BSND") {
    *attentionOutShape = *query_shape;
    attentionOutShape->SetDim(0, query_shape->GetDim(0));
    attentionOutShape->SetDim(1, query_shape->GetDim(2));       // 2: Obtain the second dimension
    attentionOutShape->SetDim(2, query_shape->GetDim(1));       // 2: DIM_NUM_2
    attentionOutShape->SetDim(3, query_shape->GetDim(3));       // 3: DIM_NUM_3
  } else if (inputLayout == "TND") {
    *attentionOutShape = *query_shape;
    attentionOutShape->SetDim(0, query_shape->GetDim(0));
    attentionOutShape->SetDim(1, query_shape->GetDim(1));
    attentionOutShape->SetDim(2, value_shape->GetDim(2));
  } else {
    *attentionOutShape = *query_shape;
  }
  OPS_LOG_I(context->GetNodeName(), "PromptFlashAttention infershape end.");
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePromptFlashAttention(gert::InferDataTypeContext* context) {
  OPS_LOG_I(context->GetNodeName(), "Enter PromptFlashAttention infershape impl.");
  // default set q's dtype as ifa's output type
  ge::DataType outputType = context->GetInputDataType(0);
  if (context->GetOptionalInputDataType(9) != ge::DT_UNDEFINED) { // 9 is quant_scale2's index
    outputType = ge::DT_INT8;
  } else if (outputType == ge::DT_INT8) {
    outputType = ge::DT_FLOAT16;
  }
  // attention_out, outidx:0
  context->SetOutputDataType(0, outputType);

  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeIncreFlashAttention(gert::InferDataTypeContext* context) {
  OPS_LOG_I(context->GetNodeName(), "Enter IncreFlashAttention inferDataShape impl.");
  // default set q's dtype as ifa's output type
  ge::DataType outputType = context->GetInputDataType(0);
  if (context->GetOptionalInputDataType(9) != ge::DT_UNDEFINED) { // 9 is quant_scale2's index
    outputType = ge::DT_INT8;
  } else if (outputType == ge::DT_INT8) {
    outputType = ge::DT_FLOAT16;
  }
  // attention_out, outidx:0
  context->SetOutputDataType(0, outputType);

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PromptFlashAttention).InferShape(InferShapePromptFlashAttention).InferDataType(InferDataTypePromptFlashAttention);
IMPL_OP_INFERSHAPE(IncreFlashAttention).InferShape(InferShapePromptFlashAttention).InferDataType(InferDataTypeIncreFlashAttention);

}  // namespace ops