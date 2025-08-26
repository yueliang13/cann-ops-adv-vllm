/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file swin_transformer_ln_qkv_quant.cc
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "error/ops_error.h"

namespace {
constexpr size_t OUTPUT_CHANNEL = 4;
constexpr int64_t UNKNOWN_DIM_VALUE = -1;
}

using namespace ge;
namespace ops {
static const int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;
static const int64_t UNKNOWN_DIM_VALUE_ = -1;

static inline bool IsUnknownRank(const gert::Shape *check_shape)
{
    return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

inline ge::graphStatus SetUnknownRank(gert::Shape* outShape) {
  outShape->SetDimNum(0);
  outShape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);
  return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape) {

  output_shape->SetDimNum(rank);
  for (int64_t i = 0; i < rank; ++i) {
    output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeSwinTransformerLnQkvQuant(gert::InferShapeContext* context) {
  OPS_LOG_I(context, "Enter SwinTransformerLnQkvQuant infershape impl.");
  // xShape shape : (B, S, H)
  const gert::Shape* xShape = context->GetInputShape(0);
  OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED);

  gert::Shape* qShape = context->GetOutputShape(0);
  OPS_LOG_E_IF_NULL(context, qShape, return ge::GRAPH_FAILED);

  gert::Shape* kShape = context->GetOutputShape(1);
  OPS_LOG_E_IF_NULL(context, kShape, return ge::GRAPH_FAILED);

  gert::Shape* vShape = context->GetOutputShape(2);
  OPS_LOG_E_IF_NULL(context, vShape, return ge::GRAPH_FAILED);
  uint32_t inputSize = 1;
  ge::graphStatus ret;
  if (IsUnknownRank(xShape)) {
    OPS_LOG_I(context, "input shape is UnknownRank, set output shape to -2");
    ret = SetUnknownRank(qShape);
    if (ret != ge::GRAPH_SUCCESS || (SetUnknownRank(kShape) != ge::GRAPH_SUCCESS) ||
                (SetUnknownRank(vShape) != ge::GRAPH_SUCCESS)) {
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }
  for (uint32_t dimIdx = 0; dimIdx < xShape->GetDimNum(); dimIdx++) {
      auto shapeValue = xShape->GetDim(dimIdx);
      if (shapeValue == UNKNOWN_DIM_VALUE_) {
        ret = SetAllUnknownDim(OUTPUT_CHANNEL, qShape);
        if ((ret != ge::GRAPH_SUCCESS) || (SetAllUnknownDim(OUTPUT_CHANNEL, kShape) != ge::GRAPH_SUCCESS) ||
                (SetAllUnknownDim(OUTPUT_CHANNEL, vShape) != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
      }
      inputSize *= shapeValue;
  }

  OPS_LOG_I(context, "inputSize: %u", inputSize);
  qShape->SetDimNum(OUTPUT_CHANNEL);
  kShape->SetDimNum(OUTPUT_CHANNEL);
  vShape->SetDimNum(OUTPUT_CHANNEL);
  auto attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);

  const auto *headNum = attrs->GetAttrPointer<int64_t>(0);
  OPS_LOG_E_IF_NULL(context, headNum, return ge::GRAPH_FAILED);
  const auto *seqLength = attrs->GetAttrPointer<int64_t>(1);
  const auto *hWinSize = attrs->GetAttrPointer<int64_t>(5);   // attr param 5 is hWin 
  const auto *wWinSize = attrs->GetAttrPointer<int64_t>(6);  //  attr params 6 is wWin 
  
  const uint32_t outputChannel1 = *headNum;
  const uint32_t outputChannel2 = (*hWinSize) * (*wWinSize);
  const uint32_t outputChannel3 = *seqLength;
  const uint32_t outputChannel0 = inputSize / (outputChannel1 * outputChannel2 * outputChannel3);
  const uint32_t outputDim0 = 0;
  const uint32_t outputDim1 = 1;
  const uint32_t outputDim2 = 2;
  const uint32_t outputDim3 = 3;
  OPS_LOG_I(context, "headNum: %ld, hWinSize is %ld, wWinSize is %ld, outputDim0 is %u, \
    outputDim1 is %u,outputDim2 is %u,outputDim3 is %u", *headNum, *hWinSize, *wWinSize, outputChannel0, \
    outputChannel1, outputChannel2, outputChannel3);
  qShape->SetDim(outputDim0, outputChannel0);
  qShape->SetDim(outputDim1, outputChannel1);
  qShape->SetDim(outputDim2, outputChannel2);
  qShape->SetDim(outputDim3, outputChannel3);
  *kShape = *qShape;
  *vShape = *qShape;
  OPS_LOG_I(context, "SwinTransformerLnQKV infershape end.");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSwinTransformerLnQkvQuant(gert::InferDataTypeContext *context) {
  OPS_LOG_I(context, "Enter SwinTransformerLnQkvQuant inferDataType impl.");
  const ge::DataType dtype = context->GetInputDataType(0);
  // q_out, outidx:0
  ge::graphStatus ret0 = context->SetOutputDataType(0, dtype);
  // k_out, outidx:1
  context->SetOutputDataType(1, dtype);
  // v_out, outidx:2
  context->SetOutputDataType(2, dtype);
  OPS_LOG_I(context, "SwinTransformerLnQkvQuant inferDataType end.");
  return ret0;
}

IMPL_OP_INFERSHAPE(SwinTransformerLnQkvQuant)
    .InferShape(InferShapeSwinTransformerLnQkvQuant)
    .InferDataType(InferDataTypeSwinTransformerLnQkvQuant);
}  // namespace ops