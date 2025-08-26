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
 * \file moe_finalize_routing_v2_ops.cc
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "log/ops_log.h"
#include "error/ops_error.h"

using namespace ge;
namespace ops {
static const size_t INDEX_IN_EXPANDED_X = 0;
static const size_t INDEX_IN_EXPANDED_ROW_IDX = 1;
static const size_t INDEX_IN_SKIP1 = 2;
static const size_t INDEX_IN_SKIP2 = 3;
static const size_t INDEX_IN_BIAS = 4;
static const size_t INDEX_IN_SCALES = 5;
static const size_t INDEX_IN_EXPERT_IDX = 6;
static constexpr size_t INDEX_OUT = 0;
static constexpr size_t SHAPE_SIZE = 2;
static constexpr size_t INPUT_NUM = 7;
static constexpr size_t ATTR_DROP_PAD_MODE = 0;
static constexpr int64_t VALUE_MODE_0 = 0;
static constexpr int64_t VALUE_MODE_1 = 1;
static constexpr int64_t VALUE_MODE_2 = 2;
static constexpr int64_t VALUE_MODE_3 = 3;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;

static inline bool IsValidType(const DataType dtype) {
  return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

static ge::graphStatus InferDataTypeMoeFinalizeRoutingV2(gert::InferDataTypeContext* context) {
  OPS_LOG_D(context->GetNodeName(), "Begin to do MoeFinalizeRoutingV2InferDataType.");
  OPS_LOG_E_IF(!IsValidType(context->GetInputDataType(INDEX_IN_EXPANDED_X)), context, return ge::GRAPH_FAILED,
             "the dtype of expanded_x should be float, float16 or bf16.");
  
  OPS_LOG_E_IF(context->GetInputDataType(INDEX_IN_EXPANDED_ROW_IDX) != ge::DT_INT32, context,
             return ge::GRAPH_FAILED, "the dtype of expanded_row_idx should be int32.");
  
  DataType parameterDtype = context->GetOptionalInputDataType(INDEX_IN_SKIP1);
  OPS_LOG_E_IF(parameterDtype != ge::DT_UNDEFINED && !IsValidType(parameterDtype), context, return ge::GRAPH_FAILED,
             "the dtype of skip1 should be float, float16 or bf16.");
  
  parameterDtype = context->GetOptionalInputDataType(INDEX_IN_SKIP2);
  OPS_LOG_E_IF(parameterDtype != ge::DT_UNDEFINED && !IsValidType(parameterDtype), context, return ge::GRAPH_FAILED,
             "the dtype of skip2 should be float, float16 or bf16.");
  
  parameterDtype = context->GetOptionalInputDataType(INDEX_IN_BIAS);
  OPS_LOG_E_IF(parameterDtype != ge::DT_UNDEFINED && !IsValidType(parameterDtype), context, return ge::GRAPH_FAILED,
             "the dtype of bias should be float, float16 or bf16.");
  
  parameterDtype = context->GetOptionalInputDataType(INDEX_IN_SCALES);
  OPS_LOG_E_IF(parameterDtype != ge::DT_UNDEFINED && !IsValidType(parameterDtype), context, return ge::GRAPH_FAILED,
             "the dtype of scales should be float, float16 or bf16.");
  
  parameterDtype = context->GetOptionalInputDataType(INDEX_IN_EXPERT_IDX);
  OPS_LOG_E_IF(parameterDtype != ge::DT_UNDEFINED && parameterDtype != ge::DT_INT32, context,
             return ge::GRAPH_FAILED, "the dtype of expert_idx should be int32.");
  
  context->SetOutputDataType(0, context->GetInputDataType(INDEX_IN_EXPANDED_X));
  return ge::GRAPH_SUCCESS;
}

static bool IsValidShape(const int64_t shape1, const int64_t shape2, const int64_t shape3, const int64_t shape4) {
  std::vector<int64_t> validValue = {-1};
  std::vector<int64_t> currentValue = {shape1, shape2, shape3, shape4};
  for (auto value : currentValue) {
    if (value == -1) {
      continue;
    }
    if (validValue.size() == 1) {
      validValue.push_back(value);
    } else if (validValue[1] != value) {
      return false;
    }
  }
  return true;
}

static inline bool IsUnknownRank(const gert::Shape* check_shape) {
  return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

static inline bool IsUnknownShape(const gert::Shape* check_shape) {
  for (size_t i = 0; i < check_shape->GetDimNum(); i++) {
    if (check_shape->GetDim(i) == UNKNOWN_DIM_VALUE_) {
      return true;
    }
  }
  return false;
}

inline ge::graphStatus SetAllUnknownDim(gert::Shape* outShape) {
  outShape->SetDimNum(0);
  outShape->AppendDim(UNKNOWN_DIM_VALUE_);
  outShape->AppendDim(UNKNOWN_DIM_VALUE_);
  return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus SetUnknownRank(gert::Shape* outShape) {
  outShape->SetDimNum(0);
  outShape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeCopyShapeInput2OutputWithIdx(gert::InferShapeContext* context) {
  auto outShape = context->GetOutputShape(INDEX_OUT);
  outShape->SetDimNum(0);
  const gert::Shape* expandedXInputShape = context->GetInputShape(INDEX_IN_EXPANDED_X);
  const gert::Shape* expandedSrcToDstRowInputShape = context->GetInputShape(INDEX_IN_EXPANDED_ROW_IDX);
  const gert::Shape* scalesInputShape = context->GetOptionalInputShape(INDEX_IN_SCALES);
  int64_t valueDim0 = expandedSrcToDstRowInputShape->GetDim(0);
  if (scalesInputShape != nullptr) {
    valueDim0 = scalesInputShape->GetDim(0);
  }
  outShape->AppendDim(valueDim0);
  outShape->AppendDim(expandedXInputShape->GetDim(expandedXInputShape->GetDimNum() - 1));
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Infershape4MoeFinalizeRoutingV2(gert::InferShapeContext* context) {
  // get and check input param

  auto attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
  const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE);
  OPS_LOG_E_IF_NULL(context, dropPadModePtr, return ge::GRAPH_FAILED);
  const int64_t dropPadMode = *dropPadModePtr;

  auto outShape = context->GetOutputShape(INDEX_OUT);
  OPS_LOG_E_IF(dropPadMode < VALUE_MODE_0 || dropPadMode > VALUE_MODE_3, context, return ge::GRAPH_FAILED,
             "attr drop_pad_mode should be [0,3].");

  const gert::Shape* expandedXShape = context->GetInputShape(INDEX_IN_EXPANDED_X);
  OPS_LOG_E_IF_NULL(context, expandedXShape, return ge::GRAPH_FAILED);
  if (IsUnknownRank(expandedXShape)) {
    return SetUnknownRank(outShape);
  } 
  if (IsUnknownShape(expandedXShape)) {
    return SetAllUnknownDim(outShape);
  }
  const size_t expandedXShapeSize = expandedXShape->GetDimNum();
  auto lastDimExpandedX = expandedXShape->GetDim(expandedXShapeSize - 1);

  OPS_LOG_E_IF((dropPadMode == VALUE_MODE_0 || dropPadMode == VALUE_MODE_2) && expandedXShapeSize != SHAPE_SIZE,
              context, return ge::GRAPH_FAILED,               "the expanded_x of input should be 2D tensor when drop_pad_mode is 0 or 2.");
  OPS_LOG_E_IF((dropPadMode == VALUE_MODE_1 || dropPadMode == VALUE_MODE_3) && expandedXShapeSize != SHAPE_SIZE + 1,
              context, return ge::GRAPH_FAILED,               "the expanded_x of input should be 3D tensor when drop_pad_mode is 1 or 3.");

  const gert::Shape* expandedSrcToDstRowInputShape = context->GetInputShape(INDEX_IN_EXPANDED_ROW_IDX);
  OPS_LOG_E_IF_NULL(context, expandedSrcToDstRowInputShape, return ge::GRAPH_FAILED);
  if (IsUnknownRank(expandedSrcToDstRowInputShape)) {
    return SetUnknownRank(outShape);
  } 
  if (IsUnknownShape(expandedSrcToDstRowInputShape)) {
    return SetAllUnknownDim(outShape);
  }
  OPS_LOG_E_IF(expandedSrcToDstRowInputShape->GetDimNum() != 1, context, return ge::GRAPH_FAILED,
             "the expanded_row_idx of input should be 1D tensor.");
  
  const gert::Tensor* x1Tensor = context->GetOptionalInputTensor(INDEX_IN_SKIP1);
  const gert::Shape* skip1InputShape = nullptr;
  if (x1Tensor != nullptr && x1Tensor->GetShapeSize() != 0) {
    skip1InputShape = context->GetOptionalInputShape(INDEX_IN_SKIP1);  
    if (IsUnknownRank(skip1InputShape)) {
      return SetUnknownRank(outShape);
    } 
    if (IsUnknownShape(skip1InputShape)) {
      return SetAllUnknownDim(outShape);
    }
    OPS_LOG_E_IF(skip1InputShape->GetDimNum() != SHAPE_SIZE, context, return ge::GRAPH_FAILED,
        "the skip1 of input should be 2D tensor.");
  }
  
  const gert::Tensor* x2Tensor = context->GetOptionalInputTensor(INDEX_IN_SKIP2);
  const gert::Shape* skip2InputShape = nullptr;
  if (x2Tensor != nullptr && x2Tensor->GetShapeSize() != 0) {
    skip2InputShape = context->GetOptionalInputShape(INDEX_IN_SKIP2);
    if (IsUnknownRank(skip2InputShape)) {
      return SetUnknownRank(outShape);
    } 
    if (IsUnknownShape(skip2InputShape)) {
      return SetAllUnknownDim(outShape);
    }
    OPS_LOG_E_IF(skip2InputShape->GetDimNum() != SHAPE_SIZE, context, return ge::GRAPH_FAILED,
              "the skip2 of input should be 2D tensor.");
  }
  const gert::Tensor* biasTensor = context->GetOptionalInputTensor(INDEX_IN_BIAS);
  const gert::Shape* biasInputShape = nullptr;
  if (biasTensor != nullptr && biasTensor->GetShapeSize() != 0) {
    biasInputShape = context->GetOptionalInputShape(INDEX_IN_BIAS);
    if (IsUnknownRank(biasInputShape)) {
      return SetUnknownRank(outShape);
    } 
    if (IsUnknownShape(biasInputShape)) {
      return SetAllUnknownDim(outShape);
    }
    OPS_LOG_E_IF(biasInputShape->GetDimNum() != SHAPE_SIZE, context, return ge::GRAPH_FAILED, 
        "the bias of input should be 2D tensor.");
  }
  const gert::Tensor* scalesTensor = context->GetOptionalInputTensor(INDEX_IN_SCALES);
  const gert::Shape* scalesInputShape = nullptr;
  if (scalesTensor != nullptr && scalesTensor->GetShapeSize() != 0) {
    scalesInputShape = context->GetOptionalInputShape(INDEX_IN_SCALES);
    if (IsUnknownRank(scalesInputShape)) {
      return SetUnknownRank(outShape);
    } 
    if (IsUnknownShape(scalesInputShape)) {
      return SetAllUnknownDim(outShape);
    }

    OPS_LOG_E_IF(scalesInputShape->GetDimNum() != SHAPE_SIZE, context, return ge::GRAPH_FAILED, 
        "the scales of input should be 2D tensor.");
  }
  const gert::Tensor* expertIdxTensor = context->GetOptionalInputTensor(INDEX_IN_EXPERT_IDX);
  const gert::Shape* expertIdxShape = nullptr;
  if (expertIdxTensor != nullptr && expertIdxTensor->GetShapeSize() != 0) {
    expertIdxShape = context->GetOptionalInputShape(INDEX_IN_EXPERT_IDX);
    if (IsUnknownRank(expertIdxShape)) {
      return SetUnknownRank(outShape);
    } 
    if (IsUnknownShape(expertIdxShape)) {
      return SetAllUnknownDim(outShape);
    }
    OPS_LOG_E_IF(expertIdxShape->GetDimNum() != SHAPE_SIZE, context, return ge::GRAPH_FAILED,
              "the expert_idx of input should be 2D tensor.");
  }
  bool validColK = (scalesInputShape == nullptr || expertIdxTensor == nullptr) || 
    ((scalesInputShape != nullptr && expertIdxTensor != nullptr) &&
    (scalesInputShape->GetDim(1) == -1 || expertIdxShape->GetDim(1) == -1 ||
    scalesInputShape->GetDim(1) == expertIdxShape->GetDim(1)));
  OPS_LOG_E_IF(!validColK, context, return ge::GRAPH_FAILED,              "the dim 1 of scales and expert_idx should be same.");

  int64_t skip2Row = skip2InputShape != nullptr ? skip2InputShape->GetDim(0) : -1;
  int64_t skip1Row = skip1InputShape != nullptr ? skip1InputShape->GetDim(0) : -1;
  int64_t scaleRow = scalesInputShape != nullptr ? scalesInputShape->GetDim(0) : -1;
  int64_t expertIdxRow = expertIdxShape != nullptr? expertIdxShape->GetDim(0) : -1;
  OPS_LOG_E_IF(!IsValidShape(skip1Row, skip2Row, scaleRow,
             expertIdxRow), context, return ge::GRAPH_FAILED,              "the dim 0 of skip1, skip2, scales and expert_idx should be same.");

  int64_t skip2Col = skip2InputShape != nullptr ? skip2InputShape->GetDim(1) : -1;
  int64_t skip1Col = skip1InputShape != nullptr ? skip1InputShape->GetDim(1) : -1;
  int64_t biasCol = biasInputShape != nullptr? biasInputShape->GetDim(1) : -1;
  OPS_LOG_E_IF(!IsValidShape(skip1Col, skip2Col, lastDimExpandedX,
             biasCol), context, return ge::GRAPH_FAILED,              "the dim 1 of skip1, skip2, bias and last dim of expanded_x should be same.");

  if (dropPadMode == VALUE_MODE_0 || dropPadMode == VALUE_MODE_2) {
    bool validDim = expandedSrcToDstRowInputShape->GetDim(0) == -1 || expandedXShape->GetDim(0) == -1 ||
      (expandedSrcToDstRowInputShape->GetDim(0) == expandedXShape->GetDim(0));
    OPS_LOG_E_IF(!validDim, context, return ge::GRAPH_FAILED,               "the dim 0 of expanded_x and expanded_row_idx should be same when drop_pad_mode is 0.");
  }
  // infershape output
  OPS_CHECK(MoeCopyShapeInput2OutputWithIdx(context) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Infershape4MoeFinalizeRoutingV2 failed!"),
            return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeFinalizeRoutingV2).InferShape(Infershape4MoeFinalizeRoutingV2).InferDataType(InferDataTypeMoeFinalizeRoutingV2);
} // namespace ops