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
 * \file moe_init_routing_quant_v2_ops.cc
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "log/ops_log.h"
#include "error/ops_error.h"
using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr int64_t OTHER_SHAPE = -1;
static constexpr int64_t INDEX_INPUT_X = 0;
static constexpr int64_t INDEX_INPUT_EXPERT_IDX = 1;
static constexpr int64_t INDEX_INPUT_SCALE = 2;
static constexpr int64_t INDEX_INPUT_OFFSET = 3;
static constexpr int64_t OUTOUT_EXPANDED_X = 0;
static constexpr int64_t OUTOUT_EXPANDED_ROW_IDX = 1;
static constexpr int64_t OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM = 2;
static constexpr int64_t OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY = 3;
static constexpr int64_t OUTOUT_DYNAMIC_QUANT_SCALE = 4;
static constexpr int64_t ATTR_ACTIVE_ROWS = 0;
static constexpr int64_t ATTR_EXPERT_CAPACITY = 1;
static constexpr int64_t ATTR_EXPERT_NUM = 2;
static constexpr int64_t ATTR_DROP_PAD_MODE = 3;
static constexpr int64_t ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG = 4;
static constexpr int64_t ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG = 5;
static constexpr int64_t ATTR_QUANT_MODE = 6;
static constexpr int64_t EXPERT_TOKENS_COUNT = 2;
static constexpr int64_t ONE_BLOCK_NUM = 8;

static bool isSameDim(int64_t dim1, int64_t dim2) {
  if (dim1 == OTHER_SHAPE || dim2 == OTHER_SHAPE) {
    return true;
  }
  return dim1 == dim2;
}

static ge::graphStatus CheckInputShape(gert::InferShapeContext* context, const gert::Shape* xShape,
                                       const gert::Shape* expertIdxShape) {
  int64_t x_n = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(0);
  int64_t cols = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(1);
  if (x_n < OTHER_SHAPE || cols < OTHER_SHAPE) {
    OPS_LOG_E(context->GetNodeName(), "Invalid x shape, shape is %s.", Shape2String(*xShape).c_str());
    return ge::GRAPH_FAILED;
  }

  int64_t expert_idx_n = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(0);
  int64_t expert_idx_k = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(1);
  if (expert_idx_n < OTHER_SHAPE || expert_idx_k < OTHER_SHAPE) {
    OPS_LOG_E(context->GetNodeName(), "Invalid expertIdx shape, shape is %s.", Shape2String(*expertIdxShape).c_str());
    return ge::GRAPH_FAILED;
  }

  if (!isSameDim(x_n, expert_idx_n)) {
    OPS_LOG_E(context->GetNodeName(), "The first dim of x(%ld) and expertIdx(%ld) should be equal.", x_n, expert_idx_n);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckParm(gert::InferShapeContext* context, const gert::Shape* xShape,
                                 const gert::Shape* expertIdxShape, const int64_t activeNum,
                                 const int64_t expertCapacity, const int64_t expertNum, const int64_t dropPadMode,
                                 const int64_t expertTokensCountOrCumsumFlag, bool expertTokensBeforeCapacityFlag,
                                 const int64_t quantMode) {
  if (xShape->GetDimNum() == 1U) {
    if (xShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
      OPS_LOG_E(context->GetNodeName(), "The dynamic dim of x should be -2, current shape is %s.",
              Shape2String(*xShape).c_str());
      return ge::GRAPH_FAILED;
    }
  } else if (xShape->GetDimNum() != DIM_TWO) {
    OPS_LOG_E(context->GetNodeName(), "The dim of x should be 2 or dynamic, current shape is %s.",
            Shape2String(*xShape).c_str());
    return ge::GRAPH_FAILED;
  }

  if (expertIdxShape->GetDimNum() == 1U) {
    if (expertIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
      OPS_LOG_E(context->GetNodeName(), "The dynamic dim of expertIdx should be -2, current shape is %s.",
              Shape2String(*expertIdxShape).c_str());
      return ge::GRAPH_FAILED;
    }
  } else if (expertIdxShape->GetDimNum() != DIM_TWO) {
    OPS_LOG_E(context->GetNodeName(), "The dim of expertIdx should be 2 or dynamic, current shape is %s.",
            Shape2String(*expertIdxShape).c_str());
    return ge::GRAPH_FAILED;
  }
  if (activeNum < 0) {
    OPS_LOG_E(context->GetNodeName(), "activeNum cannot be less than 0.");
    return ge::GRAPH_FAILED;
  }
  if (expertCapacity < 0) {
    OPS_LOG_E(context->GetNodeName(), "The expertCapacity cannot be less than 0.");
    return ge::GRAPH_FAILED;
  }
  if (expertNum < 0) {
    OPS_LOG_E(context->GetNodeName(), "The expertNum should cannot be less than 0.");
    return ge::GRAPH_FAILED;
  }
  if (dropPadMode < 0 || dropPadMode > 1) {
    OPS_LOG_E(context->GetNodeName(), "The dropPadMode should be 0 or 1.");
    return ge::GRAPH_FAILED;
  }
  if (dropPadMode > 0 && (expertCapacity < 1 || expertNum < 1)) {
    OPS_LOG_E(context->GetNodeName(), "The expertCapacity and expertNum should be greater 0 when dropPadMode is 1");
    return ge::GRAPH_FAILED;
  }
  if (expertTokensCountOrCumsumFlag < 0 || expertTokensCountOrCumsumFlag > EXPERT_TOKENS_COUNT) {
    OPS_LOG_E(context->GetNodeName(), "The expertTokensCountOrCumsumFlag should be 0, 1 or 2.");
    return ge::GRAPH_FAILED;
  }
  if (expertTokensCountOrCumsumFlag > 0 && expertNum <= 0) {
    OPS_LOG_E(context->GetNodeName(),
            "The expertNum should be greater than 0 when expertTokensCountOrCumsumFlag is greater than 0");
    return ge::GRAPH_FAILED;
  }
  if (dropPadMode > 0 && xShape->GetDim(0) > 0 && expertCapacity > xShape->GetDim(0)) {
    OPS_LOG_E(context->GetNodeName(), "The first dim of x cannot be less than expertCapacity when dropPadMode is 1");
    return ge::GRAPH_FAILED;
  }
  if (quantMode < 0 || quantMode > 1) {
    OPS_LOG_E(context->GetNodeName(), "The quantMode should be 0 or 1.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckScaleOffset(gert::InferShapeContext* context, const gert::Shape* shape, const char* tag) {
  if (shape->GetDimNum() == 1U) {
    if (shape->GetDim(0) != ge::UNKNOWN_DIM_NUM && shape->GetDim(0) != OTHER_SHAPE && shape->GetDim(0) != 1) {
      OPS_LOG_E(context->GetNodeName(), "Invalid %s shape, current shape is %s.", tag, Shape2String(*shape).c_str());
      return ge::GRAPH_FAILED;
    }
  } else {
    OPS_LOG_E(context->GetNodeName(), "Invalid %s dim num, current dim num is %lu, should be 1.", tag,
            shape->GetDimNum());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckScaleOffsetInput(gert::InferShapeContext* context, const int64_t quantMode,
                                             const int64_t dropPadMode, const int64_t expertNum, const int64_t cols) {
  const gert::Shape* scaleShape = context->GetInputShape(INDEX_INPUT_SCALE);
  if (quantMode == 0) {
    OPS_LOG_E_IF_NULL(context, scaleShape, return ge::GRAPH_FAILED);
    if (CheckScaleOffset(context, scaleShape, "scale") == ge::GRAPH_FAILED) {
      return ge::GRAPH_FAILED;
    }

    const gert::Shape* offsetShape = context->GetInputShape(INDEX_INPUT_OFFSET);
    OPS_LOG_E_IF_NULL(context, offsetShape, return ge::GRAPH_FAILED);
    if (CheckScaleOffset(context, offsetShape, "offset") == ge::GRAPH_FAILED) {
      return ge::GRAPH_FAILED;
    }
  } else {
    if (scaleShape == nullptr) {
      return ge::GRAPH_SUCCESS;
    }
    if (scaleShape->GetDimNum() == 1U) {
      if (scaleShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
        OPS_LOG_E(context->GetNodeName(), "The dynamic dim of scale should be -2, current shape is %s.",
                Shape2String(*scaleShape).c_str());
        return ge::GRAPH_FAILED;
      }
    } else if (scaleShape->GetDimNum() != DIM_TWO) {
      OPS_LOG_E(context->GetNodeName(), "The dim of scale should be 2 or dynamic, current shape is %s.",
              Shape2String(*scaleShape).c_str());
      return ge::GRAPH_FAILED;
    }
    int64_t fdm = scaleShape->GetDimNum() == 1U ? OTHER_SHAPE : scaleShape->GetDim(0);
    int64_t sdm = scaleShape->GetDimNum() == 1U ? OTHER_SHAPE : scaleShape->GetDim(1);
    if (fdm < OTHER_SHAPE || sdm < OTHER_SHAPE) {
      OPS_LOG_E(context->GetNodeName(), "Invalid scale shape, shape is %s.", Shape2String(*scaleShape).c_str());
      return ge::GRAPH_FAILED;
    }
    if (fdm != OTHER_SHAPE && fdm != 1 && fdm != expertNum) {
      OPS_LOG_E(context->GetNodeName(), "Invalid scale first dim, which is %ld. should be -1, 1 or expertNum.", fdm);
      return ge::GRAPH_FAILED;
    }
    if (sdm != OTHER_SHAPE && cols != OTHER_SHAPE && sdm != cols) {
      OPS_LOG_E(context->GetNodeName(), "The second dim of scale(%ld) and x(%ld) should be equal.", sdm, cols);
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

static void InferOutputShape(const gert::Shape* xShape, const gert::Shape* expertIdxShape, gert::Shape* expandedXShape,
                             gert::Shape* expandedRowIdx, gert::Shape* expertTokensBeforeCapacityShape,
                             gert::Shape* expertTokensCountOrCumsumShape, gert::Shape* dynamicQuantScaleShape,
                             const int64_t activeNum, const int64_t expertNum, const int64_t expertCapacity,
                             const int64_t dropPadMode, bool expertTokensBeforeCapacityFlag,
                             const int64_t expertTokensCountOrCumsumFlag, const int64_t quantMode) {
  int64_t n = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(0);
  int64_t cols = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(1);
  int64_t k = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(1);
  int64_t outActiveNum = OTHER_SHAPE;
  int64_t expandedRowIdxNum = OTHER_SHAPE;

  if (n > 0 && k > 0) {
    expandedRowIdxNum = n * k;
    outActiveNum = activeNum > 0 ? std::min(n * k, activeNum) : n * k;
  }

  if (dropPadMode > 0) {
    expandedXShape->SetDimNum(DIM_THREE);
    expandedXShape->SetDim(0U, expertNum);
    expandedXShape->SetDim(1U, expertCapacity);
    expandedXShape->SetDim(2U, cols < 0 ? OTHER_SHAPE : cols);
  } else {
    expandedXShape->SetDimNum(DIM_TWO);
    expandedXShape->SetDim(0U, outActiveNum);
    expandedXShape->SetDim(1U, cols < 0 ? OTHER_SHAPE : cols);
  }

  expandedRowIdx->SetDimNum(DIM_ONE);
  expandedRowIdx->SetDim(0U, expandedRowIdxNum);

  if (dropPadMode == 1 && expertTokensBeforeCapacityFlag) {
    expertTokensBeforeCapacityShape->SetDimNum(DIM_ONE);
    expertTokensBeforeCapacityShape->SetDim(0U, expertNum);
  }

  if (dropPadMode == 0 && expertTokensCountOrCumsumFlag > 0) {
    expertTokensCountOrCumsumShape->SetDimNum(DIM_ONE);
    expertTokensCountOrCumsumShape->SetDim(0U, expertNum);
  }

  if (quantMode == 1) {
    dynamicQuantScaleShape->SetDimNum(DIM_ONE);
    int64_t fdim = dropPadMode == 0 ? outActiveNum : expertNum * expertCapacity;
    dynamicQuantScaleShape->SetDim(0U, fdim);
  }
}

static ge::graphStatus InferShape4MoeInitRoutingQuantV2(gert::InferShapeContext* context) {
  OPS_LOG_D(context->GetNodeName(), "Begin to do MoeInitRountingQuantV2Infershape.");
  // 获取attr
  auto attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
  const int64_t* activeNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVE_ROWS);
  const int64_t activeNum = (activeNumPtr == nullptr) ? 0 : *activeNumPtr;
  const int64_t* expertCapacityPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_CAPACITY);
  const int64_t expertCapacity = (expertCapacityPtr == nullptr) ? 0 : *expertCapacityPtr;
  const int64_t* expertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_NUM);
  const int64_t expertNum = (expertNumPtr == nullptr) ? 0 : *expertNumPtr;
  const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE);
  const int64_t dropPadMode = (dropPadModePtr == nullptr) ? 0 : *dropPadModePtr;
  const int64_t* expertTokensCountOrCumsumFlagPtr =
      attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG);
  const int64_t expertTokensCountOrCumsumFlag =
      (expertTokensCountOrCumsumFlagPtr == nullptr) ? 0 : *expertTokensCountOrCumsumFlagPtr;
  const bool* expertTokensBeforeCapacityFlagPtr = attrs->GetAttrPointer<bool>(ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG);
  const bool expertTokensBeforeCapacityFlag =
      (expertTokensBeforeCapacityFlagPtr == nullptr) ? false : *expertTokensBeforeCapacityFlagPtr;
  const int64_t* quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE);
  const int64_t quantMode = (quantModePtr == nullptr) ? 0 : *quantModePtr;

  // 获取输入shape
  const gert::Shape* xShape = context->GetInputShape(INDEX_INPUT_X);
  OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED);
  const gert::Shape* expertIdxShape = context->GetInputShape(INDEX_INPUT_EXPERT_IDX);
  OPS_LOG_E_IF_NULL(context, expertIdxShape, return ge::GRAPH_FAILED);
  gert::Shape* expandedXShape = context->GetOutputShape(OUTOUT_EXPANDED_X);
  OPS_LOG_E_IF_NULL(context, expandedXShape, return ge::GRAPH_FAILED);
  gert::Shape* expandedRowIdx = context->GetOutputShape(OUTOUT_EXPANDED_ROW_IDX);
  OPS_LOG_E_IF_NULL(context, expandedRowIdx, return ge::GRAPH_FAILED);

  gert::Shape* expertTokensCountOrCumsumShape = context->GetOutputShape(OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM);
  if (dropPadMode == 0 && expertTokensCountOrCumsumFlag > 0) {
    OPS_LOG_E_IF_NULL(context, expertTokensCountOrCumsumShape, return ge::GRAPH_FAILED);
  }
  gert::Shape* expertTokensBeforeCapacityShape = context->GetOutputShape(OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY);
  if (dropPadMode == 1 && expertTokensBeforeCapacityFlag) {
    OPS_LOG_E_IF_NULL(context, expertTokensBeforeCapacityShape, return ge::GRAPH_FAILED);
  }
  gert::Shape* dynamicQuantScaleShape = context->GetOutputShape(OUTOUT_DYNAMIC_QUANT_SCALE);
  if (quantMode == 1) {
    OPS_LOG_E_IF_NULL(context, dynamicQuantScaleShape, return ge::GRAPH_FAILED);
  }

  if (CheckInputShape(context, xShape, expertIdxShape) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  if (CheckParm(context, xShape, expertIdxShape, activeNum, expertCapacity, expertNum, dropPadMode,
                expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, quantMode) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  int64_t cols = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(1);
  if (CheckScaleOffsetInput(context, quantMode, dropPadMode, expertNum, cols) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  InferOutputShape(xShape, expertIdxShape, expandedXShape, expandedRowIdx, expertTokensBeforeCapacityShape,
                   expertTokensCountOrCumsumShape, dynamicQuantScaleShape, activeNum, expertNum, expertCapacity,
                   dropPadMode, expertTokensBeforeCapacityFlag, expertTokensCountOrCumsumFlag, quantMode);

  OPS_LOG_D(context->GetNodeName(), "End to do MoeInitRountingQuantV2Infershape.");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeInitRoutingQuantV2(gert::InferDataTypeContext* context) {
  OPS_LOG_D(context->GetNodeName(), "Begin to do MoeInitRountingQuantV2InferDataType.");
  context->SetOutputDataType(OUTOUT_EXPANDED_X, ge::DT_INT8);
  context->SetOutputDataType(OUTOUT_EXPANDED_ROW_IDX, ge::DT_INT32);
  context->SetOutputDataType(OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM, ge::DT_INT32);
  context->SetOutputDataType(OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY, ge::DT_INT32);
  context->SetOutputDataType(OUTOUT_DYNAMIC_QUANT_SCALE, ge::DT_FLOAT);
  OPS_LOG_D(context->GetNodeName(), "End to do MoeInitRountingQuantV2InferDataType.");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeInitRoutingQuantV2)
    .InferShape(InferShape4MoeInitRoutingQuantV2)
    .InferDataType(InferDataType4MoeInitRoutingQuantV2);
}  // namespace ops
