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
 * \file moe_gating_top_k_softmax_proto.cpp
 * \brief
 */
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)        \
  do {                                                               \
    std::printf("op[%s], %s", op_name, err_msg);                     \
  } while (0)

namespace ops {
static const int64_t SIZE_2 = 2;
static const int64_t SIZE_3 = 3;
static const int INDEX_0 = 0;
static const int INDEX_1 = 1;
static const int INDEX_2 = 2;
static const int MAX_K = 1024;
static const int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;
static const int64_t UNKNOWN_DIM_VALUE_ = -1;

static inline bool IsUnknownRank(const gert::Shape *check_shape)
{
    return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

static inline bool IsUnknownShape(const gert::Shape *check_shape)
{
    for (size_t i = 0; i < check_shape->GetDimNum(); i++) {
        if (check_shape->GetDim(i) == UNKNOWN_DIM_VALUE_) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus InferShapeMoeGatingTopKSoftmax(gert::InferShapeContext* context) {
  const gert::Shape* gatingShape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, gatingShape);

  gert::Shape* outShape = context->GetOutputShape(0);
  gert::Shape* indicesShape = context->GetOutputShape(1);
  gert::Shape* sourceRowOutShape = context->GetOutputShape(2);

  if (IsUnknownRank(gatingShape)) {
    *outShape = *gatingShape;
    *indicesShape = *gatingShape;
    *sourceRowOutShape = *gatingShape;
    return ge::GRAPH_SUCCESS;
  }

  const int64_t gatingDimNum = gatingShape->GetDimNum();
  if (gatingDimNum != SIZE_2 && gatingDimNum != SIZE_3) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "x dimensions not equal 2 and 3!");
    return ge::GRAPH_FAILED;
  }

  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

  const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(0);
  const int64_t k = *kPtr;

  if (k <= 0 || k > MAX_K || (!IsUnknownShape(gatingShape) && k > gatingShape->GetDim(gatingDimNum - 1))) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "k value error!");
    return ge::GRAPH_FAILED;
  }

  outShape->SetDimNum(gatingDimNum);
  indicesShape->SetDimNum(gatingDimNum);
  sourceRowOutShape->SetDimNum(gatingDimNum);
  for (int64_t i = 0; i < gatingDimNum - 1; i++) {
    outShape->SetDim(i, gatingShape->GetDim(i));
    indicesShape->SetDim(i, gatingShape->GetDim(i));
    sourceRowOutShape->SetDim(i, gatingShape->GetDim(i));
  }
  outShape->SetDim(gatingDimNum - 1, k);
  indicesShape->SetDim(gatingDimNum - 1, k);
  sourceRowOutShape->SetDim(gatingDimNum - 1, k);

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeGatingTopKSoftmax(gert::InferDataTypeContext* context) {
  auto xDtype = context->GetInputDataType(INDEX_0);
  context->SetOutputDataType(INDEX_0, xDtype);
  context->SetOutputDataType(INDEX_1, ge::DT_INT32);
  context->SetOutputDataType(INDEX_2, ge::DT_INT32);
  OP_LOGD(context->GetNodeName(), "End MoeGatingTopKSoftmaxInferDataType.");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeGatingTopKSoftmax)
    .InferShape(InferShapeMoeGatingTopKSoftmax)
    .InferDataType(InferDataType4MoeGatingTopKSoftmax);
}  // namespace ops