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
 * \file moe_token_permute.cc
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"
using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t UNKNOWN_RANK_DIM_VALUE = -2;
static ge::graphStatus InferShape4MoeTokenPermute(gert::InferShapeContext* context) {
  OPS_LOG_D(context->GetNodeName(), "Begin to do MoeTokenPermuteInfershape.");
  // 获取输入shape
  const gert::Shape* xShape = context->GetInputShape(0);
  OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED);
  const gert::Shape* indicesShape = context->GetInputShape(1);
  OPS_LOG_E_IF_NULL(context, indicesShape, return ge::GRAPH_FAILED);

  gert::Shape* permuteXShape = context->GetOutputShape(0);
  OPS_LOG_E_IF_NULL(context, permuteXShape, return ge::GRAPH_FAILED);
  gert::Shape* sortedIndices = context->GetOutputShape(1);
  OPS_LOG_E_IF_NULL(context, sortedIndices, return ge::GRAPH_FAILED);

  // 获取attr
  auto attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
  const int64_t* numOutTokensPtr = attrs->GetAttrPointer<int64_t>(0);

  int64_t sortedIndicesLen = NEG_ONE;
  sortedIndices->SetDimNum(DIM_ONE);
  if (sortedIndices->GetDim(0) < 0) {
    sortedIndices->SetDim(0U, NEG_ONE);
  } else {
    int64_t IndicesDimNnum = indicesShape->GetDimNum();
    if (IndicesDimNnum != DIM_TWO && IndicesDimNnum != DIM_ONE) {
      OPS_LOG_E(context->GetNodeName(), "The dim of indices should 1 or 2,but got %ld.", IndicesDimNnum);
      return ge::GRAPH_FAILED;
    }
    int64_t topK = (IndicesDimNnum == 1) ? 1 : indicesShape->GetDim(1);
    int64_t N = indicesShape->GetDim(0);
    sortedIndicesLen = (topK * N > 0) ? topK * N : NEG_ONE;
    sortedIndices->SetDim(0U, sortedIndicesLen);
  }

  *permuteXShape = *xShape;
  bool isUnknownRank = xShape->GetDimNum() == 1 && xShape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE;
  if (sortedIndicesLen != NEG_ONE && !isUnknownRank) {
    int64_t outTokens = (*numOutTokensPtr <= 0) ? (*numOutTokensPtr) + sortedIndicesLen : *numOutTokensPtr;
    outTokens = std::min(outTokens, sortedIndicesLen);
    outTokens = std::max(outTokens, (int64_t)0);
    permuteXShape->SetDim(0, outTokens);
  }

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeTokenPermute(gert::InferDataTypeContext* context) {
  OPS_LOG_D(context->GetNodeName(), "Begin to do MoeTokenPermuteInferDataType.");
  auto xDtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, xDtype);
  context->SetOutputDataType(1, ge::DT_INT32);
  OPS_LOG_D(context->GetNodeName(), "End to do MoeTokenPermuteInferDataType.");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenPermute).InferShape(InferShape4MoeTokenPermute).InferDataType(InferDataType4MoeTokenPermute);
}  // namespace ops
