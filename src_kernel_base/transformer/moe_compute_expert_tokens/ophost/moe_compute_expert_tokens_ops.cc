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
 * \file moe_compute_expert_tokens_ops.cc
 * \brief
 */
#include <cmath>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"
#include "error/ops_error.h"

using namespace ge;
namespace ops {
// -------------------MoeComputeExpertTokens Ops START---------------------
static const size_t INDEX_IN_SORTED_EXPERTS_X1 = 0;
static constexpr size_t INDEX_out = 0;
constexpr size_t numExpertsAttrIdx = 0U;


static ge::graphStatus InfershapeForMoeComputeExpertTokens(gert::InferShapeContext* context) {
  OPS_LOG_D(context->GetNodeName(), "Begin to do MoeComputeExpertTokensInfershape.");
  // 获取输入值shape
  const gert::Shape* inputShape = context->GetInputShape(INDEX_IN_SORTED_EXPERTS_X1);
  OPS_LOG_E_IF_NULL(context, inputShape, return ge::GRAPH_FAILED);

  // 获取属性值
  auto attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
  int64_t numExperts = *(attrs->GetInt(numExpertsAttrIdx));
  OPS_CHECK(numExperts <= 0,
           OPS_REPORT_VECTOR_INNER_ERR(
              context->GetNodeName(), "Number of experts should greater than 0!"),
           return GRAPH_FAILED);

  // 获取输出值shape
  gert::Shape* output_y_shape = context->GetOutputShape(INDEX_out);
  OPS_LOG_E_IF_NULL(context, output_y_shape, return ge::GRAPH_FAILED);

  const size_t input_dim_num = 1;
  output_y_shape->SetDimNum(input_dim_num);
  output_y_shape->SetDim(0, numExperts);

  OPS_LOG_D(context->GetNodeName(), "End to do MoeComputeExpertTokensInfershape.");

 return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeComputeExpertTokens).InferShape(InfershapeForMoeComputeExpertTokens);
// -------------------MoeComputeExpertTokens Ops END---------------------
}  // namespace ops