/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
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
 * \file matmul_reduce_scatter_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "../../common/ophost/op_util.h"
#include "../../common/ophost/hcom_topo_info.h"
#include "log/ops_log.h"

using namespace ge;
namespace ops {
const size_t GROUP = 0;
const size_t IS_TRANS_A = 2;
const size_t IS_TRANS_B = 3;
const size_t RANK_SIZE = 5;
const size_t SUPPORT_DIM_SIZE = 2;

static ge::graphStatus InferShapeMatmulReduceScatter(gert::InferShapeContext* context) {
  const gert::Shape* x1_matrix_shape = context->GetInputShape(0);
  OPS_LOG_E_IF_NULL(context, x1_matrix_shape, return GRAPH_FAILED);
  const gert::Shape* x2_matrix_shape = context->GetInputShape(1);
  OPS_LOG_E_IF_NULL(context, x2_matrix_shape, return GRAPH_FAILED);
  if (x1_matrix_shape->GetDimNum() != SUPPORT_DIM_SIZE || x2_matrix_shape->GetDimNum() != SUPPORT_DIM_SIZE) {
    OPS_LOG_E(context->GetNodeName(), "Input x1 and Input x2 must be the same with 2 dims.");
    return ge::GRAPH_FAILED;
  }
  auto attrs = context->GetAttrs();
  const bool* is_trans_a = attrs->GetAttrPointer<bool>(IS_TRANS_A);
  const bool* is_trans_b = attrs->GetAttrPointer<bool>(IS_TRANS_B);
  const int64_t* rank_size_attr = attrs->GetAttrPointer<int64_t>(RANK_SIZE);

  const char* groupStr = attrs->GetAttrPointer<char>(GROUP);
  OPS_LOG_E_IF(groupStr == nullptr, context, GRAPH_FAILED, "Get group failed.");
  int64_t rank_size = -1;
  if (*rank_size_attr <= 0) {
    if ((HcomTopoInfo::Instance().GetGroupRankSize(groupStr, rank_size)) != ge::GRAPH_SUCCESS || rank_size <= 0) {
      OPS_LOG_E(context->GetNodeName(), "Get rank size failed, group [%s], rank_size [%ld]", groupStr, rank_size);
      return ge::GRAPH_FAILED;
    }
  } else {
    rank_size = *rank_size_attr;
  }

  int64_t dim_M = !(*is_trans_a) ? x1_matrix_shape->GetDim(0) : x1_matrix_shape->GetDim(1);
  int64_t dim_K_x1 = !(*is_trans_a) ? x1_matrix_shape->GetDim(1) : x1_matrix_shape->GetDim(0);
  int64_t dim_K_x2 = !(*is_trans_b) ? x2_matrix_shape->GetDim(0) : x2_matrix_shape->GetDim(1);
  int64_t dim_N = !(*is_trans_b) ? x2_matrix_shape->GetDim(1) : x2_matrix_shape->GetDim(0);

  OPS_LOG_I(context->GetNodeName(),
          "InferReduceScatter, group = %s isTransA %d isTransB %d x1.M = [%ld] x1.K = [%ld] x2.K = [%ld] x2.M = [%ld], rankSize = [%ld].",
          groupStr, (*is_trans_a), (*is_trans_b), x1_matrix_shape->GetDim(0), x1_matrix_shape->GetDim(1),
          x2_matrix_shape->GetDim(0), x2_matrix_shape->GetDim(1), rank_size);
  if (dim_K_x1 != dim_K_x2) {
    OPS_LOG_E(context->GetNodeName(), "Input x1/x2 dim k must be same, but given x1.k %ld, x2.k %ld.", dim_K_x1, dim_K_x2);
    return ge::GRAPH_FAILED;
  }

  if (dim_M % rank_size != 0 && dim_M != -1) {
    OPS_LOG_E(context->GetNodeName(), "Dim M [%ld] cant divide exactly by rank_size [%ld]", dim_M, rank_size);
    return ge::GRAPH_FAILED;
  }
  // 动态shape入图时 m轴-1时，不再进行(dim_M / rank_size)的处理
  if (dim_M == -1) {
    rank_size = 1;
  }
  if (dim_K_x1 == 0) {
    dim_M = dim_N = 0;
    OPS_LOG_E(context->GetNodeName(), "X1/X2 are empty tensors with zero dim_K");
    return ge::GRAPH_FAILED;
  }
  gert::Shape* y_shape = context->GetOutputShape(0);
  OPS_LOG_E_IF_NULL(context, y_shape, return GRAPH_FAILED);
  y_shape->SetDimNum(SUPPORT_DIM_SIZE);
  y_shape->SetDim(0, dim_M / rank_size);
  y_shape->SetDim(1, dim_N);
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMatmulReduceScatter(gert::InferDataTypeContext* context) {
  auto d_type = context->GetInputDataType(0);
  context->SetOutputDataType(0, d_type);
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatmulReduceScatter).InferShape(InferShapeMatmulReduceScatter).InferDataType(InferDataTypeMatmulReduceScatter);
}  // namespace ops
