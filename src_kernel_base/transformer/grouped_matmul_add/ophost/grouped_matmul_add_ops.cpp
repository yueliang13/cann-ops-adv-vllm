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
 * \file grouped_matmul_add_ops.cpp
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "log/ops_log.h"

using namespace ge;
namespace ops
{
static constexpr size_t INDEX_IN_X = 0;
static constexpr size_t INDEX_IN_WEIGHT = 1;
static constexpr size_t INDEX_IN_GROUP_LIST = 2;
static constexpr size_t INDEX_IN_Y = 3;
static constexpr size_t INDEX_OUT_Y = 0;

static constexpr size_t DIM_NUM_1 = 1;
static constexpr size_t DIM_NUM_2 = 2;
static constexpr size_t DIM_O = 0;

static graphStatus InferShapeGroupedMatmulAdd(gert::InferShapeContext* context)
{
    auto yShape = context->GetInputShape(INDEX_IN_Y);
    OPS_LOG_E_IF_NULL(context, yShape, return ge::GRAPH_FAILED);
    auto yRefShape = context->GetOutputShape(INDEX_OUT_Y);
    OPS_LOG_E_IF_NULL(context, yRefShape, return ge::GRAPH_FAILED);

    *yRefShape = *yShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeGroupedMatmulAdd(gert::InferDataTypeContext* context)
{
    auto xDtype = context->GetInputDataType(INDEX_IN_X);
    auto wDtype = context->GetInputDataType(INDEX_IN_WEIGHT);

    OPS_CHECK(xDtype != wDtype, OPS_LOG_E(context, "xDtype must equal to wDtype"),
             return GRAPH_FAILED);

    context->SetOutputDataType(INDEX_OUT_Y, ge::DataType::DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmulAdd)
    .InferShape(InferShapeGroupedMatmulAdd)
    .InferDataType(InferDataTypeGroupedMatmulAdd);
} // namespace ops