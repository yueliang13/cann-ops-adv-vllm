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
 * \file moe_distribute_combine_proto.cc
 * \brief
 */
#include "platform/platform_info.h"
#include "log/ops_log.h"
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "hcom_topo_info.h" 

using namespace ge;
namespace ops {
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;

static constexpr size_t COMBINE_INPUT_EXPERT_X_INDEX = 0;
static constexpr size_t COMBINE_INPUT_EXPERT_IDX_INDEX = 1;
static constexpr size_t COMBINE_OUTPUT_X_INDEX = 0;

static ge::graphStatus InferShapeMoeDistributeCombine(gert::InferShapeContext *context)
{
    const char* name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    OPS_LOG_D(name, "Begin to do InferShapeMoeDistributeCombine.");
    // 获取输入shape
    const gert::Shape *expandXShape = context->GetInputShape(COMBINE_INPUT_EXPERT_X_INDEX);
    OPS_LOG_E_IF_NULL(name, expandXShape, return GRAPH_FAILED);
    const gert::Shape *expandIdsShape = context->GetInputShape(COMBINE_INPUT_EXPERT_IDX_INDEX);
    OPS_LOG_E_IF_NULL(name, expandIdsShape, return GRAPH_FAILED);
    gert::Shape *xShape = context->GetOutputShape(COMBINE_OUTPUT_X_INDEX);
    OPS_LOG_E_IF_NULL(name, xShape, return GRAPH_FAILED);

    int64_t n = expandIdsShape->GetDimNum() == 1U ? NEG_ONE : expandIdsShape->GetDim(0);
    int64_t h = expandXShape->GetDimNum() == 1U ? NEG_ONE : expandXShape->GetDim(1);

    xShape->SetDimNum(DIM_TWO);
    xShape->SetDim(0U, n);
    xShape->SetDim(1U, h);

    OPS_LOG_D(name, "x shape shape is :%s after infershape.",
        Shape2String(*xShape).c_str());
    OPS_LOG_D(name, "End to do InferShapeMoeDistributeCombine.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDistributeCombine(gert::InferDataTypeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeCombine.");
    auto xDtype = context->GetInputDataType(COMBINE_INPUT_EXPERT_X_INDEX);
    context->SetOutputDataType(COMBINE_OUTPUT_X_INDEX, xDtype);
    OPS_LOG_D(context->GetNodeName(), "End to do InferDataTypeMoeDistributeCombine.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDistributeCombine)
    .InferShape(InferShapeMoeDistributeCombine)
    .InferDataType(InferDataTypeMoeDistributeCombine);
}  // namespace ops