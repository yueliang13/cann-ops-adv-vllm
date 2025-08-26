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
 * \file grouped_bias_add_grad_infer.cc
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "log/ops_log.h"
#include "error/ops_error.h"

using namespace ge;
namespace ops {
static constexpr size_t INDEX_GRAD_Y = 0;
static constexpr size_t INDEX_GROUP_IDX = 1;

static constexpr size_t GRAD_Y_SHAPE_WITH_GROUP_IDX = 2;
static constexpr size_t GRAD_Y_SHAPE_NO_GROUP_IDX = 3;

static constexpr size_t GROUP_INDEX_SHAPE = 1;
static constexpr size_t GRAD_BIAS_SHAPE_SIZE = 2;

static constexpr size_t OTHER_SHAPE = -1;

static constexpr int64_t MAX_GROUP_NUM = 2048;

static ge::graphStatus GroupedBiasAddGradInferShape(gert::InferShapeContext* context) {
    OPS_LOG_D(context->GetNodeName(), "Begin to do GroupedBiasAddGradInferShape.");

    const gert::Shape* gradYShape = context->GetInputShape(INDEX_GRAD_Y);
    OPS_LOG_E_IF_NULL(context, gradYShape, return ge::GRAPH_FAILED);

    auto gradYDimNum = gradYShape->GetDimNum();
    auto groupNum = gradYShape->GetDim(0);
    int64_t hNum = 0;

    if (gradYDimNum == 1 && groupNum == ge::UNKNOWN_DIM_NUM) {
        gert::Shape* gradBiasShape = context->GetOutputShape(0);
        gradBiasShape->SetDimNum(GRAD_BIAS_SHAPE_SIZE);
        gradBiasShape->SetDim(0, OTHER_SHAPE);
        gradBiasShape->SetDim(1, OTHER_SHAPE);
        return ge::GRAPH_SUCCESS;
    }

    const gert::Shape* groupIdxShape = context->GetOptionalInputShape(INDEX_GROUP_IDX);
    if (groupIdxShape == nullptr) {
        OPS_LOG_E_IF(gradYDimNum != GRAD_Y_SHAPE_NO_GROUP_IDX, context, return ge::GRAPH_FAILED,
                "the grad_y of input should be 3D tensor when group_idx is null.");
        hNum = gradYShape->GetDim(GRAD_Y_SHAPE_NO_GROUP_IDX - 1);
    } else {
        OPS_LOG_E_IF(gradYDimNum != GRAD_Y_SHAPE_WITH_GROUP_IDX || groupIdxShape->GetDimNum() != GROUP_INDEX_SHAPE,
                context, return ge::GRAPH_FAILED,
                "the grad_y of input should be 2D tensor when group_idx is not null. and group_idx must be 1D tensor.");
        
        auto groupIdxShapeSize = groupIdxShape->GetDim(0);
        OPS_LOG_E_IF(groupIdxShapeSize != -1 && groupIdxShapeSize > MAX_GROUP_NUM,
                context, return ge::GRAPH_FAILED,
                "the shape size of group_idx can not be larger than 2048, but got %ld.", groupIdxShapeSize);
        
        hNum = gradYShape->GetDim(GRAD_Y_SHAPE_WITH_GROUP_IDX - 1);
        groupNum = groupIdxShapeSize;
    }

    gert::Shape* gradBiasShape = context->GetOutputShape(0);
    gradBiasShape->SetDimNum(GRAD_BIAS_SHAPE_SIZE);
    gradBiasShape->SetDim(0, groupNum);
    gradBiasShape->SetDim(1, hNum);

    OPS_LOG_D(context->GetNodeName(), "End to do GroupedBiasAddGradInferShape.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedBiasAddGrad).InferShape(GroupedBiasAddGradInferShape);
}  // namespace ops
