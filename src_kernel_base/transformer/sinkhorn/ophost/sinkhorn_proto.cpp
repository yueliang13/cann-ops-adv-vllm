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
 * \file sinkhorn_proto.cc
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;
static constexpr size_t INPUT_COST_INDEX = 0;
static constexpr size_t OUTPUT_P_INDEX = 0;

namespace ops {
static ge::graphStatus InferShapeForSinkhorn(gert::InferShapeContext *context)
{
    OPS_LOG_D(context, "Begin to do InferShapeForSinkhorn.");
    const gert::Shape *costShape = context->GetInputShape(INPUT_COST_INDEX);
    OPS_LOG_E_IF_NULL(context, costShape, return ge::GRAPH_FAILED)
    gert::Shape *pShape = context->GetOutputShape(OUTPUT_P_INDEX);
    OPS_LOG_E_IF_NULL(context, pShape, return ge::GRAPH_FAILED);

    *pShape = *costShape;

    OPS_LOG_D(context, "End to do InferShapeForSinkhorn.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForSinkhorn(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(OUTPUT_P_INDEX, context->GetInputDataType(INPUT_COST_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Sinkhorn)
    .InferShape(InferShapeForSinkhorn)
    .InferDataType(InferDataTypeForSinkhorn);
} // namespace ops