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
 * \file apply_rotary_pos_emb_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT0 = 0;
static constexpr size_t INPUT1 = 1;

static ge::graphStatus ApplyRotaryPosEmbInferShape(gert::InferShapeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do ApplyRotaryPosEmbInferShape");

    // get input shapes
    const gert::Shape *x_shape = context->GetInputShape(INPUT0);
    OPS_LOG_E_IF_NULL(context, x_shape, return ge::GRAPH_FAILED);
    const gert::Shape *x2_shape = context->GetInputShape(INPUT1);
    OPS_LOG_E_IF_NULL(context, x2_shape, return ge::GRAPH_FAILED);
    // get output shapes
    gert::Shape *y1_shape = context->GetOutputShape(INPUT0);
    OPS_LOG_E_IF_NULL(context, y1_shape, return ge::GRAPH_FAILED);
    gert::Shape *y2_shape = context->GetOutputShape(INPUT1);
    OPS_LOG_E_IF_NULL(context, y2_shape, return ge::GRAPH_FAILED);

    *y1_shape = *x_shape;
    *y2_shape = *x2_shape;

    OPS_LOG_D(context->GetNodeName(), "End to do ApplyRotaryPosEmbInferShape");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyRotaryPosEmb).InferShape(ApplyRotaryPosEmbInferShape);
} // namespace ops
