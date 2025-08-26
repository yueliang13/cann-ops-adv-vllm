/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ffn_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShapeFFN(gert::InferShapeContext *context)
{
    auto in_shape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL(context, in_shape, return ge::GRAPH_FAILED);
    auto out_shape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, out_shape, return ge::GRAPH_FAILED);
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFFN(gert::InferDataTypeContext *context)
{
    auto input_x_dtype = context->GetInputDataType(0);
    if (input_x_dtype == ge::DT_INT8) {
        auto attrs = context->GetAttrs();
        const int64_t *output_dtype = attrs->GetInt(2);
        if (output_dtype != NULL && *output_dtype == 1) {
            context->SetOutputDataType(0, ge::DT_BF16);
        } else {
            context->SetOutputDataType(0, ge::DT_FLOAT16);
        }
    } else {
        context->SetOutputDataType(0, input_x_dtype);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FFN).InferShape(InferShapeFFN).InferDataType(InferDataTypeFFN);
} // namespace ops
