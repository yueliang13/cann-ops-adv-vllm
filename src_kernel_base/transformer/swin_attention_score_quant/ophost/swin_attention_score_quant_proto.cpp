/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file swin_attention_score_quant.cc
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "error/ops_error.h"

namespace ops {
static ge::graphStatus InferShapeSwinAttentionScoreQuant(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSwinAttentionScoreQuant(gert::InferDataTypeContext *context)
{
    const ge::DataType x1_data_type = context->GetInputDataType(3);
    ge::graphStatus ret = context->SetOutputDataType(0, x1_data_type);
    return ret;
}

IMPL_OP_INFERSHAPE(SwinAttentionScoreQuant)
    .InferShape(InferShapeSwinAttentionScoreQuant)
    .InferDataType(InferDataTypeSwinAttentionScoreQuant);
}  // namespace ops