/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file moe_token_unpermute_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeForMoeTokenUnpermute(gert::InferShapeContext *context)
{
    const gert::Shape *permutedTokensShape = context->GetInputShape(0);
    const gert::Shape *probsShape = context->GetInputShape(2);
    int64_t tokensNum;
    if (probsShape == nullptr) {
        const gert::Shape *indicesShape = context->GetInputShape(1);
        tokensNum = indicesShape->GetDim(0);
    } else {
        tokensNum = probsShape->GetDim(0);
    }

    gert::Shape *outShape = context->GetOutputShape(0);
    outShape->SetDimNum(2);
    outShape->SetDim(0, tokensNum);
    outShape->SetDim(1, permutedTokensShape->GetDim(1));

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMoeTokenUnpermute(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenUnpermute)
    .InferShape(InferShapeForMoeTokenUnpermute)
    .InferDataType(InferDataTypeForMoeTokenUnpermute);
} // namespace ops
