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
 * \file flash_attention_score_grad_proto.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;

namespace ops {

static const uint64_t DIM_NUM_2 = 2;

ge::graphStatus InferShape4FlashAttentionScoreGrad(gert::InferShapeContext *context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_I(context, "Enter FlashAttentionScoreGrad runtime infershape impl.");
    const gert::Shape *queryShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    const gert::Shape *keyShape = context->GetInputShape(1);
    OPS_LOG_E_IF_NULL(context, keyShape, return ge::GRAPH_FAILED)
    const gert::Shape *valueShape = context->GetInputShape(2);
    OPS_LOG_E_IF_NULL(context, valueShape, return ge::GRAPH_FAILED)

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED)

    auto headNum = attrs->GetInt(4); // N1
    OPS_LOG_E_IF_NULL(context, headNum, return ge::GRAPH_FAILED)
    const char *inputLayout = attrs->GetAttrPointer<char>(5);
    OPS_LOG_E_IF_NULL(context, inputLayout, return ge::GRAPH_FAILED)
    std::string inputLayoutStr = std::string(inputLayout);
    for (auto &c : inputLayoutStr) {
        c = toupper(c);
    }
    if (inputLayoutStr != "BSH" && inputLayoutStr != "SBH" && inputLayoutStr != "BSND" && inputLayoutStr != "BNSD" &&
        inputLayoutStr != "TND") {
        OPS_LOG_E(context, "The inputLayout should be BSH/SBH/BSND/BNSD/TND(case-insensitive), but got %s.",
                  inputLayoutStr.c_str());
        return GRAPH_FAILED;
    }

    int64_t b = 0;
    int64_t sQ = 0;
    int64_t sKv = 0;
    int64_t t = 0;
    if (inputLayoutStr == "BSH" || inputLayoutStr == "BSND") {
        b = queryShape->GetDim(0);
        sQ = queryShape->GetDim(1);
        sKv = keyShape->GetDim(1);
    } else if (inputLayoutStr == "SBH") {
        b = queryShape->GetDim(1);
        sQ = queryShape->GetDim(0);
        sKv = keyShape->GetDim(0);
    } else if (inputLayoutStr == "TND") {
        t = queryShape->GetDim(0);
    } else {
        // BNSD
        b = queryShape->GetDim(0);
        sQ = queryShape->GetDim(DIM_NUM_2);
        sKv = keyShape->GetDim(DIM_NUM_2);
    }
    OPS_LOG_I(context, "B=%ld, N=%ld, T=%ld, Sq=%ld, Skv=%ld, inputLayout=%s", b, *headNum, t, sQ, sKv,
              inputLayoutStr.c_str());

    gert::Shape *dqShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, dqShape, return ge::GRAPH_FAILED)
    *dqShape = *queryShape;

    gert::Shape *dkShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL(context, dkShape, return ge::GRAPH_FAILED)
    *dkShape = *keyShape;

    gert::Shape *dvShape = context->GetOutputShape(2);
    OPS_LOG_E_IF_NULL(context, dvShape, return ge::GRAPH_FAILED)
    *dvShape = *valueShape;

    // dpse output
    gert::Shape *dpseShape = context->GetOutputShape(3);
    OPS_LOG_E_IF_NULL(context, dpseShape, return ge::GRAPH_FAILED)

    const gert::Shape *pseShape = context->GetOptionalInputShape(4);
    if (pseShape != nullptr && pseShape->GetShapeSize() != 0) {
        OPS_LOG_D(context, "pse_shift is not nullptr");
        *dpseShape = *pseShape;
    } else {
        OPS_LOG_D(context, "pse_shift is nullptr");
        dpseShape->SetDimNum(1);
        dpseShape->SetDim(0, 0);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4FlashAttentionScoreGrad(gert::InferDataTypeContext *context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_I(context, "Enter FlashAttentionScoreGrad infer data type impl.");

    auto dtype = context->GetInputDataType(0);
    // dq, outidx:0
    context->SetOutputDataType(0, dtype);
    // dk, outidx:1
    context->SetOutputDataType(1, dtype);
    // dv, outidx:2
    context->SetOutputDataType(2, dtype);
    // dpse, outidx:3
    // 后续针对pse内部生成的场景，Dtype就不能跟随qkv了
    context->SetOutputDataType(3, dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP(FlashAttentionScoreGrad)
    .InferShape(InferShape4FlashAttentionScoreGrad)
    .InferDataType(InferDataType4FlashAttentionScoreGrad);

} // namespace ops
