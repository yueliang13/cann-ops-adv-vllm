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
 * \file flash_attention_score_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;

namespace ops {

constexpr int FLA_SOFTMAXMAX_F32_DIM0SHAPE = 8;
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;

ge::graphStatus InferShapeFlashAttentionScore(gert::InferShapeContext *context)
{
    OPS_LOG_I(context, "Enter FlashAttentionScore runtime infershape impl.");

    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    auto attrs = context->GetAttrs();
    const auto *queryDesc = context->GetInputDesc(0);
    OPS_LOG_E_IF_NULL(context, queryDesc, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED)

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

    int64_t shapeB = 1;
    int64_t shapeS = 1;
    int64_t shapeT = 0;

    if (inputLayoutStr == "BSH" || inputLayoutStr == "BSND") {
        shapeB = queryShape->GetDim(0);
        shapeS = queryShape->GetDim(1);
    } else if (inputLayoutStr == "SBH") {
        shapeB = queryShape->GetDim(1);
        shapeS = queryShape->GetDim(0);
    } else if (inputLayoutStr == "TND") {
        shapeT = queryShape->GetDim(0);
    } else {
        // BNSD
        shapeB = queryShape->GetDim(0);
        // 2: BNSD中S的shape
        shapeS = queryShape->GetDim(DIM_NUM_2);
    }

    auto headNum = attrs->GetInt(4);
    OPS_LOG_E_IF_NULL(context, headNum, return ge::GRAPH_FAILED)
    OPS_LOG_I(context, "B=%ld, N=%ld, T=%ld, S=%ld, inputLayout=%s, dtype=%s", shapeB, *headNum, shapeT, shapeS,
              inputLayoutStr.c_str(), ge::TypeUtils::DataTypeToSerialString(queryDesc->GetDataType()).c_str());

    // softmaxMax, fp32: (B, N, S, 8)
    gert::Shape *softmaxMaxShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, softmaxMaxShape, return ge::GRAPH_FAILED)
    if (inputLayoutStr == "TND") {
        softmaxMaxShape->SetDimNum(DIM_NUM_3);
        softmaxMaxShape->SetDim(0, shapeT);
        softmaxMaxShape->SetDim(1, *headNum);
        softmaxMaxShape->SetDim(DIM_NUM_2, FLA_SOFTMAXMAX_F32_DIM0SHAPE);
    } else {
        // 0, 1, 2, 3, 4 : dim idx
        softmaxMaxShape->SetDimNum(DIM_NUM_4);
        softmaxMaxShape->SetDim(0, shapeB);
        softmaxMaxShape->SetDim(1, *headNum);
        softmaxMaxShape->SetDim(DIM_NUM_2, shapeS);
        softmaxMaxShape->SetDim(DIM_NUM_3, FLA_SOFTMAXMAX_F32_DIM0SHAPE);
    }

    // softmaxSum, shape same as softmaxMax
    gert::Shape *softmaxSumShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL(context, softmaxSumShape, return ge::GRAPH_FAILED)
    *softmaxSumShape = *softmaxMaxShape;

    // softmaxOut, shape: (B, N, S, S)
    gert::Shape *softmaxOutShape = context->GetOutputShape(2);
    OPS_LOG_E_IF_NULL(context, softmaxOutShape, return ge::GRAPH_FAILED)
    // 0, 1, 2, 3, 4 : dim idx
    softmaxOutShape->SetDimNum(DIM_NUM_4);
    softmaxOutShape->SetDim(0, 0);
    softmaxOutShape->SetDim(1, 0);
    softmaxOutShape->SetDim(DIM_NUM_2, 0);
    softmaxOutShape->SetDim(DIM_NUM_3, 0);

    gert::Shape *attentionOutShape = context->GetOutputShape(3);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)
    *attentionOutShape = *queryShape;

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeFlashAttentionScore(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDataType(0);
    // softmax_max, outidx:0
    context->SetOutputDataType(0, DT_FLOAT);
    // softmax_sum, outidx:1
    context->SetOutputDataType(1, DT_FLOAT);
    // softmax_out, outidx:2
    context->SetOutputDataType(2, dtype);
    // attention_out, outidx:3
    context->SetOutputDataType(3, dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP(FlashAttentionScore).InferShape(InferShapeFlashAttentionScore).InferDataType(InferDataTypeFlashAttentionScore);

} // namespace ops
