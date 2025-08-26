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
 * \file moe_token_unpermute_grad_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;

namespace ops {
static constexpr size_t INPUT_PERMUTED_TOKENS_IDX = 0;
static constexpr size_t INPUT_UNPERMUTEDOUTPUTD_IDX = 1;
static constexpr size_t INPUT_ROWIDMAP_IDX = 2;
static constexpr size_t INPUT_PROB_IDX = 3;
static constexpr size_t OUTPUT_PERMUTEDTOKENSGRAD_IDX = 0;
static constexpr size_t OUTPUT_PROBGRAD_IDX = 1;
static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_NUM_TWO = 2;
static constexpr size_t DIM_NUM_ONE = 1;
static constexpr size_t OUTPUT_INPUTGRAD_DIMNUM = 2;
static constexpr int64_t UNKNOWN_RANK_DIM_VALUE = -2;

static graphStatus InferShape4MoeTokenUnpermuteGrad(gert::InferShapeContext* context) {
    OPS_LOG_D(context, "Begin to do InferShape4MoeTokenUnpermuteGrad.");
    const gert::Shape* permutedTokensShape = context->GetInputShape(INPUT_PERMUTED_TOKENS_IDX);
    OPS_LOG_E_IF_NULL(context, permutedTokensShape, return ge::GRAPH_FAILED)
    const gert::Shape* unpermutedOutputDShape = context->GetInputShape(INPUT_UNPERMUTEDOUTPUTD_IDX);
    OPS_LOG_E_IF_NULL(context, unpermutedOutputDShape, return ge::GRAPH_FAILED)
    const gert::Shape* rowIdMapShape = context->GetInputShape(INPUT_ROWIDMAP_IDX);
    OPS_LOG_E_IF_NULL(context, rowIdMapShape, return ge::GRAPH_FAILED)
    const gert::Shape* probShape = context->GetOptionalInputShape(INPUT_PROB_IDX);

    gert::Shape* permutedTokensGradShape = context->GetOutputShape(OUTPUT_PERMUTEDTOKENSGRAD_IDX);
    OPS_LOG_E_IF_NULL(context, permutedTokensGradShape, return ge::GRAPH_FAILED)
    gert::Shape* probGradShape = context->GetOutputShape(OUTPUT_PROBGRAD_IDX);
    OPS_LOG_E_IF_NULL(context, probGradShape, return ge::GRAPH_FAILED)

    // permutedTokensGrad
    bool isUnknownRank = permutedTokensShape->GetDimNum() == 1 && permutedTokensShape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE;
    if (isUnknownRank) { // [-2]输入
        OPS_LOG_D(context, "Input shape is -2, set output shape to (-2).");
        permutedTokensGradShape->SetDim(DIM_0, UNKNOWN_RANK_DIM_VALUE);
    } else {
        if (permutedTokensShape->GetDimNum() != DIM_NUM_TWO || unpermutedOutputDShape->GetDimNum() != DIM_NUM_TWO ||
            rowIdMapShape->GetDimNum() != DIM_NUM_ONE) {
            OPS_LOG_E(context, "The dim number of input must be 2, indices must be 1, but got: input %zu, indices %zu.",
                permutedTokensShape->GetDimNum(), rowIdMapShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        uint32_t inputGradDim0 = rowIdMapShape->GetDim(DIM_0);
        uint32_t hiddenSize = unpermutedOutputDShape->GetDim(DIM_1);
        permutedTokensGradShape->SetDimNum(OUTPUT_INPUTGRAD_DIMNUM);
        permutedTokensGradShape->SetDim(DIM_0, inputGradDim0);
        permutedTokensGradShape->SetDim(DIM_1, hiddenSize);
    }
    // probGrad
    if (probShape != nullptr) {
        OPS_LOG_D(context, "InferShape4MoeTokenUnpermuteGrad: probShape is not null.");
        *probGradShape = *probShape;
    } else {
        OPS_LOG_D(context, "InferShape4MoeTokenUnpermuteGrad: probShape is null.");
        probGradShape = nullptr;
    }
    OPS_LOG_D(context, "End to do InferShape4MoeTokenUnpermuteGrad.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeTokenUnpermuteGrad(gert::InferDataTypeContext* context) {
    OPS_LOG_D(context, "Begin to do InferDataType4MoeTokenUnpermuteGrad.");
    context->SetOutputDataType(OUTPUT_PERMUTEDTOKENSGRAD_IDX, context->GetInputDataType(INPUT_PERMUTED_TOKENS_IDX));
    context->SetOutputDataType(OUTPUT_PROBGRAD_IDX, context->GetInputDataType(INPUT_PERMUTED_TOKENS_IDX));
    // 混精场景PROBGRAD dtype与PROB一致，与PERMUTED_TOKENS不一致
    if (context->GetOptionalInputDataType(INPUT_PROB_IDX) != ge::DT_UNDEFINED) {
        context->SetOutputDataType(OUTPUT_PROBGRAD_IDX, context->GetInputDataType(INPUT_PROB_IDX));
    }
    OPS_LOG_D(context, "End to do InferDataType4MoeTokenUnpermuteGrad.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenUnpermuteGrad)
    .InferShape(InferShape4MoeTokenUnpermuteGrad)
    .InferDataType(InferDataType4MoeTokenUnpermuteGrad);
}  // namespace ops
