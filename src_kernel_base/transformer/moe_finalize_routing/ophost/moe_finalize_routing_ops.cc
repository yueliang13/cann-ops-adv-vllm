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
 * \file moe_finalize_routing_ops.cc
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "log/ops_log.h"
#include "error/ops_error.h"

using namespace ge;
namespace ops {
static const size_t INDEX_IN_EXPAND_PERMUTED_ROWS = 0;
static const size_t INDEX_IN_SKIP1 = 1;
static const size_t INDEX_IN_SKIP2 = 2;
static const size_t INDEX_IN_BIAS = 3;
static const size_t INDEX_IN_SCALES = 4;
static const size_t INDEX_IN_EXPANDED_SRC_TO_DST_ROW = 5;
static const size_t INDEX_IN_EXPERT_FOR_SOURCE_ROW = 6;
static constexpr size_t INDEX_out = 0;
static constexpr size_t SHAPE_SIZE = 2;
static constexpr size_t INPUT_NUM = 7;

static inline bool IsValidType(const DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

static ge::graphStatus InferDataTypeMoeFinalizeRouting(gert::InferDataTypeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do MoeFinalizeRoutingInferDataType.");
    OPS_LOG_E_IF(!IsValidType(context->GetInputDataType(INDEX_IN_EXPAND_PERMUTED_ROWS)), context,
                 return ge::GRAPH_FAILED, 
                 "the dtype of expanded_permuted_rows should be float, float16 or bf16.");
    OPS_LOG_E_IF(!IsValidType(context->GetInputDataType(INDEX_IN_SKIP1)), context, return ge::GRAPH_FAILED,
                 "the dtype of skip1 should be float, float16 or bf16.");
    const DataType skip2Dtype = context->GetOptionalInputDataType(INDEX_IN_SKIP2);
    OPS_LOG_E_IF(skip2Dtype != ge::DT_UNDEFINED && !IsValidType(skip2Dtype), context, return ge::GRAPH_FAILED,
                  "the dtype of skip2 should be float, float16 or bf16.");
    size_t offset = (context->GetComputeNodeInputNum() == INPUT_NUM) ? 0 : 1;
    OPS_LOG_E_IF(!IsValidType(context->GetInputDataType(INDEX_IN_BIAS - offset)), context, return ge::GRAPH_FAILED,
                  "the dtype of bias should be float, float16 or bf16.");
    OPS_LOG_E_IF(!IsValidType(context->GetInputDataType(INDEX_IN_SCALES - offset)), context, return ge::GRAPH_FAILED,
                  "the dtype of scales should be float, float16 or bf16.");
    OPS_LOG_E_IF(context->GetInputDataType(INDEX_IN_EXPANDED_SRC_TO_DST_ROW - offset) != ge::DT_INT32, context,
                 return ge::GRAPH_FAILED, "the dtype of expanded_src_to_dst_row should be int32.");
    OPS_LOG_E_IF(context->GetInputDataType(INDEX_IN_EXPERT_FOR_SOURCE_ROW - offset) != ge::DT_INT32, context,
                 return ge::GRAPH_FAILED, "the dtype of expert_for_source_row should be int32.");
    context->SetOutputDataType(0, context->GetInputDataType(INDEX_IN_SKIP1));
    return ge::GRAPH_SUCCESS;
}

static bool IsValidShape(const int64_t shape1, const int64_t shape2, const int64_t shape3, const int64_t shape4)
{
    std::vector<int64_t> validValue = {-1};
    std::vector<int64_t> currentValue = {shape1, shape2, shape3, shape4};
    for (auto value : currentValue) {
        if (value == -1) {
            continue;
        }
        if (validValue.size() == 1) {
            validValue.push_back(value);
        } else if (validValue[1] != value) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus MoeCopyShapeInput2OutputWithIdx(gert::InferShapeContext *context, int64_t input_idx,
                                                       int64_t output_idx)
{
    auto in_shape = context->GetInputShape(input_idx);
    OPS_LOG_E_IF_NULL(context, in_shape, return ge::GRAPH_FAILED);
    auto out_shape = context->GetOutputShape(output_idx);
    OPS_LOG_E_IF_NULL(context, out_shape, return ge::GRAPH_FAILED);
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Infershape4MoeFinalizeRouting(gert::InferShapeContext *context)
{
    // get and check input param
    const gert::Shape *expandedPermutedRowsInputShape = context->GetInputShape(INDEX_IN_EXPAND_PERMUTED_ROWS);
    OPS_LOG_E_IF_NULL(context, expandedPermutedRowsInputShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF(expandedPermutedRowsInputShape->GetDimNum() != SHAPE_SIZE, context, return ge::GRAPH_FAILED,
                "the expanded_permuted_rows of input should be 2D tensor.");

    const gert::Shape *skip1InputShape = context->GetInputShape(INDEX_IN_SKIP1);
    OPS_LOG_E_IF_NULL(context, skip1InputShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF(skip1InputShape->GetDimNum() != SHAPE_SIZE, context, return ge::GRAPH_FAILED,
                 "the skip1 of input should be 2D tensor.");

    const gert::Tensor *x2Tensor = context->GetOptionalInputTensor(INDEX_IN_SKIP2);
    const gert::Shape *skip2InputShape = (x2Tensor == nullptr || x2Tensor->GetShapeSize() == 0) ?
                                             nullptr :
                                             context->GetOptionalInputShape(INDEX_IN_SKIP2);
    OPS_LOG_E_IF(skip2InputShape != nullptr && skip2InputShape->GetDimNum() != SHAPE_SIZE, context,
                 return ge::GRAPH_FAILED, "the skip2 of input should be 2D tensor.");

    size_t offset = (context->GetComputeNodeInputNum() == INPUT_NUM) ? 0 : 1;
    const gert::Shape *biasInputShape = context->GetInputShape(INDEX_IN_BIAS - offset);
    OPS_LOG_E_IF_NULL(context, biasInputShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF(biasInputShape->GetDimNum() != SHAPE_SIZE, context, return ge::GRAPH_FAILED,
                 "the bias of input should be 2D tensor.");

    const gert::Shape *scalesInputShape = context->GetInputShape(INDEX_IN_SCALES - offset);
    OPS_LOG_E_IF_NULL(context, scalesInputShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF(scalesInputShape->GetDimNum() != SHAPE_SIZE, context, return ge::GRAPH_FAILED,
                 "the scales of input should be 2D tensor.");

    const gert::Shape *expandedSrcToDstRowInputShape =
        context->GetInputShape(INDEX_IN_EXPANDED_SRC_TO_DST_ROW - offset);
    OPS_LOG_E_IF_NULL(context, expandedSrcToDstRowInputShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF(expandedSrcToDstRowInputShape->GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "the expanded_src_to_dst_row of input should be 1D tensor.");

    const gert::Shape *expertForSourceRowInputShape = context->GetInputShape(INDEX_IN_EXPERT_FOR_SOURCE_ROW - offset);
    OPS_LOG_E_IF_NULL(context, expertForSourceRowInputShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF(expertForSourceRowInputShape->GetDimNum() != SHAPE_SIZE, context, return ge::GRAPH_FAILED,
                 "the expert_for_source_row of input should be 2D tensor.");

    bool validColK = scalesInputShape->GetDim(1) == -1 || expertForSourceRowInputShape->GetDim(1) == -1 ||
                     scalesInputShape->GetDim(1) == expertForSourceRowInputShape->GetDim(1);
    OPS_LOG_E_IF(!validColK, context, return ge::GRAPH_FAILED,
                 "the dim 1 of scales and expert_for_source_row should be same.");

    int64_t skip2Row = skip2InputShape != nullptr ? skip2InputShape->GetDim(0) : -1;
    OPS_LOG_E_IF(!IsValidShape(skip1InputShape->GetDim(0), skip2Row, scalesInputShape->GetDim(0),
                               expertForSourceRowInputShape->GetDim(0)),
                 context, return ge::GRAPH_FAILED, 
                 "the dim 0 of skip1, skip2, scales and expert_for_source_row should be same.");

    int64_t skip2Col = skip2InputShape != nullptr ? skip2InputShape->GetDim(1) : -1;
    OPS_LOG_E_IF(!IsValidShape(skip1InputShape->GetDim(1), skip2Col, expandedPermutedRowsInputShape->GetDim(1),
                               biasInputShape->GetDim(1)),
                 context, return ge::GRAPH_FAILED,
                 "the dim 1 of skip1, skip2, expanded_permuted_rows and bias should be same.");

    bool validDim = expandedSrcToDstRowInputShape->GetDim(0) == -1 || expandedPermutedRowsInputShape->GetDim(0) == -1 ||
                    expandedSrcToDstRowInputShape->GetDim(0) == expandedPermutedRowsInputShape->GetDim(0);
    OPS_LOG_E_IF(!validDim, context, return ge::GRAPH_FAILED,
                 "the dim 0 of expanded_permuted_rows and expanded_src_to_dst_row should be same.");
    // infershape output
    OPS_CHECK(MoeCopyShapeInput2OutputWithIdx(context, INDEX_IN_SKIP1, INDEX_out) != ge::GRAPH_SUCCESS,
              OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Infershape4MoeFinalizeRouting failed!"),
              return ge::GRAPH_FAILED);

    OPS_LOG_D(context->GetNodeName(), "End to do MoeFinalizeRoutingInfershape.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeFinalizeRouting)
    .InferShape(Infershape4MoeFinalizeRouting)
    .InferDataType(InferDataTypeMoeFinalizeRouting);
} // namespace ops