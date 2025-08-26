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
 * \file moe_init_routing_quant_ops.cc
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "log/ops_log.h"
#include "error/ops_error.h"
namespace {
#define OPS_CHECK_NULL_WITH_CONTEXT(OPS_DESC, PTR) OPS_LOG_E_IF_NULL(OPS_DESC, PTR, return ge::GRAPH_FAILED)
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t EXPENDED_X_IDX = 0;
static constexpr int64_t EXPENDED_ROW_IDX = 1;
static constexpr int64_t EXPENDED_EXPERT_IDX = 2;

static bool isSameDim(int64_t dim1, int64_t dim2)
{
    if (dim1 == NEG_ONE || dim2 == NEG_ONE) {
        return true;
    }
    return dim1 == dim2;
}
}
using namespace ge;
namespace ops {
static ge::graphStatus CheckInputShape(gert::InferShapeContext *context, const gert::Shape *xShape,
    const gert::Shape *rowIdxShape, const gert::Shape *expertIdxShape)
{
    int64_t x_n = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(1);
    if (x_n < NEG_ONE || cols < NEG_ONE) {
        OPS_LOG_E(context->GetNodeName(), "Invalid x shape, shape is %s.", Shape2String(*xShape).c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t row_idx_n = rowIdxShape->GetDimNum() == 1U ? NEG_ONE : rowIdxShape->GetDim(0);
    int64_t row_idx_k = rowIdxShape->GetDimNum() == 1U ? NEG_ONE : rowIdxShape->GetDim(1);
    if (row_idx_n < NEG_ONE || row_idx_k < NEG_ONE) {
        OPS_LOG_E(context->GetNodeName(), "Invalid row_idx shape, shape is %s.", Shape2String(*rowIdxShape).c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t expert_idx_n = expertIdxShape->GetDimNum() == 1U ? NEG_ONE : expertIdxShape->GetDim(0);
    int64_t expert_idx_k = expertIdxShape->GetDimNum() == 1U ? NEG_ONE : expertIdxShape->GetDim(1);
    if (expert_idx_n < NEG_ONE || expert_idx_k < NEG_ONE) {
        OPS_LOG_E(context->GetNodeName(), "Invalid expert_idx shape, shape is %s.",
            Shape2String(*expertIdxShape).c_str());
        return ge::GRAPH_FAILED;
    }

    if (!isSameDim(x_n, row_idx_n) || !isSameDim(x_n, expert_idx_n) || !isSameDim(row_idx_n, expert_idx_n)) {
        OPS_LOG_E(context->GetNodeName(), "The first dim of x, row_idx and expert_idx should be same.");
        return ge::GRAPH_FAILED;
    }

    if (!isSameDim(row_idx_k, expert_idx_k)) {
        OPS_LOG_E(context->GetNodeName(), "The second dim of row_idx and expert_idx should be same.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDimsAndAttr(gert::InferShapeContext *context, const gert::Shape *xShape,
    const gert::Shape *rowIdxShape, const gert::Shape *expertIdxShape, const int64_t activeNum)
{
    if (xShape->GetDimNum() == 1U) {
        if (xShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OPS_LOG_E(context->GetNodeName(), "The dynamic dim of x should be -2, current shape is %s.",
                Shape2String(*xShape).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (xShape->GetDimNum() != DIM_TWO) {
        OPS_LOG_E(context->GetNodeName(), "The dim of x should be 2 or dynamic, current shape is %s.",
            Shape2String(*xShape).c_str());
        return ge::GRAPH_FAILED;
    }

    if (rowIdxShape->GetDimNum() == 1U) {
        if (rowIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OPS_LOG_E(context->GetNodeName(), "The dynamic dim of row_idx should be -2, current shape is %s.",
                Shape2String(*rowIdxShape).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (rowIdxShape->GetDimNum() != DIM_TWO) {
        OPS_LOG_E(context->GetNodeName(), "The dim of row_idx should be 2 or dynamic, current shape is %s.",
            Shape2String(*rowIdxShape).c_str());
        return ge::GRAPH_FAILED;
    }

    if (expertIdxShape->GetDimNum() == 1U) {
        if (expertIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OPS_LOG_E(context->GetNodeName(), "The dynamic dim of expert_idx should be -2, current shape is %s.",
                Shape2String(*expertIdxShape).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (expertIdxShape->GetDimNum() != DIM_TWO) {
        OPS_LOG_E(context->GetNodeName(), "The dim of expert_idx should be 2 or dynamic, current shape is %s.",
            Shape2String(*expertIdxShape).c_str());
        return ge::GRAPH_FAILED;
    }

    if (activeNum < 0) {
        OPS_LOG_E(context->GetNodeName(), "active_num must be a non-negative number.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static void ShowInputShapeInfo(gert::InferShapeContext *context, const gert::Shape *xShape, const gert::Shape *rowIdxShape,
    const gert::Shape *expertIdxShape, const int64_t activeNum)
{
    OPS_LOG_D(context->GetNodeName(), "x shape is: %s.", Shape2String(*xShape).c_str());
    OPS_LOG_D(context->GetNodeName(), "row_idx shape is: %s.", Shape2String(*rowIdxShape).c_str());
    OPS_LOG_D(context->GetNodeName(), "expert_idx shape is: %s.", Shape2String(*expertIdxShape).c_str());
    OPS_LOG_D(context->GetNodeName(), "activeNum is: %ld.", activeNum);
}

static void ShowOutputShapeInfo(gert::InferShapeContext *context, const gert::Shape *expandedXShape,
    const gert::Shape *expandedRowIdx, const gert::Shape *expandedExpertIdxShape)
{
    OPS_LOG_D(context->GetNodeName(), "expanded_x shape is: %s after infershape.",
        Shape2String(*expandedXShape).c_str());
    OPS_LOG_D(context->GetNodeName(), "expanded_row_idx shape is: %s after infershape.",
        Shape2String(*expandedRowIdx).c_str());
    OPS_LOG_D(context->GetNodeName(), "expanded_expert_idx shape is: %s after infershape.",
        Shape2String(*expandedExpertIdxShape).c_str());
}

static ge::graphStatus InferShape4MoeInitRoutingQuant(gert::InferShapeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do MoeInitRoutingQuantInfershape.");
    // 获取输入shape
    const gert::Shape *xShape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *rowIdxShape = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, rowIdxShape);
    const gert::Shape *expertIdxShape = context->GetInputShape(2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdxShape);
    gert::Shape *expandedXShape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandedXShape);
    gert::Shape *expandedRowIdx = context->GetOutputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandedRowIdx);
    gert::Shape *expandedExpertIdxShape = context->GetOutputShape(2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandedExpertIdxShape);
    // 获取attr
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *activeNumPtr = attrs->GetAttrPointer<int64_t>(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, activeNumPtr);
    const int64_t activeNum = *activeNumPtr;
    ShowInputShapeInfo(context, xShape, rowIdxShape, expertIdxShape, activeNum);

    // 参数校验
    if (CheckInputDimsAndAttr(context, xShape, rowIdxShape, expertIdxShape, activeNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInputShape(context, xShape, rowIdxShape, expertIdxShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t x_n = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(1);

    int64_t row_idx_n = rowIdxShape->GetDimNum() == 1U ? NEG_ONE : rowIdxShape->GetDim(0);
    int64_t row_idx_k = rowIdxShape->GetDimNum() == 1U ? NEG_ONE : rowIdxShape->GetDim(1);

    int64_t expert_idx_n = expertIdxShape->GetDimNum() == 1U ? NEG_ONE : expertIdxShape->GetDim(0);
    int64_t expert_idx_k = expertIdxShape->GetDimNum() == 1U ? NEG_ONE : expertIdxShape->GetDim(1);

    int64_t n = x_n > row_idx_n ? (x_n > expert_idx_n ? x_n : expert_idx_n) :
                                  (row_idx_n > expert_idx_n ? row_idx_n : expert_idx_n);
    int64_t k = std::max(row_idx_k, expert_idx_k);
    int64_t outActiveNum = (n == NEG_ONE || k == NEG_ONE) ? NEG_ONE : (std::min(n, activeNum) * k);
    int64_t expertForSourceRowNum = (n == NEG_ONE || k == NEG_ONE) ? NEG_ONE : (n * k);

    expandedXShape->SetDimNum(DIM_TWO);
    expandedXShape->SetDim(0U, outActiveNum);
    expandedXShape->SetDim(1U, cols);

    expandedRowIdx->SetDimNum(DIM_ONE);
    expandedRowIdx->SetDim(0U, expertForSourceRowNum);

    expandedExpertIdxShape->SetDimNum(DIM_ONE);
    expandedExpertIdxShape->SetDim(0U, expertForSourceRowNum);

    ShowOutputShapeInfo(context, expandedXShape, expandedRowIdx, expandedExpertIdxShape);
    OPS_LOG_D(context->GetNodeName(), "End to do MoeInitRoutingQuantInfershape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeInitRoutingQant(gert::InferDataTypeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do MoeInitRoutingQuntInferDataType.");
    context->SetOutputDataType(EXPENDED_X_IDX, ge::DT_INT8);
    context->SetOutputDataType(EXPENDED_ROW_IDX, ge::DT_INT32);
    context->SetOutputDataType(EXPENDED_EXPERT_IDX, ge::DT_INT32);
    OPS_LOG_D(context->GetNodeName(), "End to do MoeInitRoutingQuntInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeInitRoutingQuant)
    .InferShape(InferShape4MoeInitRoutingQuant)
    .InferDataType(InferDataType4MoeInitRoutingQant);
} // namespace ops
