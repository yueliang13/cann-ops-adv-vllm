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
 * \file context_transfer.cpp
 * \brief
 */
#ifndef _CONTEXT_TRANSFER_CC_
#define _CONTEXT_TRANSFER_CC_

#include "context_transfer.h"
#include "op_mc2.h"

namespace optiling {
ge::graphStatus ContextTransfer::AssembleMMRCtxInfoFromMRNCtx(const gert::TilingContext *const context,
                                                              MMRCtxInfo &mmrCtxInfo)
{
    GE_ASSERT_NOTNULL(context);
    GE_ASSERT_NOTNULL(context->GetAttrs());
    // attr
    uint32_t index = 0U;
    mmrCtxInfo.group = context->GetAttrs()->GetAttrPointer<char>(index++);
    mmrCtxInfo.reduceOp = context->GetAttrs()->GetAttrPointer<char>(index++);
    mmrCtxInfo.isTransA = context->GetAttrs()->GetAttrPointer<bool>(index++);
    mmrCtxInfo.isTransB = context->GetAttrs()->GetAttrPointer<bool>(index++);
    mmrCtxInfo.commTurn = *context->GetAttrs()->GetAttrPointer<int>(index++);
    if (context->GetAttrs()->GetAttrNum() > index) {
        mmrCtxInfo.groupSizePtr = context->GetAttrs()->GetAttrPointer<int64_t>(index++);
    }
    // io tensordesc
    index = 0U;
    mmrCtxInfo.x1 = context->GetInputDesc(index++);
    GE_ASSERT_NOTNULL(mmrCtxInfo.x1);
    mmrCtxInfo.x2 = context->GetInputDesc(index++);
    GE_ASSERT_NOTNULL(mmrCtxInfo.x2);
    mmrCtxInfo.bias = context->GetOptionalInputDesc(index++);
    mmrCtxInfo.x3 = nullptr; // mrn当前不支持融合带有x3的mmr
    index++; // 跳过mrn中的arn的residual
    index++; // 跳过mrn中的arn的gamma
    mmrCtxInfo.antiquant_scale = context->GetOptionalInputDesc(index++);
    mmrCtxInfo.antiquant_offset = context->GetOptionalInputDesc(index++);
    mmrCtxInfo.dequant_scale = context->GetOptionalInputDesc(index++);
    index = 0U;
    mmrCtxInfo.y = nullptr; // 输出体现在arn中
    // io shape
    index = 0U;
    mmrCtxInfo.x1_shape = context->GetInputShape(index++);
    GE_ASSERT_NOTNULL(mmrCtxInfo.x1_shape);
    mmrCtxInfo.x2_shape = context->GetInputShape(index++);
    GE_ASSERT_NOTNULL(mmrCtxInfo.x2_shape);
    mmrCtxInfo.bias_shape = context->GetOptionalInputShape(index++);
    mmrCtxInfo.x3_shape = nullptr; // mrn当前不支持融合带有x3的mmr
    index++; // 跳过mrn中的arn的residual
    index++; // 跳过mrn中的arn的gamma
    mmrCtxInfo.antiquant_scale_shape = context->GetOptionalInputShape(index++);
    mmrCtxInfo.antiquant_offset_shape = context->GetOptionalInputShape(index++);
    mmrCtxInfo.dequant_scale_shape = context->GetOptionalInputShape(index++);
    index = 0U;
    mmrCtxInfo.y_shape = nullptr;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus ContextTransfer::AssembleARNCtxInfoFromMRNCtx(const gert::TilingContext *const context,
                                                              ARNCtxInfo &arnCtxInfo)
{
    GE_ASSERT_NOTNULL(context);

    const auto attrs = context->GetAttrs();
    GE_ASSERT_NOTNULL(attrs);
    arnCtxInfo.epsilon = attrs->GetAttrPointer<float>(static_cast<size_t>(ops::MmAllReduceAttrIdx::K_EPSILON));

    // x1在mrn的ctx中不体现，因为x1被融合没了
    arnCtxInfo.x1 = nullptr;
    arnCtxInfo.x1_shape = nullptr;

    uint32_t index = 3U; // 前面3个是mrn中的mmr的x1,x2,bias
    const size_t real_in_total = context->GetComputeNodeInfo()->GetInputsNum();
    const size_t ir_in_total = context->GetComputeNodeInfo()->GetIrInputsNum();
    if (context->GetOptionalInputShape(index - 1) == nullptr && real_in_total != ir_in_total) {
        index -= 1;
    }
    OPS_LOG_D(context->GetNodeName(), "Real input num %zu, total ir input num %zu, x2 input index %u.",
            real_in_total, ir_in_total, index);
    arnCtxInfo.x2 = context->GetInputDesc(index);
    GE_ASSERT_NOTNULL(arnCtxInfo.x2);
    arnCtxInfo.x2_shape = context->GetInputShape(index);
    GE_ASSERT_NOTNULL(arnCtxInfo.x2_shape);

    ++index;
    arnCtxInfo.gamma = context->GetInputDesc(index);
    GE_ASSERT_NOTNULL(arnCtxInfo.gamma);
    arnCtxInfo.gamma_shape = context->GetInputShape(index);
    GE_ASSERT_NOTNULL(arnCtxInfo.gamma_shape);

    index = 0U;
    arnCtxInfo.x = context->GetOutputDesc(index);
    GE_ASSERT_NOTNULL(arnCtxInfo.x);
    arnCtxInfo.x_shape = context->GetOutputShape(index);
    GE_ASSERT_NOTNULL(arnCtxInfo.x_shape);

    // mrn当前不支持融合带有rstd的arn
    arnCtxInfo.rstd = nullptr;
    arnCtxInfo.rstd_shape = nullptr;

    ++index;
    arnCtxInfo.y = context->GetOutputDesc(index);
    GE_ASSERT_NOTNULL(arnCtxInfo.y);
    arnCtxInfo.y_shape = context->GetOutputShape(index);
    GE_ASSERT_NOTNULL(arnCtxInfo.y_shape);

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus ContextTransfer::AssembleMMRCtxInfoFromIMRNCtx(const gert::TilingContext *const context,
                                                               MMRCtxInfo &mmrCtxInfo)
{
    // Inplace的场景，mmr的解析跟非inplace是一样的，区别体现在arn的解析
    return AssembleMMRCtxInfoFromMRNCtx(context, mmrCtxInfo);
}
ge::graphStatus ContextTransfer::AssembleARNCtxInfoFromIMRNCtx(const gert::TilingContext *const context,
                                                               ARNCtxInfo &arnCtxInfo)
{
    // Inplace的场景，arn的原型个数跟inplace arn一样，排列也一样，只是名字不一样
    // 暂时按照认为arnCtxInfo等价于inplace arnCtxInfo
    return AssembleARNCtxInfoFromMRNCtx(context, arnCtxInfo);
}
ge::graphStatus ContextTransfer::AssembleMRNCtxInfoFromMRNCtx(const gert::TilingContext *const context,
                                                              MRNCtxInfo &mrnCtxInfo)
{
    AssembleMMRCtxInfoFromMRNCtx(context, mrnCtxInfo.mmrCtxInfo);
    AssembleARNCtxInfoFromMRNCtx(context, mrnCtxInfo.arnCtxInfo);
    mrnCtxInfo.mmrCtxInfo.y_shape = mrnCtxInfo.arnCtxInfo.x2_shape;
    mrnCtxInfo.mmrCtxInfo.y = mrnCtxInfo.arnCtxInfo.x2;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus ContextTransfer::CheckMRNCtxInfo(const gert::TilingContext *context, const MRNCtxInfo &mrnCtxInfo)
{
    // x1 和residual的b*s
    const gert::StorageShape* x1Shape = mrnCtxInfo.mmrCtxInfo.x1_shape;
    const gert::StorageShape* residualShape = mrnCtxInfo.arnCtxInfo.x2_shape;
    uint64_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    OPS_LOG_D(context->GetNodeName(), "the dim of x1 is %lu.", x1DimNum);
    OP_TILING_CHECK(x1DimNum < DIM_ONE,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Expect x1 dim to be more than 0, but got x1 dim [%lu].",
                                                    x1DimNum),
                    return ge::GRAPH_FAILED);
    uint64_t x1MValue = x1Shape->GetStorageShape().GetDim(0);
    if (x1DimNum >= DIM_THREE) {
        x1MValue *= x1Shape->GetStorageShape().GetDim(1);
    }
    OP_TILING_CHECK(residualShape->GetStorageShape().GetDimNum() != DIM_THREE,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Expect dim of residual from arn to be 3, but got"
                                                    " residual_dim:[%lu].",
                                                    residualShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    uint64_t residualMValue = residualShape->GetStorageShape().GetDim(0) * residualShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(x1MValue != residualMValue,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Expect b * s of x1 (when dim of x1 is 2, b = 1 as default) and"
                                                    " residual to be same, but got x1_b*s:[%lu], residual_b*s:[%lu].",
                                                    x1MValue, residualMValue),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus ContextTransfer::AssembleIMRNCtxInfoFromIMRNCtx(const gert::TilingContext *const context,
                                                                IMRNCtxInfo &imrnCtxInfo)
{
    AssembleMMRCtxInfoFromIMRNCtx(context, imrnCtxInfo.mmrCtxInfo);
    AssembleARNCtxInfoFromIMRNCtx(context, imrnCtxInfo.arnCtxInfo);
    imrnCtxInfo.mmrCtxInfo.y_shape = imrnCtxInfo.arnCtxInfo.x2_shape;
    imrnCtxInfo.mmrCtxInfo.y = imrnCtxInfo.arnCtxInfo.x2;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus ContextTransfer::AssembleMMRCtxInfoFromMMRCtx(const gert::TilingContext *const context,
                                                              MMRCtxInfo &mmrCtxInfo)
{
    GE_ASSERT_NOTNULL(context);
    GE_ASSERT_NOTNULL(context->GetAttrs());
    // attr
    uint32_t index = 0U;
    mmrCtxInfo.group = context->GetAttrs()->GetAttrPointer<char>(index++);
    mmrCtxInfo.reduceOp = context->GetAttrs()->GetAttrPointer<char>(index++);
    mmrCtxInfo.isTransA = context->GetAttrs()->GetAttrPointer<bool>(index++);
    mmrCtxInfo.isTransB = context->GetAttrs()->GetAttrPointer<bool>(index++);
    mmrCtxInfo.commTurn = *context->GetAttrs()->GetAttrPointer<int>(index++);
    if (context->GetAttrs()->GetAttrNum() > index) {
        mmrCtxInfo.groupSizePtr = context->GetAttrs()->GetAttrPointer<int64_t>(index++);
    }
    // io tensordesc
    index = 0U;
    mmrCtxInfo.x1 = context->GetInputDesc(index++);
    GE_ASSERT_NOTNULL(mmrCtxInfo.x1);
    mmrCtxInfo.x2 = context->GetInputDesc(index++);
    GE_ASSERT_NOTNULL(mmrCtxInfo.x2);
    mmrCtxInfo.bias = context->GetOptionalInputDesc(index++);
    mmrCtxInfo.x3 = context->GetOptionalInputDesc(index++);
    mmrCtxInfo.antiquant_scale = context->GetOptionalInputDesc(index++);
    mmrCtxInfo.antiquant_offset = context->GetOptionalInputDesc(index++);
    mmrCtxInfo.dequant_scale = context->GetOptionalInputDesc(index++);
    mmrCtxInfo.pertoken_scale = context->GetOptionalInputDesc(index++);
    mmrCtxInfo.comm_quant_scale_1 = context->GetOptionalInputDesc(index++);
    mmrCtxInfo.comm_quant_scale_2 = context->GetOptionalInputDesc(index++);
    index = 0U;
    mmrCtxInfo.y = context->GetOutputDesc(index++);
    GE_ASSERT_NOTNULL(mmrCtxInfo.y);
    // io shape
    index = 0U;
    mmrCtxInfo.x1_shape = context->GetInputShape(index++);
    GE_ASSERT_NOTNULL(mmrCtxInfo.x1_shape);
    mmrCtxInfo.x2_shape = context->GetInputShape(index++);
    GE_ASSERT_NOTNULL(mmrCtxInfo.x2_shape);
    mmrCtxInfo.bias_shape = context->GetOptionalInputShape(index++);
    mmrCtxInfo.x3_shape = context->GetOptionalInputShape(index++);
    mmrCtxInfo.antiquant_scale_shape = context->GetOptionalInputShape(index++);
    mmrCtxInfo.antiquant_offset_shape = context->GetOptionalInputShape(index++);
    mmrCtxInfo.dequant_scale_shape = context->GetOptionalInputShape(index++);
    mmrCtxInfo.pertoken_scale_shape = context->GetOptionalInputShape(index++);
    mmrCtxInfo.comm_quant_scale_1_shape = context->GetOptionalInputShape(index++);
    mmrCtxInfo.comm_quant_scale_2_shape = context->GetOptionalInputShape(index++);
    index = 0U;
    mmrCtxInfo.y_shape = context->GetOutputShape(index++);
    return ge::GRAPH_SUCCESS;
}
}
#endif // _CONTEXT_TRANSFER_CC_
