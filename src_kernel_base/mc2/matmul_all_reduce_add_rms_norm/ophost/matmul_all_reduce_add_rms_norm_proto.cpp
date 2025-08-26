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
 * \file matmul_all_reduce_add_rms_norm_proto.cc
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "hcom_topo_info.h"
#include "op_mc2.h"
#include "error_util.h"
#include "log/ops_log.h"

namespace ops {
constexpr size_t kMC2MinShapeSize = 2U;
constexpr size_t kMC2MaxShapeSize = 3U;
static const char *kInnerDebug = "MC2 Inner Debug";

struct MatmulShapeInfo {
    size_t output_dim;
    int64_t s = 1;
    int64_t m;
    int64_t n;
    int64_t k;
};

static ge::graphStatus CheckScaleShape(gert::InferShapeContext *context, int64_t group_size, MatmulShapeInfo &shape,
                                       bool is_trans_b)
{
    const size_t scale_idx = static_cast<size_t>(MC2AddRmsNormInputIdx::K_SCALE);
    const gert::Shape *scale_shape = context->GetOptionalInputShape(scale_idx);
    if (scale_shape == nullptr || group_size == 0) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t k_value = -1;
    if (shape.k != 0 && group_size != 0) {
        int64_t quotient = static_cast<int64_t>(shape.k / group_size);
        k_value = ((shape.k % group_size) != 0 && (shape.k ^ group_size) >= 0) ? (quotient + 1) : quotient;
    } else {
        k_value = static_cast<int64_t>(shape.k);
    }
    gert::Shape expect_scale;
    if (is_trans_b) {
        expect_scale = {shape.n, k_value};
    } else {
        expect_scale = {k_value, shape.n};
    }
    if (expect_scale != *scale_shape) {
        OPS_LOG_E(context->GetNodeName(), "antiquant scale shape is not equal to expect_scale");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForMatmul(gert::InferShapeContext *context, MatmulShapeInfo &shape)
{
    const auto shape_x1 = context->GetInputShape(static_cast<size_t>(MC2InputIdx::K_X1));
    OPS_LOG_E_IF_NULL(context, shape_x1, return ge::GRAPH_FAILED);
    const size_t dim_num_x1 = shape_x1->GetDimNum();
    if (dim_num_x1 < kMC2MinShapeSize || dim_num_x1 > kMC2MaxShapeSize) {
        OPS_LOG_E(context->GetNodeName(), "x1 dim number invalid");
        return ge::GRAPH_FAILED;
    }
    const auto shape_x2 = context->GetInputShape(static_cast<size_t>(MC2InputIdx::K_X2));
    OPS_LOG_E_IF_NULL(context, shape_x2, return ge::GRAPH_FAILED);
    const size_t dim_num_x2 = shape_x2->GetDimNum();
    if (dim_num_x2 != kMC2MinShapeSize) {
        OPS_LOG_E(context->GetNodeName(), "x2 dim number invalid");
        return ge::GRAPH_FAILED;
    }
    const auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const bool *trans_a = attrs->GetAttrPointer<bool>(static_cast<size_t>(MmAllReduceAttrIdx::K_TRANS_X1));
    if (trans_a != nullptr && *trans_a) {
        OPS_LOG_E(context->GetNodeName(), "x1 does not support transpose");
        return ge::GRAPH_FAILED;
    }
    const bool *trans_b = attrs->GetAttrPointer<bool>(static_cast<size_t>(MmAllReduceAttrIdx::K_TRANS_X2));
    const bool is_trans_b = ((trans_b != nullptr) && (*trans_b));
    if (dim_num_x1 == kMC2MaxShapeSize) {
        shape.s = shape_x1->GetDim(0U);
    }
    shape.m = shape_x1->GetDim(dim_num_x1 - 2U);
    shape.k = shape_x1->GetDim(dim_num_x1 - 1U);
    shape.n = is_trans_b ? shape_x2->GetDim(0U) : shape_x2->GetDim(1U);
    const auto shapeX2KIndex = is_trans_b ? 1U : 0U;
    // shape range推导的最大范围是[1,-1]
    bool is_dynamic_shape =
        (shape.k == -1 || shape_x2->GetDim(shapeX2KIndex) == -1 || shape.k == 0 ||
         shape_x2->GetDim(shapeX2KIndex) == 0 || shape.k == 1 || shape_x2->GetDim(shapeX2KIndex) == 1);
    if (!is_dynamic_shape) {
        if (shape.k != shape_x2->GetDim(shapeX2KIndex)) {
            OPS_LOG_E(context->GetNodeName(), "Invalid shape for x1(k) x2(k)");
            return ge::GRAPH_FAILED;
        }
        const int64_t *p = attrs->GetInt(static_cast<size_t>(MmAllReduceAttrIdx::K_ANTIQUANT_GROUP_SIZE));
        const int64_t group_size = (p != nullptr ? *p : 0);
        if (CheckScaleShape(context, group_size, shape, is_trans_b) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(context->GetNodeName(), "Failed to check antiquant scale shape");
            return ge::GRAPH_FAILED;
        }
    }
    shape.output_dim = dim_num_x1;
    OPS_LOG_D(context->GetNodeName(), "Matmul x1 dim %zu s %ld m %ld n %ld k %ld.", dim_num_x1, shape.s, shape.m,
              shape.n, shape.k);
    return ge::GRAPH_SUCCESS;
}

static bool InferAllZeroShape(gert::InferShapeContext *context)
{
    const auto shape_x1 = context->GetInputShape(static_cast<size_t>(MC2AddRmsNormInputIdx::K_X1));
    if (shape_x1 == nullptr) {
        return false;
    }
    const size_t dim_num_x1 = shape_x1->GetDimNum();
    int64_t k = -1;
    if (dim_num_x1 == kMC2MaxShapeSize) {
        k = shape_x1->GetDim(2U);
    } else if (dim_num_x1 == kMC2MinShapeSize) {
        k = shape_x1->GetDim(1U);
    }
    if (k != 0) {
        return false;
    }
    OPS_LOG_I(context->GetNodeName(), "kValue of x1 is 0.");
    return true;
}

static ge::graphStatus InferShapeForMC2AddRmsNorm(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OPS_LOG_E(kInnerDebug, "context is null");
        return ge::GRAPH_FAILED;
    }
    MatmulShapeInfo shape;
    if (InferShapeForMatmul(context, shape) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Failed to infer shape for matmul all reduce");
        return ge::GRAPH_FAILED;
    }
    size_t residual_idx = static_cast<size_t>(MC2AddRmsNormInputIdx::K_RESIDUAL);
    if (context->GetComputeNodeInputNum() != static_cast<size_t>(MC2AddRmsNormInputIdx::K_MAX) &&
        context->GetOptionalInputShape(static_cast<size_t>(MC2AddRmsNormInputIdx::K_BIAS)) == nullptr) {
        residual_idx -= 1U;
    }
    OPS_LOG_I(context->GetNodeName(), "Start to infer shape, residual tensor idx %zu.", residual_idx);
    const auto res_shape = context->GetInputShape(residual_idx);
    auto shape_out_y = context->GetOutputShape(static_cast<size_t>(MC2AddRmsNormOutputIdx::K_Y));
    auto shape_out_norm = context->GetOutputShape(static_cast<size_t>(MC2AddRmsNormOutputIdx::K_NORM_OUT));
    OPS_LOG_E_IF_NULL(context, res_shape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, shape_out_y, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, shape_out_norm, return ge::GRAPH_FAILED);
    if (InferAllZeroShape(context)) {
        OPS_LOG_E(context->GetNodeName(), "MatmulAllReduceAddRmsNorm does not support k = 0.");
        return ge::GRAPH_FAILED;
    }
    const size_t dim_num = res_shape->GetDimNum();
    if (dim_num != 3U) {
        OPS_LOG_E(context->GetNodeName(), "Invalid dim number %lu for residual.", dim_num);
        return ge::GRAPH_FAILED;
    }
    shape_out_y->SetDimNum(dim_num);
    shape_out_norm->SetDimNum(dim_num);
    for (size_t i = 0U; i < dim_num; ++i) {
        const int64_t dim = res_shape->GetDim(i);
        shape_out_y->SetDim(i, dim);
        shape_out_norm->SetDim(i, dim);
        OPS_LOG_D(context->GetNodeName(), "Output shape dim idx %lu, dim value %ld.", i, dim);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForMC2AddRmsNorm(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OPS_LOG_E(kInnerDebug, "context is null");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType out_type = context->GetInputDataType(static_cast<size_t>(MC2AddRmsNormInputIdx::K_RESIDUAL));
    OPS_LOG_I(context->GetNodeName(), "Start to infer dtype, output dtype is %d.", out_type);
    ge::graphStatus ret = context->SetOutputDataType(static_cast<size_t>(MC2AddRmsNormOutputIdx::K_Y), out_type);
    if (ret != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Failed to set dtype %d to y.", out_type);
        return ge::GRAPH_FAILED;
    }
    ret = context->SetOutputDataType(static_cast<size_t>(MC2AddRmsNormOutputIdx::K_NORM_OUT), out_type);
    if (ret != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Failed to set dtype %d to norm.", out_type);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatmulAllReduceAddRmsNorm)
    .InferShape(InferShapeForMC2AddRmsNorm)
    .InferDataType(InferDtypeForMC2AddRmsNorm);
IMPL_OP_INFERSHAPE(InplaceMatmulAllReduceAddRmsNorm)
    .InferShape(InferShapeForMC2AddRmsNorm)
    .InferDataType(InferDtypeForMC2AddRmsNorm);
} // namespace ops
