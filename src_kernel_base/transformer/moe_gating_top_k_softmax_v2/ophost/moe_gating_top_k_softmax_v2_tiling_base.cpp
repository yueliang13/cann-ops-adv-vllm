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
 * \file moe_gating_top_k_softmax_v2_tiling_base.cpp
 * \brief
 */
#include "moe_gating_top_k_softmax_v2_tiling.h"
#include "tiling/tiling_templates_registry.h"
using namespace AscendC;
using namespace ge;

namespace optiling {

static const int32_t OUT_INDEX = 0;
static const int32_t INDICES_INDEX = 1;
static const int32_t SOFTMAX_RESULT_INDEX = 2;
static const int32_t MAX_K = 1024;
static const uint32_t MAX_INT32 = 2147483647;

ge::graphStatus MoeGatingTopKSoftmaxV2BaseTiling::GetPlatformInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    coreNum = ascendcPlatform.GetCoreNum();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize = ubSizePlatForm;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2BaseTiling::CheckOutShape(const gert::Shape &outShape, gert::Shape &gatingShape,
                                                                bool isSoftmax)
{
    if (outShape.GetDimNum() != gatingShape.GetDimNum()) {
        OPS_REPORT_VECTOR_INNER_ERR(context_,
                                    "Tiling4MoeGatingTopKSoftmaxV2 out and x shape num not equal, please check.");
        return ge::GRAPH_FAILED;
    }
    if (outShape.GetDim(0) != gatingShape.GetDim(0)) {
        OPS_REPORT_VECTOR_INNER_ERR(context_, "Tiling4MoeGatingTopKSoftmaxV2 out and x dim 0 not equal, please check.");
        return ge::GRAPH_FAILED;
    }
    if (gatingShape.GetDimNum() == 3U) {
        if (outShape.GetDim(1) != gatingShape.GetDim(1)) {
            OPS_REPORT_VECTOR_INNER_ERR(context_,
                                        "Tiling4MoeGatingTopKSoftmaxV2 out and x dim 1 not equal, please check.");
            return ge::GRAPH_FAILED;
        }
    }
    size_t lastDimNum = gatingShape.GetDimNum() - 1;
    if (isSoftmax) {
        auto softmaxOutDtype = context_->GetOutputDesc(SOFTMAX_RESULT_INDEX)->GetDataType();
        if (softmaxOutDtype != ge::DataType::DT_FLOAT) {
            OPS_REPORT_VECTOR_INNER_ERR(
                context_,
                "Tiling4MoeGatingTopKSoftmaxV2 get softmax result type error, should be float, please check.");
            return ge::GRAPH_FAILED;
        }
        if (outShape.GetDim(lastDimNum) != gatingShape.GetDim(lastDimNum)) {
            OPS_REPORT_VECTOR_INNER_ERR(
                context_, "Tiling4MoeGatingTopKSoftmaxV2 softmax and x last dim not equal, please check.");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (outShape.GetDim(lastDimNum) != k) {
            OPS_REPORT_VECTOR_INNER_ERR(
                context_, "Tiling4MoeGatingTopKSoftmaxV2 out or expertIdx last dim and k not equal, please check.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2BaseTiling::CheckInShape(const gert::Shape &gatingShape)
{
    if (gatingShape.GetDimNum() != 2U && gatingShape.GetDimNum() != 3U) {
        OPS_REPORT_VECTOR_INNER_ERR(context_,
                                    "Tiling4MoeGatingTopKSoftmaxV2 get x shape dim(=%zu) is not 2 or 3, please check.",
                                    gatingShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < gatingShape.GetDimNum(); i++) {
        if (gatingShape.GetDim(i) == 0) {
            OPS_REPORT_VECTOR_INNER_ERR(context_, "Tiling4MoeGatingTopKSoftmax x shape contain zero, please check.");
            return ge::GRAPH_FAILED;
        }

        if (gatingShape.GetDim(i) > MAX_INT32) {
            OPS_REPORT_VECTOR_INNER_ERR(
                context_, "Tiling4MoeGatingTopKSoftmaxV2 x shape larger than (=%u), please check.", MAX_INT32);
            return ge::GRAPH_FAILED;
        }
    }

    auto finished = context_->GetOptionalInputShape(1);
    if (finished != nullptr) {
        auto finishDtype = context_->GetOptionalInputDesc(1)->GetDataType();
        if (finishDtype != ge::DataType::DT_BOOL) {
            OPS_REPORT_VECTOR_INNER_ERR(
                context_, "Tiling4MoeGatingTopKSoftmaxV2 get finished type error, should be BOOL, please check.");
            return ge::GRAPH_FAILED;
        }

        auto finishedShape = finished->GetStorageShape();
        if (finishedShape.GetDimNum() != gatingShape.GetDimNum() - 1) {
            OPS_REPORT_VECTOR_INNER_ERR(
                context_,
                "Tiling4MoeGatingTopKSoftmaxV2 get finished dim num (=%zu) error, should be (=%zu), please check.",
                finishedShape.GetDimNum(), gatingShape.GetDimNum() - 1);
            return ge::GRAPH_FAILED;
        }

        for (size_t i = 0; i < gatingShape.GetDimNum() - 1; i++) {
            if (finishedShape.GetDim(i) != gatingShape.GetDim(i)) {
                OPS_REPORT_VECTOR_INNER_ERR(
                    context_, "Tiling4MoeGatingTopKSoftmaxV2 finished and x shape not equal, please check.");
                return ge::GRAPH_FAILED;
            }
        }
    }

    return ge::SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2BaseTiling::CheckOptionalAttr(gert::Shape &gatingShape)
{
    auto attrs = context_->GetAttrs();
    const int64_t *renormPtr = attrs->GetAttrPointer<int64_t>(1);
    renorm = 0;
    if (renormPtr) {
        renorm = *renormPtr;
        if (renorm < 0 || renorm > 1) {
            OPS_REPORT_VECTOR_INNER_ERR(
                context_, "Tiling4MoeGatingTopKSoftmaxV2 attr renorm(=%d) is wrong, please check.", renorm);
            return ge::GRAPH_FAILED;
        }
    }

    const bool *softmaxFlagPtr = attrs->GetAttrPointer<bool>(2);
    softmaxFlag = 0;
    if (softmaxFlagPtr) {
        softmaxFlag = int(*softmaxFlagPtr);
    }

    if (renorm == 0 && softmaxFlag == 1) {
        if (context_->GetOutputShape(SOFTMAX_RESULT_INDEX) == nullptr) {
            OPS_REPORT_VECTOR_INNER_ERR(context_, "Tiling4MoeGatingTopKSoftmaxV2 softmax is nullptr, please check.");
            return ge::GRAPH_FAILED;
        }

        auto ret = CheckOutShape(context_->GetOutputShape(SOFTMAX_RESULT_INDEX)->GetStorageShape(), gatingShape, true);
        if (ret != ge::SUCCESS) {
            return ret;
        }
    }

    return ge::SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2BaseTiling::GetShapeAttrsInfo()
{
    auto gating = context_->GetInputDesc(0);
    OPS_LOG_E_IF_NULL(context_, gating, return ge::GRAPH_FAILED);
    dtype = gating->GetDataType();
    auto gatingShape = context_->GetInputShape(0)->GetStorageShape();
    auto ret = CheckInShape(gatingShape);
    if (ret != ge::SUCCESS) {
        return ret;
    }

    if (gatingShape.GetDimNum() == 2U) {
        row = gatingShape.GetDim(0);
        col = gatingShape.GetDim(1);
    } else {
        row = gatingShape.GetDim(0) * gatingShape.GetDim(1);
        col = gatingShape.GetDim(2U);
    }

    auto attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);
    const int64_t *kPtr = attrs->GetAttrPointer<int64_t>(0);
    OPS_LOG_E_IF_NULL(context_, kPtr, return ge::GRAPH_FAILED);
    k = *kPtr;
    if (k <= 0 || k > int(col)) {
        OPS_REPORT_VECTOR_INNER_ERR(context_,
                                    "Tiling4MoeGatingTopKSoftmaxV2 attr k(=%d) is wrong, please check. col=%u", k, col);
        return ge::GRAPH_FAILED;
    }
    if (k > MAX_K) {
        OPS_REPORT_VECTOR_INNER_ERR(context_, "Tiling4MoeGatingTopKSoftmaxV2 attr k(=%d) is too large, please check.",
                                    k);
        return ge::GRAPH_FAILED;
    }

    ret = CheckOutShape(context_->GetOutputShape(OUT_INDEX)->GetStorageShape(), gatingShape, false);
    if (ret != ge::SUCCESS) {
        return ret;
    }

    ret = CheckOutShape(context_->GetOutputShape(INDICES_INDEX)->GetStorageShape(), gatingShape, false);
    if (ret != ge::SUCCESS) {
        return ret;
    }

    ret = CheckOptionalAttr(gatingShape);
    if (ret != ge::SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling