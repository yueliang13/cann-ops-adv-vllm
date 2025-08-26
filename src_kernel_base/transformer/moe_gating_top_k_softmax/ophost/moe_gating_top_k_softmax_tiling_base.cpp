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
 * \file moe_gating_top_k_softmax_tiling_base.cpp
 * \brief
 */
#include "moe_gating_top_k_softmax_tiling_base.h"
#include "tiling/tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

static const int32_t OUT_INDEX = 0;
static const int32_t INDICES_INDEX = 1;
static const int32_t SOURCE_ROW_OUT_INDEX = 2;
static const int32_t MAX_K = 1024;
static const uint32_t MAX_INT32 = 2147483647;

ge::graphStatus MoeGatingTopKSoftmaxBaseTiling::GetPlatformInfo() {
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
  coreNum = ascendcPlatform.GetCoreNum();
  uint64_t ubSizePlatForm;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
  ubSize = ubSizePlatForm;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxBaseTiling::CheckOutShape(const gert::Shape& outShape, gert::Shape& gatingShape) {
  OPS_ERR_IF(
      (outShape.GetDimNum() != gatingShape.GetDimNum()),
      OPS_REPORT_VECTOR_INNER_ERR(context_,
                                      "Tiling4MoeGatingTopKSoftmax out and x shape num not equal, please check."),
      return ge::GRAPH_FAILED);
  OPS_ERR_IF(
      (outShape.GetDim(0) != gatingShape.GetDim(0)),
      OPS_REPORT_VECTOR_INNER_ERR(context_,
                                      "Tiling4MoeGatingTopKSoftmax out and x dim 1 not equal, please check."),
      return ge::GRAPH_FAILED);
  if (gatingShape.GetDimNum() == 3U) {
    OPS_ERR_IF(
        (outShape.GetDim(1) != gatingShape.GetDim(1)),
        OPS_REPORT_VECTOR_INNER_ERR(context_,
                                        "Tiling4MoeGatingTopKSoftmax out and x dim 2 not equal, please check."),
        return ge::GRAPH_FAILED);
  }
  return ge::SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxBaseTiling::GetShapeAttrsInfo() {
  auto gating = context_->GetInputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, gating);
  dtype = gating->GetDataType();
  auto gatingShape = context_->GetInputShape(0)->GetStorageShape();
  OPS_ERR_IF((gatingShape.GetDimNum() != 2U && gatingShape.GetDimNum() != 3U),
                  OPS_REPORT_VECTOR_INNER_ERR(
                      context_,
                      "Tiling4MoeGatingTopKSoftmax get x shape dim(=%zu) is not 2 or 3, please check.",
                      gatingShape.GetDimNum()),
                  return ge::GRAPH_FAILED);
  for (size_t i = 0; i < gatingShape.GetDimNum(); i++) {
    OPS_ERR_IF(
        (gatingShape.GetDim(i) == 0),
        OPS_REPORT_VECTOR_INNER_ERR(context_,
                                        "Tiling4MoeGatingTopKSoftmax x shape contain zero, please check."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(
        (gatingShape.GetDim(i) > MAX_INT32),
        OPS_REPORT_VECTOR_INNER_ERR(context_,
                                        "Tiling4MoeGatingTopKSoftmax x shape larger than (=%u), please check.",
                                        MAX_INT32),
        return ge::GRAPH_FAILED);
  }

  auto finished = context_->GetOptionalInputShape(1);
  if (finished != nullptr) {
    auto finishDtype = context_->GetOptionalInputDesc(1)->GetDataType();
    OPS_ERR_IF((finishDtype != ge::DataType::DT_BOOL),
                    OPS_REPORT_VECTOR_INNER_ERR(context_,
                        "Tiling4MoeGatingTopKSoftmax get finish type error, should be BOOL, please check."),
                    return ge::GRAPH_FAILED);
    auto finishedShape = finished->GetStorageShape();
    OPS_ERR_IF((finishedShape.GetDimNum() != gatingShape.GetDimNum() - 1),
                    OPS_REPORT_VECTOR_INNER_ERR(
                        context_,
                        "Tiling4MoeGatingTopKSoftmax get finish dim num (=%zu) error, should be (=%zu), please check.",
                        finishedShape.GetDimNum(), gatingShape.GetDimNum() - 1),
                    return ge::GRAPH_FAILED);
    for (size_t i = 0; i < gatingShape.GetDimNum() - 1; i++) {
      OPS_ERR_IF(
          (finishedShape.GetDim(i) != gatingShape.GetDim(i)),
          OPS_REPORT_VECTOR_INNER_ERR(
              context_, "Tiling4MoeGatingTopKSoftmax finish and x shape not equal, please check."),
          return ge::GRAPH_FAILED);
    }
  }

  auto ret = CheckOutShape(context_->GetOutputShape(OUT_INDEX)->GetStorageShape(), gatingShape);
  if (ret != ge::SUCCESS) {
    return ret;
  }

  ret = CheckOutShape(context_->GetOutputShape(INDICES_INDEX)->GetStorageShape(), gatingShape);
  if (ret != ge::SUCCESS) {
    return ret;
  }

  ret = CheckOutShape(context_->GetOutputShape(SOURCE_ROW_OUT_INDEX)->GetStorageShape(), gatingShape);
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
  OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
  const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, kPtr);
  k = *kPtr;
  OPS_ERR_IF(
      (k <= 0 || k > col),
      OPS_REPORT_VECTOR_INNER_ERR(
          context_, "Tiling4MoeGatingTopKSoftmax attr k(=%ld) is wrong, please check. col=%u", k, col),
      return ge::GRAPH_FAILED);
  OPS_ERR_IF(
      (k > MAX_K),
      OPS_REPORT_VECTOR_INNER_ERR(
          context_, "Tiling4MoeGatingTopKSoftmax attr k(=%ld) is too large, please check.", k),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling