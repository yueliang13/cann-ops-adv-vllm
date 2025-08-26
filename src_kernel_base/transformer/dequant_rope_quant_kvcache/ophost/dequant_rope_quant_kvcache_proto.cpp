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
 * \file dequant_rope_quant_kvcache_proto.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "error/ops_error.h"

namespace{
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;
static inline bool IsUnknownRank(const gert::Shape* check_shape) {
  return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}
static inline ge::graphStatus SetUnknownRank(gert::Shape* outShape) {
    outShape->SetDimNum(0);
    outShape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);
    return ge::GRAPH_SUCCESS;
} 
}

using namespace ge;
namespace {
static constexpr size_t INPUT_X_INDEX = 0;
static constexpr size_t INPUT_COS_INDEX = 1;
static constexpr size_t INPUT_SIN_INDEX = 2;
static constexpr size_t INPUT_K_CACHE_INDEX = 3;
static constexpr size_t INPUT_V_CACHE_INDEX = 4;
static constexpr size_t INPUT_INDICES_INDEX = 5;
static constexpr size_t INPUT_SCALE_K_INDEX = 6;
static constexpr size_t INPUT_SCALE_V_INDEX = 7;
static constexpr size_t INPUT_OFFSET_K_INDEX = 8;
static constexpr size_t INPUT_OFFSET_V_INDEX = 9;
static constexpr size_t INPUT_WEIGHT_SCALE_INDEX = 10;
static constexpr size_t INPUT_ACTIVATION_SCALE_INDEX = 11;
static constexpr size_t INPUT_BIAS_INDEX = 12;
static constexpr size_t OUTPUT_Q_INDEX = 0;
static constexpr size_t OUTPUT_K_INDEX = 1;
static constexpr size_t OUTPUT_V_INDEX = 2;
static constexpr size_t OUTPUT_K_CACHE_INDEX = 3;
static constexpr size_t OUTPUT_V_CACHE_INDEX = 4;
static constexpr size_t ATTR_SIZE_SPLITS_Q_INDEX=0;
static constexpr size_t ATTR_SIZE_SPLITS_K_INDEX=0;
static constexpr size_t ATTR_SIZE_SPLITS_V_INDEX=0;
static constexpr size_t ATTR_IFKVOUT_INDEX = 3;
static constexpr size_t TOTAL_DIM = 4;
static constexpr size_t X_DIM = 3;
static constexpr size_t THIRD_DIM = 2;
static constexpr size_t FORTH_DIM = 3;
static constexpr size_t NEG_ONE = -1;
static constexpr size_t BASE_2 = 2;
static constexpr size_t BASE_3 = 3;

template <typename T>
static auto GetDiv(const T& value1, const T& value2) -> T {
  if (value2 == 0) {
    return value2;
  }
  return (value1) / value2;
}
}  // namespace

namespace ops {
static ge::graphStatus InferShapeForDequantRopeQuantKvcache(gert::InferShapeContext *context) {
  // input shape
  OPS_LOG_D(context->GetNodeName(), "Begin to do InferShapeForDequantRopeQuantKvcache.");
  const gert::Shape* qkvShape = context->GetInputShape(INPUT_X_INDEX);
  OPS_LOG_E_IF_NULL(context, qkvShape, return ge::GRAPH_FAILED);

  const gert::Shape* cacheKShape = context->GetInputShape(INPUT_K_CACHE_INDEX);
  OPS_LOG_E_IF_NULL(context, cacheKShape, return ge::GRAPH_FAILED);

  const gert::Shape* cacheVShape = context->GetInputShape(INPUT_V_CACHE_INDEX);
  OPS_LOG_E_IF_NULL(context, cacheVShape, return ge::GRAPH_FAILED);

  gert::Shape* qOutShape = context->GetOutputShape(OUTPUT_Q_INDEX);
  OPS_LOG_E_IF_NULL(context, qOutShape, return ge::GRAPH_FAILED);

  gert::Shape* kOutShape = context->GetOutputShape(OUTPUT_K_INDEX);
  OPS_LOG_E_IF_NULL(context, kOutShape, return ge::GRAPH_FAILED);

  gert::Shape* vOutShape = context->GetOutputShape(OUTPUT_V_INDEX);
  OPS_LOG_E_IF_NULL(context, vOutShape, return ge::GRAPH_FAILED);

  gert::Shape* kCacheShape = context->GetOutputShape(OUTPUT_K_CACHE_INDEX);
  OPS_LOG_E_IF_NULL(context, kCacheShape, return ge::GRAPH_FAILED);

  gert::Shape* vCacheShape = context->GetOutputShape(OUTPUT_V_CACHE_INDEX);
  OPS_LOG_E_IF_NULL(context, vCacheShape, return ge::GRAPH_FAILED);

  *kCacheShape = *cacheKShape;
  *vCacheShape = *cacheVShape;

  auto attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);

  const gert::ContinuousVector* splitSize = attrs->GetAttrPointer<gert::ContinuousVector>(0);
  const int64_t* splitSizeArray = reinterpret_cast<const int64_t*>(splitSize->GetData());
  int64_t ifKVout = *attrs->GetAttrPointer<bool>(ATTR_IFKVOUT_INDEX) == true ? 1 : 0;
  int64_t batch = -1;
  int64_t qkvSeqlen = -1;
  int64_t kvHead = -1;
  int64_t hiddenSize = -1;

  if (!IsUnknownRank(qkvShape)) {
    OPS_ERR_IF((qkvShape->GetDimNum() != X_DIM && qkvShape->GetDimNum() != BASE_2),
             OPS_REPORT_VECTOR_INNER_ERR(context,
             "input_x's dimnum should be 3 or 2."),
             return ge::GRAPH_FAILED);
    batch = qkvShape->GetDim(0);
    if (qkvShape->GetDimNum() == BASE_2) {
      qkvSeqlen = 1;
    } else {
      qkvSeqlen = qkvShape->GetDim(1);
    }
  }
  if (!IsUnknownRank(kCacheShape)) {
    OPS_ERR_IF(kCacheShape->GetDimNum() != TOTAL_DIM,
             OPS_REPORT_VECTOR_INNER_ERR(context,
             "kCache's dimnum should be 4."),
             return ge::GRAPH_FAILED);
    kvHead = cacheKShape->GetDim(THIRD_DIM);
    hiddenSize = cacheKShape->GetDim(FORTH_DIM);
  }
  OPS_ERR_IF(splitSizeArray[0] < 0,
           OPS_REPORT_VECTOR_INNER_ERR(context,
           "size_splits[0] should not less than 0."),
           return ge::GRAPH_FAILED);
  int64_t qHead = hiddenSize < 0 ? -1 : GetDiv(splitSizeArray[0], hiddenSize);

  if (IsUnknownRank(qkvShape)) {
    SetUnknownRank(qOutShape);
    SetUnknownRank(kOutShape);
    SetUnknownRank(vOutShape);
  } else if (qkvShape->GetDimNum() == BASE_2) {
    qOutShape->SetDimNum(BASE_3);
    qOutShape->SetDim(0, batch);
    qOutShape->SetDim(1, qHead);
    qOutShape->SetDim(THIRD_DIM, hiddenSize);

    if (ifKVout == 1) {
      kOutShape->SetDimNum(BASE_3);
      kOutShape->SetDim(0, batch);
      kOutShape->SetDim(1, kvHead);
      kOutShape->SetDim(THIRD_DIM, hiddenSize);

      vOutShape->SetDimNum(BASE_3);
      vOutShape->SetDim(0, batch);
      vOutShape->SetDim(1, kvHead);
      vOutShape->SetDim(THIRD_DIM, hiddenSize);
    } else {
      kOutShape->SetDimNum(BASE_3);
      kOutShape->SetDim(0, 0);

      vOutShape->SetDimNum(BASE_3);
      vOutShape->SetDim(0, 0);
    }
  } else {
    qOutShape->SetDimNum(TOTAL_DIM);
    qOutShape->SetDim(0, batch);
    qOutShape->SetDim(1, qkvSeqlen);
    qOutShape->SetDim(THIRD_DIM, qHead);
    qOutShape->SetDim(FORTH_DIM, hiddenSize);

    if (ifKVout == 1) {
      kOutShape->SetDimNum(TOTAL_DIM);
      kOutShape->SetDim(0, batch);
      kOutShape->SetDim(1, qkvSeqlen);
      kOutShape->SetDim(THIRD_DIM, kvHead);
      kOutShape->SetDim(FORTH_DIM, hiddenSize);

      vOutShape->SetDimNum(TOTAL_DIM);
      vOutShape->SetDim(0, batch);
      vOutShape->SetDim(1, qkvSeqlen);
      vOutShape->SetDim(THIRD_DIM, kvHead);
      vOutShape->SetDim(FORTH_DIM, hiddenSize);
    } else {
      kOutShape->SetDimNum(1);
      kOutShape->SetDim(0, 0);

      vOutShape->SetDimNum(1);
      vOutShape->SetDim(0, 0);
    }
  }

  OPS_LOG_D(context->GetNodeName(), "End to do InferShapeForDequantRopeQuantKvcache.");
  return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForDequantRopeQuantKvcache(gert::InferDataTypeContext *context) {
  context->SetOutputDataType(OUTPUT_Q_INDEX, context->GetInputDataType(INPUT_COS_INDEX));
  context->SetOutputDataType(OUTPUT_K_INDEX, context->GetInputDataType(INPUT_COS_INDEX));
  context->SetOutputDataType(OUTPUT_V_INDEX, context->GetInputDataType(INPUT_COS_INDEX));
  context->SetOutputDataType(OUTPUT_K_CACHE_INDEX, context->GetInputDataType(INPUT_K_CACHE_INDEX));
  context->SetOutputDataType(OUTPUT_V_CACHE_INDEX, context->GetInputDataType(INPUT_V_CACHE_INDEX));
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DequantRopeQuantKvcache).InferShape(InferShapeForDequantRopeQuantKvcache)
                                     .InferDataType(InferDataTypeForDequantRopeQuantKvcache);
}  // namespace ops