/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback_comm.h"
#include "fallback.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;

constexpr size_t INDEX_GMM_INPUT_X = 0;
constexpr size_t INDEX_GMM_INPUT_WEIGHT = 1;
constexpr size_t INDEX_GMM_INPUT_BIAS = 2;
constexpr size_t INDEX_GMM_INPUT_SCALE = 3;
constexpr size_t INDEX_GMM_INPUT_OFFSET = 4;
constexpr size_t INDEX_GMM_INPUT_ANTIQUANT_SCALE = 5;
constexpr size_t INDEX_GMM_INPUT_ANTIQUANT_OFFSET = 6;
constexpr size_t INDEX_GMM_INPUT_GROUP_LIST = 7;
constexpr size_t INDEX_GMM_INPUT_PER_TOKEN_SCALE = 8;
constexpr size_t INDEX_GMM_OUTPUT_Y = 0;
constexpr size_t INDEX_GMM_ATTR_SPLIT_ITEM = 0;
constexpr size_t INDEX_GMM_ATTR_TRANSPOSE_WEIGHT = 2;
constexpr size_t INDEX_GMM_ATTR_TRANSPOSE_X = 3;
constexpr size_t INDEX_GMM_ATTR_GROUP_TYPE = 4;
constexpr size_t INDEX_GMM_ATTR_GROUP_LIST_TYPE = 5;
constexpr size_t INDEX_GMM_ATTR_ACT_TYPE = 6;

inline aclTensorList* ConvertType(aclTensorList* geTensorList) {
  return geTensorList;
}

inline aclTensor* GeTensor2AclTensor(const gert::Tensor* geTensor, bool enableTranspose, bool enableNZ=false) {
  if (geTensor == nullptr) {
    return nullptr;
  }
  auto storageShape = geTensor->GetStorageShape();
  if (storageShape.GetDimNum() <= 1) {
    return ConvertType(geTensor);
  }
  std::vector<int64_t> storageShapeVec;
  for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
    storageShapeVec.push_back(storageShape.GetDim(i));
  }

  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  OPS_CHECK(aclCreateTensor == nullptr, OPS_LOG_E("aclnnfallback", "aclCreateTensor nullptr"), return nullptr);

  void* deviceAddr = (void*)geTensor->GetAddr();
  // convert data type
  auto dataType_ge = geTensor->GetDataType();
  auto dataType = ToAclDataType(dataType_ge);

  // convert view shape
  auto origin_shape = geTensor->GetOriginShape();
  std::vector<int64_t> viewShape;
  for (size_t i = 0; i < origin_shape.GetDimNum(); ++i) {
    viewShape.push_back(origin_shape.GetDim(i));
  }
  // Compute the strides of contiguous tensor
  std::vector<int64_t> strides(viewShape.size(), 1);
  for (int64_t i = viewShape.size() - 2; i >= 0; i--) {
    strides[i] = viewShape[i + 1] * strides[i + 1];
  }
  // when tensor is transposed, last two dims in strides and viewShape should swap
  if (enableTranspose) {
    // dimM the second-to-last dim， dimN the last dim
    auto dimM = viewShape.size() - 2;
    auto dimN = viewShape.size() - 1;
    auto swap =  strides[dimN];
    strides[dimN] = strides[dimM];
    strides[dimM] = swap;
    // swap viewShape
    swap = viewShape[dimN];
    viewShape[dimN] = viewShape[dimM];
    viewShape[dimM] = swap;
  }
  auto aclFormat = aclFormat::ACL_FORMAT_ND;
  if (enableNZ && GetPrimaryFormat(geTensor->GetStorageFormat()) == ge::Format::FORMAT_FRACTAL_NZ) {
    aclFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ;
  }
  aclTensor* out = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, strides.data(),
                                   0, aclFormat, storageShapeVec.data(), storageShapeVec.size(), deviceAddr);
  OPS_CHECK(out == nullptr, OPS_LOG_E("aclnnfallback", "out nullptr"), return nullptr);

  return out;
}

graphStatus PrepareGeTensorVector(OpExecuteContext* host_api_ctx, std::vector<const gert::Tensor*>& tensorVector, size_t index) {
  size_t cnt = 0;
  while (true) {
    auto inputGe = host_api_ctx->GetDynamicInputTensor(index, cnt);
    if (inputGe == nullptr) {
      break;
    }
    tensorVector.push_back(inputGe);
    cnt++;
  }
  return GRAPH_SUCCESS;
}

graphStatus PrepareAclTensorVector(OpExecuteContext* host_api_ctx, std::vector<const aclTensor*>& tensorVector, size_t index, bool enableTranspose, bool enableNZ) {
  size_t cnt = 0;
  while (true) {
    auto inputGe = host_api_ctx->GetDynamicInputTensor(index, cnt);
    if (inputGe == nullptr) {
      break;
    }
    auto inputAcl = GeTensor2AclTensor(inputGe, enableTranspose, enableNZ);
    tensorVector.push_back(inputAcl);
    cnt++;
  }
  return GRAPH_SUCCESS;
}

graphStatus PrepareOutputTensorVector(OpExecuteContext* host_api_ctx, std::vector<const gert::Tensor*>& tensorVector, size_t index, size_t numGeWeight, int32_t splitItem) {
  size_t numGeY = 0;
  if (0 == splitItem || 1 == splitItem) { // Length of tensorListY equals that of tensorListWeight when split_item = 0 / 1
    numGeY = numGeWeight;
  }
  else if (2 == splitItem || 3 == splitItem) { // Length of tensorListY equals 1 when split_item = 2 / 3
    numGeY = 1;
  }
  else {
    OPS_LOG_E("aclnnfallback", "Invalid value of split_item: %d, which must be one of 0/1/2/3.", splitItem);
    return GRAPH_FAILED;
  }

  for (size_t k = 0; k < numGeY; k++) {
    auto outputGe = host_api_ctx->GetOutputTensor(index + k);
    if (outputGe == nullptr) {return GRAPH_FAILED;}
    tensorVector.push_back(outputGe);
  }
  return GRAPH_SUCCESS;
}

static graphStatus GroupedMatmulExecuteFunc(OpExecuteContext* host_api_ctx)
{
  OPS_CHECK(host_api_ctx == nullptr, OPS_LOG_E("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

  auto attrs = host_api_ctx->GetAttrs();
  OPS_CHECK(attrs == nullptr, OPS_LOG_E("aclnnfallback", "attrs is null"), return GRAPH_FAILED);
  const int64_t* splitItemGe = attrs->GetAttrPointer<int64_t>(INDEX_GMM_ATTR_SPLIT_ITEM);
  OPS_CHECK(splitItemGe == nullptr, OPS_LOG_E("aclnnfallback", "splitItemGe is null"), return GRAPH_FAILED);
  const bool* isWeightTransposed = attrs->GetAttrPointer<bool>(INDEX_GMM_ATTR_TRANSPOSE_WEIGHT);
  OPS_CHECK(isWeightTransposed == nullptr, OPS_LOG_E("aclnnfallback", "isWeightTransposed is null"), return GRAPH_FAILED);
  const bool* isXTransposed = attrs->GetAttrPointer<bool>(INDEX_GMM_ATTR_TRANSPOSE_X);
  OPS_CHECK(isXTransposed == nullptr, OPS_LOG_E("aclnnfallback", "isXTransposed is null"), return GRAPH_FAILED);
  const int64_t* groupTypeGe = attrs->GetAttrPointer<int64_t>(INDEX_GMM_ATTR_GROUP_TYPE);
  OPS_CHECK(groupTypeGe == nullptr, OPS_LOG_E("aclnnfallback", "groupTypeGe is null"), return GRAPH_FAILED);
  const int64_t* groupListTypeGe = attrs->GetAttrPointer<int64_t>(INDEX_GMM_ATTR_GROUP_LIST_TYPE);
  OPS_CHECK(groupListTypeGe == nullptr, OPS_LOG_E("aclnnfallback", "groupListTypeGe is null"), return GRAPH_FAILED);
  const int64_t* actTypeGe = attrs->GetAttrPointer<int64_t>(INDEX_GMM_ATTR_ACT_TYPE);
  OPS_CHECK(actTypeGe == nullptr, OPS_LOG_E("aclnnfallback", "actTypeGe is null"), return GRAPH_FAILED);

  static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
  OPS_CHECK(aclCreateTensorList == nullptr,
            OPS_LOG_E("aclnnfallback", "Get opapi func aclCreateTensorList failed"), return GRAPH_FAILED);

  std::vector<const aclTensor*> aclTensorVectorX;
  PrepareAclTensorVector(host_api_ctx, aclTensorVectorX, INDEX_GMM_INPUT_X, *isXTransposed, false);
  auto aclTensorListX = aclCreateTensorList(aclTensorVectorX.data(), aclTensorVectorX.size());

  std::vector<const aclTensor*> aclTensorVectorWeight;
  PrepareAclTensorVector(host_api_ctx, aclTensorVectorWeight, INDEX_GMM_INPUT_WEIGHT, *isWeightTransposed, true);
  size_t numGeWeight = aclTensorVectorWeight.size();
  auto aclTensorListWeight = aclCreateTensorList(aclTensorVectorWeight.data(), aclTensorVectorWeight.size());

  std::vector<const gert::Tensor*> geTensorVectorBias;
  PrepareGeTensorVector(host_api_ctx, geTensorVectorBias, INDEX_GMM_INPUT_BIAS);

  std::vector<const gert::Tensor*> geTensorVectorScale;
  PrepareGeTensorVector(host_api_ctx, geTensorVectorScale, INDEX_GMM_INPUT_SCALE);

  std::vector<const gert::Tensor*> geTensorVectorOffset;
  PrepareGeTensorVector(host_api_ctx, geTensorVectorOffset, INDEX_GMM_INPUT_OFFSET);

  std::vector<const gert::Tensor*> geTensorVectorAntiquantScale;
  PrepareGeTensorVector(host_api_ctx, geTensorVectorAntiquantScale, INDEX_GMM_INPUT_ANTIQUANT_SCALE);

  std::vector<const gert::Tensor*> geTensorVectorAntiquantOffset;
  PrepareGeTensorVector(host_api_ctx, geTensorVectorAntiquantOffset, INDEX_GMM_INPUT_ANTIQUANT_OFFSET);

  auto groupListTensor = host_api_ctx->GetOptionalInputTensor(INDEX_GMM_INPUT_GROUP_LIST);

  auto perTokenScale = ConvertType(host_api_ctx->GetOptionalInputTensor(INDEX_GMM_INPUT_PER_TOKEN_SCALE));
  if (perTokenScale == nullptr) {
    std::vector<int64_t> shape{0};
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    OPS_CHECK(aclCreateTensor == nullptr, OPS_LOG_E("aclnnfallback", "aclCreateTensor nullptr"), return GRAPH_FAILED);
    perTokenScale = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(),
                                    0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    OPS_CHECK(perTokenScale == nullptr, OPS_LOG_E("aclnnfallback", "perTokenScale nullptr"), return GRAPH_FAILED);
  }
  std::vector<const aclTensor*> geTensorVectorPerTokenScale{perTokenScale};
  auto aclTensorListPerTokenScale = aclCreateTensorList(geTensorVectorPerTokenScale.data(),
                                                        geTensorVectorPerTokenScale.size());

  std::vector<const gert::Tensor*> geTensorVectorY;
  PrepareOutputTensorVector(host_api_ctx, geTensorVectorY, INDEX_GMM_OUTPUT_Y, numGeWeight, *splitItemGe);

  aclTensorList* activationInputOptional = nullptr;
  aclTensorList* activationQuantScaleOptional = nullptr;
  aclTensorList* activationQuantOffsetOptional = nullptr;
  aclTensorList* actFeatureOutOptional = nullptr;
  aclTensorList* dynQuantScaleOutOptional = nullptr;

  // execute opapi
  auto api_ret = EXEC_OPAPI_CMD(aclnnGroupedMatmulV4, aclTensorListX, aclTensorListWeight, geTensorVectorBias,
                                geTensorVectorScale, geTensorVectorOffset, geTensorVectorAntiquantScale,
                                geTensorVectorAntiquantOffset, aclTensorListPerTokenScale, groupListTensor,
                                activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                                *splitItemGe, *groupTypeGe, *groupListTypeGe, *actTypeGe,
                                geTensorVectorY, actFeatureOutOptional, dynQuantScaleOutOptional);
  OPS_CHECK(api_ret != GRAPH_SUCCESS, OPS_LOG_E("aclnnfallback", "api_ret failed:%u", api_ret), return GRAPH_FAILED);
  return GRAPH_SUCCESS;
}

IMPL_OP(GroupedMatmul).OpExecuteFunc(GroupedMatmulExecuteFunc);

}  // namespace fallback

#ifdef __cplusplus
}
#endif
