/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

#include "fallback_comm.h"
#include "fallback.h"
#include "error/ops_error.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;
static const size_t QUERY_INDEX = 0;
static const size_t KEY_INDEX = 1;
static const size_t VALUE_INDEX = 2;

static graphStatus PromptHostExecuteFunc(OpExecuteContext* host_api_ctx)
{
  OPS_ERR_IF(host_api_ctx == nullptr,
    OPS_LOG_E("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

  auto query = host_api_ctx->GetInputTensor(QUERY_INDEX);
  OPS_ERR_IF(query == nullptr,
    OPS_LOG_E("aclnnfallback", "query is null"), return GRAPH_FAILED);

  auto key = host_api_ctx->GetInputTensor(KEY_INDEX);
  OPS_ERR_IF(key == nullptr,
    OPS_LOG_E("aclnnfallback", "key is null"), return GRAPH_FAILED);

  auto value = host_api_ctx->GetInputTensor(VALUE_INDEX);
  OPS_ERR_IF(value == nullptr,
    OPS_LOG_E("aclnnfallback", "value is null"), return GRAPH_FAILED);

  auto output = host_api_ctx->GetOutputTensor(0);
  OPS_ERR_IF(output == nullptr,
    OPS_LOG_E("aclnnfallback", "output is null"), return GRAPH_FAILED);

  auto pseShiftGe = host_api_ctx->GetOptionalInputTensor(3);
  auto attenMaskGe = host_api_ctx->GetOptionalInputTensor(4);
  auto actualSeqLengthsGe = host_api_ctx->GetOptionalInputTensor(5);

  auto actualSeqLengthsGeKv = host_api_ctx->GetOptionalInputTensor(6);
  auto deq_scale1 = host_api_ctx->GetOptionalInputTensor(7);
  auto quant_scale1 = host_api_ctx->GetOptionalInputTensor(8);
  auto deq_scale2 = host_api_ctx->GetOptionalInputTensor(9);
  auto quant_scale2 = host_api_ctx->GetOptionalInputTensor(10);
  auto quant_offset2 = host_api_ctx->GetOptionalInputTensor(11);

  std::vector<int64_t> actSeqArray;
  if (actualSeqLengthsGe != nullptr) {
    const int64_t* actSeqData = actualSeqLengthsGe->GetData<int64_t>();
    const size_t len = static_cast<size_t>(actualSeqLengthsGe->GetShapeSize());
    for (size_t i = 0; i < len; i++) {
      actSeqArray.push_back(actSeqData[i]);
    }
  }

  std::vector<int64_t> actSeqArrayKv;
  if (actualSeqLengthsGeKv != nullptr) {
    const int64_t* actSeqData = actualSeqLengthsGeKv->GetData<int64_t>();
    const size_t len = static_cast<size_t>(actualSeqLengthsGeKv->GetShapeSize());
    for (size_t i = 0; i < len; i++) {
      actSeqArrayKv.push_back(actSeqData[i]);
    }
  }

  auto attrs = host_api_ctx->GetAttrs();
  const uint32_t* getNumHeads = attrs->GetAttrPointer<uint32_t>(0);
  const float* scaleValue = attrs->GetAttrPointer<float>(1);
  const int64_t* getPreTokens = attrs->GetAttrPointer<int64_t>(2);
  const int64_t* getNextTokens = attrs->GetAttrPointer<int64_t>(3);
  const char* layout = attrs->GetAttrPointer<char>(4);
  const uint32_t* getKVHeadNum = attrs->GetAttrPointer<uint32_t>(5);
  const uint32_t* getSparseMode = attrs->GetAttrPointer<uint32_t>(6);
  const uint32_t* getInnerPrecise = attrs->GetAttrPointer<uint32_t>(7);

  int64_t numHeads = *getNumHeads;
  double dScaleValue = *scaleValue;
  int64_t preTokens = *getPreTokens;
  int64_t nextTokens = *getNextTokens;
  int64_t kvHeadNum = *getKVHeadNum;
  int64_t sparseMode = *getSparseMode;
  int64_t innerPrecise = *getInnerPrecise;

  if (innerPrecise < 0 || innerPrecise > 3) {   // innerPrecise=2,3 corresponds to high-precision mode with invalid rows.
    OPS_LOG_E("aclnnfallback", "invalid innerPrecise(%ld). Only support 0~3 now.", innerPrecise);
    return GRAPH_FAILED;
  }
  OPS_LOG_D("aclnnFallback",
          "PromptFlashAttentionV3 fallback begin, numHeads = %ld, dScaleValue = %lf",
          numHeads, dScaleValue);
  OPS_LOG_D("aclnnFallback",
          "preTokens = %ld, nextTokens = %ld, kvHeadNum = %ld, sparseMode = %ld, innerPrecise = %ld",
          preTokens, nextTokens, kvHeadNum, sparseMode, innerPrecise);

  if (sparseMode >= 10 && sparseMode <= 14) {  // 10: min  14: max 
    innerPrecise = 0;
    sparseMode -= 10;  // subtract 10 to modify sparseMode
    OPS_LOG_D("aclnnFallback",
            "because sparseMode in range [10, 14], after modification, sparseMode = %ld, innerPrecise = %ld.",
            sparseMode, innerPrecise);
  }

  auto api_ret = EXEC_OPAPI_CMD(aclnnPromptFlashAttentionV3, query, key, value, pseShiftGe, attenMaskGe, actSeqArray,
                                actSeqArrayKv, deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2,
                                numHeads, dScaleValue, preTokens, nextTokens, layout, kvHeadNum, sparseMode,
                                innerPrecise, output);

  OPS_ERR_IF(api_ret != GRAPH_SUCCESS,
    OPS_LOG_E("aclnnfallback", "api_ret faild:%u", api_ret), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

IMPL_OP(PromptFlashAttention).OpExecuteFunc(PromptHostExecuteFunc).HostInputs({5, 6});
}  // namespace fallback

#ifdef __cplusplus
}
#endif
