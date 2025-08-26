/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "graph/types.h"
#include "aclnn_fused_infer_attention_score.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {
extern aclnnStatus aclnnInnerFusedInferAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1, const aclTensor *quantScale1,
    const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blockTable, const aclTensor *queryPaddingSize, const aclTensor *kvPaddingSize,
    const aclTensor *keyAntiquantScale, const aclTensor *keyAntiquantOffset,
    const aclTensor *valueAntiquantScale, const aclTensor *valueAntiquantOffset,
    const aclTensor *keySharedPrefix, const aclTensor *valueSharedPrefix,
    const aclIntArray *actualSharedPrefixLen, const aclTensor *query_rope,
    const aclTensor *key_rope, const aclTensor *keyRopeAntiquantScale,
    int64_t numHeads, double scaleValue, int64_t preTokens,
    int64_t nextTokens, char *inputLayout,
    int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    int64_t keyAntiquantMode, int64_t valueAntiquantMode,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerFusedInferAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                      const aclrtStream stream);

aclnnStatus aclnnFusedInferAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1, const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blockTable, const aclTensor *queryPaddingSize, const aclTensor *kvPaddingSize, int64_t numHeads,
    double scaleValue, int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;
    if (softmaxLseFlag == false) { // Do not use softmaxLse
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), (void *)&addr);
        placeHolder = tempTensor;
    } else {
        placeHolder = softmaxLse;
    }
    aclnnStatus ret = aclnnInnerFusedInferAttentionScoreGetWorkspaceSize(
        query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKv, deqScale1, quantScale1, deqScale2,
        quantScale2, quantOffset2, antiquantScale, antiquantOffset, blockTable, queryPaddingSize, kvPaddingSize,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        numHeads, scaleValue, preTokens, nextTokens,
        inputLayout, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, 0, 0,
        attentionOut, placeHolder, workspaceSize, executor);
    if (softmaxLseFlag == false) {
        aclDestroyTensor(tempTensor);
    }
    return ret;
}

aclnnStatus aclnnFusedInferAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          const aclrtStream stream)
{
    return aclnnInnerFusedInferAttentionScore(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
