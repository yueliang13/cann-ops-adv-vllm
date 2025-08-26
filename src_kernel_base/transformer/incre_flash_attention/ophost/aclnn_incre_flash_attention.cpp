/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string.h>
#include "graph/types.h"
#include "aclnn_incre_flash_attention.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerIncreFlashAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclTensor *deqScale1,
    const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blocktable, const aclTensor *kvPaddingSize, int64_t numHeads, double scaleValue, char *inputLayout,
    int64_t numKeyValueHeads, int64_t blockSize, int64_t innerPrecise, const aclTensor *attentionOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerIncreFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                 const aclrtStream stream);

aclnnStatus aclnnIncreFlashAttentionGetWorkspaceSize(const aclTensor *query, const aclTensorList *key,
                                                     const aclTensorList *value, const aclTensor *pseShift,
                                                     const aclTensor *attenMask, const aclIntArray *actualSeqLengths,
                                                     int64_t numHeads, double scaleValue, char *inputLayout,
                                                     int64_t numKeyValueHeads, const aclTensor *attentionOut,
                                                     uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus ret = aclnnInnerIncreFlashAttentionGetWorkspaceSize(
        query, key, value, nullptr, attenMask, actualSeqLengths, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, numHeads, scaleValue, inputLayout, numKeyValueHeads, 0, 1, attentionOut,
        workspaceSize, executor);

    return ret;
}

aclnnStatus aclnnIncreFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerIncreFlashAttention(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
