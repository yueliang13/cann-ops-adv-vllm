/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sparse_paged_fusion_attention.h"
#include "graph/types.h"

#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnInnerSparsePagedFusionAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclTensor *deqScale1,
    const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blocktable, const aclTensor *kvPaddingSize,
    const aclTensor *l1_cent, const aclTensor *block_ids, const aclTensor *total_seq_len,
    int64_t numHeads, double scaleValue, char *inputLayout, int64_t numKeyValueHeads, int64_t blockSize, int64_t innerPrecise,
    const aclTensor *block_position, const aclTensor *page_position_length, const aclTensor *max_page_position_length,// 新增输出参数 也准备用来存中间变量
    const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerSparsePagedFusionAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  const aclrtStream stream);

aclnnStatus aclnnSparsePagedFusionAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclTensor *deqScale1,
    const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blocktable, const aclTensor *kvPaddingSize,
    const aclTensor *l1_cent, const aclTensor *block_ids, const aclTensor *total_seq_len,
    int64_t numHeads,double scaleValue, char *inputLayout, int64_t numKeyValueHeads, int64_t blockSize, int64_t innerPrecise,
    const aclTensor *block_position, const aclTensor *page_position_length, const aclTensor *max_page_position_length,// 新增输出参数 也准备用来存中间变量
    const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus ret = aclnnInnerSparsePagedFusionAttentionGetWorkspaceSize(
        query, key, value, pseShift, attenMask, actualSeqLengths, deqScale1, quantScale1, deqScale2, quantScale2,
        quantOffset2, antiquantScale, antiquantOffset, blocktable, kvPaddingSize,
        l1_cent, block_ids, total_seq_len,
        numHeads, scaleValue, inputLayout, numKeyValueHeads, blockSize, innerPrecise,
        block_position, page_position_length, max_page_position_length,
        attentionOut, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnSparsePagedFusionAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       const aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerSparsePagedFusionAttention(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif