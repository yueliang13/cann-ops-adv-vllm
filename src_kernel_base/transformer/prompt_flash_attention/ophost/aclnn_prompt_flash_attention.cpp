/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

#include "aclnn_prompt_flash_attention.h"
#include "aclnn_prompt_flash_attention_inner.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/slice.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
aclnnStatus aclnnPromptFlashAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask,
    const aclIntArray *actualSeqLengths,
    int64_t numHeads, // q_n
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    char *inputLayout,
    int64_t numKeyValueHeads,
    const aclTensor *attentionOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor) {       
        const aclIntArray *actualSeqLengthsKv = nullptr;
        int64_t sparseMode = 0;
        int64_t innerPrecise = 1;
        const aclTensor *deqScale1 = nullptr;
        const aclTensor *quantScale1 = nullptr;
        const aclTensor *deqScale2 = nullptr;
        const aclTensor *quantScale2 = nullptr;
        const aclTensor *quantOffset2 = nullptr;
        return aclnnInnerPromptFlashAttentionGetWorkspaceSize(query, key, value, nullptr, attenMask,
                                                              actualSeqLengths, actualSeqLengthsKv,
                                                              deqScale1, quantScale1, deqScale2,
                                                              quantScale2, quantOffset2,
                                                              numHeads, scaleValue, preTokens, nextTokens,
                                                              inputLayout, numKeyValueHeads, sparseMode,
                                                              innerPrecise, attentionOut, workspaceSize, executor);
    }

aclnnStatus aclnnPromptFlashAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream) {
        // perform attention computations.
        return aclnnInnerPromptFlashAttention(workspace, workspaceSize, executor, stream);
    }

}

#ifdef __cplusplus
}
#endif