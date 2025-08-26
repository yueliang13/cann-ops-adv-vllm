/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_PROMPT_FLASH_ATTENTION_INNER_H_
#define OP_API_INC_LEVEL2_ACLNN_PROMPT_FLASH_ATTENTION_INNER_H_
#define ACLNN_API __attribute__((visibility("default")))

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnInnerPromptFlashAttention is used to calculate the workspace size based on the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInnerPromptFlashAttentionGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* pseShift,
    const aclTensor* attenMask, const aclIntArray* actualSeqLengths, const aclIntArray* actualSeqLengthsKv,
    const aclTensor* deqScale1, const aclTensor* quantScale1, const aclTensor* deqScale2, const aclTensor* quantScale2,
    const aclTensor* quantOffset2, int64_t numHeads, double scaleValue, int64_t preTokens, int64_t nextTokens,
    char* inputLayout, int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    const aclTensor* attentionOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInnerPromptFlashAttention is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnInnerPromptFlashAttention(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                     const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif /* OP_API_INC_LEVEL2_ACLNN_PROMPT_FLASH_ATTENTION_INNER_H_ */