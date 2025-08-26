/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_FLASH_ATTENTION_SCORE_GRAD_H_
#define OP_API_INC_FLASH_ATTENTION_SCORE_GRAD_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFlashAttentionScoreGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionScoreGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    double scaleValueOptional, double keepProbOptional, int64_t preTokensOptional, int64_t nextTokensOptional,
    int64_t headNum, char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScoreGrad的第二段接口，用于执行计算。
 */
aclnnStatus aclnnFlashAttentionScoreGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         const aclrtStream stream);

/**
 * @brief aclnnFlashAttentionUnpaddingScoreGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional, double scaleValueOptional,
    double keepProbOptional, int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum,
    char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional, const aclTensor *dqOut,
    const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionUnpaddingScoreGrad的第二段接口，用于执行计算。
 */
aclnnStatus aclnnFlashAttentionUnpaddingScoreGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  const aclrtStream stream);


/**
 * @brief aclnnFlashAttentionScoreGradV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionScoreGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValueOptional,
    double keepProbOptional, int64_t preTokensOptional, int64_t nextTokensOptional,
    int64_t headNum, char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional,
    int64_t pseTypeOptional, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScoreGradV2的第二段接口，用于执行计算。
*/
aclnnStatus aclnnFlashAttentionScoreGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream);

/**
 * @brief aclnnFlashAttentionUnpaddingScoreGradV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValueOptional,
    double keepProbOptional, int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum,
    char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional, int64_t pseTypeOptional,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionUnpaddingScoreGradV2的第二段接口，用于执行计算。
*/
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_FLASH_ATTENTION_SCORE_GRAD_H_
