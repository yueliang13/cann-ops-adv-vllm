/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_INCRE_FLASH_ATTENTION_H_
#define ACLNN_INCRE_FLASH_ATTENTION_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnIncreFlashAttention的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * funtion: aclnnIncreFlashAttentionGetWorkspaceSize
 * param [in] query : required
 * param [in] key : dynamic
 * param [in] value : dynamic
 * param [in] pseShift : optional
 * param [in] attenMask : optional
 * param [in] actualSeqLengths : optional
 * param [in] numHeads : required
 * param [in] scaleValue : optional
 * param [in] inputLayout : optional
 * param [in] numKeyValueHeads : optional
 * @param [out] attentionOut : required
 * @param [out] workspaceSize : size of workspace(output).
 * @param [out] executor : executor context(output).
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnIncreFlashAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, int64_t numHeads, double scaleValue,
    char *inputLayout, int64_t numKeyValueHeads, const aclTensor *attentionOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * funtion: aclnnIncreFlashAttention
 * param [in] workspace : workspace memory addr(input).
 * param [in] workspaceSize : size of workspace(input).
 * param [in] executor : executor context(input).
 * param [in] stream : acl stream.
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnIncreFlashAttention(void *workspace, uint64_t workspaceSize,
                                                                            aclOpExecutor *executor,
                                                                            const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
