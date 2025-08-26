/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_SELECT_POSITION_H_
#define ACLNN_SELECT_POSITION_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnIncreFlashAttention的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * funtion: aclnnIncreFlashAttentionGetWorkspaceSize
 * param [in] query : required
 * param [in] l1_cent : required
 * param [in] mask_empty : required
 * @param [out] d_l1_cent : required
 * @param [out] select_nprobe : required
 * @param [out] indices : required
 * param [in] workspace : required
 * param [in] tiling : required
 * @param [out] workspaceSize : size of workspace(output).
 * @param [out] executor : executor context(output).
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnSelectPositionGetWorkspaceSize(
    const aclTensor *key_ids, const aclTensor *indices, const aclTensor *token_position, const aclTensor *token_position_length,
    const aclTensor *workspace, const aclTensor *tiling,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * funtion: aclnnSelectPosition
 * param [in] workspace : workspace memory addr(input).
 * param [in] workspaceSize : size of workspace(input).
 * param [in] executor : executor context(input).
 * param [in] stream : acl stream.
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnSelectPosition(void *workspace, uint64_t workspaceSize,
                                                                            aclOpExecutor *executor,
                                                                            const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
