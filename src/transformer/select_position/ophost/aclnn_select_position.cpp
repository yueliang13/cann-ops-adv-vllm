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
#include "aclnn_select_position.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerSelectPositionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *l1_cent, const aclTensor *d_l1_cent, const aclTensor *mask_empty,
    const aclTensor *select_nprobe, const aclTensor *indices, const aclTensor *workspace, const aclTensor *tiling,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerSelectPosition(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                 const aclrtStream stream);

aclnnStatus aclnnSelectPositionGetWorkspaceSize(const aclTensor *query, const aclTensor *l1_cent, const aclTensor *d_l1_cent, const aclTensor *mask_empty, const aclTensor *select_nprobe, const aclTensor *indices, const aclTensor *workspace, const aclTensor *tiling, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus ret = aclnnInnerSelectPositionGetWorkspaceSize(
        query, l1_cent, d_l1_cent, mask_empty, select_nprobe, indices, workspace, tiling,
        workspaceSize, executor);

    return ret;
}

aclnnStatus aclnnSelectPosition(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerSelectPosition(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
