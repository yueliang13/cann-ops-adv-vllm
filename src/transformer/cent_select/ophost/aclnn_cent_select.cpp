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
#include "aclnn_cent_select.h"

#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnInnerCentSelectGetWorkspaceSize(
    const aclTensor *query, const aclTensor *l1_cent, const aclTensor *block_ids, const aclTensor *block_table, const aclTensor *seq_len, const aclTensor *page_position, const aclTensor *page_position_length,
    const aclTensor *workspace, const aclTensor *tiling,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerCentSelect(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                 const aclrtStream stream);

aclnnStatus aclnnCentSelectGetWorkspaceSize(const aclTensor *query, const aclTensor *l1_cent, const aclTensor *block_ids, const aclTensor *block_table, const aclTensor *seq_len, const aclTensor *page_position, const aclTensor *page_position_length, const aclTensor *workspace, const aclTensor *tiling, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus ret = aclnnInnerCentSelectGetWorkspaceSize(
        query, l1_cent, block_ids, block_table, seq_len, page_position, page_position_length, workspace, tiling,
        workspaceSize, executor);

    return ret;
}

aclnnStatus aclnnCentSelect(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerCentSelect(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
