/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_moe_init_routing_v2.cpp
 * \brief
 */
#include <string>
#include "graph/types.h"
#include "aclnn_moe_init_routing_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMoeInitRoutingV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIdx, int64_t activeNumOptional, int64_t expertCapacityOptional,
    int64_t expertNumOptional, int64_t dropPadModeOptional, int64_t expertTokensCountOrCumsumFlagOptional,
    bool expertTokensBeforeCapacityFlagOptional, const aclTensor *expandedXOut, const aclTensor *expandedRowIdxOut,
    const aclTensor *expertTokensCountOrCumsumOutOptional, const aclTensor *expertTokensBeforeCapacityOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMoeInitRoutingV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream);

aclnnStatus aclnnMoeInitRoutingV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIdx, int64_t activeNumOptional, int64_t expertCapacityOptional,
    int64_t expertNumOptional, int64_t dropPadModeOptional, int64_t expertTokensCountOrCumsumFlagOptional,
    bool expertTokensBeforeCapacityFlagOptional, const aclTensor *expandedXOut, const aclTensor *expandedRowIdxOut,
    const aclTensor *expertTokensCountOrCumsumOutOptional, const aclTensor *expertTokensBeforeCapacityOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerMoeInitRoutingV2GetWorkspaceSize(
        x, expertIdx, activeNumOptional, expertCapacityOptional, expertNumOptional, dropPadModeOptional,
        expertTokensCountOrCumsumFlagOptional, expertTokensBeforeCapacityFlagOptional, expandedXOut, expandedRowIdxOut,
        expertTokensCountOrCumsumOutOptional, expertTokensBeforeCapacityOutOptional, workspaceSize, executor);
}

aclnnStatus aclnnMoeInitRoutingV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnInnerMoeInitRoutingV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
