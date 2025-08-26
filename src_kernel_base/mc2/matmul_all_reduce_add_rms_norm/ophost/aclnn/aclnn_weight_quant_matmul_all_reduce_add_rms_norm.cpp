/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_weight_quant_matmul_all_reduce_add_rms_norm.h"
#include "aclnn_inplace_weight_quant_matmul_all_reduce_add_rms_norm.h"
#include "securec.h"

#include "acl/acl.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "matmul_util.h"
#include "aclnn_kernels/contiguous.h"
#include "matmul_all_reduce_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMatmulAllReduceAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                       const aclrtStream stream);

aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *antiquantScale,
    const aclTensor *antiquantOffset, const aclTensor *residual, const aclTensor *gamma, double epsilon,
    const char *group, const char *reduceOp, int64_t commTurn, int64_t streamMode, int64_t antiquantGroupSize,
    const aclTensor *y, const aclTensor *normOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(ArnCheckNotNull(x1, x2, residual, gamma), ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(antiquantScale, return ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(ArnCheckShape(x1, x2, residual), ACLNN_ERR_PARAM_INVALID);
    // 处理空tensor,x1,x2不为空，scale为空也报错，offset、bias、可选不做判断
    if (x1->IsEmpty() || x2->IsEmpty() || antiquantScale->IsEmpty() || residual->IsEmpty() || gamma->IsEmpty()) {
        // 根据实际支持情况补充
        OP_LOGD("WeightQuantMatmulAllReduceAddRmsNorm, dealing with empty tensor.");
        // 处理k = 0 场景，报错
        int64_t kValue = x1->GetViewShape().GetDim(x1->GetViewShape().GetDimNum() - 1);
        OP_LOGD("WeightQuantMatmulAllReduceAddRmsNorm, kValue: %ld.", kValue);
        if (kValue == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "WeightQuantMatmulAllReduceAddRmsNorm does not support k = 0.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 固定写法，创建OpExecutor
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // check streamMode = NUM_ACL_STOP_ON_FAILURE
    CHECK_RET(streamMode == NUM_ACL_STOP_ON_FAILURE, ACLNN_ERR_PARAM_INVALID);
    aclnnStatus ret = InnerMatmulAllReduceAddRmsNormGetWorkspaceSize(
        x1, x2, bias, antiquantScale, antiquantOffset, nullptr, residual, gamma, epsilon, group, reduceOp, commTurn,
        antiquantGroupSize, y, normOut, workspaceSize, executor);
    OP_LOGD("WeightQuantMatmulAllReduceAddRmsNorm, end ret %d", ret);
    return ret;
}

aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                      const aclrtStream stream)
{
    if (workspace == nullptr || workspaceSize == 0UL) {
        OP_LOGD("Skip the api for empty tensor, workspace addr %p, size %lu.", workspace, workspaceSize);
        return ACLNN_SUCCESS;
    }

    aclnnStatus ret = aclnnInnerMatmulAllReduceAddRmsNorm(workspace, workspaceSize, executor, stream);
    OP_LOGD("WeightQuantMatmulAllReduceAddRmsNorm, aclnnWeightQuantMatmulAllReduceAddRmsNorm ret %d", ret);
    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "WeightQuantMatmulAllReduceAddRmsNorm, This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *antiquantScale,
    const aclTensor *antiquantOffset, const aclTensor *residual, const aclTensor *gamma, double epsilon,
    const char *group, const char *reduceOp, int64_t commTurn, int64_t streamMode, int64_t antiquantGroupSize,
    const aclTensor *normOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(
        x1, x2, bias, antiquantScale, antiquantOffset, residual, gamma, epsilon, group, reduceOp, commTurn, streamMode,
        antiquantGroupSize, residual, normOut, workspaceSize, executor);
}

aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm(void *workspace, uint64_t workspaceSize,
                                                             aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnWeightQuantMatmulAllReduceAddRmsNorm(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif