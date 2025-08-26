/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_quant_matmul_all_reduce_v2.cpp
 * \brief
 */
#include "aclnn_quant_matmul_all_reduce_v2.h"
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
#include "hccl_util.h"
#include "matmul_all_reduce_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMatmulAllReduce(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             const aclrtStream stream);
extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);

aclnnStatus aclnnQuantMatmulAllReduceV2GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                        const aclTensor *biasOptional, const aclTensor *x3Optional,
                                                        const aclTensor *dequantScale,
                                                        const aclTensor *pertokenScaleOptional, const char* group,
                                                        const char *reduceOp, int64_t commTurn,
                                                        int64_t streamMode, const aclTensor *output,
                                                        uint64_t *workspaceSize, aclOpExecutor **executor) {
  uint64_t timeStamp = NnopbaseMsprofSysTime();
  // 固定写法，参数检查
  auto retParam = QuantMatmulAllReduceCheckParams(x1, x2, biasOptional, dequantScale, pertokenScaleOptional,
                              x3Optional, reduceOp, streamMode, output);
  CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
  // dequantScale转为uint64
  auto dequant = const_cast<aclTensor*>(dequantScale);
  if (dequant == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantMatmulAllReduce, dequant is nullptr.");
    return ACLNN_ERR_INNER;
  }
  if (dequant->GetDataType() == op::DataType::DT_INT64) {
    dequant->SetDataType(op::DataType::DT_UINT64);
  }
  // 处理空tensor,x1,x2不为空，dequantscale为空也报错，bias、x3可选不做判断
  if (x1->IsEmpty() || x2->IsEmpty() || dequant->IsEmpty()) {
    // 根据实际支持情况补充
    OP_LOGD("QuantMatmulAllReduceV2, dealing with empty tensor.");
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  aclnnStatus ret = InnerQuantMatmulAllReduceGetWorkspaceSize(x1, x2, biasOptional, x3Optional, dequant,
                                                                pertokenScaleOptional, group, reduceOp, commTurn,
                                                                output, workspaceSize, executor);
  OP_LOGD("QuantMatmulAllReduceV2, end ret %d", ret);
  static NnopbaseDfxId dfxId = {0x60000, __func__, false};
  NnopbaseReportApiInfo(timeStamp, dfxId);
  return ret;
}

aclnnStatus aclnnQuantMatmulAllReduceV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        const aclrtStream stream) {
  uint64_t timeStamp = NnopbaseMsprofSysTime();
  if (workspace == nullptr || workspaceSize == 0UL) {
    OP_LOGD("Skip the api for empty tensor, workspace addr %p, size %lu.", workspace, workspaceSize);
    return ACLNN_SUCCESS;
  }

  aclnnStatus ret = aclnnInnerMatmulAllReduce(workspace, workspaceSize, executor, stream);
  OP_API_CHECK(ret != ACLNN_SUCCESS, {
      OP_LOGE(ACLNN_ERR_INNER, "QuantMatmulAllReduceV2LaunchTask fail, ret: %d.", ret);
      return ACLNN_ERR_INNER;
  });
  static NnopbaseDfxId dfxId = {0x60000, __func__, false};
  NnopbaseReportApiInfo(timeStamp, dfxId);
  return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
