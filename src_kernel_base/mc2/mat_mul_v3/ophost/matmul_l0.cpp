/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(MatMulV2);
OP_TYPE_REGISTER(MatMulV3);

static const aclTensor *MatMulV3Common(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                const op::DataType output_dtype, const op::Format output_format,
                                const op::Format output_ori_format, const bool transposeX1, const bool transposeX2,
                                const bool offsetX, const bool enableHf32, aclOpExecutor *executor) {
  const aclTensor *offsetW = nullptr;
  L0_DFX(MatMulV3Common, x1, x2, transposeX1, transposeX2, offsetX, enableHf32);
  auto mm_out = executor->AllocTensor(output_dtype, output_format, output_ori_format);
  auto ret = INFER_SHAPE(MatMulV3, OP_INPUT(x1, x2, bias, offsetW), OP_OUTPUT(mm_out),
                         OP_ATTR(transposeX1, transposeX2, offsetX, enableHf32));
  if (ret != ACLNN_SUCCESS) {
    OPS_LOG_E(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
    return nullptr;
  }
  uint32_t execMode = enableHf32 ? static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32) : 0U;
  ret = ADD_TO_LAUNCHER_LIST_AICORE(MatMulV3, OP_INPUT(x1, x2, bias, offsetW), OP_OUTPUT(mm_out),
                                    OP_ATTR(transposeX1, transposeX2, offsetX, enableHf32), OP_MODE(execMode));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
  return mm_out;
}

static const aclTensor *MatMulCommon(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *offsetW,
                              const op::DataType output_dtype, const op::Format output_format,
                              const op::Format output_ori_format, const bool transposeX1, const bool transposeX2,
                              const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor *executor) {
  L0_DFX(MatMulCommon, x1, x2, transposeX1, transposeX2, offsetX);
  auto mm_out = executor->AllocTensor(output_dtype, output_format, output_ori_format);
  // 当前暂时不支持input size和hiddenSize两个参数的设置
  auto ret = INFER_SHAPE(MatMulV2, OP_INPUT(x1, x2, bias, offsetW), OP_OUTPUT(mm_out),
                         OP_ATTR(transposeX1, transposeX2, offsetX, -1L, -1L, opImplModeEnum, 1L));
  if (ret != ACLNN_SUCCESS) {
    OPS_LOG_E(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
    return nullptr;
  }
  uint32_t execMode = opImplModeEnum == 0x40 ? static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32) : 0U;
  ret = ADD_TO_LAUNCHER_LIST_AICORE(MatMulV2, OP_INPUT(x1, x2, bias, offsetW), OP_OUTPUT(mm_out),
                                    OP_ATTR(transposeX1, transposeX2, offsetX, -1L, -1L, opImplModeEnum, 1L),
                                    OP_MODE(execMode));

  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
  return mm_out;
}

const aclTensor *MatMulNd(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *offsetW,
                          const bool transposeX1, const bool transposeX2, const bool offsetX,
                          const int64_t opImplModeEnum, aclOpExecutor *executor) {
  L0_DFX(MatMulNd);
  return MatMulCommon(x1, x2, bias, offsetW, x1->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND, transposeX1,
                      transposeX2, offsetX, opImplModeEnum, executor);
};

const aclTensor *MatMulNdFp162Fp32(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                   const aclTensor *offsetW, const bool transposeX1, const bool transposeX2,
                                   const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor *executor) {
  L0_DFX(MatMulNdFp162Fp32);
  return MatMulCommon(x1, x2, bias, offsetW, DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND, transposeX1,
                      transposeX2, offsetX, opImplModeEnum, executor);
};

const aclTensor *MatMulNz(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *offsetW,
                          const bool transposeX1, const bool transposeX2, const bool offsetX,
                          const int64_t opImplModeEnum, aclOpExecutor *executor) {
  L0_DFX(MatMulNz);
  return MatMulCommon(x1, x2, bias, offsetW, x1->GetDataType(), Format::FORMAT_FRACTAL_NZ, Format::FORMAT_ND,
                      transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
};

const aclTensor* MatMulNzNzNd(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW,
                              const bool transposeX1, const bool transposeX2, const bool offsetX,
                              const int64_t opImplModeEnum, aclOpExecutor* executor) {
  L0_DFX(MatMulNzNzNd);
  return MatMulCommon(x1, x2, bias, offsetW, x1->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND, transposeX1,
                      transposeX2, offsetX, opImplModeEnum, executor);
};

const aclTensor *MatMulNzFp162Fp32(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                   const aclTensor *offsetW, const bool transposeX1, const bool transposeX2,
                                   const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor *executor) {
  L0_DFX(MatMulNzFp162Fp32);
  return MatMulCommon(x1, x2, bias, offsetW, DataType::DT_FLOAT, Format::FORMAT_FRACTAL_NZ, Format::FORMAT_ND,
                      transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
};

const aclTensor *MatMulNdNz(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *offsetW,
                            const bool transposeX1, const bool transposeX2, const bool offsetX,
                            const int64_t opImplModeEnum, aclOpExecutor *executor) {
  L0_DFX(MatMulNdNz);
  return MatMulCommon(x1, x2, bias, offsetW, x1->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND, transposeX1,
                      transposeX2, offsetX, opImplModeEnum, executor);
};

const aclTensor *MatMulV3Nd(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                            const bool transposeX1, const bool transposeX2, const bool offsetX, const bool enableHf32,
                            aclOpExecutor *executor) {
  L0_DFX(MatMulV3Nd);
  return MatMulV3Common(x1, x2, bias, x1->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND, transposeX1,
                        transposeX2, offsetX, enableHf32, executor);
};
}  // namespace l0op

