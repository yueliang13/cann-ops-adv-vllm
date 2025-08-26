/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_SRC_LEVEL2_MATMUL_ALL_REDUCE_UTIL_H_
#define OP_API_SRC_LEVEL2_MATMUL_ALL_REDUCE_UTIL_H_

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "acl/acl.h"
#include "hcom_topo_info.h"
#include "hccl_util.h"

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t FOUR_DIMS = 4;
constexpr size_t THREE_DIMS = 3;
constexpr size_t TWO_DIMS = 2;
constexpr size_t ONE_DIM = 1;
constexpr int64_t NUM_ACL_STOP_ON_FAILURE = 1;
constexpr uint8_t MC2_DEBUG_ONLY_CUBE = 1;
constexpr char DEBUG_MODE_ENV[] = "ASCEND_MC2_DEBUG_MODE";
constexpr size_t DIM_LEN_ONE = 1;
constexpr size_t DIM_LEN_TWO = 2;

struct NnopbaseDfxId {
    uint32_t id;
    const char *funcName;
    bool hasReg;
};

aclnnStatus MatmulAllReduceCheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *x3,
                                        const aclTensor *bias, const char *reduceOp, int64_t streamMode,
                                        const aclTensor *output);
aclnnStatus QuantMatmulAllReduceCheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                            const aclTensor *dequantScale, const aclTensor *pertokenScale,
                                            const aclTensor *x3, const char *reduceOp, int64_t streamMode,
                                            const aclTensor *output);
bool MatmulAllReduceCheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* output);
bool MatmulAllReduceCheckAttr(const char *reduceOp, int64_t streamMode);

bool MatmulAllReduceCheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3,
                                    const aclTensor* bias, const aclTensor* output);
bool QuantMatmulAllReduceCheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                            const aclTensor *dequantScale, const aclTensor *pertokenScale,
                                            const aclTensor *x3, const aclTensor* output);
bool MatmulAllReduceCheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3, const aclTensor* bias,
                                const aclTensor* output);
bool QuantMatmulAllReduceCheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                    const aclTensor *dequantScale, const aclTensor *pertokenScale, const aclTensor *x3,
                                    const aclTensor* output);
bool MatmulAllReduceIsWeightNZFormat(const aclTensor* x2);
bool QuantMatmulAllReduceIsWeightNZFormat(const aclTensor* x2);

// 全量化
bool QuantMatmulAllReduceIsAclnnPreTransposed(const aclTensor* x2);
void QuantMatmulAllReduceProcessTransposedX2(const aclTensor* x2, uint64_t& x2Dim0, uint64_t& x2Dim1,
                                                ge::AscendString& x2ShapeStr);
bool QuantMatmulAllReduceCheckPertokenScaleShape(const aclTensor *pertokenScale, const aclTensor* x1,
                                                    const size_t x1Len);

aclTensor *QuantMatmulAllReduceCopyTensor(const aclTensor *x2);

// MatmulAllReduceAddRmsNorm
bool ArnCheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* residual,
                     const aclTensor* gamma);
bool ArnCheckShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *residual);

aclnnStatus InnerQuantMatmulAllReduceGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                      const aclTensor *biasOptional, const aclTensor *x3Optional,
                                                      const aclTensor *dequant,
                                                      const aclTensor *pertokenScaleOptional, const char* group,
                                                      const char *reduceOp, int64_t commTurn,
                                                      const aclTensor *output,
                                                      uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus InnerMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                            const aclTensor *bias, const aclTensor *antiquantScale,
                                                            const aclTensor *antiquantOffset,
                                                            const aclTensor *dequant, const aclTensor *residual,
                                                            const aclTensor *gamma, double epsilon,
                                                            const char *group, const char *reduceOp,
                                                            int64_t commTurn, int64_t antiquantGroupSize,
                                                            const aclTensor *y, const aclTensor *normOut,
                                                            uint64_t *workspaceSize, aclOpExecutor **executor);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_SRC_LEVEL2_MATMUL_ALL_REDUCE_UTIL_H_