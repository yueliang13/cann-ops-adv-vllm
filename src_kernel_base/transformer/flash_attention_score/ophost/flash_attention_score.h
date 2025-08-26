/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_OP_FLASH_ATTENTION_SCORE_OP_H_
#define OP_API_INC_LEVEL0_OP_FLASH_ATTENTION_SCORE_OP_H_

#include "opdev/op_executor.h"

namespace l0op {

const std::array<const aclTensor *, 4>
FlashAttentionScore(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                    const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,
                    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
                    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
                    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
                    const aclIntArray *kvStartIdxOptional, double scaleValueOptional, double keepProbOptional,
                    int64_t preTockensOptional, int64_t nextTockensOptional, int64_t headNum, const char *inputLayout,
                    int64_t innerPreciseOptional, int64_t sparseModeOptional, int64_t pseTypeOptional,
                    aclOpExecutor *executor);
}

#endif // OP_API_INC_LEVEL0_OP_FLASH_ATTENTION_SCORE_OP_H_
