/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_OP_GROUPED_MATMUL_OP_H
#define OP_API_INC_LEVEL0_OP_GROUPED_MATMUL_OP_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensorList *GroupedMatmul(const aclTensorList *x,
                                   const aclTensorList *weight,
                                   const aclTensorList *biasOptional,
                                   const aclTensorList *scaleOptional,
                                   const aclTensorList *offsetOptional,
                                   const aclTensorList *antiquantScaleOptional,
                                   const aclTensorList *antiquantOffsetOptional,
                                   const aclTensor *groupListOptional,
                                   const aclTensor *perTokenScaleOptional,
                                   int64_t splitItem,
                                   op::DataType yDtype,
                                   bool transposeWeight,
                                   bool transposeX,
                                   int64_t groupType,
                                   int64_t groupListType,
                                   int64_t actType,
                                   size_t outLength,
                                   aclOpExecutor *executor);
}

#endif