/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_H
#define OP_API_INC_GROUPED_MATMUL_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: 表示公式中的x，数据类型支持FLOAT16、BFLOAT16、INT8、FLOAT32数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] weight:
 * 表示公式中的weight，数据类型支持FLOAT16、BFLOAT16、INT8、FLOAT32数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] biasOptional:
 * 表示公式中的bias，数据类型支持FLOAT16、FLOAT32、INT32数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] scaleOptional: 表示量化参数，数据类型支持UINT64数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] offsetOptional: 表示量化参数，数据类型支持FLOAT32数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] antiquantScaleOptional:
 * 表示伪量化参数，数据类型支持FLOAT16，BFLOAT16数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] antiquantOffsetOptional:
 * 表示伪量化参数，数据类型支持FLOAT16，BFLOAT16数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] groupListOptional: 可选参数，代表输入和输出M轴上的索引情况，数据类型支持INT64，支持的最大长度为128个。
 * @param [in] splitItem:
 * 整数型参数，代表输出是否要做tensor切分，0/1代表输出为多tensor；2/3代表输出为单tensor，默认值为0。
 * @param [out] y: 表示公式中的y，数据类型支持FLOAT16、BFLOAT16、INT8、FLOAT32数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulGetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight, const aclTensorList* biasOptional,
    const aclTensorList* scaleOptional, const aclTensorList* offsetOptional,
    const aclTensorList* antiquantScaleOptional, const aclTensorList* antiquantOffsetOptional,
    const aclIntArray* groupListOptional, int64_t splitItem, const aclTensorList* y, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnGroupedMatmul的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnGtTensorGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif