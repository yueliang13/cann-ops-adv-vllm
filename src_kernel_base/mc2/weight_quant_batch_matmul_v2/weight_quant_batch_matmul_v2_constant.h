/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_constant.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_CONSTANT_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_CONSTANT_H
namespace WeightQuantBatchMatmulV2 {
using HighPreciseType = int32_t;
using HighPerformanceType = half;
enum class QuantType {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
};

enum class PrecisionType {
    NONE = 0,
    HIGH_PRECISION = 1,
};

} // namespace WeightQuantBatchMatmulV2
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_CONSTANT_H