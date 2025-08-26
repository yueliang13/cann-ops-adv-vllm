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
 * \file basic_block_config.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_CONFIG_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_CONFIG_H

#include "../tool.h"
#include "../weight_quant_batch_matmul_v2_constant.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"


namespace WeightQuantBatchMatmulV2 {

constexpr static uint16_t WEIGHT_F16_UB_NZ_STRIDE = 65;

struct WqmmConfig {
    bool aTrans;
    bool bTrans;
    QuantType antiQuantType;
    bool hasAntiQuantOffset;
    QuantType quantType;
    CubeFormat weightFormat;
};

struct BasicBlockOffsetParam {
    uint32_t mL1Size;
    uint32_t kaL1Size;
    uint32_t kbL1Size;
    uint32_t nL1Size;

    uint64_t mOffset;
    uint64_t nOffset;

    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
};

struct VecAntiQuantConfig {
    int32_t ubMte2BufferNum = 2;
    int32_t ubMte2InnerSize = 512;
};

struct L1DbConfig {
    uint32_t aF16L1DbOffset;
    uint32_t biasL1DbOffset;
    uint32_t weightF16L1DbOffset;
};

struct UbConsumeConfig {
    uint32_t l1RequireVfComputeRealK;
    uint32_t l1RequireVfComputeRealN;
    uint32_t kWeightLowBitUbOffset;
    uint32_t nWeightLowBitUbOffset;
};

struct L1ConsumeConfig {
    uint32_t l1SplitTwoVecExternalOffset;
    uint32_t l1RealExternalLen;
};

}
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_CONFIG_H