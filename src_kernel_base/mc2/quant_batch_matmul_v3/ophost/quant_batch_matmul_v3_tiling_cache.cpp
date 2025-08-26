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
 * \file quant_batch_matmul_v3_tiling_cache.cc
 * \brief function of tiling_cache
 */
#include "quant_batch_matmul_v3_tiling_cache.h"

namespace optiling {
QuantBatchMatmulV3HashInput::QuantBatchMatmulV3HashInput(const QuantBatchMatmulInfo &params)
{
    mSize = params.mSize;
    mSizePerNpu = params.mSizePerNpu;
    kSize = params.kSize;
    nSize = params.nSize;

    batchA1 = params.batchA1;
    batchA2 = params.batchA2;
    batchA3 = params.batchA3;
    batchA4 = params.batchA4;

    batchB1 = params.batchB1;
    batchB2 = params.batchB2;
    batchB3 = params.batchB3;
    batchB4 = params.batchB4;

    batchBias = params.batchBias;

    aDtype = static_cast<int32_t>(params.aDtype);
    bDtype = static_cast<int32_t>(params.bDtype);
    cDtype = static_cast<int32_t>(params.cDtype);
    biasDtype = static_cast<int32_t>(params.biasDtype);
    scaleDtype = static_cast<int32_t>(params.scaleDtype);
    outDtype = static_cast<int32_t>(params.outDtype);

    bitField.transA = static_cast<uint16_t>(params.transA);
    bitField.transB = static_cast<uint16_t>(params.transB);
    bitField.hasBias = static_cast<uint16_t>(params.hasBias);
    bitField.aFormatNd = static_cast<uint16_t>(params.aFormat == ge::FORMAT_ND);
    bitField.bFormatNd = static_cast<uint16_t>(params.bFormat == ge::FORMAT_ND);
    bitField.cFormatNd = static_cast<uint16_t>(params.cFormat == ge::FORMAT_ND);

    bitField.isPerTensor = static_cast<uint16_t>(params.isPerTensor);
    bitField.isPertoken = static_cast<uint16_t>(params.isPertoken);

    bitField.reserved = 0;
}
}  // namespace optiling

