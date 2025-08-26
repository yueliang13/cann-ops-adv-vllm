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
 * \file rotary_position_embedding_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_GRAD_H

#ifdef ASCENDC_OP_TEST
#define ROPE_GRAD_EXTERN_C extern "C"
#else
#define ROPE_GRAD_EXTERN_C
#endif

#include "register/tilingdata_base.h"
namespace optiling {

BEGIN_TILING_DATA_DEF(RopeHalfGradParams)
    TILING_DATA_FIELD_DEF(uint64_t, layout);
    TILING_DATA_FIELD_DEF(uint64_t, xShapeSize);
    TILING_DATA_FIELD_DEF(uint64_t, cosShapeSize);
    TILING_DATA_FIELD_DEF(uint64_t, dimB);
    TILING_DATA_FIELD_DEF(uint64_t, dimS);
    TILING_DATA_FIELD_DEF(uint64_t, dimN);
    TILING_DATA_FIELD_DEF(uint64_t, dimD);
    TILING_DATA_FIELD_DEF(uint64_t, cosDimB);
    TILING_DATA_FIELD_DEF(uint64_t, cosDimN);
    TILING_DATA_FIELD_DEF(uint64_t, halfDimDAlignNum);

    TILING_DATA_FIELD_DEF(uint64_t, coreData);
    TILING_DATA_FIELD_DEF(uint64_t, coreLast);
    TILING_DATA_FIELD_DEF(uint64_t, copyLoop);
    TILING_DATA_FIELD_DEF(uint64_t, copyTail);
    TILING_DATA_FIELD_DEF(uint64_t, lastCopyLoop);
    TILING_DATA_FIELD_DEF(uint64_t, lastCopyTail);
    TILING_DATA_FIELD_DEF(uint64_t, alignUbSize);
    TILING_DATA_FIELD_DEF(uint64_t, calcUbSize);
    TILING_DATA_FIELD_DEF(uint64_t, coreUsed);
    TILING_DATA_FIELD_DEF(uint64_t, coreNum);

    TILING_DATA_FIELD_DEF(uint64_t, firstReduce);
    TILING_DATA_FIELD_DEF(uint64_t, secondReduce);
    TILING_DATA_FIELD_DEF(uint64_t, ubLoopGap);
    TILING_DATA_FIELD_DEF(uint64_t, blockLenInner);
    TILING_DATA_FIELD_DEF(uint64_t, strideInner);
    TILING_DATA_FIELD_DEF(uint64_t, blockLenPadInner);
    TILING_DATA_FIELD_DEF(uint64_t, stridePadInner);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RopeHalfGradParamsOp, RopeHalfGradParams)

BEGIN_TILING_DATA_DEF(RopeInterleavedGradParams)
  TILING_DATA_FIELD_DEF(uint64_t, batchSize);
  TILING_DATA_FIELD_DEF(uint64_t, seqLen);
  TILING_DATA_FIELD_DEF(uint64_t, numHeads);
  TILING_DATA_FIELD_DEF(uint64_t, headDim);
  TILING_DATA_FIELD_DEF(uint64_t, alignHeadDim);
  TILING_DATA_FIELD_DEF(uint64_t, padHeadDim);
  
  TILING_DATA_FIELD_DEF(uint64_t, frontCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, seqFrontLen);
  TILING_DATA_FIELD_DEF(uint64_t, seqTailLen);

  TILING_DATA_FIELD_DEF(uint64_t, seqFrontCalcNum);
  TILING_DATA_FIELD_DEF(uint64_t, seqFrontCalcLoop);
  TILING_DATA_FIELD_DEF(uint64_t, seqFrontCalcTail);

  TILING_DATA_FIELD_DEF(uint64_t, seqTailCalcNum);
  TILING_DATA_FIELD_DEF(uint64_t, seqTailCalcLoop);
  TILING_DATA_FIELD_DEF(uint64_t, seqTailCalcTail);
  
  TILING_DATA_FIELD_DEF(uint64_t, numHeadsLength);
  TILING_DATA_FIELD_DEF(uint64_t, numHeadsLoop);
  TILING_DATA_FIELD_DEF(uint64_t, numHeadsTail);

  TILING_DATA_FIELD_DEF(uint64_t, batchNumHeadsLength);
  TILING_DATA_FIELD_DEF(uint64_t, batchNumHeadsLoop);
  TILING_DATA_FIELD_DEF(uint64_t, batchNumHeadsTail);

  TILING_DATA_FIELD_DEF(uint64_t, layout);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RopeInterleavedGradParamsOp, RopeInterleavedGradParams)


BEGIN_TILING_DATA_DEF(RotaryPositionEmbeddingGradTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(RopeHalfGradParams, ropeHalfGradParams);
  TILING_DATA_FIELD_DEF_STRUCT(RopeInterleavedGradParams, ropeInterleavedGradParams);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RotaryPositionEmbeddingGrad, RotaryPositionEmbeddingGradTilingData)

struct RotaryPositionEmbeddingGradCompileInfo {
  int64_t totalCoreNum = 0;
  int64_t totalUbSize = 0;
};

}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_GRAD_H