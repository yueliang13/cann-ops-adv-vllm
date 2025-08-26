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
 * \file rotary_position_embedding_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_H
#ifdef ASCENDC_OP_TEST
#define ROPE_EXTERN_C extern "C"
#else
#define ROPE_EXTERN_C
#endif

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RotateHalfParams)
  TILING_DATA_FIELD_DEF(uint64_t, tilingMode);  // layout code
  TILING_DATA_FIELD_DEF(uint64_t, gmLength);
  TILING_DATA_FIELD_DEF(uint64_t, broadcastFirstDim);  // B
  TILING_DATA_FIELD_DEF(uint64_t, broadcastSecondDim);  // N
  TILING_DATA_FIELD_DEF(uint64_t, dLength);  // D dim length
  TILING_DATA_FIELD_DEF(uint64_t, halfDLength);  // D/2
  TILING_DATA_FIELD_DEF(uint64_t, halfDPadLength);  // D/2 pads block aligned length
  TILING_DATA_FIELD_DEF(uint64_t, dPadLength);  // D pads block aligned length when D/2 is not aligned
  TILING_DATA_FIELD_DEF(uint64_t, isAligned);  // is D/2 block aligned, 0: false; 1: true
  TILING_DATA_FIELD_DEF(uint64_t, totalSLines);  // S dim length
  TILING_DATA_FIELD_DEF(uint64_t, storeSLines);  // number of S lines ub can store
  TILING_DATA_FIELD_DEF(uint64_t, storeDataLength);
  TILING_DATA_FIELD_DEF(uint64_t, storePadDataLength);
  // former cores tiling data
  TILING_DATA_FIELD_DEF(uint64_t, formerCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, formerSLines);  // number of S lines in each former core
  TILING_DATA_FIELD_DEF(uint64_t, formerUbLoop);  // ub loop times in former cores
  TILING_DATA_FIELD_DEF(uint64_t, formerUbLast);  // S lines left after ub loop
  TILING_DATA_FIELD_DEF(uint64_t, formerXDataLength);  // x data length in each former core
  TILING_DATA_FIELD_DEF(uint64_t, formerRDataLength);  // r data length in each former core
  TILING_DATA_FIELD_DEF(uint64_t, formerXCoreOffset);  // x total data length former core
  TILING_DATA_FIELD_DEF(uint64_t, formerRCoreOffset);  // r total data length former core
  TILING_DATA_FIELD_DEF(uint64_t, formerUbLastDataLength);  // ub last processed data length
  TILING_DATA_FIELD_DEF(uint64_t, formerUbLastPadDataLength);  // ub last processed pad data length
  // tail cores tiling data
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailSLines);
  TILING_DATA_FIELD_DEF(uint64_t, tailUbLoop);
  TILING_DATA_FIELD_DEF(uint64_t, tailUbLast);
  TILING_DATA_FIELD_DEF(uint64_t, tailXDataLength);
  TILING_DATA_FIELD_DEF(uint64_t, tailRDataLength);
  TILING_DATA_FIELD_DEF(uint64_t, tailUbLastDataLength);
  TILING_DATA_FIELD_DEF(uint64_t, tailUbLastPadDataLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RotateHalfParamsOp, RotateHalfParams)

BEGIN_TILING_DATA_DEF(RopeInterleavedParams)
  TILING_DATA_FIELD_DEF(uint64_t, batchSize);
  TILING_DATA_FIELD_DEF(uint64_t, seqLen);
  TILING_DATA_FIELD_DEF(uint64_t, numHeads);
  TILING_DATA_FIELD_DEF(uint64_t, headDim);
  TILING_DATA_FIELD_DEF(uint64_t, frontCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, coreCalcNum);
  TILING_DATA_FIELD_DEF(uint64_t, coreCalcTail);
  TILING_DATA_FIELD_DEF(uint64_t, ubCalcNum);
  TILING_DATA_FIELD_DEF(uint64_t, ubCalcLoop);
  TILING_DATA_FIELD_DEF(uint64_t, ubCalcTail);
  TILING_DATA_FIELD_DEF(uint64_t, ubCalcTailNum);
  TILING_DATA_FIELD_DEF(uint64_t, ubCalcTailLoop);
  TILING_DATA_FIELD_DEF(uint64_t, ubCalcTailTail);
  TILING_DATA_FIELD_DEF(uint64_t, ubCalcBNum);
  TILING_DATA_FIELD_DEF(uint64_t, ubCalcBLoop);
  TILING_DATA_FIELD_DEF(uint64_t, ubCalcBTail);
  TILING_DATA_FIELD_DEF(uint64_t, ubCalcNNum);
  TILING_DATA_FIELD_DEF(uint64_t, ubCalcNLoop);
  TILING_DATA_FIELD_DEF(uint64_t, ubCalcNTail);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RopeInterleavedParamsOp, RopeInterleavedParams)


BEGIN_TILING_DATA_DEF(RotaryPositionEmbeddingTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(RotateHalfParams, rotateHalfParams);
  TILING_DATA_FIELD_DEF_STRUCT(RopeInterleavedParams, ropeInterleavedParams);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RotaryPositionEmbedding, RotaryPositionEmbeddingTilingData)

struct RotaryPositionEmbeddingCompileInfo {
  int64_t totalCoreNum = 0;
  int64_t totalUbSize = 0;
};

}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_H
