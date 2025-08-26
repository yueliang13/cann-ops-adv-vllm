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
 * \file rope_rotate_half_tiling.cpp
 * \brief
 */
#include "rope_rotate_half_tiling.h"
#include "rotary_position_embedding_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/ops_log.h"


namespace {
const uint64_t INDEX_INPUT_X = 0;
const uint64_t INDEX_INPUT_COS = 1;
const uint64_t INDEX_INPUT_SIN = 2;
const uint64_t BYTE_PER_DATA_4 = 4;
const uint64_t BYTE_PER_DATA_2 = 2;
const uint64_t BYTE_OF_BLOCK = 32;
const uint64_t BYTE_OF_REPEAT = 256;
const uint64_t DIM_NUM = 4;
const uint64_t DIM_FIRST = 0;
const uint64_t DIM_SECOND = 1;
const uint64_t DIM_THIRD = 2;
const uint64_t DIM_FOURTH = 3;
const uint64_t D_LENGTH_LIMIT = 896;  // keep the same  support D length as the grad
const uint64_t BNSD_ALIGNED_BLOCK_S_SCALE = 8;  // empiric value
const uint64_t BNSD_UNALIGNED_BLOCK_BN_SCALE = 2;  // empiric value
const uint64_t BNSD_UNALIGNED_BLOCK_D_LENGTH = 80; // empiric value
const uint64_t NO_BROADCAST_DIM = 1;
const uint64_t UB_NUM = 4;           // ub split number for float and float16
const uint64_t UB_NUM_BF16 = 3 * 4;  // ub split number for bfloat16 dtype
const uint64_t TWO = 2;
const uint64_t SINGLE_BUFFER = 1;
const uint64_t DOUBLE_BUFFER = 2;
const uint64_t TILING_KEY_PREFIX = 1000;
const uint64_t TILING_MODE_WEIGHT = 10;
constexpr size_t UB_RESERVE_SIZE = 2 * 1024;

const uint64_t TILING_MODE_UNKNOWN = 0;
const uint64_t TILING_MODE_BNSD = 1;
const uint64_t TILING_MODE_BSND = 2;
const uint64_t TILING_MODE_SBND = 3;
const uint64_t TILING_MODE_NO_BROADCAST = 4;  // SD, B=1, N=1
const uint64_t TILING_MODE_BND = 5;           // S=1
const uint64_t TILING_MODE_R_B1SD = 6;        // r: B1SD
const uint64_t TILING_DTYPE_UNKNOWN = 0;
const uint64_t TILING_DTYPE_FP32 = 1;
const uint64_t TILING_DTYPE_FP16 = 2;
const uint64_t TILING_DTYPE_BF16 = 3;

__attribute__((always_inline)) inline uint64_t GetCeilDiv(uint64_t value1, uint64_t value2) {
  if (value2 == 0) {
    return value2;
  }
  return (value1 + value2 - 1) / value2;
}

__attribute__((always_inline)) inline uint64_t GetDiv(uint64_t value1, uint64_t value2) {
  if (value2 == 0) {
    return value2;
  }
  return value1 / value2;
}

__attribute__((always_inline)) inline uint64_t GetRem(uint64_t value1, uint64_t value2) {
  if (value2 == 0) {
    return value2;
  }
  return value1 % value2;
}

__attribute__((always_inline)) inline uint64_t GetBytePerData(const ge::DataType& dtype) {
  if (dtype == ge::DT_FLOAT) {
    return BYTE_PER_DATA_4;
  }
  if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
    return BYTE_PER_DATA_2;
  }
  return BYTE_PER_DATA_4;
}

__attribute__((always_inline)) inline uint64_t GetTilingDtype(const ge::DataType& dtype) {
  if (dtype == ge::DT_FLOAT) {
    return TILING_DTYPE_FP32;
  }
  if (dtype == ge::DT_FLOAT16) {
    return TILING_DTYPE_FP16;
  }
  if (dtype == ge::DT_BF16) {
    return TILING_DTYPE_BF16;
  }
  return TILING_DTYPE_UNKNOWN;
}

__attribute__((always_inline)) inline uint64_t GetTilingKey(uint64_t tilingMode, uint64_t tilingDtype) {
  return TILING_KEY_PREFIX + tilingMode * TILING_MODE_WEIGHT + tilingDtype;
}
}  // namespace

namespace optiling {
class RotateHalfTiling {
 public:
  explicit RotateHalfTiling(gert::TilingContext* context) : context(context){};
  uint64_t coreNum = 0;
  uint64_t tilingDtype = TILING_DTYPE_UNKNOWN;
  uint64_t tilingKey = TILING_KEY_PREFIX;
  RotaryPositionEmbeddingTilingData tiling;
  RotateHalfParams& tilingData_ = tiling.rotateHalfParams;
  ge::graphStatus DoRotateHalfTiling();

 private:
  gert::TilingContext* context = nullptr;
  inline void GetCoreDataTiling();
  inline void PrintTilingParams();
  inline void GetUbLoopTiling();
  inline bool CheckBnsdBlockSkip();
  inline void GetAlignedInfo(const ge::DataType inputDtype, uint64_t dLength);
  inline void GetStoreLines(const uint64_t ubSize, const uint64_t bytePerData, const ge::DataType& dtype);
  inline void ChooseTilingMode(const gert::Shape& xShape, const gert::Shape& rShape);
  inline void CalcRotateHalfTiling(const ge::DataType& dtype, uint64_t ubSize);
  ge::graphStatus CheckShapeSupport(const gert::Shape& xShape, const gert::Shape& cosShape, const gert::Shape& sinShape,
                                    uint64_t dLength);
};

inline void RotateHalfTiling::PrintTilingParams() {
  OPS_LOG_D(context, ">>>>>>>>>>>>>>> Start to print RotateHalf tiling data <<<<<<<<<<<<<<<<");
  OPS_LOG_D(context, ">>> tilingKey:           %lu", tilingKey);
  OPS_LOG_D(context, ">>> coreNum:             %lu", coreNum);
  OPS_LOG_D(context, ">>> tilingMode:          %lu", tilingData_.get_tilingMode());
  OPS_LOG_D(context, ">>> tilingDtype:         %lu", tilingDtype);
  OPS_LOG_D(context, ">>> gmLength:            %lu", tilingData_.get_gmLength());
  OPS_LOG_D(context, ">>> broadcastFirstDim:   %lu", tilingData_.get_broadcastFirstDim());
  OPS_LOG_D(context, ">>> broadcastSecondDim:  %lu", tilingData_.get_broadcastSecondDim());
  OPS_LOG_D(context, ">>> dLength:             %lu", tilingData_.get_dLength());
  OPS_LOG_D(context, ">>> halfDLength:         %lu", tilingData_.get_halfDLength());
  OPS_LOG_D(context, ">>> halfDPadLength:      %lu", tilingData_.get_halfDPadLength());
  OPS_LOG_D(context, ">>> dPadLength:          %lu", tilingData_.get_dPadLength());
  OPS_LOG_D(context, ">>> isAligned:           %lu", tilingData_.get_isAligned());
  OPS_LOG_D(context, ">>> totalSLines:         %lu", tilingData_.get_totalSLines());
  OPS_LOG_D(context, ">>> storeSLines:         %lu", tilingData_.get_storeSLines());
  OPS_LOG_D(context, ">>> storeDataLength:     %lu", tilingData_.get_storeDataLength());
  OPS_LOG_D(context, ">>> storePadDataLength:  %lu", tilingData_.get_storePadDataLength());
  OPS_LOG_D(context, ">>>>>>>>>>>>>>>       former core tiling data      <<<<<<<<<<<<<<<<");
  OPS_LOG_D(context, ">>> formerCoreNum:             %lu", tilingData_.get_formerCoreNum());
  OPS_LOG_D(context, ">>> formerSLines:              %lu", tilingData_.get_formerSLines());
  OPS_LOG_D(context, ">>> formerUbLoop:              %lu", tilingData_.get_formerUbLoop());
  OPS_LOG_D(context, ">>> formerUbLast:              %lu", tilingData_.get_formerUbLast());
  OPS_LOG_D(context, ">>> formerXDataLength:         %lu", tilingData_.get_formerXDataLength());
  OPS_LOG_D(context, ">>> formerRDataLength:         %lu", tilingData_.get_formerRDataLength());
  OPS_LOG_D(context, ">>> formerXCoreOffset:         %lu", tilingData_.get_formerXCoreOffset());
  OPS_LOG_D(context, ">>> formerRCoreOffset:         %lu", tilingData_.get_formerRCoreOffset());
  OPS_LOG_D(context, ">>> formerUbLastDataLength:    %lu", tilingData_.get_formerUbLastDataLength());
  OPS_LOG_D(context, ">>> formerUbLastPadDataLength: %lu", tilingData_.get_formerUbLastPadDataLength());
  OPS_LOG_D(context, ">>>>>>>>>>>>>>>        tail core tiling data        <<<<<<<<<<<<<<<<");
  OPS_LOG_D(context, ">>> tailCoreNum:               %lu", tilingData_.get_tailCoreNum());
  OPS_LOG_D(context, ">>> tailSLines:                %lu", tilingData_.get_tailSLines());
  OPS_LOG_D(context, ">>> tailUbLoop:                %lu", tilingData_.get_tailUbLoop());
  OPS_LOG_D(context, ">>> tailUbLast:                %lu", tilingData_.get_tailUbLast());
  OPS_LOG_D(context, ">>> tailXDataLength:           %lu", tilingData_.get_tailXDataLength());
  OPS_LOG_D(context, ">>> tailRDataLength:           %lu", tilingData_.get_tailRDataLength());
  OPS_LOG_D(context, ">>> tailUbLastDataLength:      %lu", tilingData_.get_tailUbLastDataLength());
  OPS_LOG_D(context, ">>> tailUbLastPadDataLength:   %lu", tilingData_.get_tailUbLastPadDataLength());
  OPS_LOG_D(context, ">>>>>>>>>>>>>>> Print RotateHalf tiling data end <<<<<<<<<<<<<<<<");
}

/* Split S lines to each core, former core maybe deal one more line than tail core. */
inline void RotateHalfTiling::GetCoreDataTiling() {
  uint64_t formerCoreNum, tailCoreNum;
  uint64_t totalS = tilingData_.get_totalSLines();
  // calc former and tail core num
  if (totalS <= coreNum) {
    formerCoreNum = totalS;
    tailCoreNum = 0;
    coreNum = totalS;
  } else {
    formerCoreNum = GetRem(totalS, coreNum);
    tailCoreNum = coreNum - formerCoreNum;
  }
  // S lines each core
  uint64_t formerSLines = GetCeilDiv(totalS, coreNum);
  uint64_t tailSLines = GetDiv(totalS, coreNum);

  tilingData_.set_formerCoreNum(formerCoreNum);
  tilingData_.set_formerSLines(formerSLines);
  tilingData_.set_tailCoreNum(tailCoreNum);
  tilingData_.set_tailSLines(tailSLines);
}

/* Calc ub loop times and left S lines in each core */
inline void RotateHalfTiling::GetUbLoopTiling() {
  uint64_t tilingMode = tilingData_.get_tilingMode();
  uint64_t formerSLines = tilingData_.get_formerSLines();
  uint64_t tailSLines = tilingData_.get_tailSLines();
  uint64_t storeSLines = tilingData_.get_storeSLines();
  uint64_t dLength = tilingData_.get_dLength();
  uint64_t dPadLength = tilingData_.get_dPadLength();
  uint64_t bSize = tilingData_.get_broadcastFirstDim();
  uint64_t nSize = tilingData_.get_broadcastSecondDim();
  uint64_t bnSize = bSize * nSize;
  uint64_t bdSize = bSize * dLength;
  uint64_t ndSize = nSize * dLength;
  uint64_t bndSize = bnSize * dLength;

  // each core ub loop
  tilingData_.set_formerUbLoop(GetDiv(formerSLines, storeSLines));
  tilingData_.set_tailUbLoop(GetDiv(tailSLines, storeSLines));
  // each core ub last lines after loop
  tilingData_.set_formerUbLast(GetRem(formerSLines, storeSLines));
  tilingData_.set_tailUbLast(GetRem(tailSLines, storeSLines));
  // each core processed data length
  if (tilingMode == TILING_MODE_BNSD || tilingMode == TILING_MODE_BSND || tilingMode == TILING_MODE_SBND ||
      tilingMode == TILING_MODE_NO_BROADCAST) {
    tilingData_.set_formerXDataLength(formerSLines * bndSize);
    tilingData_.set_tailXDataLength(tailSLines * bndSize);
    tilingData_.set_formerRDataLength(formerSLines * dLength);
    tilingData_.set_tailRDataLength(tailSLines * dLength);
  } else if (tilingMode == TILING_MODE_BND) {
    tilingData_.set_formerXDataLength(formerSLines * dLength);
    tilingData_.set_tailXDataLength(tailSLines * dLength);
    tilingData_.set_formerRDataLength(dLength);
    tilingData_.set_tailRDataLength(dLength);
  } else if (tilingMode == TILING_MODE_R_B1SD) {
    tilingData_.set_formerXDataLength(formerSLines * bndSize);
    tilingData_.set_tailXDataLength(tailSLines * bndSize);
    tilingData_.set_formerRDataLength(formerSLines * bdSize);
    tilingData_.set_tailRDataLength(tailSLines * bdSize);
  }
  // each ub loop processed data length == storeDataLength
  // ub last processed data length
  tilingData_.set_formerUbLastDataLength(tilingData_.get_formerUbLast() * dLength);
  tilingData_.set_tailUbLastDataLength(tilingData_.get_tailUbLast() * dLength);
  tilingData_.set_formerUbLastPadDataLength(tilingData_.get_formerUbLast() * dPadLength);
  tilingData_.set_tailUbLastPadDataLength(tilingData_.get_tailUbLast() * dPadLength);
  // former core processed total data length(the start offset of tail cores)
  tilingData_.set_formerRCoreOffset(tilingData_.get_formerCoreNum() * formerSLines * dLength);
  if (tilingMode == TILING_MODE_BNSD || tilingMode == TILING_MODE_NO_BROADCAST || tilingMode == TILING_MODE_R_B1SD) {
    tilingData_.set_formerXCoreOffset(tilingData_.get_formerCoreNum() * formerSLines * dLength);
  } else if (tilingMode == TILING_MODE_BSND) {
    tilingData_.set_formerXCoreOffset(tilingData_.get_formerCoreNum() * formerSLines * ndSize);
  } else if (tilingMode == TILING_MODE_SBND) {
    tilingData_.set_formerXCoreOffset(tilingData_.get_formerCoreNum() * formerSLines * bndSize);
  } else if (tilingMode == TILING_MODE_BND) {
    tilingData_.set_formerXCoreOffset(tilingData_.get_formerCoreNum() * formerSLines * dLength);
    tilingData_.set_formerRCoreOffset(0);
  }
}

/* Cut UB size to ubNum, compute how many Lines of D_pad(split S) can store in each split UB */
inline void RotateHalfTiling::GetStoreLines(const uint64_t ubSize, const uint64_t bytePerData,
                                            const ge::DataType& dtype) {
  uint64_t ubNum = dtype == ge::DT_FLOAT ? DOUBLE_BUFFER * UB_NUM : DOUBLE_BUFFER * UB_NUM_BF16;
  uint64_t splitUbSize = GetDiv(ubSize - UB_RESERVE_SIZE, ubNum);
  uint64_t storeSLines = GetDiv(splitUbSize, bytePerData * tilingData_.get_dPadLength());
  tilingData_.set_storeSLines(storeSLines);
  tilingData_.set_storeDataLength(storeSLines * tilingData_.get_dLength());
  tilingData_.set_storePadDataLength(storeSLines * tilingData_.get_dPadLength());
}

inline void RotateHalfTiling::ChooseTilingMode(const gert::Shape& xShape, const gert::Shape& rShape) {
  uint64_t xFirstDim = static_cast<int64_t>(xShape.GetDim(DIM_FIRST));
  uint64_t xSecondDim = static_cast<int64_t>(xShape.GetDim(DIM_SECOND));
  uint64_t xThirdDim = static_cast<int64_t>(xShape.GetDim(DIM_THIRD));
  uint64_t rFirstDim = static_cast<int64_t>(rShape.GetDim(DIM_FIRST));
  uint64_t rSecondDim = static_cast<int64_t>(rShape.GetDim(DIM_SECOND));
  uint64_t rThirdDim = static_cast<int64_t>(rShape.GetDim(DIM_THIRD));
  if (rFirstDim == xFirstDim && rSecondDim == xSecondDim && rThirdDim == xThirdDim) {  // x = r, NO_BROADCAST
    OPS_LOG_D(context, "RotateHalf layout: x = r, NO_BROADCAST");
    tilingData_.set_tilingMode(TILING_MODE_NO_BROADCAST);
    tilingData_.set_broadcastFirstDim(NO_BROADCAST_DIM);
    tilingData_.set_broadcastSecondDim(NO_BROADCAST_DIM);
    tilingData_.set_totalSLines(xFirstDim * xSecondDim * xThirdDim);
  } else if (rFirstDim == 1 && rSecondDim == 1 && rThirdDim == 1) {  // S = 1 --> x: BND, r: D
    OPS_LOG_D(context, "RotateHalf layout: BND");
    tilingData_.set_tilingMode(TILING_MODE_BND);
    tilingData_.set_broadcastFirstDim(NO_BROADCAST_DIM);
    tilingData_.set_broadcastSecondDim(NO_BROADCAST_DIM);
    if (xFirstDim == 1) {
      tilingData_.set_totalSLines(xSecondDim * xThirdDim);
    } else if (xSecondDim == 1) {
      tilingData_.set_totalSLines(xFirstDim * xThirdDim);
    } else if (xThirdDim == 1) {
      tilingData_.set_totalSLines(xFirstDim * xSecondDim);
    }
  } else if (xFirstDim != rFirstDim && xSecondDim != rSecondDim) {  // BNSD
    OPS_LOG_D(context, "RotateHalf layout: BNSD");
    tilingData_.set_tilingMode(TILING_MODE_BNSD);
    tilingData_.set_broadcastFirstDim(xFirstDim);
    tilingData_.set_broadcastSecondDim(xSecondDim);
    tilingData_.set_totalSLines(xThirdDim);
  } else if (xFirstDim != rFirstDim && xThirdDim != rThirdDim) {  // BSND
    OPS_LOG_D(context, "RotateHalf layout: BSND");
    tilingData_.set_tilingMode(TILING_MODE_BSND);
    tilingData_.set_broadcastFirstDim(xFirstDim);
    tilingData_.set_broadcastSecondDim(xThirdDim);
    tilingData_.set_totalSLines(xSecondDim);
  } else if (xSecondDim != rSecondDim && xThirdDim != rThirdDim) {  // SBND
    OPS_LOG_D(context, "RotateHalf layout: SBND");
    tilingData_.set_tilingMode(TILING_MODE_SBND);
    tilingData_.set_broadcastFirstDim(xSecondDim);
    tilingData_.set_broadcastSecondDim(xThirdDim);
    tilingData_.set_totalSLines(xFirstDim);
  } else if (xFirstDim != rFirstDim && (rSecondDim == 1 || rThirdDim == 1)) {  // x: B1SD or BS1D
    OPS_LOG_D(context, "RotateHalf layout: x B1SD or BS1D");
    tilingData_.set_tilingMode(TILING_MODE_BSND);
    tilingData_.set_broadcastFirstDim(xFirstDim);
    tilingData_.set_broadcastSecondDim(NO_BROADCAST_DIM);
    tilingData_.set_totalSLines(xSecondDim * xThirdDim);
  } else if (xSecondDim != rSecondDim && rThirdDim == 1) {  // x: SB1D
    OPS_LOG_D(context, "RotateHalf layout: x SB1D");
    tilingData_.set_tilingMode(TILING_MODE_SBND);
    tilingData_.set_broadcastFirstDim(xSecondDim);
    tilingData_.set_broadcastSecondDim(NO_BROADCAST_DIM);
    tilingData_.set_totalSLines(xFirstDim);
  } else if (rThirdDim != xThirdDim && rFirstDim == xFirstDim && rSecondDim == xSecondDim) {  // r: BS1D or SB1D
    OPS_LOG_D(context, "RotateHalf layout: r BS1D or SB1D");
    tilingData_.set_tilingMode(TILING_MODE_SBND);
    tilingData_.set_broadcastFirstDim(NO_BROADCAST_DIM);
    tilingData_.set_broadcastSecondDim(xThirdDim);
    tilingData_.set_totalSLines(xFirstDim * xSecondDim);
  } else if (rSecondDim != xSecondDim && rFirstDim == xFirstDim && rThirdDim == xThirdDim) {  // r: B1SD
    OPS_LOG_D(context, "RotateHalf layout: r B1SD");
    tilingData_.set_tilingMode(TILING_MODE_R_B1SD);
    tilingData_.set_broadcastFirstDim(xFirstDim);
    tilingData_.set_broadcastSecondDim(xSecondDim);
    tilingData_.set_totalSLines(xThirdDim);
  } else {
    tilingData_.set_tilingMode(TILING_MODE_UNKNOWN);
  }
}

inline void RotateHalfTiling::CalcRotateHalfTiling(const ge::DataType& dtype, uint64_t ubSize) {
  uint64_t bytePerData = GetBytePerData(dtype);
  GetCoreDataTiling();
  GetStoreLines(ubSize, bytePerData, dtype);
  GetUbLoopTiling();
}

/* judge D/2 aligned */
inline void RotateHalfTiling::GetAlignedInfo(const ge::DataType inputDtype, uint64_t dLength) {
  uint64_t halfDLength = GetDiv(dLength, TWO);
  uint64_t bytePerData = GetBytePerData(inputDtype);
  uint64_t dataEachBlock = GetDiv(BYTE_OF_BLOCK, bytePerData);
  tilingData_.set_dLength(dLength);
  tilingData_.set_halfDLength(halfDLength);
  if (GetRem(halfDLength, dataEachBlock) == 0) {
    tilingData_.set_isAligned(1);
    tilingData_.set_halfDPadLength(halfDLength);
    tilingData_.set_dPadLength(dLength);
  } else {
    uint64_t halfDPadLength = GetCeilDiv(halfDLength, dataEachBlock) * dataEachBlock;
    tilingData_.set_isAligned(0);
    tilingData_.set_halfDPadLength(halfDPadLength);
    tilingData_.set_dPadLength(halfDPadLength * TWO);
  }
}

/* Check input shape */
ge::graphStatus RotateHalfTiling::CheckShapeSupport(const gert::Shape& xShape, const gert::Shape& cosShape,
                                                    const gert::Shape& sinShape, uint64_t dLength) {
  if (xShape.GetDimNum() != DIM_NUM || cosShape.GetDimNum() != DIM_NUM || sinShape.GetDimNum() != DIM_NUM) {
    OPS_LOG_E(context, "the input shape must be 4-dimensional.");
    return ge::GRAPH_FAILED;
  }
  if (dLength > D_LENGTH_LIMIT) {
    OPS_LOG_E(context, "input last dim (head_dim) should be less than %lu.", D_LENGTH_LIMIT);
    return ge::GRAPH_FAILED;
  }
  if (GetRem(dLength, TWO) != 0) {
    OPS_LOG_E(context, "input last dim (head_dim) must be an even number.");
    return ge::GRAPH_FAILED;
  }
  if (cosShape != sinShape) {
    OPS_LOG_E(context, "cos shape and sin shape should be equal.");
    return ge::GRAPH_FAILED;
  }
  uint64_t cosDLength = static_cast<uint64_t>(cosShape.GetDim(DIM_FOURTH));
  if (cosDLength != dLength) {
    OPS_LOG_E(context, "input last dim (head_dim) should be equal, but get x [%lu] and cos [%lu].", dLength, cosDLength);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

/**
 * Check do not use Rope: 
 * 1. layout=BNSD
 * 2. if D is aligned: check B * N > 8 * S --> return true
 * 3. if D is unaligned: check 2 * B * N > S / coreNum && D < 80 --> return true
 * else false
 * 
 * **Note: The above judgment conditions are empirical values obtained through testing**
 */
inline bool RotateHalfTiling::CheckBnsdBlockSkip() {
  if (tilingData_.get_tilingMode() == TILING_MODE_BNSD) {
    int64_t broadcastDim = tilingData_.get_broadcastFirstDim() * tilingData_.get_broadcastSecondDim();
    int64_t unalignedBroadcastTime = broadcastDim * BNSD_UNALIGNED_BLOCK_BN_SCALE;
    int64_t alignedSThreshold = tilingData_.get_totalSLines() * BNSD_ALIGNED_BLOCK_S_SCALE;
    int64_t eachCoreSLines = GetCeilDiv(tilingData_.get_totalSLines(), coreNum);
    
    if (tilingData_.get_isAligned() == 1 && broadcastDim > alignedSThreshold) {
      return true;
    }
    if (tilingData_.get_isAligned() == 0 && unalignedBroadcastTime > eachCoreSLines &&
        tilingData_.get_dLength() < BNSD_UNALIGNED_BLOCK_D_LENGTH) {
      return true;
    }
  }
  return false;
}

ge::graphStatus RotateHalfTiling::DoRotateHalfTiling() {
  // get chip core num and ub size
  const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const auto aivNum = ascendcPlatform.GetCoreNumAiv();
  uint64_t ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  coreNum = aivNum;

  // get x and r shape. layout: [..., head_dim]
  auto inputXShapePtr = context->GetInputShape(INDEX_INPUT_X);
  OPS_LOG_E_IF_NULL(context, inputXShapePtr, return ge::GRAPH_FAILED);
  const gert::Shape& xShape = inputXShapePtr->GetStorageShape();
  auto inputCosShapePtr = context->GetInputShape(INDEX_INPUT_COS);
  OPS_LOG_E_IF_NULL(context, inputCosShapePtr, return ge::GRAPH_FAILED);
  const gert::Shape& cosShape = inputCosShapePtr->GetStorageShape();
  auto inputSinShapePtr = context->GetInputShape(INDEX_INPUT_SIN);
  OPS_LOG_E_IF_NULL(context, inputSinShapePtr, return ge::GRAPH_FAILED);
  const gert::Shape& sinShape = inputSinShapePtr->GetStorageShape();
  uint64_t dLength = static_cast<uint64_t>(xShape.GetDim(DIM_FOURTH));
  uint64_t gmLength = static_cast<uint64_t>(xShape.GetShapeSize());
  tilingData_.set_gmLength(gmLength);
  auto shapeCheckRes = CheckShapeSupport(xShape, cosShape, sinShape, dLength);
  if (ge::GRAPH_SUCCESS != shapeCheckRes) {
    OPS_LOG_E(context, "input shape does not meet the requirements.");
    return shapeCheckRes;
  }

  // check input data type
  auto inputInfoPtr = context->GetInputDesc(INDEX_INPUT_X);
  OPS_LOG_E_IF_NULL(context, inputInfoPtr, return ge::GRAPH_FAILED);
  auto cosInfoPtr = context->GetInputDesc(INDEX_INPUT_COS);
  OPS_LOG_E_IF_NULL(context, cosInfoPtr, return ge::GRAPH_FAILED);
  auto sinInfoPtr = context->GetInputDesc(INDEX_INPUT_SIN);
  OPS_LOG_E_IF_NULL(context, sinInfoPtr, return ge::GRAPH_FAILED);
  const ge::DataType inputDtype = inputInfoPtr->GetDataType();
  const ge::DataType cosDtype = cosInfoPtr->GetDataType();
  const ge::DataType sinDtype = sinInfoPtr->GetDataType();

  if (inputDtype != cosDtype || inputDtype != sinDtype) {
    OPS_LOG_E(context, "the dtype of input x, cos and sin must be the same.");
    return ge::GRAPH_FAILED;
  }

  tilingDtype = GetTilingDtype(inputDtype);
  if (tilingDtype == TILING_DTYPE_UNKNOWN) {
    OPS_LOG_E(context, "only supports float, float16 and bfloat16 data type.");
    return ge::GRAPH_FAILED;
  }

  // check D/2 aligned
  GetAlignedInfo(inputDtype, dLength);
  if (tilingData_.get_isAligned() == 0 && ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910) {
    OPS_LOG_E(context, "current soc does not support non aligned calculations.");
    return ge::GRAPH_FAILED;
  }

  ChooseTilingMode(xShape, cosShape);
  if (tilingData_.get_tilingMode() == TILING_MODE_UNKNOWN) {
    OPS_LOG_E(context, "unknow input layout, unable to calculate.");
    return ge::GRAPH_FAILED;
  }

  // block some layout=BNSD case, use small operators for higher performance
  if (CheckBnsdBlockSkip()) {
    OPS_LOG_E(context, "when input is BNSD layout and  B * N is large or D is not aligned, "
                       "please do not use RotaryPositionEmbedding fusion operator.");
    return ge::GRAPH_FAILED;
  }

  tilingKey = GetTilingKey(tilingData_.get_tilingMode(), tilingDtype);
  CalcRotateHalfTiling(inputDtype, ubSize);
  if (tilingData_.get_storeSLines() <= 0) {
    OPS_LOG_E(context, "head_dim shape is too large to compute.");
    return ge::GRAPH_FAILED;
  }

  PrintTilingParams();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4RopeRotateHalf(gert::TilingContext* context) {
  OPS_LOG_D(context, "RotateHalf tiling start.");
  RotateHalfTiling rotateHalfTiling(context);

  auto tilingRes = rotateHalfTiling.DoRotateHalfTiling();
  if (ge::GRAPH_SUCCESS != tilingRes) {
    OPS_LOG_E(context, "DoRotateHalfTiling failed.");
    return tilingRes;
  }

  context->SetTilingKey(rotateHalfTiling.tilingKey);
  context->SetBlockDim(rotateHalfTiling.coreNum);
  rotateHalfTiling.tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                                       context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(rotateHalfTiling.tiling.GetDataSize());
  size_t usrWorkspaceSize = 0;
  size_t sysWorkspaceSize = 16 * 1024 * 1024;
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = usrWorkspaceSize + sysWorkspaceSize;

  OPS_LOG_D(context, "RotateHalf tiling end.");
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling