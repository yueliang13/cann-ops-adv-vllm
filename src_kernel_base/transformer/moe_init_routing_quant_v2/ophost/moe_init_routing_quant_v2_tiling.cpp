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
 * \file moe_init_routing_quant_v2_tiling.cpp
 * \brief
 */
#include "moe_init_routing_quant_v2_tiling.h"

namespace optiling {

const static int64_t ATTR_QUANT_MODE = 6;
const static int64_t TILING_KEY_BASE = 10000;
const static int64_t TILING_KEY_PERF_BASE = 20000;
const static int64_t TILING_KEY_QUANT_BASE = 1000;
const static int64_t TILING_KEY_DROP_MODE_BASE = 100;
const static int64_t TILING_KEY_SORT_BASE = 10;
const static int64_t NUM_TWO = 2;
const static int64_t ONE_BLOCK_BYTE = 32;
const static int64_t FOUR_BLOCK_BYTE = 128;
const static int64_t MAX_COLS_ONE_LOOP = 8192;
const static int64_t INDEX_SCALE = 2;
const static int64_t INDEX_OFFSET = 3;
const static int64_t SMOOTH_NONE = 0;
const static int64_t SMOOTH_1H = 1;
const static int64_t SMOOTH_EH = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t MAX_COLS_DYNAMIC_QUANT = 6144;
const static int64_t SORT32_ALIGN_ELEMENT = 32;
const static int64_t ONE_CORE_SORT_BUFFER = 6;
const static int64_t DYNAMIC_QUANT_SRC_TO_DST_BUFFER = 15;
const static int64_t DYNAMIC_QUANT_COLS_BUFFER = 21;
const static int64_t DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER = 13;
const static int64_t DYNAMIC_QUANT_SCALE_SIZE_64 = 64;
const static int64_t DYNAMIC_QUANT_SCALE_SIZE_128 = 128;
const static int64_t OUTOUT_DYNAMIC_QUANT_SCALE = 4;
const static int64_t FULLLOAD_H_LIMIT = 7168;

class MoeInitRoutingQuantV2TilingBase : public InnerMoeInitRoutingV2TilingBase {
 public:
  explicit MoeInitRoutingQuantV2TilingBase(gert::TilingContext* context) : InnerMoeInitRoutingV2TilingBase(context) {
  }
  ~MoeInitRoutingQuantV2TilingBase() override = default;

  void Reset(gert::TilingContext* context) override {
    InnerMoeInitRoutingV2TilingBase::Reset(context);
  }

 protected:
  // 2、获取INPUT/OUTPUT/ATTR信息
  ge::graphStatus GetShapeAttrsInfo() override;
  // 5、计算TilingKey
  uint64_t GetTilingKey() const override;
  // 6、计算Workspace 大小
  ge::graphStatus GetWorkspaceSize() override;
  // 7、保存Tiling数据
  ge::graphStatus PostTiling() override;

 private:
  ge::graphStatus CheckOutShape() override;
  bool IsFullLoadQuant(int64_t space);
  bool IsFullLoadDynamicQuant(int64_t space);
  bool IsFullLoad() override;
  void SetGatherTilingData(InnerMoeV2GatherOutComputeTilingData* tilingData, int64_t perCoreRows, int64_t lastCoreRows,
                           int64_t cols);
  void SetGatherTilingDataCols(InnerMoeV2GatherOutComputeTilingData* tilingData, int64_t baseMaxCols, int64_t cols);
  void SetGatherTilingDataRows(InnerMoeV2GatherOutComputeTilingData* tilingData, int64_t perCoreRows,
                               int64_t lastCoreRows, int64_t basePerLoopMaxRows);
  void Tiling4GatherQuant();
  void Tiling4GatherDynamicQuant();
  void Tiling4SrcToDstCapacityCompute() override;
  void Tiling4GatherOutCompute() override;
  void CopyGatherOutTiling(InnerMoeV2GatherOutComputeTilingData& dst, InnerMoeV2GatherOutComputeTilingData& src);
  void CopyTilingData();

  int64_t quantMode;
  MoeInitRoutingQuantV2TilingData quantTilingData;
};

#define CHECK_NULL(context, ptr, ...)                                                              \
  do {                                                                                             \
    if ((ptr) == nullptr) {                                                                        \
      const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
      OPS_LOG_E_WITHOUT_REPORT(name, "%s is nullptr!", ##__VA_ARGS__);                               \
      return ge::GRAPH_FAILED;                                                                     \
    }                                                                                              \
  } while (0)

#define CHECK_FAIL(context, cond, ...)                \
  do {                                                \
    if (cond) {                                       \
      OPS_LOG_E(context->GetNodeName(), ##__VA_ARGS__); \
      return ge::GRAPH_FAILED;                        \
    }                                                 \
  } while (0)

inline static int64_t GetPerOrLastValue(int64_t x, int64_t y) {
  if (y == 0) {
    return 0;
  }
  return x <= y ? x : x % y;
}

inline static int64_t AlignOneBlockByte(int64_t x) {
  return (x + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
}

inline static int64_t AlignOneBlockByteCeil(int64_t x) {
  return x / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
}

ge::graphStatus MoeInitRoutingQuantV2TilingBase::CheckOutShape() {
  if (InnerMoeInitRoutingV2TilingBase::CheckOutShape() == ge::GRAPH_FAILED) {
    return ge::GRAPH_FAILED;
  }

  if (quantMode != 0) {
    auto dynamicShapePtr = context_->GetOutputShape(OUTOUT_DYNAMIC_QUANT_SCALE);
    CHECK_NULL(context_, dynamicShapePtr, "DynamicQuantScale");

    auto dynamicQuantScaleDesc = context_->GetOutputDesc(OUTOUT_DYNAMIC_QUANT_SCALE);
    CHECK_NULL(context_, dynamicQuantScaleDesc, "DynamicQuantScale");
    auto dt = dynamicQuantScaleDesc->GetDataType();
    CHECK_FAIL(context_, dt != ge::DT_FLOAT, "The data type of dynamicQuantScale should be FLOAT.");

    const gert::Shape dynamicQuantScaleShape = dynamicShapePtr->GetStorageShape();
    size_t dynamicQuantScaleDimNum = dynamicQuantScaleShape.GetDimNum();
    CHECK_FAIL(context_, dynamicQuantScaleDimNum != 1, "The dim number of dynamicQuantScale should be 1.");
    if (dropPadMode == 0) {
      int64_t firstDim = moeInitRoutingTilingData.get_n() * moeInitRoutingTilingData.get_k();
      firstDim = activateNum == 0 ? firstDim : std::min(firstDim, activateNum);
      CHECK_FAIL(context_, dynamicQuantScaleShape.GetDim(0) != firstDim,
                 "The first dim of dynamicQuantScale should be %ld.", firstDim);
    } else {
      CHECK_FAIL(context_, dynamicQuantScaleShape.GetDim(0) != expertNum * expertCapacity,
                 "The first dim of dynamicQuantScale should be %ld.", expertNum * expertCapacity);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingQuantV2TilingBase::GetShapeAttrsInfo() {
  auto attrs = context_->GetAttrs();
  const int64_t* quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE);
  if (quantModePtr != nullptr) {
    quantMode = *quantModePtr;
  }
  CHECK_FAIL(context_, quantMode < 0 || quantMode > 1, "The quantMode should be 0 or 1.");

  if (InnerMoeInitRoutingV2TilingBase::GetShapeAttrsInfo() == ge::GRAPH_FAILED) {
    return ge::GRAPH_FAILED;
  }

  auto scaleShapePtr = context_->GetOptionalInputShape(INDEX_SCALE);
  if (quantMode == 0) {
    CHECK_NULL(context_, scaleShapePtr, "scale");
    auto scaleDesc = context_->GetOptionalInputDesc(INDEX_SCALE);
    CHECK_NULL(context_, scaleDesc, "scale");
    auto dt_scale = scaleDesc->GetDataType();
    CHECK_FAIL(context_, dt_scale != ge::DT_FLOAT, "The data type of scale should be float.");

    auto offsetShapePtr = context_->GetOptionalInputShape(INDEX_OFFSET);
    CHECK_NULL(context_, offsetShapePtr, "offset");
    auto offsetDesc = context_->GetOptionalInputDesc(INDEX_OFFSET);
    CHECK_NULL(context_, offsetDesc, "offset");
    auto dt_offset = offsetDesc->GetDataType();
    CHECK_FAIL(context_, dt_offset != ge::DT_FLOAT, "The data type of offset should be float.");

    auto scaleShape = scaleShapePtr->GetStorageShape();
    CHECK_FAIL(context_, scaleShape.GetDimNum() != 1, "The dim number of scale should be 1.");
    CHECK_FAIL(context_, scaleShape.GetDim(0) != 1, "The first dim of scale should be 1.");
    auto offsetShape = offsetShapePtr->GetStorageShape();
    CHECK_FAIL(context_, offsetShape.GetDimNum() != 1, "The dim number of offset should be 1.");
    CHECK_FAIL(context_, offsetShape.GetDim(0) != 1, "The first dim of offset should be 1.");
  } else {
    if (scaleShapePtr != nullptr) {
      auto scaleDesc = context_->GetOptionalInputDesc(INDEX_SCALE);
      CHECK_NULL(context_, scaleDesc, "scale");
      auto dt = scaleDesc->GetDataType();
      CHECK_FAIL(context_, dt != ge::DT_FLOAT, "The data type of scale should be float.");

      int64_t cols = InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_cols();
      auto smoothShape = scaleShapePtr->GetStorageShape();
      size_t smoothDimNum = smoothShape.GetDimNum();
      CHECK_FAIL(context_, smoothDimNum != NUM_TWO, "The dim number of scale should be 2.");
      CHECK_FAIL(context_, smoothShape.GetDim(0) != 1 && smoothShape.GetDim(0) != expertNum,
                 "The first dim of scale should be 1 or expert_num.");
      CHECK_FAIL(context_, smoothShape.GetDim(1) != cols, "The second dim of scale should be %ld.", cols);
      quantTilingData.set_smoothType((smoothShape.GetDim(0) == 1) ? SMOOTH_1H : SMOOTH_EH);
    } else {
      quantTilingData.set_smoothType(SMOOTH_NONE);
    }
  }
  return ge::GRAPH_SUCCESS;
}

bool MoeInitRoutingQuantV2TilingBase::IsFullLoadQuant(int64_t space) {
  int64_t perCoreXRows = moeInitRoutingTilingData.get_n() / aivNum;
  int64_t remainder = moeInitRoutingTilingData.get_n() % aivNum;
  // NUM_TWO is Max xRows need add 2 becauseof the left and right row may be another row.
  perCoreXRows = remainder <= 1 ? perCoreXRows + 1 : perCoreXRows + NUM_TWO;
  int64_t quantBaseSpace = AlignOneBlockByte(moeInitRoutingTilingData.get_cols());
  int64_t quantSpace =
      quantBaseSpace * (inuptXDtypeSize_ + sizeof(int8_t) + sizeof(float) + sizeof(int16_t)) * perCoreXRows;
  int64_t remainUbAfterSort = aicoreParams_.ubSize - space - quantSpace;
  return remainUbAfterSort > 0;
}

bool MoeInitRoutingQuantV2TilingBase::IsFullLoadDynamicQuant(int64_t space) {
  int64_t quantSpace = AlignOneBlockByte(moeInitRoutingTilingData.get_cols()) * DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER;
  int64_t scaleOutSpace = 64;
  int64_t remainUbAfterSort = aicoreParams_.ubSize - space - scaleOutSpace - quantSpace;
  return remainUbAfterSort > 0;
}

bool MoeInitRoutingQuantV2TilingBase::IsFullLoad() {
  if (totalLength > sortLoopMaxElement || moeInitRoutingTilingData.get_cols() > MAX_COLS_ONE_LOOP ||
      this->dropPadMode == 1) {
    return false;
  }
  int64_t sortSpace = AlignOneBlockByte(this->totalLength) * sizeof(int32_t) * ONE_CORE_SORT_BUFFER;
  int64_t otherSpace = AlignOneBlockByte(this->totalLength) * sizeof(int32_t) * NUM_THREE;
  int64_t expertSpace = AlignOneBlockByte(this->expertNum * sizeof(int32_t));
  if (quantMode == 0) {
    return IsFullLoadQuant(sortSpace + otherSpace + expertSpace);
  } else {
    return IsFullLoadDynamicQuant(sortSpace + otherSpace + expertSpace);
  }
}

void MoeInitRoutingQuantV2TilingBase::SetGatherTilingData(InnerMoeV2GatherOutComputeTilingData* tilingData,
                                                          int64_t perCoreRows, int64_t lastCoreRows, int64_t cols) {
  tilingData->set_perCorePerLoopRows(perCoreRows);
  tilingData->set_perCoreLastLoopRows(perCoreRows);
  tilingData->set_lastCorePerLoopRows(lastCoreRows);
  tilingData->set_lastCoreLastLoopRows(lastCoreRows);
  tilingData->set_perCoreLoops(1);
  tilingData->set_lastCoreLoops(1);
  tilingData->set_perLoopCols(cols);
  tilingData->set_lastLoopCols(cols);
  tilingData->set_colLoops(1);
}

void SetGatherTilingDatawithloop(InnerMoeV2GatherOutComputeTilingData* tilingData,
                                 int64_t perCorePerLoopRows, int64_t lastCorePerLoopRows, int64_t cols,
                                 int64_t perCoreLastLoopRows = 1, int64_t lastCoreLastLoopRows = 1,
                                 int64_t perCoreLoops = 1, int64_t lastCoreLoops = 1) {
  tilingData->set_perCorePerLoopRows(perCorePerLoopRows);
  tilingData->set_perCoreLastLoopRows(perCoreLastLoopRows);
  tilingData->set_lastCorePerLoopRows(lastCorePerLoopRows);
  tilingData->set_lastCoreLastLoopRows(lastCoreLastLoopRows);
  tilingData->set_perCoreLoops(perCoreLoops);
  tilingData->set_lastCoreLoops(lastCoreLoops);
  tilingData->set_perLoopCols(cols);
  tilingData->set_lastLoopCols(cols);
  tilingData->set_colLoops(1);
}

void MoeInitRoutingQuantV2TilingBase::SetGatherTilingDataCols(InnerMoeV2GatherOutComputeTilingData* tilingData,
                                                              int64_t baseMaxCols, int64_t cols) {
  tilingData->set_perLoopCols(std::min(baseMaxCols, cols));
  tilingData->set_lastLoopCols(GetPerOrLastValue(cols, baseMaxCols));
  tilingData->set_colLoops(baseMaxCols == 0 ? 0 : (cols + baseMaxCols - 1) / baseMaxCols);
}
void MoeInitRoutingQuantV2TilingBase::SetGatherTilingDataRows(InnerMoeV2GatherOutComputeTilingData* tilingData,
                                                              int64_t perCoreRows, int64_t lastCoreRows,
                                                              int64_t basePerLoopMaxRows) {
  tilingData->set_perCorePerLoopRows(std::min(perCoreRows, basePerLoopMaxRows));
  tilingData->set_perCoreLastLoopRows(GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
  tilingData->set_perCoreLoops(basePerLoopMaxRows == 0 ? 0
                                                       : (perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);

  tilingData->set_lastCorePerLoopRows(std::min(lastCoreRows, basePerLoopMaxRows));
  tilingData->set_lastCoreLastLoopRows(GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
  tilingData->set_lastCoreLoops(basePerLoopMaxRows == 0 ? 0
                                                        : (lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
}

void MoeInitRoutingQuantV2TilingBase::Tiling4SrcToDstCapacityCompute() {
  if (quantMode == 0 || dropPadMode == 0) {
    InnerMoeInitRoutingV2TilingBase::Tiling4SrcToDstCapacityCompute();
    return;
  }
  auto tilingData = &moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp;

  int64_t perCoreRows = CeilDiv(totalLength, aivNum);
  if (perCoreRows <= 0) {
    tilingData->set_needCoreNum(0);
    return;
  }
  tilingData->set_needCoreNum(CeilDiv(totalLength, perCoreRows));
  int64_t cols = moeInitRoutingTilingData.get_cols();
  tilingData->set_perCoreRows(perCoreRows);
  int64_t lastCoreRows = totalLength - perCoreRows * (tilingData->get_needCoreNum() - 1);
  tilingData->set_lastCoreRows(lastCoreRows);

  int64_t rowSize = AlignOneBlockByte(perCoreRows * sizeof(int32_t)) * NUM_FOUR;
  int64_t colSize = AlignOneBlockByte(cols * sizeof(int8_t)) * DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
  int64_t scaleSize = DYNAMIC_QUANT_SCALE_SIZE_64;
  if (rowSize + colSize + scaleSize < static_cast<int64_t>(aicoreParams_.ubSize)) {
    SetGatherTilingData(tilingData, perCoreRows, lastCoreRows, cols);
  } else {
    int64_t baseMaxCols = MAX_COLS_DYNAMIC_QUANT;
    int64_t totalColSize = AlignOneBlockByte(baseMaxCols * sizeof(int8_t)) * DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
    int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize);
    int64_t basePerLoopMaxRows =
        AlignOneBlockByteCeil((ubSize - totalColSize - scaleSize) / sizeof(int32_t)) / NUM_FOUR;
    if (cols < MAX_COLS_DYNAMIC_QUANT) {
      basePerLoopMaxRows = AlignOneBlockByteCeil((ubSize - colSize - scaleSize) / sizeof(int32_t)) / NUM_FOUR;
    } else if (perCoreRows < basePerLoopMaxRows) {
      baseMaxCols = AlignOneBlockByteCeil(ubSize - rowSize - scaleSize) / DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
    }
    SetGatherTilingDataCols(tilingData, baseMaxCols, cols);
    SetGatherTilingDataRows(tilingData, perCoreRows, lastCoreRows, basePerLoopMaxRows);
  }
}

void MoeInitRoutingQuantV2TilingBase::Tiling4GatherQuant() {
  auto tilingData = &quantTilingData.gatherOutComputeParamsOp;
  tilingData->set_activateRows(totalLength);
  if (dropPadMode == 0 && activateNum > 0) {
    tilingData->set_activateRows(std::min(activateNum, totalLength));
  }
  int64_t perCoreRows = CeilDiv(totalLength, aivNum);
  if (perCoreRows <= 0) {
    tilingData->set_needCoreNum(0);
    return;
  }
  tilingData->set_needCoreNum(CeilDiv(totalLength, perCoreRows));
  int64_t cols = InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_cols();
  tilingData->set_perCoreRows(perCoreRows);
  int64_t lastCoreRows = totalLength - perCoreRows * (tilingData->get_needCoreNum() - 1);
  tilingData->set_lastCoreRows(lastCoreRows);

  int64_t sizeOfCol = sizeof(int8_t) * NUM_TWO + sizeof(float) + sizeof(int16_t) + inuptXDtypeSize_ * NUM_TWO;
  int64_t rowSize = AlignOneBlockByte((perCoreRows * sizeof(int32_t) * NUM_TWO));
  int64_t colSize = AlignOneBlockByte(cols * sizeOfCol);
  if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO) {
    SetGatherTilingData(tilingData, perCoreRows, lastCoreRows, cols);
  } else {
    int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
    int64_t baseMaxColsSize = AlignOneBlockByte(baseMaxCols * sizeOfCol);
    int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize);
    int64_t basePerLoopMaxRows = AlignOneBlockByteCeil((ubSize - baseMaxColsSize) / NUM_TWO / sizeof(int32_t));
    if (cols < MAX_COLS_ONE_LOOP) {
      basePerLoopMaxRows = AlignOneBlockByteCeil((ubSize - colSize) / NUM_TWO / sizeof(int32_t));
    } else if (perCoreRows < basePerLoopMaxRows) {
      baseMaxCols = AlignOneBlockByteCeil((ubSize - rowSize) / sizeOfCol);
    }
    SetGatherTilingDataCols(tilingData, baseMaxCols, cols);
    SetGatherTilingDataRows(tilingData, perCoreRows, lastCoreRows, basePerLoopMaxRows);
  }
}

void MoeInitRoutingQuantV2TilingBase::Tiling4GatherDynamicQuant() {
  auto tilingData = &quantTilingData.gatherOutComputeParamsOp;
  tilingData->set_activateRows(totalLength);
  if (dropPadMode == 0 && activateNum > 0) {
    tilingData->set_activateRows(std::min(activateNum, totalLength));
  }
  int64_t perCoreRows = CeilDiv(totalLength, aivNum);
  if (perCoreRows <= 0) {
    tilingData->set_needCoreNum(0);
    return;
  }
  tilingData->set_needCoreNum(CeilDiv(totalLength, perCoreRows));
  int64_t cols = InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_cols();
  tilingData->set_perCoreRows(perCoreRows);
  int64_t lastCoreRows = totalLength - perCoreRows * (tilingData->get_needCoreNum() - 1);
  tilingData->set_lastCoreRows(lastCoreRows);

  int64_t rowSize = AlignOneBlockByte(perCoreRows * sizeof(int32_t)) * NUM_FOUR;
  int64_t colSize = AlignOneBlockByte(cols * sizeof(int8_t)) * DYNAMIC_QUANT_COLS_BUFFER;
  int64_t scaleSize = DYNAMIC_QUANT_SCALE_SIZE_64;
  int64_t onceRowSize = (static_cast<int64_t>(aicoreParams_.ubSize) - 
                          colSize - scaleSize - 
                          ONE_BLOCK_BYTE * NUM_FOUR * NUM_THREE) /
                        (sizeof(int32_t) * NUM_FOUR);
  int64_t oneBlockNumInt = static_cast<int64_t>(ONE_BLOCK_BYTE) / static_cast<int64_t>(sizeof(int32_t));
  onceRowSize = onceRowSize / oneBlockNumInt * oneBlockNumInt;
  bool ifOneLoop = ((static_cast<int64_t>(aicoreParams_.ubSize) > colSize +
            scaleSize + ONE_BLOCK_BYTE * NUM_FOUR * NUM_FOUR) && 
            quantTilingData.get_smoothType() == SMOOTH_NONE &&
            cols == FULLLOAD_H_LIMIT);
  int64_t perCoreOnceRowSize = ifOneLoop ? std::min(onceRowSize, perCoreRows) : perCoreRows;
  int64_t lastCoreOnceRowSize = ifOneLoop ? std::min(onceRowSize, lastCoreRows) : lastCoreRows;
  int64_t perCoreLoops = ifOneLoop ? CeilDiv(perCoreRows, perCoreOnceRowSize) : 1;
  int64_t lastCoreLoops = ifOneLoop ? CeilDiv(lastCoreRows, lastCoreOnceRowSize) : 1;
  int64_t perCoreLastLoopRows = ifOneLoop ? GetPerOrLastValue(perCoreRows, perCoreOnceRowSize) : perCoreRows;
  int64_t lastCoreLastLoopRows =ifOneLoop ?  GetPerOrLastValue(lastCoreRows, lastCoreOnceRowSize) : lastCoreRows;
  if (rowSize + colSize + scaleSize < static_cast<int64_t>(aicoreParams_.ubSize) || ifOneLoop) {
    SetGatherTilingDatawithloop(tilingData, perCoreOnceRowSize, lastCoreOnceRowSize, cols,
                                perCoreLastLoopRows, lastCoreLastLoopRows,
                                perCoreLoops, lastCoreLoops);
  } else {
    int64_t baseMaxCols = MAX_COLS_DYNAMIC_QUANT;
    int64_t totalColSize = AlignOneBlockByte(baseMaxCols * sizeof(int8_t)) * DYNAMIC_QUANT_COLS_BUFFER;
    int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize);
    int64_t basePerLoopMaxRows =
        AlignOneBlockByteCeil((ubSize - totalColSize - scaleSize) / sizeof(int32_t)) / NUM_FOUR;
    if (cols < MAX_COLS_DYNAMIC_QUANT) {
      basePerLoopMaxRows = AlignOneBlockByteCeil((ubSize - colSize - scaleSize) / sizeof(int32_t)) / NUM_FOUR;
    } else if (perCoreRows < basePerLoopMaxRows) {
      baseMaxCols = AlignOneBlockByteCeil(ubSize - rowSize - scaleSize) / DYNAMIC_QUANT_COLS_BUFFER;
    }
    SetGatherTilingDataCols(tilingData, baseMaxCols, cols);
    SetGatherTilingDataRows(tilingData, perCoreRows, lastCoreRows, basePerLoopMaxRows);
  }
}

void MoeInitRoutingQuantV2TilingBase::Tiling4GatherOutCompute() {
  if (quantMode == 0) {
    Tiling4GatherQuant();
  } else {
    Tiling4GatherDynamicQuant();
  }
}

void MoeInitRoutingQuantV2TilingBase::CopyGatherOutTiling(InnerMoeV2GatherOutComputeTilingData& dst,
                                                          InnerMoeV2GatherOutComputeTilingData& src) {
  dst.set_needCoreNum(src.get_needCoreNum());
  dst.set_activateRows(src.get_activateRows());
  dst.set_perCoreRows(src.get_perCoreRows());
  dst.set_perCorePerLoopRows(src.get_perCorePerLoopRows());
  dst.set_perCoreLastLoopRows(src.get_perCoreLastLoopRows());
  dst.set_lastCoreRows(src.get_lastCoreRows());
  dst.set_lastCorePerLoopRows(src.get_lastCorePerLoopRows());
  dst.set_lastCoreLastLoopRows(src.get_lastCoreLastLoopRows());
  dst.set_perCoreLoops(src.get_perCoreLoops());
  dst.set_lastCoreLoops(src.get_lastCoreLoops());
  dst.set_perLoopCols(src.get_perLoopCols());
  dst.set_lastLoopCols(src.get_lastLoopCols());
  dst.set_colLoops(src.get_colLoops());
}

void MoeInitRoutingQuantV2TilingBase::CopyTilingData() {
  quantTilingData.set_coreNum(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_coreNum());
  quantTilingData.set_n(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_n());
  quantTilingData.set_cols(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_cols());
  quantTilingData.set_k(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_k());
  quantTilingData.set_expertCapacity(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_expertCapacity());
  quantTilingData.set_expertNum(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_expertNum());
  quantTilingData.set_dropPadMode(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_dropPadMode());
  quantTilingData.set_expertTokensCountOrCumsumFlag(
      InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_expertTokensCountOrCumsumFlag());
  quantTilingData.set_expertTokensBeforeCapacityFlag(
      InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_expertTokensBeforeCapacityFlag());

  auto vbsTilingData = &InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.vbsComputeParamsOp;
  quantTilingData.vbsComputeParamsOp.set_needCoreNum(vbsTilingData->get_needCoreNum());
  quantTilingData.vbsComputeParamsOp.set_perCoreElements(vbsTilingData->get_perCoreElements());
  quantTilingData.vbsComputeParamsOp.set_perCoreLoops(vbsTilingData->get_perCoreLoops());
  quantTilingData.vbsComputeParamsOp.set_perCorePerLoopElements(vbsTilingData->get_perCorePerLoopElements());
  quantTilingData.vbsComputeParamsOp.set_perCoreLastLoopElements(vbsTilingData->get_perCoreLastLoopElements());
  quantTilingData.vbsComputeParamsOp.set_lastCoreElements(vbsTilingData->get_lastCoreElements());
  quantTilingData.vbsComputeParamsOp.set_lastCoreLoops(vbsTilingData->get_lastCoreLoops());
  quantTilingData.vbsComputeParamsOp.set_lastCorePerLoopElements(vbsTilingData->get_lastCorePerLoopElements());
  quantTilingData.vbsComputeParamsOp.set_lastCoreLastLoopElements(vbsTilingData->get_lastCoreLastLoopElements());
  quantTilingData.vbsComputeParamsOp.set_oneLoopMaxElements(vbsTilingData->get_oneLoopMaxElements());

  quantTilingData.vmsMiddleComputeParamsOp.set_needCoreNum(
      InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.vmsMiddleComputeParamsOp.get_needCoreNum());
  quantTilingData.sortOutComputeParamsOp.set_oneLoopMaxElements(
      InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.sortOutComputeParamsOp.get_oneLoopMaxElements());

  CopyGatherOutTiling(quantTilingData.srcToDstComputeParamsOp,
                      InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.srcToDstComputeParamsOp);
  CopyGatherOutTiling(quantTilingData.srcToDstCapacityComputeParamsOp,
                      InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp);
}

ge::graphStatus MoeInitRoutingQuantV2TilingBase::GetWorkspaceSize() {
  InnerMoeInitRoutingV2TilingBase::GetWorkspaceSize();
  bool useCols =
      (dropPadMode == 0 && quantTilingData.gatherOutComputeParamsOp.get_colLoops() > 1) ||
      (dropPadMode == 1 &&
       InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_colLoops() > 1);
  if (quantMode == 1 && useCols) {
    workspaceSize_ += aivNum * InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_cols() * sizeof(float);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingQuantV2TilingBase::PostTiling() {
  CopyTilingData();
  context_->SetBlockDim(aivNum);
  size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
  currentWorkspace[0] = workspaceSize_;
  quantTilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(quantTilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

uint64_t MoeInitRoutingQuantV2TilingBase::GetTilingKey() const {
  if (isFullLoad) {
    return TILING_KEY_PERF_BASE + quantMode * TILING_KEY_QUANT_BASE;
  }
  return TILING_KEY_BASE + quantMode * TILING_KEY_QUANT_BASE + dropPadMode * TILING_KEY_DROP_MODE_BASE +
         (totalLength > sortLoopMaxElement) * TILING_KEY_SORT_BASE;
}

ASCENDC_EXTERN_C ge::graphStatus TilingForMoeInitRoutingQuantV2(gert::TilingContext* context) {
  MoeInitRoutingQuantV2TilingBase tiling(context);
  return tiling.DoTiling();
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForMoeInitRoutingQuantV2(gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeInitRoutingQuantV2)
    .Tiling(TilingForMoeInitRoutingQuantV2)
    .TilingParse<MoeInitRoutingQuantV2CompileInfo>(TilingPrepareForMoeInitRoutingQuantV2);

}  // namespace optiling
