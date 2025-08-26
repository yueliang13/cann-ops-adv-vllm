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
 * \file moe_gating_top_k_softmax_tiling_perf.cpp
 * \brief
 */
#include "tiling/tiling_templates_registry.h"
#include "moe_gating_top_k_softmax_tiling_base.h"

using namespace AscendC;

namespace optiling {
static const int32_t DIM_0 = 0;
static const int32_t DIM_1 = 1;
static const int32_t DIM_2 = 2;
static const int32_t SIZE_2 = 2;
static const int32_t SIZE_3 = 3;
static const int32_t TWO = 2;
static const int32_t SIX = 6;
static const int32_t FP32_SIZE = 4;
static const int32_t FP16_SIZE = 2;
static const int32_t BF16_SIZE = 2;
static const int32_t INT32_SIZE = 4;
static const int32_t BOOL_SIZE = 1;
static const int64_t BLOCK_SIZE = 32;
static const int64_t ALIGN_NUM = 32;
static const int64_t BUFFER_SIZE = 96;
static const int64_t BLOCK_B32_SIZE = 8;
static const int64_t REPEAT_B32_SIZE = 64;
static const int64_t MAX_COL = 1024;
static const bool IS_SOFTMAX_REUSE_SOURCE = true;
static const bool IS_TOP_K_REUSE_SOURCE = true;

static inline uint32_t CeilDiv(uint32_t value, uint32_t factor) {
  if (factor == 0) {
    return value;
  }
  return (value + factor - 1) / factor;
}

static inline int64_t calcUbAlignBufferSize(const uint32_t curRowInUb, const uint32_t col, const int typeSize) {
  return CeilDiv(col * typeSize, BLOCK_SIZE) * BLOCK_SIZE * curRowInUb;
}

static inline uint32_t calcGatingAlignCol(const uint32_t col, const ge::DataType dtype) {
  // 对齐成32个数处理
  return CeilDiv(col, ALIGN_NUM) * ALIGN_NUM;
}

static uint32_t GenIndicesMask(int64_t k) {
  uint32_t mask = TWO;  // 10
  for (int i = 1; i < k; ++i) {
    mask = (mask << TWO) | TWO;
  }
  return mask;
}

static uint32_t GenValuesMask(int64_t k) {
  uint32_t mask = 1;  // 01
  for (int i = 1; i < k; ++i) {
    mask = (mask << TWO) | 1;
  }
  return mask;
}

class MoeGatingTopKSoftmaxPerfTiling : public MoeGatingTopKSoftmaxBaseTiling {
 public:
  explicit MoeGatingTopKSoftmaxPerfTiling(gert::TilingContext* context) : MoeGatingTopKSoftmaxBaseTiling(context) {
  }

 protected:
  bool IsCapable() override;

  ge::graphStatus DoOpTiling() override;

  ge::graphStatus DoLibApiTiling() override;

  uint64_t GetTilingKey() const override;

  ge::graphStatus GetWorkspaceSize() override;

  ge::graphStatus PostTiling() override;

 private:
  uint32_t maxRow;
  uint32_t gatingAlignCol;
  MoeGatingTopKSoftmaxPerfTilingData tilingData;

  uint32_t calcMaxRowInUb(const int64_t ubSize, const ge::DataType dtype, const uint32_t k, const uint32_t blockRow,
                          const uint32_t col);

  bool isBufferSizeEnough(const uint32_t curRowInUb, const uint32_t gatingAlignCol, const int64_t tmpUbSize,
                          const uint32_t k);
};

bool MoeGatingTopKSoftmaxPerfTiling::IsCapable() {
  if ((col <= MAX_COL && col % BLOCK_B32_SIZE == 0 && k <= BLOCK_B32_SIZE) || (col < BLOCK_B32_SIZE)) {
    return true;
  }
  return false;
}

ge::graphStatus MoeGatingTopKSoftmaxPerfTiling::DoOpTiling() {
  gatingAlignCol = calcGatingAlignCol(col, dtype);
  maxRow = calcMaxRowInUb(ubSize, dtype, k, CeilDiv(row, coreNum), gatingAlignCol);

  tilingData.set_row(row);
  tilingData.set_col(col);

  tilingData.set_k(k);
  tilingData.set_kAlign(CeilDiv(k, BLOCK_B32_SIZE) * BLOCK_B32_SIZE);

  tilingData.set_blockFormer(CeilDiv(row, coreNum));
  tilingData.set_blockNum(CeilDiv(row, tilingData.get_blockFormer()));
  tilingData.set_blockTail(row - (tilingData.get_blockNum() - 1) * tilingData.get_blockFormer());
  tilingData.set_colBytesAlign(CeilDiv(col, BLOCK_SIZE / FP32_SIZE) * (BLOCK_SIZE / FP32_SIZE));
  tilingData.set_colAlign(gatingAlignCol);
  tilingData.set_ubLoopOfFormerBlock(CeilDiv(tilingData.get_blockFormer(), maxRow));
  tilingData.set_ubFormer(maxRow);
  tilingData.set_ubTailOfFormerBlock(tilingData.get_blockFormer() -
                                     (tilingData.get_ubLoopOfFormerBlock() - 1) * tilingData.get_ubFormer());
  tilingData.set_ubLoopOfTailBlock(CeilDiv(tilingData.get_blockTail(), maxRow));
  tilingData.set_ubTailOfTailBlock(tilingData.get_blockTail() -
                                   (tilingData.get_ubLoopOfTailBlock() - 1) * tilingData.get_ubFormer());
  tilingData.set_topKValuesMask(GenValuesMask(k));
  tilingData.set_topKIndicesMask(GenIndicesMask(k));
  tilingData.set_bufferElemSize(maxRow * gatingAlignCol);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxPerfTiling::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKSoftmaxPerfTiling::GetTilingKey() const {
  if (col <= BLOCK_B32_SIZE) {
    switch (dtype) {
      case ge::DataType::DT_FLOAT16:
        return MOE_GATING_SOFTMAX_PERF_FLOAT16_COL_SMALLER_THAN_8;
      case ge::DataType::DT_FLOAT:
        return MOE_GATING_SOFTMAX_PERF_FLOAT_COL_SMALLER_THAN_8;
      case ge::DataType::DT_BF16:
        return MOE_GATING_SOFTMAX_PERF_BF16_COL_SMALLER_THAN_8;
      default:
        break;
    }    
  } else if (col <= REPEAT_B32_SIZE) {
    switch (dtype) {
      case ge::DataType::DT_FLOAT16:
        return MOE_GATING_SOFTMAX_PERF_FLOAT16_COL_FROM_8_TO_64;
      case ge::DataType::DT_FLOAT:
        return MOE_GATING_SOFTMAX_PERF_FLOAT_COL_FROM_8_TO_64;
      case ge::DataType::DT_BF16:
        return MOE_GATING_SOFTMAX_PERF_BF16_COL_FROM_8_TO_64;
      default:
        break;
    }      
  } else {
    switch (dtype) {
      case ge::DataType::DT_FLOAT16:
        return MOE_GATING_SOFTMAX_PERF_FLOAT16_COL_BIGGER_THAN_64;
      case ge::DataType::DT_FLOAT:
        return MOE_GATING_SOFTMAX_PERF_FLOAT_COL_BIGGER_THAN_64;
      case ge::DataType::DT_BF16:
        return MOE_GATING_SOFTMAX_PERF_BF16_COL_BIGGER_THAN_64;
      default:
        break;
    }  
  }
  return tilingKey_;
}

ge::graphStatus MoeGatingTopKSoftmaxPerfTiling::GetWorkspaceSize() {
  workspaceSize_ = 0;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxPerfTiling::PostTiling() {
  tilingData.set_tilingKey(GetTilingKey());
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(tilingData.get_blockNum());
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

bool MoeGatingTopKSoftmaxPerfTiling::isBufferSizeEnough(const uint32_t curRowInUb, const uint32_t gatingAlignCol,
                                                        const int64_t tmpUbSize, const uint32_t k) {
  // 输出row_idx使用的内存
  int64_t rowIdxOutBufferSize = calcUbAlignBufferSize(curRowInUb, k, INT32_SIZE);

  // 输入finished使用的内存，因为启用了double buffer，因此共计使用两块
  int64_t finishedBufferSize = CeilDiv(BOOL_SIZE * curRowInUb, BLOCK_SIZE) * BLOCK_SIZE;

  // 通用内存，大小为r*E，元素大小取可能的最大值，即32位
  // 共计使用六块：Double Buffer输入x使用两块，输出y及expert_idx使用两块，通用缓存使用两块
  int64_t generalBufferSize = curRowInUb * gatingAlignCol * INT32_SIZE;

  // 判断内存是否越界
  if (rowIdxOutBufferSize + BUFFER_SIZE + finishedBufferSize * TWO + generalBufferSize * SIX > tmpUbSize) {
    return false;
  }
  return true;
}

uint32_t MoeGatingTopKSoftmaxPerfTiling::calcMaxRowInUb(const int64_t ubSize, const ge::DataType dtype,
                                                        const uint32_t k, const uint32_t blockRow, const uint32_t col) {
  uint32_t ubOuter = 1;
  int64_t tmpUbSize = ubSize;
  uint32_t curRowInUb;
  while (true) {
    curRowInUb = CeilDiv(blockRow, ubOuter);
    if (isBufferSizeEnough(curRowInUb, gatingAlignCol, tmpUbSize, k)) {
      break;
    }
    ubOuter++;
    if (ubOuter > blockRow) {
      return 0;
    }
  }
  return curRowInUb;
}

REGISTER_TILING_TEMPLATE("MoeGatingTopKSoftmax", MoeGatingTopKSoftmaxPerfTiling, 1000);
}  // namespace optiling