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
 * \file moe_gating_top_k_softmax_tiling_k_full_load.cpp
 * \brief
 */
#include "tiling/tiling_templates_registry.h"
#include "moe_gating_top_k_softmax_tiling_base.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static const int32_t FP32_SIZE = 4;
static const int64_t ALIGN_NUM = 32;
static const int64_t DOUBLE_BUFFER = 2;
static const int32_t MAX_COL_IN_UB = 98304; // ubSize/minTypeSize

static inline uint32_t CeilDiv(uint32_t value, uint32_t factor) {
  if (factor == 0) {
    return value;
  }
  return (value + factor - 1) / factor;
}

class MoeGatingTopKSoftmaxKFullLoadTiling : public MoeGatingTopKSoftmaxBaseTiling {
 public:
  explicit MoeGatingTopKSoftmaxKFullLoadTiling(gert::TilingContext* context)
      : MoeGatingTopKSoftmaxBaseTiling(context) {
  }

 protected:
  bool IsCapable() override;
  ge::graphStatus DoOpTiling() override;
  ge::graphStatus DoLibApiTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus GetWorkspaceSize() override;
  ge::graphStatus PostTiling() override;

 private:
  uint32_t ubOuter;
  uint32_t ubLoop;
  uint32_t ubFormer;
  uint32_t ubTail;
  uint32_t ubFormerAlign;
  uint32_t ubTailAlign;
  MoeGatingTopKSoftmaxKFullLoadTilingData tilingData;
  bool CalculateParamInUb();
  uint64_t GetTotalTmpSizeInUb(uint32_t kAlign);
};

bool MoeGatingTopKSoftmaxKFullLoadTiling::IsCapable() {
  return true;
}

uint64_t MoeGatingTopKSoftmaxKFullLoadTiling::GetTotalTmpSizeInUb(uint32_t kAlign) {
  int32_t dataTypeSize;
  // if dtype is bf16,will cast fp32
  if (dtype == ge::DataType::DT_BF16) {
    dataTypeSize = FP32_SIZE;
  } else {
    dataTypeSize = ge::GetSizeByDataType(dtype);
  }
  uint64_t gatingUbTmpSize = ubFormerAlign * dataTypeSize;
  uint64_t topkOutUbTmpSize = (ubFormerAlign + kAlign) * dataTypeSize;
  uint64_t topkIndicesOutUbTmpSize = (ubFormerAlign + kAlign) * FP32_SIZE;
  uint64_t rowsOutUbTmpSize = 0;
  // if dtype is bf16,will cast fp32
  if (dtype == ge::DataType::DT_BF16) {
    rowsOutUbTmpSize = ubFormerAlign * FP32_SIZE;
  } else {
    rowsOutUbTmpSize = kAlign * FP32_SIZE;
  }

  uint64_t finishUbTmpSize = ALIGN_NUM;
  auto shape = ge::Shape({ubFormerAlign});
  uint64_t softmaxUbTmpSize = GetSoftMaxFlashV2MaxTmpSize(shape, dataTypeSize, dataTypeSize, true);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());

  uint32_t maxValue = 0;
  uint32_t minValue = 0;
  (void)GetTopKMaxMinTmpSize(ascendcPlatform, kAlign + ubFormerAlign, 1, false, true, TopKMode::TOPK_NORMAL, true,
                             dataTypeSize, maxValue, minValue);
  maxValue = maxValue > softmaxUbTmpSize ? maxValue : softmaxUbTmpSize;

  uint64_t additionalUbtmpSize = ALIGN_NUM + ALIGN_NUM + ALIGN_NUM;

  uint64_t totalSize = gatingUbTmpSize + topkOutUbTmpSize + topkIndicesOutUbTmpSize + rowsOutUbTmpSize +
                       finishUbTmpSize + maxValue + additionalUbtmpSize;
  return totalSize;
}

bool MoeGatingTopKSoftmaxKFullLoadTiling::CalculateParamInUb() {
  ubOuter = 1;
  if (col > MAX_COL_IN_UB) {
    ubOuter = col / MAX_COL_IN_UB;
  }
  uint32_t kAlign = CeilDiv(k, ALIGN_NUM) * ALIGN_NUM;
  while (true) {
    ubFormer = col / ubOuter;
    ubFormerAlign = CeilDiv(ubFormer, ALIGN_NUM) * ALIGN_NUM;
    uint64_t totalSize = GetTotalTmpSizeInUb(kAlign);
    if (totalSize <= (ubSize / DOUBLE_BUFFER)) {
      break;
    }
    if (ubFormerAlign < kAlign) {
      ubOuter = 0;
      break;
    }
    ubOuter++;
  }
  if (ubOuter == 0) {
    return false;
  }
  auto ubFormerDownAlign = (ubFormer / ALIGN_NUM) * ALIGN_NUM;
  auto ubOuterDownAlign = (col + ubFormerDownAlign - 1) / ubFormerDownAlign;
  ubLoop = ubOuterDownAlign;
  ubFormer = ubFormerDownAlign;
  ubFormerAlign = ubFormer;
  ubTail = col - (ubLoop - 1) * ubFormerAlign;
  ubTailAlign = CeilDiv(ubTail, ALIGN_NUM) * ALIGN_NUM;
  return true;
}

ge::graphStatus MoeGatingTopKSoftmaxKFullLoadTiling::DoOpTiling() {
  tilingData.set_row(row);
  tilingData.set_col(col);
  tilingData.set_k(k);
  tilingData.set_kAlign(CeilDiv(k, ALIGN_NUM) * ALIGN_NUM);
  tilingData.set_blockFormer(CeilDiv(row, coreNum));
  tilingData.set_blockNum(CeilDiv(row, tilingData.get_blockFormer()));
  tilingData.set_blockTail(row - (tilingData.get_blockNum() - 1) * tilingData.get_blockFormer());

  if (!CalculateParamInUb()) {
    OP_LOGE("[MoeGatingTopKSoftmaxK] AutoTiling failed, the K is too large.");
    return ge::GRAPH_FAILED;
  }
  tilingData.set_ubLoop(ubLoop);
  tilingData.set_ubFormer(ubFormer);
  tilingData.set_ubFormerAlign(ubFormerAlign);
  tilingData.set_ubTail(ubTail);
  tilingData.set_ubTailAlign(ubTailAlign);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxKFullLoadTiling::DoLibApiTiling() {
  int dataTypeSize;
  if (dtype == ge::DataType::DT_BF16) {
    dataTypeSize = FP32_SIZE;
  } else {
    dataTypeSize = ge::GetSizeByDataType(dtype);
  }
  uint32_t ubFormerAlign = CeilDiv(ubFormer, ALIGN_NUM) * ALIGN_NUM;
  uint32_t kAlign = CeilDiv(k, ALIGN_NUM) * ALIGN_NUM;
  auto softmaxShape = ge::Shape({tilingData.get_ubFormer()});
  SoftMaxFlashV2TilingFunc(softmaxShape, dataTypeSize, dataTypeSize,
                           GetSoftMaxFlashV2MaxTmpSize(softmaxShape, dataTypeSize, true, true),
                           tilingData.ubFormerSoftmaxTilingData, true);

  softmaxShape = ge::Shape({tilingData.get_ubTail()});
  SoftMaxFlashV2TilingFunc(softmaxShape, dataTypeSize, dataTypeSize,
                           GetSoftMaxFlashV2MaxTmpSize(softmaxShape, dataTypeSize, true, true),
                           tilingData.ubTailSoftmaxTilingData, true);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());

  TopKTilingFunc(ascendcPlatform, kAlign + ubFormerAlign, 1, kAlign, dataTypeSize, true, TopKMode::TOPK_NORMAL,
                 true, tilingData.topkFormerTilingData);

  TopKTilingFunc(ascendcPlatform, kAlign + ubTailAlign, 1, kAlign, dataTypeSize, true, TopKMode::TOPK_NORMAL,
                 true, tilingData.topkTailTilingData);
  return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKSoftmaxKFullLoadTiling::GetTilingKey() const {
  switch (dtype) {
    case ge::DataType::DT_FLOAT16:
      return MOE_GATING_SOFTMAX_K_FULL_LOAD_FLOAT16;
    case ge::DataType::DT_FLOAT:
      return MOE_GATING_SOFTMAX_K_FULL_LOAD_FLOAT;
    case ge::DataType::DT_BF16:
      return MOE_GATING_SOFTMAX_K_FULL_LOAD_BF16;
    default:
      break;
  }
  return tilingKey_;
}

ge::graphStatus MoeGatingTopKSoftmaxKFullLoadTiling::GetWorkspaceSize() {
  workspaceSize_ = 0;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxKFullLoadTiling::PostTiling() {
  tilingData.set_tilingKey(GetTilingKey());
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(tilingData.get_blockNum());
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MoeGatingTopKSoftmax", MoeGatingTopKSoftmaxKFullLoadTiling, 20000);
}  // namespace optiling