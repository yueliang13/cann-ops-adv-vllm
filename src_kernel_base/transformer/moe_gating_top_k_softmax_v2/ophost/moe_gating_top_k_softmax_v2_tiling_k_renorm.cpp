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
 * \file moe_gating_top_k_softmax_v2_tiling_k_renorm.cpp
 * \brief
 */
#include "tiling/tiling_templates_registry.h"
#include "moe_gating_top_k_softmax_v2_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static const bool IS_SOFTMAX_REUSE_SOURCE = true;
static const int32_t FP32_SIZE = 4;
static const int32_t MAX_COL_IN_UB = 98304; // ubSize/minTypeSize

class MoeGatingTopKSoftmaxV2KRenormTiling : public MoeGatingTopKSoftmaxV2BaseTiling {
 public:
  explicit MoeGatingTopKSoftmaxV2KRenormTiling(gert::TilingContext* context)
      : MoeGatingTopKSoftmaxV2BaseTiling(context) {
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
  MoeGatingTopKSoftmaxV2KFullLoadTilingData tilingData;
  bool CalculateParamInUb();
  uint64_t GetTotalTmpSizeInUb(uint32_t kAlign);
};

bool MoeGatingTopKSoftmaxV2KRenormTiling::IsCapable() {
  return renorm == 1;
}

uint64_t MoeGatingTopKSoftmaxV2KRenormTiling::GetTotalTmpSizeInUb(uint32_t kAlign) {
  int32_t dataTypeSize;
  // if dtype is bf16,will cast fp32
  if (dtype == ge::DataType::DT_BF16) {
    dataTypeSize = FP32_SIZE;
  } else {
    dataTypeSize = ge::GetSizeByDataType(dtype);
  }
  uint64_t gatingUbTmpSize = (ubFormerAlign + kAlign) * FP32_SIZE;
  uint64_t finishUbTmpSize = ALIGN_NUM;
  uint64_t topkOutUbTmpSize = kAlign * FP32_SIZE;
  uint64_t topkIndicesOutUbTmpSize = (ubFormerAlign + kAlign) * FP32_SIZE;

  auto shape = ge::Shape({1, k});
  uint64_t softmaxUbTmpSize = GetSoftMaxMaxTmpSize(shape, dataTypeSize, IS_SOFTMAX_REUSE_SOURCE);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());

  uint32_t maxValue = 0;
  uint32_t minValue = 0;
  (void)GetTopKMaxMinTmpSize(ascendcPlatform, kAlign + ubFormerAlign, 1, false, true, TopKMode::TOPK_NORMAL, true,
                             dataTypeSize, maxValue, minValue);
  maxValue = maxValue > softmaxUbTmpSize ? maxValue : softmaxUbTmpSize;

  uint64_t totalSize = gatingUbTmpSize + topkOutUbTmpSize + topkIndicesOutUbTmpSize + finishUbTmpSize + maxValue;
  return totalSize;
}

bool MoeGatingTopKSoftmaxV2KRenormTiling::CalculateParamInUb() {
  ubOuter = 1;
  if (col > MAX_COL_IN_UB) {
    ubOuter = col / MAX_COL_IN_UB;
  }
  uint32_t kAlign = CeilDiv(k, ALIGN_NUM) * ALIGN_NUM;
  while (true) {               // find best ubOuter
    ubFormer = col / ubOuter;
    ubFormerAlign = CeilDiv(ubFormer, ALIGN_NUM) * ALIGN_NUM;
    uint64_t totalSize = GetTotalTmpSizeInUb(kAlign);
    int doubleBuffer = 2;
    if (totalSize <= (ubSize / doubleBuffer)) {
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
  ubFormer = ubFormerDownAlign;
  ubLoop = ubOuterDownAlign;
  ubFormerAlign = ubFormer;
  ubTail = col - (ubLoop - 1) * ubFormerAlign;
  ubTailAlign = CeilDiv(ubTail, ALIGN_NUM) * ALIGN_NUM;
  return true;
}

ge::graphStatus MoeGatingTopKSoftmaxV2KRenormTiling::DoOpTiling() {
  tilingData.set_row(row);
  tilingData.set_col(col);
  tilingData.set_k(k);
  tilingData.set_kAlign(CeilDiv(k, ALIGN_NUM) * ALIGN_NUM);
  tilingData.set_blockFormer(CeilDiv(row, coreNum));
  tilingData.set_blockNum(CeilDiv(row, tilingData.get_blockFormer()));
  tilingData.set_blockTail(row - (tilingData.get_blockNum() - 1) * tilingData.get_blockFormer());

  if (!CalculateParamInUb()) {
    OPS_LOG_E("[MoeGatingTopKSoftmaxV2K Renorm]", "AutoTiling failed, the K is too large.");
    return ge::GRAPH_FAILED;
  }
  tilingData.set_ubLoop(ubLoop);
  tilingData.set_ubFormer(ubFormer);
  tilingData.set_ubFormerAlign(ubFormerAlign);
  tilingData.set_ubTail(ubTail);
  tilingData.set_ubTailAlign(ubTailAlign);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2KRenormTiling::DoLibApiTiling() {
  int dataTypeSize;
  if (dtype == ge::DataType::DT_BF16) {
    dataTypeSize = FP32_SIZE;
  } else {
    dataTypeSize = ge::GetSizeByDataType(dtype);
  }
  uint32_t ubFormerAlign = CeilDiv(ubFormer, ALIGN_NUM) * ALIGN_NUM;
  uint32_t kAlign = CeilDiv(k, ALIGN_NUM) * ALIGN_NUM;

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());

  TopKTilingFunc(ascendcPlatform, kAlign + ubFormerAlign, 1, kAlign, dataTypeSize, true, TopKMode::TOPK_NORMAL,
                 true, tilingData.topkFormerTilingData);

  TopKTilingFunc(ascendcPlatform, kAlign + ubTailAlign, 1, kAlign, dataTypeSize, true, TopKMode::TOPK_NORMAL,
                 true, tilingData.topkTailTilingData);

  auto softmaxShape = ge::Shape({1, kAlign});
  SoftMaxTilingFunc(softmaxShape, dataTypeSize,
                    GetSoftMaxMaxTmpSize(softmaxShape, dataTypeSize, IS_SOFTMAX_REUSE_SOURCE),
                    tilingData.ubFormerSoftmaxTilingData);
  return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKSoftmaxV2KRenormTiling::GetTilingKey() const {
  int sceneValue = 2;
  int doubleBufferFlag = 1;
  return TOPK_SOFTMAX_TILING_KEY_BASE_ALL + sceneValue * TOPK_SOFTMAX_TILING_KEY_BASE_SCENE +
         renorm * TOPK_SOFTMAX_TILING_KEY_BASE_RENORM + dtypeKey(dtype) * TOPK_SOFTMAX_TILING_KEY_BASE_DTYPE +
         doubleBufferFlag;
}

ge::graphStatus MoeGatingTopKSoftmaxV2KRenormTiling::GetWorkspaceSize() {
  workspaceSize_ = SYSTEM_WORKSPACE;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2KRenormTiling::PostTiling() {
  tilingData.set_softmaxFlag(softmaxFlag);
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(tilingData.get_blockNum());
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MoeGatingTopKSoftmaxV2", MoeGatingTopKSoftmaxV2KRenormTiling, 20001);
}  // namespace optiling