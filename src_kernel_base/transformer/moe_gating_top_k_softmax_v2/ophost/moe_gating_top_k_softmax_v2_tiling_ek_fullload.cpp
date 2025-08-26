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
 * \file moe_gating_top_k_softmax_v2_tiling_ek_fullload.cpp
 * \brief
 */
#include "tiling/tiling_templates_registry.h"
#include "moe_gating_top_k_softmax_v2_tiling.h"
using namespace AscendC;

namespace optiling {
static const int32_t SIZE_2 = 2;
static const int32_t FP32_SIZE = 4;
static const int32_t FP16_SIZE = 2;
static const int32_t INT32_SIZE = 4;
static const int32_t BOOL_SIZE = 1;
static const bool IS_SOFTMAX_REUSE_SOURCE = true;
static const bool IS_TOP_K_REUSE_SOURCE = true;
static const int32_t MAX_COL_IN_UB = 98304; // ubSize/minTypeSize

class MoeGatingTopKSoftmaxV2EKFullLoadTiling : public MoeGatingTopKSoftmaxV2BaseTiling {
 public:
  explicit MoeGatingTopKSoftmaxV2EKFullLoadTiling(gert::TilingContext* context)
      : MoeGatingTopKSoftmaxV2BaseTiling(context) {
  }

 protected:
  ge::graphStatus GetWorkspaceSize() override;
  bool IsCapable() override;
  ge::graphStatus DoOpTiling() override;
  ge::graphStatus DoLibApiTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus PostTiling() override;

 private:
  uint32_t maxRow;
  uint32_t gatingAlignCol;
  bool doubleBufferFlag;
  MoeGatingTopKSoftmaxV2EKFullLoadTilingData tilingData;
  uint32_t calcMaxRowInUb(const bool doubleBufferFlag, const uint64_t ubSize, const ge::DataType dtype,
                          const uint32_t k, const uint32_t blockRow, const uint32_t col);
  bool isBufferSizeEnough(const uint32_t curRowInUb, const uint32_t gatingAlignCol, const uint64_t tmpUbSize,
                          const ge::DataType dtype, const uint32_t k);
  bool getDoubleBufferFlag(const uint32_t gatingAlignCol, const uint64_t ubSize, const ge::DataType dtype,
                           const uint32_t k);
};

bool MoeGatingTopKSoftmaxV2EKFullLoadTiling::IsCapable() {
  if (col > MAX_COL_IN_UB) {
    return false;
  }
  gatingAlignCol = calcGatingAlignCol(col, dtype);
  doubleBufferFlag = getDoubleBufferFlag(gatingAlignCol, ubSize, dtype, k);
  maxRow = calcMaxRowInUb(doubleBufferFlag, ubSize, dtype, k, CeilDiv(row, coreNum), gatingAlignCol);
  if (maxRow == 0) {
    return false;
  }
  return true;
}

ge::graphStatus MoeGatingTopKSoftmaxV2EKFullLoadTiling::DoOpTiling() {
  tilingData.set_row(row);
  tilingData.set_col(col);

  tilingData.set_k(k);
  tilingData.set_kAlignB16(CeilDiv(k * FP16_SIZE, BLOCK_SIZE) * BLOCK_SIZE);
  tilingData.set_kAlignB32(CeilDiv(k * FP32_SIZE, BLOCK_SIZE) * BLOCK_SIZE);
  tilingData.set_kAlignT(CeilDiv(k * ge::GetSizeByDataType(dtype), BLOCK_SIZE) * BLOCK_SIZE);

  tilingData.set_blockFormer(CeilDiv(row, coreNum));
  tilingData.set_blockNum(CeilDiv(row, tilingData.get_blockFormer()));
  tilingData.set_blockTail(row - (tilingData.get_blockNum() - 1) * tilingData.get_blockFormer());

  tilingData.set_colAlign(gatingAlignCol);
  tilingData.set_ubLoopOfFormerBlock(CeilDiv(tilingData.get_blockFormer(), maxRow));
  tilingData.set_ubFormer(maxRow);
  tilingData.set_ubTailOfFormerBlock(tilingData.get_blockFormer() -
                                     (tilingData.get_ubLoopOfFormerBlock() - 1) * tilingData.get_ubFormer());
  tilingData.set_ubLoopOfTailBlock(CeilDiv(tilingData.get_blockTail(), maxRow));
  tilingData.set_ubTailOfTailBlock(tilingData.get_blockTail() -
                                   (tilingData.get_ubLoopOfTailBlock() - 1) * tilingData.get_ubFormer());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2EKFullLoadTiling::DoLibApiTiling() {
  int dataTypeSize;
  if (renorm == 0 || dtype == ge::DataType::DT_BF16) {
    dataTypeSize = FP32_SIZE;
  } else {
    dataTypeSize = ge::GetSizeByDataType(dtype);
  }

  auto softmaxShape = ge::Shape({tilingData.get_ubFormer(), gatingAlignCol});
  if (renorm == 1) {
    softmaxShape.SetDim(1, tilingData.get_kAlignB32() / FP32_SIZE);
  }
  SoftMaxTilingFunc(softmaxShape, dataTypeSize,
                    GetSoftMaxMaxTmpSize(softmaxShape, dataTypeSize, IS_SOFTMAX_REUSE_SOURCE),
                    tilingData.formerSoftmaxTilingData);

  softmaxShape = ge::Shape({tilingData.get_ubTailOfFormerBlock(), gatingAlignCol});
  if (renorm == 1) {
    softmaxShape.SetDim(1, tilingData.get_kAlignB32() / FP32_SIZE);
  }
  SoftMaxTilingFunc(softmaxShape, dataTypeSize,
                    GetSoftMaxMaxTmpSize(softmaxShape, dataTypeSize, IS_SOFTMAX_REUSE_SOURCE),
                    tilingData.formerBlockTailSoftmaxTilingData);

  softmaxShape = ge::Shape({tilingData.get_ubTailOfTailBlock(), gatingAlignCol});
  if (renorm == 1) {
    softmaxShape.SetDim(1, tilingData.get_kAlignB32() / FP32_SIZE);
  }
  SoftMaxTilingFunc(softmaxShape, dataTypeSize,
                    GetSoftMaxMaxTmpSize(softmaxShape, dataTypeSize, IS_SOFTMAX_REUSE_SOURCE),
                    tilingData.tailBlockTailSoftmaxTilingData);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
  TopKTilingFunc(ascendcPlatform, gatingAlignCol, tilingData.get_ubFormer(), k, dataTypeSize, true,
                TopKMode::TOPK_NORMAL, true, tilingData.formerTopkTilingData);

  TopKTilingFunc(ascendcPlatform, gatingAlignCol, tilingData.get_ubTailOfFormerBlock(), k, dataTypeSize, true,
                 TopKMode::TOPK_NORMAL, true, tilingData.formerBlockTailTopkTilingData);

  TopKTilingFunc(ascendcPlatform, gatingAlignCol, tilingData.get_ubTailOfTailBlock(), k, dataTypeSize, true,
                 TopKMode::TOPK_NORMAL, true, tilingData.tailBlockTailTopkTilingData);
  return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKSoftmaxV2EKFullLoadTiling::GetTilingKey() const {
  int sceneValue = 1;
  return TOPK_SOFTMAX_TILING_KEY_BASE_ALL + sceneValue * TOPK_SOFTMAX_TILING_KEY_BASE_SCENE +
         renorm * TOPK_SOFTMAX_TILING_KEY_BASE_RENORM + dtypeKey(dtype) * TOPK_SOFTMAX_TILING_KEY_BASE_DTYPE +
         doubleBufferFlag;
}

ge::graphStatus MoeGatingTopKSoftmaxV2EKFullLoadTiling::GetWorkspaceSize() {
  workspaceSize_ = SYSTEM_WORKSPACE;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2EKFullLoadTiling::PostTiling() {
  tilingData.set_softmaxFlag(softmaxFlag);
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(tilingData.get_blockNum());
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

bool MoeGatingTopKSoftmaxV2EKFullLoadTiling::getDoubleBufferFlag(const uint32_t gatingAlignCol, const uint64_t ubSize,
                                                                 const ge::DataType dtype, const uint32_t k) {
  // 判断一行的数据是否能搬进一半大小的ub空间
  return isBufferSizeEnough(1, gatingAlignCol, ubSize / SIZE_2, dtype, k);
}

bool MoeGatingTopKSoftmaxV2EKFullLoadTiling::isBufferSizeEnough(const uint32_t curRowInUb, const uint32_t gatingAlignCol,
                                                                const uint64_t tmpUbSize, const ge::DataType dtype,
                                                                const uint32_t k) {
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
  // bf16类型cast成fp32处理
  int typeSize = ge::GetSizeByDataType(dtype);
  if (renorm == 0 || dtype == ge::DataType::DT_BF16) {
    // 1.bf16场景gating复用缓存softmax输出结果，按fp32分配内存
    uint64_t gatingBufferSize = curRowInUb * gatingAlignCol;
    gatingBufferSize = gatingBufferSize * FP32_SIZE;
    if (gatingBufferSize > tmpUbSize) {
      return false;
    }
    // 2.计算softmax
    auto shape = ge::Shape({curRowInUb, gatingAlignCol});
    uint64_t softMaxMaxTmpSize = GetSoftMaxMaxTmpSize(shape, FP32_SIZE, IS_SOFTMAX_REUSE_SOURCE);

    // 3.计算topk
    uint64_t topKUbOutBufferSize = calcUbAlignBufferSize(curRowInUb, k, typeSize);
    uint64_t topKIndicesBufferSize = calcUbAlignBufferSize(curRowInUb, k, INT32_SIZE);

    uint32_t maxValue;
    uint32_t minValue;
    GetTopKMaxMinTmpSize(ascendcPlatform, gatingAlignCol, curRowInUb, IS_TOP_K_REUSE_SOURCE, true, TopKMode::TOPK_NORMAL,
                         true, FP32_SIZE, maxValue, minValue);

    maxValue = maxValue > softMaxMaxTmpSize ? maxValue : softMaxMaxTmpSize;

    uint64_t finishedUbBufferSize = CeilDiv(BOOL_SIZE * curRowInUb, BLOCK_SIZE) * BLOCK_SIZE;

    // sourceRowOut用来复用缓存softmax输出，按r*E分配大小
    uint64_t softmaxOutAlignBufferSize = curRowInUb * gatingAlignCol * INT32_SIZE;

    if (gatingAlignCol * INT32_SIZE + gatingBufferSize + topKUbOutBufferSize + topKIndicesBufferSize + maxValue +
            finishedUbBufferSize + softmaxOutAlignBufferSize >
        tmpUbSize) {
      return false;
    }
    return true;
  }

  // 1.搬入gating
  uint64_t gatingAlignBufferSize = curRowInUb * gatingAlignCol;
  gatingAlignBufferSize = gatingAlignBufferSize * typeSize;
  if (gatingAlignBufferSize > tmpUbSize) {
    return false;
  }

  // 2.执行softmax
  auto shape = ge::Shape({curRowInUb, gatingAlignCol});
  uint64_t softMaxMaxTmpSize = GetSoftMaxMaxTmpSize(shape, typeSize, IS_SOFTMAX_REUSE_SOURCE);

  // 3.计算topk
  uint32_t maxValue;
  uint32_t minValue;
  GetTopKMaxMinTmpSize(ascendcPlatform, gatingAlignCol, curRowInUb, IS_TOP_K_REUSE_SOURCE, true, TopKMode::TOPK_NORMAL,
                       true, typeSize, maxValue, minValue);

  maxValue = maxValue > softMaxMaxTmpSize ? maxValue : softMaxMaxTmpSize;

  uint64_t topKOutAlignBufferSize = calcUbAlignBufferSize(curRowInUb, k, typeSize);
  uint64_t topKIndicesAlignBufferSize = calcUbAlignBufferSize(curRowInUb, k, INT32_SIZE);

  uint64_t finishedUbBufferSize = CeilDiv(BOOL_SIZE * curRowInUb, BLOCK_SIZE) * BLOCK_SIZE;

  // sourceRowOut用来复用缓存softmax输出，按r*E分配大小
  uint64_t softmaxOutAlignBufferSize = curRowInUb * gatingAlignCol * INT32_SIZE;

  if (gatingAlignCol * INT32_SIZE + gatingAlignBufferSize + topKOutAlignBufferSize + topKIndicesAlignBufferSize +
          softmaxOutAlignBufferSize + maxValue + finishedUbBufferSize >
      tmpUbSize) {
    return false;
  }
  return true;
}

uint32_t MoeGatingTopKSoftmaxV2EKFullLoadTiling::calcMaxRowInUb(const bool doubleBufferFlag, const uint64_t ubSize,
                                                                const ge::DataType dtype, const uint32_t k,
                                                                const uint32_t blockRow, const uint32_t col) {
  uint32_t ubOuter = 1;
  uint64_t tmpUbSize = doubleBufferFlag ? ubSize / 2 : ubSize;
  uint64_t curRowInUb;
  while (true) {
    curRowInUb = CeilDiv(blockRow, ubOuter);
    if (isBufferSizeEnough(curRowInUb, gatingAlignCol, tmpUbSize, dtype, k)) {
      break;
    }
    ubOuter++;
    if (ubOuter > blockRow) {
      return 0;
    }
  }
  return curRowInUb;
}

REGISTER_TILING_TEMPLATE("MoeGatingTopKSoftmaxV2", MoeGatingTopKSoftmaxV2EKFullLoadTiling, 10000);
}  // namespace optiling