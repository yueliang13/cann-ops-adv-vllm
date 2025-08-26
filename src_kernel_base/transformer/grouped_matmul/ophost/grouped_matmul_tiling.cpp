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
 * \file grouped_matmul_tiling.cpp
 * \brief
 */
#include "grouped_matmul_tiling.h"

#include <climits>
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "log/ops_log.h"
#include "error/ops_error.h"
#include "tiling/tiling_base.h"
using namespace ge;
using namespace AscendC;

template <typename T1, typename T2>
static T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

namespace optiling {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t ANTIQUANT_SCALE_INDEX = 5;
constexpr uint32_t GROUPLIST_INDEX = 7;
constexpr uint32_t PER_TOKEN_SCALE_INDEX = 8;
constexpr uint32_t SCALE_INDEX = 3;
constexpr uint32_t Y_INDEX = 0;
constexpr int64_t BEST_L1_PARTA = 256 * 1024;
constexpr int64_t BEST_L1_PARTB = 128 * 1024;
constexpr uint32_t L1_PARTA_SIZE = 256 * 1024;
constexpr int32_t BEST_BASEN = 256;
constexpr int32_t BEST_BASEN_QUANT_ONE_GROUP = 128;
constexpr int32_t BEST_BASEM_QUANT_ONE_GROUP = 256;
constexpr int32_t BEST_BASEK_QUANT_ONE_GROUP = 128;
constexpr int32_t BEST_BASEN_MSD = 512;
constexpr int32_t BEST_UB_BASEK = 256;
constexpr int32_t BEST_UB_BASEN = 512;
constexpr int32_t MAX_BASEM = 256;
constexpr uint32_t A16W8_MSD_STEP = 2;
constexpr uint32_t A16W8_MSD_KN_BASE_BLOCK = 128;
constexpr uint32_t A16W8_MSD_AVERAGE_TOKEN_NUM = 64;
constexpr uint32_t A16W8_MSD_MAX_K = 12 * 1024;
constexpr uint32_t A16W8_MSD_MIN_N = 1024;
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32;  // 32: a block has 32 bytes data
constexpr uint32_t UB_ANTIQUANT_PER_BLOCK_ALIGN = 4 * 1024;
constexpr uint32_t UB_A16W8_BLOCK_NUM_FP16 = 6;  // 2 * sizeof(int8) + 2 * sizeof(half)
constexpr uint32_t UB_A16W8_IO_USED_BLOCK_FP16 = 6;
constexpr uint32_t UB_A16W8_BLOCK_NUM_BF16 = 8;  // tmpUb used 2 blks
constexpr uint32_t UB_A16W8_IO_USED_BLOCK_BF16 = 6;
constexpr uint32_t UB_A16W4_BLOCK_NUM_FP16 = 5;  // 2 * sizeof(int4) + 2 * sizeof(half)
constexpr uint32_t UB_A16W4_IO_USED_BLOCK_FP16 = 5;
constexpr uint32_t UB_A16W4_BLOCK_NUM_BF16 = 7;  // tmpUb used 2 blks
constexpr uint32_t UB_A16W4_IO_USED_BLOCK_BF16 = 5;
constexpr uint32_t UB_DYNAMIC_QUANT_BLOCK_NUM = 28;
constexpr uint32_t UB_DUNAMIC_QUANT_IO_USED_BLOCK = 12;
constexpr uint32_t UB_QUANT_BLOCK_ALIGN = 2 * 1024;
constexpr uint32_t UB_A16W8_MSD_BLOCK_NUM = 30;
constexpr uint32_t UB_A16W8_MSD_IO_USED_BLOCK = 6;
constexpr uint32_t UB_A16W8_MSD_BLOCK_ALIGN = 512;
constexpr uint32_t UB_STATIC_QUANT_BLOCK_NUM_BF16 = 20;
constexpr uint32_t UB_STATIC_QUANT_BLOCK_NUM_FP16 = 24;
constexpr uint32_t UB_STATIC_QUANT_IO_USED_BLOCK = 12;
constexpr uint32_t QUEUE_DOUBLE_BUFFER = 2;
constexpr uint32_t FP32_DATATYPE_SIZE = 4;
constexpr uint64_t TILING_KEY = 0;
constexpr uint64_t TILING_KEY_TRANS_X = 1;
constexpr uint64_t TILING_KEY_TRANS_W = 2;
constexpr uint64_t TILING_KEY_ANTIQUANT_PERFORMANCE = 3;
constexpr uint64_t TILING_KEY_QUANT_2VECTOR = 4;
constexpr uint64_t TILING_KEY_QUANT_2VECTOR_TRANS_W = 5;
constexpr uint64_t TILING_KEY_A16W8_MSD = 6;
constexpr uint64_t TILING_KEY_A16W8_MSD_TRANS_W = 7;
constexpr uint64_t TILING_KEY_A8W4_MSD = 8;
constexpr uint64_t ATTR_INDEX_SPLIT_ITEM = 0;
constexpr uint64_t ATTR_INDEX_TRANS_W = 2;
constexpr uint64_t ATTR_INDEX_TRANS_X = 3;
constexpr uint64_t ATTR_INDEX_GROUPTYPE = 4;
constexpr uint32_t ATTR_INDEX_GROUP_LIST_TYPE = 5;
constexpr uint64_t ATTR_INDEX_ACT_TYPE = 6;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int32_t NO_SPLIT = -1;
constexpr int32_t SPLIT_M = 0;
constexpr int32_t SPLIT_K = 2;
constexpr int64_t ANTIQUANT_PERFORMANCE_THRESHOLD = 5 * 1024 * 1024;  // used for whether going into performance branch in antiquant case, by experiment
constexpr int64_t ACT_TYPE_GELU = 2;
constexpr uint16_t MAX_TENSOR_CONT = 128;
constexpr int64_t FULL_K_SINGLE_N = 1280; // used for fullload k case, by experiment
constexpr int64_t FULL_K_N_THRESHOLD = 2560; // used for fullload k case, by experiment
constexpr int64_t FULL_K_M_THRESHOLD = 2048; // used for fullload k case, by experiment
constexpr int64_t FULL_K_M_E_THRESHOLD = 256; // used for fullload k case, by experiment
constexpr int64_t FULL_K_MAX_K_THRESHOLD = 384; // used for fullload k case, by experiment
constexpr int64_t FULL_K_MIN_K_THRESHOLD = 320; // used for fullload k case, by experiment

static inline uint32_t SixteenAlign(uint32_t a, bool up = false) {
    if (up) {
        a += 15;  // 15: 16 bytes up-align
    }
    return a & ~15;  // ~15: 16 bytes down-align
}

struct GMMCompileInfo {
  uint32_t aicNum;
  uint32_t aivNum;
  uint64_t ubSize;
  uint64_t l1Size;
  uint64_t l2Size;
  uint64_t l0CSize;
  uint64_t l0ASize;
  uint64_t l0BSize;
  platform_ascendc::SocVersion socVersion;
};

class GMMTiling {
 public:
  GMMTilingData tilingData;
  ge::graphStatus Init(const gert::TilingContext* context);
  ge::graphStatus RunFusionKernelTiling(gert::TilingContext* context);

 protected:
  ge::graphStatus CalMMTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr);
  ge::graphStatus GMMSetMMTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr);
  void GMMSetTilingKey(gert::TilingContext* context) const;
  ge::graphStatus GMMGetAttrs(const gert::TilingContext* context);
  ge::graphStatus GMMSetUbDivideBlk();
  ge::graphStatus GMMSetUbDivideBlkAntiquant();
  ge::graphStatus GMMSetUbDivideBlkQuant();
  ge::graphStatus GMMCalUbSize(const gert::TilingContext* context, uint32_t ubSize);
  int64_t GMMGetBS(const gert::Shape xShape) const;
  ge::graphStatus PrepareTilingData(const gert::TilingContext* context);
  ge::graphStatus CheckWeightNZShape(const gert::TilingContext* context, int64_t numInOneBlk) const;
  ge::graphStatus GMMGetTensorShapeSplitM(const gert::TilingContext* context, const gert::Shape xShape,
                                          const gert::Shape wShape);
  ge::graphStatus GMMGetTensorShapeSplitK(const gert::TilingContext* context, const gert::Shape xShape,
                                          const gert::Shape wShape);
  ge::graphStatus SplitMSingleXSingleWeightSingleY(const gert::Shape xShape, const gert::Shape wShape);
  ge::graphStatus SplitMSingleXSeparatedWeight(const gert::TilingContext* context, const gert::Shape xShape);
  ge::graphStatus SeparatedXSeparatedWeight(const gert::TilingContext* context);
  ge::graphStatus SeparatedXSingleWeight(const gert::TilingContext* context, const gert::Shape wShape);
  ge::graphStatus SplitKSingleXSingleWeightSingleY(const gert::TilingContext* context, const gert::Shape xShape,
                                                   const gert::Shape wShape);
  ge::graphStatus DivideUbAndSetWorkspace(gert::TilingContext* context, const uint32_t& aicNum);
  void DivideUbAndSetWorkspaceAntiquant(size_t* workspaces, const uint32_t& aicNum, uint32_t &ubSize);
  ge::graphStatus CalcStepKaKb(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr,
                               int64_t mInMM, uint32_t& mmStepKa, uint32_t& mmStepKb);
  ge::graphStatus SetBias(const gert::TilingContext* context, matmul_tiling::MultiCoreMatmulTiling& mm) const;
  int32_t FindBestSingleNPertoken(const uint32_t aicNum) const;
  void FindBestUsedCoreNumOneGroup(const uint32_t aicNum);
  ge::graphStatus SetWorkspscesPerTokenQuant(const uint32_t aicNum, size_t* workspaces);
  void SetTilingDataIsSingleTensor();
  ge::graphStatus GetPerGroupNum(const gert::TilingContext* context);
  ge::graphStatus CheckMKN(const gert::TilingContext* context);
  void FullLoadK(const GMMCompileInfo* compileInfoPtr);

 private:
  int32_t mList_[MAX_TENSOR_CONT] = {0};
  int32_t kList_[MAX_TENSOR_CONT] = {0};
  int32_t nList_[MAX_TENSOR_CONT] = {0};
  int64_t maxM_ = 0;
  int64_t maxN_ = 0;
  int64_t maxK_ = 0;
  int32_t minK_ = INT32_MAX;
  int32_t baseM_;
  int32_t baseN_;
  int32_t baseK_;
  uint64_t ubSize_;
  uint32_t mmDataTypeSize_;
  uint32_t ubDivideBlkNum_;
  uint32_t ubIoBlkNum_;
  uint32_t ubBlockAlign_;
  uint64_t workspacesSize_ = 0;  // for antiquant
  uint32_t groupNum_ = 0;
  bool transposeWeight_;
  bool transposeX_;
  bool isSingleWeight_;
  bool isSingleX_;
  bool isSingleY_;
  bool isAllSingleTensor_;
  bool hasBias_;
  int32_t groupType_;
  int64_t splitItem_;
  uint32_t groupListType_;
  uint32_t xKDim_;
  uint32_t weightNDim_;
  uint32_t xDimNum_;
  bool antiquantPerformance_ = false;
  uint32_t actType_;
  uint32_t usedCoreNum_ = 0;

  ge::DataType xDType_ = ge::DT_UNDEFINED;
  ge::DataType mmDType_ = ge::DT_UNDEFINED;
  ge::DataType weightDtype_ = ge::DT_UNDEFINED;
  ge::DataType scaleDtype_ = ge::DT_UNDEFINED;
  ge::DataType yDtype_ = ge::DT_UNDEFINED;
  uint32_t perTokenOrPerGroupSize_ = 0;  // in quant case, it indicates pertoken flag; in antiquant case, it represents pergroup size
  bool isA16W8Msd_ = false;
  uint32_t totalM_ = 0;
  matmul_tiling::CubeFormat wFormat_;
  int32_t nzFactor_;  // for weight nz format
};

ge::graphStatus GMMTiling::CheckWeightNZShape(const gert::TilingContext* context, int64_t numInOneBlk) const {
  OPS_ERR_IF(numInOneBlk <= 0, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "numInOneBlk, the "
             "input of CheckWeightNZShape has an invaild value %ld", numInOneBlk), return ge::GRAPH_FAILED);
  size_t i = 0;
  while (true) {
    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, i++);
    if (wTensor == nullptr) { break; }
    gert::Shape wOriginShape = wTensor->GetOriginShape();
    int64_t lastDimValue = wOriginShape.GetDim(wOriginShape.GetDimNum() - 1);  // inner axis
    OPS_ERR_IF(lastDimValue % numInOneBlk != 0,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
               "the inner axis size of nz weight is expected to be a multiple of 32B, "
               "but now the inner axis size is %ld.", lastDimValue),
               return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::CheckMKN(const gert::TilingContext* context) {
  mmDataTypeSize_ = GetSizeByDataType(mmDType_);
  OPS_ERR_IF(mmDataTypeSize_ == 0, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "GMM get mm dtype[%s] size is 0.", TypeUtils::DataTypeToAscendString(mmDType_).GetString()),
             return ge::GRAPH_FAILED);
  uint32_t numInOneBlk = ONE_BLK_SIZE / mmDataTypeSize_;
  OPS_ERR_IF(numInOneBlk == 0, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "GMM numInOneBlk cannot be 0."), return ge::GRAPH_FAILED);
  int64_t maxMKN = INT_MAX / numInOneBlk * numInOneBlk;
  OPS_ERR_IF(maxM_ > maxMKN || maxN_ > maxMKN || maxK_ > maxMKN,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "32B-aligned m, n or k axis is out of range int32!"),
             return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

void GMMTiling::SetTilingDataIsSingleTensor() {
  tilingData.gmmBaseParams.set_singleWeight(static_cast<uint32_t>(isSingleWeight_));
  tilingData.gmmBaseParams.set_singleX(static_cast<uint32_t>(isSingleX_));
  tilingData.gmmBaseParams.set_singleY(static_cast<uint32_t>(isSingleY_));
}

ge::graphStatus GMMTiling::PrepareTilingData(const gert::TilingContext* context) {
  // get transpose and groupType
  OPS_ERR_IF(GMMGetAttrs(context) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMMGetAttrs failed"),
             return ge::GRAPH_FAILED);
  // get the first tensor's shape of weight and x
  auto xTensor = context->GetDynamicInputTensor(X_INDEX, 0);  // 0: get first tensor
  OPS_LOG_E_IF_NULL(context, xTensor, return ge::GRAPH_FAILED);
  gert::Shape xShape = xTensor->GetStorageShape();
  xDimNum_ = static_cast<uint32_t>(xShape.GetDimNum());
  xKDim_ = transposeX_ ? 0 : xDimNum_ - 1;  // 0: when x is transposed, the first dim is k; -1：otherwise, the last dim is k

  auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
  OPS_LOG_E_IF_NULL(context, wTensor, return ge::GRAPH_FAILED);
  gert::Shape wShape = wTensor->GetOriginShape();
  uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
  weightNDim_ = transposeWeight_ ? wDimNum - 2 : wDimNum - 1;  // -2: when w is transposed, the last 2 dim is n; -1: otherwise, the last dim is n
  nzFactor_ = 1;  // init
  if (wFormat_ == matmul_tiling::CubeFormat::NZ) {
    uint32_t numInOneBlk = UB_BLOCK_UNIT_SIZE / std::max(1, GetSizeByDataType(weightDtype_));
    if (wDimNum >= 4) {  // 4: least dim num of nz format tensor
      weightNDim_ = transposeWeight_ ? wDimNum - 3 : wDimNum - 4;  // -3: when w is transposed, the last 3 dim is n/nzFactor; -4: when w has nz format, the last 4 dim is n/nzFactor
      // nzFactor_ is a factor used to compute n axis size. If weight is transposed, nzFactor_ is 16; otherwise nzFactor_ is 16 for bf16, 32 for int8
      nzFactor_ = transposeWeight_ ? 16 : static_cast<int32_t>(numInOneBlk);
    } else {
      OPS_ERR_IF(CheckWeightNZShape(context, static_cast<int64_t>(numInOneBlk)) != ge::GRAPH_SUCCESS,
                 OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "the shape of nz weight is invaild."),
                 return ge::GRAPH_FAILED);
    }
  }
  isSingleWeight_ = (context->GetDynamicInputTensor(WEIGHT_INDEX, 1) == nullptr);
  isSingleX_ = (context->GetDynamicInputTensor(X_INDEX, 1) == nullptr);
  isSingleY_ = (splitItem_ == 2 || splitItem_ == 3);  // 2: when x is multi-tensor, y is single-tensor; 3: when x is single-tensor, y is single-tensor
  SetTilingDataIsSingleTensor();

  if (groupType_ == SPLIT_M) {
    return GMMGetTensorShapeSplitM(context, xShape, wShape);
  }
  if (groupType_ == SPLIT_K) {
    return GMMGetTensorShapeSplitK(context, xShape, wShape);
  }
  if (groupType_ == NO_SPLIT) {  // not split any axis
    if (isSingleWeight_ && wDimNum > 2) {  // 2: dim of splited weight tensor
      return SeparatedXSingleWeight(context, wShape);
    }
    return SeparatedXSeparatedWeight(context);
  }
  OPS_LOG_E(context->GetNodeName(), "GMM_tiling: not support groupType_=%d, isSingleWeight_=%d, isSingleX_=%d, isSingleY_=%d",
            groupType_, isSingleWeight_, isSingleX_, isSingleY_);
  return ge::GRAPH_FAILED;
}

ge::graphStatus GMMTiling::GMMGetTensorShapeSplitM(const gert::TilingContext* context, const gert::Shape xShape,
    const gert::Shape wShape) {
    if (isSingleX_ && isSingleWeight_ && isSingleY_) {  // split M, s-s-s
      return SplitMSingleXSingleWeightSingleY(xShape, wShape);
    }
    if (isSingleX_ && !isSingleWeight_ && isSingleY_) {  // split M, s-m-s
      return SplitMSingleXSeparatedWeight(context, xShape);
    }
    if (isSingleX_ && !isSingleWeight_ && !isSingleY_) {  // splitM, s-m-m
      return SplitMSingleXSeparatedWeight(context, xShape);
    }
    if (!isSingleX_ && !isSingleWeight_ && isSingleY_) {  // split M, m-m-s
      return SeparatedXSeparatedWeight(context);
    }
    if (!isSingleX_ && isSingleWeight_) {  // split M, m-s-m/m-s-s
      return SeparatedXSingleWeight(context, wShape);
    }
    if (!isSingleX_ && !isSingleWeight_ && !isSingleY_) {  // split M, m-m-m
      return SeparatedXSeparatedWeight(context);
    }
    OPS_LOG_E(context->GetNodeName(), "GMM_tiling: not support groupType_=%d, isSingleWeight_=%d, isSingleX_=%d, isSingleY_=%d",
              groupType_, isSingleWeight_, isSingleX_, isSingleY_);
    return ge::GRAPH_FAILED;
}

ge::graphStatus GMMTiling::GMMGetTensorShapeSplitK(const gert::TilingContext* context, const gert::Shape xShape,
    const gert::Shape wShape) {
    if (isSingleX_ && isSingleWeight_ && isSingleY_) {  // splitK, s-s-s
      return SplitKSingleXSingleWeightSingleY(context, xShape, wShape);
    }
    if (!isSingleX_ && isSingleWeight_) {  // splitK, m-s-m/m-s-s
      return SeparatedXSingleWeight(context, wShape);
    }
    OPS_LOG_E(context->GetNodeName(), "GMM_tiling: not support groupType_=%d, isSingleWeight_=%d, isSingleX_=%d, isSingleY_=%d",
              groupType_, isSingleWeight_, isSingleX_, isSingleY_);
    return ge::GRAPH_FAILED;
}

/** @brief split M：single-single-single(s-s-s)
*/
ge::graphStatus GMMTiling::SplitMSingleXSingleWeightSingleY(const gert::Shape xShape, const gert::Shape wShape) {
  groupNum_ = static_cast<int32_t>(wShape.GetDim(0));
  int64_t m = GMMGetBS(xShape);
  int64_t k = xShape.GetDim(xKDim_);
  int64_t n = wShape.GetDim(weightNDim_) * static_cast<int64_t>(nzFactor_);
  kList_[0] = static_cast<int32_t>(k);  // if split M axis, the K axis values of x tensorList are all the same.
  nList_[0] = static_cast<int32_t>(n);
  mList_[0] = -1;
  maxM_ = m;
  maxK_ = k;
  maxN_ = n;
  totalM_ = m;
  return ge::GRAPH_SUCCESS;
}

/** @brief split M：single-multi-single(s-m-s)/single-multi-multi(s-m-m), share the same function.
*/
ge::graphStatus GMMTiling::SplitMSingleXSeparatedWeight(const gert::TilingContext* context, const gert::Shape xShape) {
  int64_t m = GMMGetBS(xShape);
  int64_t k = xShape.GetDim(xKDim_);
  for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, i);
    if (wTensor == nullptr) { break; }  // when x has multi tensors, xTensor is allowed to be empty
    auto wShape = wTensor->GetOriginShape();

    groupNum_ += 1;
    kList_[i] = static_cast<int32_t>(k);
    int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
    nList_[i] = static_cast<int32_t>(n);
    maxN_ = std::max(maxN_, n);
  }
  mList_[0] = -1;  // mList is unknown right now
  maxM_ = m;
  maxK_ = k;
  totalM_ = m;

  return ge::GRAPH_SUCCESS;
}

/** @brief split M：multi-multi-single(m-m-s); no split: multi-multi-multi(m-m-m), share the same function
*/
ge::graphStatus GMMTiling::SeparatedXSeparatedWeight(const gert::TilingContext* context) {
  for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, i);
    auto xTensor = context->GetDynamicInputTensor(X_INDEX, i);
    if (wTensor == nullptr || xTensor == nullptr) { break; }
    auto wShape = wTensor->GetOriginShape();
    auto xShape = xTensor->GetStorageShape();
    groupNum_ += 1;
    int64_t m = GMMGetBS(xShape);
    int64_t k = xShape.GetDim(xKDim_);
    int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
    mList_[i] = static_cast<int32_t>(m);
    kList_[i] = static_cast<int32_t>(k);
    nList_[i] = static_cast<int32_t>(n);
    maxM_ = std::max(maxM_, m);
    maxK_ = std::max(maxK_, k);
    maxN_ = std::max(maxN_, n);
    totalM_ += m;
  }
  groupType_ = NO_SPLIT;
  return ge::GRAPH_SUCCESS;
}

/** @brief split M : multi-single-multi(m-s-m), split K : multi-single-multi(m-s-m), share the same function
*/
ge::graphStatus GMMTiling::SeparatedXSingleWeight(const gert::TilingContext* context, const gert::Shape wShape) {
  int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
  for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
    auto xTensor = context->GetDynamicInputTensor(X_INDEX, i);
    if (xTensor == nullptr) { break; }  // when x has multi tensors, xTensor is allowed to be empty
    auto xShape = xTensor->GetStorageShape();
    groupNum_ += 1;
    int64_t m = GMMGetBS(xShape);
    int64_t k = xShape.GetDim(xKDim_);
    mList_[i] = static_cast<int32_t>(m);
    kList_[i] = static_cast<int32_t>(k);
    nList_[i] = static_cast<int32_t>(n);
    maxM_ = std::max(maxM_, m);
    maxK_ = std::max(maxK_, k);
    totalM_ += m;
  }
  maxN_ = n;
  groupType_ = NO_SPLIT;
  return ge::GRAPH_SUCCESS;
}

/** @brief split K single-single-single
*/
ge::graphStatus GMMTiling::SplitKSingleXSingleWeightSingleY(const gert::TilingContext* context,
    const gert::Shape xShape, const gert::Shape wShape) {
  int64_t m = GMMGetBS(xShape);
  int64_t k = xShape.GetDim(xKDim_);
  int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;

  auto groupListTensor = context->GetDynamicInputTensor(GROUPLIST_INDEX, 0);
  if (groupListTensor == nullptr) {
    OPS_LOG_E(context->GetNodeName(), "groupListTensor is nullptr");
    return ge::GRAPH_FAILED;
  }
  gert::Shape groupListShape = groupListTensor->GetStorageShape();
  groupNum_ = static_cast<int32_t>(groupListShape.GetDim(0));  // 0: the first dim of groupList is groupNum
  mList_[0] = static_cast<int32_t>(m);
  nList_[0] = static_cast<int32_t>(n);
  kList_[0] = -1;
  maxM_ = m;
  maxN_ = n;
  maxK_ = k;
  totalM_ = m;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::Init(const gert::TilingContext* context) {
  OPS_ERR_IF(PrepareTilingData(context) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM PrepareTilingData failed."),
             return ge::GRAPH_FAILED);
  auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
  OPS_LOG_E_IF_NULL(context, compileInfoPtr, return ge::GRAPH_FAILED);  // check compileInfoPtr is not null
  // check whether x, weight and y are all single tensor
  isAllSingleTensor_ = isSingleX_ && isSingleWeight_ && isSingleY_;
  bool isA16W8 = (xDType_ == ge::DT_FLOAT16 || xDType_ == ge::DT_BF16) && weightDtype_ == ge::DT_INT8;
  // check whether k and n are supported in msd
  bool isKNForA16W8MSD = maxN_ % A16W8_MSD_KN_BASE_BLOCK == 0 && maxK_ % A16W8_MSD_KN_BASE_BLOCK == 0 &&
                         maxK_ <= A16W8_MSD_MAX_K && maxN_ >= A16W8_MSD_MIN_N;
  // check whether total token num and average token num are supported in msd
  bool isMForA16W8MSD = totalM_ <= A16W8_MSD_AVERAGE_TOKEN_NUM * groupNum_;
  isA16W8Msd_ = isAllSingleTensor_ && groupType_ == SPLIT_M && isA16W8 && isKNForA16W8MSD && isMForA16W8MSD;
  mmDType_ = isA16W8Msd_ ? ge::DT_INT8 : xDType_;
  OPS_ERR_IF(CheckMKN(context) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM CheckMKN failed."),
             return ge::GRAPH_FAILED);
  auto biasPtr = context->GetDynamicInputTensor(BIAS_INDEX, 0);  // 0: obtain the first tensor of the tensorList
  hasBias_ = !(biasPtr == nullptr || biasPtr->GetStorageShape().GetShapeSize() == 0);

  tilingData.gmmArray.set_mList(mList_);
  tilingData.gmmArray.set_kList(kList_);
  tilingData.gmmArray.set_nList(nList_);
  tilingData.gmmBaseParams.set_groupNum(groupNum_);
  tilingData.gmmBaseParams.set_m(totalM_);
  tilingData.gmmBaseParams.set_hasBias(static_cast<uint32_t>(hasBias_));
  tilingData.gmmBaseParams.set_groupType(static_cast<int32_t>(groupType_));
  tilingData.gmmBaseParams.set_activeType(actType_);
  tilingData.gmmBaseParams.set_quantParam(perTokenOrPerGroupSize_);
  tilingData.gmmBaseParams.set_groupListType(groupListType_);
  OPS_LOG_I(context->GetNodeName(), "GMM_tiling: groupNum_ is %u, maxM_ is %ld, maxK_ is %ld, maxN_ is %ld.",
            groupNum_, maxM_, maxK_, maxN_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GetPerGroupNum(const gert::TilingContext* context) {
  auto antiquantScale = context->GetDynamicInputTensor(ANTIQUANT_SCALE_INDEX, 0);
  OPS_LOG_E_IF_NULL(context, antiquantScale, return ge::GRAPH_FAILED);
  auto antiquantScaleShape = antiquantScale->GetStorageShape();
  int64_t dimNum = antiquantScaleShape.GetDimNum();
  auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
  OPS_LOG_E_IF_NULL(context, wTensor, return ge::GRAPH_FAILED);
  gert::Shape wShape = wTensor->GetOriginShape();
  size_t wDimNum = wShape.GetDimNum();
  if ((isSingleWeight_ && wDimNum > 2 && dimNum == 3) || (!isSingleWeight_ && dimNum == 2)) {  // 2 and 3: dim threshold
    int64_t g = antiquantScaleShape.GetDim(dimNum - 2);
    perTokenOrPerGroupSize_ = g > 1 ? kList_[0] / g : 0;
    tilingData.gmmBaseParams.set_quantParam(perTokenOrPerGroupSize_);
  }
  return ge::GRAPH_SUCCESS;
}

void GMMTiling::DivideUbAndSetWorkspaceAntiquant(size_t* workspaces, const uint32_t& aicNum, uint32_t &ubSize) {
    if (isA16W8Msd_) {
      // whole workspace for a16w8 msd scene is combined by workspace for global max of each row (m * 8),
      // workspace for local reduce sum of each row (m * aicNum), workspace for data after prerpocessing x (2 * m * k)
      // and workspace for matmul output (2 * m * n)
      // 32: need 32 byte to store global max
      workspaces[0] += (aicNum * sizeof(float) + 32 +
                        A16W8_MSD_STEP * (maxK_ * sizeof(int8_t) + maxN_ * sizeof(int32_t))) * totalM_;
      // 7: make aicnum align up to 8
      uint32_t alignedAicNum = (aicNum + 7) & (~7);
      ubSize = static_cast<uint32_t>(ubSize_ - (baseM_ / A16W8_MSD_STEP) * alignedAicNum * sizeof(float));
      // workspacesSize in GMMBaseParams is size of matmul left input matrix of matmul.
      workspacesSize_ += A16W8_MSD_STEP * static_cast<uint64_t>(maxK_) * static_cast<uint64_t>(maxM_);
    } else {
      for (uint32_t i = 0; i < groupNum_; i++) {
        bool isAllSingleTensor = isSingleX_ && isSingleWeight_ && isSingleY_;
        int32_t kInList = isAllSingleTensor ? kList_[0] : kList_[i];  // in s-s-s case，k only exits in the first of the list
        int32_t nInList = isAllSingleTensor ? nList_[0] : nList_[i];  // in s-s-s case，n only exits in the first of the list
        int32_t k = kList_[0] == -1 ? static_cast<int32_t>(maxK_) : kInList;
        int32_t n = nList_[0] == -1 ? static_cast<int32_t>(maxN_) : nInList;
        minK_ = std::min(minK_, k);
        workspacesSize_ += static_cast<uint64_t>(k) * static_cast<uint64_t>(n);
      }
      // when minK * baseN * coreNum * sizeof(float16) > 12M, it goes into antiquantPerformance branch (12M is obtained by test).
      int32_t dimMN =
        CeilDiv(CeilDiv(maxM_, groupNum_), baseM_) * CeilDiv(maxN_, baseN_);
      bool goodCubeUtility = dimMN * (xDType_ == ge::DT_BF16 ? 2 : 1) >= static_cast<int32_t>(aicNum * 0.4);  // 0.4: a factor, in practice.
      antiquantPerformance_ =
        goodCubeUtility && static_cast<int64_t>(minK_) * baseN_ * aicNum >= ANTIQUANT_PERFORMANCE_THRESHOLD;
      uint32_t maxUbBaseN = BEST_UB_BASEN;
      if (transposeWeight_) {
        maxUbBaseN = baseN_;
      } else if (antiquantPerformance_) {
        // 2: use 2 pieces of workspace in antiquantPerformance branch
        workspacesSize_ = static_cast<uint64_t>(maxN_) * maxK_ * 2;
      }
      // 2: 2 InQueue(antiquant_scale,antiquant_offset)
      ubSize = static_cast<uint32_t>(ubSize_ - maxUbBaseN * mmDataTypeSize_ * QUEUE_DOUBLE_BUFFER * 2);
      workspaces[0] += workspacesSize_ * mmDataTypeSize_;
    }
}

ge::graphStatus GMMTiling::DivideUbAndSetWorkspace(gert::TilingContext* context, const uint32_t& aicNum) {
  size_t* workspaces = context->GetWorkspaceSizes(1);  // get second variable
  OPS_LOG_E_IF_NULL(context, workspaces, return ge::GRAPH_FAILED);  // check workspaces is not null
  workspaces[0] = SYS_WORKSPACE_SIZE;  // default size
  if (weightDtype_ != ge::DT_INT8 && weightDtype_ != ge::DT_INT4) {
    return ge::GRAPH_SUCCESS;
  }
  uint32_t ubSize = static_cast<uint32_t>(ubSize_);
  if ((xDType_ == ge::DT_BF16 || xDType_ == ge::DT_FLOAT16)) {
    DivideUbAndSetWorkspaceAntiquant(workspaces, aicNum, ubSize);
    OPS_ERR_IF(GetPerGroupNum(context) != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(
               context->GetNodeName(), "GetPerGroupNum failed."), return ge::GRAPH_FAILED);
  } else if (xDType_ == ge::DT_INT8 && yDtype_ != ge::DT_INT32) {
    uint32_t scaleDataTypeSize = GetSizeByDataType(scaleDtype_);
    ubSize = perTokenOrPerGroupSize_ == 1 ?  // is perToken
      static_cast<uint32_t>(ubSize_ -
                            (baseN_ * scaleDataTypeSize + baseM_ * sizeof(float)) * QUEUE_DOUBLE_BUFFER) :
      static_cast<uint32_t>(ubSize_ - baseN_ * scaleDataTypeSize * QUEUE_DOUBLE_BUFFER);
    OPS_ERR_IF(SetWorkspscesPerTokenQuant(aicNum, workspaces) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SetWorkspscesPerTokenQuant failed."),
               return ge::GRAPH_FAILED);
  } else if (xDType_ == ge::DT_INT8 && yDtype_ == ge::DT_INT32) {
    FindBestUsedCoreNumOneGroup(aicNum);
    return ge::GRAPH_SUCCESS;
  }
  OPS_ERR_IF(GMMSetUbDivideBlk() != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "GMMSetUbDivideBlk failed."),
             return ge::GRAPH_FAILED);
  OPS_ERR_IF(GMMCalUbSize(context, ubSize) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "GMMCalUbSize failed."),
             return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

int32_t GMMTiling::FindBestSingleNPertoken(const uint32_t aicNum) const {
  if (CeilDiv(maxN_, baseN_) * groupNum_ <= aicNum) {  // if all matmuls only occupy a part of cores
    return baseN_;
  }
  if (maxN_  >= 2048) {  // 2048: a threshold
    return 1024;  // 1024: max singleN
  }
  int32_t bestSingleN = baseN_;  // init bestSingleN
  uint32_t bestLastCycleCoreNum = (groupNum_ * CeilDiv(maxN_, bestSingleN)) % aicNum;  // init lastCycleCoreNum
  // 1024: max singleN
  for (int32_t tempSingleN = 1024 / baseN_ * baseN_; tempSingleN > baseN_; tempSingleN -= baseN_) {
    uint32_t lastCycleCoreNum = (groupNum_ * CeilDiv(maxN_, tempSingleN)) % aicNum;
    if (lastCycleCoreNum == 0) {
      bestSingleN = tempSingleN;
      break;
    }
    if (lastCycleCoreNum > bestLastCycleCoreNum ||
      (lastCycleCoreNum == bestLastCycleCoreNum && maxN_ % tempSingleN == 0)) {
      bestSingleN = tempSingleN;
      bestLastCycleCoreNum = lastCycleCoreNum;
    }
  }
  return bestSingleN;
}

void GMMTiling::FindBestUsedCoreNumOneGroup(const uint32_t aicNum) {
  if (groupNum_ > 1) {
    return;
  }
  uint32_t totalCoreNums = CeilDiv(maxN_, baseN_);
  // 3: if cube iterNum less or equal to 3, and more than half cores are unused in last iter, use less cores each iter
  if ((aicNum * 3 > totalCoreNums && totalCoreNums % aicNum <= aicNum / 2) || totalCoreNums < aicNum) {  // 2: half of aicNum
    uint32_t  cubeIterNum = CeilDiv(totalCoreNums, aicNum);
    usedCoreNum_ = CeilDiv(totalCoreNums, cubeIterNum);
  }
}

ge::graphStatus GMMTiling::SetWorkspscesPerTokenQuant(const uint32_t aicNum, size_t* workspaces) {
  if (aicNum == 0) {  // invaild value
    return ge::GRAPH_FAILED;
  }
  bool opt = (maxM_ <= 32 * groupNum_ && wFormat_ == matmul_tiling::CubeFormat::NZ) &&
             (!transposeWeight_ || maxN_ >= 2048);  // 32: a factor, 2048: a threshold.
  if (opt) {  // non-basic strategy. matmul output in non-continugous mode with singleN >= baseN
    workspaces[0] += maxM_ * maxN_ * sizeof(int32_t);
    int32_t bestSingleN = FindBestSingleNPertoken(aicNum);
    tilingData.gmmBaseParams.set_singleN(bestSingleN);
  } else {
    FindBestUsedCoreNumOneGroup(aicNum);
    // 4： when do cv parallelism, four pieces of workspace are used for storing four cycles of matmul output
    workspaces[0] += 4 * baseM_ * baseN_ * usedCoreNum_ * sizeof(int32_t);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::RunFusionKernelTiling(gert::TilingContext* context) {
  OPS_LOG_I(context->GetNodeName(), "Begin Run GMM Tiling");
  auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
  OPS_LOG_E_IF_NULL(context, compileInfoPtr, return ge::GRAPH_FAILED);  // check compileInfoPtr is not null

  ubSize_ = compileInfoPtr->ubSize;  // get ubSize from compileInfo
  const uint32_t& aicNum = compileInfoPtr->aicNum;  // get aicNum from compileInfo
  if (aicNum == 0) {  // invaild value
    return ge::GRAPH_FAILED;
  }
  usedCoreNum_ = aicNum;

  OPS_ERR_IF(CalMMTiling(context, compileInfoPtr) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM CalMMTiling failed"),
             return ge::GRAPH_FAILED);

  OPS_ERR_IF(GMMSetMMTiling(context, compileInfoPtr) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM GMMSetMMTiling failed"),
             return ge::GRAPH_FAILED);
  tilingData.gmmBaseParams.set_singleN(0);  // 0 is the default value
  FullLoadK(compileInfoPtr);
  OPS_ERR_IF(DivideUbAndSetWorkspace(context, aicNum) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM DivideUbAndSetWorkspace failed"),
             return ge::GRAPH_FAILED);
  tilingData.gmmBaseParams.set_workspaceSize(workspacesSize_);
  tilingData.mmTilingData.set_usedCoreNum(usedCoreNum_);  // usedCoreNum is ai_core num
  tilingData.gmmBaseParams.set_coreNum(usedCoreNum_);  // ai cube number
  GMMSetTilingKey(context);  // set tilingkey
  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->SetBlockDim(usedCoreNum_);  // block dim is the number of aicube
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  OPS_LOG_I(context->GetNodeName(), "End Run GMM Tiling");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GMMCalUbSize(const gert::TilingContext* context, uint32_t ubSize) {
  OPS_ERR_IF((ubDivideBlkNum_ == 0 || ubBlockAlign_ == 0),
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "ubDivideBlkNum and ubBlockAlign cannot be 0"),
             return ge::GRAPH_FAILED);
  uint32_t ubCalSize = ubSize / ubDivideBlkNum_;  // divide the UB into ubDivideBlkNum_ pieces
  ubCalSize = ubCalSize / ubBlockAlign_ * ubBlockAlign_;  // 16k/8k/4k align.
  uint32_t ubRestBytes = ubSize - ubCalSize * ubIoBlkNum_;  // compute the rest memory in UB space
  ubRestBytes = ubRestBytes / UB_BLOCK_UNIT_SIZE * UB_BLOCK_UNIT_SIZE;  // 32B align.
  OPS_ERR_IF((ubCalSize == 0 || ubRestBytes == 0),
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "ubCalSize and ubRestBytes cannot be 0"),
             return ge::GRAPH_FAILED);
  uint32_t ubBaseN = 0;  // init
  uint32_t ubBaseK = 0;  // init
  if (transposeWeight_) {
    ubBaseK = BEST_UB_BASEK;
    ubBaseN = ubCalSize / ubBaseK;
    uint32_t alignFactor = UB_BLOCK_UNIT_SIZE;
    if (weightDtype_ == ge::DT_INT4) {
        alignFactor <<= 1;  // int4 need 64 elements algin.
    }
    ubBaseN = ubBaseN / alignFactor * alignFactor;
  } else {
    if ((xDType_ == ge::DT_BF16 || xDType_ == ge::DT_FLOAT16) &&
        (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4)) {
      if (perTokenOrPerGroupSize_ > 0) {
        ubBaseK = perTokenOrPerGroupSize_;
        static const uint32_t MIN_UB_BASEN = 128;  // a threshold
        ubBaseN = std::min<uint32_t>(BEST_UB_BASEN, std::max<uint32_t>(MIN_UB_BASEN, (ubCalSize / ubBaseK + MIN_UB_BASEN - 1) / MIN_UB_BASEN * MIN_UB_BASEN));
      } else if (antiquantPerformance_) {
        ubBaseN = BEST_UB_BASEN;
      } else {
        ubBaseN = baseN_;
      }
    } else {
      ubBaseN = baseN_;
    }
    ubBaseK = ubCalSize / ubBaseN;  // ubCalSize is the number of elements, not in bytes unit.
  }
  if (xDType_ == ge::DT_BF16 && (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4) && !isA16W8Msd_) {
    OPS_ERR_IF(ubBaseK == 0 || ubBaseN == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "ubBaseK or ubBaseN cannot be 0"),
               return ge::GRAPH_FAILED);
  }
  tilingData.gmmBaseParams.set_ubCalSize(ubCalSize);
  tilingData.gmmBaseParams.set_ubRestBytes(ubRestBytes);  // in byte unit
  tilingData.gmmBaseParams.set_ubBaseK(ubBaseK);
  tilingData.gmmBaseParams.set_ubBaseN(ubBaseN);
  return ge::GRAPH_SUCCESS;
}

int64_t GMMTiling::GMMGetBS(const gert::Shape xShape) const {
    int64_t bs = 0;  // init bs
    if (transposeX_) {
      bs = xShape.GetDim(1);  // x shape is [k, m] if x is transpose_
    } else {
      if (groupType_ == -1) {  // -1: no group case, may exits a situation that multi dims product equals to bs.
        bs = xShape.GetDim(0);  // 0: x first dim
        size_t bsDimNum = xDimNum_ >= 1 ? xDimNum_ - 1 : 0;  // 1: x last dim k, the other dimensions are bs
        for (size_t i = 1; i < bsDimNum; i++) {
            bs *= xShape.GetDim(i);
        }
      } else {
        bs = xShape.GetDim(0);  // in group case，x's shapeis [m,k], 0 is the m axis.
      }
    }
    return bs;
}

void GMMTiling::GMMSetTilingKey(gert::TilingContext* context) const {
    bool transposeXSupportDtype = (weightDtype_ == ge::DT_FLOAT16 || weightDtype_ == ge::DT_BF16 ||
                                   weightDtype_ == ge::DT_FLOAT);
    if (isA16W8Msd_) {
      context->SetScheduleMode(1);  // set as batchmod for template using SyncAll
      context->SetTilingKey(transposeWeight_ ? TILING_KEY_A16W8_MSD_TRANS_W : TILING_KEY_A16W8_MSD);
      return;
    }
    if (xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT8 && actType_ == ACT_TYPE_GELU) {
      if (transposeWeight_) {
        context->SetTilingKey(TILING_KEY_QUANT_2VECTOR_TRANS_W);
      } else {
        context->SetTilingKey(TILING_KEY_QUANT_2VECTOR);
      }
      return;
    }
    if (transposeWeight_) {
      context->SetTilingKey(TILING_KEY_TRANS_W);
    } else if (transposeX_ && transposeXSupportDtype) {
      context->SetTilingKey(TILING_KEY_TRANS_X);
    } else if (antiquantPerformance_) {
      context->SetTilingKey(TILING_KEY_ANTIQUANT_PERFORMANCE);
      context->SetScheduleMode(1);  // set as batchmod for template using SyncAll
    } else {
      context->SetTilingKey(TILING_KEY);
    }
}

ge::graphStatus GMMTiling::GMMGetAttrs(const gert::TilingContext* context) {
  auto attr = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attr, return ge::GRAPH_FAILED);  // check attr is not null
  const bool* transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
  const bool* transposeXPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_TRANS_X);
  const int32_t* groupTypePtr = attr->GetAttrPointer<int32_t>(ATTR_INDEX_GROUPTYPE);
  const int64_t* splitItemPtr = attr->GetAttrPointer<int64_t>(ATTR_INDEX_SPLIT_ITEM);
  const int64_t* actTypePtr = attr->GetAttrPointer<int64_t>(ATTR_INDEX_ACT_TYPE);
  const uint32_t* groupListTypePtr = attr->GetAttrPointer<uint32_t>(ATTR_INDEX_GROUP_LIST_TYPE);
  transposeWeight_ = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
  transposeX_ = transposeXPtr != nullptr ? *transposeXPtr : false;
  groupType_ = groupTypePtr != nullptr ? *groupTypePtr : NO_SPLIT;
  splitItem_ = splitItemPtr != nullptr ? *splitItemPtr : 0;  // 0: 默认split_item
  actType_ = actTypePtr != nullptr ? *actTypePtr : 0;
  groupListType_ = groupListTypePtr != nullptr ? *groupListTypePtr : 0;

  auto xDesc = context->GetDynamicInputDesc(X_INDEX, 0);
  OPS_LOG_E_IF_NULL(context, xDesc, return ge::GRAPH_FAILED);  // check xDesc is not null
  xDType_ = xDesc->GetDataType();
  auto w0Desc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
  OPS_LOG_E_IF_NULL(context, w0Desc, return ge::GRAPH_FAILED);
  weightDtype_ = w0Desc->GetDataType();
  auto perTokenScalePtr = context->GetOptionalInputTensor(PER_TOKEN_SCALE_INDEX);
  if (perTokenScalePtr != nullptr && perTokenScalePtr->GetStorageShape().GetShapeSize() != 0) {
    perTokenOrPerGroupSize_ = 1;
  }
  tilingData.gmmBaseParams.set_quantParam(perTokenOrPerGroupSize_);
  auto yDesc = context->GetOutputDesc(Y_INDEX);
  OPS_LOG_E_IF_NULL(context, yDesc, return ge::GRAPH_FAILED);
  yDtype_ = yDesc->GetDataType();
  if (weightDtype_ == ge::DT_INT8 && xDType_ == ge::DT_INT8 && yDtype_ != ge::DT_INT32) {
    auto scale0Desc = context->GetDynamicInputDesc(SCALE_INDEX, 0);
    OPS_LOG_E_IF_NULL(context, scale0Desc, return ge::GRAPH_FAILED);
    scaleDtype_ = scale0Desc->GetDataType();
  }
  auto wFormat0 = static_cast<ge::Format>(ge::GetPrimaryFormat(w0Desc->GetStorageFormat()));
  wFormat_ = wFormat0 == ge::FORMAT_FRACTAL_NZ ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GMMSetUbDivideBlkAntiquant() {
  if (isA16W8Msd_) {
    ubDivideBlkNum_ = UB_A16W8_MSD_BLOCK_NUM;
    ubIoBlkNum_ = UB_A16W8_MSD_IO_USED_BLOCK;
    ubBlockAlign_ = UB_A16W8_MSD_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  if (xDType_ == ge::DT_FLOAT16 && (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4)) {
    if (weightDtype_ == ge::DT_INT8) {
      ubDivideBlkNum_ = UB_A16W8_BLOCK_NUM_FP16;
      ubIoBlkNum_ = UB_A16W8_IO_USED_BLOCK_FP16;
    } else {  // int4
      ubDivideBlkNum_ = UB_A16W4_BLOCK_NUM_FP16;
      ubIoBlkNum_ = UB_A16W4_IO_USED_BLOCK_FP16;
    }
    ubBlockAlign_ = UB_ANTIQUANT_PER_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  if (xDType_ == ge::DT_BF16 && (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4)) {
    if (weightDtype_ == ge::DT_INT8) {
      ubDivideBlkNum_ = UB_A16W8_BLOCK_NUM_BF16;
      ubIoBlkNum_ = UB_A16W8_IO_USED_BLOCK_BF16;
    } else {
      ubDivideBlkNum_ = UB_A16W4_BLOCK_NUM_BF16;
      ubIoBlkNum_ = UB_A16W4_IO_USED_BLOCK_BF16;
    }
    ubBlockAlign_ = UB_ANTIQUANT_PER_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

ge::graphStatus GMMTiling::GMMSetUbDivideBlkQuant() {
  if (weightDtype_ == ge::DT_INT8 && (perTokenOrPerGroupSize_ == 1 || actType_ != 0)) {
    // include case per-token without activation, per-token with activation and per-tensor with activation
    ubDivideBlkNum_ = UB_DYNAMIC_QUANT_BLOCK_NUM;
    ubIoBlkNum_ = UB_DUNAMIC_QUANT_IO_USED_BLOCK;
    ubBlockAlign_ = UB_QUANT_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  if (weightDtype_ == ge::DT_INT8 && perTokenOrPerGroupSize_ != 1) {
    // include case per-tensor without activation
    if (yDtype_ == ge::DT_FLOAT16) {
      ubDivideBlkNum_ = UB_STATIC_QUANT_BLOCK_NUM_FP16;
    } else {
      ubDivideBlkNum_ = UB_STATIC_QUANT_BLOCK_NUM_BF16;
    }
    ubIoBlkNum_ = UB_STATIC_QUANT_IO_USED_BLOCK;
    ubBlockAlign_ = UB_QUANT_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

ge::graphStatus GMMTiling::GMMSetUbDivideBlk() {
  ubDivideBlkNum_ = 0;  // init ubDivideBlkNum_
  ubIoBlkNum_ = 0;  // init ubIoBlkNum_
  ubBlockAlign_ = 0;  // init ubBlockAlign_
  if (xDType_ == ge::DT_INT8) {
    return GMMSetUbDivideBlkQuant();
  } else {
    return GMMSetUbDivideBlkAntiquant();
  }
  return ge::GRAPH_FAILED;
}

ge::graphStatus GMMTiling::SetBias(const gert::TilingContext* context, matmul_tiling::MultiCoreMatmulTiling& mm) const {
  if (!hasBias_ || isA16W8Msd_) {
    mm.SetBias(false);
  } else {
    mm.SetBias(true);
    auto biasTensor = context->GetDynamicInputTensor(BIAS_INDEX, 0);
    OPS_ERR_IF(biasTensor == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Get bias tensor failed."),
               return ge::GRAPH_FAILED);
    mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                   static_cast<matmul_tiling::DataType>(biasTensor->GetDataType()));
  }
  return ge::GRAPH_SUCCESS;
}

static void InitPlatformInfo(const GMMCompileInfo* compileInfoPtr, matmul_tiling::PlatformInfo& platformInfo) {
  platformInfo.socVersion = compileInfoPtr->socVersion;
  platformInfo.l1Size = compileInfoPtr->l1Size;
  platformInfo.l0CSize = compileInfoPtr->l0CSize;
  platformInfo.ubSize = compileInfoPtr->ubSize;
  platformInfo.l0ASize = compileInfoPtr->l0ASize;
  platformInfo.l0BSize = compileInfoPtr->l0BSize;
}

void GMMTiling::FullLoadK(const GMMCompileInfo* compileInfoPtr) {
  if (wFormat_ == matmul_tiling::CubeFormat::ND && !transposeWeight_ && !transposeX_ && xDType_ == weightDtype_ &&
      (weightDtype_ == ge::DT_FLOAT16 || weightDtype_ == ge::DT_BF16) && 
      maxM_ >= FULL_K_M_E_THRESHOLD * groupNum_ && maxN_ == FULL_K_N_THRESHOLD &&
      maxK_ <= FULL_K_MAX_K_THRESHOLD &&
      maxK_ >= FULL_K_MIN_K_THRESHOLD) {
    int64_t fullLoadStepKa = CeilDiv(maxK_ , baseK_);
    int64_t fullLoadStepKb = fullLoadStepKa / QUEUE_DOUBLE_BUFFER;
    int64_t fullLoadDepthKa = fullLoadStepKa * QUEUE_DOUBLE_BUFFER;
    int64_t fullLoadDepthKb = fullLoadStepKb * QUEUE_DOUBLE_BUFFER;    
    bool ifFullLoad = (maxM_ > FULL_K_M_THRESHOLD) && isAllSingleTensor_ && groupType_ == SPLIT_M &&
                      (((baseM_ * baseK_ * mmDataTypeSize_) * fullLoadDepthKa + 
                        (baseN_ * baseK_ * mmDataTypeSize_) * fullLoadDepthKb) <= static_cast<int64_t>(compileInfoPtr->l1Size));
    if (ifFullLoad) {
      tilingData.mmTilingData.set_stepKa(fullLoadStepKa);  // set precomputed mmStepKa
      tilingData.mmTilingData.set_depthA1(fullLoadDepthKa);  // set precomputed mmDepthA1
      tilingData.mmTilingData.set_stepKb(fullLoadStepKb);  // set precomputed mmStepKb
      tilingData.mmTilingData.set_depthB1(fullLoadDepthKb);  // set precomputed mmDepthB1
      tilingData.mmTilingData.set_iterateOrder(1);  // set precomputed stepN
      tilingData.gmmBaseParams.set_singleN(FULL_K_SINGLE_N);  // 0 is the default value
    }
  }
}

ge::graphStatus GMMTiling::CalcStepKaKb(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr,
                                        int64_t mInMM, uint32_t& mmStepKa, uint32_t& mmStepKb) {
  uint64_t availableL1Size = compileInfoPtr->l1Size;
  if (xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT8) {
    availableL1Size -= baseN_ * sizeof(uint64_t);
  }
  if (hasBias_) {
    availableL1Size -= baseN_ * 4;  // 4: size of float32 or int32
  }
  if (compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
    availableL1Size = BEST_L1_PARTA + BEST_L1_PARTB;
  }
  OPS_ERR_IF(availableL1Size < L1_PARTA_SIZE,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "availableL1Size is less than 256k"),
             return ge::GRAPH_FAILED);
  // according to double buffer, recompute the params used for data movement from GM to L1
  uint64_t l1ASize = baseM_ > baseN_ ? L1_PARTA_SIZE : availableL1Size - L1_PARTA_SIZE; 
  uint64_t l1BSize = availableL1Size - l1ASize;
  // 2: double buffer
  mmStepKa = (l1ASize / 2) / (baseM_ * baseK_ * mmDataTypeSize_);
  if (compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND310P &&
                                    wFormat_ == matmul_tiling::CubeFormat::NZ && mInMM <= baseM_) {
    mmStepKa = std::min<uint32_t>(mmStepKa, std::max(1, 128 / baseK_));  // 128: nz inner block size. In practice, baseK_*mmStepKa=128 makes performance better.
  }
  // 2: double buffer
  mmStepKb = (l1BSize / 2) / (baseN_ * baseK_ * mmDataTypeSize_);
  OPS_ERR_IF(mmStepKa == 0 || mmStepKb == 0,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "stepka or stepkb cannot be 0"),
             return ge::GRAPH_FAILED);

  if (mmStepKa > mmStepKb) {
    mmStepKa = mmStepKa / mmStepKb * mmStepKb;
  } else if (mmStepKa < mmStepKb) {
    mmStepKb = mmStepKb / mmStepKa * mmStepKa;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GMMSetMMTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr) {
  matmul_tiling::DataType matmulDtype = static_cast<matmul_tiling::DataType>(mmDType_);
  matmul_tiling::PlatformInfo platformInfo;
  InitPlatformInfo(compileInfoPtr, platformInfo);
  matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
  int64_t mInMM = isA16W8Msd_ ? A16W8_MSD_STEP * maxM_ : maxM_;  // if msd, m in matmul should mul steps
  mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
  mm.SetBType(matmul_tiling::TPosition::GM, wFormat_, matmulDtype, false);
  mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN, matmul_tiling::DataType::DT_FLOAT16);
  OPS_ERR_IF(SetBias(context, mm) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SetBias failed."), return ge::GRAPH_FAILED);
  mm.SetOrgShape(mInMM, maxN_, maxK_);
  mm.SetShape(mInMM, baseN_, maxK_);
  mm.SetFixSplit(baseM_, baseN_, baseK_);
  mm.SetBufferSpace(compileInfoPtr->l1Size, compileInfoPtr->l0CSize, ubSize_);
  OPS_ERR_IF(mm.GetTiling(tilingData.mmTilingData) == -1,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "matmul getTiling failed."),
             return ge::GRAPH_FAILED);
  uint32_t mmStepKa = 1;
  uint32_t mmStepKb = 1;
  OPS_ERR_IF(CalcStepKaKb(context, compileInfoPtr, mInMM, mmStepKa, mmStepKb) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "matmul calc stepka or stepkb failed."),
             return ge::GRAPH_FAILED);

  constexpr uint32_t stepM = 1;  // 1: stepM set fixed value 1
  constexpr uint32_t stepN = 1;  // 1: stepN set fixed value 1
  uint32_t mmDepthA1 = mmStepKa * DOUBLE_BUFFER_STEPKA_STEPKB * stepM;
  uint32_t mmDepthB1 = mmStepKb * DOUBLE_BUFFER_STEPKA_STEPKB * stepN;
  tilingData.mmTilingData.set_shareMode(0);
  if (compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
    tilingData.mmTilingData.set_shareUbSize(0);
    tilingData.mmTilingData.set_transLength(131072);  // 131072: 128KB size
  }
  tilingData.mmTilingData.set_dbL0C(1);  // disable double buffer for LOC
  tilingData.mmTilingData.set_baseM(baseM_);  // set precomputed baseM
  tilingData.mmTilingData.set_baseN(baseN_);  // set precomputed baseN
  tilingData.mmTilingData.set_baseK(baseK_);  // set precomputed baseK
  tilingData.mmTilingData.set_stepKa(mmStepKa);  // set precomputed mmStepKa
  tilingData.mmTilingData.set_depthA1(mmDepthA1);  // set precomputed mmDepthA1
  tilingData.mmTilingData.set_stepKb(mmStepKb);  // set precomputed mmStepKb
  tilingData.mmTilingData.set_depthB1(mmDepthB1);  // set precomputed mmDepthB1
  tilingData.mmTilingData.set_stepM(stepM);  // set precomputed stepM
  tilingData.mmTilingData.set_stepN(stepN);  // set precomputed stepN
  OPS_LOG_I(context->GetNodeName(), "GMM_tiling: baseM is %d, baseK is %d, baseN is %d.", baseM_, baseK_, baseN_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::CalMMTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr) {
  // 2048: min n for a16w8 msd to set baseN 512
  if (isA16W8Msd_ && maxN_ >= 2048 && !transposeWeight_) {
    baseN_ = BEST_BASEN_MSD;
  } else if (xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT8 && groupNum_ == 1) {
    baseN_ = BEST_BASEN_QUANT_ONE_GROUP;
    baseM_ = BEST_BASEM_QUANT_ONE_GROUP;
    baseK_ = BEST_BASEK_QUANT_ONE_GROUP;
    baseM_ = baseM_ > maxM_ ? SixteenAlign(maxM_, true) : baseM_;
    return ge::GRAPH_SUCCESS;
  } else {
    baseN_ = BEST_BASEN;
  }
  // according to the double buffer enabled L0B, compute baseK
  baseK_ = (compileInfoPtr->l0BSize / DOUBLE_BUFFER_L0A_L0B) / (baseN_ * mmDataTypeSize_);
  baseK_ = SixteenAlign(baseK_);
  OPS_ERR_IF(baseK_ == 0,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "baseK_ cannot be 0."),
             return ge::GRAPH_FAILED);
  // according to the double buffer enabled L0A/L0C, compute baseM(cube)
  uint32_t maxBaseM = compileInfoPtr->l0CSize / (baseN_ * FP32_DATATYPE_SIZE);
  baseM_ = std::min<uint32_t>((compileInfoPtr->l0ASize / DOUBLE_BUFFER_L0A_L0B) / (baseK_ * mmDataTypeSize_),
                              maxBaseM);

  if (!isA16W8Msd_) {
    baseM_ = baseM_ > maxM_ ? SixteenAlign(maxM_, true) : SixteenAlign(baseM_);
  } else {
    baseM_ = baseM_ > A16W8_MSD_STEP * maxM_ ? SixteenAlign(A16W8_MSD_STEP * maxM_, true) : SixteenAlign(baseM_);
  }
  if (baseM_ > MAX_BASEM) {
    baseM_ = MAX_BASEM;
  }
  OPS_ERR_IF(baseM_ == 0,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "baseM_ cannot be 0."),
             return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingGMM(gert::TilingContext* context) {
  auto xDesc = context->GetDynamicInputDesc(X_INDEX, 0);
  OPS_LOG_E_IF_NULL(context, xDesc, return ge::GRAPH_FAILED);  // check xDesc is not null
  ge::DataType xDType_ = xDesc->GetDataType();
  auto w0Desc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
  OPS_LOG_E_IF_NULL(context, w0Desc, return ge::GRAPH_FAILED);
  ge::DataType weightDtype_ = w0Desc->GetDataType();
  if(xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4) {     // A8W4 MSD
      GMMTilingData tilingData;
      constexpr uint32_t cvParallNum = 4; // for cv collaboration
      constexpr uint32_t THIRTY_TWO = 32;
      constexpr uint32_t UBCALSIZE = 32 * 256;  // for vector compute
      constexpr uint32_t UBRESTBYTES = 9 * 32 * 256;  // for vector compute
      constexpr uint32_t TWO = 2;
      constexpr uint32_t EIGHT = 8;
      uint32_t singleN = 256;
      uint32_t singleM = 128;
      auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
      OPS_LOG_E_IF_NULL(context, compileInfoPtr, return ge::GRAPH_FAILED);  // check compileInfoPtr is not null
      const uint32_t& aicNum = compileInfoPtr->aicNum;  // get aicNum from compileInfo
      if (aicNum == 0) {  // invaild value
        return ge::GRAPH_FAILED;
      }

      uint32_t usedCoreNum_ = aicNum;
      uint32_t n = context->GetDynamicInputTensor(SCALE_INDEX, 0)->GetStorageShape().GetDim(TWO);
      uint32_t k = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(1);
      uint32_t m = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(0);
      tilingData.gmmBaseParams.set_coreNum(usedCoreNum_);
      tilingData.gmmBaseParams.set_groupNum(context->GetDynamicInputTensor(WEIGHT_INDEX, 0)->GetStorageShape().GetDim(0));
      tilingData.gmmBaseParams.set_totalInGroup(m);
      tilingData.gmmBaseParams.set_k(k);
      tilingData.gmmBaseParams.set_n(n);
      tilingData.gmmBaseParams.set_vBaseM(THIRTY_TWO);
      tilingData.gmmBaseParams.set_ubCalSize(UBCALSIZE);
      tilingData.gmmBaseParams.set_ubRestBytes(UBRESTBYTES);
      tilingData.gmmBaseParams.set_parallNum(cvParallNum);
      tilingData.gmmBaseParams.set_quantGroupNum(context->GetDynamicInputTensor(SCALE_INDEX, 0)->GetStorageShape().GetDim(1));
      tilingData.gmmBaseParams.set_m(context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(0));
      context->SetBlockDim(aicNum);

      //GEMM Tiling
      constexpr uint32_t A8W4_MSD_BASE_M = 64;
      constexpr uint32_t A8W4_MSD_BASE_K = 256;
      constexpr uint32_t A8W4_MSD_BASE_N = 512;
      matmul_tiling::PlatformInfo platformInfo;
      InitPlatformInfo(compileInfoPtr, platformInfo);
      matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
      mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
      mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT4, false);
      mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
      mm.SetBias(false);
      mm.SetOrgShape(A8W4_MSD_BASE_M, n, k);
      mm.SetShape(A8W4_MSD_BASE_M, A8W4_MSD_BASE_N, k);
      mm.SetFixSplit(A8W4_MSD_BASE_M, A8W4_MSD_BASE_N, A8W4_MSD_BASE_K);
      if (mm.GetTiling(tilingData.mmTilingData) == -1){
        std::cout << "GetTiling FAIED!!\n";
        return ge::GRAPH_FAILED;
      }

      constexpr uint32_t FOUR = 4;
      tilingData.mmTilingData.set_dbL0C(1);  // disable double buffer for LOC
      tilingData.mmTilingData.set_stepKa(FOUR);  // set precomputed mmStepKa
      tilingData.mmTilingData.set_stepKb(FOUR);  // set precomputed mmStepKb
      tilingData.mmTilingData.set_depthA1(EIGHT);  // set precomputed mmDepthA1
      tilingData.mmTilingData.set_depthB1(EIGHT);  // set precomputed mmDepthB1
      tilingData.mmTilingData.set_stepM(1);  // set precomputed stepM
      tilingData.mmTilingData.set_stepN(1);  // set precomputed stepN
      OPS_LOG_I(context->GetNodeName(), "GMM_tiling: baseM is %u, baseK is %u, baseN is %u.", A8W4_MSD_BASE_M, A8W4_MSD_BASE_K, A8W4_MSD_BASE_N);

      context->SetScheduleMode(1);  // set as batchmod for template using SyncAll
      context->SetTilingKey(TILING_KEY_A8W4_MSD);

      tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
      context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

      size_t* workspaces = context->GetWorkspaceSizes(1);  // get second variable
      OPS_LOG_E_IF_NULL(context, workspaces, return ge::GRAPH_FAILED);  // check workspaces is not null
      workspaces[0] = SYS_WORKSPACE_SIZE;  // default size
      workspaces[0] += (cvParallNum * aicNum * singleN * singleM * sizeof(int32_t) * EIGHT);
      return ge::GRAPH_SUCCESS;
  }

  GMMTiling tiling;
  OPS_ERR_IF(tiling.Init(context) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM tiling init failed"),
             return ge::GRAPH_FAILED);
  return tiling.RunFusionKernelTiling(context);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGMM(gert::TilingParseContext* context) {
  fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
  OPS_LOG_E_IF_NULL(context, platformInfoPtr, return ge::GRAPH_FAILED);
  auto compileInfoPtr = context->GetCompiledInfo<GMMCompileInfo>();
  OPS_LOG_E_IF_NULL(context, compileInfoPtr, return ge::GRAPH_FAILED);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
  compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
  compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

  OPS_ERR_IF((compileInfoPtr->aicNum == 0 || compileInfoPtr->aivNum == 0 || compileInfoPtr->ubSize == 0 || \
             compileInfoPtr->l1Size == 0 || compileInfoPtr->l0CSize == 0 || compileInfoPtr->l0ASize == 0 || \
             compileInfoPtr->l0BSize == 0),
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "platform info is invalid, aicNum=%u, aivNum=%u, ubSize=%lu, l1Size=%lu, l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
             compileInfoPtr->aicNum, compileInfoPtr->aivNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size,
             compileInfoPtr->l0CSize, compileInfoPtr->l0ASize, compileInfoPtr->l0BSize),
             return ge::GRAPH_FAILED);

  OPS_LOG_I(context->GetNodeName(), "Parse compile info success, soc: %d",
            static_cast<int>(compileInfoPtr->socVersion));
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmul)
.Tiling(TilingGMM)
.TilingParse<GMMCompileInfo>(TilingPrepareForGMM);  // regist into the framework
}  // namespace optiling
