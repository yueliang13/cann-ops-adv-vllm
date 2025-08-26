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
 * \file grouped_matmul_add_tiling.cpp
 * \brief
 */

#include "grouped_matmul_add_tiling.h"
#include <climits>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "log/ops_log.h"
#include "tiling/tiling_templates_registry.h"

namespace optiling {
constexpr uint64_t BEST_L1_PARTA = 256 * 1024;
constexpr uint64_t BEST_L1_PARTB = 128 * 1024;
constexpr int32_t BEST_BASEN = 256;
constexpr int32_t MAX_BASEM = 256;
constexpr uint32_t DATATYPE_SIZE = 2;
constexpr uint32_t FP32_DATATYPE_SIZE = 4;
constexpr uint64_t TILING_KEY_DEFAULT_FP16 = 0;
constexpr uint64_t TILING_KEY_DEFAULT_BF16 = 1;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr int32_t MAX_TENSOR_CONT = 128;
constexpr int64_t INDEX_GROUP_LIST = 2;
constexpr size_t INDEX_IN_X = 0;
constexpr size_t INDEX_IN_W = 1;
constexpr size_t INDEX_IN_GROUP_LIST = 2;
constexpr int32_t DIM0 = 0;
constexpr int32_t DIM1 = 1;
constexpr int32_t DIM2 = 2;
constexpr int32_t DIMNUM_1D = 1;
constexpr int32_t DIMNUM_2D = 2;
constexpr int32_t ATTR_TRANSPOSE_X_INDEX = 0;
constexpr int32_t ATTR_TRANSPOSE_W_INDEX = 1;
constexpr int32_t ATTR_GROUP_TYPE_INDEX = 2;
constexpr int32_t AIC_AIV_RATION = 2;

static inline uint32_t SixteenAlign(uint32_t a, bool up = false) {
  if (up) {
    a += 15;  // 15: 16 bytes up-align
  }
  return a & ~15;  // ~15: 16 bytes down-align
};

static ge::graphStatus CalTCubeTiling(gert::TilingContext* context, GroupedMatmulAddTilingData& tiling, int32_t m,
                                   int32_t k, int32_t n, int32_t baseM, int32_t baseN, int32_t baseK) {
  auto xType = context->GetInputDesc(INDEX_IN_X)->GetDataType();
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  matmul_tiling::DataType matmulDtype = static_cast<matmul_tiling::DataType>(xType);
  matmul_tiling::MultiCoreMatmulTiling mm(ascendcPlatform);
  mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
  mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
  mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
  mm.SetBias(false);
  mm.SetOrgShape(m, n, k);
  mm.SetShape(m, baseN, k);
  mm.SetFixSplit(baseM, baseN, baseK);

  uint64_t l1Size, l0_cSize, ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0_cSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

  mm.SetBufferSpace(l1Size, l0_cSize, ubSize);
  if (mm.GetTiling(tiling.mmTilingData) == -1) {
    OPS_LOG_E(context, "GroupedMatmulAdd GetTiling error");
    return ge::GRAPH_FAILED;
  }
  uint64_t productMK = static_cast<uint64_t>(baseM) * static_cast<uint64_t>(baseK);
  uint64_t productNK = static_cast<uint64_t>(baseN) * static_cast<uint64_t>(baseK);
  if (productMK > UINT32_MAX || productNK > UINT32_MAX) {
    OPS_LOG_E(context, "productMK or productNK > uint32_t max value, productMK:%lu, productNK:%lu", productNK, productMK);
    return ge::GRAPH_FAILED;
  }
  uint32_t mmStepKa = (BEST_L1_PARTB >> 1) / (static_cast<uint32_t>(productMK) * DATATYPE_SIZE);
  uint32_t mmStepKb = (BEST_L1_PARTA >> 1) / (static_cast<uint32_t>(productNK) * DATATYPE_SIZE);
  if (mmStepKa > mmStepKb) {
    mmStepKa = mmStepKa / mmStepKb * mmStepKb;
  } else if (mmStepKa < mmStepKb) {
    mmStepKb = mmStepKb / mmStepKa * mmStepKa;
  }
  constexpr uint32_t stepM = 1;  // 1: stepM set fixed value 1
  constexpr uint32_t stepN = 1;  // 1: stepN set fixed value 1
  uint32_t mmDepthA1 = mmStepKa * DOUBLE_BUFFER_STEPKA_STEPKB * stepM;
  uint32_t mmDepthB1 = mmStepKb * DOUBLE_BUFFER_STEPKA_STEPKB * stepN;

  tiling.mmTilingData.set_stepKa(mmStepKa);    // set precomputed mmStepKa
  tiling.mmTilingData.set_depthA1(mmDepthA1);  // set precomputed mmDepthA1
  tiling.mmTilingData.set_stepKb(mmStepKb);    // set precomputed mmStepKb
  tiling.mmTilingData.set_depthB1(mmDepthB1);  // set precomputed mmDepthB1
  tiling.mmTilingData.set_stepM(stepM);        // set precomputed stepM
  tiling.mmTilingData.set_stepN(stepN);        // set precomputed stepN
  return ge::GRAPH_SUCCESS; 
}

static ge::graphStatus CalMmTiling(gert::TilingContext* context, GroupedMatmulAddTilingData& tiling, int32_t m,
                                   int32_t k, int32_t n) {
  int32_t baseN = BEST_BASEN;
  int32_t baseK, baseM, maxM;
  uint64_t l0_A_size, l0_B_size, l0_C_size;

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  // 先根据 baseN 和 L0_B的大小确定baseK
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0_B_size);
  baseK = static_cast<int32_t>((l0_B_size / DOUBLE_BUFFER_L0A_L0B) /
                               static_cast<uint64_t>(baseN * static_cast<int32_t>(DATATYPE_SIZE)));
  baseK = static_cast<int32_t>(SixteenAlign(baseK));
  // L0_C大小会限制 BaseM
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0_C_size);
  uint32_t maxBaseM = static_cast<uint32_t>(l0_C_size / static_cast<uint64_t>(baseN * FP32_DATATYPE_SIZE));
  // L0_A大小会限制 BaseM
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0_A_size);
  baseM = std::min<uint32_t>((l0_A_size / DOUBLE_BUFFER_L0A_L0B) / (baseK * DATATYPE_SIZE), maxBaseM);
  auto xShape = context->GetInputShape(0)->GetOriginShape();
  maxM = xShape.GetDim(xShape.GetDimNum() - 1);
  if (baseM > maxM) {
    baseM = static_cast<int32_t>(SixteenAlign(maxM, true));
  } else {
    baseM = static_cast<int32_t>(SixteenAlign(baseM));
  }
  if (baseM > MAX_BASEM) {
    baseM = MAX_BASEM;
  }

  // 设置矩阵相关tiling参数
  auto xType = context->GetInputDesc(INDEX_IN_X)->GetDataType();
  if (xType == ge::DataType::DT_FLOAT16) {
    context->SetTilingKey(TILING_KEY_DEFAULT_FP16);
  } else if(xType == ge::DataType::DT_BF16) {
    context->SetTilingKey(TILING_KEY_DEFAULT_BF16);
  } else {
    OPS_LOG_E(context, "GroupedMatmulAdd GetTiling error : type not support");
  }
  auto ret = CalTCubeTiling(context, tiling, m, k, n, baseM, baseN, baseK);
  OPS_CHECK(ge::GRAPH_SUCCESS != ret,
                OPS_LOG_E(context, "GroupedMatmulAdd CalTCubeTiling error"),
                return ret);
  tiling.mmTilingData.set_shareMode(0);
  tiling.mmTilingData.set_baseM(baseM);        // set precomputed baseM
  tiling.mmTilingData.set_baseN(baseN);        // set precomputed baseN
  tiling.mmTilingData.set_baseK(baseK);        // set precomputed baseK

  return ge::GRAPH_SUCCESS;
}

static void PrintInfo(gert::TilingContext* context, GroupedMatmulAddTilingData &tiling) {
  OPS_LOG_D(context, ">>>>>>>>>>>>>>> Start to print GroupedMatmulAdd tiling data <<<<<<<<<<<<<<<<");
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [GmmBaseParams]: groupNum = %ld", tiling.gmmBaseParams.get_groupNum());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [GmmBaseParams]: coreNum = %ld", tiling.gmmBaseParams.get_coreNum());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [GmmBaseParams]: groupType = %ld", tiling.gmmBaseParams.get_groupType());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [GmmArray]: mList[0] = %ld", tiling.gmmArray.get_mList()[0]);
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [GmmArray]: kList[0] = %ld", tiling.gmmArray.get_kList()[0]);
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [GmmArray]: nList[0] = %ld", tiling.gmmArray.get_nList()[0]);
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: usedCoreNum = %d", tiling.mmTilingData.get_usedCoreNum());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: M = %d", tiling.mmTilingData.get_M());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: N = %d", tiling.mmTilingData.get_N());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: Ka = %d", tiling.mmTilingData.get_Ka());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: Kb = %d", tiling.mmTilingData.get_Kb());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: singleCoreM = %d", tiling.mmTilingData.get_singleCoreM());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: singleCoreN = %d", tiling.mmTilingData.get_singleCoreN());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: singleCoreK = %d", tiling.mmTilingData.get_singleCoreK());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: baseM = %d", tiling.mmTilingData.get_baseM());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: baseN = %d", tiling.mmTilingData.get_baseN());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: baseK = %d", tiling.mmTilingData.get_baseK());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: depthA1 = %d", tiling.mmTilingData.get_depthA1());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: depthB1 = %d", tiling.mmTilingData.get_depthB1());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: stepM = %d", tiling.mmTilingData.get_stepM());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: stepN = %d", tiling.mmTilingData.get_stepN());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: stepKa = %d", tiling.mmTilingData.get_stepKa());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: stepKb = %d", tiling.mmTilingData.get_stepKb());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: isBias = %d", tiling.mmTilingData.get_isBias());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: transLength = %d", tiling.mmTilingData.get_transLength());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: iterateOrder = %d", tiling.mmTilingData.get_iterateOrder());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: dbL0A = %d", tiling.mmTilingData.get_dbL0A());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: dbL0B = %d", tiling.mmTilingData.get_dbL0B());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TCubeTiling]: dbL0C = %d", tiling.mmTilingData.get_dbL0C());
  OPS_LOG_D(context, ">>> GroupedMatmulAdd [TilingKey]: %lu", context->GetTilingKey());
  OPS_LOG_D(context, ">>>>>>>>>>>>>>> Print GroupedMatmulAdd tiling data end <<<<<<<<<<<<<<<<");
}

ASCENDC_EXTERN_C ge::graphStatus TilingCheck4GroupedMatmulAdd(gert::TilingContext* context) {
  auto xShape = context->GetInputShape(INDEX_IN_X)->GetOriginShape();
  auto wShape = context->GetInputShape(INDEX_IN_W)->GetOriginShape();
  auto groupListShape = context->GetInputShape(INDEX_IN_GROUP_LIST)->GetOriginShape();
  //检查输入dim是否符合预期
  OPS_CHECK(DIMNUM_2D != xShape.GetDimNum(),
    OPS_LOG_E(context, "xShape error xShape.GetDimNum():%lu", xShape.GetDimNum()),
    return ge::GRAPH_FAILED);
  OPS_CHECK(DIMNUM_2D != wShape.GetDimNum(),
    OPS_LOG_E(context, "wShape error wShape.GetDimNum():%lu", wShape.GetDimNum()),
    return ge::GRAPH_FAILED);
  OPS_CHECK(DIMNUM_1D != groupListShape.GetDimNum(),
    OPS_LOG_E(context, "groupListShape error groupListShape.GetDimNum():%lu", groupListShape.GetDimNum()),
    return ge::GRAPH_FAILED);
  //检查shape是否可以对应
  auto xk = xShape.GetDim(DIM0);
  auto wk = wShape.GetDim(DIM0);
  OPS_CHECK(xk != wk,
    OPS_LOG_E(context, "shape Error xk:%ld wk:%ld", xk, wk),
    return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus Tiling4GroupedMatmulAdd(gert::TilingContext* context) {
  GroupedMatmulAddTilingData tiling;
  auto xShape = context->GetInputShape(0)->GetOriginShape();
  auto wShape = context->GetInputShape(1)->GetOriginShape();
  auto groupListShape = context->GetInputShape(2)->GetOriginShape();

  OPS_CHECK(ge::GRAPH_SUCCESS != TilingCheck4GroupedMatmulAdd(context),
    OPS_LOG_E(context, "GroupedMatmulAdd TilingCheck4GroupedMatmulAdd error"),
    return ge::GRAPH_FAILED);

  int64_t m, n, k;
  m = xShape.GetDim(DIM1);
  k = xShape.GetDim(DIM0);
  n = wShape.GetDim(DIM1);

  // 先设置最大核数
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const int64_t aicNum = ascendcPlatform.GetCoreNumAic();

  //设置gmmBaseParams
  tiling.gmmBaseParams.set_groupNum(groupListShape.GetDim(DIM0));
  tiling.gmmBaseParams.set_coreNum(aicNum * AIC_AIV_RATION);//AIC_AIV 1:2
  tiling.gmmBaseParams.set_groupType(INDEX_GROUP_LIST);

  //设置GmmArray
  int64_t kList[MAX_TENSOR_LIST_SIZE] = {0};
  int64_t mList[MAX_TENSOR_LIST_SIZE] = {0};
  int64_t nList[MAX_TENSOR_LIST_SIZE] = {0};
  kList[0] = -1;
  mList[0] = m;
  nList[0] = n;
  tiling.gmmArray.set_mList(mList);
  tiling.gmmArray.set_kList(kList);
  tiling.gmmArray.set_nList(nList);
  //设置其余
  auto ret = CalMmTiling(context, tiling, m, k, n);
  OPS_CHECK(ge::GRAPH_SUCCESS != ret,
                OPS_LOG_E(context, "GroupedMatmulAdd CalMmTiling error"),
                return ret);
  context->SetBlockDim(aicNum);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  auto baseM = tiling.mmTilingData.get_baseM();
  auto baseN = tiling.mmTilingData.get_baseN();
  uint32_t userWorkspaceSize = baseM * baseN * FP32_DATATYPE_SIZE * aicNum * AIC_AIV_RATION;
  size_t* workspaces = context->GetWorkspaceSizes(1);
  workspaces[0] = sysWorkspaceSize + userWorkspaceSize;

  PrintInfo(context, tiling);

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GroupedMatmulAdd(gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}

struct GroupedMatmulAddCompileInfo {};

IMPL_OP_OPTILING(GroupedMatmulAdd)
    .Tiling(Tiling4GroupedMatmulAdd)
    .TilingParse<GroupedMatmulAddCompileInfo>(TilingPrepare4GroupedMatmulAdd);
}  // namespace optiling