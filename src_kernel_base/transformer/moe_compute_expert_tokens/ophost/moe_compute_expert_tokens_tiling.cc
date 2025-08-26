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
 * \file moe_compute_expert_tokens_tiling.cc
 * \brief
 */
#include <cmath>
#include "moe_compute_expert_tokens_tiling.h"

using namespace std;
using namespace ge;
using namespace AscendC;

namespace optiling {

static const size_t SHAPE_SIZE = 1;
static const int64_t ONCE_HANDLE_COUNT = 320;
static const int64_t ONCE_RPG_MAX_COUNT = 255;
static const int64_t SYS_WORKSPACE = 16 * 1024 * 1024;

static const int64_t E_BOUND_NUM = 256;
static const int64_t BSK_BOUND_NUM = 12000;
static const int64_t NET_E_BOUND_NUM = 8;
static const int64_t NET_BSK_BOUND_NUM = 65536;
static const int64_t EXTRA_WORKSAPCE = 32;
static const int64_t E_BOUND_NUM_UP_LIMIT = 2048;
static const int64_t BSK_BOUND_NUM_UP_LIMIT = std::pow(2, 24);

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t valueNum = 0;
    if (factor == 0) {
        return value;
    }
    if (value % factor == 0) {
        valueNum = value / factor;
    } else {
        valueNum = value / factor + 1;
    }
    return valueNum;
}

inline static int64_t RoundUp(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

inline static int64_t CalcWorkLocal(int64_t handleNum)
{
    int64_t elementsPerBlock = 32 / sizeof(float);
    int64_t elementsPerRepeat = 256 / sizeof(float);

    int64_t firstMaxRepeat = RoundUp(handleNum, elementsPerRepeat);
    int64_t iter1OutputCount = firstMaxRepeat * 2;
    int64_t iter2AlignStart = RoundUp(iter1OutputCount, elementsPerBlock) * elementsPerBlock;
    int64_t iter2OutputCount = RoundUp(iter1OutputCount, elementsPerRepeat) * 2;
    int64_t iter3AlignStart = RoundUp(iter2OutputCount, elementsPerBlock) * elementsPerBlock;
    int64_t iter3OutputCount = RoundUp(iter2OutputCount, elementsPerRepeat) * 2;

    int64_t iter3AlignEnd = RoundUp(iter3OutputCount, elementsPerBlock) * elementsPerBlock;
    int64_t finalWorkLocalNeedSize = iter2AlignStart + iter3AlignStart + iter3AlignEnd;
    return finalWorkLocalNeedSize;
}

inline static ge::graphStatus MoeComputeExpertTokensSetTilingData(
    gert::TilingContext* context, MoeComputeExpertTokensTilingData& tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static bool CheckParamsShape(const gert::TilingContext* context)
{
  auto sortedExpertShapePtr = context->GetInputShape(0);
  OPS_LOG_E_IF_NULL(context, sortedExpertShapePtr, return false);
  auto sortedExpertShape = sortedExpertShapePtr->GetStorageShape();

  auto outShapePtr = context->GetOutputShape(0);
  OPS_LOG_E_IF_NULL(context, outShapePtr, return false);
  auto outShape = outShapePtr->GetStorageShape();

  auto sortedExpertTypePtr = context->GetInputDesc(0);
  OPS_LOG_E_IF_NULL(context, sortedExpertTypePtr, return false);
  auto sortedExpertType = sortedExpertTypePtr->GetDataType();
  int32_t dtypeSize = ge::GetSizeByDataType(sortedExpertType);

  auto attrs = context->GetAttrs();
  auto numOfExpert = *(attrs->GetAttrPointer<int32_t>(0));

  OPS_CHECK(sortedExpertShape.GetDimNum() != SHAPE_SIZE,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                                  "the sorted_expert of input should be 1D tensor."),
                  return false);

  OPS_CHECK(outShape.GetDimNum() != SHAPE_SIZE,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                                  "the output should be 1D tensor."),
                  return false);

  OPS_CHECK(sortedExpertType != ge::DT_INT32 || dtypeSize != 4,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                                  "the sorted_expert of input type should be INT32."),
                  return false);
 
  OPS_CHECK(numOfExpert == 0,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                                  "the numOfExpert of attr should not be 0."),
                  return false);

  return true;
}

static inline void PrintTilingData(MoeComputeExpertTokensTilingData &tilingData)
{
    OPS_LOG_I("[MoeComputeExpertTokens]", "[totalCoreNum]: %ld", tilingData.get_totalCoreNum());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[usedCoreNumBefore]: %ld", tilingData.get_usedCoreNumBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[usedCoreNumBefore3]: %ld", tilingData.get_usedCoreNumBefore3());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[usedCoreNumAfter]: %ld", tilingData.get_usedCoreNumAfter());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[ubSize]: %ld", tilingData.get_ubSize());   
    OPS_LOG_I("[MoeComputeExpertTokens]", "[workLocalNeedSize]: %ld", tilingData.get_workLocalNeedSize()); 
    OPS_LOG_I("[MoeComputeExpertTokens]", "[sortedExpertNum]: %ld", tilingData.get_sortedExpertNum());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[normalCoreHandleNumBefore]: %ld", tilingData.get_normalCoreHandleNumBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[normalCoreLoopNumBefore]: %ld", tilingData.get_normalCoreLoopNumBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[normalCoreHandleNumPerLoopBefore]: %ld", tilingData.get_normalCoreHandleNumPerLoopBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[normalCoreHandleNumTailLoopBefore]: %ld", tilingData.get_normalCoreHandleNumTailLoopBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[tailCoreHandleNumBefore]: %ld", tilingData.get_tailCoreHandleNumBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[tailCoreLoopNumBefore]: %ld", tilingData.get_tailCoreLoopNumBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[tailCoreHandleNumPerLoopBefore]: %ld", tilingData.get_tailCoreHandleNumPerLoopBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[tailCoreHandleNumTailLoopBefore]: %ld", tilingData.get_tailCoreHandleNumTailLoopBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[numOfExpert]: %ld", tilingData.get_numOfExpert());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[normalCoreHandleNumAfter]: %ld", tilingData.get_normalCoreHandleNumAfter());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[normalCoreLoopNumAfter]: %ld", tilingData.get_normalCoreLoopNumAfter());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[normalCoreHandleNumPerLoopAfter]: %ld", tilingData.get_normalCoreHandleNumPerLoopAfter());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[normalCoreHandleNumTailLoopAfter]: %ld", tilingData.get_normalCoreHandleNumTailLoopAfter());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[tailCoreHandleNumAfter]: %ld", tilingData.get_tailCoreHandleNumAfter());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[tailCoreLoopNumAfter]: %ld", tilingData.get_tailCoreLoopNumAfter());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[tailCoreHandleNumPerLoopAfter]: %ld", tilingData.get_tailCoreHandleNumPerLoopAfter());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[tailCoreHandleNumTailLoopAfter]: %ld", tilingData.get_tailCoreHandleNumTailLoopAfter());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[handleNumPerCoreBefore]: %ld", tilingData.get_handleNumPerCoreBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[handleNumTailCoreBefore]: %ld", tilingData.get_handleNumTailCoreBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[loopCountBefore]: %ld", tilingData.get_loopCountBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[loopCountTailBefore]: %ld", tilingData.get_loopCountTailBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[handleNumPerLoopBefore]: %ld", tilingData.get_handleNumPerLoopBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[handleNumTailCorePerLoopBefore]: %ld", tilingData.get_handleNumTailCorePerLoopBefore());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[handleExpertNumLoopCount]: %ld", tilingData.get_handleExpertNumLoopCount());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[handleExpertNumMainCorePerLoop]: %ld", tilingData.get_handleExpertNumMainCorePerLoop());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[handleExpertNumTailCorePerLoop]: %ld", tilingData.get_handleExpertNumTailCorePerLoop());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[handleNumTailCoreMainLoop]: %ld", tilingData.get_handleNumTailCoreMainLoop());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[loopCountTailCoreMainLoop]: %ld", tilingData.get_loopCountTailCoreMainLoop());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[handleNumTailCoreTailLoop]: %ld", tilingData.get_handleNumTailCoreTailLoop());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[loopCountTailCoreTailLoop]: %ld", tilingData.get_loopCountTailCoreTailLoop());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[userWorkspaceSize]: %ld", tilingData.get_userWorkspaceSize());
    OPS_LOG_I("[MoeComputeExpertTokens]", "[tilingKey]: %ld", tilingData.get_tilingKey());
}

static inline ge::graphStatus CalcNumOfExpertTiling(const gert::TilingContext* context, MoeComputeExpertTokensTilingData& tilingData)
{
    OPS_LOG_D(context->GetNodeName(), "TilingComputeExpertTokens Enter CalcNumOfExpertTiling.");
    
    int64_t normalCoreHandleNumAfter = (tilingData.get_numOfExpert() + tilingData.get_totalCoreNum() - 1) / tilingData.get_totalCoreNum();
    int64_t usedCoreNumAfter = min(RoundUp(tilingData.get_numOfExpert(), normalCoreHandleNumAfter), tilingData.get_totalCoreNum());
    int64_t tailCoreHandleNumAfter = tilingData.get_numOfExpert() - (usedCoreNumAfter - 1) * normalCoreHandleNumAfter;

    int64_t normalCoreHandleNumPerLoopAfter = (normalCoreHandleNumAfter > ONCE_RPG_MAX_COUNT) ? ONCE_RPG_MAX_COUNT : normalCoreHandleNumAfter;
    int64_t normalCoreLoopNumAfter =  CeilDiv(normalCoreHandleNumAfter, normalCoreHandleNumPerLoopAfter);
    int64_t normalCoreHandleNumTailLoopAfter = normalCoreHandleNumAfter - (normalCoreLoopNumAfter - 1) * normalCoreHandleNumPerLoopAfter;

    int64_t tailCoreHandleNumPerLoopAfter = (tailCoreHandleNumAfter > ONCE_RPG_MAX_COUNT) ? ONCE_RPG_MAX_COUNT : tailCoreHandleNumAfter;
    int64_t tailCoreLoopNumAfter =  CeilDiv(tailCoreHandleNumAfter, tailCoreHandleNumPerLoopAfter);
    int64_t tailCoreHandleNumTailLoopAfter = tailCoreHandleNumAfter  - (tailCoreLoopNumAfter - 1) * tailCoreHandleNumPerLoopAfter;

    tilingData.set_normalCoreHandleNumAfter(normalCoreHandleNumAfter);
    tilingData.set_normalCoreLoopNumAfter(normalCoreLoopNumAfter);
    tilingData.set_normalCoreHandleNumPerLoopAfter(normalCoreHandleNumPerLoopAfter);
    tilingData.set_normalCoreHandleNumTailLoopAfter(normalCoreHandleNumTailLoopAfter);

    tilingData.set_tailCoreHandleNumAfter(tailCoreHandleNumAfter);
    tilingData.set_tailCoreLoopNumAfter(tailCoreLoopNumAfter);
    tilingData.set_tailCoreHandleNumPerLoopAfter(tailCoreHandleNumPerLoopAfter);
    tilingData.set_tailCoreHandleNumTailLoopAfter(tailCoreHandleNumTailLoopAfter);
    tilingData.set_usedCoreNumAfter(usedCoreNumAfter);

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcSortedExpertTiling(const gert::TilingContext* context, MoeComputeExpertTokensTilingData& tilingData)
{
    OPS_LOG_D(context->GetNodeName(), "TilingComputeExpertTokens Enter CalcSortedExpertTiling.");

    int64_t normalCoreHandleNumBefore = (tilingData.get_sortedExpertNum() + tilingData.get_totalCoreNum() - 1) / tilingData.get_totalCoreNum();
    int64_t usedCoreNumBefore = min(RoundUp(tilingData.get_sortedExpertNum(), normalCoreHandleNumBefore), tilingData.get_totalCoreNum());
    int64_t tailCoreHandleNumBefore = tilingData.get_sortedExpertNum() - (usedCoreNumBefore - 1) * normalCoreHandleNumBefore;

    int64_t normalCoreHandleNumPerLoopBefore = (normalCoreHandleNumBefore > ONCE_HANDLE_COUNT) ? ONCE_HANDLE_COUNT : normalCoreHandleNumBefore;
    int64_t normalCoreLoopNumBefore =  CeilDiv(normalCoreHandleNumBefore, normalCoreHandleNumPerLoopBefore);
    int64_t normalCoreHandleNumTailLoopBefore = normalCoreHandleNumBefore - (normalCoreLoopNumBefore - 1) * normalCoreHandleNumPerLoopBefore;

    int64_t tailCoreHandleNumPerLoopBefore = (tailCoreHandleNumBefore > ONCE_HANDLE_COUNT) ? ONCE_HANDLE_COUNT : tailCoreHandleNumBefore;
    int64_t tailCoreLoopNumBefore =  CeilDiv(tailCoreHandleNumBefore, tailCoreHandleNumPerLoopBefore);
    int64_t tailCoreHandleNumTailLoopBefore = tailCoreHandleNumBefore  - (tailCoreLoopNumBefore - 1) * tailCoreHandleNumPerLoopBefore;

    tilingData.set_normalCoreHandleNumBefore(normalCoreHandleNumBefore);
    tilingData.set_normalCoreLoopNumBefore(normalCoreLoopNumBefore);
    tilingData.set_normalCoreHandleNumPerLoopBefore(normalCoreHandleNumPerLoopBefore);
    tilingData.set_normalCoreHandleNumTailLoopBefore(normalCoreHandleNumTailLoopBefore);

    tilingData.set_tailCoreHandleNumBefore(tailCoreHandleNumBefore);
    tilingData.set_tailCoreLoopNumBefore(tailCoreLoopNumBefore);
    tilingData.set_tailCoreHandleNumPerLoopBefore(tailCoreHandleNumPerLoopBefore);
    tilingData.set_tailCoreHandleNumTailLoopBefore(tailCoreHandleNumTailLoopBefore);
    tilingData.set_usedCoreNumBefore(usedCoreNumBefore);

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcTemplate3ParamTiling(const gert::TilingContext* context, MoeComputeExpertTokensTilingData& tilingData)
{
    OPS_LOG_D(context->GetNodeName(), "TilingComputeExpertTokens Enter CalcTemplate3ParamTiling.");

    int64_t handleNumPerCoreBefore = CeilDiv(CeilDiv(tilingData.get_sortedExpertNum(), tilingData.get_totalCoreNum()), ONCE_HANDLE_COUNT) * ONCE_HANDLE_COUNT;
    int64_t loopCountBefore = CeilDiv(handleNumPerCoreBefore, ONCE_HANDLE_COUNT);
    int64_t handleNumPerLoopBefore = ONCE_HANDLE_COUNT; 
    int64_t usedCoreNumBefore3 = min(RoundUp(tilingData.get_sortedExpertNum(), handleNumPerCoreBefore), tilingData.get_totalCoreNum());
    int64_t handleNumTailCoreBefore = tilingData.get_sortedExpertNum() - (usedCoreNumBefore3 - 1) * handleNumPerCoreBefore;
    int64_t loopCountTailBefore = 1;
    int64_t handleNumTailCorePerLoopBefore = handleNumTailCoreBefore;
    
    int64_t handleNumTailCoreMainLoop = (handleNumTailCoreBefore > ONCE_HANDLE_COUNT) ? ONCE_HANDLE_COUNT : handleNumTailCoreBefore;
    int64_t loopCountTailCoreMainLoop = (handleNumTailCoreBefore > ONCE_HANDLE_COUNT)? handleNumTailCoreBefore / ONCE_HANDLE_COUNT: 1;
    int64_t handleNumTailCoreTailLoop = handleNumTailCoreBefore - loopCountTailCoreMainLoop * handleNumTailCoreMainLoop;
    int64_t loopCountTailCoreTailLoop = handleNumTailCoreTailLoop == 0 ? 0: 1;
    int64_t handleExpertNumMainCorePerLoop = (tilingData.get_numOfExpert() >= E_BOUND_NUM) ? E_BOUND_NUM : tilingData.get_numOfExpert();
    int64_t handleExpertNumLoopCount = CeilDiv(tilingData.get_numOfExpert(), E_BOUND_NUM);
    int64_t handleExpertNumTailCorePerLoop = tilingData.get_numOfExpert() - (handleExpertNumLoopCount - 1) * handleExpertNumMainCorePerLoop;

    tilingData.set_handleNumPerCoreBefore(handleNumPerCoreBefore);
    tilingData.set_handleNumTailCoreBefore(handleNumTailCoreBefore);
    tilingData.set_loopCountBefore(loopCountBefore);
    tilingData.set_loopCountTailBefore(loopCountTailBefore);
    tilingData.set_handleNumPerLoopBefore(handleNumPerLoopBefore);
    tilingData.set_handleNumTailCorePerLoopBefore(handleNumTailCorePerLoopBefore);

    tilingData.set_handleNumTailCoreMainLoop(handleNumTailCoreMainLoop);
    tilingData.set_loopCountTailCoreMainLoop(loopCountTailCoreMainLoop);
    tilingData.set_handleNumTailCoreTailLoop(handleNumTailCoreTailLoop);
    tilingData.set_loopCountTailCoreTailLoop(loopCountTailCoreTailLoop);
    tilingData.set_handleExpertNumMainCorePerLoop(handleExpertNumMainCorePerLoop);
    tilingData.set_handleExpertNumLoopCount(handleExpertNumLoopCount);
    tilingData.set_handleExpertNumTailCorePerLoop(handleExpertNumTailCorePerLoop);
    tilingData.set_usedCoreNumBefore3(usedCoreNumBefore3);

    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus Tiling4MoeComputeExpertTokens(gert::TilingContext* context)
{
    OPS_LOG_D(context->GetNodeName(), "[MoeComputeExpertTokens] Tiling4MoeComputeExpertTokens running begin");
    OPS_CHECK(!CheckParamsShape(context),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "Tiling4MoeComputeExpertTokens check shape fail."),
                    return ge::GRAPH_FAILED);
    
    auto platformInfo = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platformInfo, return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OPS_CHECK((totalCoreNum <= 0),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "TilingPrepare4MoeComputeExpertTokens fail to get core num."),
                    return ge::GRAPH_FAILED);
    
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OPS_CHECK((ubSizePlatForm <= 0),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "TilingPrepare4MoeComputeExpertTokens fail to get ub size."),
                    return ge::GRAPH_FAILED);

    // 实例化对象op
    MoeComputeExpertTokensTilingData tilingData;

    // 设置totalCoreNum
    tilingData.set_totalCoreNum(totalCoreNum);

    // 设置总ubsize
    int64_t ubSize = static_cast<int64_t>(ubSizePlatForm);
    tilingData.set_ubSize(ubSize);

    // 获取sortedExpert的维度
    auto sortedExpertInput = context->GetInputTensor(0);
    OPS_LOG_E_IF_NULL(context, sortedExpertInput, return ge::GRAPH_FAILED);
    auto sortedExpertInputShape = sortedExpertInput->GetStorageShape();
    OPS_CHECK(sortedExpertInputShape.GetDimNum() != 1,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "DimNum of sorted expert should be 1D, please check."),
                    return ge::GRAPH_FAILED);

    tilingData.set_sortedExpertNum(sortedExpertInputShape.GetDim(0));
    OPS_CHECK(tilingData.get_sortedExpertNum() > BSK_BOUND_NUM_UP_LIMIT,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "Number of sorted expert should not larger than 2**24, please check."),
                    return ge::GRAPH_FAILED);

    // 获取numExpert的输入数据的属性
    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int32_t* numOfExpertPtr = attrs->GetAttrPointer<int32_t>(0);
    OPS_LOG_E_IF_NULL(context, numOfExpertPtr, return ge::GRAPH_FAILED);
    const int32_t numOfExpert = *numOfExpertPtr;
    tilingData.set_numOfExpert(numOfExpert);
    OPS_CHECK(tilingData.get_numOfExpert() > E_BOUND_NUM_UP_LIMIT,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "Number of expert should not larger than 2048, please check."),
                    return ge::GRAPH_FAILED);

    // 设置syncAll之前before模板3的参数
    OPS_CHECK(CalcTemplate3ParamTiling(context, tilingData) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "CalcTemplate3ParamTiling fail."),
                    return ge::GRAPH_FAILED);

    // 设置syncAll之前的参数设置
    OPS_CHECK(CalcSortedExpertTiling(context, tilingData) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "CalcSortedExpertTiling fail."),
                    return ge::GRAPH_FAILED);

    // 设置syncAll之后的参数设置
    OPS_CHECK(CalcNumOfExpertTiling(context, tilingData) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "CalcNumOfExpertTiling fail."),
                    return ge::GRAPH_FAILED);

    // 设置tilingKey
    bool isSortedExpertOverBound = (tilingData.get_sortedExpertNum() > BSK_BOUND_NUM);
    bool isNumOfExpertOverBound = (tilingData.get_numOfExpert() > E_BOUND_NUM);
    if (!isSortedExpertOverBound && !isNumOfExpertOverBound) {
        tilingData.set_tilingKey(COM_SCENE_FLAG_1);
    } else if (isSortedExpertOverBound && !isNumOfExpertOverBound) {
        tilingData.set_tilingKey(COM_SCENE_FLAG_2);
    } else {
        tilingData.set_tilingKey(COM_SCENE_FLAG_3);
    }

    // 特殊网络case场景
    bool isNetScene = (tilingData.get_sortedExpertNum() == NET_BSK_BOUND_NUM) &&
                      (tilingData.get_numOfExpert() == NET_E_BOUND_NUM);
    if (isNetScene) {
        tilingData.set_tilingKey(COM_SCENE_FLAG_1);
    }

     // 计算workLocal使用空间
    int64_t handleNum = tilingData.get_normalCoreHandleNumBefore();
    int64_t workLocalNeedSize = CalcWorkLocal(handleNum);
    tilingData.set_workLocalNeedSize(workLocalNeedSize);

    // 设置workspace
    size_t userSize = tilingData.get_numOfExpert() * tilingData.get_totalCoreNum() * sizeof(int32_t);
    size_t sysWorkspaceSize = SYS_WORKSPACE;
    size_t* userWorkspaceSize = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL(context, userWorkspaceSize, return ge::GRAPH_FAILED);
    userWorkspaceSize[0] = userSize + sysWorkspaceSize;
    tilingData.set_userWorkspaceSize(userWorkspaceSize[0]);

    OPS_CHECK(MoeComputeExpertTokensSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "MoeComputeExpertTokensSetTilingData set tiling data fail."),
                    return ge::GRAPH_FAILED);
    
    context->SetBlockDim(tilingData.get_totalCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());

    PrintTilingData(tilingData);

    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForMoeComputeExpertTokens(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeComputeExpertTokens)
    .Tiling(Tiling4MoeComputeExpertTokens)
    .TilingParse<MoeComputeExpertTokensCompileInfo>(TilingPrepareForMoeComputeExpertTokens);

}  // namespace optiling
