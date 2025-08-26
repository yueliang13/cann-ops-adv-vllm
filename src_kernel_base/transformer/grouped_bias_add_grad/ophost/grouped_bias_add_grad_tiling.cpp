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
 * \file grouped_bias_add_grad_tiling.cpp
 * \brief
 */

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "register/op_def_registry.h"
#include "log/ops_log.h"
#include "error/ops_error.h"
#include "grouped_bias_add_grad_tiling_base.h"
#include "grouped_bias_add_grad_tiling.h"

namespace optiling {
namespace groupedbiasaddgrad {
template <typename T>
typename std::enable_if <std::is_signed<T>::value, T>::type CeilDiv(T x, T y) {
  if (y != 0 && x != 0) {
    const T quotient = x / y;
    return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
  }

  return x;
}

/**
 * if y is 0, return x
 */
template <typename T>
typename std::enable_if <std::is_unsigned<T>::value, T>::type CeilDiv(T x, T y) {
  if (y != 0 && x != 0) {
    const T quotient = x / y;
    return (x % y != 0) ? (quotient + 1) : quotient;
  }

  return x;
}

static const uint32_t ATTR_GROUP_IDX_TYPE_INDEX = 0;

ge::graphStatus GroupedBiasAddGradTiling::GetPlatformInfo()
{
    OPS_LOG_D(nodeName_, "[GroupedBiasAddGrad] GetPlatformInfo start running.");
    auto compileInfo = reinterpret_cast<const GroupedBiasAddGradCompileInfo *>(context_->GetCompileInfo());
    OPS_LOG_E_IF_NULL(context_, compileInfo, return ge::GRAPH_FAILED);

    baseInfoOp_.vectorCoreNum = compileInfo->coreNum;
    OPS_CHECK((baseInfoOp_.vectorCoreNum <= 0),
                    OPS_REPORT_VECTOR_INNER_ERR(
                        nodeName_, "GroupedBiasAddGradTiling get num of vector core is less than or equal to 0."),
                    return ge::GRAPH_FAILED);

    baseInfoOp_.ubSize = compileInfo->ubSize;
    OPS_CHECK(
        (baseInfoOp_.ubSize <= 0),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "GroupedBiasAddGradTiling get ub size is less than or equal to 0."),
        return ge::GRAPH_FAILED);

    baseInfoOp_.ubSize -= RESERVED_UB_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::GetInputInfo()
{
    // grad_y 处理包含dtype、shape校验，获取G、H、C大小
    // 可选输入group_idx，包含shape校验
    // grad_bias 处理包含shape校验
    OPS_LOG_D(nodeName_, "[GroupedBiasAddGrad] GetInputInfo start running.");
    auto gradYInputDesc = context_->GetInputDesc(GRAD_Y_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, gradYInputDesc, return ge::GRAPH_FAILED);
    baseInfoOp_.gradYInputDtype = gradYInputDesc->GetDataType();
    OPS_CHECK(
        (baseInfoOp_.gradYInputDtype != ge::DT_FLOAT && baseInfoOp_.gradYInputDtype != ge::DT_FLOAT16 &&
         baseInfoOp_.gradYInputDtype != ge::DT_BF16),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "the dtype of input grad_y should be one of FP32/FP16/BF16."),
        return ge::GRAPH_FAILED);

    auto gradYInputShapePtr = context_->GetInputShape(GRAD_Y_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, gradYInputShapePtr, return ge::GRAPH_FAILED);
    auto gradYInputShape = gradYInputShapePtr->GetStorageShape();
    baseInfoOp_.gradYDimNum = gradYInputShape.GetDimNum();

    // group_idx
    auto groupIdxInputShapePtr = context_->GetOptionalInputShape(GROUP_IDX_INPUT_INDEX);
    if (groupIdxInputShapePtr == nullptr) {
        OPS_CHECK((baseInfoOp_.gradYDimNum != THREE_NUM),
            OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "the input grad_y should be 3D tensor when group_idx is null."),
            return ge::GRAPH_FAILED);
        baseInfoOp_.dimG = gradYInputShape.GetDim(0);
        baseInfoOp_.dimC = gradYInputShape.GetDim(1);
        baseInfoOp_.dimH = gradYInputShape.GetDim(TWO_NUM);
    } else {
        OPS_CHECK(
            (baseInfoOp_.gradYDimNum != TWO_NUM || groupIdxInputShapePtr->GetStorageShape().GetDimNum() != 1),
            OPS_REPORT_VECTOR_INNER_ERR(nodeName_,
            "the input grad_y should be 2D tensor when group_idx is not null. and group_idx must be 1D tensor."),
            return ge::GRAPH_FAILED);
        baseInfoOp_.existGroupIdx = 1;
        baseInfoOp_.dimGB = gradYInputShape.GetDim(0);
        baseInfoOp_.dimG = groupIdxInputShapePtr->GetStorageShape().GetShapeSize();
        baseInfoOp_.dimH = gradYInputShape.GetDim(1);
        auto groupIdxInputDesc = context_->GetInputDesc(GROUP_IDX_INPUT_INDEX);
        OPS_LOG_E_IF_NULL(context_, groupIdxInputDesc, return ge::GRAPH_FAILED);
        baseInfoOp_.groupIdxInputDtype = groupIdxInputDesc->GetDataType();
        OPS_CHECK((baseInfoOp_.groupIdxInputDtype != ge::DT_INT32 &&
            baseInfoOp_.groupIdxInputDtype != ge::DT_INT64),
            OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "the dtype of input group_idx should be INT32 or INT64."),
            return ge::GRAPH_FAILED);
    }

    if (baseInfoOp_.existGroupIdx == 1 && baseInfoOp_.dimG > INPUT_MAX_GROUP) {
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "input group_idx shape not support more than %ld, but got %ld.",
                                        INPUT_MAX_GROUP, baseInfoOp_.dimG);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::GetAttrInfo()
{
    OPS_LOG_D(nodeName_, "[GroupedBiasAddGrad] GetAttrInfo start running.");
    auto* attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);

    auto *attrGroupIdxType = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_IDX_TYPE_INDEX);
    OPS_LOG_E_IF_NULL(context_, attrGroupIdxType, return ge::GRAPH_FAILED);

    baseInfoOp_.groupIdxType = static_cast<int32_t>(*attrGroupIdxType);
    OPS_CHECK((baseInfoOp_.groupIdxType != 0 && baseInfoOp_.groupIdxType != 1),
            OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "the value of group_idx_type should be 0 or 1, but got %d",
            baseInfoOp_.groupIdxType), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::CheckOutput()
{
    // grad_bias
    auto gradBiasOutputShapePtr = context_->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context_, gradBiasOutputShapePtr, return ge::GRAPH_FAILED);
    auto gradBiasOutputShape = gradBiasOutputShapePtr->GetStorageShape();
    OPS_CHECK((gradBiasOutputShape.GetDimNum() != TWO_NUM),
                    OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "the dim of grad_bias should be 2, but got %zu.",
                                                    gradBiasOutputShape.GetDimNum()),
                    return ge::GRAPH_FAILED);
    auto gradBiasdim0 = gradBiasOutputShape.GetDim(0);
    auto gradBiasdim1 = gradBiasOutputShape.GetDim(1);
    OPS_CHECK(((gradBiasdim0 != baseInfoOp_.dimG) || (gradBiasdim1 != baseInfoOp_.dimH)),
                    OPS_REPORT_VECTOR_INNER_ERR(nodeName_,
                                                    "the shape of grad_bias should be [%ld, %ld], bug got [%ld, %ld].",
                                                    baseInfoOp_.dimG, baseInfoOp_.dimH, gradBiasdim0, gradBiasdim1),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::DoSplitTiling()
{
    OPS_LOG_D(nodeName_, "[GroupedBiasAddGrad] DoUnequalCTiling start running.");
    int64_t coreNum = baseInfoOp_.vectorCoreNum;
    int64_t totalSplitNum = CeilDiv(baseInfoOp_.dimH, splitCoreOp_.baseH) * baseInfoOp_.dimG;
    splitCoreOp_.usedCoreNum = totalSplitNum > coreNum ? coreNum : totalSplitNum;
    splitCoreOp_.normalCoreProcessNum = CeilDiv(totalSplitNum, splitCoreOp_.usedCoreNum);
    splitCoreOp_.tailCoreProcessNum = splitCoreOp_.normalCoreProcessNum - 1;
    int64_t tailCoreNum = splitCoreOp_.normalCoreProcessNum * splitCoreOp_.usedCoreNum - totalSplitNum;
    splitCoreOp_.normalCoreNum = splitCoreOp_.usedCoreNum - tailCoreNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::DoUnequalCPerformanceTiling()
{
    OPS_LOG_D(nodeName_, "[GroupedBiasAddGrad] DoUnequalCTiling start running.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::DoTiling()
{
    // 分别走C等长和不等长模板切分，C不等长情况还可以根据H大小判断走高性能模板
    splitCoreOp_.baseH = H_BASE_SIZE / sizeof(float);
    int64_t avilableUbsize = baseInfoOp_.ubSize - H_BASE_SIZE * BUFFER_NUM - UB_GROUP_SUM_NUM * H_BASE_SIZE;
    splitCoreOp_.baseC = avilableUbsize / ACTIVE_NODES_NUM / sizeof(float) / splitCoreOp_.baseH;
    DoSplitTiling();
    bool perfTemp = baseInfoOp_.dimG >= PERF_G_NUM &&
                    splitCoreOp_.usedCoreNum >= baseInfoOp_.vectorCoreNum &&
                    baseInfoOp_.dimG % TWO_NUM == 0 &&
                    baseInfoOp_.dimH / splitCoreOp_.baseH >= splitCoreOp_.usedCoreNum / TWO_NUM;
    int64_t unitLength = baseInfoOp_.groupIdxInputDtype == ge::DT_INT32 ? sizeof(int32_t) : sizeof(int64_t);
    if (baseInfoOp_.existGroupIdx == 0) {
        splitCoreOp_.loopCNum = CeilDiv(baseInfoOp_.dimC, splitCoreOp_.baseC);
        splitCoreOp_.wsUnitNum = splitCoreOp_.loopCNum;
    } else if (perfTemp && baseInfoOp_.dimGB < INT32_MAX) {
        baseInfoOp_.performance = 1;
        int64_t groupIdxAlign = CeilDiv(baseInfoOp_.dimG, BLOCK_SIZE / unitLength) * BLOCK_SIZE / unitLength;
        int64_t sortBuffer =
                CeilDiv(baseInfoOp_.dimG, BLOCK_SIZE) * BLOCK_SIZE * sizeof(int32_t) * TWO_NUM * TWO_NUM;
        int64_t groupIdxExtraBuffer = groupIdxAlign * unitLength * TWO_NUM + sortBuffer;
        avilableUbsize = avilableUbsize - groupIdxExtraBuffer;
        splitCoreOp_.baseC = avilableUbsize / ACTIVE_NODES_NUM / sizeof(float) / splitCoreOp_.baseH;
        splitCoreOp_.wsUnitNum = CeilDiv(baseInfoOp_.dimGB, splitCoreOp_.baseC);
        splitCoreOp_.normalCoreProcessNum = (baseInfoOp_.dimG + 1) / TWO_NUM;
        splitCoreOp_.tailCoreProcessNum = baseInfoOp_.dimG / TWO_NUM;
        splitCoreOp_.normalCoreNum = splitCoreOp_.usedCoreNum / TWO_NUM;
    } else {
        int64_t groupIdxAlign = (baseInfoOp_.dimG * unitLength + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        avilableUbsize = avilableUbsize - groupIdxAlign - groupIdxAlign;
        splitCoreOp_.baseC = avilableUbsize / ACTIVE_NODES_NUM / sizeof(float) / splitCoreOp_.baseH;
        splitCoreOp_.wsUnitNum = CeilDiv(baseInfoOp_.dimGB, splitCoreOp_.baseC);
    }
    return ge::GRAPH_SUCCESS;
}

void GroupedBiasAddGradTiling::SaveToTilingData()
{
    tilingData_.set_usedCoreNum(splitCoreOp_.usedCoreNum);
    tilingData_.set_normalCoreNum(splitCoreOp_.normalCoreNum);
    tilingData_.set_normalCoreProcessNum(splitCoreOp_.normalCoreProcessNum);
    tilingData_.set_tailCoreProcessNum(splitCoreOp_.tailCoreProcessNum);
    tilingData_.set_wsUnitNum(splitCoreOp_.wsUnitNum);
    tilingData_.set_dimG(baseInfoOp_.dimG);
    tilingData_.set_dimC(baseInfoOp_.dimC);
    tilingData_.set_dimH(baseInfoOp_.dimH);
    tilingData_.set_dimGB(baseInfoOp_.dimGB);
    tilingData_.set_baseH(splitCoreOp_.baseH);
    tilingData_.set_baseC(splitCoreOp_.baseC);
    tilingData_.set_loopCNum(splitCoreOp_.loopCNum);
    tilingData_.set_groupIdxType(baseInfoOp_.groupIdxType);
}

ge::graphStatus GroupedBiasAddGradTiling::PostTiling()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL(context_, workspaces, return ge::GRAPH_FAILED);
    size_t workspaceSize = WORKSPACE_BASE_CAL;
    workspaces[0] = workspaceSize;
    workspaces[0] += splitCoreOp_.wsUnitNum * H_BASE_SIZE * splitCoreOp_.usedCoreNum;

    SaveToTilingData();

    context_->SetBlockDim(splitCoreOp_.usedCoreNum);
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetTilingKey(GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedBiasAddGradTiling::ComputeTiling(const std::vector<uint32_t>& args) const
{
    uint64_t result = 1000000UL;
    uint64_t startValue = 1;
    constexpr uint64_t incrementCoefficient = 10;
    for (auto it = args.rbegin(); it != args.rend(); ++it) {
        result += *it * startValue;
        startValue *= incrementCoefficient;
    }
    return result;
}

uint64_t GroupedBiasAddGradTiling::GetTilingKey() const
{
    // 计算tiling，kernel入口根据不同tiling进入不同模板
    DtypeEnum inDtype = DtypeEnum::FLOAT16;
    if (baseInfoOp_.gradYInputDtype == ge::DT_BF16) {
        inDtype = DtypeEnum::BFLOAT16;
    }
    if (baseInfoOp_.gradYInputDtype == ge::DT_FLOAT) {
        inDtype = DtypeEnum::FLOAT32;
    }

    uint32_t useUBSum = 0;
    if (splitCoreOp_.wsUnitNum <= UB_GROUP_SUM_NUM) {
        useUBSum = 1;
    }

    uint32_t groupIdxDtype = baseInfoOp_.groupIdxInputDtype == ge::DT_INT32 ? 0 : 1;
    auto tilingKey =
        ComputeTiling({static_cast<uint32_t>(groupIdxDtype), baseInfoOp_.performance, useUBSum,
                       baseInfoOp_.existGroupIdx, static_cast<uint32_t>(inDtype)});
    OPS_LOG_I(nodeName_, "[GroupedBiasAddGrad] GetTilingKey [%lu].", tilingKey);
    return tilingKey;
}

void GroupedBiasAddGradTiling::DumpTilingInfo() const
{
    std::ostringstream info;
    info << "baseInfoOp_.vectorCoreNum: " << baseInfoOp_.vectorCoreNum << std::endl;
    info << "baseInfoOp_.gradYDimNum: " << baseInfoOp_.gradYDimNum << std::endl;
    info << "baseInfoOp_.dimG: " << baseInfoOp_.dimG << std::endl;
    info << "baseInfoOp_.dimC: " << baseInfoOp_.dimC << std::endl;
    info << "baseInfoOp_.dimH: " << baseInfoOp_.dimH << std::endl;
    info << "baseInfoOp_.dimGB: " << baseInfoOp_.dimGB << std::endl;
    info << "baseInfoOp_.gradYInputDtype: " << baseInfoOp_.gradYInputDtype << std::endl;
    info << "baseInfoOp_.existGroupIdx: " << baseInfoOp_.existGroupIdx << std::endl;
    info << "baseInfoOp_.performance: " << baseInfoOp_.performance << std::endl;
    info << "baseInfoOp_.groupIdxType: " << baseInfoOp_.groupIdxType << std::endl;

    info << "splitCoreOp_.usedCoreNum: " << splitCoreOp_.usedCoreNum << std::endl;
    info << "splitCoreOp_.normalCoreNum: " << splitCoreOp_.normalCoreNum << std::endl;
    info << "splitCoreOp_.normalCoreProcessNum: " << splitCoreOp_.normalCoreProcessNum << std::endl;
    info << "splitCoreOp_.tailCoreProcessNum: " << splitCoreOp_.tailCoreProcessNum << std::endl;
    info << "splitCoreOp_.baseH: " << splitCoreOp_.baseH << std::endl;
    info << "splitCoreOp_.baseC: " << splitCoreOp_.baseC << std::endl;
    info << "splitCoreOp_.loopCNum: " << splitCoreOp_.loopCNum << std::endl;
    info << "splitCoreOp_.wsUnitNum: " << splitCoreOp_.wsUnitNum << std::endl;

    OPS_LOG_I(nodeName_, "%s", info.str().c_str());
}

ge::graphStatus GroupedBiasAddGradTiling::RunGroupedBiasAddGradTiling()
{
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ret = GetInputInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetAttrInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutput();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = DoTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingForGroupedBiasAddGrad(gert::TilingContext* context)
{
    OPS_CHECK(context == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR("GroupedBiasAddGrad", "context should not be nullptr."),
                    return ge::GRAPH_FAILED);

    GroupedBiasAddGradTiling tiling(context);
    return tiling.RunGroupedBiasAddGradTiling();
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGroupedBiasAddGrad(gert::TilingParseContext* context)
{
    OPS_CHECK(context == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR("GroupedBiasAddGrad", "context should not be nullptr."),
                    return ge::GRAPH_FAILED);
    auto nodeName = context->GetNodeName();
    OPS_LOG_D(nodeName, "TilingPrepareForGroupedBiasAddGrad start.");

    auto compileInfo = context->GetCompiledInfo<GroupedBiasAddGradCompileInfo>();
    OPS_LOG_E_IF_NULL(context, compileInfo, return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platformInfo, return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OPS_CHECK((compileInfo->coreNum  <= 0),
                    OPS_REPORT_VECTOR_INNER_ERR(
                        nodeName, "Failed to get core number."),
                    return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OPS_CHECK(
        (compileInfo->ubSize <= 0),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName, "Failed to get ub size."),
        return ge::GRAPH_FAILED);

    OPS_LOG_D(nodeName, "TilingPrepareForGroupedBiasAddGrad end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedBiasAddGrad)
    .Tiling(TilingForGroupedBiasAddGrad)
    .TilingParse<GroupedBiasAddGradCompileInfo>(TilingPrepareForGroupedBiasAddGrad);
} // namespace groupedbiasaddgrad
} // namespace optiling