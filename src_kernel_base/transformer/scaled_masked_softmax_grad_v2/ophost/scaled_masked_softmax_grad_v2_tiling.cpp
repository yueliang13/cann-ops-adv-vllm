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
 * \file scaled_masked_softmax_grad_v2_tiling.cpp
 * \brief
 */
#include "scaled_masked_softmax_grad_v2_tiling.h"

namespace {
    constexpr uint64_t INPUT_Y_GRAD_IDX = 0;
    constexpr uint64_t INPUT_Y_IDX = 1;
    constexpr uint64_t INPUT_MASK_IDX = 2;
    constexpr uint64_t DIM_0 = 0;
    constexpr uint64_t DIM_1 = 1;
    constexpr uint64_t DIM_2 = 2;
    constexpr uint64_t DIM_3 = 3;
    constexpr uint64_t DIM_NUM = 4;
    constexpr uint64_t REQUIRED_INPUT_NUM = 2;
    constexpr uint64_t REQUIRED_OUTPUT_NUM = 1;
    constexpr uint64_t ATTR_0 = 0;
    constexpr uint64_t ATTR_1 = 1;
    constexpr uint64_t ALIGNED_NUM = 64;
    constexpr uint64_t SIZE_2 = 2;
    constexpr uint64_t SIZE_4 = 4;
    constexpr uint64_t SELECT_MAX_SIZE = 18 * 1024;
    constexpr uint64_t LAST_DIM_MAX_SIZE = 4096;
    constexpr uint64_t MAX_NORM_HEAD_DIM = 1024;
    constexpr uint64_t TILING_KEY_FP16 = 1;
    constexpr uint64_t TILING_KEY_FP32 = 2;
    constexpr uint64_t MASK_MODE_BNSD = 0;
    constexpr uint64_t MASK_MODE_1NSD = 1;
    constexpr uint64_t MASK_MODE_B1SD = 2;
    constexpr uint64_t MASK_MODE_11SD = 3;
}

namespace optiling {
class ScaledMaskedSoftmaxGradV2Tiling {
public:
    explicit ScaledMaskedSoftmaxGradV2Tiling(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus DoTiling();
    void PrintInfo();
    uint64_t CeilDiv(const uint64_t& dividend, const uint64_t& divisor);

private:
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckCoreInfo();
    ge::graphStatus CalcTilingParams();
    ge::graphStatus CalcLargeHeadDimInfo();
    ge::graphStatus CalcNormHeadDimInfo();
    ge::graphStatus SetMaskMode();
    ge::graphStatus SetTilingKey();
    ge::graphStatus SetAttrParams();
    ge::graphStatus SetTilingParams();

private:
    ScaledMaskedSoftmaxGradV2TilingData tiling;
    gert::TilingContext* context_ = nullptr;
    ge::DataType dataType;
    const char *opName = nullptr;

    int64_t coreNum;
    uint64_t batch;
    uint64_t channel;
    uint64_t seqLength;
    uint64_t headDim;
    uint64_t maskBatch;
    uint64_t maskChannel;
    uint64_t ubSize;
    uint64_t usedUbSize;
    uint64_t totalLinePerHeadCore;
    uint64_t paddedHeadDim;
    uint64_t selectSize;
    uint64_t maxLinePerLoop;
    uint64_t dataSize;
};

void ScaledMaskedSoftmaxGradV2Tiling::PrintInfo() {
    OPS_LOG_D(opName, "----------------Start to print ScaledMaskedSoftmaxGradV2 tiling data.----------------");
    OPS_LOG_D(opName, ">>> usedCoreNum:                     %lu", tiling.get_usedCoreNum());
    OPS_LOG_D(opName, ">>> batch:                           %lu", tiling.get_batch());
    OPS_LOG_D(opName, ">>> channel:                         %lu", tiling.get_channel());
    OPS_LOG_D(opName, ">>> seqLength:                       %lu", tiling.get_seqLength());
    OPS_LOG_D(opName, ">>> headDim:                         %lu", tiling.get_headDim());
    OPS_LOG_D(opName, ">>> totalLine:                       %lu", tiling.get_totalLine());
    OPS_LOG_D(opName, ">>> paddedHeadDim:                   %lu", tiling.get_paddedHeadDim());
    OPS_LOG_D(opName, ">>> totalLinePerHeadCore:            %lu", tiling.get_totalLinePerHeadCore());
    OPS_LOG_D(opName, ">>> totalLinePerTailCore:            %lu", tiling.get_totalLinePerTailCore());
    OPS_LOG_D(opName, ">>> maxLinePerLoop:                  %lu", tiling.get_maxLinePerLoop());
    OPS_LOG_D(opName, ">>> tailLinePerHeadCore:             %lu", tiling.get_tailLinePerHeadCore());
    OPS_LOG_D(opName, ">>> tailLinePerTailCore:             %lu", tiling.get_tailLinePerTailCore());
    OPS_LOG_D(opName, ">>> headCoreNum:                     %lu", tiling.get_headCoreNum());
    OPS_LOG_D(opName, ">>> maskMoveMode:                    %lu", tiling.get_maskMoveMode());
    OPS_LOG_D(opName, ">>> selectSize:                      %lu", tiling.get_selectSize());
    OPS_LOG_D(opName, ">>> scale:                           %f",  tiling.get_scale());
    OPS_LOG_D(opName, ">>> fixedTriuMask:                   %u",  tiling.get_fixedTriuMask());
    OPS_LOG_D(opName, "----------------Print ScaledMaskedSoftmaxGradV2 tiling data end.<<<<<<<<<<<<<<<<");
}

uint64_t ScaledMaskedSoftmaxGradV2Tiling::CeilDiv(const uint64_t& dividend, const uint64_t& divisor) {
    if (divisor == 0) {
        return divisor;
    }
    return (dividend + divisor - 1) / divisor;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::CheckInputShape() {
    // check y_grad
    auto yGradShapePtr = context_->GetInputShape(INPUT_Y_GRAD_IDX);
    OPS_LOG_E_IF_NULL(context_, yGradShapePtr, return ge::GRAPH_FAILED);
    auto yGradShape = yGradShapePtr->GetStorageShape();
    int64_t yGradDimNum = yGradShape.GetDimNum();
    OPS_CHECK((yGradDimNum != DIM_NUM), OPS_REPORT_VECTOR_INNER_ERR(opName,
        "yGradDimNum must be 4."), return ge::GRAPH_FAILED);
    batch = yGradShape.GetDim(DIM_0);
    channel = yGradShape.GetDim(DIM_1);
    seqLength = yGradShape.GetDim(DIM_2);
    headDim = yGradShape.GetDim(DIM_3);
    OPS_CHECK((batch <= 0 || channel <= 0 || seqLength <= 0 || headDim <= 0),
        OPS_REPORT_VECTOR_INNER_ERR(opName, "The length of yGradDim must be greater than 0."),
        return ge::GRAPH_FAILED);
    OPS_CHECK((headDim > LAST_DIM_MAX_SIZE), OPS_REPORT_VECTOR_INNER_ERR(opName,
        "The length yGrad dim 3 must be less than or equal to 4096."), return ge::GRAPH_FAILED);

    // check y
    auto yShapePtr = context_->GetInputShape(INPUT_Y_IDX);
    OPS_LOG_E_IF_NULL(context_, yShapePtr, return ge::GRAPH_FAILED);
    auto yShape = yShapePtr->GetStorageShape();
    OPS_CHECK((yShape != yGradShape), OPS_REPORT_VECTOR_INNER_ERR(opName,
        "yShape should be same as yGradShape"), return ge::GRAPH_FAILED);

    // check mask
    auto maskShapePtr = context_->GetInputShape(INPUT_MASK_IDX);
    OPS_LOG_E_IF_NULL(context_, maskShapePtr, return ge::GRAPH_FAILED);
    auto maskShape = maskShapePtr->GetStorageShape();
    int64_t maskDimNum = maskShape.GetDimNum();
    OPS_CHECK((maskDimNum != DIM_NUM), OPS_REPORT_VECTOR_INNER_ERR(opName,
        "maskDimNum must be 4."), return ge::GRAPH_FAILED);

    // check input and mask dim
    maskBatch = maskShape.GetDim(DIM_0);
    maskChannel = maskShape.GetDim(DIM_1);
    uint64_t maskSeqLength = maskShape.GetDim(DIM_2);
    uint64_t maskHeadDim = maskShape.GetDim(DIM_3);
    OPS_CHECK((seqLength != maskSeqLength || headDim != maskHeadDim), OPS_REPORT_VECTOR_INNER_ERR(
        opName, "The last two dims of mask must be equal to the last two dims of yGrad."),
        return ge::GRAPH_FAILED);
    OPS_CHECK((maskBatch <= 0 || maskChannel <= 0), OPS_REPORT_VECTOR_INNER_ERR(opName,
        "The length of maskDim must be greater than 0."), return ge::GRAPH_FAILED);
    OPS_CHECK(((batch != maskBatch && maskBatch != 1) || (channel != maskChannel && maskChannel != 1)),
        OPS_REPORT_VECTOR_INNER_ERR(opName, "mask shape must be broadcast to yGrad shape."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::CheckInputDtype() {
    auto inputYGrad = context_->GetInputDesc(INPUT_Y_GRAD_IDX);
    OPS_LOG_E_IF_NULL(context_, inputYGrad, return ge::GRAPH_FAILED);
    auto yGradDtype = inputYGrad->GetDataType();
    OPS_CHECK((yGradDtype != ge::DT_FLOAT16 && yGradDtype != ge::DT_FLOAT && yGradDtype != ge::DT_BF16),
        OPS_REPORT_VECTOR_INNER_ERR(opName, "yGradDtype should be FP16/BF16/FP32"),
        return ge::GRAPH_FAILED);

    dataType = yGradDtype;
    if (dataType == ge::DT_FLOAT) {
        dataSize = SIZE_4;
    } else {
        dataSize = SIZE_2;
    }

    // check y
    auto inputY = context_->GetInputDesc(INPUT_Y_IDX);
    OPS_LOG_E_IF_NULL(context_, inputY, return ge::GRAPH_FAILED);
    auto yDtype = inputY->GetDataType();
    OPS_CHECK((yDtype != yGradDtype), OPS_REPORT_VECTOR_INNER_ERR(opName,
                    "yDtype should be same as yGradDtype"), return ge::GRAPH_FAILED);

    auto inputMask = context_->GetInputDesc(INPUT_MASK_IDX);
    OPS_LOG_E_IF_NULL(context_, inputMask, return ge::GRAPH_FAILED);
    auto maskDtype = inputMask->GetDataType();
    OPS_CHECK((maskDtype != ge::DT_BOOL), OPS_REPORT_VECTOR_INNER_ERR(opName,
                    "maskDtype should be bool"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::CheckCoreInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OPS_LOG_D(opName, "Start to check core info.");
    OPS_LOG_D(opName, "Get total core num:%ld", coreNum);
    OPS_CHECK((coreNum <= 0), OPS_REPORT_VECTOR_INNER_ERR(opName, "Failed to get core num."),
                    return ge::GRAPH_FAILED);
    OPS_LOG_D(opName, "Get total ub size:%lu", ubSize);
    OPS_LOG_D(opName, "Check core info ends.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::CalcLargeHeadDimInfo()
{
    selectSize = SELECT_MAX_SIZE;
    usedUbSize = paddedHeadDim * (REQUIRED_INPUT_NUM * dataSize + SIZE_4 *
                (dataSize == SIZE_2 ? REQUIRED_INPUT_NUM : 0));
    maxLinePerLoop = (ubSize - selectSize) / usedUbSize;
    maxLinePerLoop = std::min(maxLinePerLoop, totalLinePerHeadCore);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::CalcNormHeadDimInfo()
{
    std::vector<int64_t> shapeVec = {1, static_cast<int64_t>(paddedHeadDim)};
    ge::Shape srcShape(shapeVec);
    uint64_t oneLineSoftmaxGradSize = AscendC::GetSoftMaxGradMaxTmpSize(srcShape, SIZE_4, false, false);
    // inQue*2+outQue + cast*2 + maskQue + api_tmp_buf
    usedUbSize = paddedHeadDim * ((REQUIRED_INPUT_NUM + REQUIRED_OUTPUT_NUM) * dataSize +
        SIZE_4 * (dataSize == SIZE_2 ? REQUIRED_INPUT_NUM : 0)) + paddedHeadDim + oneLineSoftmaxGradSize;
    maxLinePerLoop = ubSize / usedUbSize;
    uint64_t maxLocalWorkSpaceSize = oneLineSoftmaxGradSize * maxLinePerLoop;
    selectSize = std::max(maxLocalWorkSpaceSize, SELECT_MAX_SIZE);
    if (selectSize > maxLocalWorkSpaceSize) {
        maxLinePerLoop = (ubSize - selectSize) / (usedUbSize - oneLineSoftmaxGradSize);
    }
    maxLinePerLoop = std::min(maxLinePerLoop, totalLinePerHeadCore);
    shapeVec = {static_cast<int64_t>(maxLinePerLoop), static_cast<int64_t>(paddedHeadDim)};
    srcShape = ge::Shape(shapeVec);
    AscendC::SoftMaxGradTilingFunc(srcShape, SIZE_4, selectSize, tiling.softmaxGradTilingData, false);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::CalcTilingParams()
{
    uint64_t totalLine = batch * channel * seqLength;
    uint64_t usedCoreNum = std::min(totalLine, static_cast<uint64_t>(coreNum));
    totalLinePerHeadCore = CeilDiv(totalLine, usedCoreNum);
    uint64_t totalLinePerTailCore = 0;
    uint64_t headCoreNum = 0;
    if (usedCoreNum != 0) {
        totalLinePerTailCore = totalLine / usedCoreNum;
        headCoreNum = totalLine % usedCoreNum;
    }

    paddedHeadDim = CeilDiv(headDim, ALIGNED_NUM) * ALIGNED_NUM;
    if (paddedHeadDim > MAX_NORM_HEAD_DIM) {
        CalcLargeHeadDimInfo();
    } else {
        CalcNormHeadDimInfo();
    }

    uint64_t tailLinePerHeadCore = totalLinePerHeadCore % maxLinePerLoop;
    uint64_t tailLinePerTailCore = totalLinePerTailCore % maxLinePerLoop;
    if (tailLinePerHeadCore == 0) {
        tailLinePerHeadCore = maxLinePerLoop;
    }
    if (tailLinePerTailCore == 0) {
        tailLinePerTailCore = maxLinePerLoop;
    }

    // set tiling data
    context_->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_totalLine(totalLine);
    tiling.set_totalLinePerTailCore(totalLinePerTailCore);
    tiling.set_tailLinePerHeadCore(tailLinePerHeadCore);
    tiling.set_tailLinePerTailCore(tailLinePerTailCore);
    tiling.set_headCoreNum(headCoreNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::SetMaskMode()
{
    uint64_t maskMoveMode = 0u;
    if (batch == maskBatch && channel == maskChannel) {
        maskMoveMode = MASK_MODE_BNSD;
    } else if (1 == maskBatch && channel == maskChannel) {
        maskMoveMode = MASK_MODE_1NSD;
    } else if (batch == maskBatch && 1 == maskChannel) {
        maskMoveMode = MASK_MODE_B1SD;
    } else if (1 == maskBatch && 1 == maskChannel) {
        maskMoveMode = MASK_MODE_11SD;
    }

    tiling.set_maskMoveMode(maskMoveMode);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::SetTilingKey()
{
    uint64_t tilingKey = (paddedHeadDim <= MAX_NORM_HEAD_DIM ? 1000 : 2000);
    if (dataType == ge::DT_FLOAT16) {
        tilingKey += TILING_KEY_FP16;
    } else if (dataType == ge::DT_FLOAT) {
        tilingKey += TILING_KEY_FP32;
    }
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::SetAttrParams()
{
    auto attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);
    const float* scale = attrs->GetAttrPointer<float>(ATTR_0);
    const bool* fixedTriuMask = attrs->GetAttrPointer<bool>(ATTR_1);

    tiling.set_scale(*scale);
    tiling.set_fixedTriuMask(static_cast<uint32_t>(*fixedTriuMask));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::SetTilingParams()
{
    tiling.set_batch(batch);
    tiling.set_channel(channel);
    tiling.set_seqLength(seqLength);
    tiling.set_headDim(headDim);
    tiling.set_paddedHeadDim(paddedHeadDim);
    tiling.set_maxLinePerLoop(maxLinePerLoop);
    tiling.set_totalLinePerHeadCore(totalLinePerHeadCore);
    tiling.set_selectSize(selectSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::DoTiling()
{
    opName = context_->GetNodeName();
    OPS_LOG_D(opName, "Start running Tiling4ScaledMaskedSoftmaxGradV2.");
    OPS_CHECK((CheckInputShape() != ge::GRAPH_SUCCESS), OPS_REPORT_VECTOR_INNER_ERR(opName,
                    "InputShape is invalid."), return ge::GRAPH_FAILED);
    OPS_CHECK((CheckInputDtype() != ge::GRAPH_SUCCESS), OPS_REPORT_VECTOR_INNER_ERR(opName,
                    "InputShape is invalid."), return ge::GRAPH_FAILED);       
    OPS_CHECK((CheckCoreInfo() != ge::GRAPH_SUCCESS), OPS_REPORT_VECTOR_INNER_ERR(opName,
                    "CoreNum or ubSize is invalid."), return ge::GRAPH_FAILED);

    CalcTilingParams();

    SetTilingParams();
    SetMaskMode();
    SetTilingKey();
    SetAttrParams();

    PrintInfo();

    // 固定写法
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL(context_, currentWorkspace, return ge::GRAPH_FAILED);
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus Tiling4ScaledMaskedSoftmaxGradV2(gert::TilingContext* context)
{
    ScaledMaskedSoftmaxGradV2Tiling tilingObj(context);
    return tilingObj.DoTiling();
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepare4ScaledMaskedSoftmaxGradV2(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScaledMaskedSoftmaxGradV2)
    .Tiling(Tiling4ScaledMaskedSoftmaxGradV2)
    .TilingParse<ScaledMaskedSoftmaxGradV2CompileInfo>(TilingPrepare4ScaledMaskedSoftmaxGradV2);
} // namespace optiling