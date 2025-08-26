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
 * \file scaled_masked_softmax_v2_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "log/ops_log.h"
#include "error/ops_error.h"
#include "scaled_masked_softmax_v2_tiling.h"

namespace {
    constexpr uint64_t X_INDEX = 0;
    constexpr uint64_t MASK_INDEX = 1;
    constexpr uint64_t DIM_0 = 0;
    constexpr uint64_t DIM_1 = 1;
    constexpr uint64_t DIM_2 = 2;
    constexpr uint64_t DIM_3 = 3;
    constexpr uint64_t DIM_NUM = 4;
    constexpr uint64_t ATTR_0 = 0;
    constexpr uint64_t ATTR_1 = 1;
    constexpr uint64_t AlignedBytes = 32;
    constexpr uint64_t BOOL_SIZE = 1;
    constexpr uint64_t FP32_SIZE = 4;
    constexpr uint64_t XY_PARAMS = 2;
    constexpr uint64_t BROADCAST_BATCH = 1;
    constexpr uint64_t BROADCAST_CHANNEL = 2;

    constexpr uint64_t DTYPE_DIGIT = 1;
    constexpr uint64_t FP32_TILINGKEY = 0;
    constexpr uint64_t FP16_TILINGKEY = 1;
    constexpr uint64_t BF16_TILINGKEY = 2;

    constexpr uint64_t MAX_DIM_NUM = 4096;
    constexpr uint64_t SELECT_BUF_SIZE = 16 * 1024;
    constexpr uint64_t SOFTMAX_BUF_SIZE = 32 * 1024;

    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    }
}

namespace optiling {
class ScaledMaskedSoftmaxV2Tiling {
public:
    explicit ScaledMaskedSoftmaxV2Tiling(gert::TilingContext* context) : context(context) {};
    ge::graphStatus Init();
    ge::graphStatus DoTiling();
    void TilingDataPrint();

private:
    uint64_t getDataTypeSize(int32_t dtype);
    bool InitPlatformInfo();
    bool InitInputDtype();
    bool InitInputShape();
    bool InitAttr();

    void SetInputShape();
    void SetPaddingInfo();
    void SetMaskAdapt();
    bool SetUbSplitInfo();
    void SetSoftmaxTiling();
    void SetTilingKey();
    
private:
    ScaledMaskedSoftmaxV2TilingData tiling;
    gert::TilingContext* context = nullptr;

    uint64_t batch;
    uint64_t channel;
    uint64_t height;
    uint64_t width;
    uint64_t maskBatch;
    uint64_t maskChannel;
    uint64_t xDtypeSize;
    uint64_t totalCoreNum;

    float scale;
};

bool ScaledMaskedSoftmaxV2Tiling::InitInputDtype() {
    auto inputDtype = context->GetInputDesc(X_INDEX)->GetDataType();
    xDtypeSize = getDataTypeSize(inputDtype);

    OPS_CHECK(
        xDtypeSize == 0,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "[ScaledMaskedSoftmaxV2Tiling] X Dtype invalid."),
        return false);

    return true;
}

uint64_t ScaledMaskedSoftmaxV2Tiling::getDataTypeSize(int32_t dtype) {
    switch(dtype) {
        case ge::DT_FLOAT:
            return sizeof(int32_t);
        case ge::DT_BF16:
        case ge::DT_FLOAT16:
            return sizeof(int16_t);
        default:
            return 0;
    }
}

bool ScaledMaskedSoftmaxV2Tiling::InitPlatformInfo() {
    const auto ascendCPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    totalCoreNum = ascendCPlatform.GetCoreNumAiv();
    if(!(totalCoreNum > 0)) {
        return false;
    }
    context->SetBlockDim(totalCoreNum);
    tiling.set_coreNum(totalCoreNum);
    
    size_t sysWorkspaceSize = ascendCPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    return true;
}

ge::graphStatus ScaledMaskedSoftmaxV2Tiling::Init() {
    OPS_CHECK(!InitPlatformInfo(),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "InitPlatformInfo failed."),
        return false);
    OPS_CHECK(!InitAttr(),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "InitAttr failed."),
        return false);
    OPS_CHECK(!InitInputDtype(),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "InitInputDtype failed."),
        return false);
    OPS_CHECK(!InitInputShape(),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "InitInputShape failed."),
        return false);
    return ge::GRAPH_SUCCESS;
}

bool ScaledMaskedSoftmaxV2Tiling::InitInputShape() {
    auto const xShape = context->GetInputShape(X_INDEX);
    auto xShapeVal = xShape->GetStorageShape();
    auto const maskShape = context->GetInputShape(MASK_INDEX);
    auto maskShapeVal = maskShape->GetStorageShape();

    OPS_CHECK(
        xShapeVal.GetDimNum() != DIM_NUM || maskShapeVal.GetDimNum() != DIM_NUM,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "x or mask DimNum is not 4."),
        return false);

    batch = xShapeVal.GetDim(DIM_0);
    channel = xShapeVal.GetDim(DIM_1);
    height = xShapeVal.GetDim(DIM_2);
    width = xShapeVal.GetDim(DIM_3);

    OPS_CHECK(
        width > MAX_DIM_NUM || width <= 0,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "x dim 3 should in (0, 4096]."),
        return false);

    maskBatch = maskShapeVal.GetDim(DIM_0);
    maskChannel = maskShapeVal.GetDim(DIM_1);
    uint64_t maskHeight = maskShapeVal.GetDim(DIM_2);
    uint64_t maskWidth = maskShapeVal.GetDim(DIM_3);

    OPS_CHECK(
        maskHeight != height || maskWidth != width,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "x and mask height or width not equal."),
        return false);

    OPS_CHECK(
        (batch != maskBatch && maskBatch != 1) || (channel != maskChannel && maskChannel != 1),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "x and mask batch or channel can't broadcast."),
        return false);
    return true;
}

bool ScaledMaskedSoftmaxV2Tiling::InitAttr() {
    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr, 
        OPS_LOG_E("ScaledMaskedSoftmaxV2Tiling", "attrs is null"), return false);
    scale = *attrs->GetAttrPointer<float>(ATTR_0);
    tiling.set_scale(scale);
    return true;
}

void ScaledMaskedSoftmaxV2Tiling::SetInputShape() {
    tiling.set_batch(batch);
    tiling.set_channel(channel);
    tiling.set_height(height);
    tiling.set_width(width);

    tiling.set_maskBatch(maskBatch);
    tiling.set_maskChannel(maskChannel);
    tiling.set_maskHeight(height);
    tiling.set_maskWidth(width);
}

void ScaledMaskedSoftmaxV2Tiling::SetPaddingInfo() {
    uint64_t alignedXBlock = AlignedBytes / xDtypeSize;
    uint64_t xLeft = width % alignedXBlock;
    uint64_t xPaddingNum = xLeft > 0 ? alignedXBlock - xLeft : 0u;
    tiling.set_paddingNum(xPaddingNum);
    tiling.set_padLineNum(width + xPaddingNum);

    uint64_t alignedMaskBlock = AlignedBytes / BOOL_SIZE;
    uint64_t maskLeft = width % alignedMaskBlock;
    uint64_t maskPaddingNum = maskLeft > 0 ? alignedMaskBlock - maskLeft : 0u;
    tiling.set_alignedMaskPadding(maskPaddingNum);
    tiling.set_alignedMaskWidth(width + maskPaddingNum);
}

void ScaledMaskedSoftmaxV2Tiling::SetMaskAdapt() {
    uint64_t maskMode = 0;
    if (batch != maskBatch) {
        maskMode |= BROADCAST_BATCH;
        tiling.set_nStep(0);
    } else {
        tiling.set_nStep(1);
    }
        
    if (channel != maskChannel) {
        maskMode |= BROADCAST_CHANNEL;
        tiling.set_cStep(0);
    } else {
        tiling.set_cStep(1);
    }
    tiling.set_maskMode(maskMode);
}

bool ScaledMaskedSoftmaxV2Tiling::SetUbSplitInfo() {
    uint64_t totalUbSize;
    const auto ascendCPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendCPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, totalUbSize);
    int64_t availableUbSize = totalUbSize;

    uint64_t padedWidth = tiling.get_padLineNum();
    uint64_t maskPaddedWidth = tiling.get_alignedMaskWidth();    
    uint64_t maxByteLine = padedWidth * XY_PARAMS * xDtypeSize
                           + padedWidth * FP32_SIZE
                           + maskPaddedWidth * BOOL_SIZE;
    // 高阶api使用tmpBuf复用
    uint64_t availableLinePerIter = (availableUbSize - SOFTMAX_BUF_SIZE) / maxByteLine;

    uint64_t coreNum = tiling.get_coreNum(); 
    uint64_t totalLine = batch * channel * height;
    if (coreNum == 0 || availableLinePerIter == 0) {
        return false;
    }

    uint64_t lineTailCore = totalLine / coreNum;
    uint64_t headCoreNum = totalLine % coreNum;
    uint64_t lineHeadCore = lineTailCore + 1;

    uint64_t iterHeadCore = CeilDiv(lineHeadCore, availableLinePerIter);
    uint64_t lineHeadIter = availableLinePerIter;
    uint64_t lineLastHeadIter = lineHeadCore - (iterHeadCore - 1) * lineHeadIter;

    uint64_t iterTailCore = CeilDiv(lineTailCore, availableLinePerIter);
    uint64_t lineTailIter = availableLinePerIter;
    uint64_t lineLastTailIter = lineTailCore - (iterTailCore - 1) * lineTailIter;

    tiling.set_headCoreNum(headCoreNum);
    tiling.set_lineHeadCore(lineHeadCore);
    tiling.set_iterHeadCore(iterHeadCore);
    tiling.set_lineHeadIter(lineHeadIter);
    tiling.set_lineLastHeadIter(lineLastHeadIter);

    tiling.set_lineTailCore(lineTailCore);
    tiling.set_iterTailCore(iterTailCore);
    tiling.set_lineTailIter(lineTailIter);
    tiling.set_lineLastTailIter(lineLastTailIter);
    return true;
}

void ScaledMaskedSoftmaxV2Tiling::SetSoftmaxTiling() {
    auto shape = ge::Shape({static_cast<int64_t>(tiling.get_lineHeadIter()),
                            static_cast<int64_t>(tiling.get_padLineNum())});
    auto size = AscendC::GetSoftMaxMaxTmpSize(shape, FP32_SIZE, false);
    if (size > SOFTMAX_BUF_SIZE) {
        size = SOFTMAX_BUF_SIZE;
    }
    AscendC::SoftMaxTilingFunc(shape, FP32_SIZE, size, tiling.softmaxTilingData);
}

void ScaledMaskedSoftmaxV2Tiling::SetTilingKey() {
    uint64_t dtypeKey = FP32_TILINGKEY;
    auto xDataType = context->GetInputDesc(X_INDEX)->GetDataType();
    if (xDataType == ge::DT_FLOAT16) {
        dtypeKey = FP16_TILINGKEY;
    } else if (xDataType == ge::DT_BF16) {
        dtypeKey = BF16_TILINGKEY;
    }
    uint64_t tilingKey = dtypeKey * DTYPE_DIGIT;
    context->SetTilingKey(tilingKey);
    OPS_LOG_D(context->GetNodeName(), ">>> [ScaledMaskedSoftmaxV2Tiling] tilingKey: %lu", tilingKey);
}

ge::graphStatus ScaledMaskedSoftmaxV2Tiling::DoTiling() {
    OPS_LOG_D("ScaledMaskedSoftmaxV2 tiling start");
    SetInputShape();
    SetPaddingInfo();
    SetMaskAdapt();
    if (!SetUbSplitInfo()) {
        return ge::GRAPH_FAILED;
    }
    SetSoftmaxTiling();
    SetTilingKey();

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    TilingDataPrint();
    OPS_LOG_D("ScaledMaskedSoftmaxV2 tiling end");
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingScaledMaskedSoftmaxV2(gert::TilingContext* context) {
    ScaledMaskedSoftmaxV2Tiling tilingObj(context);
    tilingObj.Init();
    return tilingObj.DoTiling();
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForScaledMaskedSoftmaxV2(gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

void ScaledMaskedSoftmaxV2Tiling::TilingDataPrint() {
    auto nodeName = context->GetNodeName();
    OPS_LOG_D(nodeName, ">>>>>>>>>>>>>>> Start to print ScaledMaskedSoftmaxV2 tiling data <<<<<<<<<<<<<<<<");
    OPS_LOG_D(nodeName, ">>> coreNum:                         %lu", tiling.get_coreNum());
    OPS_LOG_D(nodeName, ">>> batch:                           %lu", tiling.get_batch());
    OPS_LOG_D(nodeName, ">>> channel:                         %lu", tiling.get_channel());
    OPS_LOG_D(nodeName, ">>> height:                          %lu", tiling.get_height());
    OPS_LOG_D(nodeName, ">>> width:                           %lu", tiling.get_width());
    OPS_LOG_D(nodeName, ">>> maskbatch:                       %lu", tiling.get_maskBatch());
    OPS_LOG_D(nodeName, ">>> maskchannel:                     %lu", tiling.get_maskChannel());
    OPS_LOG_D(nodeName, ">>> maskheight:                      %lu", tiling.get_maskHeight());
    OPS_LOG_D(nodeName, ">>> maskwidth:                       %lu", tiling.get_maskWidth());
    OPS_LOG_D(nodeName, ">>> scale:                           %f",  tiling.get_scale());
    OPS_LOG_D(nodeName, ">>> maskMode:                        %lu", tiling.get_maskMode());
    OPS_LOG_D(nodeName, ">>> paddingNum:                      %lu", tiling.get_paddingNum());
    OPS_LOG_D(nodeName, ">>> padLineNum:                      %lu", tiling.get_padLineNum());
    OPS_LOG_D(nodeName, ">>> alignedMaskPadding:              %lu", tiling.get_alignedMaskPadding());
    OPS_LOG_D(nodeName, ">>> alignedMaskWidth:                %lu", tiling.get_alignedMaskWidth());
    OPS_LOG_D(nodeName, ">>> nStep:                           %lu", tiling.get_nStep());
    OPS_LOG_D(nodeName, ">>> cStep:                           %lu", tiling.get_cStep());
    OPS_LOG_D(nodeName, ">>> headCoreNum:                     %lu", tiling.get_headCoreNum());
    OPS_LOG_D(nodeName, ">>> lineHeadCore:                    %lu", tiling.get_lineHeadCore());
    OPS_LOG_D(nodeName, ">>> iterHeadCore:                    %lu", tiling.get_iterHeadCore());
    OPS_LOG_D(nodeName, ">>> lineHeadIter:                    %lu", tiling.get_lineHeadIter());
    OPS_LOG_D(nodeName, ">>> lineLastHeadIter:                %lu", tiling.get_lineLastHeadIter());
    OPS_LOG_D(nodeName, ">>> lineTailCore:                    %lu", tiling.get_lineTailCore());
    OPS_LOG_D(nodeName, ">>> iterTailCore:                    %lu", tiling.get_iterTailCore());
    OPS_LOG_D(nodeName, ">>> lineTailIter:                    %lu", tiling.get_lineTailIter());
    OPS_LOG_D(nodeName, ">>> lineLastTailIter:                %lu", tiling.get_lineLastTailIter());
    OPS_LOG_D(nodeName, ">>>>>>>>>>>>>>> Print ScaledMaskedSoftmaxV2 tiling data end <<<<<<<<<<<<<<<<");
}

struct ScaledMaskedSoftmaxV2CompileInfo {};

IMPL_OP_OPTILING(ScaledMaskedSoftmaxV2)
    .Tiling(TilingScaledMaskedSoftmaxV2)
    .TilingParse<ScaledMaskedSoftmaxV2CompileInfo>(TilingPrepareForScaledMaskedSoftmaxV2);
} // namespace optiling
