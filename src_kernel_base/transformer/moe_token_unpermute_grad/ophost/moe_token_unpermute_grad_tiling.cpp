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
 * \file moe_token_unpermute_grad_tiling.cpp
 * \brief
 */
#include <iostream>
#include "moe_token_unpermute_grad_tiling.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE_32 = 32;
constexpr uint32_t BLOCK_SIZE_512 = 512;
constexpr uint32_t TOPK_MAX_NUM = 512;
constexpr uint32_t REDUCESUM_ONEREPEAT_CALNUM = static_cast<uint32_t>(256 / sizeof(float));
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t INDICES_RESERVE_MAX_NUM = 256;
constexpr size_t INPUT_PERMUTED_TOKENS_IDX = 0;
constexpr size_t INPUT_UNPERMUTEDOUTPUTD_IDX = 1;
constexpr size_t INPUT_ROWIDMAP_IDX = 2;
constexpr size_t INPUT_PROB_IDX = 3;
constexpr size_t ATTR_PADDEDMODE_IDX = 0;
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;

template<typename T>
inline T AlignUp(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

template<typename T>
inline T AlignDown(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : ((num) / (rnd) * (rnd)));
}

template<typename T>
inline T CeilDiv(T num, T div)
{
    return (((div) == 0) ? 0 : (((num) + (div) - 1) / (div)));
}

inline int64_t GetDiv(int64_t num, int64_t div)
{
    return div == 0 ? 0 : num / div;
}

inline int64_t GetRem(int64_t num, int64_t div)
{
    return div == 0 ? 0 : num % div;
}

inline uint32_t GetLengthByType(int32_t dtype) {
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
        case ge::DT_BF16:
            return sizeof(int16_t);
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return sizeof(int32_t);
        case ge::DT_DOUBLE:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            return sizeof(int64_t);
        default:
            return 0;
    }
}

inline int64_t GetGreatestDivisor(int64_t num, int64_t div) {
    if (div == 0) {
        return div;
    }
    while (num % div != 0) {
        div--;
    }
    return div;
}

static void MoeTokenUnpermuteGradPrintParam(gert::TilingContext* context, MoeTokenUnpermuteGradTilingData& tiling) {
    OPS_LOG_D(context, ">>>>>>>>>>>>>>> Start to print MoeTokenUnpermuteGrad tiling data <<<<<<<<<<<<<<<<");
    OPS_LOG_D(context, ">>> tokensNum:                %ld", tiling.get_tokensNum());
    OPS_LOG_D(context, ">>> tokensNum:                %ld", tiling.get_tokensNum());
    OPS_LOG_D(context, ">>> topK:                     %ld", tiling.get_topK());
    OPS_LOG_D(context, ">>> hiddenSize:               %ld", tiling.get_hiddenSize());
    OPS_LOG_D(context, ">>> numOutTokens:             %ld", tiling.get_numOutTokens());
    OPS_LOG_D(context, ">>> formerCoreNum:            %ld", tiling.get_formerCoreNum());
    OPS_LOG_D(context, ">>> tailCoreNum:              %ld", tiling.get_tailCoreNum());
    OPS_LOG_D(context, ">>> tokenNumEachCore:         %ld", tiling.get_tokenNumEachCore());
    OPS_LOG_D(context, ">>> tokenNumTailCore:         %ld", tiling.get_tokenNumTailCore());
    OPS_LOG_D(context, ">>> rowIdMapEachCore:         %ld", tiling.get_rowIdMapEachCore());
    OPS_LOG_D(context, ">>> rowIdMapTailCore:         %ld", tiling.get_rowIdMapTailCore());
    OPS_LOG_D(context, ">>>>>>>>>>>>>>>       in core information      <<<<<<<<<<<<<<<<");
    OPS_LOG_D(context, ">>> hiddenSizeAlign:          %ld", tiling.get_hiddenSizeAlign());
    OPS_LOG_D(context, ">>> hiddenSizeLoopTimes:      %ld", tiling.get_hiddenSizeLoopTimes());
    OPS_LOG_D(context, ">>> hiddenSizeTail:           %ld", tiling.get_hiddenSizeTail());
    OPS_LOG_D(context, ">>> inputReserveNum:          %ld", tiling.get_inputReserveNum());
    OPS_LOG_D(context, ">>> indicesReserveNum:        %ld", tiling.get_indicesReserveNum());
    OPS_LOG_D(context, ">>> indicesReserveNumAlign:   %ld", tiling.get_indicesReserveNumAlign());
    OPS_LOG_D(context, ">>> totalUbSize:              %ld", tiling.get_totalUbSize());
    OPS_LOG_D(context, ">>>>>>>>>>>>>>> Print MoeTokenUnpermuteGrad tiling data end <<<<<<<<<<<<<<<<");
}


// 核间切分策略
static void MoeTokenUnpermuteGradInitSplitInfo(const gert::TilingContext* context, MoeTokenUnpermuteGradTilingData& tiling) {
    int64_t tokensNum = tiling.get_tokensNum();
    int64_t topK = tiling.get_topK();
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t totalUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, totalUbSize);
    int64_t ubSize = static_cast<int64_t>(totalUbSize);

    // formerCoreNum个核是多1的，tailCoreNum个核是少1的
    int64_t tokenNumTailCore = GetDiv(tokensNum, static_cast<int64_t>(totalCoreNum));
    int64_t formerCoreNum = GetRem(tokensNum, static_cast<int64_t>(totalCoreNum));
    int64_t tokenNumEachCore = formerCoreNum == 0 ? tokenNumTailCore : tokenNumTailCore + 1;
    int64_t tailCoreNum = totalCoreNum - formerCoreNum;
    int64_t rowIdMapEachCore = tokenNumEachCore * topK;
    int64_t rowIdMapTailCore = tokenNumTailCore * topK;
    tiling.set_formerCoreNum(formerCoreNum);
    tiling.set_tailCoreNum(tailCoreNum);
    tiling.set_tokenNumEachCore(tokenNumEachCore);
    tiling.set_tokenNumTailCore(tokenNumTailCore);
    tiling.set_rowIdMapEachCore(rowIdMapEachCore);
    tiling.set_rowIdMapTailCore(rowIdMapTailCore);
    tiling.set_totalUbSize(ubSize);
}

static bool CoreSplitInfoProbIsNone(MoeTokenUnpermuteGradTilingData& tiling, uint32_t inputTypeLength, uint32_t rowIdMapTypeLength, uint64_t totalUbSize) {
    int64_t hiddenSize = tiling.get_hiddenSize();
    uint32_t inputBlockAlignEleNum = BLOCK_SIZE_32 / inputTypeLength;
    uint32_t rowIdMapBlockAlignEleNum = BLOCK_SIZE_32 / rowIdMapTypeLength;

    int64_t hiddenSizeAlign = AlignUp<int64_t>(hiddenSize, static_cast<int64_t>(inputBlockAlignEleNum)); // h=hiddensize,全载
    int64_t indicesReserveNum = totalUbSize / (BUFFER_NUM * (hiddenSizeAlign * inputTypeLength + rowIdMapTypeLength)); // 剩余空间全部尽可能分满,inque(indicesNumPerLoop,h),indices(indicesNumPerLoop,)
    indicesReserveNum = AlignDown<int64_t>(indicesReserveNum, static_cast<int64_t>(rowIdMapBlockAlignEleNum)); // indicesNumPerLoop 32B对齐
    indicesReserveNum = std::max(indicesReserveNum, static_cast<int64_t>(rowIdMapBlockAlignEleNum)); // indicesNumPerLoop<32B的边界值处理,最小保证32B对齐
    if (indicesReserveNum <= 0) {
        return false;
    }
    int64_t hiddenSizeTmpMax = (totalUbSize - indicesReserveNum * BUFFER_NUM * rowIdMapTypeLength) / (BUFFER_NUM * inputTypeLength * indicesReserveNum);
    if (hiddenSizeTmpMax < hiddenSizeAlign) { // hiddensize需要切分
        hiddenSizeAlign = AlignDown<int64_t>(hiddenSizeTmpMax, static_cast<int64_t>(inputBlockAlignEleNum)); // hiddenSize很大，一定不为0
    }
    int64_t hiddenSizeLoopTimes = (hiddenSize + hiddenSizeAlign - 1) / hiddenSizeAlign;
    int64_t hiddenSizeTail = hiddenSize - (hiddenSizeLoopTimes - 1) * hiddenSizeAlign;
    tiling.set_hiddenSizeAlign(hiddenSizeAlign);
    tiling.set_hiddenSizeLoopTimes(hiddenSizeLoopTimes);
    tiling.set_hiddenSizeTail(hiddenSizeTail);
    tiling.set_inputReserveNum(indicesReserveNum); // permutedTokenNumPerLoop = indicesNumPerLoop
    tiling.set_indicesReserveNum(indicesReserveNum); // indicesNumPerLoop
    tiling.set_indicesReserveNumAlign(indicesReserveNum); // indicesNumPerLoopAlign
    return true;
}

static bool CoreSplitInfoProbIsNotNone(MoeTokenUnpermuteGradTilingData& tiling, uint32_t inputTypeLength,uint32_t probTypeLength, uint64_t totalUbSize) {
    int64_t hiddenSize = tiling.get_hiddenSize();
    int64_t topK = tiling.get_topK();
    uint32_t inputBlock512AlignEleNum = BLOCK_SIZE_512 / inputTypeLength;
    uint32_t inputBlock32AlignEleNum = BLOCK_SIZE_32 / probTypeLength;
    uint32_t fp32TypeLength = sizeof(float);

    int64_t hiddenSizeAlign = AlignUp<int64_t>(hiddenSize, static_cast<int64_t>(inputBlock512AlignEleNum)); // h=hiddensize,全载
    int64_t indicesReserveNumMax = tiling.get_tokenNumEachCore() * topK;
    int64_t indicesReserveNum = 1;
    if (topK <= INDICES_RESERVE_MAX_NUM) { // indicesReserveNum最大预留256个元素，占据内存BUFFER_NUM*(4*n+2*n)+4*n=(BUFFER_NUM * 6 + 4)*n
        indicesReserveNum = std::min(indicesReserveNumMax, AlignDown<int64_t>(INDICES_RESERVE_MAX_NUM, topK)); // indicesNumPerLoop
    } else { // indicesReserveNum最大预留topK个元素
        indicesReserveNum = topK; // indicesNumPerLoop
    }
    int64_t indicesReserveNumAlign = AlignUp<int64_t>(indicesReserveNum, static_cast<int64_t>(inputBlock32AlignEleNum)); // indicesNumPerLoopAlign
    int64_t inputReserveNumMax = static_cast<int64_t>(totalUbSize - BUFFER_NUM * probTypeLength * indicesReserveNumAlign * 2 -
        indicesReserveNumAlign * fp32TypeLength * 3 - 256 - (BUFFER_NUM * inputTypeLength * 2 + 3 * fp32TypeLength) * hiddenSizeAlign) / static_cast<int64_t>(BUFFER_NUM * inputTypeLength * hiddenSizeAlign);
    int64_t inputReserveNum = 1;
    int64_t hiddenSizeLoopTimes = 1;
    if (inputReserveNumMax < 1) {
        // 切hiddensize
        OPS_LOG_D("MoeTokenUnpermuteGradTiling, hiddensize need to be split.");
        int64_t hiddenSizeLoopMax = (totalUbSize - BUFFER_NUM * probTypeLength * indicesReserveNumAlign * 2 -
            indicesReserveNumAlign * fp32TypeLength * 3 - 256) / (3 * BUFFER_NUM * inputTypeLength + 3 * fp32TypeLength); // 一个切片最大能载入的hidden_size大小
        hiddenSizeLoopMax = AlignDown<int64_t>(hiddenSizeLoopMax, static_cast<int64_t>(inputBlock512AlignEleNum)); // 512B能存放元素数量对齐
        if (hiddenSizeLoopMax == 0) {
            OPS_LOG_D("MoeTokenUnpermuteGradTiling tiling error, hiddenSizeLoopMax == 0");
            return false;
        }
        hiddenSizeLoopTimes = (hiddenSize + hiddenSizeLoopMax -1) / hiddenSizeLoopMax; // 切几次
        hiddenSize = hiddenSize - (hiddenSizeLoopTimes - 1) * hiddenSizeLoopMax; // 尾块大小
        hiddenSizeAlign = hiddenSizeLoopMax; // 非尾块的对齐大小
    } else { // indicesNumPerLoop是topK的倍数，topK是permutedTokenNumPerLoop的倍数
        OPS_LOG_D("MoeTokenUnpermuteGradTiling, hiddensize is not split.");
        inputReserveNumMax = std::min(inputReserveNumMax, topK);
        inputReserveNum = GetGreatestDivisor(topK, inputReserveNumMax);
    }
    if (indicesReserveNum == 0 || inputReserveNum == 0) {
        OPS_LOG_D("MoeTokenUnpermuteGradTiling tiling error, indicesReserveNum == 0 or inputReserveNum == 0");
        return false;
    }
    tiling.set_hiddenSizeAlign(hiddenSizeAlign);
    tiling.set_hiddenSizeLoopTimes(hiddenSizeLoopTimes);
    tiling.set_hiddenSizeTail(hiddenSize);
    tiling.set_inputReserveNum(inputReserveNum); // permutedTokenNumPerLoop
    tiling.set_indicesReserveNum(indicesReserveNum); // indicesNumPerLoop
    tiling.set_indicesReserveNumAlign(indicesReserveNumAlign); // indicesNumPerLoopAlign
    return true;
}

// 核内切分策略
static ge::graphStatus MoeTokenUnpermuteGradCoreSplitInfo(const gert::TilingContext* context, MoeTokenUnpermuteGradTilingData& tiling) {
    auto tokensDtype = context->GetInputDesc(INPUT_PERMUTED_TOKENS_IDX)->GetDataType();
    uint32_t tokensTypeLength = GetLengthByType(tokensDtype); // sizeof(bfloat16)

    auto probTensor = context->GetOptionalInputTensor(INPUT_PROB_IDX);
    auto probDtype = tokensDtype;
    uint32_t probTypeLength = tokensTypeLength;
    if (probTensor != nullptr) {
        probDtype = context->GetInputDesc(INPUT_PROB_IDX)->GetDataType();
        probTypeLength = GetLengthByType(probDtype);
    }

    auto rowIdMapDtype = context->GetInputDesc(INPUT_ROWIDMAP_IDX)->GetDataType();
    uint32_t rowIdMapTypeLength = GetLengthByType(rowIdMapDtype); // sizeof(int32)
    if (tokensTypeLength == 0 || rowIdMapTypeLength == 0 || probTypeLength == 0) {
        OPS_LOG_E(context->GetNodeName(), "[MoeTokenUnpermuteGrad] input or rowIdMap GetLengthByType is: 0.");
        return ge::GRAPH_FAILED;
    }

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t totalUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, totalUbSize);

    if (probTensor == nullptr) { // prob为None
        if (!CoreSplitInfoProbIsNone(tiling, tokensTypeLength, rowIdMapTypeLength, totalUbSize)) {
            OPS_LOG_E(context->GetNodeName(), "[MoeTokenUnpermuteGrad] In this case prob is None, split in core operation illegal.");
            return ge::GRAPH_FAILED;
        }
    } else { // prob非None
        if (!CoreSplitInfoProbIsNotNone(tiling, tokensTypeLength, probTypeLength, totalUbSize)) {
            OPS_LOG_E(context->GetNodeName(), "[MoeTokenUnpermuteGrad] In this case prob is not None, split in core operation illegal.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus Tiling4MoeTokenUnpermuteGrad(gert::TilingContext* context) {
    OPS_LOG_D("MoeTokenUnpermuteGradTiling tiling start");
    MoeTokenUnpermuteGradTilingData tiling;
    const gert::StorageShape* permutedTokensShape = context->GetInputShape(INPUT_PERMUTED_TOKENS_IDX);
    const gert::StorageShape* unpermutedOutputDShape = context->GetInputShape(INPUT_UNPERMUTEDOUTPUTD_IDX);
    OPS_LOG_E_IF_NULL(context, permutedTokensShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, unpermutedOutputDShape, return ge::GRAPH_FAILED);
    auto probTensor = context->GetOptionalInputTensor(INPUT_PROB_IDX);
    const gert::StorageShape* probShape = (probTensor == nullptr)? nullptr : context->GetOptionalInputShape(INPUT_PROB_IDX);
    int64_t tokensNum = unpermutedOutputDShape->GetStorageShape().GetDim(DIM_0);
    int64_t hiddenSize = unpermutedOutputDShape->GetStorageShape().GetDim(DIM_1);
    int64_t topK = (probShape == nullptr)? 1 : probShape->GetStorageShape().GetDim(DIM_1);
    int64_t numOutTokens = permutedTokensShape->GetStorageShape().GetDim(DIM_0);
    tiling.set_tokensNum(tokensNum);
    tiling.set_topK(topK);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_numOutTokens(numOutTokens);
    if (tokensNum == 0 || topK == 0 || hiddenSize == 0 || numOutTokens == 0) {
        OPS_LOG_E(context->GetNodeName(), "[MoeTokenUnpermuteGrad] input shape has 0.");
        return ge::GRAPH_FAILED;
    }
    if (topK > TOPK_MAX_NUM) {
        OPS_LOG_E(context->GetNodeName(), "[MoeTokenUnpermuteGrad] topK only support no greater than 512.");
        return ge::GRAPH_FAILED;
    }
    
    MoeTokenUnpermuteGradInitSplitInfo(context, tiling); // 核间切分
    OPS_LOG_D("MoeTokenUnpermuteGradTiling MoeTokenUnpermuteGradInitSplitInfo finished");
    MoeTokenUnpermuteGradCoreSplitInfo(context, tiling); // 核内切分
    OPS_LOG_D("MoeTokenUnpermuteGradTiling MoeTokenUnpermuteGradCoreSplitInfo finished");
    MoeTokenUnpermuteGradPrintParam(context, tiling);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(totalCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    
    auto attrPtr = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrPtr, return ge::GRAPH_FAILED);
    const bool* paddedModePtr = attrPtr->GetAttrPointer<bool>(ATTR_PADDEDMODE_IDX);
    OPS_LOG_E_IF_NULL(context, paddedModePtr, return ge::GRAPH_FAILED);
    bool paddedMode = *paddedModePtr;
    OPS_LOG_D(context, ">>> [MoeTokenUnpermuteGradTiling] paddedMode: %d", paddedMode);
    uint32_t paddedModeKey = 0;
    uint32_t probKey = (probTensor == nullptr)? 0 : 1;
    uint32_t tilingKey = paddedModeKey * 10 + probKey;
    // 0: padded_mode = False, 不存在prob
    // 1: padded_mode = False, 存在prob
    // 10: padded_mode = True, 不存在prob
    // 11: padded_mode = True, 存在prob
    context->SetTilingKey(tilingKey);
    OPS_LOG_D(context, ">>> [MoeTokenUnpermuteGradTiling] tilingKey: %u", tilingKey);
    OPS_LOG_D("MoeTokenUnpermuteGradTiling tiling end");
    return ge::GRAPH_SUCCESS; 
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepare4MoeTokenUnpermuteGrad(gert::TilingParseContext* context) {
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct MoeTokenUnpermuteGradCompileInfo {};

IMPL_OP_OPTILING(MoeTokenUnpermuteGrad)
    .Tiling(Tiling4MoeTokenUnpermuteGrad)
    .TilingParse<MoeTokenUnpermuteGradCompileInfo>(TilingPrepare4MoeTokenUnpermuteGrad);

}  // namespace optiling