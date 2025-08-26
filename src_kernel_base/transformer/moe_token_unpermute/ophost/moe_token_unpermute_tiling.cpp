/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_token_unpermute_tiling.cpp
 * \brief
 */
#include "moe_token_unpermute_tiling.h"

namespace optiling {

ASCENDC_EXTERN_C ge::graphStatus TilingMoeTokenUnpermute(gert::TilingContext *context)
{
    return TilingCompute(context, -1);
}

static inline int64_t AlignN(const int64_t x, const int64_t N)
{
    return (x + N - 1) & ~(N - 1);
}

static inline int64_t GetLengthByType(const int32_t dtype)
{
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

static inline int64_t safeMod(const int64_t a, const int64_t b)
{
    return b == 0 ? 0 : a % b;
}

static inline int64_t safeDiv(const int64_t a, const int64_t b)
{
    return b == 0 ? 0 : a / b;
}

static inline bool isFloatDtype(const int64_t inputDtypeSize)
{
    return inputDtypeSize == GetLengthByType(ge::DT_FLOAT);
}

/*
 * 计算hidden_size=1所需要的Btye
 */
static inline int64_t ComputeUnitHSpace(const int64_t inputDtypeSize, const int64_t bufferNum)
{
    int64_t castNum = isFloatDtype(inputDtypeSize) ? 0 : CAST_NUM;
    return inputDtypeSize * (QUE_NUM + bufferNum - 1) + FLOAT_DATA_SIZE * castNum;
}

static inline int64_t ComputeMaxHiddenSize(MoeTokenUnpermuteParam &param, int64_t bufferNum)
{
    // sorted_indices和probs的预留空间；topK_num为最大值512时，至少需要5120 Btye。
    const int64_t reserveSpace = 5120;
    int64_t maxHiddenSize =
        safeDiv((param.core.maxCoreMemery - reserveSpace), ComputeUnitHSpace(param.input.tokensDtypeSize, bufferNum));

    return AlignN(maxHiddenSize - ALIGN_512, ALIGN_512);
}

static inline void Init(gert::TilingContext *context, const int64_t topK, MoeTokenUnpermuteParam &param)
{
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    auto ascendPlaform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    param.core.maxCoreNum = static_cast<int64_t>(ascendPlaform.GetCoreNumAiv());
    uint64_t maxCoreMemery;
    ascendPlaform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxCoreMemery);
    param.core.maxCoreMemery = static_cast<int64_t>(maxCoreMemery);

    const gert::StorageShape *tokensShape = context->GetInputShape(0);
    const gert::StorageShape *sortedIndicesShape = context->GetInputShape(1);
    const gert::StorageShape *probsShape = context->GetInputShape(2);
    auto dataTensor0 = context->GetInputTensor(0);
    auto dataTensor1 = context->GetInputTensor(1);

    param.input.tokensDtypeSize = GetLengthByType(dataTensor0->GetDataType());
    param.input.indicesDtypeSize = GetLengthByType(dataTensor1->GetDataType());
    param.input.numOutTokens = tokensShape->GetStorageShape().GetDim(0); // numOutTokens根据tokens第0维获取；
    param.input.hiddenSize = tokensShape->GetStorageShape().GetDim(1);
    param.input.totalLength = sortedIndicesShape->GetStorageShape().GetDim(
        0); // tokens可能存在numOutTokens，因此totalLength从sortedIndices获取。
    if (probsShape != nullptr) {
        auto dataTensor2 = context->GetInputTensor(2);
        param.input.probsDtypeSize = GetLengthByType(dataTensor2->GetDataType());
        param.input.haveProbs = true;
        param.input.tokensNum = probsShape->GetStorageShape().GetDim(0);
        param.input.topK = probsShape->GetStorageShape().GetDim(1);
    } else {
        param.input.topK = topK == -1 ? 1 : topK;
        param.input.tokensNum = safeDiv(param.input.totalLength, param.input.topK);
    }
}

static void SetCoreNum(MoeTokenUnpermuteParam &param)
{
    if (param.input.tokensNum < param.core.maxCoreNum) {
        param.core.usedCoreNum = param.input.tokensNum;
    } else {
        param.core.usedCoreNum = param.core.maxCoreNum;
    }
}

static inline void TilingHiddenSize(MoeTokenUnpermuteParam &param)
{
    int64_t maxHiddenSize = ComputeMaxHiddenSize(param, MIN_BUFFER_NUM);
    if (AlignN(param.input.hiddenSize, ALIGN_512) <= maxHiddenSize) {
        param.hiddenTiling.length = param.input.hiddenSize;
        param.hiddenTiling.remain = 0;
        param.hiddenTiling.num = 1;
    } else {
        param.hiddenTiling.length = maxHiddenSize;
        param.hiddenTiling.remain = safeMod(param.input.hiddenSize, maxHiddenSize);
        param.hiddenTiling.num = safeDiv(param.input.hiddenSize, maxHiddenSize);
    }
}

static inline void SetBufferNum(MoeTokenUnpermuteParam &param)
{
    const int64_t maxBufferNum = 4;
    int64_t bufferNum = maxBufferNum;
    while (bufferNum > MIN_BUFFER_NUM && param.hiddenTiling.length > ComputeMaxHiddenSize(param, bufferNum)) {
        bufferNum--;
    }
    param.core.bufferNum = bufferNum;
}

static inline void ComputeRemainMemerySpace(MoeTokenUnpermuteParam &param)
{
    param.core.remainMemerySpace =
        param.core.maxCoreMemery - AlignN(param.hiddenTiling.length, ALIGN_512) *
                                       ComputeUnitHSpace(param.input.tokensDtypeSize, param.core.bufferNum);
    param.core.remainMemerySpace -= ALIGN_256;
}

static inline void TilingToken(MoeTokenUnpermuteParam &param)
{
    param.tokenPerCore.length = safeDiv(param.input.tokensNum, param.core.usedCoreNum);
    param.tokenPerCore.num = param.core.usedCoreNum;
    param.tokenPerCore.remain = safeMod(param.input.tokensNum, param.core.usedCoreNum);

    int64_t unitTokenSpace = param.input.indicesDtypeSize;
    if (param.input.haveProbs) {
        unitTokenSpace += param.input.probsDtypeSize;
        if (!isFloatDtype(param.input.probsDtypeSize)) {
            unitTokenSpace += sizeof(float);
        }
    }

    int64_t probIndiceSpace = param.tokenPerCore.length * param.input.topK * unitTokenSpace;

    if (param.core.remainMemerySpace >= probIndiceSpace) {
        param.tokenTiling.length = param.tokenPerCore.length;
        param.tokenTiling.remain = 0;
        param.tokenTiling.num = 1;
    } else {
        int64_t maxTokenSize = safeDiv(param.core.remainMemerySpace, (param.input.topK * unitTokenSpace));
        param.tokenTiling.length = maxTokenSize;
        param.tokenTiling.remain = safeMod(param.tokenPerCore.length, maxTokenSize);
        param.tokenTiling.num = safeDiv(param.tokenPerCore.length, maxTokenSize);
    }
}
/*
  tilingKey计算规则
  第0位：
    0表示probs为None，1表示prob非None。
  第1-2位:
    00 表示bfloat16数据类型;
    01 表示float16数据类型;
    10 表示float32数据类型。
  第3-4位（mix位，probs与与permuted_tokens数据类型不一致时生效）：
    00 表示probs不存在，或probs与permuted_tokens数据类型保持一致;
    01 表示probs数据类型为bfloat16数据类型;
    10 表示probs数据类型为float16数据类型;
    11 表示probs数据类型为float32数据类型。
 */
static inline void SetTilingKey(const gert::TilingContext *context, MoeTokenUnpermuteParam &param)
{
    auto permuted_tokens_dtype = context->GetInputTensor(0)->GetDataType();
    if (permuted_tokens_dtype == ge::DT_FLOAT16) {
        param.core.tilingKey += TILINGKEY_FLOAT16;
    } else if (permuted_tokens_dtype == ge::DT_FLOAT) {
        param.core.tilingKey += TILINGKEY_FLOAT;
    }
    if (param.input.haveProbs) {
        // 存在probs
        param.core.tilingKey += TILINGKEY_PROBS;
        auto probs_dtype = context->GetInputTensor(2)->GetDataType();
        if (permuted_tokens_dtype != probs_dtype) {
            // 支持混合精度类型
            if (probs_dtype == ge::DT_BF16) {
                param.core.tilingKey += TILINGKEY_MIX_BF16;
            } else if (probs_dtype == ge::DT_FLOAT16) {
                param.core.tilingKey += TILINGKEY_MIX_FP16;
            } else if (probs_dtype == ge::DT_FLOAT) {
                param.core.tilingKey += TILINGKEY_MIX_FP32;
            }
        }
    }
}

static inline void SetTilingData(gert::TilingContext *context, const MoeTokenUnpermuteParam &param)
{
    MoeTokenUnpermuteTilingData tilingData;
    tilingData.set_hidden_size(param.input.hiddenSize);
    tilingData.set_top_k(param.input.topK);
    tilingData.set_num_out_tokens(param.input.numOutTokens);
    tilingData.set_hidden_splited_length(param.hiddenTiling.length);
    tilingData.set_hidden_splited_num(param.hiddenTiling.num);
    tilingData.set_hidden_splited_remain(param.hiddenTiling.remain);
    tilingData.set_tokens_core_length(param.tokenPerCore.length);
    tilingData.set_tokens_core_remain(param.tokenPerCore.remain);
    tilingData.set_tokens_splited_length(param.tokenTiling.length);
    tilingData.set_tokens_splited_remain(param.tokenTiling.remain);
    tilingData.set_tokens_splited_num(param.tokenTiling.num);
    tilingData.set_buffer_num(param.core.bufferNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(param.core.usedCoreNum);
}

ge::graphStatus TilingCompute(gert::TilingContext *context, const int64_t topK)
{
    MoeTokenUnpermuteParam param;

    Init(context, topK, param);
    SetCoreNum(param);
    TilingHiddenSize(param);
    SetBufferNum(param);
    ComputeRemainMemerySpace(param);
    TilingToken(param);
    SetTilingKey(context, param);
    SetTilingData(context, param);
    return context->SetTilingKey(param.core.tilingKey);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForMoeTokenUnpermute(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct MoeTokenUnpermuteCompileInfo {};

IMPL_OP_OPTILING(MoeTokenUnpermute)
    .Tiling(TilingMoeTokenUnpermute)
    .TilingParse<MoeTokenUnpermuteCompileInfo>(TilingPrepareForMoeTokenUnpermute);
} // namespace optiling
