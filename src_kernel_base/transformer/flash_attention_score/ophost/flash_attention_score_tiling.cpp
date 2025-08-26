/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_tiling.cpp
 * \brief
 */

#include "flash_attention_score_tiling.h"
#include <queue>
#include <cmath>
#include <cfloat>
#include <register/op_impl_registry.h>
#include "tiling/data_copy_transpose_tiling.h"
#include "tiling/tiling_templates_registry.h"
#include "flash_attention_score_tiling_common.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;
constexpr size_t VALUE_INPUT_INDEX = 2;
constexpr size_t SOFTMAXSUM_OUPUT_INDEX = 1;
constexpr size_t ATTENTIONOUT_OUPUT_INDEX = 3;
constexpr size_t INPUTLAYOUT_ATTRS_INDEX = 5;
constexpr size_t MIN_COPY_UINT_SIZE = 32;

static uint32_t Ceil(uint32_t num1, uint32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

class FlashAttentionScoreEmptyInputTiling {
public:
    FlashAttentionScoreTilingData tilingData;

    void FlashAttentionScoreSetEmptyInputTilingData(gert::TilingContext *context,
                                                    FlashAttentionScoreTilingData &faTilingData);
    void GetTilingKeyAttentionScore4EmptyInput(uint32_t &tilingKey, const gert::TilingContext *context);
};

void FlashAttentionScoreEmptyInputTiling::GetTilingKeyAttentionScore4EmptyInput(uint32_t &tilingKey,
                                                                                const gert::TilingContext *context)
{
    OPS_LOG_E_IF_NULL(context, context->GetInputDesc(KEY_INPUT_INDEX), return)
    auto kernelType = context->GetInputDesc(KEY_INPUT_INDEX)->GetDataType();
    if (kernelType == ge::DT_FLOAT16) {
        tilingKey = 90;
    } else if (kernelType == ge::DT_FLOAT) {
        tilingKey = 92;
    } else {
        tilingKey = 94;
    }
}

void FlashAttentionScoreEmptyInputTiling::FlashAttentionScoreSetEmptyInputTilingData(
    gert::TilingContext *context, FlashAttentionScoreTilingData &faTilingData)
{
    OPS_LOG_E_IF_NULL(context, context->GetRawTilingData(), return)
    faTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(faTilingData.GetDataSize());
}

static ge::graphStatus CheckParams(const gert::TilingContext *context)
{
    if (context->GetInputShape(QUERY_INPUT_INDEX) != nullptr && context->GetInputShape(KEY_INPUT_INDEX) != nullptr &&
        context->GetInputShape(VALUE_INPUT_INDEX) != nullptr && context->GetAttrs() != nullptr) {
        auto &queryShape = context->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
        auto &keyShape = context->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
        auto &valueShape = context->GetInputShape(VALUE_INPUT_INDEX)->GetStorageShape();
        const char *inputLayout = context->GetAttrs()->GetAttrPointer<char>(INPUTLAYOUT_ATTRS_INDEX);
        if (strlen(inputLayout) == 3) { // 3: BSH or SBH
            OPS_ERR_IF((keyShape != valueShape), OPS_REPORT_VECTOR_INNER_ERR(context, "key or value shape is invalid"),
                       return ge::GRAPH_FAILED);
            if (inputLayout[0] == 'B') {
                // layout is BSH
                OPS_ERR_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
                           OPS_REPORT_VECTOR_INNER_ERR(context, "query or key shape is invalid"),
                           return ge::GRAPH_FAILED);
            } else {
                if (inputLayout[0] == 'T') { // TND  N1 != N2
                    return ge::SUCCESS;
                }
                // layout is SBH
                OPS_ERR_IF((queryShape.GetDim(1) != keyShape.GetDim(1)),
                           OPS_REPORT_VECTOR_INNER_ERR(context, "query or key shape is invalid"),
                           return ge::GRAPH_FAILED);
            }
        } else if (strlen(inputLayout) == 4) { // 4: layout is BNSD or BSND
            OPS_ERR_IF((keyShape != valueShape), OPS_REPORT_VECTOR_INNER_ERR(context, "key or value shape is invalid"),
                       return ge::GRAPH_FAILED);
            OPS_ERR_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
                       OPS_REPORT_VECTOR_INNER_ERR(context, "query or key shape is invalid"), return ge::GRAPH_FAILED);
            OPS_ERR_IF((queryShape.GetDim(3) != keyShape.GetDim(3)),
                       OPS_REPORT_VECTOR_INNER_ERR(context, "query or key shape is invalid"), return ge::GRAPH_FAILED);
        } else {
            OPS_LOG_W(context, "invalid input_layout[%s].", inputLayout);
            return ge::GRAPH_FAILED;
        }
        return ge::SUCCESS;
    }
    OPS_LOG_W(context, "fail to get shape or attr from context");
    return ge::GRAPH_FAILED;
}

static bool IsEmptyInput(gert::TilingContext *context)
{
    auto attenOutShape = context->GetOutputShape(ATTENTIONOUT_OUPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, attenOutShape, return false)

    auto queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return false)

    auto keyShape = context->GetInputShape(KEY_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, keyShape, return false)

    auto softmaxSumShape = context->GetOutputShape(SOFTMAXSUM_OUPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, softmaxSumShape, return false)

    int64_t attentionOutShapeSize = attenOutShape->GetStorageShape().GetShapeSize();
    int64_t queryShapeSize = queryShape->GetStorageShape().GetShapeSize();
    int64_t keyShapeSize = keyShape->GetStorageShape().GetShapeSize();
    int64_t softmaxSumShapeSize = softmaxSumShape->GetStorageShape().GetShapeSize();
    if ((queryShapeSize == 0 || keyShapeSize == 0) && (attentionOutShapeSize != 0 || softmaxSumShapeSize != 0)) {
        /* 以 MIN_COPY_UINT_SIZE 为 32Byte说明, blocks为数据的块数, blocks与coreNum存在三种关系:
          (1) blocks % coreNum == 0
             主核数量为coreNum,主核处理块数为blocks / coreNum, 最后一个核处理非32Byte对齐的数据, 尾核数量为0
          (2) blocks % coreNum != 0
              (2.1) blocks < coreNum
                    主核数量为blocks,主核处理块数为1, 最后一个核处理非32Byte对齐的数据, 尾核数量为0
              (2.2) blocks > coreNum
                    主核数量为blocks % coreNum, 尾核数量为coreNum - (blocks % coreNum), 尾核处理块数为blocks / coreNum
                    主核处理块数为blocks / coreNum + 1,最后一个尾核处理非对齐场景
        (2.2)情况如下:
        |-------------主核块-----------------|------------尾核块-----------|非对齐块|
        |                                   |                             |       |
        |                                   |                             |       |
        |--------n*(blocks/coreNum+1)-------|-----m*(blocks/coreNum)------|<32Byte|
        */
        auto kernelType = context->GetInputDesc(KEY_INPUT_INDEX)->GetDataType();
        FlashAttentionScoreEmptyInputTiling emptyInputTiling;
        auto compileInfoPtr = reinterpret_cast<const FlashAttentionScoreCompileInfo *>(context->GetCompileInfo());
        OPS_ERR_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context, "compileInfoPtr is null"),
                   return false);
        uint32_t coreNum = compileInfoPtr->aivNum;
        OPS_ERR_IF((coreNum <= 0),
                   OPS_REPORT_VECTOR_INNER_ERR(context, "platform info is invalid, coreNum=%u.", coreNum), return false);
        OPS_ERR_IF((kernelType != ge::DT_FLOAT16 && kernelType != ge::DT_FLOAT && kernelType != ge::DT_BF16),
                   OPS_REPORT_VECTOR_INNER_ERR(context, "kernelType is invalid, kernelType is %d", kernelType),
                   return false);
        uint32_t attentionOutFormerNum;          // attentionOut的主核
        uint32_t attentionOutTailNum;            // attentionOut的尾核
        uint32_t softmaxMaxFormerNum;            // softmaxMax 和 softmaxSum的主核
        uint32_t softmaxMaxTailNum;              // softmaxMax 和 softmaxSum的尾核
        uint64_t attentionOutSingleCoreDataSize; // attentionOut的每个主核处理的数据个数
        uint64_t attentionOutTailCoreDataSize;   // attentionOut的每个尾核处理的数据个数
        uint64_t softmaxMaxSingleCoreDataSize;
        uint64_t softmaxMaxTailCoreDataSize;
        uint64_t attentionOutLastCoreDataSize = 0; // 最后一个核应该处理的数据量
        uint64_t attentionOutLastCoreIndex = 0;    // 最后一个核起始地址
        uint32_t tilingKey = 0;
        uint64_t attentionOutBlockSize = 0;
        uint64_t softmaxSumBlockSize = 0;

        // 计算 MIN_COPY_UINT_SIZE 块数
        attentionOutBlockSize = Ceil(attentionOutShapeSize * ge::GetSizeByDataType(kernelType), MIN_COPY_UINT_SIZE);
        // softmaxSum 和 softmaxMax 输出为 fp32
        softmaxSumBlockSize = Ceil(softmaxSumShapeSize * ge::GetSizeByDataType(ge::DT_FLOAT), MIN_COPY_UINT_SIZE);

        if (attentionOutShapeSize != 0) {
            if (attentionOutBlockSize % coreNum == 0) {
                attentionOutTailCoreDataSize = 0;
                attentionOutFormerNum = coreNum;
                attentionOutTailNum = 0;
                attentionOutSingleCoreDataSize =
                    attentionOutBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType);
                attentionOutLastCoreDataSize =
                    attentionOutSingleCoreDataSize -
                    (attentionOutBlockSize * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType) -
                     attentionOutShapeSize);
                attentionOutLastCoreIndex = (attentionOutFormerNum - 1) * attentionOutSingleCoreDataSize;
            } else {
                attentionOutTailCoreDataSize =
                    attentionOutBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType);
                attentionOutSingleCoreDataSize =
                    attentionOutTailCoreDataSize + MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType);
                if (attentionOutBlockSize > coreNum) {
                    attentionOutFormerNum = attentionOutBlockSize % coreNum;
                    attentionOutTailNum = coreNum - attentionOutFormerNum;
                    attentionOutLastCoreIndex = attentionOutFormerNum * attentionOutSingleCoreDataSize +
                                                (attentionOutTailNum - 1) * attentionOutTailCoreDataSize;
                    attentionOutLastCoreDataSize =
                        attentionOutTailCoreDataSize -
                        (attentionOutSingleCoreDataSize * attentionOutFormerNum +
                         attentionOutTailCoreDataSize * attentionOutTailNum - attentionOutShapeSize);
                } else {
                    attentionOutFormerNum = attentionOutBlockSize;
                    attentionOutTailNum = 0;
                    attentionOutLastCoreIndex = (attentionOutFormerNum - 1) * attentionOutSingleCoreDataSize;
                    attentionOutLastCoreDataSize =
                        attentionOutSingleCoreDataSize -
                        (attentionOutFormerNum * attentionOutSingleCoreDataSize - attentionOutShapeSize);
                }
            }
        } else {
            attentionOutFormerNum = 0;
            attentionOutTailNum = 0;
            attentionOutSingleCoreDataSize = 0;
            attentionOutTailCoreDataSize = 0;
            attentionOutLastCoreDataSize = 0;
            attentionOutLastCoreIndex = 0;
        }

        if (softmaxSumBlockSize % coreNum == 0) {
            softmaxMaxSingleCoreDataSize =
                softmaxSumBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(ge::DT_FLOAT);
            softmaxMaxTailCoreDataSize = 0;
            softmaxMaxFormerNum = coreNum;
            softmaxMaxTailNum = 0;
        } else {
            if (softmaxSumBlockSize > coreNum) {
                softmaxMaxFormerNum = softmaxSumBlockSize % coreNum;
                softmaxMaxTailNum = coreNum - softmaxMaxFormerNum;
            } else {
                softmaxMaxFormerNum = softmaxSumBlockSize;
                softmaxMaxTailNum = 0;
            }
            softmaxMaxTailCoreDataSize =
                softmaxSumBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(ge::DT_FLOAT);
            softmaxMaxSingleCoreDataSize =
                softmaxMaxTailCoreDataSize + MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(ge::DT_FLOAT);
        }

        emptyInputTiling.tilingData.emptyInputTilingData.set_coreNum(coreNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutFormerNum(attentionOutFormerNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutTailNum(attentionOutTailNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_softmaxMaxFormerNum(softmaxMaxFormerNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_softmaxMaxTailNum(softmaxMaxTailNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutSingleCoreDataSize(
            attentionOutSingleCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutTailCoreDataSize(attentionOutTailCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_softmaxMaxSingleCoreDataSize(softmaxMaxSingleCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_softmaxMaxTailCoreDataSize(softmaxMaxTailCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutLastCoreDataSize(attentionOutLastCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutLastCoreIndex(attentionOutLastCoreIndex);
        emptyInputTiling.FlashAttentionScoreSetEmptyInputTilingData(context, emptyInputTiling.tilingData);
        emptyInputTiling.GetTilingKeyAttentionScore4EmptyInput(tilingKey, context);
        context->SetTilingKey(tilingKey);
        uint32_t aivActualNum =
            std::max((attentionOutFormerNum + attentionOutTailNum), (softmaxMaxFormerNum + softmaxMaxTailNum));
        context->SetBlockDim(optiling::CalcTschBlockDim(aivActualNum, 0, compileInfoPtr->aivNum));
        size_t *workspaces = context->GetWorkspaceSizes(1);
        // workspace上预留100M
        workspaces[0] = 100 * 1024 * 1024;

        return true;
    }
    return false;
}

ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionScore(gert::TilingContext *context)
{
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (IsEmptyInput(context)) {
        return ge::GRAPH_SUCCESS;
    } else {
        auto resultCode = TilingRegistry::GetInstance().DoTilingImpl(context);
        return resultCode;
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFlashAttentionScore(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OPS_LOG_E_IF(platformInfoPtr == nullptr, context, return ge::GRAPH_FAILED, "platformInfoPtr is null");

    auto compileInfoPtr = context->GetCompiledInfo<FlashAttentionScoreCompileInfo>();
    OPS_LOG_E_IF(compileInfoPtr == nullptr, context, return ge::GRAPH_FAILED, "compileInfoPtr is null");

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FlashAttentionScore)
    .Tiling(TilingFlashAttentionScore)
    .TilingInputsDataDependency({7, 8, 9, 10, 11})
    .TilingParse<FlashAttentionScoreCompileInfo>(TilingPrepareForFlashAttentionScore);  // 向框架注册入口函数

} // namespace optiling
