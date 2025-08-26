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
 * \file flash_attention_score_grad_tiling.cpp
 * \brief
 */

#include "flash_attention_score_grad_tiling.h"
#include <register/op_impl_registry.h>
#include "tiling/data_copy_transpose_tiling.h"
#include "tiling/tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
constexpr uint32_t OUTPUT_IDX_DQ = 0;
constexpr uint32_t OUTPUT_IDX_DK = 1;
constexpr uint32_t OUTPUT_IDX_DV = 2;
constexpr uint32_t OUTPUT_IDX_DPSE = 3;

constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t DY_INPUT_INDEX = 3;
constexpr uint32_t SOFTMAX_MAX = 8;
constexpr uint32_t SOFTMAX_SUM = 9;
constexpr uint32_t ATTENTION_IN = 11;

constexpr uint32_t HEAD_NUM_IDX =4;
constexpr uint32_t LAYOUT_ATTR_IDX = 5;

constexpr uint32_t FAG_EMPTY_TILING_KEY = 90;

static uint32_t CalculateTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
{
    uint32_t ration;
    if (aicCoreNum == 0 || aivCoreNum == 0 || aicCoreNum > aivCoreNum) {
        return sliceNum;
    }
    ration = aivCoreNum / aicCoreNum;
    return (sliceNum + (ration - 1)) / ration;
}

// tiling func + tiling prepare
class FlashAttentionScoreGradTiling {
public:
    FlashAttentionScoreGradTilingData tilingData;
    FlashAttentionScoreGradTiling(){};

    ge::graphStatus RunEmptyTiling(gert::TilingContext *context)
    {
        uint64_t aicNum = 40; // 40: B3 default aicNum
        uint64_t aivNum = 20; // 20: B3 default aivNum
        auto platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compilePtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo *>(context->GetCompileInfo());
            OPS_ERR_IF(compilePtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context, "compile_info is null"),
                       return ge::GRAPH_FAILED);
            aivNum = compilePtr->aivNum;
            aicNum = compilePtr->aicNum;
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            aicNum = ascendcPlatform.GetCoreNumAic();
            aivNum = ascendcPlatform.GetCoreNumAiv();
        }
        OPS_ERR_IF(aivNum == 0, OPS_REPORT_VECTOR_INNER_ERR("flash_attention_score_grad", "num of aiv is 0."),
                   return GRAPH_FAILED);
        uint64_t dqNum = context->GetOutputShape(OUTPUT_IDX_DQ)->GetStorageShape().GetShapeSize();
        if (dqNum % aivNum == 0) {
            tilingData.emptyTensorTilingData.set_formerDqNum(aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDqNum(dqNum / aivNum);
            tilingData.emptyTensorTilingData.set_tailCoreDqNum(0);
        } else {
            tilingData.emptyTensorTilingData.set_formerDqNum(dqNum % aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDqNum(dqNum / aivNum + 1);
            tilingData.emptyTensorTilingData.set_tailCoreDqNum(dqNum / aivNum);
        }
        uint64_t dkNum = context->GetOutputShape(OUTPUT_IDX_DK)->GetStorageShape().GetShapeSize();
        if (dkNum % aivNum == 0) {
            tilingData.emptyTensorTilingData.set_formerDkNum(aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDkNum(dkNum / aivNum);
            tilingData.emptyTensorTilingData.set_tailCoreDkNum(0);
        } else {
            tilingData.emptyTensorTilingData.set_formerDkNum(dkNum % aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDkNum(dkNum / aivNum + 1);
            tilingData.emptyTensorTilingData.set_tailCoreDkNum(dkNum / aivNum);
        }
        const gert::StorageShape *dpseShape = context->GetOutputShape(OUTPUT_IDX_DPSE);
        uint64_t dpseNum = (dpseShape == nullptr) ? 0 : dpseShape->GetStorageShape().GetShapeSize();
        if (dpseNum % aivNum == 0) {
            tilingData.emptyTensorTilingData.set_formerDpseNum(aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDpseNum(dpseNum / aivNum);
            tilingData.emptyTensorTilingData.set_tailCoreDpseNum(0);
        } else {
            tilingData.emptyTensorTilingData.set_formerDpseNum(dpseNum % aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDpseNum(dpseNum / aivNum + 1);
            tilingData.emptyTensorTilingData.set_tailCoreDpseNum(dpseNum / aivNum);
        }

        context->SetTilingKey(FAG_EMPTY_TILING_KEY);
        auto sliceNum =
            (dqNum < aivNum && dkNum < aivNum && dpseNum < aivNum) ? std::max(std::max(dqNum, dkNum), dpseNum) : aivNum;
        context->SetBlockDim(CalculateTschBlockDim(sliceNum, aicNum, aivNum));
        size_t *workspaces = context->GetWorkspaceSizes(1);
        workspaces[0] = 100 * 1024 * 1024;
        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
};

static bool IsEmptyOutput(gert::TilingContext *context)
{
    const gert::StorageShape *dqShape = context->GetOutputShape(OUTPUT_IDX_DQ);
    const gert::StorageShape *dkShape = context->GetOutputShape(OUTPUT_IDX_DK);
    const gert::StorageShape *dvShape = context->GetOutputShape(OUTPUT_IDX_DV);
    if (dqShape != nullptr && dkShape != nullptr && dvShape != nullptr) {
        if (dqShape->GetStorageShape().GetShapeSize() == 0 || dkShape->GetStorageShape().GetShapeSize() == 0 ||
            dvShape->GetStorageShape().GetShapeSize() == 0) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus CheckAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED)
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto keepProbPtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<uint32_t>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    size_t *workspaces = context->GetWorkspaceSizes(1);

    OPS_LOG_E_IF_NULL(context, scaleValuePtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context, keepProbPtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context, preTokensPtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context, nextTokensPtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context, n1SizePtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context, inputLayoutPtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context, workspaces, return ge::GRAPH_FAILED)
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckBaseInput(gert::TilingContext *context){
    auto &queryShape = context->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto &keyShape = context->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto &valueShape = context->GetInputShape(VALUE_INPUT_INDEX)->GetStorageShape();
    int64_t headNum = *context->GetAttrs()->GetAttrPointer<int>(HEAD_NUM_IDX);
    OPS_ERR_IF(headNum == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context, "headNum is 0."),
               return ge::GRAPH_FAILED);
    const char *inputLayout = context->GetAttrs()->GetAttrPointer<char>(LAYOUT_ATTR_IDX);
    if (strlen(inputLayout) == 3) { // 3: BSH or SBH or TND
        OPS_LOG_E_IF(keyShape != valueShape, context, return ge::GRAPH_FAILED, "key or value shape is invalid");
        if (inputLayout[0] == 'B') {
            // layout is BSH
            OPS_LOG_E_IF((queryShape.GetDim(0) != keyShape.GetDim(0)), context, return ge::GRAPH_FAILED,
                         "query or key shape is invalid");
            OPS_ERR_IF(queryShape.GetDim(2) % headNum != 0,
               OPS_REPORT_VECTOR_INNER_ERR(context, "h1 [%ld] should be a multiple of headNum [%ld].",
               queryShape.GetDim(2), headNum),
               return ge::GRAPH_FAILED);
        } else if (inputLayout[0] == 'T') { // TND  N1 != N2
            OPS_ERR_IF(headNum != queryShape.GetDim(1),
               OPS_REPORT_VECTOR_INNER_ERR(context, "headNum is [%ld], but got n1 [%ld].",
               headNum, queryShape.GetDim(1)),
               return ge::GRAPH_FAILED);
            return ge::SUCCESS;
        } else {
            // layout is SBH
            OPS_LOG_E_IF((queryShape.GetDim(1) != keyShape.GetDim(1)), context, return ge::GRAPH_FAILED,
                         "query or key shape is invalid");
            OPS_ERR_IF(queryShape.GetDim(2) % headNum != 0,
               OPS_REPORT_VECTOR_INNER_ERR(context, "h1 [%ld] should be a multiple of headNum [%ld].",
               queryShape.GetDim(2), headNum),
               return ge::GRAPH_FAILED);
        }
    } else if (strlen(inputLayout) == 4) { // 4: layout is BNSD or BSND
        OPS_LOG_E_IF((queryShape.GetDim(0) != keyShape.GetDim(0)), context, return ge::GRAPH_FAILED,
                     "query or key shape is invalid");
        OPS_LOG_E_IF((queryShape.GetDim(3) != keyShape.GetDim(3)), context, return ge::GRAPH_FAILED,
                     "query or key shape is invalid");
        if (inputLayout[1] == 'N') {
            OPS_ERR_IF(headNum != queryShape.GetDim(1),
                   OPS_REPORT_VECTOR_INNER_ERR(context, "headNum is [%ld], but got n1 [%ld].",
                   headNum, queryShape.GetDim(1)),
                   return ge::GRAPH_FAILED);
        } else {
            OPS_ERR_IF(headNum != queryShape.GetDim(2),
                   OPS_REPORT_VECTOR_INNER_ERR(context, "headNum is [%ld], but got n1 [%ld].",
                   headNum, queryShape.GetDim(2)),
                   return ge::GRAPH_FAILED);  
        }
    } else {
        OPS_LOG_E(context, "invalid input_layout[%s].", inputLayout);
        return ge::GRAPH_FAILED;
    }
    return ge::SUCCESS;
}

static ge::graphStatus CheckParams(gert::TilingContext *context)
{
    OPS_LOG_E_IF(context == nullptr, context, return ge::GRAPH_FAILED, "context is null");
    OPS_ERR_IF(CheckAttrs(context) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "invalid attrs"), return ge::GRAPH_FAILED);
    if (context->GetInputShape(QUERY_INPUT_INDEX) != nullptr && context->GetInputShape(KEY_INPUT_INDEX) != nullptr &&
        context->GetInputShape(VALUE_INPUT_INDEX) != nullptr && context->GetInputShape(DY_INPUT_INDEX) != nullptr &&
        context->GetOptionalInputShape(SOFTMAX_MAX) != nullptr &&
        context->GetOptionalInputShape(SOFTMAX_SUM) != nullptr &&
        context->GetOptionalInputShape(ATTENTION_IN) != nullptr) {
        if (CheckBaseInput(context) == ge::GRAPH_SUCCESS) {
            return ge::SUCCESS;
        }
    }
    OPS_LOG_E(context, "fail to get shape or attr from context");
    return ge::GRAPH_FAILED;
}

ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionGradScore(gert::TilingContext *context)
{
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (IsEmptyOutput(context)) {
        FlashAttentionScoreGradTiling flashAttentionScoreGradTiling;
        return flashAttentionScoreGradTiling.RunEmptyTiling(context);
    } else {
        return TilingRegistry::GetInstance().DoTilingImpl(context);
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFlashAttentionScoreGrad(gert::TilingParseContext *context)
{
    OPS_LOG_E_IF(context == nullptr, context, return ge::GRAPH_FAILED, "context is null.");
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OPS_LOG_E_IF(platformInfoPtr == nullptr, context, return ge::GRAPH_FAILED, "platformInfoPtr is null.");

    auto compileInfoPtr = context->GetCompiledInfo<FlashAttentionScoreGradCompileInfo>();
    OPS_LOG_E_IF(compileInfoPtr == nullptr, context, return ge::GRAPH_FAILED, "compileInfoPtr is null.");

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);

    OPS_LOG_I(context,
              "parse TilingParseContext succ. aivNum:%u, aicNum:%u, "
              "ubSize:%lu, l1Size:%lu, l0aSize:%lu, l0bSize:%lu, l0cSize:%lu, l2CacheSize:%lu",
              compileInfoPtr->aivNum, compileInfoPtr->aicNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size,
              compileInfoPtr->l0aSize, compileInfoPtr->l0bSize, compileInfoPtr->l0cSize, compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FlashAttentionScoreGrad)
    .Tiling(TilingFlashAttentionGradScore)
    .TilingInputsDataDependency({12, 13, 14, 15, 16})
    .TilingParse<FlashAttentionScoreGradCompileInfo>(TilingPrepareForFlashAttentionScoreGrad); // 向框架注册入口函数

} // namespace optiling
