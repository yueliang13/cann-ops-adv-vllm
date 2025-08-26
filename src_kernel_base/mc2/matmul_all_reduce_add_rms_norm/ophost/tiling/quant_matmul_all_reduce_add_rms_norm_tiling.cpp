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
 * \file quant_matmul_all_reduce_add_rms_norm_tiling.cc
 * \brief
 */
#ifndef _QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_
#define _QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_

#include "quant_matmul_all_reduce_add_rms_norm_tiling.h"
namespace optiling {
namespace {
constexpr char MRN[] = "MatmulAllReduceAddRmsNorm";
constexpr char IMRN[] = "InplaceMatmulAllReduceAddRmsNorm";
} // namespace
QuantMMNTilingTransferHelper::QuantMMNTilingTransferHelper(
    QuantMatmulAllReduceAddRmsNormTiling &quantMatmulAllReduceAddRmsNormTiling, QuantMatmulAllReduceTilingData &data)
    : QuantMatmulAllReduceTiling(quantMatmulAllReduceAddRmsNormTiling.context_,
                                 &quantMatmulAllReduceAddRmsNormTiling.mrnCtxInfo_.mmrCtxInfo, &data),
      tilingProcesser_(quantMatmulAllReduceAddRmsNormTiling)
{
}
ge::graphStatus QuantMMNTilingTransferHelper::GetShapeAttrsInfo()
{
    return MatmulAllReduceTilingBase::AnalyzeShapeAttr();
}

bool QuantMatmulAllReduceAddRmsNormTiling::HasTail() const
{
    return hasTail_;
}
ge::graphStatus QuantMatmulAllReduceAddRmsNormTiling::CheckMRNInput(const MRNCtxInfo &mrnCtxInfo)
{
    // dequantScale数据类型为bf16时, residual为bf16;其他时,residual为fp16
    auto dequantScaleType = mrnCtxInfo.mmrCtxInfo.dequant_scale->GetDataType();
    auto residualType = mrnCtxInfo.arnCtxInfo.x2->GetDataType();
    if (dequantScaleType == ge::DT_BF16) {
        OP_TILING_CHECK(residualType != ge::DT_BF16,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "when dequantScaleType = bf16, Expect type of"
                                                        " residual to be bf16."),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(residualType != ge::DT_FLOAT16,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "when dequantScaleType != bf16, Expect type of"
                                                        " residual to be fp16"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus QuantMatmulAllReduceAddRmsNormTiling::DoOpTiling()
{
    helper_->DoOpTiling();
    CommonAddResNormTiling::CheckAddRmsNormInput(context_, mrnCtxInfo_.arnCtxInfo);
    ContextTransfer::CheckMRNCtxInfo(context_, mrnCtxInfo_);
    CheckMRNInput(mrnCtxInfo_);
    hasTail_ = (tilingData_.qunatMatmulAllReduceTilingData.param.get_tailCnt() != 0);
    AddRmsNormTilingInputFromMM addRmsNormTilingInputFromMm;
    addRmsNormTilingInputFromMm.m = helper_->tileMValue_;
    addRmsNormTilingInputFromMm.n = helper_->args_.nValue;
    addRmsNormTilingInputFromMm.x1Dtype = helper_->args_.geCType;
    GE_ASSERT_TRUE(context_->GetPlatformInfo() != nullptr);
    AddRMSNormTilingDepend addRmsNormTilingDepend = {context_->GetNodeName(),
                                                     *context_->GetPlatformInfo(),
                                                     mrnCtxInfo_.arnCtxInfo,
                                                     addRmsNormTilingInputFromMm,
                                                     true,
                                                     false};

    AddRMSNormTilingOutput addRmsNormTilingOutput = {tilingData_.addRMSNormTileTilingData, tilingOutAddRmsNormTile_};

    CommonAddResNormTiling::Tiling4AddRmsNorm(addRmsNormTilingDepend, addRmsNormTilingOutput);
    tilingData_.addRmsNormTilingeKeyData.set_ARNKeyTile(tilingOutAddRmsNormTile_.tilingKey);
    tilingData_.addRmsNormTilingeKeyData.set_ARNBlockDimTile(tilingOutAddRmsNormTile_.blockDim);

    if (HasTail()) {
        addRmsNormTilingDepend.addRmsNormTilingInputFromMm.m = helper_->tailMValue_;
        AddRMSNormTilingOutput addRmsNormTilingOutputTail = {tilingData_.addRMSNormTailTilingData,
                                                             tilingOutAddRmsNormTail_};
        CommonAddResNormTiling::Tiling4AddRmsNorm(addRmsNormTilingDepend, addRmsNormTilingOutputTail);
        tilingData_.addRmsNormTilingeKeyData.set_ARNKeyTail(tilingOutAddRmsNormTail_.tilingKey);
        tilingData_.addRmsNormTilingeKeyData.set_ARNBlockDimTail(tilingOutAddRmsNormTail_.blockDim);
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus QuantMatmulAllReduceAddRmsNormTiling::GetShapeAttrsInfo()
{
    if (strcmp(context_->GetNodeType(), MRN) == 0) {
        ContextTransfer::AssembleMRNCtxInfoFromMRNCtx(context_, mrnCtxInfo_);
    } else if (strcmp(context_->GetNodeType(), IMRN) == 0) {
        ContextTransfer::AssembleIMRNCtxInfoFromIMRNCtx(context_, mrnCtxInfo_);
    } else {
        OPS_LOG_E(context_->GetNodeName(), "Unsupported node type %s", context_->GetNodeType());
        return ge::GRAPH_FAILED;
    }
    GE_ASSERT_NOTNULL(helper_);
    return helper_->GetShapeAttrsInfo();
}
ge::graphStatus QuantMatmulAllReduceAddRmsNormTiling::GetPlatformInfo()
{
    return helper_->GetPlatformInfo();
}
ge::graphStatus QuantMatmulAllReduceAddRmsNormTiling::DoLibApiTiling()
{
    return helper_->DoLibApiTiling();
}
bool QuantMatmulAllReduceAddRmsNormTiling::IsCapable()
{
    return helper_->IsCapable();
}
QuantMatmulAllReduceAddRmsNormTiling::QuantMatmulAllReduceAddRmsNormTiling(gert::TilingContext *context)
    : TilingBaseClass(context)
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
    helper_ = std::move(std::unique_ptr<QuantMMNTilingTransferHelper>(
        new (std::nothrow) QuantMMNTilingTransferHelper(*this, tilingData_.qunatMatmulAllReduceTilingData)));
}

ge::graphStatus QuantMatmulAllReduceAddRmsNormTiling::GetWorkspaceSize()
{
    helper_->GetWorkspaceSize();
    const auto mc2_workspace = helper_->myWorkSpaceSize_;
    GE_ASSERT_TRUE(mc2_workspace >= SYS_WORKSPACE_SIZE);
    if (HasTail()) {
        GE_ASSERT_EQ(tilingOutAddRmsNormTile_.workSpaceSize, tilingOutAddRmsNormTail_.workSpaceSize);
    }
    // 系统空间用mc2申请的就好了， arn的key减去这部分
    GE_ASSERT_TRUE(tilingOutAddRmsNormTile_.workSpaceSize >= SYS_WORKSPACE_SIZE);
    const auto arn_workspace = tilingOutAddRmsNormTile_.workSpaceSize - SYS_WORKSPACE_SIZE;
    const auto my_workspace = mc2_workspace + arn_workspace;
    OPS_LOG_I(helper_->opName_, " Workspace %lu with detail: mc2: %lu arn：%u", my_workspace, mc2_workspace,
              arn_workspace);
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = my_workspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceAddRmsNormTiling::PostTiling()
{
    OPS_LOG_D(helper_->opName_, "final tiling data size: %zu and context capacity size: %zu ",
              tilingData_.GetDataSize(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    OP_TILING_CHECK(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(helper_->opName_,
                                                    "tiling data size[%zu] not aligned to"
                                                    " 8",
                                                    tilingData_.GetDataSize()),
                    return ge::GRAPH_FAILED);
    helper_->PrintTilingData();
    auto blockDimOfArn = static_cast<uint64_t>(tilingOutAddRmsNormTile_.blockDim);
    if (HasTail()) {
        blockDimOfArn = std::max(blockDimOfArn, static_cast<uint64_t>(tilingOutAddRmsNormTail_.blockDim));
    }
    OPS_LOG_I(helper_->opName_, "ctx block dim: %lu, mc2 block dim %lu, arn block dim %lu", helper_->args_.aicCoreNum,
              helper_->args_.aicCoreNum, blockDimOfArn);
    // 当前mc2给的aicCoreNum是硬件规格的最大个数, blockDimOfArn取了尾和非尾的最大值，最大值应该小于等于硬件规格的aiv num
    GE_ASSERT_TRUE(helper_->args_.aicCoreNum * 2 >= blockDimOfArn);
    context_->SetBlockDim(helper_->args_.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

uint64_t QuantMatmulAllReduceAddRmsNormTiling::GetTilingKey() const
{
    const auto mc2_key = helper_->GetTilingKey();
    const auto my_key = mc2_key; // use mc2 key as mrn key
    OPS_LOG_I(helper_->opName_, " tilingKey %lu with detail: mc2_key: %lu arn_key tile：%u arn_key tail: %u", my_key,
              mc2_key, tilingOutAddRmsNormTile_.tilingKey, tilingOutAddRmsNormTail_.tilingKey);
    return my_key;
}
using InplaceQuantMatmulAllReduceAddRmsNormTiling = QuantMatmulAllReduceAddRmsNormTiling;
REGISTER_TILING_TEMPLATE(MRN, QuantMatmulAllReduceAddRmsNormTiling, 0);
REGISTER_TILING_TEMPLATE(IMRN, InplaceQuantMatmulAllReduceAddRmsNormTiling, 0);
} // namespace optiling
#endif // _QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_