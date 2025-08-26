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
 * \file weight_quant_matmul_all_reduce_add_rms_norm_tiling.cc
 * \brief
 */
#ifndef _WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_
#define _WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_

#include "weight_quant_matmul_all_reduce_add_rms_norm_tiling.h"
namespace optiling {
namespace {
constexpr char MRN[] = "MatmulAllReduceAddRmsNorm";
constexpr char IMRN[] = "InplaceMatmulAllReduceAddRmsNorm";
} // namespace
WeightQuantMMNTilingTransferHelper::WeightQuantMMNTilingTransferHelper(
    WeightQuantMatmulAllReduceAddRmsNormTiling &weightQuantMatmulAllReduceAddRmsNormTiling,
    WeightQuantMatmulAllReduceTilingData &data)
    : WeightQuantMatmulAllReduceTiling(weightQuantMatmulAllReduceAddRmsNormTiling.context_,
                                       &weightQuantMatmulAllReduceAddRmsNormTiling.mrnCtxInfo_.mmrCtxInfo, &data),
      tilingProcesser_(weightQuantMatmulAllReduceAddRmsNormTiling)
{
}
ge::graphStatus WeightQuantMMNTilingTransferHelper::GetShapeAttrsInfo()
{
    return MatmulAllReduceTilingBase::AnalyzeShapeAttr();
}

bool WeightQuantMatmulAllReduceAddRmsNormTiling::HasTail() const
{
    return hasTail_;
}
ge::graphStatus WeightQuantMatmulAllReduceAddRmsNormTiling::CheckMRNInput(const MRNCtxInfo &mrnCtxInfo)
{
    // x1和residual数据类型是否相同
    auto x1Type = mrnCtxInfo.mmrCtxInfo.x1->GetDataType();
    auto residualType = mrnCtxInfo.arnCtxInfo.x2->GetDataType();
    OP_TILING_CHECK(x1Type != residualType,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "In the antiquant scenario, type of x1 and residual should be"
                                                    " same"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus WeightQuantMatmulAllReduceAddRmsNormTiling::DoOpTiling()
{
    helper_->DoOpTiling();
    CommonAddResNormTiling::CheckAddRmsNormInput(context_, mrnCtxInfo_.arnCtxInfo);
    ContextTransfer::CheckMRNCtxInfo(context_, mrnCtxInfo_);
    CheckMRNInput(mrnCtxInfo_);
    hasTail_ = (tilingData_.weightQuantMatmulAllReduceTilingData.param.get_tailCnt() != 0);
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
ge::graphStatus WeightQuantMatmulAllReduceAddRmsNormTiling::GetShapeAttrsInfo()
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
ge::graphStatus WeightQuantMatmulAllReduceAddRmsNormTiling::GetPlatformInfo()
{
    return helper_->GetPlatformInfo();
}
ge::graphStatus WeightQuantMatmulAllReduceAddRmsNormTiling::DoLibApiTiling()
{
    return helper_->DoLibApiTiling();
}
bool WeightQuantMatmulAllReduceAddRmsNormTiling::IsCapable()
{
    return helper_->IsCapable();
}
WeightQuantMatmulAllReduceAddRmsNormTiling::WeightQuantMatmulAllReduceAddRmsNormTiling(gert::TilingContext *context)
    : TilingBaseClass(context)
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
    helper_ = std::move(std::unique_ptr<WeightQuantMMNTilingTransferHelper>(new (
        std::nothrow) WeightQuantMMNTilingTransferHelper(*this, tilingData_.weightQuantMatmulAllReduceTilingData)));
}

ge::graphStatus WeightQuantMatmulAllReduceAddRmsNormTiling::GetWorkspaceSize()
{
    helper_->GetWorkspaceSize();
    const auto mc2_workspace = helper_->myWorkSpaceSize_;
    GE_ASSERT_TRUE(mc2_workspace >= SYS_WORKSPACE_SIZE);
    if (HasTail()) {
        GE_ASSERT_TRUE(tilingOutAddRmsNormTile_.workSpaceSize == tilingOutAddRmsNormTail_.workSpaceSize);
    }
    // 系统空间用mc2申请的就好了， arn的减去这部分
    GE_ASSERT_TRUE(tilingOutAddRmsNormTile_.workSpaceSize >= SYS_WORKSPACE_SIZE);
    const auto arn_workspace = tilingOutAddRmsNormTile_.workSpaceSize - SYS_WORKSPACE_SIZE;
    const auto my_workspace = mc2_workspace + arn_workspace;
    OPS_LOG_I(helper_->opName_, " Workspace %lu with detail: mc2: %lu arn：%u", my_workspace, mc2_workspace,
              arn_workspace);
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = my_workspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantMatmulAllReduceAddRmsNormTiling::PostTiling()
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

uint64_t WeightQuantMatmulAllReduceAddRmsNormTiling::GetTilingKey() const
{
    const auto mc2_key = helper_->GetTilingKey();
    const auto my_key = mc2_key; // use mc2 key as mrn key
    OPS_LOG_I(helper_->opName_, " tilingKey %lu with detail: mc2_key: %lu arn_key tile：%u arn_key tail: %u", my_key,
              mc2_key, tilingOutAddRmsNormTile_.tilingKey, tilingOutAddRmsNormTail_.tilingKey);
    return my_key;
}
using InplaceWeightQuantMatmulAllReduceAddRmsNormTiling = WeightQuantMatmulAllReduceAddRmsNormTiling;
REGISTER_TILING_TEMPLATE(MRN, WeightQuantMatmulAllReduceAddRmsNormTiling, 1);
REGISTER_TILING_TEMPLATE(IMRN, InplaceWeightQuantMatmulAllReduceAddRmsNormTiling, 1);
} // namespace optiling
#endif // _WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_