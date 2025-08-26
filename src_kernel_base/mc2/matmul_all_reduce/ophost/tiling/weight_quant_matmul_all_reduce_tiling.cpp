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
 * \file weight_quant_matmul_all_reduce_tiling.cc
 * \brief
 */
#include "op_mc2.h"
#include "weight_quant_matmul_all_reduce_tiling.h"
namespace optiling {

ge::graphStatus WeightQuantTilingTransferHelper::GetShapeAttrsInfo()
{
    OPS_LOG_I(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
    auto &&tilingArgs = tilingProcesser_.args_;
    opName_ = tilingProcesser_.opName_;
    matmulInfoPtr_.reset(new WeightQuantBatchMatmulInfo());
    matmulInfoPtr_->transA = tilingArgs.isATrans;
    matmulInfoPtr_->transB = tilingArgs.isBTrans;
    matmulInfoPtr_->hasBias = tilingArgs.isBias;
    matmulInfoPtr_->hasAntiQuantOffset = tilingProcesser_.HasAntiQuantOffset();
    matmulInfoPtr_->mSize = tilingArgs.mValue;
    matmulInfoPtr_->kSize = tilingArgs.kValue;
    matmulInfoPtr_->nSize = tilingArgs.nValue;
    matmulInfoPtr_->aDtype = tilingArgs.geAType;
    matmulInfoPtr_->bDtype = tilingArgs.geBType;
    matmulInfoPtr_->cDtype = tilingArgs.geCType;
    matmulInfoPtr_->biasDtype = tilingArgs.geBiasType;
    matmulInfoPtr_->antiQuantType = tilingProcesser_.antiQuantType_;
    matmulInfoPtr_->groupSize = tilingProcesser_.antiGroupSize_;
    matmulInfoPtr_->quantType = tilingProcesser_.quantType_;
    matmulInfoPtr_->bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(
        tilingProcesser_.mmrCtxInfo_.x2->GetStorageFormat()));
    OP_TILING_CHECK((matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) && (matmulInfoPtr_->antiQuantType !=
                                                                        QuantType::PER_CHANNEL),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_,
                                                    "Nz weight input only supports per-channel scene, "
                                                    "current anti-quant type: [%d].",
                                                    static_cast<int>(matmulInfoPtr_->antiQuantType)),
                    return ge::GRAPH_FAILED);
    PrintTilingInputParam(*matmulInfoPtr_);
    return ge::GRAPH_SUCCESS;
}
void WeightQuantTilingTransferHelper::PrintTilingInputParam(
    WeightQuantBatchMatmulInfo &weightQuantBatchMatmulInfo)
{
    OPS_LOG_D(tilingProcesser_.opName_,
            " transA_ %d transB_ %d, hasBias_ %d, hasAntiQuantOffset_ %d",
            weightQuantBatchMatmulInfo.transA, weightQuantBatchMatmulInfo.transB,
            weightQuantBatchMatmulInfo.hasBias, weightQuantBatchMatmulInfo.hasAntiQuantOffset);
    OPS_LOG_D(tilingProcesser_.opName_,
            "mSize_ %ld kSize_ %ldnSize_ %ld groupSize_ %ld",
            weightQuantBatchMatmulInfo.mSize, weightQuantBatchMatmulInfo.kSize,
            weightQuantBatchMatmulInfo.nSize, weightQuantBatchMatmulInfo.groupSize);
    OPS_LOG_D(tilingProcesser_.opName_, "aDtype_ %d bDtype_ %d cDtype_ %d biasDtype_ %d",
            static_cast<int32_t>(weightQuantBatchMatmulInfo.aDtype),
            static_cast<int32_t>(weightQuantBatchMatmulInfo.bDtype),
            static_cast<int32_t>(weightQuantBatchMatmulInfo.cDtype),
            static_cast<int32_t>(weightQuantBatchMatmulInfo.biasDtype));
    OPS_LOG_D(tilingProcesser_.opName_, "antiQuantType_ %d quantType_ %d bFormat %d",
            static_cast<int32_t>(weightQuantBatchMatmulInfo.antiQuantType),
            static_cast<int32_t>(weightQuantBatchMatmulInfo.quantType),
            static_cast<int32_t>(weightQuantBatchMatmulInfo.bFormat));
}
ge::graphStatus WeightQuantTilingTransferHelper::PostTiling()
{
    tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
    OPS_LOG_I(tilingProcesser_.opName_, " set mm workspace size %lu to mc2", tilingProcesser_.myWorkSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

bool WeightQuantMatmulAllReduceTiling::IsCapable()
{
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND310P) {
        OPS_LOG_I(opName_, "skip weight quant tiling when 310p.");
        return false;
    }
    if (isA16W8_ || isA16W4_) {
        OPS_LOG_I(opName_, "start with weight quant tiling.");
        return true;
    }
    OPS_LOG_I(opName_, "skip weight quant tiling as dtype not support");
    return false;
}

void WeightQuantMatmulAllReduceTiling::DoEmptyTensorTiling()
{
    MutableTCubeTileTilingData().set_M(args_.orgMValue);
    MutableTCubeTileTilingData().set_isBias(args_.isBias);
    MutableTCubeTileTilingData().set_usedCoreNum(1);
}

ge::graphStatus WeightQuantMatmulAllReduceTiling::DoOpTiling()
{
    CheckA16W8();
    CheckInput();
    DoRCSTiling();
    DoSplitMTiling();
    if (isKZero_) {
        DoEmptyTensorTiling();
        DoAllReduceTiling(true);
        return ge::GRAPH_SUCCESS;
    }
    DoWeightQuantTiling();
    DoAllReduceTiling(true);
    return ge::GRAPH_SUCCESS;
}
uint64_t WeightQuantMatmulAllReduceTiling::GetTilingKey() const
{
    uint64_t tilingKey = context_->GetTilingKey();
    if (isKZero_) {
        tilingKey = WEIGHT_QUANT_EMPTY_TENSOR_KEY;
    }
    OPS_LOG_I(opName_, " tilingKey %lu", tilingKey);
    return tilingKey;
}
ge::graphStatus WeightQuantMatmulAllReduceTiling::GetWorkspaceSize()
{
    MatmulAllReduceTilingBase::GetWorkspaceSize();
    myWorkSpaceSize_ = myWorkSpaceSize_ + MutableRCSTilingData().get_biasLen();
    if (isKZero_) {
        myWorkSpaceSize_ = myWorkSpaceSize_ + libApiWorkSpaceSize_;
        OPS_LOG_D(opName_, " Empty tensor k is 0, set workspace size %lu to context", myWorkSpaceSize_);
    }
    OPS_LOG_I(opName_, " set max workspace size %lu to context", myWorkSpaceSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);  // set workspace
    workspaces[0] = myWorkSpaceSize_;
    return GRAPH_SUCCESS;
}
ge::graphStatus WeightQuantMatmulAllReduceTiling::PostTiling()
{
    OPS_LOG_D(opName_, "final tiling data size: %zu and context capacity size: %zu ",
            weightQuantMatmulAllReduceTilingData_.GetDataSize(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(weightQuantMatmulAllReduceTilingData_.GetDataSize());

    OP_TILING_CHECK(weightQuantMatmulAllReduceTilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data size[%zu] not aligned to 8",
                                                    weightQuantMatmulAllReduceTilingData_.GetDataSize()),
                    return ge::GRAPH_FAILED);
    PrintTilingData();
    context_->SetBlockDim(args_.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus WeightQuantMatmulAllReduceTiling::DoWeightQuantTiling()
{
    args_.mValue = tileMValue_;
    WeightQuantTilingTransferHelper mmTile(*this, weightQuantMatmulAllReduceTilingData_.tilematmulTiling);
    if (args_.enableSplitK) {
        return mmTile.DoTiling();
    } else {
        mmTile.DoTiling();
        if (MutableRCSTilingData().get_tailCnt() == 0) {
            return GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        WeightQuantTilingTransferHelper mmTail(*this, weightQuantMatmulAllReduceTilingData_.tailmatmulTiling);
        return mmTail.DoTiling();
    }
}
ge::graphStatus WeightQuantMatmulAllReduceTiling::CheckAxisSize()
{
  const uint64_t m = MatmulAllReduceTilingBase::GetMValue();
  OP_TILING_CHECK(
      m > static_cast<uint64_t>(INT32_MAX),
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "The size of m-axis(%lu) "
                                     "exceeds the upper limit(%d).", m, INT32_MAX),
      return ge::GRAPH_FAILED);
  const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
  OP_TILING_CHECK(
      k > static_cast<uint64_t>(UINT16_MAX),
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "The size of k-axis(%lu) "
                                     "exceeds the upper limit(%d).", k, UINT16_MAX),
      return ge::GRAPH_FAILED);
  const uint64_t n = MatmulAllReduceTilingBase::GetNValue();
  OP_TILING_CHECK(
      n > static_cast<uint64_t>(UINT16_MAX),
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "The size of n-axis(%lu) "
                                     "exceeds the upper limit(%d).", n, UINT16_MAX),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus WeightQuantMatmulAllReduceTiling::CheckInput()
{
    MatmulAllReduceTilingBase::CheckInput();
    const size_t x2DimNum = (static_cast<ge::Format>(ge::GetPrimaryFormat(mmrCtxInfo_.x2->GetStorageFormat())) ==
            ge::Format::FORMAT_FRACTAL_NZ ? 4 : 2);
    const size_t actualX2DimNum = mmrCtxInfo_.x2_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(x2DimNum != actualX2DimNum,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "In the antiquant scenario, Expect x2 dim to be %lu,"
                                                    " but got x2 dim:[%lu].", x2DimNum, actualX2DimNum),
                    return ge::GRAPH_FAILED);
    auto x1Type = mmrCtxInfo_.x1->GetDataType();
    OP_TILING_CHECK(!((x1Type == ge::DT_FLOAT16) || (x1Type == ge::DT_BF16)),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "In the antiquant scenario, type"
                                                    " of x1 should be fp16 or bf16"),
                    return ge::GRAPH_FAILED);
    auto x2Type = mmrCtxInfo_.x2->GetDataType();
    OP_TILING_CHECK(!((x2Type == ge::DT_INT8) || (x2Type == ge::DT_INT4)),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "In the antiquant scenario, type"
                                                    " of x2 should be int8 or int4."),
                    return ge::GRAPH_FAILED);
    if (mmrCtxInfo_.bias_shape != nullptr) {
        OP_TILING_CHECK(x1Type != mmrCtxInfo_.bias->GetDataType(),
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "In the antiquant scenario,"
                                                        " type of x1 and bias should be same"),
                        return ge::GRAPH_FAILED);
    }
    // x1,antiquantScale数据类型相同
    auto antiquantScaleType = mmrCtxInfo_.antiquant_scale->GetDataType();
    OP_TILING_CHECK(antiquantScaleType != x1Type,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "In the antiquant scenario, type"
                                                    " of antiquantScale and x1 should be same"),
                    return ge::GRAPH_FAILED);
    // antiquantScale和antiquantOffset数据类型相同
    if (mmrCtxInfo_.antiquant_offset_shape != nullptr) {
        auto antiquantOffsetType = mmrCtxInfo_.antiquant_offset->GetDataType();
        OP_TILING_CHECK(antiquantOffsetType != antiquantScaleType,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "In the antiquant scenario, type"
                                                    " of antiquantScale and antiquantOffset should be same"),
                    return ge::GRAPH_FAILED);
    }
    // antiquantgroupsize 校验
    uint64_t kValue = GetKValue();
    if (kValue != 0 && mmrCtxInfo_.groupSizePtr != nullptr) {
        const int64_t groupSize = *(mmrCtxInfo_.groupSizePtr);
        OP_TILING_CHECK(((groupSize != 0) && (groupSize % ANTIQUANT_GROUP_SIZE_MIN_VALUE != 0 ||
                        groupSize < ANTIQUANT_GROUP_SIZE_MIN_VALUE ||
                        groupSize > std::min(static_cast<int32_t>(kValue - 1), INT32_MAX))),
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "In the per-group scenario,"
                        " antiquantGroupSize should be in range [32, min(%ld, INT_MAX)], Actual is %ld.",
                        (kValue - 1), groupSize), return ge::GRAPH_FAILED);
    }
    return CheckAxisSize();
}
WeightQuantMatmulAllReduceTiling::WeightQuantMatmulAllReduceTiling(gert::TilingContext *context)
    : MatmulAllReduceTilingBase(context),
    weightQuantMatmulAllReduceTilingData_(weightQuantMatmulAllReduceTilingDataSelf_)
{
    weightQuantMatmulAllReduceTilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
}
WeightQuantMatmulAllReduceTiling::WeightQuantMatmulAllReduceTiling(
    gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo, WeightQuantMatmulAllReduceTilingData *out)
    : MatmulAllReduceTilingBase(context, mmrCtxInfo), weightQuantMatmulAllReduceTilingData_(*out)
{
}
}
