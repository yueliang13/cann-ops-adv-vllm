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
 * \file matmul_all_reduce_tiling_910.cc
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_TILING_910_CC
#define MATMUL_ALL_REDUCE_TILING_910_CC
#include "op_mc2.h"
#include "matmul_all_reduce_tiling_910.h"
namespace optiling {
bool MatmulAllReduceTiling910::IsCapable()
{
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND310P) {
        OPS_LOG_I(opName_, "skip 910 tiling when 310p.");
        return false;
    }
    OPS_LOG_I(opName_, "start with MatmulAllReduceTiling910 tiling.");
    return true;
}

void MatmulAllReduceTiling910::DoEmptyTensorTiling()
{
    MutableTCubeTileTilingData().set_M(args_.orgMValue);
    MutableTCubeTileTilingData().set_N(args_.orgNValue);
    MutableTCubeTileTilingData().set_isBias(args_.isBias);
    MutableTCubeTileTilingData().set_usedCoreNum(1);
}

ge::graphStatus MatmulAllReduceTiling910::DoOpTiling()
{
    CheckA16W16();
    CheckInput();
    DoRCSTiling();
    DoSplitMTiling();
    if (!isKZero_) {
        Do910Tiling();
    } else {
        DoEmptyTensorTiling();
    }
    DoAllReduceTiling(true);
    return ge::GRAPH_SUCCESS;
}

uint64_t MatmulAllReduceTiling910::GetTilingKey() const
{
    if (unlikely(isKZero_)) {
        OPS_LOG_I(opName_, "Get tiling key %lu for empty tensor.", EMPTY_TENSOR_KEY);
        return EMPTY_TENSOR_KEY;
    }

    const uint64_t tilingKeyInCtx = context_->GetTilingKey();
    if (tilingKeyInCtx == MM_ALINGNED_TILING_KEY && !enableBiasConvert_ &&
        !matmulAllReduce910TilingData_.param.get_isAdd()) {
        OPS_LOG_I(opName_, "Get tiling key %lu for cube only case.", CUBE_ONLY_KEY);
        return CUBE_ONLY_KEY;
    }

    OPS_LOG_I(opName_, "Get tiling key %lu.", tilingKeyInCtx);
    return tilingKeyInCtx;
}

ge::graphStatus MatmulAllReduceTiling910::GetWorkspaceSize()
{
    MatmulAllReduceTilingBase::GetWorkspaceSize();
    OPS_LOG_I(opName_, "select max workspace size to context, myWorkSpaceSize_:%lu, workspaceSize_:%lu",
            myWorkSpaceSize_, workspaceSize_);
    myWorkSpaceSize_ = std::max(myWorkSpaceSize_, workspaceSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = myWorkSpaceSize_;
    return GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::PostTiling()
{
    OPS_LOG_D(opName_, "final tiling data size: %zu and context capacity size: %zu ",
            matmulAllReduce910TilingData_.GetDataSize(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(matmulAllReduce910TilingData_.GetDataSize());

    OP_TILING_CHECK(matmulAllReduce910TilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data size[%zu] not aligned to 8",
                                                    matmulAllReduce910TilingData_.GetDataSize()),
                    return ge::GRAPH_FAILED);
    PrintTilingData();
    context_->SetBlockDim(args_.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::Do910Tiling()
{
    args_.mValue = tileMValue_;
    TilingTransferHelper mmTile(*this, matmulAllReduce910TilingData_.tilematmulTiling);
    if (args_.enableSplitK) {
        OPS_LOG_D(opName_, "Enable SplitK Tiling.");
        return mmTile.DoTiling();
    } else {
        mmTile.DoTiling();
        if (MutableRCSTilingData().get_tailCnt() == 0) {
            return GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        TilingTransferHelper mmTail(*this, matmulAllReduce910TilingData_.tailmatmulTiling);
        return mmTail.DoTiling();
    }
}

Mc2Msg &MatmulAllReduceTiling910::MutableMc2MsgData()
{
    return matmulAllReduce910TilingData_.msg;
}

RCSTiling &MatmulAllReduceTiling910::MutableRCSTilingData()
{
    return matmulAllReduce910TilingData_.param;
}

TCubeTiling &MatmulAllReduceTiling910::MutableTCubeTileTilingData()
{
    return matmulAllReduce910TilingData_.tilematmulTiling.matmulTiling;
}

TCubeTiling &MatmulAllReduceTiling910::MutableTCubeTailTilingData()
{
    return matmulAllReduce910TilingData_.tailmatmulTiling.matmulTiling;
}

ge::graphStatus MatmulAllReduceTiling910::CheckAxisSize()
{
  const uint64_t m = MatmulAllReduceTilingBase::GetMValue();
  OP_TILING_CHECK(
      m > static_cast<uint64_t>(INT32_MAX),
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "The size of m-axis(%lu) exceeds the upper limit.", m),
      return ge::GRAPH_FAILED);
  const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
  OP_TILING_CHECK(
      k > static_cast<uint64_t>(INT32_MAX),
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "The size of k-axis(%lu) exceeds the upper limit.", k),
      return ge::GRAPH_FAILED);
  const uint64_t n = MatmulAllReduceTilingBase::GetNValue();
  OP_TILING_CHECK(
      n > static_cast<uint64_t>(INT32_MAX),
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "The size of n-axis(%lu) exceeds the upper limit.", n),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::CheckInput()
{
    MatmulAllReduceTilingBase::CheckInput();
    // x2 shape 为 2 维
    size_t x2DimNum = mmrCtxInfo_.x2_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(x2DimNum != DIM_NUM_TWO,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "In the not quant scenario, Expect x2 dim to be 2, but "
                                                    " got x2 dim:[%lu].", x2DimNum),
                    return ge::GRAPH_FAILED);
    auto x1Type = mmrCtxInfo_.x1->GetDataType();
    //  x1 为fp16 或者bf16
    OP_TILING_CHECK(!((x1Type == ge::DT_FLOAT16) || (x1Type == ge::DT_BF16)),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "In the not quant scenario, type of x1 should be"
                                                    " fp16 or bf16."),
                    return ge::GRAPH_FAILED);
    // x1，x2数据类型相同
    auto x2Type = mmrCtxInfo_.x2->GetDataType();
    OP_TILING_CHECK(x1Type != x2Type,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "In the not quant scenario, type of x1 and x2"
                                                    " should be same"),
                    return ge::GRAPH_FAILED);
    // x1,bias数据类型相同
    if (mmrCtxInfo_.bias_shape != nullptr) {
        auto biasType = mmrCtxInfo_.bias->GetDataType();
        OP_TILING_CHECK(x1Type != biasType,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "In the not quant scenario, type of x1 and bias should be"
                                                        " same."),
                        return ge::GRAPH_FAILED);
    }

    auto outputDimNum = mmrCtxInfo_.y_shape->GetStorageShape().GetDimNum();
    if (mmrCtxInfo_.x3_shape != nullptr) {
        auto x3DimNum = mmrCtxInfo_.x3_shape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(outputDimNum != x3DimNum,
            VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                            "In the not quant scenario, shape of x3 and output should be"
                                            " same."),
            return ge::GRAPH_FAILED);
        for (size_t i = 0U; i < outputDimNum; i++) {
            OP_TILING_CHECK(mmrCtxInfo_.y_shape->GetStorageShape().GetDim(i) !=
                mmrCtxInfo_.x3_shape->GetStorageShape().GetDim(i),
                VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                "In the not quant scenario, shape of x3 and output should be"
                                                " same."),
                return ge::GRAPH_FAILED);
        }
    }
    return CheckAxisSize();
}

MatmulAllReduceTiling910::MatmulAllReduceTiling910(gert::TilingContext *context)
    : MatmulAllReduceTilingBase(context), matmulAllReduce910TilingData_(matmulAllReduce910TilingDataSelf_)
{
    matmulAllReduce910TilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
}

MatmulAllReduceTiling910::MatmulAllReduceTiling910(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo,
                                                   MatmulAllReduce910TilingData *out)
    : MatmulAllReduceTilingBase(context, mmrCtxInfo), matmulAllReduce910TilingData_(*out)
{
}

ge::graphStatus TilingTransferHelper::GetShapeAttrsInfo()
{
    auto &&tilingArgs = tilingProcesser_.args_;
    args_.opName = tilingProcesser_.opName_;
    args_.isATrans = tilingArgs.isATrans;
    args_.isBTrans = tilingArgs.isBTrans;
    args_.hasBias = tilingArgs.isBias;
    args_.aType = tilingArgs.geAType;
    args_.bType = tilingArgs.geBType;
    args_.cType = tilingArgs.geCType;
    args_.biasType = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;

    args_.aFormat = ge::FORMAT_ND;
    args_.bFormat = ge::FORMAT_ND;
    args_.outFormat = ge::FORMAT_ND;

    args_.mValue = tilingArgs.mValue;
    args_.kValue = tilingArgs.kValue;
    args_.nValue = tilingArgs.nValue;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingTransferHelper::PostTiling()
{
    tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
    OPS_LOG_I(tilingProcesser_.opName_, " set mm workspace size %lu to mc2", tilingProcesser_.myWorkSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

TilingTransferHelper::TilingTransferHelper(MatmulAllReduceTiling910 &matmulAllReduceTiling910, MatmulTilingData &data)
    : MatmulV3BaseTiling(matmulAllReduceTiling910.context_, &data), tilingProcesser_(matmulAllReduceTiling910)
{
}
}
#endif // MATMUL_ALL_REDUCE_TILING_910_CC