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
 * \file quant_matmul_all_reduce_tiling.cc
 * \brief
 */
#ifndef _QUANT_MATMUL_ALL_REDUCE_TILING_CC_
#define _QUANT_MATMUL_ALL_REDUCE_TILING_CC_
#include "op_mc2.h"
#include "quant_matmul_all_reduce_tiling.h"
namespace optiling {
namespace {
const gert::Shape defaultShape = gert::Shape();
const gert::StorageShape defaultStorageShape = gert::StorageShape();
}
bool QuantMatmulAllReduceTiling::IsCapable()
{
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND310P) {
        OPS_LOG_I(opName_, "skip quant tiling when 310p.");
        return false;
    }
    if (isA8W8_) {
        OPS_LOG_I(opName_, "start with quant tiling.");
        return true;
    }
    OPS_LOG_I(opName_, "skip quant tiling as dtype not support");
    return false;
}
ge::graphStatus QuantMatmulAllReduceTiling::DoOpTiling()
{
    CheckA8W8();
    CheckInput();
    DoRCSTiling();
    DoSplitMTiling();
    DoQuantTiling();
    if (MutableRCSTilingData().get_isInputCommQuantScale() == 1) {
        isCommInt8Enable_ = true;
    }
    DoAllReduceTiling(true);
    return ge::GRAPH_SUCCESS;
}
uint64_t QuantMatmulAllReduceTiling::GetTilingKey() const
{
    uint64_t tilingKey = context_->GetTilingKey();
    if (isCommInt8Enable_ == true) {
        tilingKey += 10UL; // 适配int8 通信tilingKey
    }
    OPS_LOG_I(opName_, " tilingKey %lu", tilingKey);
    return tilingKey;
}
ge::graphStatus QuantMatmulAllReduceTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);  // set workspace
    uint64_t int8WorkSpace = 0UL;
    if (MutableRCSTilingData().get_isInputCommQuantScale() == 1) {
        uint64_t padTileM = MutableTCubeTileTilingData().get_M();
        uint64_t padTailM = MutableTCubeTailTilingData().get_M();
        if (padTileM % args_.rankDim != 0) {
            padTileM += args_.rankDim - (padTileM % args_.rankDim); // args_.rankDim :1/2/4/8 不会为0
        }
        uint64_t tempPadTileM = padTileM * MutableTCubeTileTilingData().get_N() * sizeof(uint8_t);
        if (padTailM % args_.rankDim != 0) {
            padTailM += args_.rankDim - (padTailM % args_.rankDim); // args_.rankDim :1/2/4/8 不会为0
        }
        uint64_t tempPadTailM = padTailM * MutableTCubeTailTilingData().get_N() * sizeof(uint8_t);
        int8WorkSpace = tempPadTileM * MutableRCSTilingData().get_tileCnt()+ tempPadTailM * MutableRCSTilingData().get_tailCnt();
        OPS_LOG_I(opName_, " set int8WorkSpace size %lu to context", int8WorkSpace);
    }
    uint64_t commWorkSpace = myWorkSpaceSize_ - libApiWorkSpaceSize_;
    MutableRCSTilingData().set_commWorkSpaceSize(commWorkSpace); // myWorkSpaceSize_去除系统空间后剩余大小
    myWorkSpaceSize_ = myWorkSpaceSize_ + int8WorkSpace;
    OPS_LOG_I(opName_, " set max workspace size %lu to context", myWorkSpaceSize_);
    workspaces[0] = myWorkSpaceSize_;
    return GRAPH_SUCCESS;
}
ge::graphStatus QuantMatmulAllReduceTiling::PostTiling()
{
    OPS_LOG_D(opName_, "final tiling data size: %zu and context capacity size: %zu ",
            quantMatmulAllReduceTilingData_.GetDataSize(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(quantMatmulAllReduceTilingData_.GetDataSize());

    OP_TILING_CHECK(quantMatmulAllReduceTilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data size[%zu] not aligned to 8",
                                                    quantMatmulAllReduceTilingData_.GetDataSize()),
                    return ge::GRAPH_FAILED);
    if (MutableRCSTilingData().get_rankID() == 0) {
        PrintTilingData(MutableRCSTilingData());
        PrintTilingData(MutableTCubeTileTilingData());
        PrintTilingData(MutableMc2MsgData());
        if (MutableRCSTilingData().get_tailM() > 0) {
            OPS_LOG_D(opName_, "have tail");
            PrintTilingData(MutableTCubeTailTilingData());
        }
    }

    context_->SetBlockDim(args_.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}
Mc2Msg &QuantMatmulAllReduceTiling::MutableMc2MsgData()
{
    return quantMatmulAllReduceTilingData_.msg;
}
RCSTiling &QuantMatmulAllReduceTiling::MutableRCSTilingData()
{
    return quantMatmulAllReduceTilingData_.param;
}
TCubeTiling &QuantMatmulAllReduceTiling::MutableTCubeTileTilingData()
{
    return quantMatmulAllReduceTilingData_.tilematmulTiling.matmulTiling;
}
TCubeTiling &QuantMatmulAllReduceTiling::MutableTCubeTailTilingData()
{
    return quantMatmulAllReduceTilingData_.tailmatmulTiling.matmulTiling;
}
ge::graphStatus QuantMatmulAllReduceTiling::DoQuantTiling()
{
    args_.mValue = tileMValue_;
    QuantTilingTransferHelper mmTile(*this, quantMatmulAllReduceTilingData_.tilematmulTiling);
    if (args_.enableSplitK) {
        OPS_LOG_D(opName_, "Enable SplitK Tiling.");
        return mmTile.DoTiling();
    } else {
        mmTile.DoTiling();
        if (MutableRCSTilingData().get_tailCnt() == 0) {
            return GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        QuantTilingTransferHelper mmTail(*this, quantMatmulAllReduceTilingData_.tailmatmulTiling);
        return mmTail.DoTiling();
    }
}
ge::graphStatus QuantMatmulAllReduceTiling::CheckAxisSize()
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
  uint64_t x2LastDim = args_.isBTrans ? k : n;
  OP_TILING_CHECK(
      x2LastDim > static_cast<uint64_t>(UINT16_MAX),
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "The size of x2 last-axis(%lu) "
                                     "exceeds the upper limit(%d).", x2LastDim, UINT16_MAX),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTiling::CheckDequantScaleType()
{
    auto dequantScaleType = mmrCtxInfo_.dequant_scale->GetDataType();
    auto yType = mmrCtxInfo_.y->GetDataType();
    auto pertokenScaleShape = mmrCtxInfo_.pertoken_scale_shape;
    OPS_LOG_D(opName_, "dequantScaleType %d, yType %d.", dequantScaleType, yType);
    // 1. y = bf16 时，dequantScale = bf16
    // 2. y = fp16 且 protoken 不存在时， dequantScale = int64、uint64
    // 3. y = fp16 且 protoken 存在时，dequantScale = fp32
    if (yType == ge::DT_BF16) {
        OP_TILING_CHECK(dequantScaleType != ge::DT_BF16,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "In the dequant scenario, when output type is bf16,"
                                                    " type of dequantScale should be bf16"),
                        return ge::GRAPH_FAILED);
    } else if (pertokenScaleShape == nullptr) {
        OP_TILING_CHECK(!(dequantScaleType == ge::DT_UINT64 || dequantScaleType == ge::DT_INT64),
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "In the dequant scenario, when output type is fp16,"
                                                    " type of dequantScale should be uint64 or int64 without pertoken."),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(dequantScaleType != ge::DT_FLOAT,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "In the dequant scenario, when output type is fp16,"
                                                    " type of dequantScale should be bf16 with pertoken."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTiling::CheckInput()
{
    MatmulAllReduceTilingBase::CheckInput();
    // x2 shape 为 2 维
    size_t x2DimNum = mmrCtxInfo_.x2_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(x2DimNum != DIM_NUM_TWO && x2DimNum != DIM_NUM_FOUR,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "In the dequant scenario, Expect x2 dim to be 2 or 4,"
                                                    " but got x2 dim:[%lu].", x2DimNum),
                    return ge::GRAPH_FAILED);
    // x1，x2数据类型相同
    auto x1Type = mmrCtxInfo_.x1->GetDataType();
    auto x2Type = mmrCtxInfo_.x2->GetDataType();
    OP_TILING_CHECK(x1Type != x2Type,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "In the dequant scenario, type of x1 and x2 should be same"),
                    return ge::GRAPH_FAILED);
    // bias数据类型为int32
    if (mmrCtxInfo_.bias_shape != nullptr) {
        auto biasType = mmrCtxInfo_.bias->GetDataType();
        OP_TILING_CHECK(biasType != ge::DT_INT32,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "In the dequant scenario, type of bias should be int32"),
                    return ge::GRAPH_FAILED);
    }
    // dequantScale数据类型范围
    CheckDequantScaleType();
    // comm_quant_scale不为空时校验数据类型
    if ((mmrCtxInfo_.comm_quant_scale_1_shape != nullptr) && (mmrCtxInfo_.comm_quant_scale_2_shape != nullptr)) {
        auto commQuantScaleType1 = mmrCtxInfo_.comm_quant_scale_1->GetDataType();
        auto commQuantScaleType2 = mmrCtxInfo_.comm_quant_scale_2->GetDataType();
        auto cType = mmrCtxInfo_.y->GetDataType();
        OP_TILING_CHECK((commQuantScaleType1 != cType || commQuantScaleType2 != cType),
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "The type of comm_quant_scale_1 Type(%d) or comm_quant_scale_2 Type(%d) should be"
                                                        "same to cType(%d)", static_cast<int32_t>(commQuantScaleType1),
                                                        static_cast<int32_t>(commQuantScaleType2), static_cast<int32_t>(cType)),
                        return ge::GRAPH_FAILED);
    }
    return CheckAxisSize();
}
QuantMatmulAllReduceTiling::QuantMatmulAllReduceTiling(gert::TilingContext *context)
    : MatmulAllReduceTilingBase(context), quantMatmulAllReduceTilingData_(quantMatmulAllReduceTilingDataSelf_)
{
    quantMatmulAllReduceTilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
}
const gert::Shape QuantTilingTransferHelper::GetX1Shape(const size_t index)
{
    (void) index;
    if (tilingProcesser_.args_.isATrans) {
        return gert::Shape({static_cast<int64_t>(tilingProcesser_.args_.kValue),
                            static_cast<int64_t>(tilingProcesser_.args_.mValue)});
    }
    return gert::Shape({static_cast<int64_t>(tilingProcesser_.args_.mValue),
                        static_cast<int64_t>(tilingProcesser_.args_.kValue)});
}
const gert::Shape QuantTilingTransferHelper::GetX2Shape(const size_t index)
{
    (void) index;
    if (tilingProcesser_.args_.isBTrans) {
        return gert::Shape({static_cast<int64_t>(tilingProcesser_.args_.nValue),
                            static_cast<int64_t>(tilingProcesser_.args_.kValue)});
    }
    return gert::Shape({static_cast<int64_t>(tilingProcesser_.args_.kValue),
                        static_cast<int64_t>(tilingProcesser_.args_.nValue)});
}
// 使用外部传入的tilingdata和ctxinfo
QuantMatmulAllReduceTiling::QuantMatmulAllReduceTiling(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo,
                                                       QuantMatmulAllReduceTilingData *out)
    : MatmulAllReduceTilingBase(context, mmrCtxInfo), quantMatmulAllReduceTilingData_(*out)
{
}
const gert::Shape &QuantTilingTransferHelper::GetScaleShape(const size_t index)
{
    (void) index;
    if (tilingProcesser_.mmrCtxInfo_.dequant_scale_shape == nullptr) {
        OPS_LOG_E(inputParams_.opName, "Op is quant, but has no quant shape");
        return defaultShape;
    }
    return tilingProcesser_.mmrCtxInfo_.dequant_scale_shape->GetStorageShape();
}

const gert::StorageShape *QuantTilingTransferHelper::GetPertokenShape(const size_t index)
{
    (void) index;
    if (tilingProcesser_.mmrCtxInfo_.pertoken_scale_shape != nullptr) {
        return &defaultStorageShape;
    }
    return nullptr;
}

const gert::StorageShape *QuantTilingTransferHelper::GetBiasShape(const size_t index)
{
    (void) index;
    return tilingProcesser_.mmrCtxInfo_.bias_shape;
}
ge::graphStatus QuantTilingTransferHelper::GetShapeAttrsInfo()
{
    OPS_LOG_I(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
    auto &&tilingArgs = tilingProcesser_.args_;
    inputParams_.opName = tilingProcesser_.opName_;
    inputParams_.transA = tilingArgs.isATrans;
    inputParams_.transB = tilingArgs.isBTrans;
    inputParams_.hasBias = tilingArgs.isBias;
    inputParams_.libApiWorkSpaceSize = tilingProcesser_.libApiWorkSpaceSize_;
    inputParams_.aDtype = tilingArgs.geAType;
    inputParams_.bDtype = tilingArgs.geBType;
    inputParams_.cDtype = tilingArgs.geCType;
    inputParams_.outDtype = static_cast<int64_t>(tilingArgs.geCType);
    inputParams_.biasDtype = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;
    inputParams_.scaleDtype = tilingProcesser_.mmrCtxInfo_.dequant_scale->GetDataType();
    optiling::PlatformInfo::GetInstance().intrinsic_fix_pipe_l0c2out = tilingProcesser_.supportL0c2Out_;
    AnalyzeInputs();
    inputParams_.isPerTensor
            = (tilingProcesser_.quantType_ == QuantType::PER_TENSOR);
    PrintTilingInputParam(inputParams_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantTilingTransferHelper::PostTiling()
{
    PrintTilingData();
    tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
    OPS_LOG_I(tilingProcesser_.opName_, " set mm workspace size %lu to mc2", tilingProcesser_.myWorkSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

void QuantTilingTransferHelper::PrintTilingInputParam(QuantBatchMatmulInfo quantBatchMatmulInfo)
{
    OPS_LOG_D(tilingProcesser_.opName_,
            " transA_ %d transB_ %d, hasBias_ %d",
            quantBatchMatmulInfo.transA, quantBatchMatmulInfo.transB,
            quantBatchMatmulInfo.hasBias);
    OPS_LOG_D(tilingProcesser_.opName_,
            "mSize_ %ld kSize_ %ld nSize_ %ld libApiWorkSpaceSize %u",
            quantBatchMatmulInfo.mSize, quantBatchMatmulInfo.kSize,
            quantBatchMatmulInfo.nSize,
            quantBatchMatmulInfo.libApiWorkSpaceSize);
    OPS_LOG_D(tilingProcesser_.opName_, "aDtype_ %d bDtype_ %d cDtype_ %d biasDtype_ %d outDtype %ld",
            static_cast<int32_t>(quantBatchMatmulInfo.aDtype),
            static_cast<int32_t>(quantBatchMatmulInfo.bDtype),
            static_cast<int32_t>(quantBatchMatmulInfo.cDtype),
            static_cast<int32_t>(quantBatchMatmulInfo.biasDtype),
            quantBatchMatmulInfo.outDtype);
    OPS_LOG_D(tilingProcesser_.opName_, "batchA %lu batchA1-A4[%lu:%lu:%lu:%lu];"
                                      " batchB %lu batchB1-B4[%lu:%lu:%lu:%lu]; batchC %lu; batchBias %lu",
            quantBatchMatmulInfo.batchA, quantBatchMatmulInfo.batchA1, quantBatchMatmulInfo.batchA2,
            quantBatchMatmulInfo.batchA3, quantBatchMatmulInfo.batchA4, quantBatchMatmulInfo.batchB,
            quantBatchMatmulInfo.batchB1, quantBatchMatmulInfo.batchB2, quantBatchMatmulInfo.batchB3,
            quantBatchMatmulInfo.batchB4, quantBatchMatmulInfo.batchC, quantBatchMatmulInfo.batchBias);
    OPS_LOG_D(tilingProcesser_.opName_, "isPerTensor %d",
            static_cast<int32_t>(quantBatchMatmulInfo.isPerTensor));
}
QuantTilingTransferHelper::QuantTilingTransferHelper(QuantMatmulAllReduceTiling &quantMatmulAllReduceTiling,
                                                     QuantBatchMatmulV3TilingData &data)
    : QuantBatchMatmulV3Tiling(quantMatmulAllReduceTiling.context_, &data),
    tilingProcesser_(quantMatmulAllReduceTiling)
{
}
}

#endif //_QUANT_MATMUL_ALL_REDUCE_TILING_CC_