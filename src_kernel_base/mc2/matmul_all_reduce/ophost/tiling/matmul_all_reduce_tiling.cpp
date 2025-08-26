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
 * \file matmul_all_reduce_tiling.cc
 * \brief
 */
#include "matmul_all_reduce_tiling.h"
#include <queue>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>

#include "log/ops_log.h"
#include "op_mc2.h"
#include "op_util.h"
#include "error_log.h"
#include "cube_tiling_runtime.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/quant_matmul_all_reduce_tiling.h"
#include "tiling/matmul_all_reduce_tiling_910.h"
#include "tiling/weight_quant_matmul_all_reduce_tiling.h"

#include "tiling/tiling_templates_registry.h"
#include "tiling/tiling_type.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
constexpr char HCCL_BUFFSIZE[] = "HCCL_BUFFSIZE";
// 每卡数据分配几次计算
constexpr uint64_t COMM_TILE = 8;
constexpr int64_t CORE_THRESHOLD = 20;
constexpr uint64_t LENGTH_SUM_THRESHOLD = 8192 * 3;
constexpr uint64_t MAX_SHAPE_DIM = 65535;
constexpr uint64_t MIN_SHAPE_DIM = 1;
constexpr int32_t MAX_M_SHAPE_DIM = 3;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t ADD_X3_FP16_UB_BUF_FACTOR = 6; // 对应add x3算子中FP16数据的切分数
constexpr uint32_t ADD_X3_BF16_UB_BUF_FACTOR = 10; // 对应add x3算子中BF16数据的切分数
constexpr uint32_t ALIGN_DATA_SIZE = 32;
constexpr uint64_t L2_CACHE_SIZE_910_B4 = 100663296;
constexpr uint32_t ratio = 2; // size 单位/MB
constexpr uint32_t kByte = 1024;

struct HcclAicpuOpParam {
    uint8_t res[64];
};

struct KFCMsgBody {
    // Rank* aiv * MsgSize * sizeof(消息)
    HcclAicpuOpParam msgSndArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
    HcclAicpuOpParam msgRcvArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
};

const std::map<ge::DataType, matmul_tiling::DataType> D_TYPE_MAP =
{
    {ge::DT_BF16, matmul_tiling::DataType::DT_BFLOAT16},
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
    {ge::DT_INT32, matmul_tiling::DataType::DT_INT32},
    {ge::DT_INT4, matmul_tiling::DataType::DT_INT4},
};

const std::map<ge::DataType, mc2tiling::HcclDataType> HCCL_DATA_TYPE =
{
    {ge::DataType::DT_INT8, mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT8},
    {ge::DataType::DT_UINT8, mc2tiling::HcclDataType::HCCL_DATA_TYPE_UINT8},
    {ge::DataType::DT_INT16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT16},
    {ge::DataType::DT_UINT16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_UINT16},
    {ge::DataType::DT_INT32, mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT32},
    {ge::DataType::DT_UINT32, mc2tiling::HcclDataType::HCCL_DATA_TYPE_UINT32},
    {ge::DataType::DT_FLOAT16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_FP16},
    {ge::DataType::DT_FLOAT, mc2tiling::HcclDataType::HCCL_DATA_TYPE_FP32},
    {ge::DataType::DT_BF16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_BFP16},
};

ge::graphStatus MatmulAllReduceTilingFunc(gert::TilingContext *context);
ge::graphStatus TilingParseForMatmulAllReduce(gert::TilingParseContext *context);

void MatmulAllReduceTilingBase::Reset()
{
    tileMValue_ = 0;
    tailMValue_ = 0;
    isQuantKey_ = false;
    isPerTensor_ = false;
    antiQuantType_ = AntiQuantType::NONE;
    quantType_ = QuantType::PER_TENSOR;
    antiGroupSize_ = 0;
    isUbQuant_ = false;
    enableL2Cache_ = false;
    enableBiasConvert_ = false;
    opName_ = nullptr;
    reduceOp_ = nullptr;
    rankSize_ = 8U;
    libApiWorkSpaceSize_ = 0;
    supportL0c2Out_ = false;
    isKZero_ = false;
    isA8W8_ = false;
    isA16W8_ = false;
    isA16W4_ = false;
}

void MatmulAllReduceTilingBase::DoAllReduceTiling(bool useHcclApi)
{
    auto &&args = MutableMc2MsgData();
    auto debugMode = mc2tiling::Mc2TilingUtils::GetDebugMode();
    args.set_debugMode(debugMode);
    args.set_commType(MutableRCSTilingData().get_commtype());
    args.set_reduceOp(MutableRCSTilingData().get_subtype());

    args.set_waitPolicy(1);
    args.set_rspPolicy(1);
    args.set_exitPolicy(0);
    args.set_commAlg(0);
    args.set_taskType(static_cast<uint8_t>(mc2tiling::KfcTaskType::KFC_TASK_HCC_TASK_DELIVER));

    args.set_commOrder(1); // 0先AiCPU后MM;  1为先MM后AICPU
    args.set_reuseMode(MutableRCSTilingData().get_tileCnt() + MutableRCSTilingData().get_tailCnt()); // 数据空间被使用

    // 只通信不计算模式下，如果K < N，sendOff的offset和sendCnt需要根据K计算
    auto columnNum = args_.orgNValue;
    if (debugMode == MC2_DEBUG_ONLY_AICPU && args_.orgKValue < args_.orgNValue) {
        columnNum = args_.orgKValue;
    }

    // AllReduce
    args.set_sendOff(MutableTCubeTileTilingData().get_M() * args_.orgNValue * args_.outputDtypeSize);
    args.set_recvOff(MutableTCubeTileTilingData().get_M() * columnNum * args_.outputDtypeSize);
    args.set_sendCnt(MutableTCubeTileTilingData().get_M() * args_.orgNValue);
    args.set_recvCnt(MutableTCubeTileTilingData().get_M() * columnNum);

    // 通信公式化Tiling计算中，可能有多个尾块
    args.set_tailSendOff(MutableTCubeTailTilingData().get_M() * args_.orgNValue * args_.outputDtypeSize);
    args.set_tailRecvOff(MutableTCubeTailTilingData().get_M() * columnNum * args_.outputDtypeSize);
    args.set_tailSendCnt(MutableTCubeTailTilingData().get_M() * args_.orgNValue);
    args.set_tailRecvCnt(MutableTCubeTailTilingData().get_M() * columnNum);

    // 总共发送的次数
    args.set_totalCnt(MutableRCSTilingData().get_rankM() * MutableRCSTilingData().get_rankN());
    args.set_turnNum(MutableRCSTilingData().get_tileCnt() + MutableRCSTilingData().get_tailCnt()); // 总轮次
    args.set_tailNum(MutableRCSTilingData().get_tailCnt()); // 尾块的轮次
    args.set_stride(0); // 跳写间隔

    // workspace 地址
    setUseBufferType();
    args.set_workspaceOff(libApiWorkSpaceSize_);

    // 消息队列的开始  device notify write/read value偏移
    args.set_notifyOff(sizeof(KFCMsgBody));
    args.set_notifyBeginCnt(mc2tiling::NOTIFY_WRITE_CNT); // notify write value的使用个数
    args.set_notifyEndCnt(1);   // notify read value的使用个数

    args.set_funID(mc2tiling::ALL_REDUCE_FUNC_ID);
    args.set_dataType(static_cast<uint8_t>(GetDataType(args_.geCType))); // hccl 数据类型
    args.set_groupNum(1);
    args.set_sendArgIndex(0);
    args.set_recvArgIndex(context_->GetComputeNodeInfo()->GetIrInputsNum() +
                          context_->GetComputeNodeInfo()->GetIrOutputsNum() - 1);
    OPS_LOG_I(opName_, "IR inputNum: %zu, IR outputNum: %zu", context_->GetComputeNodeInfo()->GetIrInputsNum(),
            context_->GetComputeNodeInfo()->GetIrOutputsNum());
    if (useHcclApi) {
        args.set_preparePosition(1); // 使用HCCLAPI
        args.set_hasCommOut(1);
    } else {
        args.set_preparePosition(0);
    }
}

void MatmulAllReduceTilingBase::setUseBufferType()
{
    uint8_t buffer_type;
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND910B) {
        buffer_type = static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
        OPS_LOG_I(opName_, "Set buffer type to output for non-910B soc.");
    } else if (MutableMc2MsgData().get_debugMode() == MC2_DEBUG_ONLY_AICPU) {
        buffer_type = static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
        OPS_LOG_I(opName_, "Set buffer type to output for aicpu debug mode.");
    } else if (MutableMc2MsgData().get_reuseMode() == 0) {
        buffer_type = static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
        OPS_LOG_I(opName_, "Set buffer type to output for non-reuse mode.");
    } else if (isKZero_) {
        buffer_type = static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
        OPS_LOG_I(opName_, "Set buffer type to output for empty tensor.");
    } else {
        uint64_t defaultWindowSize = 200;
        if (getenv(HCCL_BUFFSIZE) == nullptr) {
            OPS_LOG_D(opName_, "Env HCCL_BUFFSIZE don't set");
        } else {
            try {
                std::string envStr(getenv(HCCL_BUFFSIZE));
                defaultWindowSize = std::stoi(envStr);
            } catch (...) {
                OPS_LOG_E(opName_, "Unknown Exception encountered when parser env HCCL_BUFFERSIZE");
            }
        }
        const uint64_t maxWindowSize = defaultWindowSize * 1024 * 1024;
        uint64_t tileSendOff = static_cast<uint64_t>(MutableMc2MsgData().get_sendOff())
                                     * MutableRCSTilingData().get_tileCnt();
        uint64_t tailSendOff = static_cast<uint64_t>(MutableMc2MsgData().get_tailSendOff())
                                     * MutableRCSTilingData().get_tailCnt();
        if (MutableRCSTilingData().get_isInputCommQuantScale() == 1) {  // int8低bit通信做alltoall需要pad M使其可以被卡数整除
            uint64_t padTileM = MutableTCubeTileTilingData().get_M();
            uint64_t padTailM = MutableTCubeTailTilingData().get_M();
            if (padTileM % args_.rankDim != 0) {
                padTileM += args_.rankDim - (padTileM % args_.rankDim); // args_.rankDim :1/2/4/8 不会为0
            }
            tileSendOff = static_cast<uint64_t>(padTileM * MutableTCubeTileTilingData().get_N() * sizeof(uint8_t))
                            * MutableRCSTilingData().get_tileCnt();
            if (padTailM % args_.rankDim != 0) {
                padTailM += args_.rankDim - (padTailM % args_.rankDim); // args_.rankDim :1/2/4/8 不会为0
            }
            tailSendOff = static_cast<uint64_t>(padTailM * MutableTCubeTailTilingData().get_N() * sizeof(uint8_t))
                            * MutableRCSTilingData().get_tailCnt();
        }
        if (UINT64_MAX - tileSendOff < tailSendOff || tileSendOff + tailSendOff >= maxWindowSize) {
            buffer_type = static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
        } else {
            buffer_type = static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_WINDOW_IN);
        }
        OPS_LOG_I(opName_, "Set buffer type to %u, window size %lu/%lu, max %lu.",
                static_cast<uint32_t>(buffer_type), tileSendOff, tailSendOff, maxWindowSize);
    }
    MutableMc2MsgData().set_useBufferType(buffer_type);
}

void MatmulAllReduceTilingBase::DoRCSTiling()
{
    MutableRCSTilingData().set_rankDim(args_.rankDim);
    MutableRCSTilingData().set_isTransposeA(args_.isATrans);
    MutableRCSTilingData().set_isTransposeB(args_.isBTrans);
    MutableRCSTilingData().set_commtype(static_cast<uint32_t>(args_.cmdType));
    if (strncmp(reduceOp_, "sum", 3) == 0)  { // 3 is index
        OPS_LOG_D(opName_, "reduceOp_ is SUM.");
        MutableRCSTilingData().set_subtype(static_cast<uint8_t>(mc2tiling::HcclReduceOp::HCCL_REDUCE_SUM));
    } else {
        OPS_LOG_D(opName_, "reduceOp_ is RESERVED.");
        MutableRCSTilingData().set_subtype(static_cast<uint8_t>(mc2tiling::HcclReduceOp::HCCL_REDUCE_RESERVED));
    }
    OPS_LOG_D(opName_, "MatMulAllReduce DoRCSTiling, args_.orgMValue: %lu, args_.orgNValue: %lu, args_.orgKValue: %lu.",
            args_.orgMValue, args_.orgNValue, args_.orgKValue);
    MutableRCSTilingData().set_rankM(args_.orgMValue);
    MutableRCSTilingData().set_rankN(args_.orgNValue);
    MutableRCSTilingData().set_rankK(args_.orgKValue);
    MutableRCSTilingData().set_aicCoreNum(args_.aicCoreNum);
    if (MutableRCSTilingData().get_isAdd()) {
        CalcUbTiling();
    }
    SetCommQuantScale();
}

void MatmulAllReduceTilingBase::SetMCutSocVersion(SocVersion& inputSocVersion)
{
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND310P) {
        inputSocVersion = SocVersion::SOC310_P;
        OPS_LOG_D(opName_, "TileCnt enter 310P branch.");
        return;
    }
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t socMemSize = L2_CACHE_SIZE_910_B4;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, socMemSize);
    inputSocVersion = (socMemSize == L2_CACHE_SIZE_910_B4) ? SocVersion::SOC910_B4 : inputSocVersion;
    if (socMemSize == L2_CACHE_SIZE_910_B4) {
        inputSocVersion = SocVersion::SOC910_B4;
        OPS_LOG_D(opName_, "TileCnt enter 910B4 branch.");
    }
}

void MatmulAllReduceTilingBase::DoSplitMTiling()
{
    auto &&param = MutableRCSTilingData();
    if (args_.enableSplitK || isKZero_) {
        param.set_tileCnt(1);
        param.set_tailCnt(0);
        param.set_tailM(0);
    } else {
        OPS_LOG_D(opName_, "start formulaic tiling.");
        SocVersion inputSocVersion = SocVersion::SOC910_B;
        SetMCutSocVersion(inputSocVersion); // 判断是否是310P或者910B4
        MMPlusAllReduce allReduceTilingHccl(args_, args_.rankDim, KernelType::ALL_REDUCE, inputSocVersion);
        allReduceTilingHccl.GetTiling();
        CutResult mCutAllreduce = allReduceTilingHccl.tilingM_.cutRes;
        const gert::StorageShape *commQuantScaleShape1 = mmrCtxInfo_.comm_quant_scale_1_shape;
        const gert::StorageShape *commQuantScaleShape2 = mmrCtxInfo_.comm_quant_scale_2_shape;
        if (commQuantScaleShape1 != nullptr && commQuantScaleShape2 != nullptr) { // 低bit通信
            OPS_LOG_D(opName_, "TileCnt enter comm quant.");
            MMPlusQuantAllReduce quantAllReduceTilingHccl(args_, args_.rankDim, KernelType::ALL_REDUCE, inputSocVersion);
            quantAllReduceTilingHccl.GetTiling();
            mCutAllreduce = quantAllReduceTilingHccl.tilingM_.cutRes;
        }
        if (mCutAllreduce.shortTileAtBack ||
            mCutAllreduce.numShortTile == 0) {
            param.set_tileCnt(mCutAllreduce.numLongTile);
            param.set_tailM(mCutAllreduce.shortTileLen);
            tileMValue_ = mCutAllreduce.longTileLen;
            if (mCutAllreduce.numShortTile > 0) { // 有优化空间，不大于零，那就等于零
                tailMValue_ = mCutAllreduce.shortTileLen;
                param.set_tailCnt(mCutAllreduce.numShortTile);
            } else {
                param.set_tailCnt(0);
            }
        } else {
            param.set_tileCnt(mCutAllreduce.numShortTile);
            param.set_tailM(mCutAllreduce.longTileLen);
            tileMValue_ = mCutAllreduce.shortTileLen;
            if (mCutAllreduce.numLongTile > 0) {
                tailMValue_ = mCutAllreduce.longTileLen;
                param.set_tailCnt(mCutAllreduce.numLongTile);
            } else {
                param.set_tailCnt(0);
            }
        }
    }
}

void MatmulAllReduceTilingBase::SetCommQuantScale() {
    bool isInput = false;
    const gert::StorageShape *commQuantScaleShape1 = mmrCtxInfo_.comm_quant_scale_1_shape;
    const gert::StorageShape *commQuantScaleShape2 = mmrCtxInfo_.comm_quant_scale_2_shape;
    if (commQuantScaleShape1 != nullptr && commQuantScaleShape2 != nullptr) {
        isInput = true;
    }

    MutableRCSTilingData().set_isInputCommQuantScale(isInput);
    OPS_LOG_D(opName_, "is inpit comm_quant_scale_1_shape and comm_quant_scale_2_shape? %d", isInput ? 1:0);
}


ge::graphStatus MatmulAllReduceTilingBase::DoMatmulTiling(matmul_tiling::MultiCoreMatmulTiling &mm1,
                                                          TCubeTiling& cubeTiling)
{
    uint64_t mValue = args_.mValue;
    uint64_t nValue = args_.nValue;
    uint64_t kValue = args_.kValue;
    auto bmmFormat = matmul_tiling::CubeFormat::ND;
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(mmrCtxInfo_.x2->GetStorageFormat())) ==
        ge::Format::FORMAT_FRACTAL_NZ) {
        bmmFormat = matmul_tiling::CubeFormat::NZ;
        is_weight_nz_ = true;
    }
    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args_.aType, args_.isATrans);
    mm1.SetBType(matmul_tiling::TPosition::GM, bmmFormat, args_.bType, args_.isBTrans);
    mm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args_.cType);
    if (args_.isBias) {
        mm1.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args_.biasType);
        mm1.SetBias(true);
    }
    else {
        mm1.SetBias(false);
    }
    mm1.SetDim(args_.aicCoreNum);

    mm1.SetShape(mValue, nValue, kValue);
    mm1.SetOrgShape(mValue, nValue, kValue);
    mm1.SetBufferSpace(512 * 1024, -1, -1); // 512 * 1024 is buffer size
    int32_t fixCoreM = -1;
    int32_t fixCoreK = -1;
    int32_t fixCoreN = -1;
    mm1.SetSingleShape(fixCoreM, fixCoreN, fixCoreK);
    if (mm1.GetTiling(cubeTiling) == -1) {
        OPS_LOG_E(opName_, "mValue %lu, nValue %lu, kValue %lu, aicCoreNum %lu",
                mValue, nValue, kValue, args_.aicCoreNum);
        return ge::GRAPH_FAILED;
    }
    mc2tiling::MatmulFormulaicTiling reduceTiling("MatmulAllReduce");
    reduceTiling.SetSocVersion(socVersion_);
    reduceTiling.SetWeightFormat(bmmFormat);
    reduceTiling.GetCubeTiling(args_, cubeTiling);
    return ge::GRAPH_SUCCESS;
}

void MatmulAllReduceTilingBase::DoL2CacheTiling(L2cacheTilePara& l2cacheTiling)
{
    L2TilePara tileL2;
    bool enableL2Tile = CalL2TilePara(tileL2, args_.mValue, args_.kValue, args_.nValue, args_.aicCoreNum);
    enableL2Cache_ = enableL2Cache_ && enableL2Tile;
    OPS_LOG_D(opName_, "enableL2Tile %d", enableL2Tile);
    if (enableL2Tile) {
        l2cacheTiling.set_mTileCntL2(tileL2.mTile);
        l2cacheTiling.set_nTileCntL2(tileL2.nTile);
        l2cacheTiling.set_mTileBlock(tileL2.mTileBlock);
        l2cacheTiling.set_nTileBlock(tileL2.nTileBlock);
        OPS_LOG_D(opName_, "tileL2.mTile %u, tileL2.nTile %u, tileL2.mTileBlock %u, tileL2.nTileBlock %u",
                tileL2.mTile, tileL2.nTile, tileL2.mTileBlock, tileL2.nTileBlock);
    }
}

// 根据ctx填充mmrCtxInfo, 这个函数结束之后从context读的操作应该都从mmrCtxInfo读
// 根据ctxinfo做后续处理的AnalyzeShapeAttr操作, 子类重写之后也应该调用一下
ge::graphStatus MatmulAllReduceTilingBase::GetShapeAttrsInfo()
{
    ContextTransfer::AssembleMMRCtxInfoFromMMRCtx(context_, mmrCtxInfo_);
    return AnalyzeShapeAttr();
}

ge::graphStatus MatmulAllReduceTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opName_, "fail to get platform info"),
                    return ge::GRAPH_FAILED);
    std::string intrinsicName = "Intrinsic_fix_pipe_l0c2out";
    std::string val;
    (void)platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", intrinsicName.c_str(), val);
    supportL0c2Out_ = !val.empty();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    OP_TILING_CHECK(CheckRanksizePlatformSupported() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "Check Ranksize Platform Supported failed"),
                    return ge::GRAPH_FAILED);
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    auto coreNum = ascendcPlatform.GetCoreNumAic();
    args_.aicCoreNum = coreNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;

    OP_TILING_CHECK(!CheckPlatformInfo(),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "Check Platform Info failed"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Get workspace size failed"),
	    return ge::GRAPH_FAILED);

    uint32_t biasLen = 0;
    if (args_.isBias && args_.biasType == matmul_tiling::DataType::DT_BFLOAT16) {
        enableBiasConvert_ = true;
        biasLen = mc2tiling::AlignUp(args_.orgNValue, mc2tiling::SHAPE_ALIGN_SIZE) * sizeof(float);
    }
    MutableRCSTilingData().set_biasLen(biasLen);
    uint32_t mmOutInt32Len = 0;
    if (isUbQuant_) {
        uint32_t maxM = std::max(tilingData_.matmulTiling.get_M(), tilingData_.tailTiling.get_M());
        mmOutInt32Len = (maxM * tilingData_.matmulTiling.get_N()) * sizeof(int32_t);
    }
    uint32_t softSyncSize = mc2tiling::AC_MAX_AIV * 32;  // aiv_cnt * 32bytes
    workspaces[0] = libApiWorkSpaceSize_ + biasLen + softSyncSize + mmOutInt32Len;
    workspaceSize_ = workspaces[0];
    OPS_LOG_I(opName_, "libApiWorkSpaceSize=%u, biasLen=%d, softSyncSize=%u, mmOutInt32Len=%u, workspaces[0] size=%ld",
            libApiWorkSpaceSize_, biasLen, softSyncSize, mmOutInt32Len, workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::PostTiling()
{
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    PrintTilingData();
    context_->SetBlockDim(args_.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::CheckRanksizePlatformSupported() const
{
    bool rankSizeSupported =
        (socVersion_ == platform_ascendc::SocVersion::ASCEND310P &&
            (rankSize_ == 1 || rankSize_ == 2 || rankSize_ == 4)) ||
        (socVersion_ != platform_ascendc::SocVersion::ASCEND310P &&
            (rankSize_ == 1 || rankSize_ == 2 || rankSize_ == 4 || rankSize_ == 8));
    OP_TILING_CHECK(!rankSizeSupported,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                    "rank size %u is not supported by socversion id:%d yet;"
                    "Ascend 910B supports rank size 1,2,4,8"
                    "Ascend 310P supports rank size 1,2,4",
                    rankSize_, static_cast<int32_t>(socVersion_)),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool MatmulAllReduceTilingBase::AnalyzeAttrs()
{
    auto group = mmrCtxInfo_.group;
    reduceOp_ = mmrCtxInfo_.reduceOp;
    auto isTransA = mmrCtxInfo_.isTransA;
    auto isTransB = mmrCtxInfo_.isTransB;
    auto commTurn = mmrCtxInfo_.commTurn;
    if (commTurn == 0) {
        commTurn = COMM_TILE;
    }
    const int64_t *groupSizePtr = mmrCtxInfo_.groupSizePtr;
    if (groupSizePtr != nullptr) {
        antiGroupSize_ = static_cast<uint64_t>(*groupSizePtr);
    }
    rankSize_ = mc2tiling::MatmulFormulaicTiling::GetRankSize(group);

    OPS_LOG_D(opName_,
            "group is %s, rankSize_ is %u, reduceOp_ is %s, isTransA is %d, isTransB is %d,"
            "commTurn is %d antiGroupSize_ is %lu",
            group, rankSize_, reduceOp_, *isTransA, *isTransB, commTurn, antiGroupSize_);
    args_.isATrans = isTransA? *isTransA : 0;
    args_.isBTrans = isTransB? *isTransB : 0;
    args_.cmdType  = mc2tiling::AicpuComType::HCCL_CMD_ALLREDUCE;
    args_.rankDim = rankSize_;
    args_.commTurn = commTurn;
    return true;
}

void MatmulAllReduceTilingBase::SetQuantData()
{
    // 判断是否是量化场景，以及per-tensor还是per-channel
    const gert::StorageShape *matrixDequant = mmrCtxInfo_.dequant_scale_shape;
    if (matrixDequant != nullptr) {
        isQuantKey_ = true;
        const auto &dequantShape = matrixDequant->GetStorageShape();
        isPerTensor_ = (dequantShape.GetDimNum() == 1 && dequantShape.GetDim(0) == 1);
        quantType_ = isPerTensor_ ? QuantType::PER_TENSOR : QuantType::PER_CHANNEL;
    }
    OPS_LOG_D(opName_, "Tiling isQuantKey_ is %d, isPerTensor is %d quantType_ is %d", isQuantKey_ ? 1 : 0,
            isPerTensor_ ? 1 : 0, static_cast<int32_t>(quantType_));
}

void MatmulAllReduceTilingBase::SetAntiQuantData()
{
    antiQuantType_ = GetAntiQuantType();
    OPS_LOG_D(opName_, "Tiling antiQuantType_is %d", static_cast<int32_t>(antiQuantType_));
}

void MatmulAllReduceTilingBase::GetAtomicAddData()
{
    // 判断是否需要作Atomic add
    bool isAdd = false;
    const gert::StorageShape* matrixAdd = mmrCtxInfo_.x3_shape;
    if (matrixAdd != nullptr) {
        isAdd = true;
    }

    MutableRCSTilingData().set_isAdd(isAdd);
    OPS_LOG_D(opName_, "is add? %d", isAdd ? 1 : 0);
}

uint64_t MatmulAllReduceTilingBase::GetNValue() const
{
    const uint16_t maxDimNumForND = 3U;
    const uint16_t nIndexFor3Dim = maxDimNumForND - 1U;
    const uint16_t nIndexFor2Dim = maxDimNumForND - 2U;
    const gert::StorageShape* yShape = mmrCtxInfo_.y_shape;
    return yShape->GetStorageShape().GetDimNum() == maxDimNumForND ? yShape->GetStorageShape().GetDim(nIndexFor3Dim)
                                                                   : yShape->GetStorageShape().GetDim(nIndexFor2Dim);
}

uint64_t MatmulAllReduceTilingBase::GetKValue()
{
    const gert::StorageShape* inputShape = mmrCtxInfo_.x1_shape;
    size_t inputLen = inputShape->GetStorageShape().GetDimNum();
    return inputShape->GetStorageShape().GetDim(inputLen - 1);
}

uint64_t MatmulAllReduceTilingBase::GetMValue()
{
    const gert::StorageShape* inputShape = mmrCtxInfo_.x1_shape;
    size_t inputLen = inputShape->GetStorageShape().GetDimNum();
    uint64_t mValue = inputShape->GetStorageShape().GetDim(0);
    if (inputLen == MAX_M_SHAPE_DIM) {
        mValue *= inputShape->GetStorageShape().GetDim(1);
    }
    return mValue;
}

ge::graphStatus MatmulAllReduceTilingBase::CheckInput()
{
    // x1 shape 为2-3维
    size_t x1DimNum = mmrCtxInfo_.x1_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(!((x1DimNum >= DIM_NUM_TWO) && (x1DimNum <= DIM_NUM_THREE)),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "Expect x1 dim to be 2 or 3, but got x1 dim:[%lu].", x1DimNum),
                    return ge::GRAPH_FAILED);
    // bias为n值
    if (mmrCtxInfo_.bias_shape != nullptr) {
        size_t biasDimNum = mmrCtxInfo_.bias_shape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(biasDimNum != DIM_NUM_ONE,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "Expect dim of bias to be 1, but got"
                                                        " bias dim:[%lu].", biasDimNum),
                        return ge::GRAPH_FAILED);
        int64_t biasNValue = mmrCtxInfo_.bias_shape->GetStorageShape().GetDim(0);
        int64_t nValue = static_cast<int64_t>(GetNValue());
        OP_TILING_CHECK(biasNValue != nValue,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "Expect nValue of bias and output(or residual) to be same,"
                                                        " but got bias_n:[%lu], output_n:[%lu]", biasNValue, nValue),
                        return ge::GRAPH_FAILED);
    }
    // reduceOp为sum
    OP_TILING_CHECK(strncmp(mmrCtxInfo_.reduceOp, "sum", 3) != 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "Expect reduceOp to be sum."),
                        return ge::GRAPH_FAILED);
    // commTurn为0
    OP_TILING_CHECK(mmrCtxInfo_.commTurn != 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "Expect commTurn to be 0."),
                        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::CheckA16W16()
{
    OP_TILING_CHECK(((mmrCtxInfo_.antiquant_scale_shape != nullptr) ||
                     (mmrCtxInfo_.antiquant_offset_shape != nullptr) ||
                     (mmrCtxInfo_.dequant_scale_shape != nullptr)),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "when neither dtype of x1 or dtype of x2 is equal to int8,"
                                                    "antiquantScale, antiquantOffset and dequantScale "
                                                    "should be null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((mmrCtxInfo_.comm_quant_scale_1_shape != nullptr) ||
                     (mmrCtxInfo_.comm_quant_scale_2_shape != nullptr)),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "Parameter comm_quant_scale is not support in A16W16"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MatmulAllReduceTilingBase::CheckA8W8()
{
    uint64_t mValue = GetMValue();
    uint64_t nValue = GetNValue();
    OP_TILING_CHECK(!CheckDequantScaleShape(nValue),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "DequantScale shape is wrong"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckPertokenScaleShape(mValue),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "PertokenScale shape is wrong"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((mmrCtxInfo_.antiquant_scale_shape != nullptr) ||
                     (mmrCtxInfo_.antiquant_offset_shape != nullptr)),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "when both dtype of x1 and dtype of x2 are equal to int8,"
                                                    "antiquantScale, antiquantOffset should be null"),
                    return ge::GRAPH_FAILED);
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND910B) {
        OP_TILING_CHECK(!CheckCommQuantScaleShape(nValue),
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                       "CommQuantScale shape is wrong"),
                        return ge::GRAPH_FAILED);
    } else if (socVersion_ == platform_ascendc::SocVersion::ASCEND310P) {
        OP_TILING_CHECK(((mmrCtxInfo_.comm_quant_scale_1_shape != nullptr) ||
                         (mmrCtxInfo_.comm_quant_scale_2_shape != nullptr)),
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                       "Parameter comm_quant_scale is not support in A16W8"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MatmulAllReduceTilingBase::CheckA16W8()
{
    uint64_t kValue = GetKValue();
    uint64_t nValue = GetNValue();
    if (kValue == 0) {
        OPS_LOG_D(context_->GetNodeName(), "kValue equals zero. "
                "There is no need to check antiquantScale shape in the situation that tensor is empty");
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK((!CheckAntiQuantScaleShape(kValue, nValue) || mmrCtxInfo_.dequant_scale_shape != nullptr),
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "when antiquant , dequantScale should be null"),
                        return ge::GRAPH_FAILED);
    if (!CheckAntiQuantOffsetValid()) {
        OPS_LOG_E(context_->GetNodeName(), "anti quant offset input valid.");
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(((mmrCtxInfo_.comm_quant_scale_1_shape != nullptr) ||
                     (mmrCtxInfo_.comm_quant_scale_2_shape != nullptr)),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "Parameter comm_quant_scale is not support in A16W8"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool MatmulAllReduceTilingBase::SetArgs(ge::DataType aType, ge::DataType bType, ge::DataType cType,
                                        ge::DataType biasType, bool isBias)
{
    uint64_t mValue = GetMValue();
    uint64_t kValue = GetKValue();
    uint64_t nValue = GetNValue();

    uint64_t inputDtypeSize = D_TYPE_SIZE_MAP.at(aType);
    uint64_t outputDtypeSize = D_TYPE_SIZE_MAP.at(cType);

    args_.orgMValue = mValue;
    args_.orgNValue = nValue;
    args_.orgKValue = kValue;
    isKZero_ = args_.orgKValue == 0;
    args_.mValue = mValue;
    args_.nValue = nValue;
    args_.kValue = kValue;
    args_.inputDtypeSize = inputDtypeSize;
    args_.outputDtypeSize = outputDtypeSize;
    args_.enablePad = false;
    args_.enableSplitK = false;
    args_.isBias = isBias;
    args_.geAType = aType;
    args_.geBType = bType;
    args_.geCType = cType;
    args_.geBiasType = biasType;
    try {
        args_.aType = D_TYPE_MAP.at(aType);
        args_.bType = D_TYPE_MAP.at(bType);
        args_.cType = D_TYPE_MAP.at(cType);
        args_.biasType = D_TYPE_MAP.at(biasType);
    } catch (const std::out_of_range &e) {
        OPS_LOG_E(opName_, "Unsupported{ aType: %d bType: %d cType: %d biasType %d }.", static_cast<int32_t>(aType),
                static_cast<int32_t>(bType), static_cast<int32_t>(cType), static_cast<int32_t>(biasType));
        return false;
    }

    OPS_LOG_D(opName_, "aType: %d bType: %d cType: %d biasType %d.", static_cast<int32_t>(args_.aType),
            static_cast<int32_t>(args_.bType), static_cast<int32_t>(args_.cType), static_cast<int32_t>(args_.biasType));
    return true;
}

bool MatmulAllReduceTilingBase::AnalyzeInputs()
{
    ge::DataType biasType;
    bool isBias = true;

    auto aType = mmrCtxInfo_.x1->GetDataType();
    auto bType = mmrCtxInfo_.x2->GetDataType();
    auto cType = mmrCtxInfo_.y->GetDataType();
    const gert::StorageShape *antiquantScale = mmrCtxInfo_.antiquant_scale_shape;
    const gert::StorageShape *dequantScale = mmrCtxInfo_.dequant_scale_shape;

    isA8W8_ = (aType == ge::DT_INT8 && bType ==ge::DT_INT8 && (dequantScale != nullptr));
    isA16W8_ = (aType != ge::DT_INT8 && bType ==ge::DT_INT8 && (antiquantScale != nullptr));
    isA16W4_ = (aType != ge::DT_INT8 && bType ==ge::DT_INT4 && (antiquantScale != nullptr));
    OPS_LOG_D(opName_, "isA8W8_ is %d, isA16W8_ is %d, isA16W4_ is %d", isA8W8_ ? 1 : 0, isA16W8_ ? 1 : 0,
            isA16W4_ ? 1 : 0);

    auto dequantDesc = mmrCtxInfo_.dequant_scale;
    isUbQuant_ = ((dequantDesc != nullptr) ? (dequantDesc->GetDataType() == ge::DT_BF16) : isUbQuant_);
    OPS_LOG_D(opName_, "MatmulAllReduceTilingBase::AnalyzeInput isUbQuant_: %d ", isUbQuant_ ? 1 : 0);
    const gert::StorageShape* matrixBias = mmrCtxInfo_.bias_shape;
    if (matrixBias == nullptr) {
        isBias = false;
        biasType = cType;
    } else {
        biasType = mmrCtxInfo_.bias->GetDataType();
    }

    SetQuantData();
    SetAntiQuantData();
    GetAtomicAddData();

    size_t yDimNum = mmrCtxInfo_.y_shape->GetStorageShape().GetDimNum();
    OPS_LOG_D(context_->GetNodeName(), "dim of output is %lu.", yDimNum);
    OP_TILING_CHECK((yDimNum < DIM_NUM_TWO || yDimNum > DIM_NUM_THREE),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "Expect output(residual) dim to be 2 or 3, but got output"
                                                    " dim:[%lu].", yDimNum),
                    return false);
    return SetArgs(aType, bType, cType, biasType, isBias);
}

mc2tiling::HcclDataType MatmulAllReduceTilingBase::GetDataType(ge::DataType type)
{
    if (HCCL_DATA_TYPE.find(type) != HCCL_DATA_TYPE.end()) {
        return HCCL_DATA_TYPE.at(type);
    }
    return mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED;
}

bool MatmulAllReduceTilingBase::CalL2TilePara(L2TilePara& tileL2, uint64_t mValue, uint64_t kValue,
                                              uint64_t nValue, uint32_t cubeCoreNum)
{
    uint64_t blockBaseM = 1;
    uint64_t blockBaseN = 1;
    uint64_t blockBaseK = 1;
    mc2tiling::MatmulFormulaicTiling::GetBaseBlockParm(
        socVersion_, blockBaseM, blockBaseN, blockBaseK, blockBaseK, blockBaseK);
    // GetBaseBlockParm方法实际上只会给blockBaseM赋值成128或256，此处防止下文除0
    if (blockBaseM == 0ULL || blockBaseN == 0ULL) {
        OPS_LOG_E(opName_, "matmulAllreduce get base block parm fail");
        return false;
    }
    uint64_t sizeA = mValue * kValue * D_MTYPE_SIZE_MAP.at(args_.aType) / kByte / kByte;
    uint64_t sizeB = kValue * nValue * D_MTYPE_SIZE_MAP.at(args_.bType) / kByte / kByte;
    uint64_t sizeC = mValue * nValue * D_MTYPE_SIZE_MAP.at(args_.cType) / kByte / kByte;
    uint64_t totalSize = sizeA + sizeB + sizeC;
    uint64_t l2CacheSize = 0;
    uint64_t singleMatrixSize = 0;
    uint32_t tileSize = 0;
    uint32_t tileLimit = 0;
    bool useNewPara = (cubeCoreNum > CORE_THRESHOLD && (mValue + kValue + nValue) < LENGTH_SUM_THRESHOLD);
    GetL2CacheParm(l2CacheSize, singleMatrixSize, tileSize, tileLimit, useNewPara);
    if (totalSize >= l2CacheSize || sizeA >= singleMatrixSize ||
        sizeB >= singleMatrixSize || sizeC >= singleMatrixSize) {
        // 仅考虑fp16场景
        tileL2.mTileBlock = (tileSize * kByte * kByte / ratio / kValue + blockBaseM - 1) / blockBaseM;
        tileL2.nTileBlock = (tileSize * kByte * kByte / ratio / kValue + blockBaseN - 1) / blockBaseN;
        // 该处保证L2cache切分的合法性，否则kernel侧切分策略不合法，会导致地址溢出
        tileL2.mTile = (mValue + tileL2.mTileBlock * blockBaseM - 1) / (tileL2.mTileBlock * blockBaseM);
        tileL2.nTile = (nValue + tileL2.nTileBlock * blockBaseN - 1) / (tileL2.nTileBlock * blockBaseN);
        // 如果A或B实际小于tileSize，导致切分的mTileBlock太大，此时可以不用切分
        if (tileL2.mTileBlock >= (mValue / blockBaseM)) {
            tileL2.mTile = 1;
            tileL2.mTileBlock = (mValue + blockBaseM - 1) / blockBaseM;
        }
        if (tileL2.nTileBlock >= (nValue / blockBaseN)) {
            tileL2.nTile = 1;
            tileL2.nTileBlock = (nValue + blockBaseN - 1) / blockBaseN;
        }
        tileL2.mTile = (tileL2.mTile > 0 && sizeA > tileLimit) ? tileL2.mTile : 1;
        tileL2.nTile = (tileL2.nTile > 0 && sizeB > tileLimit) ? tileL2.nTile : 1;
        if (!is_weight_nz_ && (tileL2.mTileBlock * tileL2.nTileBlock < cubeCoreNum)) {
            return false;
        }
        if ((tileL2.mTile > 1 || tileL2.nTile > 1) && mValue >= blockBaseM) {
            return true;
        }
    }
    return false;
}

void MatmulAllReduceTilingBase::PrintTilingData(optiling::TCubeTiling& tiling)
{
    OPS_LOG_I(opName_, " tiling.usedCoreNum %d", tiling.get_usedCoreNum());
    OPS_LOG_I(opName_, " tiling.M %d", tiling.get_M());
    OPS_LOG_I(opName_, " tiling.N %d", tiling.get_N());
    OPS_LOG_I(opName_, " tiling.Ka %d", tiling.get_Ka());
    OPS_LOG_I(opName_, " tiling.Kb %d", tiling.get_Kb());
    OPS_LOG_I(opName_, " tiling.singleCoreM %d", tiling.get_singleCoreM());
    OPS_LOG_I(opName_, " tiling.singleCoreN %d", tiling.get_singleCoreN());
    OPS_LOG_I(opName_, " tiling.singleCoreK %d", tiling.get_singleCoreK());
    OPS_LOG_I(opName_, " tiling.baseM %d", tiling.get_baseM());
    OPS_LOG_I(opName_, " tiling.baseN %d", tiling.get_baseN());
    OPS_LOG_I(opName_, " tiling.baseK %d", tiling.get_baseK());
    OPS_LOG_I(opName_, " tiling.depthA1 %d", tiling.get_depthA1());
    OPS_LOG_I(opName_, " tiling.depthB1 %d", tiling.get_depthB1());
    OPS_LOG_I(opName_, " tiling.stepM %d", tiling.get_stepM());
    OPS_LOG_I(opName_, " tiling.stepN %d", tiling.get_stepN());
    OPS_LOG_I(opName_, " tiling.stepka %d", tiling.get_stepKa());
    OPS_LOG_I(opName_, " tiling.stepkb %d", tiling.get_stepKb());
    OPS_LOG_I(opName_, " tiling.isBias %d", tiling.get_isBias());
    OPS_LOG_I(opName_, " tiling.transLength %d", tiling.get_transLength());
    OPS_LOG_I(opName_, " tiling.iterateOrder %s", ((tiling.get_iterateOrder() == 1)? "orderM" : "orderN"));
    OPS_LOG_I(opName_, " tiling.usedL1Size %d", tiling.get_shareL1Size());
    OPS_LOG_I(opName_, " tiling.usedL0CSize %d", tiling.get_shareL0CSize());
    OPS_LOG_I(opName_, " tiling.usedUBSize %d", tiling.get_shareUbSize());
    OPS_LOG_I(opName_, " tiling.batchM %d", tiling.get_batchM());
    OPS_LOG_I(opName_, " tiling.batchN %d", tiling.get_batchN());
    OPS_LOG_I(opName_, " tiling.singleBatchM %d", tiling.get_singleBatchM());
    OPS_LOG_I(opName_, " tiling.singleBatchN %d", tiling.get_singleBatchN());
}

void MatmulAllReduceTilingBase::PrintTilingData(optiling::RCSTiling& tiling)
{
    OPS_LOG_I(opName_, " tiling.commtype %d", tiling.get_commtype());
    OPS_LOG_I(opName_, " tiling.subtype %d", tiling.get_subtype());
    OPS_LOG_I(opName_, " tiling.rankDim %d", tiling.get_rankDim());
    OPS_LOG_I(opName_, " tiling.rankID %d", tiling.get_rankID());
    OPS_LOG_I(opName_, " tiling.tileCnt %d", tiling.get_tileCnt());
    OPS_LOG_I(opName_, " tiling.tailM %d", tiling.get_tailM());
    OPS_LOG_I(opName_, " tiling.tailCnt %d", tiling.get_tailCnt());
    OPS_LOG_I(opName_, " tiling.isTransA %d", tiling.get_isTransposeA());
    OPS_LOG_I(opName_, " tiling.isTransB %d", tiling.get_isTransposeB());
    OPS_LOG_I(opName_, " tiling.get_rankM %d", tiling.get_rankM());
    OPS_LOG_I(opName_, " tiling.get_rankN %d", tiling.get_rankN());
    OPS_LOG_I(opName_, " tiling.get_rankK %d", tiling.get_rankK());
    OPS_LOG_I(opName_, " tiling.gatherIndex %d", tiling.get_gatherIndex());
    OPS_LOG_I(opName_, " tiling.cToFloatLen %lu", tiling.get_cToFloatLen());
    OPS_LOG_I(opName_, " tiling.nd2NzWorkLen %lu", tiling.get_nd2NzWorkLen());
    OPS_LOG_I(opName_, " tiling.gatherLen %lu", tiling.get_gatherLen());
    OPS_LOG_I(opName_, " tiling.aicCoreNum %d", tiling.get_aicCoreNum());
}

void MatmulAllReduceTilingBase::PrintTilingData(optiling::Mc2Msg& msg)
{
    OPS_LOG_I(opName_, " msg.sendOff %lu", msg.get_sendOff());
    OPS_LOG_I(opName_, " msg.recvOff %lu", msg.get_recvOff());
    OPS_LOG_I(opName_, " msg.tailSendOff %lu", msg.get_tailSendOff());
    OPS_LOG_I(opName_, " msg.tailRecvOff %lu", msg.get_tailRecvOff());
    OPS_LOG_I(opName_, " msg.sendCnt %lu", msg.get_sendCnt());
    OPS_LOG_I(opName_, " msg.recvCnt %lu", msg.get_recvCnt());
    OPS_LOG_I(opName_, " msg.tailSendCnt %lu", msg.get_tailSendCnt());
    OPS_LOG_I(opName_, " msg.tailRecvCnt %lu", msg.get_tailRecvCnt());
    OPS_LOG_I(opName_, " msg.totalCnt %lu", msg.get_totalCnt());
    OPS_LOG_I(opName_, " msg.turnNum %u", msg.get_turnNum());
    OPS_LOG_I(opName_, " msg.tailNum %u", msg.get_tailNum());
    OPS_LOG_I(opName_, " msg.stride %u", msg.get_stride());
    OPS_LOG_I(opName_, " msg.workspaceOff %u", msg.get_workspaceOff());
    OPS_LOG_I(opName_, " msg.notifyOff %u", msg.get_notifyOff());

    OPS_LOG_I(opName_, " msg.notifyBeginCnt %u", msg.get_notifyBeginCnt());
    OPS_LOG_I(opName_, " msg.notifyEndCnt %u", msg.get_notifyEndCnt());
    OPS_LOG_I(opName_, " msg.useBufferType %u", static_cast<uint32_t>(msg.get_useBufferType()));
    OPS_LOG_I(opName_, " msg.funID %u", static_cast<uint32_t>(msg.get_funID()));
    OPS_LOG_I(opName_, " msg.dataType %u", static_cast<uint32_t>(msg.get_dataType()));
    OPS_LOG_I(opName_, " msg.groupNum %u", static_cast<uint32_t>(msg.get_groupNum()));

    OPS_LOG_I(opName_, " msg.reuseMode %u", static_cast<uint32_t>(msg.get_reuseMode()));
    OPS_LOG_I(opName_, " msg.commType %u", static_cast<uint32_t>(msg.get_commType()));
    OPS_LOG_I(opName_, " msg.reduceOp %u", static_cast<uint32_t>(msg.get_reduceOp()));
    OPS_LOG_I(opName_, " msg.commOrder %u", static_cast<uint32_t>(msg.get_commOrder()));
    OPS_LOG_I(opName_, " msg.waitPolicy %u", static_cast<uint32_t>(msg.get_waitPolicy()));
    OPS_LOG_I(opName_, " msg.rspPolicy %u", static_cast<uint32_t>(msg.get_rspPolicy()));
    OPS_LOG_I(opName_, " msg.exitPolicy %u", static_cast<uint32_t>(msg.get_exitPolicy()));

    OPS_LOG_I(opName_, " msg.commAlg %u", static_cast<uint32_t>(msg.get_commAlg()));
    OPS_LOG_I(opName_, " msg.taskType %u", static_cast<uint32_t>(msg.get_taskType()));
    OPS_LOG_D(opName_, " msg.preparePosition %u", msg.get_preparePosition());
}

bool MatmulAllReduceTilingBase::HasAntiQuantOffset() const
{
    return mmrCtxInfo_.antiquant_offset != nullptr;
}

bool MatmulAllReduceTilingBase::CheckDequantScaleShape(const uint64_t nValue) const
{
    const auto bias = mmrCtxInfo_.bias_shape;
    if (bias != nullptr) {
        const auto scaleShapeSize = static_cast<size_t>(bias->GetStorageShape().GetShapeSize());
        uint64_t dimNum = bias->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            dimNum != 1U || scaleShapeSize != nValue,
            VECTOR_INNER_ERR_REPORT_TILIING(opName_,
                                            "Expected shape of bias is [n] where n is %lu in current case, "
                                            "but got bias shape: %s, bias dim num is: %lu.",
                                            nValue, ge::Shape2String(bias->GetStorageShape()).c_str(), dimNum),
            return false);
    } else {
        OPS_LOG_D(context_->GetNodeName(), "No Bias.");
    }

    const auto scale = mmrCtxInfo_.dequant_scale_shape;
    OP_TILING_CHECK(scale == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opName_, "DequantScale is nullptr "),
                    return false);
    uint64_t scaleDimNum = scale->GetStorageShape().GetDimNum();
    OPS_LOG_D(context_->GetNodeName(), "dim of scale is %lu.", scaleDimNum);
    OP_TILING_CHECK(scaleDimNum > DIM_NUM_TWO,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "DequantScale dim should be 1 or 2, but got"
                                                    " dequantScale dim num is: %lu", scaleDimNum),
                    return false);
    const auto scaleShapeSize = static_cast<size_t>(scale->GetStorageShape().GetShapeSize());
    if (scaleShapeSize == 1) {
        return true;
    }
    OP_TILING_CHECK(
        scaleShapeSize != nValue,
        VECTOR_INNER_ERR_REPORT_TILIING(opName_,
                                        "Expected shape of dequantScale to be [1] or [n] or [1,n] for "
                                        "per-tensor/per-channel. n is %lu in these cases, "
                                        "but got scale shape: %s.",
                                        nValue, ge::Shape2String(scale->GetStorageShape()).c_str()),
        return false);
    return true;
}

bool MatmulAllReduceTilingBase::CheckPertokenScaleShape(const uint64_t mValue) const
{
    const auto pertokenScale = mmrCtxInfo_.pertoken_scale_shape;
    if (pertokenScale == nullptr) {
        return true;
    }

    uint64_t pertokenScaleDimNum = pertokenScale->GetStorageShape().GetDimNum();
    OPS_LOG_D(opName_, "dim of pertokenScale is %lu.", pertokenScaleDimNum);
    OP_TILING_CHECK(pertokenScaleDimNum > DIM_NUM_ONE,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "PertokenScale dim should be 1, but got"
                                                    " pertokenScale dim num is: %lu", pertokenScaleDimNum),
                    return false);

    const auto pertokenScaleShapeSize = static_cast<size_t>(pertokenScale->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(
        pertokenScaleShapeSize != mValue,
        VECTOR_INNER_ERR_REPORT_TILIING(opName_,
                                        "Expected shape of pertokenScale to be [m]."
                                        "m is %lu in these cases, "
                                        "but got pertokenScale shape: %s.",
                                        mValue, ge::Shape2String(pertokenScale->GetStorageShape()).c_str()),
        return false);
    return true;
}

bool MatmulAllReduceTilingBase::CheckCommQuantScaleShape(const uint64_t nValue) const
{
    const auto commQuantScale1Shape = mmrCtxInfo_.comm_quant_scale_1_shape;
    const auto commQuantScale2Shape = mmrCtxInfo_.comm_quant_scale_2_shape;
    if ((commQuantScale1Shape == nullptr) && (commQuantScale2Shape == nullptr)) {
        return true;
    }
    OP_TILING_CHECK((commQuantScale1Shape == nullptr) || (commQuantScale2Shape == nullptr) ,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "comm_quant_scale_1 or comm_quant_scale_2 dim is nullpter"),
                    return false);

    uint64_t commQuantScaleOneDimNum = commQuantScale1Shape->GetStorageShape().GetDimNum();
    uint64_t commQuantScaleTwoDimNum = commQuantScale2Shape->GetStorageShape().GetDimNum();
    OPS_LOG_D(opName_, "dim of comm_quant_scale_1 and comm_quant_scale_2 is %lu and %lu", commQuantScaleOneDimNum, commQuantScaleTwoDimNum);
    OP_TILING_CHECK((commQuantScaleOneDimNum > DIM_NUM_TWO) || (commQuantScaleTwoDimNum > DIM_NUM_TWO) ,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "comm_quant_scale_1 and comm_quant_scale_2 dim should be 1 or 2, but got"
                                                    "comm_quant_scale_1 dim is: %lu, comm_quant_scale_2 dim is: %lu",
                                                    commQuantScaleOneDimNum, commQuantScaleTwoDimNum),
                    return false);

    const auto commQuantScaleShapeSize1 = static_cast<size_t>(commQuantScale1Shape->GetStorageShape().GetShapeSize());
    const auto commQuantScaleShapeSize2 = static_cast<size_t>(commQuantScale2Shape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK((commQuantScaleShapeSize1 != nValue) || (commQuantScaleShapeSize2 != nValue) ,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "comm_quant_scale_1 and comm_quant_scale_2 dim should be [n],"
                                                    "n is %lu in these case,"
                                                    "but got comm_quant_scale_1 shape is: %s, comm_quant_scale_2 shape is: %s",
                                                    nValue, ge::Shape2String(commQuantScale1Shape->GetStorageShape()).c_str(),
                                                    ge::Shape2String(commQuantScale2Shape->GetStorageShape()).c_str()),
                    return false);
    return true;
}


bool MatmulAllReduceTilingBase::CheckAntiQuantScaleShape(const uint64_t kValue, const uint64_t nValue)
{
    const auto scale = mmrCtxInfo_.antiquant_scale_shape;
    if (scale == nullptr) {
        OPS_LOG_D(context_->GetNodeName(), "No antiquantScale.");
        return false;
    }
    // 校验scale维度
    int32_t twoDim = 2;
    int32_t scaleShapeDim = scale->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(scaleShapeDim != 1 && scaleShapeDim != twoDim,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
        "Dim size of MatmulAllReduce weight quant antiquantScale param must be 1 or 2."), return false);
    const auto scaleShapeSize = static_cast<size_t>(scale->GetStorageShape().GetShapeSize());
    OPS_LOG_D(context_->GetNodeName(), "scaleShapeSize %lu, antiGroupSize_ %lu", scaleShapeSize, antiGroupSize_);
    if (scaleShapeSize == 1) {
        OP_TILING_CHECK(antiGroupSize_ != 0,
            VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "when scale shape size is 1, antigroupsize must be 0."), return false);
        return true;
    } else if (antiGroupSize_ > 0) {
        OP_TILING_CHECK(kValue < 33, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "in per-group, the kValue must be greater than 33."), return false);
        return true;
    } else {
        OP_TILING_CHECK(
            scaleShapeSize != nValue,
            VECTOR_INNER_ERR_REPORT_TILIING(opName_,
                                            "Expected shape of antiquantScale to be [1] or [n] or [1,n] for "
                                            "per-tensor/per-channel. n is %lu in these cases, "
                                            "but got scale shape: %s.",
                                            nValue, ge::Shape2String(scale->GetStorageShape()).c_str()),
            return false);
        return true;
    }
    return false;
}

bool MatmulAllReduceTilingBase::CheckAntiQuantOffsetValid() const
{
    const auto scale = mmrCtxInfo_.antiquant_scale_shape;
    if (scale == nullptr) {
        OPS_LOG_E(context_->GetNodeName(), "No antiquantScale.");
        return false;
    }
    // 校验bias dim
    const gert::StorageShape* matrixBias = mmrCtxInfo_.bias_shape;
    int32_t biasShapeSize = 0;
    int64_t nValue = GetNValue();
    if (matrixBias != nullptr) {
        biasShapeSize = matrixBias->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(biasShapeSize != 1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "Dim size of MatmulAllReduce weight quant bias must be 1."), return false);
        // 校验bias和x2最后一维是否一致
        OP_TILING_CHECK(matrixBias->GetStorageShape().GetDim(biasShapeSize - 1) != nValue,
            VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "Bias size must be the same size of last dim of x2."), return false);
    }
    // 校验offset维度 与 scale一致
    const gert::StorageShape *antiquantOffset = mmrCtxInfo_.antiquant_offset_shape;
    if (antiquantOffset != nullptr) {
        int32_t offsetShapeDim = antiquantOffset->GetStorageShape().GetDimNum();
        int32_t scaleShapeDim = scale->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(offsetShapeDim != scaleShapeDim, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "The offset dim must be the same of the scale dim."), return false);
        // 校验offset和x2最后一维是否一致
        for (int32_t i = 0; i < offsetShapeDim; ++i) {
            int64_t offsetValue = antiquantOffset->GetStorageShape().GetDim(i);
            int64_t scaleValue = scale->GetStorageShape().GetDim(i);
            OP_TILING_CHECK(offsetValue != scaleValue, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "Offset shape must be the same of scale shape."), return false);
        }
    }
    return true;
}

bool MatmulAllReduceTilingBase::CheckA16W4Shape(const uint64_t kValue, const uint64_t nValue)
{
    uint64_t innerN = (MutableRCSTilingData().get_isTransposeB() != 0) ? kValue : nValue;
    OP_TILING_CHECK((innerN & 1) != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_,
                    "In the int4 scenario, the inner shaft of x2 should be an even number. k[%lu], n[%lu]",
                    kValue, nValue),
                    return false);
    return true;
}

bool MatmulAllReduceTilingBase::CheckPlatformInfo() const
{
    if (isA16W8_ || isA16W4_) {
        OP_TILING_CHECK(supportL0c2Out_ && ((args_.mValue < MIN_SHAPE_DIM)
                        || (args_.kValue != 0 && (args_.kValue > MAX_SHAPE_DIM || args_.kValue < MIN_SHAPE_DIM))
                        || (args_.nValue > MAX_SHAPE_DIM || args_.nValue < MIN_SHAPE_DIM)),
                        VECTOR_INNER_ERR_REPORT_TILIING(opName_,
                        "only support MKN in range [%lu, %lu], get actual value[%lu, %lu, %lu]",
                        MIN_SHAPE_DIM, MAX_SHAPE_DIM, args_.mValue, args_.kValue, args_.nValue),
                        return false);
    }
    return true;
}

AntiQuantType MatmulAllReduceTilingBase::GetAntiQuantType()
{
    const auto scale = mmrCtxInfo_.antiquant_scale_shape;
    if (scale == nullptr) {
        OPS_LOG_D(context_->GetNodeName(), "No anti quant scale.");
        return AntiQuantType::NONE;
    }
    const auto scaleShapeSize = static_cast<size_t>(scale->GetStorageShape().GetShapeSize());
    OPS_LOG_D(context_->GetNodeName(), "Scale shape size %zu antiGroupSize_ %zu", scaleShapeSize, antiGroupSize_);
    if (scaleShapeSize == 1) {
        return AntiQuantType::PER_TENSOR;
    } else if (antiGroupSize_ > 0) {
        return AntiQuantType::PER_GROUP;
    } else {
        return AntiQuantType::PER_CHANNEL;
    }
    return AntiQuantType::NONE;
}

void MatmulAllReduceTilingBase::CalcUbTiling() const
{
    uint32_t addX3UbBufFac = (args_.geCType == ge::DT_BF16) ? ADD_X3_BF16_UB_BUF_FACTOR : ADD_X3_FP16_UB_BUF_FACTOR;
    addX3UbBufFac *= sizeof(int16_t);
    uint32_t addX3UbCnt = mc2tiling::AlignDown(static_cast<uint32_t>((aicoreParams_.ubSize) / addX3UbBufFac),
        ALIGN_DATA_SIZE);
    tilingData_.param.set_addX3UbCnt(addX3UbCnt);
}
ge::graphStatus MatmulAllReduceTilingBase::AnalyzeShapeAttr()
{
    opName_ = context_->GetNodeName();
    OP_TILING_CHECK(!AnalyzeAttrs() || !AnalyzeInputs(),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "fail to analyze context info"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
void MatmulAllReduceTilingBase::PrintTilingData()
{
    if (MutableRCSTilingData().get_rankID() != 0) {
        return;
    }
    PrintTilingData(MutableRCSTilingData());
    PrintTilingData(MutableTCubeTileTilingData());
    PrintTilingData(MutableMc2MsgData());
    if (MutableRCSTilingData().get_tailM() <= 0) {
        return;
    }
    OPS_LOG_D(opName_, "have tail");
    PrintTilingData(MutableTCubeTailTilingData());
}
REGISTER_TILING_TEMPLATE("MatmulAllReduce", QuantMatmulAllReduceTiling, 0);
REGISTER_TILING_TEMPLATE("MatmulAllReduce", WeightQuantMatmulAllReduceTiling, 1);
REGISTER_TILING_TEMPLATE("MatmulAllReduce", MatmulAllReduceTiling910, 2);

ge::graphStatus MatmulAllReduceTilingFunc(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

struct MatmulAllReduceCompileInfo {};
ge::graphStatus TilingParseForMatmulAllReduce(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulAllReduce)
    .Tiling(MatmulAllReduceTilingFunc)
    .TilingParse<MatmulAllReduceCompileInfo>(TilingParseForMatmulAllReduce);
}  // namespace optiling
