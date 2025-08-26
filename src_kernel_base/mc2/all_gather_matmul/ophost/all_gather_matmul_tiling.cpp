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
 * \file all_gather_matmul_tiling.cpp
 * \brief
 */
#include "vector"
#include "../../common/ophost/matmul_formulaic_tiling.h"
#include "all_gather_formulaic_tiling.h"
#include "log/ops_log.h"
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "../../common/ophost/mc2_tiling_utils.h"
#include "../../common/ophost/mc2_tiling_struct.h"
#include "../../common/ophost/op_util.h"
#include "../all_gather_matmul_tiling.h"

using namespace AscendC;
using namespace ge;

namespace {
const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK = {
    {0, {2, 4, 8}}
    };
constexpr uint32_t TILINGKEY_BIAS = 1U;
constexpr uint32_t TILINGKEY_ND2NZ = 10U;
constexpr uint32_t TILINGKEY_FULL_MESH = 100U;

static void PrintTilingData(::TCubeTiling& tiling)
{
    OPS_LOG_D("AllGatherMatmul", " tiling.usedCoreNum %d", tiling.usedCoreNum);
    OPS_LOG_D("AllGatherMatmul", " tiling.M %d", tiling.M);
    OPS_LOG_D("AllGatherMatmul", " tiling.N %d", tiling.N);
    OPS_LOG_D("AllGatherMatmul", " tiling.Ka %d", tiling.Ka);
    OPS_LOG_D("AllGatherMatmul", " tiling.Kb %d", tiling.Kb);
    OPS_LOG_D("AllGatherMatmul", " tiling.singleCoreM %d", tiling.singleCoreM);
    OPS_LOG_D("AllGatherMatmul", " tiling.singleCoreN %d", tiling.singleCoreN);
    OPS_LOG_D("AllGatherMatmul", " tiling.singleCoreK %d", tiling.singleCoreK);
    OPS_LOG_D("AllGatherMatmul", " tiling.baseM %d", tiling.baseM);
    OPS_LOG_D("AllGatherMatmul", " tiling.baseN %d", tiling.baseN);
    OPS_LOG_D("AllGatherMatmul", " tiling.baseK %d", tiling.baseK);
    OPS_LOG_D("AllGatherMatmul", " tiling.depthA1 %d", tiling.depthA1);
    OPS_LOG_D("AllGatherMatmul", " tiling.depthB1 %d", tiling.depthB1);
    OPS_LOG_D("AllGatherMatmul", " tiling.stepM %d", tiling.stepM);
    OPS_LOG_D("AllGatherMatmul", " tiling.stepN %d", tiling.stepN);
    OPS_LOG_D("AllGatherMatmul", " tiling.stepka %d", tiling.stepKa);
    OPS_LOG_D("AllGatherMatmul", " tiling.stepkb %d", tiling.stepKb);
    OPS_LOG_D("AllGatherMatmul", " tiling.isBias %d", tiling.isBias);
    OPS_LOG_D("AllGatherMatmul", " tiling.transLength %d", tiling.transLength);
    OPS_LOG_D("AllGatherMatmul", " tiling.iterateOrder %s", ((tiling.iterateOrder == 1) ? "orderM" : "orderN"));
    OPS_LOG_D("AllGatherMatmul", " tiling.usedL1Size %d", tiling.shareL1Size);
    OPS_LOG_D("AllGatherMatmul", " tiling.usedL0CSize %d", tiling.shareL0CSize);
    OPS_LOG_D("AllGatherMatmul", " tiling.dbL0C %d", tiling.dbL0C);
    OPS_LOG_D("AllGatherMatmul", " tiling.usedUBSize %d", tiling.shareUbSize);
    OPS_LOG_D("AllGatherMatmul", " tiling.batchM %d", tiling.batchM);
    OPS_LOG_D("AllGatherMatmul", " tiling.batchN %d", tiling.batchN);
    OPS_LOG_D("AllGatherMatmul", " tiling.singleBatchM %d", tiling.singleBatchM);
    OPS_LOG_D("AllGatherMatmul", " tiling.singleBatchN %d", tiling.singleBatchN);
}

static void PrintTilingData(::RCSTiling& rcsTiling)
{
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.commtype %u", rcsTiling.commtype);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.subtype %u", rcsTiling.subtype);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.rankDim %u", rcsTiling.rankDim);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.rankID %u", rcsTiling.rankID);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.tileCnt %u", rcsTiling.tileCnt);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.tailM %u", rcsTiling.tailM);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.tailCnt %u", rcsTiling.tailCnt);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.isTransA %u", rcsTiling.isTransposeA);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.isTransB %u", rcsTiling.isTransposeB);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.rankM %u", rcsTiling.rankM);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.rankN %u", rcsTiling.rankN);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.rankK %u", rcsTiling.rankK);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.gatherIndex %u", rcsTiling.gatherIndex);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.cToFloatLen %lu", rcsTiling.cToFloatLen);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.nd2NzWorkLen %lu", rcsTiling.nd2NzWorkLen);
    OPS_LOG_D("AllGatherMatmul", " rcsTiling.gatherLen %lu", rcsTiling.gatherLen);
}

static void PrintTilingData(::TileL2Tiling& tileL2Tiling)
{
    OPS_LOG_D("AllGatherMatmul", " tileL2Tiling.mL2TileCnt %u", tileL2Tiling.mL2TileCnt);
    OPS_LOG_D("AllGatherMatmul", " tileL2Tiling.nL2TileCnt %u", tileL2Tiling.nL2TileCnt);
    OPS_LOG_D("AllGatherMatmul", " tileL2Tiling.mTileBlocks %u", tileL2Tiling.mTileBlocks);
    OPS_LOG_D("AllGatherMatmul", " tileL2Tiling.nTileBlocks %u", tileL2Tiling.nTileBlocks);
    OPS_LOG_D("AllGatherMatmul", " tileL2Tiling.mTailBlocks %u", tileL2Tiling.mTailBlocks);
    OPS_LOG_D("AllGatherMatmul", " tileL2Tiling.nTailBlocks %u", tileL2Tiling.nTailBlocks);
    OPS_LOG_D("AllGatherMatmul", " tileL2Tiling.rankTileNum %u", tileL2Tiling.rankTileNum);
    OPS_LOG_D("AllGatherMatmul", " tileL2Tiling.calcOrder %u", tileL2Tiling.calcOrder);
    OPS_LOG_D("AllGatherMatmul", " tileL2Tiling.enableL2Tile %u", tileL2Tiling.enableL2Tile);
}
}

namespace optiling {

static ge::graphStatus CalcMatmulTiling(mc2tiling::TilingArgs& args, ::TCubeTiling& cubeTiling, ::TileL2Tiling &l2Tiling);

static ge::graphStatus MC2SetWorkspace(gert::TilingContext* context, AllGatherMatmulTilingData& tilingData, mc2tiling::TilingArgs& args);

static uint32_t MC2_Splite(mc2tiling::TilingArgs& args, uint32_t maxTileCnt = 64)
{
    // 检查允许通信的最大次数
    if (args.commTurn >= maxTileCnt) {
        args.commTurn = maxTileCnt;
    }

    uint64_t tileLen = 1;
    if (args.mValue > args.commTurn) {
        tileLen = args.mValue/ args.commTurn;
    }

    if (args.inputDtypeSize == 2) { // 数据长度为2, 则向 2*64 = 128，则向128对齐
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 64); // align size
    } else if (args.inputDtypeSize == 4) { // 4 is float32 type size
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 32); // align size
    }
    if (args.mValue > tileLen) {
        return tileLen;
    }
    return args.mValue;
}

static ge::graphStatus AllGatherParamsCheck(const gert::TilingContext* context)
{
    OPS_CHECK(mc2tiling::Mc2TilingUtils::CommonParamCheck(context) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "common check failed"), return ge::GRAPH_FAILED);

    const gert::StorageShape* aShape = context->GetInputShape(0);
    uint64_t valueOne = aShape->GetStorageShape().GetDim(0);
    uint64_t valueTwo = aShape->GetStorageShape().GetDim(1);

    OPS_CHECK(valueOne == 0 || valueTwo == 0,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "the value is invalid"), return ge::GRAPH_FAILED);

    if (context->GetAttrs() == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "get attrs failed");
    } else {
        auto gather_index = context->GetAttrs()->GetAttrPointer<int>(3);
        OPS_CHECK(*gather_index != 0,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "the gather_index should be 0, but real value is %d", *gather_index), return ge::GRAPH_FAILED);

        auto isTransA = context->GetAttrs()->GetAttrPointer<bool>(1);
        OPS_CHECK(*isTransA != false,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "the isTransA should be false, but real value is 1"), return ge::GRAPH_FAILED);
        OPS_CHECK((valueTwo < KVALUE_MIN || valueTwo >= KVALUE_MAX),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "The k-axis should be in range[256, 65535), but it is: %lu.", valueTwo), return ge::GRAPH_FAILED);
    }
    auto group = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    OPS_CHECK(group == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "group is nullptr. "),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetCommAlg(AllGatherMatmulTilingData &tilingData)
{
    tilingData.socParam.commAlg = COMM_ALG_FULL_MESH;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAllGatherFormulateTileCnt(const gert::TilingContext* ctx,
    AllGatherMatmulTilingData& tilingData, mc2tiling::TilingArgs& args)
{
    if (ctx->GetAttrs() == nullptr) {
        OPS_LOG_W(ctx->GetNodeName(), " ctx->GetAttrs is nullptr.");
        return ge::GRAPH_FAILED;
    }

    SocVersion inputSocVersion = SocVersion::SOC910_B;

    AllGatherPlusMM tileFormulate(args, args.rankDim, KernelType::ALL_GATHER, inputSocVersion);
    tileFormulate.GetTiling();
    CutResult mCutGather = tileFormulate.tilingM_.cutRes;
    tilingData.param.tileCnt = mCutGather.numLongTile;
    args.mValue = mCutGather.longTileLen;
    CalcMatmulTiling(args, tilingData.tileTiling, tilingData.tileL2Tiling);
    args.baseMLimit = mCutGather.longTileLen;
    args.mValue = mCutGather.longTileLen * args.rankTileNum;
    tilingData.param.tailM = mCutGather.shortTileLen;
    tilingData.param.tailCnt = 0;
    if (mCutGather.numShortTile > 0) {
        args.mValue = mCutGather.shortTileLen;
        tilingData.param.tailM = args.mValue;
        tilingData.param.tailCnt = mCutGather.numShortTile;
        CalcMatmulTiling(args, tilingData.tailTiling, tilingData.tailL2Tiling);
        args.baseMLimit = mCutGather.shortTileLen;
        args.mValue = mCutGather.shortTileLen * args.rankTileNum;
    }
    args.mValue = mCutGather.longTileLen;
    return ge::GRAPH_SUCCESS;
}

// 第一个参数m
static ge::graphStatus MCSpliteM(gert::TilingContext* ctx, AllGatherMatmulTilingData& tilingData,
                                 mc2tiling::TilingArgs& args)
{
    args.rankTileNum = args.rankDim - 1;
    // cmdType = HCCL_CMD_ALLGATHER, 是允许切K
    if (args.enableSplitK) { // 只有1份
        tilingData.param.tileCnt = 1;
        tilingData.param.tailCnt = 0;
        tilingData.param.tailM = 0;

        CalcMatmulTiling(args, tilingData.tileTiling, tilingData.tileL2Tiling);
    } else if (args.commTurn != 0) {
        uint64_t splite = MC2_Splite(args);

        // 现在找到1个合适的切分
        auto tileCnt = args.mValue / splite; // 切的份数
        auto tileTail = args.mValue % splite; // 尾巴

        tilingData.param.tileCnt = tileCnt;
        args.mValue = splite;
        tilingData.param.tailCnt = 0;
        CalcMatmulTiling(args, tilingData.tileTiling, tilingData.tileL2Tiling);
        tilingData.param.tailM = tileTail;
        if (tileTail != 0) {
            args.mValue = tileTail;
            tilingData.param.tailCnt = 1;
            CalcMatmulTiling(args, tilingData.tailTiling, tilingData.tailL2Tiling);
        }
        args.mValue = splite;
    } else {
        GetAllGatherFormulateTileCnt(ctx, tilingData, args);
    }
    MC2SetWorkspace(ctx, tilingData, args);

    return ge::GRAPH_SUCCESS;
}

static void UpdateTilingKey(uint32_t& tilingKey, AllGatherMatmulTilingData& tilingData, bool isBias)
{
    tilingKey += isBias ? TILINGKEY_BIAS : 0;
    tilingKey += (tilingData.socParam.isND2NZ == 1) ? TILINGKEY_ND2NZ : 0;
    tilingKey += (tilingData.socParam.commAlg == COMM_ALG_FULL_MESH) ? TILINGKEY_FULL_MESH : 0;
}

static ge::graphStatus SetMatmulTilingAllGatherMatmul(gert::TilingContext* context,
                                                      AllGatherMatmulTilingData& tilingData,
                                                      mc2tiling::TilingArgs& args)
{
    ge::DataType  biasType;
    bool isBias = true;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNumAic();
    auto aType = context->GetInputTensor(0)->GetDataType();
    auto bType = context->GetInputTensor(1)->GetDataType();
    auto cType = aType;
    const gert::StorageShape* matrix_bias = context->GetOptionalInputShape(2);
    if (matrix_bias == nullptr) {
        isBias = false;
        biasType = cType;
    }
    else {
        biasType = context->GetInputTensor(2)->GetDataType(); // 2 is index
    }

    const gert::StorageShape* aShape = context->GetInputShape(0);
    const gert::StorageShape* bShape = context->GetInputShape(1);
    uint64_t mValue = aShape->GetStorageShape().GetDim(0);
    uint64_t kValue = aShape->GetStorageShape().GetDim(1);
    uint64_t nValue = bShape->GetStorageShape().GetDim(1);

    if (aShape->GetStorageShape().GetDim(1) != bShape->GetStorageShape().GetDim(0)) {
        OPS_LOG_D(context->GetNodeName(), "A.shape(1) %ld B.shape(0) %ld, istransB = %d",
                aShape->GetStorageShape().GetDim(1), bShape->GetStorageShape().GetDim(0), args.isBTrans);
        nValue = bShape->GetStorageShape().GetDim(0);
    }

    uint64_t inputDtypeSize = mc2tiling::D_TYPE_SIZE_MAP.at(aType);
    uint64_t outputDtypeSize = mc2tiling::D_TYPE_SIZE_MAP.at(cType);

    tilingData.param.rankM = mValue; // 存放用户原始输入的mValue
    tilingData.param.rankN = nValue; // 存放用户原始输入的nValue
    tilingData.param.rankK = kValue; // 存放用户原始输入的kValue
    tilingData.param.aicCoreNum = coreNum;

    args.orgMValue = mValue;
    args.orgNValue = nValue;
    args.orgKValue = kValue;
    args.mValue = mValue;
    args.nValue = nValue;
    args.kValue = kValue;
    args.baseMLimit = -1;
    args.inputDtypeSize = inputDtypeSize;
    args.outputDtypeSize = outputDtypeSize;
    args.aicCoreNum = coreNum;
    args.enablePad = false;
    args.enableSplitK = false;
    args.isBias = isBias;
    args.geAType = aType;
    args.geBType = bType;
    args.geCType = cType;
    args.geBiasType = biasType;
    args.aType = mc2tiling::D_TYPE_MAP.at(aType);
    args.bType = mc2tiling::D_TYPE_MAP.at(bType);
    args.cType = mc2tiling::D_TYPE_MAP.at(cType);
    args.biasType = mc2tiling::D_TYPE_MAP.at(biasType); // 因为bias可能不存在，先采用biasType规避

    // 为通信而进行调整搬运
    if (args.cmdType == mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER) {
        // 先计算出自己的Tiling
        args.rankTileNum = 1; // 1: local matrix not tile
        args.isLocal = true;
        CalcMatmulTiling(args, tilingData.localTiling, tilingData.localL2Tiling);
        if (tilingData.param.rankID == 0) {
            PrintTilingData(tilingData.localTiling);
            PrintTilingData(tilingData.localL2Tiling);
        }
    } else {
      OPS_LOG_E(context->GetNodeName(), "args.cmdType error %d", static_cast<int>(args.cmdType));
      return ge::GRAPH_FAILED;
    }

    // 本卡一次计算完,其他卡数据按照DR搬运
    if ((tilingData.socParam.commAlg == COMM_ALG_DOUBLE_RING) && (tilingData.socParam.isStep == 1)) {
        args.mValue /= DOUBLE_RING_FACTOR;
        OPS_LOG_I(context->GetNodeName(), " args.mValue is set to be %lu under double ring + step communication algorithm.",
            args.mValue);
    }

    args.isLocal = false;

    MCSpliteM(context, tilingData, args);

    uint32_t tilingKey = 0U;
    UpdateTilingKey(tilingKey, tilingData, isBias);     // 当前GetTilingKey函数中使用了Mc2Msg结构体，因而无法归一化，此处使用自己的tilingkey计算函数，确保计算逻辑与旧的key保持一致
    OPS_LOG_D(context->GetNodeName(), "tilingKey is %u, aicCoreNum is %lu.", tilingKey, args.aicCoreNum);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(args.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcMatmulTiling(mc2tiling::TilingArgs& args, ::TCubeTiling& cubeTiling, ::TileL2Tiling &l2Tiling)
{
    uint64_t mValue = args.mValue;
    uint64_t nValue = args.nValue;
    uint64_t kValue = args.kValue;

    matmul_tiling::MultiCoreMatmulTiling mm;
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.aType, args.isATrans);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.bType, args.isBTrans);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.cType);
    if (args.isBias) {
        mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.biasType);
        mm.SetBias(true);
    }
    else {
        mm.SetBias(false);
    }
    mm.SetDim(args.aicCoreNum);
    mm.SetShape(mValue, nValue, kValue);
    mm.SetOrgShape(mValue, nValue, kValue);
    mm.SetBufferSpace(512 * 1024, -1, -1); // 512 * 1024 is buffer size
    mm.SetSingleShape(-1, -1, -1);
    if (nValue == 0) {
        cubeTiling.M = mValue;
        cubeTiling.N = nValue;
        cubeTiling.Ka = kValue;
        cubeTiling.Kb = kValue;
    } else {
        if (mm.GetTiling(cubeTiling) == -1) {
            OPS_LOG_E("AllGatherMatmul", "mValue %lu, nValue %lu, kValue %lu, aicCoreNum %lu",
                    mValue, nValue, kValue, args.aicCoreNum);
            return ge::GRAPH_FAILED;
        }
    }
    mc2tiling::MatmulFormulaicTiling gatherTiling("AllGatherMatmul");
    gatherTiling.GetCubeTiling(args, cubeTiling, l2Tiling);
    return ge::GRAPH_SUCCESS;
}

static uint64_t GetStorage_a(AllGatherMatmulTilingData& tilingData, mc2tiling::TilingArgs& args)
{
    constexpr uint64_t alignAddrLen = 512;
    auto&& cfg = tilingData.param;
    uint32_t gatherIndex = cfg.gatherIndex;
    uint64_t nd2nzLen = 0;
    uint64_t storage_a = 0;

    // step1: ND2NZ
    if (gatherIndex == 0) { // 转置B
        // 计算ND2NZ需使用空间方法保持与MMV3 tiling计算逻辑一致
        uint64_t alignByte = 256 / args.inputDtypeSize;  // 256B 对齐shape
        uint64_t kALign = ops::CeilAlign(static_cast<uint64_t>(cfg.rankK), alignByte);
        uint64_t nALign = ops::CeilAlign(static_cast<uint64_t>(cfg.rankN), alignByte);
        nd2nzLen = kALign * nALign * args.inputDtypeSize;
    }
    else {
        auto alignM = cfg.rankM + 16;
        auto alignK = cfg.rankK + 16;
        nd2nzLen = mc2tiling::AlignUp(alignM * alignK * args.inputDtypeSize, alignAddrLen);
    }

    if (args.cmdType == mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER) {
        uint64_t gmcFloat = 0; // allgatherMm 通信后数据只需放在gatherLen对应的workspace或者gatherout中，不需要gmcFloat
        uint64_t gatherLen = 0;
        if (args.isStorageGather == false) {
            if (gatherIndex == 0) { // A矩阵
                gatherLen = mc2tiling::AlignUp(cfg.rankM * cfg.rankK * args.inputDtypeSize, alignAddrLen);
            }
            else {
                gatherLen = mc2tiling::AlignUp(cfg.rankK * cfg.rankN * args.inputDtypeSize, alignAddrLen);
            }
            gatherLen *= cfg.rankDim;
        }

        tilingData.param.nd2NzWorkLen = nd2nzLen;
        tilingData.param.cToFloatLen = gmcFloat;
        tilingData.param.gatherLen = gatherLen;

        storage_a = nd2nzLen + gmcFloat + gatherLen; // 需要计算存放的A矩阵
    }
    return storage_a;
}

struct HcclAicpuOpParam {
    uint8_t res[64];
};

struct KFCMsgBody {
    // Rank* aiv * MsgSize * sizeof(消息)
    HcclAicpuOpParam msgSndArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
    HcclAicpuOpParam msgRcvArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
};
struct KFCNotify {
    // 消息通信
    HcclAicpuOpParam msgSend[16]; // 填充16个
    HcclAicpuOpParam msgCnt[16];
};

static ge::graphStatus MC2SetWorkspace(gert::TilingContext* context, AllGatherMatmulTilingData& tilingData,
                                       mc2tiling::TilingArgs& args)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OPS_CHECK(workspaces == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "get workspace failed"),
        return ge::GRAPH_FAILED);
    uint64_t storage_a = GetStorage_a(tilingData, args);

    int biasLen = 0;
    if (args.isBias) {
        biasLen = mc2tiling::AlignUp(args.orgNValue, mc2tiling::SHAPE_ALIGN_SIZE) * sizeof(float);
    }
    tilingData.param.biasLen = biasLen;
    workspaces[0] = storage_a + 16 * 1024 * 1024 + biasLen; // 16 mb, 1024 * 1024 is 1 mb
    OPS_LOG_D("AllGatherMatmul", "workspaces[0] size is %lu.", workspaces[0]);
    OPS_LOG_D("AllGatherMatmul", "biasLen is %d.", biasLen);

    tilingData.param.dataType = static_cast<uint8_t>(mc2tiling::Mc2TilingUtils::GetDataType(args.geAType));

    if (tilingData.param.rankID == 0) {
        OPS_LOG_D("AllGatherMatmul", "workspace size %lu.", workspaces[0]);

        PrintTilingData(tilingData.param);
        PrintTilingData(tilingData.tileTiling);
        PrintTilingData(tilingData.tileL2Tiling);
        if (tilingData.param.tailM != 0U) {
            OPS_LOG_D("AllGatherMatmul", "have tail.");
            PrintTilingData(tilingData.tailTiling);
            PrintTilingData(tilingData.tailL2Tiling);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static bool NeedGatherOut(const gert::TilingContext* context) {
  const gert::StorageShape* gatherOut = context->GetOutputShape(1);
  int64_t mulGatherShape = 1;
  if (gatherOut != nullptr) {
    for (unsigned int i = 0;i < gatherOut->GetStorageShape().GetDimNum(); i++) {
        mulGatherShape = mulGatherShape * gatherOut->GetStorageShape().GetDim(i);
        OPS_LOG_D("AllGatherMatmul", "gatherOut StorageShape[%u] is %ld", i, gatherOut->GetStorageShape().GetDim(i));
    }
  }

  if (gatherOut == nullptr || mulGatherShape == 0) {
    return false;
  } else {
    return true;
  }
}

static void SetSocParam(AllGatherMatmulTilingData* tilingData, const char* group)
{
  auto commSets = mc2tiling::Mc2TilingUtils::GetCommSets(group);
  tilingData->socParam.isStep = 0U;
  tilingData->socParam.isND2NZ = 1U;
}

static void InitHcclParam(AllGatherMatmulTilingData* tilingData, const char* group)
{
  std::string algConfig = "AllGather=level0:fullmesh";
  Mc2CcTilingConfig mc2CcTilingConfig(group, tilingData->param.commtype, algConfig);
  uint8_t skipBufferWindowCopy = (tilingData->param.gatherLen == 0) ?
                                 static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_DEFAULT) :
                                 static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
  mc2CcTilingConfig.SetSkipBufferWindowCopy(skipBufferWindowCopy);
  mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
  mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
}

static ge::graphStatus AllGatherMatmulTilingFunc(gert::TilingContext *context) {
  // 对参数进行校验
  int index = 0;
  AllGatherMatmulTilingData* tilingData = context->GetTilingData<AllGatherMatmulTilingData>();
  mc2tiling::TilingArgs args;
  auto group = context->GetAttrs()->GetAttrPointer<char>(index++);
  OPS_CHECK(AllGatherParamsCheck(context) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "param is invalid"), return ge::GRAPH_FAILED);

  auto is_trans_a = context->GetAttrs()->GetAttrPointer<bool>(index++);
  auto is_trans_b = context->GetAttrs()->GetAttrPointer<bool>(index++);
  auto gather_index = context->GetAttrs()->GetAttrPointer<int>(index++);
  auto comm_turn = *context->GetAttrs()->GetAttrPointer<int>(index++);

  auto rankSize = mc2tiling::MatmulFormulaicTiling::GetRankSize(group);
  OPS_CHECK(comm_turn != 0, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
      "comm_turn should be 0, but the actual value is %d.", comm_turn), return ge::GRAPH_FAILED);

  OPS_LOG_D("AllGatherMatmul"," group is %s, rankSize is %u, is_trans_a is %d, is_trans_b is %d, gather_index is %d,"
          "comm_turn is %d.", group, rankSize, *is_trans_a, *is_trans_b, *gather_index, comm_turn);
  tilingData->param.rankDim = rankSize;
  tilingData->param.isTransposeA = is_trans_a ? *is_trans_a : 0;
  tilingData->param.isTransposeB = is_trans_b ? *is_trans_b : 0;
  tilingData->param.gatherIndex = gather_index ? *gather_index : 0;
  tilingData->param.commtype = static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER);
  tilingData->param.subtype = 0;
  tilingData->param.storageGather = 0;
  SetSocParam(tilingData, group);

  OPS_CHECK(SetCommAlg(*tilingData) != ge::GRAPH_SUCCESS,
    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), " Set comm algorithm failed."), return ge::GRAPH_FAILED);
  OPS_LOG_I(context->GetNodeName(), " Communication algorithm is %u.", tilingData->socParam.commAlg);

  // distinguish between 910A2 and 910A3
  auto it = std::find(VALID_RANK.at(0).begin(),
  VALID_RANK.at(0).end(), rankSize);
  OPS_CHECK(it == VALID_RANK.at(0).end(),
    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
    "world_size value is %u, which is illegal.", rankSize), return ge::GRAPH_FAILED);

  args.isATrans = is_trans_a ? *is_trans_a : 0;
  args.isBTrans = is_trans_b ? *is_trans_b : 0;
  args.cmdType = mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER;
  args.rankDim = rankSize;
  args.commTurn = comm_turn;
  args.commAlg = tilingData->socParam.commAlg;

  if (NeedGatherOut(context)) {
    args.isStorageGather = true;
    tilingData->param.storageGather = 1;
  } else {
    args.isStorageGather = false;
  }

  SetMatmulTilingAllGatherMatmul(context, *tilingData, args);
  InitHcclParam(tilingData, group);
  return ge::GRAPH_SUCCESS;
}

struct AllGatherMatmulCompileInfo {};
static ge::graphStatus TilingParseForAllGatherMatmul(gert::TilingParseContext *context) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(AllGatherMatmul)
    .Tiling(AllGatherMatmulTilingFunc)
    .TilingParse<AllGatherMatmulCompileInfo>(TilingParseForAllGatherMatmul);
}  // namespace optiling

