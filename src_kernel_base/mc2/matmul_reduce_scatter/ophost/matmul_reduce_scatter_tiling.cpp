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
 * \file matmul_reduce_scatter_tiling.cpp
 * \brief
*/
#include "vector"
#include "graph/utils/type_utils.h"
#include "log/ops_log.h"
#include "error/ops_error.h"
#include "ophost/op_util.h"
#include "register/op_def_registry.h"
#include "ophost/mc2_tiling_utils.h"
#include "ophost/hcom_topo_info.h"
#include "ophost/matmul_formulaic_tiling.h"
#include "reduce_scatter_formulaic_tiling.h"
#include "../matmul_reduce_scatter_tiling.h"

using namespace AscendC;
using namespace ge;

namespace {
constexpr char HCCL_DETERMINISTIC[] = "HCCL_DETERMINISTIC";
const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK = {
    {0, {2, 4, 8}},
	{1, {2, 4, 8, 16, 32}}
    };

constexpr uint32_t TILINGKEY_BIAS = 1U;
constexpr uint32_t TILINGKEY_ND2NZ = 10U;
constexpr uint32_t TILINGKEY_FULL_MESH = 100U;

const std::vector<uint64_t> CALC_ND_BASIC = {6144, 4096, 2048};
constexpr uint64_t NUMBER_TWO = 2;
constexpr uint64_t BLOCK_BYTE_SIZE = 32;
constexpr uint64_t N_ALIGNED = 16;
constexpr uint64_t NCALC_THRES = 16;
constexpr uint64_t MIN_TAIL = 512;
constexpr uint64_t NUMBER_SIXTEEN = 16;

constexpr uint64_t VECTOR_D_BASE = 2048;
constexpr uint64_t UB_SIZE = 196352;

static int64_t CeilDivision(int64_t num1, int32_t num2) {
    return ops::CeilDiv(num1, static_cast<int64_t>(num2));
}

static bool CheckUbOverFlowMC2(uint64_t ubSize, uint64_t nAligned16, uint64_t nValue, const uint64_t& baseN,
                               const uint64_t& baseD, uint64_t dtypeSize)
{
    uint64_t nAlignedLoop = CeilDivision(nAligned16, baseN);
    uint64_t nValueLoop = CeilDivision(nValue, baseN);
    return ((nAlignedLoop != nValueLoop) &&
            ((nAligned16 - ((nValue / baseN) - 1) * baseN) * baseD > (ubSize / NUMBER_TWO / dtypeSize)));
}

static void CalcNd2NzTilingMC2(mc2tiling::TilingArgs& args, uint64_t ubSize, uint64_t dtypeSize, uint64_t nValue,
                               uint64_t dValue, uint64_t& baseN, uint64_t& baseD)
{
    constexpr uint64_t mataD = 16384;
    constexpr uint64_t minN = 7168;
    // mata 交织场景
    if (dValue % mataD == 0 && nValue >= minN && ubSize == UB_SIZE &&
        (args.aType == matmul_tiling::DataType::DT_FLOAT16 || args.aType == matmul_tiling::DataType::DT_BF16) &&
        args.aicCoreNum >= 24) { // 24 核最小核数
        baseN = 96;              // baseN 经验值 96
        baseD = 512;             // baseD 经验值 512
        return;
    }

    uint64_t vectorCoreNum = NUMBER_TWO * args.aicCoreNum;
    vectorCoreNum = std::max(vectorCoreNum, 1UL);
    uint64_t baseThres = VECTOR_D_BASE / dtypeSize;
    uint64_t c0 = BLOCK_BYTE_SIZE / dtypeSize;
    uint64_t nAligned16 = ops::CeilAlign(nValue, N_ALIGNED);
    uint64_t dAlignedC0 = ops::CeilAlign(dValue, c0);
    if (dValue <= baseThres) {
        baseD = std::max(ops::CeilAlign(dValue, c0), uint64_t(1));
        uint64_t nDim = vectorCoreNum;
        baseN = ubSize / NUMBER_TWO / dtypeSize / baseD;
        uint64_t round = std::max((uint64_t)CeilDivision(CeilDivision(nAligned16, nDim), baseN), 1UL);
        baseN = std::max((uint64_t)CeilDivision(CeilDivision(nAligned16, nDim), round), NCALC_THRES);
        while (baseN > NCALC_THRES) {
            if (CheckUbOverFlowMC2(ubSize, nAligned16, nValue, baseN, baseD, dtypeSize)) {
                baseN--;
                continue;
            }
            break;
        }
        return;
    }
    uint64_t lastTail = 0;
    // bestBaseD = 4K / dtype, bestBaseN = UB / 2 / 4096
    uint64_t bestBaseN = NCALC_THRES;
    uint64_t bestBaseD = CALC_ND_BASIC.at(1) / dtypeSize;
    for (auto base : CALC_ND_BASIC) {
        baseD = std::max(std::min(dAlignedC0, base / dtypeSize), uint64_t(1));
        uint64_t dLoop = CeilDivision(dAlignedC0, baseD);
        uint64_t dTail = dAlignedC0 % baseD;
        if (dTail > 0 && dTail < (MIN_TAIL / dtypeSize)) {
            if (baseD * dtypeSize == CALC_ND_BASIC.at(0)) { // if dtail < 0.5K && baseD == 3k, give up and return bestbase
                continue;
            }
            dLoop--;
            baseD = std::max(ops::CeilAlign((uint64_t)CeilDivision(dAlignedC0, dLoop),
                c0), uint64_t(1));; // dtail< 0.5K, make baseD larger
        }
        baseN = std::max(ubSize / NUMBER_TWO / dtypeSize / baseD, NUMBER_SIXTEEN);
        if (baseN * baseD * dtypeSize * NUMBER_TWO > ubSize) { // check ub overflow
            continue;
        }
        if (CheckUbOverFlowMC2(ubSize, nAligned16, nValue, baseN, baseD, dtypeSize)) {
            continue;
        }
        uint64_t nLoop = CeilDivision(nAligned16, baseN);
        uint64_t tail = nLoop * dLoop % vectorCoreNum;
        while (baseN > NCALC_THRES) {
            if (CheckUbOverFlowMC2(ubSize, nAligned16, nValue, baseN, baseD, dtypeSize)) { // check ub overflow
                baseN--;
                nLoop = CeilDivision(nAligned16, baseN);
                tail = nLoop * dLoop % vectorCoreNum;
                continue;
            }
            if (tail == 0) {
                return;
            }
            if (tail > lastTail) {
                lastTail = tail;
                bestBaseD = baseD;
                bestBaseN = baseN;
            }
            baseN--;
            nLoop = CeilDivision(nAligned16, baseN);
            tail = nLoop * dLoop % vectorCoreNum;
        }
    }
    baseD = bestBaseD;
    baseN = bestBaseN;
    return;
}

static void PrintTilingData(TCubeTiling& tiling)
{
    OPS_LOG_D("MatmulReduceScatter", " tiling.usedCoreNum %d", tiling.usedCoreNum);
    OPS_LOG_D("MatmulReduceScatter", " tiling.M %d", tiling.M);
    OPS_LOG_D("MatmulReduceScatter", " tiling.N %d", tiling.N);
    OPS_LOG_D("MatmulReduceScatter", " tiling.Ka %d", tiling.Ka);
    OPS_LOG_D("MatmulReduceScatter", " tiling.Kb %d", tiling.Kb);
    OPS_LOG_D("MatmulReduceScatter", " tiling.singleCoreM %d", tiling.singleCoreM);
    OPS_LOG_D("MatmulReduceScatter", " tiling.singleCoreN %d", tiling.singleCoreN);
    OPS_LOG_D("MatmulReduceScatter", " tiling.singleCoreK %d", tiling.singleCoreK);
    OPS_LOG_D("MatmulReduceScatter", " tiling.baseM %d", tiling.baseM);
    OPS_LOG_D("MatmulReduceScatter", " tiling.baseN %d", tiling.baseN);
    OPS_LOG_D("MatmulReduceScatter", " tiling.baseK %d", tiling.baseK);
    OPS_LOG_D("MatmulReduceScatter", " tiling.depthA1 %d", tiling.depthA1);
    OPS_LOG_D("MatmulReduceScatter", " tiling.depthB1 %d", tiling.depthB1);
    OPS_LOG_D("MatmulReduceScatter", " tiling.stepM %d", tiling.stepM);
    OPS_LOG_D("MatmulReduceScatter", " tiling.stepN %d", tiling.stepN);
    OPS_LOG_D("MatmulReduceScatter", " tiling.stepka %d", tiling.stepKa);
    OPS_LOG_D("MatmulReduceScatter", " tiling.stepkb %d", tiling.stepKb);
    OPS_LOG_D("MatmulReduceScatter", " tiling.isBias %d", tiling.isBias);
    OPS_LOG_D("MatmulReduceScatter", " tiling.transLength %d", tiling.transLength);
    OPS_LOG_D("MatmulReduceScatter", " tiling.iterateOrder %s", ((tiling.iterateOrder == 1)? "orderM" : "orderN"));
    OPS_LOG_D("MatmulReduceScatter", " tiling.usedL1Size %d", tiling.shareL1Size);
    OPS_LOG_D("MatmulReduceScatter", " tiling.usedL0CSize %d", tiling.shareL0CSize);
    OPS_LOG_D("MatmulReduceScatter", " tiling.dbL0C %d", tiling.dbL0C); // set_dbL0C(1)
    OPS_LOG_D("MatmulReduceScatter", " tiling.usedUBSize %d", tiling.shareUbSize);
    OPS_LOG_D("MatmulReduceScatter", " tiling.batchM %d", tiling.batchM);
    OPS_LOG_D("MatmulReduceScatter", " tiling.batchN %d", tiling.batchN);
    OPS_LOG_D("MatmulReduceScatter", " tiling.singleBatchM %d", tiling.singleBatchM);
    OPS_LOG_D("MatmulReduceScatter", " tiling.singleBatchN %d", tiling.singleBatchN);
}

static void PrintTilingData(RCSTiling& rcsTiling)
{
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.rankDim %u", rcsTiling.rankDim);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.rankID %u", rcsTiling.rankID);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.commtype %u", rcsTiling.commtype);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.subtype %u", rcsTiling.subtype);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.tileCnt %u", rcsTiling.tileCnt);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.tailM %u", rcsTiling.tailM);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.tailCnt %u", rcsTiling.tailCnt);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.biasLen %u", rcsTiling.biasLen);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.isAdd %u", rcsTiling.isAdd);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.rankM %u", rcsTiling.rankM);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.rankN %u", rcsTiling.rankN);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.rankK %u", rcsTiling.rankK);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.gatherIndex %u", rcsTiling.gatherIndex);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.isTransA %u", rcsTiling.isTransposeA);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.isTransB %u", rcsTiling.isTransposeB);
	OPS_LOG_D("MatmulReduceScatter", " rcsTiling.storageGather %u", rcsTiling.storageGather);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.nd2NzWorkLen %lu", rcsTiling.nd2NzWorkLen);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.cToFloatLen %lu", rcsTiling.cToFloatLen);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.gatherLen %lu", rcsTiling.gatherLen);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.workspaceAddr4 %u", rcsTiling.workspaceAddr4);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.aicCoreNum %u", rcsTiling.aicCoreNum);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.needUbBuffer %u", rcsTiling.needUbBuffer);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.addX3UbCnt %u", rcsTiling.addX3UbCnt);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.commWorkSpaceSize %u", rcsTiling.commWorkSpaceSize);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.isInputCommQuantScale %u", rcsTiling.isInputCommQuantScale);
    OPS_LOG_D("MatmulReduceScatter", " rcsTiling.dataType %u", rcsTiling.dataType);
}

static void PrintTilingData(TileL2Tiling& tileL2Tiling)
{
    OPS_LOG_D("MatmulReduceScatter", " tileL2Tiling.mL2TileCnt %u", tileL2Tiling.mL2TileCnt);
    OPS_LOG_D("MatmulReduceScatter", " tileL2Tiling.nL2TileCnt %u", tileL2Tiling.nL2TileCnt);
    OPS_LOG_D("MatmulReduceScatter", " tileL2Tiling.mTileBlocks %u", tileL2Tiling.mTileBlocks);
    OPS_LOG_D("MatmulReduceScatter", " tileL2Tiling.nTileBlocks %u", tileL2Tiling.nTileBlocks);
    OPS_LOG_D("MatmulReduceScatter", " tileL2Tiling.mTailBlocks %u", tileL2Tiling.mTailBlocks);
    OPS_LOG_D("MatmulReduceScatter", " tileL2Tiling.nTailBlocks %u", tileL2Tiling.nTailBlocks);
    OPS_LOG_D("MatmulReduceScatter", " tileL2Tiling.rankTileNum %u", tileL2Tiling.rankTileNum);
    OPS_LOG_D("MatmulReduceScatter", " tileL2Tiling.calcOrder %u", tileL2Tiling.calcOrder);
    OPS_LOG_D("MatmulReduceScatter", " tileL2Tiling.enableL2Tile %u", tileL2Tiling.enableL2Tile);
}
}

namespace optiling {

static ge::graphStatus CalcMatmulTilingReduceScatter(mc2tiling::TilingArgs& args, ::TCubeTiling& cubeTiling,
                                                     ::TileL2Tiling &l2Tiling);

static ge::graphStatus MC2SetWorkspaceReduceScatter(gert::TilingContext* context,
                                                    MatmulReduceScatterTilingData& tilingData,
                                                    mc2tiling::TilingArgs& args);

static uint32_t MC2_SpliteReduceScatter(mc2tiling::TilingArgs& args, uint32_t maxTileCnt = 64)
{
    // 检查允许通信的最大次数
    if (args.commTurn >= maxTileCnt) {
        args.commTurn = maxTileCnt;
    }

    uint64_t tileLen = 1;
    if (args.mValue > args.commTurn) {
        tileLen = args.mValue/ args.commTurn;
    }

    if (args.outputDtypeSize == 2) { // 数据长度为2, 则向 2*64 = 128，则向128对齐
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 64);
    } else if (args.outputDtypeSize == 4) { // 4 is float32 tpye size
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 32); // 32 is used to align to 128
    }
    if (args.mValue > tileLen) {
        return tileLen;
    }
    return args.mValue;
}

static ge::graphStatus ReduceScatterParamsCheck(const gert::TilingContext* context)
{
    OPS_CHECK(mc2tiling::Mc2TilingUtils::CommonParamCheck(context) != ge::GRAPH_SUCCESS,
              OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "common check failed"), return ge::GRAPH_FAILED);

    const gert::StorageShape* aShape = context->GetInputShape(0);
    const gert::StorageShape* bShape = context->GetInputShape(1);
    uint64_t valueOne = aShape->GetStorageShape().GetDim(0);
    uint64_t valueTwo = aShape->GetStorageShape().GetDim(1);
    uint64_t valueThree = bShape->GetStorageShape().GetDim(0);
    uint64_t valueFour = bShape->GetStorageShape().GetDim(1);

    OPS_CHECK(valueOne == 0 || valueTwo == 0 || valueThree == 0 || valueFour == 0,
              OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "the value is invalid"), return ge::GRAPH_FAILED);

    if (context->GetAttrs() == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "get attrs failed");
    } else {
        auto reduce_op = context->GetAttrs()->GetAttrPointer<char>(1);
        OPS_CHECK(strcmp(reduce_op, "sum") != 0,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "the reduce_op should be sum, but real value is %s", reduce_op), return ge::GRAPH_FAILED);

        auto isTransA = context->GetAttrs()->GetAttrPointer<bool>(2);
        OPS_CHECK(*isTransA != false,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "the isTransA should be false, but real value is 1"), return ge::GRAPH_FAILED);
    }

    auto group = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    OPS_CHECK(group == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "group is nullptr. "),
              return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetCommAlg(MatmulReduceScatterTilingData &tilingData)
{
    tilingData.socParam.commAlg = COMM_ALG_FULL_MESH;
    return ge::GRAPH_SUCCESS;
}

static bool IsDeterministic()
{
    if (getenv(HCCL_DETERMINISTIC) == nullptr) {
        return false;
    }
    std::string envStr(getenv(HCCL_DETERMINISTIC));
    std::transform(envStr.begin(), envStr.end(), envStr.begin(), ::toupper);
    if (envStr == "TRUE") {
        OPS_LOG_I("MatmulReduceScatter", "HCCL_DETERMINISTIC is set to be true.");
        return true;
    }
    OPS_LOG_I("MatmulReduceScatter", "HCCL_DETERMINISTIC [%s] is set to be false.", envStr.c_str());
    return false;
}

static ge::graphStatus GetReduceScatterFormulateTileCnt(const gert::TilingContext* ctx,
    MatmulReduceScatterTilingData& tilingData, mc2tiling::TilingArgs& args)
{
    if (ctx->GetAttrs() == nullptr) {
        OPS_LOG_W(ctx->GetNodeName(), " ctx->GetAttrs is nullptr.");
        return ge::GRAPH_FAILED;
    }

    SocVersion inputSocVersion = (tilingData.socParam.isA3 == 0) ? SocVersion::SOC910_B : SocVersion::SOC910_93;
    bool commDeterministic = (inputSocVersion == SocVersion::SOC910_B) ? IsDeterministic() : false;

    MMPlusReduceScatter scatterTilingHccl(args, args.rankDim, KernelType::REDUCE_SCATTER,
        inputSocVersion, commDeterministic);
    scatterTilingHccl.GetTiling();
    CutResult mCutScatter = scatterTilingHccl.tilingM_.cutRes;
    if (mCutScatter.shortTileAtBack || mCutScatter.numShortTile == 0) {
        tilingData.param.tileCnt = mCutScatter.numLongTile;
        args.mValue = mCutScatter.longTileLen;
        CalcMatmulTilingReduceScatter(args, tilingData.tileTiling, tilingData.tileL2Tiling);
        args.baseMLimit = mCutScatter.longTileLen;
        args.mValue = mCutScatter.longTileLen * args.rankTileNum;
        tilingData.param.tailM = mCutScatter.shortTileLen;
        tilingData.param.tailCnt = 0;
        if (mCutScatter.numShortTile > 0) {
            args.mValue = mCutScatter.shortTileLen;
            tilingData.param.tailCnt = mCutScatter.numShortTile;
            CalcMatmulTilingReduceScatter(args, tilingData.tailTiling, tilingData.tailL2Tiling);
            args.baseMLimit = mCutScatter.shortTileLen;
            args.mValue = mCutScatter.shortTileLen * args.rankTileNum;
        }
    } else {
        tilingData.param.tileCnt = mCutScatter.numShortTile;
        args.mValue = mCutScatter.shortTileLen;
        CalcMatmulTilingReduceScatter(args, tilingData.tileTiling, tilingData.tileL2Tiling);
        args.baseMLimit = mCutScatter.shortTileLen;
        args.mValue = mCutScatter.shortTileLen * args.rankTileNum;
        tilingData.param.tailM = mCutScatter.longTileLen;
        tilingData.param.tailCnt = 0;
        if (mCutScatter.numLongTile > 0) {
            args.mValue = mCutScatter.longTileLen;
            tilingData.param.tailCnt = mCutScatter.numLongTile;
            CalcMatmulTilingReduceScatter(args, tilingData.tailTiling, tilingData.tailL2Tiling);
            args.baseMLimit = mCutScatter.longTileLen;
            args.mValue = mCutScatter.longTileLen * args.rankTileNum;
        }
    }
    args.mValue = mCutScatter.longTileLen;
    return ge::GRAPH_SUCCESS;
}

// 第一个参数m
static ge::graphStatus MCSpliteMReduceScatter(gert::TilingContext* ctx, MatmulReduceScatterTilingData& tilingData,
                                              mc2tiling::TilingArgs& args)
{
    args.rankTileNum = args.rankDim;

    if (args.enableSplitK){ // 只有1份
        tilingData.param.tileCnt = 1;
        tilingData.param.tailCnt = 0;
        tilingData.param.tailM = 0;
        CalcMatmulTilingReduceScatter(args, tilingData.tileTiling, tilingData.tileL2Tiling);
    } else if (args.commTurn != 0) {
        uint64_t splite = MC2_SpliteReduceScatter(args);

        // 现在找到1个合适的切分
        auto tileCnt = args.mValue / splite; // 切的份数
        auto tileTail = args.mValue % splite; // 尾巴

        tilingData.param.tileCnt = tileCnt;
        args.mValue = splite;
        tilingData.param.tailCnt = 0;
        CalcMatmulTilingReduceScatter(args, tilingData.tileTiling, tilingData.tileL2Tiling);
        tilingData.param.tailM = tileTail;
        if (tileTail != 0) {
            args.mValue = tileTail;
            tilingData.param.tailCnt = 1;
            CalcMatmulTilingReduceScatter(args, tilingData.tailTiling, tilingData.tailL2Tiling);
        }
        args.mValue = splite;
    } else {
        GetReduceScatterFormulateTileCnt(ctx, tilingData, args);
    }
    MC2SetWorkspaceReduceScatter(ctx, tilingData, args);

    return ge::GRAPH_SUCCESS;
}

static void SetReduceScatterTilingArgs(gert::TilingContext* context, mc2tiling::TilingArgs& args)
{
    auto coreNum = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetCoreNumAic();
    auto aType = context->GetInputTensor(0)->GetDataType();
    auto bType = context->GetInputTensor(1)->GetDataType();
    auto cType = aType;

    // bias
    const gert::StorageShape* matrix_bias = context->GetOptionalInputShape(2);
    bool isBias = (matrix_bias == nullptr)? false : true;
    ge::DataType biasType = (matrix_bias == nullptr)? cType : context->GetInputTensor(2)->GetDataType();

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

    args.orgMValue = mValue;
    args.orgNValue = nValue;
    args.orgKValue = kValue;
    args.mValue = mValue;

    if (args.commAlg == COMM_ALG_DOUBLE_RING) {
        args.mValue /= DOUBLE_RING_FACTOR;
        OPS_LOG_D(context->GetNodeName(), " args.mValue is %lu under double ring communication algorithm.", args.mValue);
    }

    args.nValue = nValue;
    args.kValue = kValue;
    args.baseMLimit = -1;
    args.inputDtypeSize = mc2tiling::D_TYPE_SIZE_MAP.at(aType);
    args.outputDtypeSize = mc2tiling::D_TYPE_SIZE_MAP.at(cType);
    args.geAType = aType;
    args.geBType = bType;
    args.geCType = cType;
    args.geBiasType = biasType;
    args.aicCoreNum = coreNum;
    args.enablePad = false;
    args.enableSplitK = false;
    args.isBias = isBias;
    args.aType = mc2tiling::D_TYPE_MAP.at(aType);
    args.bType = mc2tiling::D_TYPE_MAP.at(bType);
    args.cType = mc2tiling::D_TYPE_MAP.at(cType);
    args.biasType = mc2tiling::D_TYPE_MAP.at(biasType); // 因为bias可能不存在，先采用biasType规避
}

struct HcclAicpuOpParam {
    uint8_t res[64];
};

struct KFCNotify {
    // 消息通信
    HcclAicpuOpParam msgSend[16]; // 填充16个
    HcclAicpuOpParam msgCnt[16];
};

struct KFCMsgBody {
    // Rank* aiv * MsgSize * sizeof(消息)
    HcclAicpuOpParam msgSndArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
    HcclAicpuOpParam msgRcvArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
};

static void GetTilingKey(uint32_t& tilingKey, MatmulReduceScatterTilingData& tilingData)
{
    tilingKey += (tilingData.socParam.isND2NZ == 1) ? TILINGKEY_ND2NZ : 0;
    tilingKey += (tilingData.socParam.commAlg == COMM_ALG_FULL_MESH) ? TILINGKEY_FULL_MESH : 0;
    uint64_t castBias = tilingData.param.biasLen == 0 ? 0 : TILINGKEY_BIAS;
    tilingKey += castBias;

    OPS_LOG_D("MatmulReduceScatterTilingData", "The final tiling Key is: %u!", tilingKey);
    return;
}

static ge::graphStatus SetMatmulTilingMatmulReduceScatter(gert::TilingContext* context, MatmulReduceScatterTilingData& tilingData,
                                                          mc2tiling::TilingArgs& args)
{
    SetReduceScatterTilingArgs(context, args);

    tilingData.param.rankM = args.orgMValue; // 存放用户原始输入的mValue
    tilingData.param.rankN = args.orgNValue; // 存放用户原始输入的nValue
    tilingData.param.rankK = args.orgKValue; // 存放用户原始输入的kValue
    tilingData.param.aicCoreNum = args.aicCoreNum;
    uint64_t baseN = 1;
    uint64_t baseD = 1;
    uint64_t ubSizeValue = 0;
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeValue);

    uint64_t nValue = args.isBTrans ? tilingData.param.rankN : tilingData.param.rankK;
    uint64_t dValue = args.isBTrans ? tilingData.param.rankK : tilingData.param.rankN;
    CalcNd2NzTilingMC2(args, ubSizeValue, args.inputDtypeSize, nValue, dValue, baseN, baseD);
    tilingData.socParam.baseBD = baseD;
    tilingData.socParam.baseBN = baseN;
    // 为通信而进行调整搬运
    if (args.cmdType == mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER) {
        if (args.rankDim <= 0 || args.orgMValue % args.rankDim) {
            OPS_LOG_E(context->GetNodeName(), "rankDim error : %u, mValue=%lu", args.rankDim, args.orgMValue);
            return ge::GRAPH_FAILED;
        }
        args.mValue /= args.rankDim; // 必须能够整数切分, 并且不能切K
    } else {
        OPS_LOG_E(context->GetNodeName(), "args.cmdType error %d", static_cast<int>(args.cmdType));
        return ge::GRAPH_FAILED;
    }

    MCSpliteMReduceScatter(context, tilingData, args);

    uint32_t tilingKey = 0U;
    GetTilingKey(tilingKey, tilingData);
    OPS_LOG_D(context->GetNodeName(), "tilingKey is %u, aicCoreNum is %lu.", tilingKey, args.aicCoreNum);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(args.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcMatmulTilingReduceScatter(mc2tiling::TilingArgs& args, ::TCubeTiling& cubeTiling,
                                                     ::TileL2Tiling &l2Tiling)
{
    uint64_t mValue = args.mValue;
    uint64_t nValue = args.nValue;
    uint64_t kValue = args.kValue;

    matmul_tiling::MultiCoreMatmulTiling mm1;
    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.aType, args.isATrans);
    mm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.bType, args.isBTrans);
    mm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.cType);
    if (args.isBias) {
        mm1.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.biasType);
        mm1.SetBias(true);
    }
    else {
        mm1.SetBias(false);
    }
    mm1.SetDim(args.aicCoreNum);

    mm1.SetShape(mValue, nValue, kValue);
    mm1.SetOrgShape(mValue, nValue, kValue);
    mm1.SetBufferSpace(512 * 1024, -1, -1); // 512 * 1024 is buffer size
    mm1.SetSingleShape(-1, -1, -1);
    if (mm1.GetTiling(cubeTiling) == -1) {
        OPS_LOG_E("MatmulReduceScatter", "mValue %lu, nValue %lu, kValue %lu, aicCoreNum %lu",
                mValue, nValue, kValue, args.aicCoreNum);
        return ge::GRAPH_FAILED;
    }
    mc2tiling::MatmulFormulaicTiling scatterTiling("MatmulReduceScatter");
    scatterTiling.GetCubeTiling(args, cubeTiling, l2Tiling);

    return ge::GRAPH_SUCCESS;
}

static void CalculateNd2nzLen(::RCSTiling& config, mc2tiling::TilingArgs& args, uint64_t& nd2nzLen) {
    constexpr uint64_t alignAddrLen = 512;
    uint32_t gatherIndex = config.gatherIndex;
    if (gatherIndex == 0) { // 转置B
        // 计算ND2NZ需使用空间方法保持与MMV3 tiling计算逻辑一致
        uint64_t alignByte = 256 / args.inputDtypeSize; // 256B 对齐shape
        uint64_t kALign = ops::CeilAlign(static_cast<uint64_t>(config.rankK), alignByte);
        uint64_t nALign = ops::CeilAlign(static_cast<uint64_t>(config.rankN), alignByte);
        nd2nzLen = kALign * nALign * args.inputDtypeSize;
    }
    else {
        auto alignM = config.rankM + 16;
        auto alignK = config.rankK + 16;
        nd2nzLen = mc2tiling::AlignUp(alignM * alignK * args.inputDtypeSize, alignAddrLen);
    }
}

static ge::graphStatus SetTilingData(const gert::TilingContext* context, MatmulReduceScatterTilingData& tilingData)
{
    const uint32_t opType = static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER);
    if (context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char* groupName = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    const std::string rsConfig = (tilingData.socParam.isA3 == 0) ?
	    "ReduceScatter=level0:fullmesh" : "ReduceScatter=level0:doublering";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, rsConfig, 0);
    mc2CcTilingConfig.GetTiling(tilingData.mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MC2SetWorkspaceReduceScatter(gert::TilingContext* context,
                                                    MatmulReduceScatterTilingData& tilingData,
                                                    mc2tiling::TilingArgs& args)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OPS_CHECK(workspaces == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "get workspace failed"), return ge::GRAPH_FAILED);

    auto&& cfg = tilingData.param;

    // step1: ND2NZ
    uint64_t nd2nzLen = 0;
    CalculateNd2nzLen(cfg, args, nd2nzLen);

    uint64_t storage_a = 0;
    if (args.cmdType == mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER) {
        // A*B 的数据，需要找地方存储，因为 C 的长度 = A*B的长度/ RankDim
        uint64_t gmcFloat = static_cast<uint64_t>(cfg.rankM) * static_cast<uint64_t>(cfg.rankN) *
                            static_cast<uint64_t>(args.outputDtypeSize);
        gmcFloat = mc2tiling::AlignUp<uint64_t>(gmcFloat, 512); // 512 is used to get gm
        OPS_LOG_D("MatmulReduceScatter", " reduce scatter gmcFloat size %lu.", gmcFloat);

        tilingData.param.nd2NzWorkLen = nd2nzLen;
        tilingData.param.cToFloatLen = gmcFloat;

        storage_a = nd2nzLen + gmcFloat;
        OPS_LOG_D("MatmulReduceScatter", " reduce scatter storage_a size %lu.", storage_a);
    }

    int biasLen = 0;
    if (args.isBias && args.bType == matmul_tiling::DataType::DT_BFLOAT16) {
        biasLen = mc2tiling::AlignUp(args.orgNValue, mc2tiling::SHAPE_ALIGN_SIZE) * sizeof(float);
    }
    tilingData.param.biasLen = biasLen;
    workspaces[0] = storage_a + 16 * 1024 * 1024 + biasLen; // 16 mb, 1024 * 1024 is one mb
    OPS_LOG_D("MatmulReduceScatter", " workspaces[0] size %lu, biasLen %d.", workspaces[0], biasLen);
	tilingData.param.dataType = static_cast<uint8_t>(mc2tiling::Mc2TilingUtils::GetDataType(args.geAType));

    if (tilingData.param.rankID == 0) {
        OPS_LOG_D("MatmulReduceScatter", "workspace size %lu", workspaces[0]);

        PrintTilingData(tilingData.param);
        PrintTilingData(tilingData.tileTiling);
        PrintTilingData(tilingData.tileL2Tiling);
        if (tilingData.param.tailM != 0U) {
            OPS_LOG_D("MatmulReduceScatter", "have tail");
            PrintTilingData(tilingData.tailTiling);
            PrintTilingData(tilingData.tailL2Tiling);
        }
    }

    SetTilingData(context, tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MatmulReduceScatterTilingFunc(gert::TilingContext *context) {
  MatmulReduceScatterTilingData* tilingData = context->GetTilingData<MatmulReduceScatterTilingData>();
    // 对参数进行校验
    OPS_CHECK(ReduceScatterParamsCheck(context) != ge::GRAPH_SUCCESS,
              OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "param is invalid"), return ge::GRAPH_FAILED);
    int index = 0;
  auto group = context->GetAttrs()->GetAttrPointer<char>(index++);
  auto reduce_op = context->GetAttrs()->GetAttrPointer<char>(index++);
  auto is_trans_a = context->GetAttrs()->GetAttrPointer<bool>(index++);
  auto is_trans_b = context->GetAttrs()->GetAttrPointer<bool>(index++);
  auto comm_turn = *context->GetAttrs()->GetAttrPointer<int>(index++);
    auto rankSize = mc2tiling::MatmulFormulaicTiling::GetRankSize(group);

  OPS_CHECK(comm_turn != 0, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
    "comm_turn should be 0, but the actual value is %d.", comm_turn), return ge::GRAPH_FAILED);

  OPS_LOG_D("MatmulReduceScatter", "group is %s, rankSize is %u, reduce_op is %s, is_trans_a is %d, is_trans_b is %d,"
    "comm_turn is %d.", group, rankSize, reduce_op, *is_trans_a, *is_trans_b, comm_turn);

    tilingData->param.rankDim = rankSize;
  tilingData->param.isTransposeA = is_trans_a? *is_trans_a : 0;
  tilingData->param.isTransposeB = is_trans_b? *is_trans_b : 0;
    auto commSets = mc2tiling::Mc2TilingUtils::GetCommSets(group);
    tilingData->socParam.isA3 = (commSets == COMM_MESH) ? 0 : 1;
    tilingData->socParam.isStep = 0U;
    tilingData->socParam.isND2NZ = 1U;
    OPS_CHECK(SetCommAlg(*tilingData) != ge::GRAPH_SUCCESS,
    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), " Set comm algorithm failed."), return ge::GRAPH_FAILED);
    OPS_LOG_I(context->GetNodeName(), " Communication algorithm is %u.", tilingData->socParam.commAlg);
  OPS_LOG_I(context->GetNodeName(), " Step comm flag is %u. ND2NZ flag is: %u", tilingData->socParam.isStep, tilingData->socParam.isND2NZ);
    // distinguish between 910A2 and 910A3
    auto it = std::find(VALID_RANK.at(tilingData->socParam.isA3).begin(),
                        VALID_RANK.at(tilingData->socParam.isA3).end(), rankSize);
  OPS_CHECK(it == VALID_RANK.at(tilingData->socParam.isA3).end(),
	OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
	"world_size value is %u, which is illegal.", rankSize), return ge::GRAPH_FAILED);

    mc2tiling::TilingArgs args;
  args.isATrans = is_trans_a? *is_trans_a : 0;
  args.isBTrans = is_trans_b? *is_trans_b : 0;
    args.cmdType = mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER;
    args.rankDim = rankSize;
    args.commTurn = comm_turn;
    args.commAlg = tilingData->socParam.commAlg;

  tilingData->param.isTransposeA = args.isATrans? 1 : 0;
  tilingData->param.isTransposeB = args.isBTrans? 1 : 0;
    tilingData->param.commtype = static_cast<uint32_t>(args.cmdType);

  if (strncmp(reduce_op, "sum", 3) == 0)  { // 3 is index
        OPS_LOG_D("MatmulReduceScatter", "strncmp(reduce_op, sum, 3) == 0");
        tilingData->param.subtype = static_cast<uint8_t>(mc2tiling::HcclReduceOp::HCCL_REDUCE_SUM);
    } else {
        OPS_LOG_D("MatmulReduceScatter", "strncmp(reduce_op, sum, 3) != 0");
        tilingData->param.subtype = static_cast<uint8_t>(mc2tiling::HcclReduceOp::HCCL_REDUCE_RESERVED);
    }

    SetMatmulTilingMatmulReduceScatter(context, *tilingData, args);

    return ge::GRAPH_SUCCESS;
}

struct MatmulReduceScatterCompileInfo {};
static ge::graphStatus TilingParseForMatmulReduceScatter(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulReduceScatter)
    .Tiling(MatmulReduceScatterTilingFunc)
    .TilingParse<MatmulReduceScatterCompileInfo>(TilingParseForMatmulReduceScatter);
}  // namespace optiling