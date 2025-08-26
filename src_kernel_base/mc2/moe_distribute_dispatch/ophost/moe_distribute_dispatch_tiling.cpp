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
 * \file moe_distribute_dispatch_tiling.cc
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "log/ops_log.h"
#include "hcom_topo_info.h"
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "../moe_distribute_dispatch_tiling.h"

using namespace AscendC;
using namespace ge;
namespace {
    constexpr uint32_t X_INDEX = 0U;
    constexpr uint32_t EXPERT_IDS_INDEX = 1U;
    constexpr uint32_t SCALES_INDEX = 2U;
    constexpr uint32_t X_ACTIVE_MASK_INDEX = 3U;
    constexpr uint32_t OUTPUT_EXPAND_X_INDEX = 0U;
    constexpr uint32_t OUTPUT_DYNAMIC_SCALES_INDEX = 1U;
    constexpr uint32_t OUTPUT_EXPAND_IDX_INDEX = 2U;
    constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3U;
    constexpr uint32_t OUTPUT_EP_RECV_COUNTS_INDEX = 4U;
    constexpr uint32_t OUTPUT_TP_RECV_COUNTS_INDEX = 5U;

    constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
    constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
    constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
    constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
    constexpr uint32_t ATTR_GROUP_TP_INDEX = 4;
    constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 5;
    constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 6;
    constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
    constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
    constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
    constexpr uint32_t ATTR_QUANT_MODE_INDEX = 10;
    constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 11;
    constexpr uint32_t ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX = 12;

    constexpr uint32_t TWO_DIMS = 2;
    constexpr uint32_t UNQUANT_MODE = 0;
    constexpr uint32_t DYNAMIC_QUANT_MODE = 2;
    constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
    constexpr uint32_t BLOCK_SIZE_A2 = 32;
    constexpr uint32_t MAX_K_VALUE_A2 = 16;
    constexpr uint32_t MIN_K_VALUE_A2 = 2;
    constexpr uint32_t LAYERED_SUPPORT_K = 8;
    constexpr int32_t MAX_HIDDEN_SIZE_A2 = 7168;
    constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 256;
    constexpr int32_t MAX_MOE_EXPERT_NUMS_A2 = 512;
    constexpr uint32_t MAX_BATCH_SIZE_LAYERED_A2 = 128;
    constexpr uint32_t MAX_BATCH_SIZE_A2 = 256;
    const char *K_INNER_DEBUG = "MoeDistributeDispatch Tiling Debug";

    constexpr uint32_t NUM_10 = 10;
    constexpr uint32_t NUM_100 = 100;
    constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
    constexpr uint32_t USER_WORKSPACE_A2 = 1 * 1024 * 1024; // moeExpertNum_ * sizeof(uint32_t) + epWorldSize_ * 2 * 32

    constexpr uint64_t INIT_TILINGKEY = 1000;
    constexpr uint64_t TILING_KEY_BASE_A2 = 2000000000;
    constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 100000000;
}

namespace optiling {

// a2函数
static ge::graphStatus MoeDistributeDispatchA2CheckAttrAndSetTiling(gert::TilingContext *context, MoeDistributeDispatchA2Info& info)
{
    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr, OPS_LOG_E(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_TP_WORLD_SIZE_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int>(ATTR_TP_RANK_ID_INDEX);
    auto expertSharedTypePtr = attrs->GetAttrPointer<int>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int>(ATTR_GLOBAL_BS_INDEX);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX);

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OPS_CHECK(expertIdStorageShape == nullptr, OPS_LOG_E(K_INNER_DEBUG, "expertIdShape is null."), return GRAPH_FAILED);
    int32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);

    OPS_CHECK(groupEpPtr == nullptr || strlen(groupEpPtr) == 0,
        OPS_LOG_E(K_INNER_DEBUG, "groupEp is invalid."), return GRAPH_FAILED);
    OPS_CHECK(epWorldSizePtr == nullptr || *epWorldSizePtr <= 0 || *epWorldSizePtr > MAX_EP_WORLD_SIZE_A2 ||
        *epWorldSizePtr % RANK_NUM_PER_NODE_A2 != 0,
        OPS_LOG_E(K_INNER_DEBUG, "epWorldSize is invalid."), return GRAPH_FAILED);
    OPS_CHECK(epRankIdPtr == nullptr || *epRankIdPtr < 0 || *epRankIdPtr >= *epWorldSizePtr,
        OPS_LOG_E(K_INNER_DEBUG, "epRankId is invalid."), return GRAPH_FAILED);
    OPS_CHECK(moeExpertNumPtr == nullptr || *moeExpertNumPtr % *epWorldSizePtr != 0 ||
        *moeExpertNumPtr <= 0 || *moeExpertNumPtr > MAX_MOE_EXPERT_NUMS_A2,
        OPS_LOG_E(K_INNER_DEBUG, "moeExpertNum is invalid."), return GRAPH_FAILED);
    OPS_CHECK(tpWorldSizePtr == nullptr,
        OPS_LOG_E(K_INNER_DEBUG, "tpWorldSize is null."), return GRAPH_FAILED);
    OPS_CHECK(tpRankIdPtr == nullptr,
        OPS_LOG_E(K_INNER_DEBUG, "tpRankId is null."), return GRAPH_FAILED);
    OPS_CHECK(expertSharedTypePtr == nullptr,
        OPS_LOG_E(K_INNER_DEBUG, "expertSharedType is null."), return GRAPH_FAILED);
    OPS_CHECK(sharedExpertRankNumPtr == nullptr,
        OPS_LOG_E(K_INNER_DEBUG, "sharedExpertRankNum is null."), return GRAPH_FAILED);
    OPS_CHECK(quantModePtr == nullptr || (*quantModePtr != UNQUANT_MODE && *quantModePtr != DYNAMIC_QUANT_MODE),
        OPS_LOG_E(K_INNER_DEBUG, "quantMode is invalid."), return GRAPH_FAILED);
    OPS_CHECK(globalBsPtr == nullptr,
        OPS_LOG_E(K_INNER_DEBUG, "globalBs is null."), return GRAPH_FAILED);
    OPS_CHECK(expertTokenNumsTypePtr == nullptr || *expertTokenNumsTypePtr < 0 || *expertTokenNumsTypePtr > 1,
        OPS_LOG_E(K_INNER_DEBUG, "expertTokenNumsType is invalid. Must be 0 or 1. "), return GRAPH_FAILED);

    info.epWorldSize = *epWorldSizePtr;
    info.tpWorldSize = static_cast<uint32_t>(0);
    info.epRankId = *epRankIdPtr;
    info.tpRankId = static_cast<uint32_t>(0);
    info.expertSharedType = static_cast<uint32_t>(0);
    info.sharedExpertRankNum = static_cast<uint32_t>(0);
    info.moeExpertNum = *moeExpertNumPtr;
    info.quantMode = *quantModePtr;
    if (*globalBsPtr == 0) {
        info.globalBs = *epWorldSizePtr * bs;
    } else {
        info.globalBs = *globalBsPtr;
    }
    info.expertTokenNumsType = *expertTokenNumsTypePtr;

    OPS_LOG_D(K_INNER_DEBUG, "quantMode=%u", info.quantMode);
    OPS_LOG_D(K_INNER_DEBUG, "globalBs=%u", info.globalBs);
    OPS_LOG_D(K_INNER_DEBUG, "expertTokenNumsType=%u", info.expertTokenNumsType);
    OPS_LOG_D(K_INNER_DEBUG, "expertSharedType=%u", info.expertSharedType);
    OPS_LOG_D(K_INNER_DEBUG, "sharedExpertRankNum=%u", info.sharedExpertRankNum);
    OPS_LOG_D(K_INNER_DEBUG, "moeExpertNum=%u", info.moeExpertNum);
    OPS_LOG_D(K_INNER_DEBUG, "epWorldSize=%u", info.epWorldSize);
    OPS_LOG_D(K_INNER_DEBUG, "tpWorldSize=%u", info.tpWorldSize);
    OPS_LOG_D(K_INNER_DEBUG, "epRankId=%u", info.epRankId);
    OPS_LOG_D(K_INNER_DEBUG, "tpRankId=%u", info.tpRankId);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA2CheckShapeAndSetTiling(gert::TilingContext *context,
                                                                     MoeDistributeDispatchA2Info &info,
                                                                     const bool isLayered)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_I(nodeName, "MoeDistributeDispatchA2 MoeDistributeDispatchA2CheckShapeAndSetTiling.");
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);

    OPS_CHECK(xStorageShape == nullptr,
        OPS_LOG_E(K_INNER_DEBUG, "xShape is null."), return GRAPH_FAILED);
    OPS_CHECK(expertIdStorageShape == nullptr,
        OPS_LOG_E(K_INNER_DEBUG, "expertIdShape is null."), return GRAPH_FAILED);
    OPS_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OPS_LOG_E(K_INNER_DEBUG, "x dims is invalid."), return false);
    OPS_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OPS_LOG_E(K_INNER_DEBUG, "expertId dims is invalid."), return false);
    OPS_LOG_D(nodeName, "X dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OPS_LOG_D(nodeName, "X dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));
    OPS_LOG_D(nodeName, "expertId dim0 = %ld", expertIdStorageShape->GetStorageShape().GetDim(0));
    OPS_LOG_D(nodeName, "expertId dim1 = %ld", expertIdStorageShape->GetStorageShape().GetDim(1));

    uint32_t h = xStorageShape->GetStorageShape().GetDim(1);
    uint32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);
    uint32_t k = expertIdStorageShape->GetStorageShape().GetDim(1);
    bool isScales = (scalesStorageShape != nullptr);
    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr, OPS_LOG_E(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);
    auto quantModePtr = attrs->GetAttrPointer<int>(ATTR_QUANT_MODE_INDEX);
    OPS_CHECK(h % BLOCK_SIZE_A2 != 0 || h <= 0 || h > MAX_HIDDEN_SIZE_A2,
        OPS_LOG_E(K_INNER_DEBUG, "hiddensize is invalid."), return GRAPH_FAILED);
    OPS_CHECK(bs <= 0,
        OPS_LOG_E(K_INNER_DEBUG, "batchsize is invalid."), return GRAPH_FAILED);
    OPS_CHECK(k < MIN_K_VALUE_A2 || k > MAX_K_VALUE_A2,
        OPS_LOG_E(K_INNER_DEBUG, "k is invalid."), return GRAPH_FAILED);
    OPS_CHECK(*quantModePtr == UNQUANT_MODE && isScales,
        OPS_LOG_E(K_INNER_DEBUG, "scales should be null when quantMode is unQuant."), return GRAPH_FAILED);
    const uint32_t maxBatchSize = isLayered ? MAX_BATCH_SIZE_LAYERED_A2 : MAX_BATCH_SIZE_A2;
    OPS_CHECK(bs > maxBatchSize,
                    OPS_LOG_E(nodeName, "Batchsize must be smaller than %u.", maxBatchSize),
                    return ge::GRAPH_FAILED);
    info.isQuant = isScales;
    info.bs = bs;
    info.k = k;
    info.h = h;

    OPS_LOG_D(K_INNER_DEBUG, "isQuant=%d", info.isQuant);
    OPS_LOG_D(K_INNER_DEBUG, "batchSize=%u", info.bs);
    OPS_LOG_D(K_INNER_DEBUG, "k=%u", info.k);
    OPS_LOG_D(K_INNER_DEBUG, "hidenSize=%u", info.h);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA2GetPlatformInfoAndSetTiling(gert::TilingContext *context, MoeDistributeDispatchA2Info& info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    // todo boxi debug
    OPS_LOG_D(K_INNER_DEBUG, "aivNum=%u", info.aivNum);
    OPS_LOG_D(K_INNER_DEBUG, "ubSize=%lu", info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

static bool MoeDistributeDispatchA2IsLayered()
{
    const char* hcclIntraPcieEnable = getenv("HCCL_INTRA_PCIE_ENABLE");
    const char* hcclIntraRoceEnable = getenv("HCCL_INTRA_ROCE_ENABLE");

    if (hcclIntraPcieEnable == nullptr || hcclIntraRoceEnable == nullptr) {
        OPS_LOG_D(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE or HCCL_INTRA_ROCE_ENABLE don't set");
        return false;
    } else if (strcmp(hcclIntraPcieEnable, "1") == 0 && strcmp(hcclIntraRoceEnable, "0") == 0) {
        OPS_LOG_D(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE = 1 and HCCL_INTRA_ROCE_ENABLE = 0, use layered solution.");
        return true;
    }
    OPS_LOG_D(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE != 1 or HCCL_INTRA_ROCE_ENABLE != 0, use default solution.");
    return false;
}

static uint64_t MoeDistributeDispatchA2CalcTilingKey(gert::TilingContext *context, const bool isLayered)
{
    uint64_t tilingKey = TILING_KEY_BASE_A2 + INIT_TILINGKEY;

    if (isLayered) {
        tilingKey += TILING_KEY_LAYERED_COMM_A2;
    }

    auto attrs = context->GetAttrs();
    auto quantModePtr = attrs->GetAttrPointer<int>(ATTR_QUANT_MODE_INDEX);
    tilingKey += static_cast<uint64_t>(*quantModePtr);

    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isScales = (scalesStorageShape != nullptr);
    tilingKey += static_cast<uint64_t>((isScales ? NUM_10 : 0));

    OPS_LOG_D(K_INNER_DEBUG, "tilingKey=%lu", tilingKey);

    return tilingKey;
}

static ge::graphStatus MoeDistributeDispatchA2TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_I(nodeName, "Enter MoeDistributeDispatchA2 tiling func.");

    // 1. tilingData
    MoeDistributeDispatchA2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchA2TilingData>();
    OPS_CHECK(tilingData == nullptr, OPS_REPORT_VECTOR_INNER_ERR(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);
    OPS_LOG_I(nodeName, "MoeDistributeDispatchA2 get tilingData.");
    MoeDistributeDispatchA2Info& info = tilingData->moeDistributeDispatchInfo;
    OPS_LOG_I(nodeName, "MoeDistributeDispatchA2 get tilingData info.");

    bool isLayered = MoeDistributeDispatchA2IsLayered();
    OPS_CHECK(MoeDistributeDispatchA2CheckShapeAndSetTiling(context, info, isLayered) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "MoeDistributeDispatchA2 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OPS_CHECK(MoeDistributeDispatchA2CheckAttrAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "MoeDistributeDispatchA2 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OPS_CHECK(MoeDistributeDispatchA2GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "MoeDistributeDispatchA2 GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);

    uint64_t tilingKey = MoeDistributeDispatchA2CalcTilingKey(context, isLayered);
    context->SetTilingKey(tilingKey);
    // 2. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OPS_CHECK(workSpaces == nullptr, OPS_REPORT_VECTOR_INNER_ERR(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + USER_WORKSPACE_A2;

    // 3. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    uint32_t opType = 18; // batch write=18,
    std::string algConfig = "MultiPut=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);

    OPS_LOG_I(nodeName, "Leave MoeDistributeDispatchA2 tiling func.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchTilingFunc(gert::TilingContext* context)
{
    return MoeDistributeDispatchA2TilingFuncImpl(context);
}

struct MoeDistributeDispatchCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeDispatch(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeDispatch)
    .Tiling(MoeDistributeDispatchTilingFunc)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatch);
} // namespace optiling