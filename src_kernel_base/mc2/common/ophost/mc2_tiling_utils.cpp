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
 * \file mc2_tiling_utils.cpp
 * \brief
 */

#include <cstdlib>
#include "log/ops_log.h"
#include "error/ops_error.h"
#include "hcom_topo_info.h"
#include "graph/utils/type_utils.h"
#include "matmul_v3_tiling_strategy.h"
#include "mc2_tiling_utils.h"

namespace mc2tiling {
namespace {
constexpr char DEUBG_MODE_ENV[] = "ASCEND_MC2_DEBUG_MODE";
constexpr char DEUBG_COMM_ALG_ENV[] = "ASCEND_MC2_DEBUG_COMM_ALG";
constexpr char DEUBG_STEP_SIZE_ENV[] = "ASCEND_MC2_DEBUG_STEP_SIZE";
constexpr char HCCL_BUFFSIZE[] = "HCCL_BUFFSIZE";
}
uint8_t Mc2TilingUtils::GetDebugMode()
{
    auto debugModeEnv = getenv(DEUBG_MODE_ENV);
    uint8_t debugMode = 0;
    if (debugModeEnv != nullptr) {
        debugMode = static_cast<uint8_t>(std::atoi(debugModeEnv));
    }
    OPS_LOG_I("", "Get ASCEND_MC2_DEBUG_MODE is %u", debugMode);
    return debugMode;
}

uint8_t Mc2TilingUtils::GetDebugCommAlg()
{
    auto debugCommAlgEnv = getenv(DEUBG_COMM_ALG_ENV);
    uint8_t debugCommAlg = 0;
    if (debugCommAlgEnv != nullptr) {
        debugCommAlg = static_cast<uint8_t>(std::atoi(debugCommAlgEnv));
    }
    OPS_LOG_I("", "Get ASCEND_MC2_DEBUG_COMM_ALG is %u", debugCommAlg);
    return debugCommAlg;
}

uint8_t Mc2TilingUtils::GetDebugStepSize()
{
    auto debugStepSizeEnv = getenv(DEUBG_STEP_SIZE_ENV);
    uint8_t debugStepSize = 0;
    if (debugStepSizeEnv != nullptr) {
        debugStepSize = static_cast<uint8_t>(std::atoi(debugStepSizeEnv));
    }
    OPS_LOG_I("", "Get ASCEND_MC2_DEBUG_STEP_SIZE is %u", debugStepSize);
    return debugStepSize;
}

matmul_tiling::DataType ConvertGeTypeToMmType(const std::string &opName, ge::DataType type)
{
    static const std::map<ge::DataType, matmul_tiling::DataType> GE_TO_MM_MAP =
    {
        {ge::DT_BF16, matmul_tiling::DataType::DT_BFLOAT16},
        {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
        {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    };

    auto iterator = GE_TO_MM_MAP.find(type);
    if (iterator != GE_TO_MM_MAP.end()) {
        return iterator->second;
    }

    OPS_LOG_I(opName, "cannot find matmul_tiling datatype according to ge datatype: %d", static_cast<int32_t>(type));
    return matmul_tiling::DataType::DT_MAX;
}

ge::DataType ConvertMmTypeToGeType(const std::string &opName, matmul_tiling::DataType type)
{
    static const std::map<matmul_tiling::DataType, ge::DataType> MM_TO_GE_MAP =
    {
        {matmul_tiling::DataType::DT_BFLOAT16, ge::DT_BF16},
        {matmul_tiling::DataType::DT_FLOAT16, ge::DT_FLOAT16},
        {matmul_tiling::DataType::DT_FLOAT, ge::DT_FLOAT},
    };

    auto iterator = MM_TO_GE_MAP.find(type);
    if (iterator != MM_TO_GE_MAP.end()) {
        return iterator->second;
    }

    OPS_LOG_I(opName, "cannot find ge datatype according to  matmul_tiling datatype: %d", static_cast<int32_t>(type));
    return ge::DT_MAX;
}

uint64_t GetDataTypeSize(const std::string &opName, ge::DataType type)
{
    static const std::map<ge::DataType, int64_t> DATA_TYPE_SIZE_MAP =
    {
        {ge::DT_BF16, 2},
        {ge::DT_FLOAT16, 2},
        {ge::DT_FLOAT, 4},
    };

    auto iterator = DATA_TYPE_SIZE_MAP.find(type);
    if (iterator != DATA_TYPE_SIZE_MAP.end()) {
        return iterator->second;
    }

    OPS_LOG_I(opName, "cannot find datatype size according to ge datatype: %d", static_cast<int32_t>(type));
    return 0;
}

HcclDataType ConvertGeTypeToHcclType(const std::string &opName, ge::DataType type)
{
    static const std::map<ge::DataType, HcclDataType> HCCL_DATA_TYPE_MAP =
    {
        {ge::DataType::DT_INT8, HcclDataType::HCCL_DATA_TYPE_INT8},
        {ge::DataType::DT_UINT8, HcclDataType::HCCL_DATA_TYPE_UINT8},
        {ge::DataType::DT_INT16, HcclDataType::HCCL_DATA_TYPE_INT16},
        {ge::DataType::DT_UINT16, HcclDataType::HCCL_DATA_TYPE_UINT16},
        {ge::DataType::DT_INT32, HcclDataType::HCCL_DATA_TYPE_INT32},
        {ge::DataType::DT_UINT32, HcclDataType::HCCL_DATA_TYPE_UINT32},
        {ge::DataType::DT_FLOAT16, HcclDataType::HCCL_DATA_TYPE_FP16},
        {ge::DataType::DT_FLOAT, HcclDataType::HCCL_DATA_TYPE_FP32},
        {ge::DataType::DT_BF16, HcclDataType::HCCL_DATA_TYPE_BFP16},
    };

    auto iterator = HCCL_DATA_TYPE_MAP.find(type);
    if (iterator != HCCL_DATA_TYPE_MAP.end()) {
        return iterator->second;
    }

    OPS_LOG_I(opName, "cannot find HcclDataType according to ge datatype: %d", static_cast<int32_t>(type));
    return HcclDataType::HCCL_DATA_TYPE_RESERVED;
}

bool CheckSuppportedFormat(ge::Format format)
{
    static const std::set<ge::Format> SUPPORT_FORMAT_SET =
    {
        ge::FORMAT_ND
    };

    return (SUPPORT_FORMAT_SET.count(format) != 0);
}

bool IsDeterministic()
{
    if (getenv(HCCL_DETERMINISTIC) == nullptr) {
        return false;
    }
    std::string envStr(getenv(HCCL_DETERMINISTIC));
    std::transform(envStr.begin(), envStr.end(), envStr.begin(), ::toupper);
    if (envStr == "FALSE") {
        return false;
    }
    OPS_LOG_I("MatmulReduceScatter", "Set HCCL_DETERMINISTIC is true");
    return true;
}

uint8_t Mc2GetCommAlgo(uint32_t rankDim, uint64_t mValue, const char *group, const gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "fail to get platform info!");
        return COMM_ALG_DEFAULT;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    auto debugCommAlg = mc2tiling::Mc2TilingUtils::GetDebugCommAlg();
    if (rankDim == 2) { // 如果是2p
        if ((debugCommAlg != 0) && (debugCommAlg != COMM_ALG_FULL_MESH)) {
            OPS_LOG_E(context->GetNodeName(), " CommAlgo %u is not supported when rank dim is 2.", debugCommAlg);
            return COMM_ALG_DEFAULT;
        }
        return COMM_ALG_FULL_MESH;
    }

    ge::HcomTopoInfo::TopoInfo topoInfo;
    if (!ge::HcomTopoInfo::Instance().TryGetGroupTopoInfo(group, topoInfo)) {
        OPS_LOG_W(context->GetNodeName(), " GroupTopoInfo not set.");
        return COMM_ALG_DEFAULT;
    }
    auto commSets = topoInfo.topo_level_descs[static_cast<int32_t>(ge::HcomTopoInfo::TopoLevel::L0)].comm_sets;
    OPS_LOG_D(context->GetNodeName(), " comm_sets from TopoInfo is %u, COMM_MESH is %u", commSets, COMM_MESH);

    // 如果平台只支持fullmesh
    if (commSets == COMM_MESH) {
        if ((debugCommAlg != 0) && (debugCommAlg != COMM_ALG_FULL_MESH)) { // 环境变量设置非fullmesh
            OPS_LOG_E(context->GetNodeName(), " CommAlg %u is not supported.", debugCommAlg);
            return COMM_ALG_DEFAULT;
        }
        return COMM_ALG_FULL_MESH;
    }

    // reducescatter支持doublering或switch
    if (debugCommAlg != 0) { // 环境变量设置非doublering/switch
        if ((debugCommAlg != COMM_ALG_DOUBLE_RING) && (debugCommAlg != COMM_ALG_SWITCH_WING)) {
            OPS_LOG_E(context->GetNodeName(), " CommAlg %u is not supported.", debugCommAlg);
            return COMM_ALG_DEFAULT;
        }
        return debugCommAlg;
    }
    if ((mValue % CHECK_VALUE_ODD != 0) || (mValue % rankDim != 0)) {
        OPS_LOG_W(context->GetNodeName(), " m value is odd or cannot be devided by rankDim.");
        return COMM_ALG_DEFAULT;
    }
    return COMM_ALG_DOUBLE_RING;
}

bool CheckRankSize(const platform_ascendc::SocVersion socVersion, const uint32_t rankSize)
{
    static const std::map<platform_ascendc::SocVersion, std::set<uint32_t>> SUPPORT_RANK_SIZE_SET =
    {
        {platform_ascendc::SocVersion::ASCEND310P, {1, 2, 4}},
        {platform_ascendc::SocVersion::ASCEND910B, {1, 2, 4, 8}},
    };
    auto it = SUPPORT_RANK_SIZE_SET.find(socVersion);
    if ( it != SUPPORT_RANK_SIZE_SET.end()) {
        return it->second.count(rankSize) != 0;
    }

    return false;
}

bool CheckDataTypeVaild(ge::DataType type, std::initializer_list<ge::DataType> supportDtypeList)
{
    return std::find(supportDtypeList.begin(), supportDtypeList.end(), type) != supportDtypeList.end();
}

void UpdateMatmulV3Args(optiling::matmul_v3_advanced::MatMulV3Args &mmV3Args, const TilingArgs &args, const char* opName)
{
    mmV3Args.opName = opName;
    mmV3Args.isATrans = args.isATrans;
    mmV3Args.isBTrans = args.isBTrans;
    mmV3Args.isHf32 = false;
    mmV3Args.hasBias = args.isBias;
    mmV3Args.aType = args.geAType;
    mmV3Args.bType = args.geBType;
    mmV3Args.cType = args.geCType;
    mmV3Args.biasType = args.geBiasType;
    mmV3Args.aFormat = ge::FORMAT_ND;
    mmV3Args.bFormat = ge::FORMAT_ND;
    mmV3Args.outFormat = ge::FORMAT_ND;
    mmV3Args.mValue = args.mValue;
    mmV3Args.nValue = args.nValue;
    mmV3Args.kValue = args.kValue;
    mmV3Args.aDtypeSize = GetDataTypeSize(opName, mmV3Args.aType);
    mmV3Args.bDtypeSize = GetDataTypeSize(opName, mmV3Args.bType);
}

uint32_t Mc2TilingUtils::GetCommSets(const char *group)
{
    ge::HcomTopoInfo::TopoInfo topoInfo;
    if (!ge::HcomTopoInfo::Instance().TryGetGroupTopoInfo(group, topoInfo)) {
        OPS_LOG_W("", " GroupTopoInfo not set.");
        return COMM_UNDEFINED;
    }
    auto commSets = topoInfo.topo_level_descs[static_cast<int32_t>(ge::HcomTopoInfo::TopoLevel::L0)].comm_sets;
    OPS_LOG_I("", "Get commSets is %u", commSets);
    return commSets;
}

ge::graphStatus Mc2TilingUtils::CommonParamCheck(const gert::TilingContext* context)
{
    const gert::StorageShape* aShape = context->GetInputShape(0);
    const gert::StorageShape* bShape = context->GetInputShape(1);
    OPS_CHECK(aShape == nullptr || bShape == nullptr,
         OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "the shape is invalid"), return ge::GRAPH_FAILED);

    uint64_t aShapeDimNum = aShape->GetStorageShape().GetDimNum();
    uint64_t bShapeDimNum = bShape->GetStorageShape().GetDimNum();
    OPS_CHECK(aShapeDimNum != 2 || bShapeDimNum != 2,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "the dimNum is not two"), return ge::GRAPH_FAILED);

    auto aTensor = context->GetInputDesc(0);
    auto bTensor = context->GetInputDesc(1);
    auto output = context->GetOutputDesc(0);
    OPS_CHECK(aTensor == nullptr || bTensor == nullptr || output == nullptr,
          OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "the tensor is invalid"), return ge::GRAPH_FAILED);

    auto aShapeFormat = aTensor->GetStorageFormat();
    auto bShapeFormat = bTensor->GetStorageFormat();
    auto outputFormat = output->GetStorageFormat();
    OPS_CHECK(aShapeFormat != outputFormat,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "a shape Format, output Format are not same"), return ge::GRAPH_FAILED);
    OPS_CHECK(
        (mc2tiling::SUPPORTED_FORMAT.count(aShapeFormat) == 0 || mc2tiling::SUPPORTED_FORMAT.count(bShapeFormat) == 0),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                        "a shape Format, b shape Format only support ND, the format is %s",
                                        ge::TypeUtils::FormatToAscendString(aShapeFormat).GetString()),
                                        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

mc2tiling::HcclDataType Mc2TilingUtils::GetDataType(ge::DataType type)
{
    if (mc2tiling::HCCL_DATA_TYPE.find(type) != mc2tiling::HCCL_DATA_TYPE.end()) {
        return mc2tiling::HCCL_DATA_TYPE.at(type);
    }
    return mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED;
}

uint64_t Mc2TilingUtils::GetMaxWindowSize()
{
    uint16_t defaultWindowSize = 200;
    if (getenv(HCCL_BUFFSIZE) == nullptr) {
        OPS_LOG_D("", "Env HCCL_BUFFSIZE don't set");
    } else {
        try {
            std::string envStr(getenv(HCCL_BUFFSIZE));
            defaultWindowSize = std::stoi(envStr);
        } catch (...) {
            OPS_LOG_E("", "Unknown Exception encountered when parser env HCCL_BUFFERSIZE");
        }
    }
    const uint64_t maxWindowSize = static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
    OPS_LOG_I("", "Get maxWindowSize is %lu", maxWindowSize);
    return maxWindowSize;
}

bool Mc2TilingUtils::CheckRankSize(platform_ascendc::SocVersion socVersion, uint32_t rankSize)
{
    auto it = supportedRankSizeSet.find(socVersion);
    if ( it != supportedRankSizeSet.end()) {
        return it->second.count(rankSize) != 0;
    }

    return false;
}


} // namespace mc2tiling
