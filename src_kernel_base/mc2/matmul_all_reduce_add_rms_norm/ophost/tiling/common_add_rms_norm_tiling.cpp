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
 * \file common_add_rms_norm_tiling.cc
 * \brief
 */
#ifndef COMMON_ADD_RMS_NORM_CC_
#define COMMON_ADD_RMS_NORM_CC_
#include "common_add_rms_norm_tiling.h"

namespace optiling {
namespace {
constexpr uint32_t DTYPE_KEY_FP16 = 1U;
constexpr uint32_t DTYPE_KEY_FP32 = 2U;
constexpr uint32_t DTYPE_KEY_BF16 = 3U;
constexpr uint32_t UB_FACTOR_B16 = 12288U;
constexpr uint32_t UB_FACTOR_B32 = 10240U;
constexpr uint32_t UB_FACTOR_B16_CUTD = 12096U;
constexpr uint32_t UB_FACTOR_B32_CUTD = 9696U;
constexpr uint32_t BLOCK_ALIGN_NUM = 16U;
constexpr uint32_t VALUE_2 = 2U;
constexpr uint32_t VALUE_16 = 16U;
constexpr uint32_t VALUE_64 = 64U;
constexpr uint32_t VALUE_256 = 256U;
constexpr uint32_t VALUE_1024 = 1024U;
constexpr uint32_t SMALL_REDUCE_NUM = 2000;
constexpr uint32_t COL_ALIGN_SIZE = 260;

using TilingInfo = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float>;
static size_t CeilDiv(size_t x, size_t y)
{
    if (y == 0) {
        return 0;
    } else {
        return ((x - 1) / y + 1);
    }
}
void SetWorkSpaceSize(AddRMSNormTilingOutput &addRmsNormTilingOutput)
{
    size_t sysWorkspaceSize = SYS_WORKSPACE_SIZE;
    size_t usrSize = 256;
    addRmsNormTilingOutput.tilingOut.workSpaceSize = usrSize + sysWorkspaceSize;
}
void SetTilingKey(const uint32_t dtypeKey, uint32_t modeKey, AddRMSNormTilingOutput &addRmsNormTilingOutput)
{
    uint32_t tilingKey = dtypeKey * 10 + modeKey;
    addRmsNormTilingOutput.tilingOut.tilingKey = tilingKey;
}
void SetBlockDim(const uint32_t numRow, const uint32_t blockFactor, AddRMSNormTilingOutput &addRmsNormTilingOutput)
{
    uint32_t useCoreNum = CeilDiv(numRow, blockFactor);
    addRmsNormTilingOutput.tilingOut.blockDim = useCoreNum;
}
void SetTilingData(const TilingInfo &tilingInfo, AddRMSNormTilingOutput &addRmsNormTilingOutput)
{
    auto &&tilingData = addRmsNormTilingOutput.addRmsNormTilingData;
    tilingData.set_num_row(std::get<0>(tilingInfo));
    tilingData.set_num_col(std::get<1>(tilingInfo));
    tilingData.set_block_factor(std::get<2>(tilingInfo));
    tilingData.set_row_factor(std::get<3>(tilingInfo));
    tilingData.set_ub_factor(std::get<4>(tilingInfo));
    tilingData.set_epsilon(std::get<5>(tilingInfo));
    tilingData.set_avg_factor(std::get<6>(tilingInfo));
}
ge::graphStatus AssembleX1Shape(const AddRMSNormTilingDepend &addRmsNormTilingDepend, gert::Shape &xShape)
{
    if (!addRmsNormTilingDepend.useMmOutputAsX1Input) {
        GE_ASSERT_NOTNULL(addRmsNormTilingDepend.arnCtxInfo.x1_shape);
        xShape = addRmsNormTilingDepend.arnCtxInfo.x1_shape->GetStorageShape();
    } else {
        xShape = {addRmsNormTilingDepend.addRmsNormTilingInputFromMm.m,
                  addRmsNormTilingDepend.addRmsNormTilingInputFromMm.n};
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AssembleX1Dtype(const AddRMSNormTilingDepend &addRmsNormTilingDepend, ge::DataType &xDtype)
{
    if (!addRmsNormTilingDepend.useMmOutputAsX1Input) {
        GE_ASSERT_NOTNULL(addRmsNormTilingDepend.arnCtxInfo.x1);
        xDtype = addRmsNormTilingDepend.arnCtxInfo.x1->GetDataType();
    } else {
        xDtype = addRmsNormTilingDepend.addRmsNormTilingInputFromMm.x1Dtype;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus CheckAddRmsNormInputDtype(const gert::TilingContext *context, const ge::DataType &x2Type,
                                          const ge::DataType &gammaType, const ge::DataType &yType,
                                          const ge::DataType &xType)
{
    // residual和gamma数据类型相同
    OP_TILING_CHECK(
        x2Type != gammaType,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "type of residual and gamma should be same"),
        return ge::GRAPH_FAILED);
    // residual和normOut数据类型相同
    OP_TILING_CHECK(
        x2Type != yType,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "type of residual and normOut should be same"),
        return ge::GRAPH_FAILED);
    // residual和输出y数据类型相同
    OP_TILING_CHECK(x2Type != xType,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "type of residual and y should be same"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAddRmsNormInputShape(const gert::TilingContext *context, const gert::StorageShape *x2Shape,
                                          const gert::StorageShape *gammaShape, const gert::StorageShape *yShape,
                                          const gert::StorageShape *xShape)
{
    // residual shape
    OP_TILING_CHECK(x2Shape->GetStorageShape().GetDimNum() != DIM_THREE,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Expect dim of residual to be 3, but got residual_dim:[%lu].",
                                                    x2Shape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    // gamma shape
    OP_TILING_CHECK(gammaShape->GetStorageShape().GetDimNum() != DIM_ONE,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Expect dim of gamma from arn to be 1, but got gamma_dim:[%lu].",
                                                    gammaShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    // normOut shape
    OP_TILING_CHECK(yShape->GetStorageShape().GetDimNum() != DIM_THREE,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Expect dim of normOut from arn to be 3, but got"
                                                    " normOut_dim:[%lu].",
                                                    yShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    // y shape
    OP_TILING_CHECK(xShape->GetStorageShape().GetDimNum() != DIM_THREE,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Expect dim of y from arn to be 3, but got y_dim:[%lu].",
                                                    xShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    // residual和gamma的n值
    OP_TILING_CHECK(x2Shape->GetStorageShape().GetDim(2) != gammaShape->GetStorageShape().GetDim(0),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Expect n of residual and gamma from arn to be same, but got"
                                                    " reisudal_n:[%lu], gamma_n:[%lu].",
                                                    x2Shape->GetStorageShape().GetDim(2),
                                                    gammaShape->GetStorageShape().GetDim(0)),
                    return ge::GRAPH_FAILED);
    // residual和y，normOut的shape
    OP_TILING_CHECK(x2Shape->GetStorageShape() != yShape->GetStorageShape(),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Expect shape of residual and normOut from arn to be same,"
                                                    " but got residual:[%s], normOut:[%s].",
                                                    ge::Shape2String(x2Shape->GetStorageShape()).c_str(),
                                                    ge::Shape2String(yShape->GetStorageShape()).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2Shape->GetStorageShape() != xShape->GetStorageShape(),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Expect shape of residual and y from arn to be same, but"
                                                    " got residual:[%s], y:[%s].",
                                                    ge::Shape2String(x2Shape->GetStorageShape()).c_str(),
                                                    ge::Shape2String(xShape->GetStorageShape()).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus CommonAddResNormTiling::CheckAddRmsNormInput(const gert::TilingContext *context,
                                                             const ARNCtxInfo &arnCtxInfo)
{
    auto x2Type = arnCtxInfo.x2->GetDataType();
    auto gammaType = arnCtxInfo.gamma->GetDataType();
    auto yType = arnCtxInfo.y->GetDataType();
    auto xType = arnCtxInfo.x->GetDataType();
    auto epsilon = arnCtxInfo.epsilon;
    const gert::StorageShape *x2Shape = arnCtxInfo.x2_shape;
    const gert::StorageShape *gammaShape = arnCtxInfo.gamma_shape;
    const gert::StorageShape *yShape = arnCtxInfo.y_shape;
    const gert::StorageShape *xShape = arnCtxInfo.x_shape;
    CheckAddRmsNormInputDtype(context, x2Type, gammaType, yType, xType);
    CheckAddRmsNormInputShape(context, x2Shape, gammaShape, yShape, xShape);
    OP_TILING_CHECK((epsilon != nullptr && (*epsilon <= 0 || *epsilon >= 1)),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Expect epsilon to be in range (0, 1)."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonAddResNormTiling::SetAddRmsNormTilingData(const AddRMSNormTilingDepend &addRmsNormTilingDepend,
                                                                const uint32_t numRow, const int64_t numCol,
                                                                const uint32_t blockFactor,
                                                                AddRMSNormTilingOutput &addRmsNormTilingOutput)
{
    ge::DataType dataType;
    AssembleX1Dtype(addRmsNormTilingDepend, dataType);

    uint32_t dtypeKey = DTYPE_KEY_FP16;
    uint32_t ubFactor = UB_FACTOR_B16;
    if (dataType == ge::DT_FLOAT) {
        dtypeKey = DTYPE_KEY_FP32;
        ubFactor = UB_FACTOR_B32;
    }
    if (dataType == ge::DT_BF16) {
        dtypeKey = DTYPE_KEY_BF16;
    }

    uint32_t modeKey = ModeKey::K_NORMAL; // 0: Normal, 1: SplitD, 2: MergeN, 3: SingleN 4: MultiN
    uint32_t numColAlign = CeilDiv(numCol, BLOCK_ALIGN_NUM) * BLOCK_ALIGN_NUM;
    uint32_t rowFactor = VALUE_64;
    uint64_t ubSize;
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(&addRmsNormTilingDepend.platFormInfos);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (numCol > ubFactor) {
        modeKey = ModeKey::K_SPLIT_D;
        ubFactor = (dataType == ge::DT_FLOAT) ? UB_FACTOR_B32_CUTD : UB_FACTOR_B16_CUTD;
        uint32_t colTileNum = CeilDiv(numCol, ubFactor);
        ubFactor = CeilDiv(numCol, colTileNum * BLOCK_ALIGN_NUM) * BLOCK_ALIGN_NUM;
    } else if (blockFactor == 1) {
        modeKey = ModeKey::K_SINGLE_N;
    } else if (numColAlign <= SMALL_REDUCE_NUM) {
        modeKey = ModeKey::K_MERGE_N;
        uint64_t numColAlignWeight = (dtypeKey == DTYPE_KEY_FP32) ? 24 : 18;
        rowFactor = ubSize / (numColAlign * numColAlignWeight + COL_ALIGN_SIZE);
        ubFactor = rowFactor * numColAlign;
    } else if (dataType == ge::DT_FLOAT16 && numCol == numColAlign) {
        modeKey = ModeKey::K_MULTI_N;
        GE_ASSERT_TRUE(ubSize >= VALUE_1024 + VALUE_256 + numColAlign * VALUE_2);
        GE_ASSERT_TRUE(numColAlign * VALUE_16 + VALUE_64 != 0);
        // reserve 1024byte
        rowFactor = (ubSize - VALUE_1024 - VALUE_256 - numColAlign * VALUE_2) / (numColAlign * VALUE_16 + VALUE_64);
        ubFactor = rowFactor * numColAlign;
        if (rowFactor == 0) {
            modeKey = ModeKey::K_NORMAL;
            rowFactor = VALUE_64;
            ubFactor = UB_FACTOR_B16;
        }
    }
    GE_ASSERT_NOTNULL(addRmsNormTilingDepend.arnCtxInfo.epsilon);
    const float epsilon = *addRmsNormTilingDepend.arnCtxInfo.epsilon;
    float avgFactor = (numCol == 0) ? 0 : 1.0F / numCol;
    SetTilingData(std::make_tuple(numRow, numCol, blockFactor, rowFactor, ubFactor, epsilon, avgFactor),
                  addRmsNormTilingOutput);
    SetTilingKey(dtypeKey, modeKey, addRmsNormTilingOutput);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonAddResNormTiling::Tiling4AddRmsNorm(const AddRMSNormTilingDepend &addRmsNormTilingDepend,
                                                          AddRMSNormTilingOutput &addRmsNormTilingOutput)
{
    const std::string node = addRmsNormTilingDepend.nodeName;
    OPS_LOG_D(node, "Enter CommonAddResNormTiling");

    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(&addRmsNormTilingDepend.platFormInfos);
    uint32_t numCore = ascendcPlatform.GetCoreNumAiv();
    if (addRmsNormTilingDepend.useHalfBlockDim) {
        numCore /= 2U; // 配比1：1时, 1/2个数
    }
    OPS_LOG_D(node, "Core Num: %u, use half %d", numCore, addRmsNormTilingDepend.useHalfBlockDim);
    uint32_t blockFactor = 1U;
    GE_ASSERT_NOTNULL(addRmsNormTilingDepend.arnCtxInfo.gamma_shape);
    const auto &gammaShape = addRmsNormTilingDepend.arnCtxInfo.gamma_shape->GetStorageShape();
    int64_t numCol = gammaShape.GetShapeSize();
    gert::Shape xShape;
    AssembleX1Shape(addRmsNormTilingDepend, xShape);

    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    GE_ASSERT_TRUE(xDimNum >= gammaDimNum);
    uint32_t numRow = 1U;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= xShape.GetDim(i);
    }
    uint32_t tileNum = CeilDiv(numRow, numCore * blockFactor);
    blockFactor *= tileNum;
    OPS_LOG_D(node, "tile num: %u, numRow: %u blockFactor: %u", tileNum, numRow, blockFactor);

    // block dim的值小于等于num_core
    SetBlockDim(numRow, blockFactor, addRmsNormTilingOutput);
    SetWorkSpaceSize(addRmsNormTilingOutput);

    SetAddRmsNormTilingData(addRmsNormTilingDepend, numRow, numCol, blockFactor, addRmsNormTilingOutput);

    OPS_LOG_I(node, "Tiling Key: %u", addRmsNormTilingOutput.tilingOut.tilingKey);
    OPS_LOG_I(node, "Block Dim: %u", addRmsNormTilingOutput.tilingOut.blockDim);
    OPS_LOG_I(node, "Workspace: %u", addRmsNormTilingOutput.tilingOut.workSpaceSize);
    OPS_LOG_I(node, "numRow: %d, numCol: %ld, blockFactor: %d, rowFactor: %d, ubFactor: %d, epsilon: %f, avgFactor: %f",
              numRow, numCol, blockFactor, addRmsNormTilingOutput.addRmsNormTilingData.get_row_factor(),
              addRmsNormTilingOutput.addRmsNormTilingData.get_ub_factor(),
              addRmsNormTilingOutput.addRmsNormTilingData.get_epsilon(),
              addRmsNormTilingOutput.addRmsNormTilingData.get_avg_factor());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

#endif // COMMON_ADD_RMS_NORM_CC_
