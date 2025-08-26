/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_reg_base_tiling.cpp
 * \brief
 */
#include <algorithm>
#include <map>
#include <numeric>

#include "cube_tiling_runtime.h"
#include "graph/utils/type_utils.h"
#include "op_tiling_util.h"
#include "register/op_impl_registry.h"
#include "weight_quant_batch_matmul_v2_reg_base_tiling.h"

using AscendC::BLOCK_CUBE;    // uint32_t
using AscendC::ONE_BLK_SIZE;  // uint32_t


namespace {
constexpr uint64_t INT4_DTYPE_PARAM = 2;
constexpr uint32_t WORKSPACE_SIZE = static_cast<uint32_t>(16 * 1024 * 1024);
constexpr int32_t DB_BUFFER = 2;
constexpr int32_t EXTRA_GROUP_NUM = 2;
constexpr uint64_t NK_ALIGN_SIZE = 64;  // 当前仅支持B矩阵64对齐

constexpr int64_t B32_BITS = 32;
constexpr int64_t B16_BITS = 16;
constexpr int64_t B8_BITS = 8;
constexpr int64_t B4_BITS = 4;
constexpr int64_t BLK_NUM_V100 = 32;
constexpr int64_t L0A_SIZE_V100 = 64 * 1024;
constexpr int64_t L0C_SIZE_V100 = 256 * 1024;
constexpr int64_t UB_SIZE_V100 = 248 * 1024; // 当前框架获取的UB空间为240KB，问题解决后删除
constexpr int64_t MTE2_MIN_LOAD_SIZE_V100 = 32 * 1024;  // 实测16KB带宽较差，此处取32KB
constexpr int64_t MIN_CACHE_LINE_V100 = 128;
constexpr int64_t CACHE_LINE_V100 = 256;
constexpr int64_t GROUP_ALIGN_SIZE = 32;

constexpr double FREQUENCY_v100 = 1.6;
constexpr double HBM_BW_V100 = 1.6;
constexpr double L2_BW_V100 = 5.4;

int64_t GetDtypeBits(ge::DataType dtype)
{
    if (dtype == ge::DT_INT4) {
        return B4_BITS;
    } else if (dtype == ge::DT_INT8) {
        return B8_BITS;
    } else if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return B16_BITS;
    } else if (dtype == ge::DT_FLOAT) {
        return B32_BITS;
    } else {
        return 0;
    }
}

}  // namespace
namespace optiling {

bool WeightQuantBatchMatmulV2RegBase::IsCapable() {
    if (matmulInfoPtr_->antiQuantType == QuantType::PER_TENSOR) {
        OPS_LOG_I(opName_, "the reg base template doesn't support the per-tensor mode");
        return false;
    }

    if (matmulInfoPtr_->bDtype != ge::DT_INT4 && matmulInfoPtr_->bDtype != ge::DT_INT8) {
        OPS_LOG_I(opName_, "the reg base template only support weight dtype int4 or int8, but is [%s]",
                ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->bDtype).GetString());
        return false;
    }

    if (matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL) {
        OPS_LOG_I(opName_, "the reg base template doesn't support the per-channel mode");
        return false;
    }

    if (matmulInfoPtr_->cDtype == ge::DT_INT8) {
        OPS_LOG_I(opName_, "the reg base template doesn't support the dtype of y as int8");
        return false;
    }

    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2RegBase::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "unable to get pointer of tiling data"),
                    return ge::GRAPH_FAILED);

    tilingData_->set_kSize(matmulInfoPtr_->kSize);
    tilingData_->set_nSize(matmulInfoPtr_->nSize);
    tilingData_->set_mSize(matmulInfoPtr_->mSize);

    PlatformParam platformParam = {compileInfoPtr_->aicNum,
                                   compileInfoPtr_->aicNum,
                                   UB_SIZE_V100,
                                   static_cast<int64_t>(compileInfoPtr_->l1Size),
                                   L0A_SIZE_V100,
                                   L0A_SIZE_V100,
                                   L0C_SIZE_V100,
                                   CACHE_LINE_V100,
                                   MIN_CACHE_LINE_V100,
                                   FREQUENCY_v100,
                                   HBM_BW_V100,
                                   L2_BW_V100};
    tilingSolver_.SetPlatformParam(platformParam);
    tilingSolver_.SetShape(matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize,
                           matmulInfoPtr_->groupSize);
    WeightQuantBmmAttr attr = {matmulInfoPtr_->transA, matmulInfoPtr_->transB, matmulInfoPtr_->hasBias,
                               matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ, matmulInfoPtr_->hasAntiQuantOffset};
    tilingSolver_.SetAttr(opName_, attr);
    tilingSolver_.SetDtypeBits(GetDtypeBits(matmulInfoPtr_->aDtype), GetDtypeBits(matmulInfoPtr_->bDtype), 0);
    tilingSolver_.SetQuantType(matmulInfoPtr_->antiQuantType);
    if (matmulInfoPtr_->hasBias) {
        tilingSolver_.SetDtypeBits(GetDtypeBits(matmulInfoPtr_->aDtype), GetDtypeBits(matmulInfoPtr_->bDtype),
                                   GetDtypeBits(matmulInfoPtr_->biasDtype));
    }
    OP_CHECK(!tilingSolver_.GetBasicBlockTiling(),
             VECTOR_INNER_ERR_REPORT_TILIING(opName_, "Unable to get matmul tiling for mnk[%lu, %lu, %lu]",
                                             matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize),
             return ge::GRAPH_FAILED);
    SetMatmulTiling();
    SetBubTiling();
    SetAdditionalParam();

    return ge::GRAPH_SUCCESS;
}

uint64_t WeightQuantBatchMatmulV2RegBase::GetTilingKey() const
{
    // biasType为10表示bias数据类型和x不同，为0表示和x相同。
    uint32_t biasType = (matmulInfoPtr_->biasDtype == ge::DT_FLOAT) ? 10 : 0;
    uint32_t isWeightNz = (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) ? 10 : 0;
    return RecursiveSum(matmulInfoPtr_->transA, matmulInfoPtr_->transB, matmulInfoPtr_->antiQuantType,
                        matmulInfoPtr_->hasAntiQuantOffset, KernelTemplateType::ANTI_REG, biasType, isWeightNz);
}

ge::graphStatus WeightQuantBatchMatmulV2RegBase::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    workspaceSize_ += tilingData_->get_cubeBlockDimN() * tilingData_->get_cubeBlockDimM() * sizeof(uintptr_t);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2RegBase::PostTiling()
{
    OPS_LOG_D(opName_, "final tiling data size: %zu", tilingData_->GetDataSize());

    OP_TILING_CHECK(tilingData_->GetDataSize() % sizeof(uint64_t) != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data size[%zu] not aligned to 8",
                                                    tilingData_->GetDataSize()),
                    return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingData_->GetDataSize());
    context_->SetBlockDim(tilingData_->get_cubeBlockDimM() * tilingData_->get_cubeBlockDimN());

    size_t *workspaces = context_->GetWorkspaceSizes(1);  // set workspace
    workspaces[0] = workspaceSize_;

    tilingData_->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    PrintCVTilingData(true);
    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2RegBase::SetBubTiling()
{
    int64_t nBubSize, kBubSize;
    if (matmulInfoPtr_->bDtype == ge::DT_INT8 && matmulInfoPtr_->groupSize > 0) {
        GetBubTilingA16W8NDPerGroup(nBubSize, kBubSize);
    } else if (matmulInfoPtr_->bDtype == ge::DT_INT8 && matmulInfoPtr_->transB) {
        GetBubTilingA16W8Trans(nBubSize, kBubSize);
    } else if (matmulInfoPtr_->bDtype == ge::DT_INT8 && !matmulInfoPtr_->transB) {
        GetBubTilingA16W8NoTrans(nBubSize, kBubSize);
    } else if (matmulInfoPtr_->bDtype == ge::DT_INT4 && matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        GetBubTilingA16W4NZ(nBubSize, kBubSize);
    } else if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        GetBubTilingA16W4ND(nBubSize, kBubSize);
    }
    tilingData_->set_nBubSize(nBubSize);
    tilingData_->set_kBubSize(kBubSize);
}

void WeightQuantBatchMatmulV2RegBase::SetMatmulTiling()
{
    const BasicBlockParam &tilingRes = tilingSolver_.GetTilingResult();
    tilingData_->set_cubeBlockDimM(static_cast<uint8_t>(tilingRes.mDim));
    tilingData_->set_cubeBlockDimN(static_cast<uint8_t>(tilingRes.nDim));

    tilingData_->matmulTiling.set_M(tilingRes.mSize);
    tilingData_->matmulTiling.set_Ka(tilingRes.kSize);
    tilingData_->matmulTiling.set_N(tilingRes.nSize);
    tilingData_->matmulTiling.set_Kb(tilingRes.kSize);
    tilingData_->matmulTiling.set_singleCoreM(tilingRes.l1Param.stepM * tilingRes.basicBlock.baseM);
    tilingData_->matmulTiling.set_singleCoreK(tilingRes.l1Param.stepKa * tilingRes.basicBlock.baseK);
    tilingData_->matmulTiling.set_singleCoreN(tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN);

    tilingData_->matmulTiling.set_baseM(tilingRes.basicBlock.baseM);
    tilingData_->matmulTiling.set_baseN(tilingRes.basicBlock.baseN);
    tilingData_->matmulTiling.set_baseK(tilingRes.basicBlock.baseK);
    tilingData_->matmulTiling.set_dbL0A(DB_BUFFER);
    tilingData_->matmulTiling.set_dbL0B(DB_BUFFER);
    int32_t dbL0C = tilingRes.basicBlock.baseM * tilingRes.basicBlock.baseN * sizeof(float) * DB_BUFFER <= L0C_SIZE_V100
                        ? DB_BUFFER
                        : 1;
    tilingData_->matmulTiling.set_dbL0C(dbL0C);

    tilingData_->matmulTiling.set_stepM(tilingRes.l1Param.stepM);
    tilingData_->matmulTiling.set_stepN(tilingRes.l1Param.stepN);
    tilingData_->matmulTiling.set_stepKa(tilingRes.l1Param.stepKa);
    tilingData_->matmulTiling.set_stepKb(tilingRes.l1Param.stepKb);
    tilingData_->matmulTiling.set_depthA1(tilingRes.l1Param.A1BufferNum * tilingRes.l1Param.stepM *
                                          tilingRes.l1Param.stepKa);
    tilingData_->matmulTiling.set_depthB1(tilingRes.l1Param.B1BufferNum * tilingRes.l1Param.stepN *
                                          tilingRes.l1Param.stepKb);
    tilingData_->matmulTiling.set_iterateOrder(tilingRes.l1Param.iterateOrder);

    tilingData_->matmulTiling.set_isBias(static_cast<int32_t>(matmulInfoPtr_->hasBias));
    tilingData_->matmulTiling.set_shareL1Size(0);
    if (matmulInfoPtr_->hasBias) {
        tilingData_->matmulTiling.set_shareL1Size(tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN *
                                                  GetSizeByDataType(matmulInfoPtr_->biasDtype));
    }
    tilingData_->matmulTiling.set_shareL0CSize(0);

    tilingData_->set_AL1Pingpong(tilingRes.l1Param.A1BufferNum);
    tilingData_->set_BL1Pingpong(tilingRes.l1Param.B1BufferNum);
    tilingData_->set_groupSize(matmulInfoPtr_->groupSize);
}

uint64_t WeightQuantBatchMatmulV2RegBase::GetGroupNumBub(uint64_t kDimSzie) const
{
    if (kDimSzie == 0) {
        return 0;
    } else if (matmulInfoPtr_->groupSize == 0) {
        return 1;
    } else if (kDimSzie % matmulInfoPtr_->groupSize == 0) {
        return kDimSzie / matmulInfoPtr_->groupSize;
    } else if (matmulInfoPtr_->groupSize % kDimSzie == 0) {
        return 1;
    } else if (kDimSzie > matmulInfoPtr_->groupSize) {
        return std::min(ops::CeilDiv(matmulInfoPtr_->kSize, matmulInfoPtr_->groupSize),
                        kDimSzie / matmulInfoPtr_->groupSize + EXTRA_GROUP_NUM);
    } else {
        return EXTRA_GROUP_NUM;
    }
}

ge::graphStatus WeightQuantBatchMatmulV2RegBase::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        tilingData_ = std::unique_ptr<WeightQuantBatchMatmulV2RegBaseTilingData>(
            new (std::nothrow) WeightQuantBatchMatmulV2RegBaseTilingData());
    }
    OP_TILING_CHECK(tilingData_ == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingData_->GetDataSize(),
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                                        context_->GetRawTilingData()->GetCapacity(), tilingData_->GetDataSize()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint64_t WeightQuantBatchMatmulV2RegBase::GetBubSize(uint64_t orgNdim, uint64_t orgDdim, bool isWeightNz) const
{
    uint64_t kDimSzie = (matmulInfoPtr_->transB || isWeightNz) ? orgDdim : orgNdim;
    uint64_t nDimSzie = (matmulInfoPtr_->transB || isWeightNz) ? orgNdim : orgDdim;
    uint64_t sizeScale = DB_BUFFER * GetSizeByDataType(ge::DT_FLOAT16) * nDimSzie;
    uint64_t sizeWeightOut;
    uint64_t sizeWeightIn = DB_BUFFER * GetSizeByDataType(ge::DT_INT8) * kDimSzie * nDimSzie;
    if (matmulInfoPtr_->transB) {
        uint64_t blockSize = (uint64_t)ONE_BLK_SIZE / GetSizeByDataType(ge::DT_FLOAT16);
        sizeWeightOut = DB_BUFFER * GetSizeByDataType(ge::DT_FLOAT16) * kDimSzie * (nDimSzie + 1);
        sizeScale = sizeScale * ops::CeilAlign(GetGroupNumBub(kDimSzie), blockSize);
        if (matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL) {
            sizeScale = DB_BUFFER * GetSizeByDataType(ge::DT_FLOAT16) * ops::CeilAlign(nDimSzie, blockSize);
        }
    } else {
        sizeWeightOut = DB_BUFFER * GetSizeByDataType(ge::DT_FLOAT16) * nDimSzie * (kDimSzie + 1);
        if (matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP) {
            sizeWeightOut = DB_BUFFER * GetSizeByDataType(ge::DT_FLOAT16) * nDimSzie * kDimSzie;
        }
        sizeScale = sizeScale * GetGroupNumBub(kDimSzie);
    }

    if (matmulInfoPtr_->antiQuantType == QuantType::PER_TENSOR) {
        sizeScale = 0;
    }
    uint64_t sizeOffset = matmulInfoPtr_->hasAntiQuantOffset ? sizeScale : 0;
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        sizeWeightIn = sizeWeightIn / INT4_DTYPE_PARAM;
    }

    return (sizeWeightIn + sizeWeightOut + sizeScale + sizeOffset);
}

void WeightQuantBatchMatmulV2RegBase::GetBubTilingA16W8NDPerGroup(int64_t &nBubSize, int64_t &kBubSize) const
{
    const BasicBlockParam &tilingRes = tilingSolver_.GetTilingResult();
    int64_t nBl1Size = std::min(tilingRes.singleN, tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN);
    int64_t kBl1Size = std::min(tilingRes.singleK, tilingRes.l1Param.stepKb * tilingRes.basicBlock.baseK);
    if (matmulInfoPtr_->transB) {
        nBubSize = ops::CeilDiv(nBl1Size, BUFF_NUM_2);
        kBubSize = ops::CeilAlign(kBl1Size, static_cast<int64_t>(GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype)));
    } else {
        kBubSize = ops::CeilDiv(kBl1Size, BUFF_NUM_2);
        nBubSize = ops::CeilAlign(nBl1Size, static_cast<int64_t>(GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype)));
    }
}

void WeightQuantBatchMatmulV2RegBase::GetBubTilingA16W8NoTrans(int64_t &nBubSize, int64_t &kBubSize) const
{
    const BasicBlockParam &tilingRes = tilingSolver_.GetTilingResult();
    int64_t kBl1Size = std::min(tilingRes.singleK, tilingRes.l1Param.stepKb * tilingRes.basicBlock.baseK);
    int64_t nBl1Size = std::min(tilingRes.singleN, tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN);
    // (1) 优先切外轴K (2)K方向不能再切分时,保证切n后的nbubsize 32对齐 (3) 不切分
    if (kBl1Size > BLOCK_CUBE) {
        kBubSize = ops::CeilAlign(ops::CeilDiv(kBl1Size, BUFF_NUM_2), BLOCK_CUBE);
        nBubSize = nBl1Size;
        //kBl1 = 16不能再分，且nBl1为64整数倍时可以切n，保证nbub是32整数倍
    } else if (nBl1Size % 64 == 0) {
        nBubSize = ops::CeilAlign(ops::CeilDiv(nBl1Size, BUFF_NUM_2), BLOCK_CUBE);
        kBubSize = kBl1Size;
    } else {
        nBubSize = nBl1Size;
        kBubSize = kBl1Size;
    }
}

void WeightQuantBatchMatmulV2RegBase::GetBubTilingA16W4NZ(int64_t &nBubSize, int64_t &kBubSize) const
{
    const BasicBlockParam &tilingRes = tilingSolver_.GetTilingResult();
    int64_t kBl1Size = std::min(tilingRes.singleK, tilingRes.l1Param.stepKb * tilingRes.basicBlock.baseK);
    int64_t nBl1Size = std::min(tilingRes.singleN, tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN);
    if (nBl1Size > BLOCK_CUBE) {
        nBubSize = ops::CeilAlign(ops::CeilDiv(nBl1Size, BUFF_NUM_2), BLOCK_CUBE);
        kBubSize = kBl1Size;
    } else {
        nBubSize = nBl1Size;
        if (matmulInfoPtr_->groupSize > 0 && kBl1Size > static_cast<int64_t>(matmulInfoPtr_->groupSize)) {
            kBubSize = ops::CeilDiv(static_cast<int64_t>(kBl1Size / matmulInfoPtr_->groupSize), BUFF_NUM_2) *
                       matmulInfoPtr_->groupSize;
        } else {
            kBubSize = ops::CeilDiv(kBl1Size / GROUP_ALIGN_SIZE, BUFF_NUM_2) * GROUP_ALIGN_SIZE;
        }
    }
}

void WeightQuantBatchMatmulV2RegBase::GetBubTilingA16W4ND(int64_t &nBubSize, int64_t &kBubSize) const
{
    const BasicBlockParam &tilingRes = tilingSolver_.GetTilingResult();
    int64_t kBl1Size = std::min(tilingRes.singleK, tilingRes.l1Param.stepKb * tilingRes.basicBlock.baseK);
    int64_t nBl1Size = std::min(tilingRes.singleN, tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN);
    if (matmulInfoPtr_->transB) {
        nBubSize = ops::CeilDiv(nBl1Size, BUFF_NUM_2);
        kBubSize = ops::CeilAlign(kBl1Size, static_cast<int64_t>(GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype)));
        if (matmulInfoPtr_->groupSize > GROUP_SIZE_64 && matmulInfoPtr_->groupSize % GROUP_SIZE_64 > 0 &&
            kBubSize > static_cast<int64_t>(matmulInfoPtr_->groupSize)) {
            // 96含义：在NK且gs非64对齐场景，跨gs计算的长度为96，因此至少保证内轴长度大等于gs+96
            kBubSize = std::max(kBubSize, static_cast<int64_t>(matmulInfoPtr_->groupSize + 96));
        }
    } else {
        nBubSize = ops::CeilAlign(nBl1Size, static_cast<int64_t>(GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype)));
        kBubSize = ops::CeilDiv(kBl1Size, BUFF_NUM_2);
    }
}

void WeightQuantBatchMatmulV2RegBase::GetBubTilingA16W8Trans(int64_t &nBubSize, int64_t &kBubSize) const
{
    const BasicBlockParam &tilingRes = tilingSolver_.GetTilingResult();
    int64_t kBl1Size = std::min(tilingRes.singleK, tilingRes.l1Param.stepKb * tilingRes.basicBlock.baseK);
    int64_t nBl1Size = std::min(tilingRes.singleN, tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN);
    // 若BL1不足最小载入量，则无需切分
    if (nBl1Size * kBl1Size <= MTE2_MIN_LOAD_SIZE_V100 &&
        GetBubSize(nBl1Size, kBl1Size, false) <= compileInfoPtr_->ubSize) {
        nBubSize = nBl1Size;
        kBubSize = kBl1Size;
        OPS_LOG_D(opName_, "No need to split BL1, set nBubSize: %ld, kBubSize: %ld", nBubSize, kBubSize);
        return;
    }

    // default解
    kBubSize = kBl1Size;
    nBubSize = ops::CeilAlign(ops::CeilDiv(MTE2_MIN_LOAD_SIZE_V100, kBubSize), BLOCK_CUBE);
    nBubSize = std::min(nBubSize, nBl1Size);
    nBubSize = std::max(BLOCK_CUBE, nBubSize);
    // 若kbl1较大（如大于5300时），通过MTE2_MIN_LOAD_SIZE_V100反算出的nbub可能小于16，导致bub放大到16后超UB，
    // 此时无法保证内轴全载，仅保证最小载入量。
    if (GetBubSize(nBubSize, kBubSize, false) > compileInfoPtr_->ubSize) {
        nBubSize = BLOCK_CUBE;
        kBubSize = std::min(MTE2_MIN_LOAD_SIZE_V100 / BLOCK_CUBE, kBl1Size);
    }

    if (kBl1Size % CACHE_LINE_V100 > 0) {
        OPS_LOG_D(opName_, "kBl1 is not %ld aligned, use default UB tiling. nBubSize: %ld, kBubSize: %ld",
                CACHE_LINE_V100, nBubSize, kBubSize);
        return;
    }

    for (int64_t tmpKBub = kBl1Size; tmpKBub >= CACHE_LINE_V100; tmpKBub -= CACHE_LINE_V100) {
        for (int64_t tmpNBub = BLOCK_CUBE; tmpNBub <= nBl1Size; tmpNBub += BLOCK_CUBE) {
            // 前提条件：满足最小载入量且不超UB
            if (GetBubSize(tmpNBub, tmpKBub, false) > compileInfoPtr_->ubSize ||
                tmpNBub * tmpKBub < MTE2_MIN_LOAD_SIZE_V100) {
                continue;
            }
            // 优先选择n、k方向都整除的特解
            if (kBl1Size % tmpKBub == 0 && nBl1Size % tmpNBub == 0) {
                nBubSize = tmpNBub;
                kBubSize = tmpKBub;
                OPS_LOG_D(opName_, "Find ideal UB tiling, nBubSize: %ld, kBubSize: %ld", nBubSize, kBubSize);
                return;
            }
        }
    }
    OPS_LOG_D(opName_, "Use default UB tiling. nBubSize: %ld, kBubSize: %ld", nBubSize, kBubSize);
}

void WeightQuantBatchMatmulV2RegBase::SetAdditionalParam()
{
    const BasicBlockParam &tilingRes = tilingSolver_.GetTilingResult();
    tilingData_->set_vecCoreParallel(0);
    if (tilingRes.l1Param.B1BufferNum == 1 &&
        ops::CeilDiv(
            static_cast<uint64_t>(std::min(tilingRes.singleK, tilingRes.l1Param.stepKb * tilingRes.basicBlock.baseK)),
            tilingData_->get_kBubSize()) == DB_BUFFER &&
        tilingData_->get_nBubSize() ==
            static_cast<uint64_t>(std::min(tilingRes.singleN, tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN))) {
        OPS_LOG_D(opName_,
                "Set vecCoreParallel to 1, nBubSize: %lu, singleN: %ld, kBubSize: %lu, singleK: %ld",
                tilingData_->get_nBubSize(), tilingRes.singleN, tilingData_->get_kBubSize(), tilingRes.singleK);
        tilingData_->set_vecCoreParallel(1);
    }
}

void WeightQuantBatchMatmulV2RegBase::PrintCVTilingData(bool debugLevel) const
{
    if (debugLevel && AlogCheckDebugLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }

    std::stringstream ss;
    ss << " kSize: " << tilingData_->get_kSize() << " groupSize: " << tilingData_->get_groupSize()
       << " nSize: " << tilingData_->get_nSize() << " mSize: " << tilingData_->get_mSize()
       << " cubeBlockDimN: " << static_cast<uint32_t>(tilingData_->get_cubeBlockDimN())
       << " cubeBlockDimM: " << static_cast<uint32_t>(tilingData_->get_cubeBlockDimM())
       << " vecCoreParallel: " << static_cast<uint32_t>(tilingData_->get_vecCoreParallel())
       << " nBubSize: " << tilingData_->get_nBubSize() << " kBubSize: " << tilingData_->get_kBubSize()
       << " AL1Pingpong: " << tilingData_->get_AL1Pingpong() << " BL1Pingpong: " << tilingData_->get_BL1Pingpong();
    int32_t logLevel = debugLevel ? DLOG_DEBUG : DLOG_ERROR;
}
}  // namespace optiling
