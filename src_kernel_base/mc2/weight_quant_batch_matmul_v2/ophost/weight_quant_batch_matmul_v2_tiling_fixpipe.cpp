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
 * \file weight_quant_batch_matmul_v2_tiling_fixpipe.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_tiling_fixpipe.h"

#include "weight_quant_batch_matmul_v2_tiling_key.h"

namespace optiling {

constexpr uint64_t INT8_BLOCK_CUBE_TRANSPOSE = 32UL;

ge::graphStatus WeightQuantBatchMatmulV2TilingFixpipe::PostTiling()
{
    OPS_LOG_D(opName_, "final tiling data size: %zu", tilingData_->GetDataSize());

    OP_TILING_CHECK(
        tilingData_->GetDataSize() % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data size[%zu] not aligned to 8", tilingData_->GetDataSize()),
        return ge::GRAPH_FAILED);

    context_->GetRawTilingData()->SetDataSize(tilingData_->GetDataSize());
    // 计算aic num n方向分核*m方向分核
    context_->SetBlockDim(tilingData_->get_nBlockNum() *
                          ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->get_singleCoreM())));
    tilingData_->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2TilingFixpipe::IsCapable()
{
    OPS_LOG_D(opName_,
            "begin to detect the Fixpipe template limit. MKN[%lu, %lu, %lu], "
            "groupSize_: [%lu]",
            matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize, matmulInfoPtr_->groupSize);

    OP_TILING_CHECK(!CheckDtypeIsCapable(),
                    OPS_LOG_D(opName_,
                            "check mkn finish, the Fixpipe template doesn't "
                            "support current shape."),
                    return false);

    OP_TILING_CHECK(!CheckShapeIsCapable(),
                    OPS_LOG_D(opName_,
                            "check mkn finish, the Fixpipe template doesn't "
                            "support current shape."),
                    return false);
    return true;
}

bool WeightQuantBatchMatmulV2TilingFixpipe::CheckDtypeIsCapable() const
{
    // 仅支持输出fp16
    OP_TILING_CHECK(matmulInfoPtr_->cDtype != ge::DT_FLOAT16,
                    OPS_LOG_D(opName_, "the Fixpipe template only support cDtype is FP16, current is [%s].",
                            ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->cDtype).GetString()),
                    return false);

    // 只支持W8场景
    OP_TILING_CHECK(matmulInfoPtr_->bDtype == ge::DT_INT4,
                    OPS_LOG_D(opName_,
                            "the Fixpipe template only support bDtype is int8, "
                            "current is [int4]."),
                    return false);

    // 仅支持antiquantScale类型是uint64_t/int64_t
    OP_TILING_CHECK(((matmulInfoPtr_->antiQuantScaleDtype != ge::DT_UINT64) &&
                     (matmulInfoPtr_->antiQuantScaleDtype != ge::DT_INT64)),
                    OPS_LOG_D(opName_,
                            "the Fixpipe template only support antiquantScaleDtype is uint64, "
                            "current is [%s].",
                            ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->antiQuantScaleDtype).GetString()),
                    return false);
    return true;
}

bool WeightQuantBatchMatmulV2TilingFixpipe::CheckShapeIsCapable() const
{
    // 仅支持n轴\k轴都是64的倍数
    OP_TILING_CHECK(matmulInfoPtr_->nSize % 64 != 0 || matmulInfoPtr_->kSize % 64 != 0,
                    OPS_LOG_D(opName_,
                            "the Fixpipe template only support n aligned to 64 "
                            "and k aligned to 64."),
                    return false);

    // 仅支持m轴是在1-96的范围
    OP_TILING_CHECK(matmulInfoPtr_->mSize > 96,
                    OPS_LOG_D(opName_, "the Fixpipe template only support mSize_ in range [1, 96]."), return false);

    // 仅支持b转置场景
    OP_TILING_CHECK(!matmulInfoPtr_->transB,
                    OPS_LOG_D(opName_,
                            "the Fixpipe template only support weight is "
                            "transposed, current transB : [%s].",
                            matmulInfoPtr_->transB ? "true" : "false"),
                    return false);

    // 仅支持a不转置场景
    OP_TILING_CHECK(matmulInfoPtr_->transA,
                    OPS_LOG_D(opName_,
                            "the Fixpipe template only support x is not "
                            "transposed, current transA : [%s].",
                            matmulInfoPtr_->transA ? "true" : "false"),
                    return false);

    // 只支持perchannel场景
    OP_TILING_CHECK(matmulInfoPtr_->antiQuantType != QuantType::PER_CHANNEL,
                    OPS_LOG_I(opName_, "the Fixpipe template only support per channel."), return false);
    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingFixpipe::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        tilingData_ = std::unique_ptr<WeightQuantBatchMatmulV2FixpipeTilingData>(
            new (std::nothrow) WeightQuantBatchMatmulV2FixpipeTilingData());
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

ge::graphStatus WeightQuantBatchMatmulV2TilingFixpipe::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "unable to get pointer of tiling data"),
                    return ge::GRAPH_FAILED);

    tilingData_->set_hasBias(matmulInfoPtr_->hasBias);
    // 保证kernel的数据32对齐，避免为了处理16的尾块而引入其他计算
    uint64_t singleCoreN = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(compileInfoPtr_->aicNum)), 32UL);
    uint64_t baseN = std::min(singleCoreN, 128UL);
    if (matmulInfoPtr_->nSize <= compileInfoPtr_->aicNum * BLOCK_CUBE) {
        // 极端场景，n方向按照16的粒度依旧可以分不满，降低n方向切分粒度
        singleCoreN = BLOCK_CUBE;
        baseN = INT8_BLOCK_CUBE_TRANSPOSE;
    }
    uint64_t nBlkNum = ops::CeilDiv(matmulInfoPtr_->nSize, singleCoreN);
    uint64_t mBlkNum = compileInfoPtr_->aicNum / nBlkNum;

    // fixp方案切分的基本块是baseK = 512，
    // 此处根据实际k值缩小基本块的k，防止mmad出错
    uint64_t baseK = matmulInfoPtr_->kSize > 512 ? 512 : matmulInfoPtr_->kSize;
    uint64_t singleCoreM = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(mBlkNum)),
                                          static_cast<uint64_t>(BLOCK_CUBE));

    // fixp基本块切分后，a的最大剩余空间是250 * 1024 byte
    uint64_t aL1MaxSize = 250 * 1024;
    if (singleCoreM * matmulInfoPtr_->kSize * sizeof(matmulInfoPtr_->aDtype) > aL1MaxSize) {
        aFullLoad_ = 0;
    } else {
        aFullLoad_ = 1;
    }
    tilingData_->set_nBlockNum(nBlkNum);
    tilingData_->set_baseK(baseK);
    tilingData_->set_baseM(singleCoreM);
    tilingData_->set_baseN(baseN);
    tilingData_->set_singleCoreM(std::min(matmulInfoPtr_->mSize, singleCoreM));  // 核内不切m
    tilingData_->set_singleCoreN(singleCoreN);
    tilingData_->set_mSize(matmulInfoPtr_->mSize);
    tilingData_->set_kSize(matmulInfoPtr_->kSize);
    tilingData_->set_nSize(matmulInfoPtr_->nSize);
    return ge::GRAPH_SUCCESS;
}

// 5、计算TilingKey
uint64_t WeightQuantBatchMatmulV2TilingFixpipe::GetTilingKey() const
{
    TilingKeyConfigure tilingKeyConfigure;
    // 平台类型占2位(平台大类， 平台小类)，平台大类在高位，需要乘10
    tilingKeyConfigure.socVersionType = static_cast<uint8_t>(SocVersionType::SUPPORT_L0C_TO_OUT) * 10;
    tilingKeyConfigure.quantizationScenario = static_cast<uint8_t>(QuantizationScenario::DEFAULT);
    // 算法类型占2位(算法大类，算法小类)，算法大类在高位，需要乘10
    tilingKeyConfigure.algorithm = static_cast<uint8_t>(OptimizationAlgorithmCategory::FIXPIPE_ANTIQUANT) * 10;
    tilingKeyConfigure.transposeSituation =
        (static_cast<uint16_t>(matmulInfoPtr_->transA) << 1) | static_cast<uint16_t>(matmulInfoPtr_->transB);
    tilingKeyConfigure.antiquantType = static_cast<uint8_t>(matmulInfoPtr_->antiQuantType);
    tilingKeyConfigure.quantType = static_cast<uint8_t>(QuantType::NONE);
    tilingKeyConfigure.optionInputSituation = ((static_cast<uint16_t>(matmulInfoPtr_->hasAntiQuantOffset) << 1) |
                                               static_cast<uint16_t>(matmulInfoPtr_->hasBias));
    tilingKeyConfigure.weightFormat = static_cast<uint8_t>(WeightFormat::ND);

    tilingKeyConfigure.templateCustom = static_cast<uint8_t>(
        aFullLoad_ ? FixpipeConfiguration::A_SINGLE_M_SINGLE_K_FULL_LOAD : FixpipeConfiguration::A_NORMAL_LOAD);
    tilingKeyConfigure.apiConstexpr = 0;
    return tilingKeyConfigure.GenTilingKey();
}

// 6、计算Workspace 大小
ge::graphStatus WeightQuantBatchMatmulV2TilingFixpipe::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = compileInfoPtr_->workspaceNum;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
