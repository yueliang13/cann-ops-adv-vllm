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
 * \file weight_quant_batch_matmul_v2_compute_matmul_tiling.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_compute_matmul_tiling.h"

namespace optiling {

constexpr uint64_t DOUBLE_BUFFER_FACTOR = 2UL;  // 1: off, 2: on
constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr uint64_t INT4_BLK_SIZE = 64UL;
constexpr uint64_t ALIGN_32 = 32UL;
constexpr int32_t BASEN_128 = 128;
constexpr int32_t BASEK_256 = 256;
constexpr int32_t DEPTHB1_4 = 4;
constexpr int32_t STEPKB_2 = 2;

bool ComputeMatmulTiling::GetTiling(TCubeTiling &matmulTiling, MatmulMultiCoreResult &multiCoreResult,
                                    const MatmulParams &params, const AiCoreParams &aicoreParams,
                                    gert::TilingContext *context)
{
    bool getTilingResult = false;
    if (params.aDtype == ge::DT_INT4 && params.bDtype == ge::DT_INT4) {
        getTilingResult = SimpleIncreTiling(matmulTiling, multiCoreResult, params, aicoreParams);
    }

    if (params.aDtype == ge::DT_INT8 && params.bDtype == ge::DT_INT8) {
        getTilingResult = MsdA16W8CommonTiling(matmulTiling, multiCoreResult, params, aicoreParams);
    }

    if (!getTilingResult) {
        OPS_LOG_W("WeightQuantBatchMatmulV2",
                "can not get tiling from SimpleIncreTiling/MsdA16W8CommonTiling. try to get tiling from cache tiling");
        return GetCacheTiling(matmulTiling, multiCoreResult, params, context);
    }
    return getTilingResult;
}

bool ComputeMatmulTiling::GetCacheTiling(TCubeTiling &matmulTiling, MatmulMultiCoreResult &multiCoreResult,
                                         const MatmulParams &params, gert::TilingContext *context)
{
    int32_t reduceSize = static_cast<int32_t>(GetBlockAlignSizeByDataType(params.aDtype));
    bool weightNz = params.format_b == ge::FORMAT_FRACTAL_NZ;
    BatchmatmulCompileParas compileParams;
    compileParams.binary_mode_flag = true;
    compileParams.bias_flag = params.hasBias;
    compileParams.pattern_flag = true;
    compileParams.zero_flag = false;

    bool alignedMKN =
        params.mSize % BLOCK_CUBE == 0 && params.kSize % reduceSize == 0 && params.nSize % BLOCK_CUBE == 0;
    BatchmatmulRunParas runParams;
    runParams.use_pre_ub = false;
    runParams.trans_a_flag = params.transA;
    runParams.trans_b_flag = params.transB;
    runParams.format_a_nd = true;
    runParams.format_b_nd = !weightNz;
    runParams.format_out_nd = true;
    runParams.format_a = params.format_a;
    runParams.format_b = params.format_b;
    runParams.format_out = params.format_out;
    runParams.reserved_bool = (params.quantType > QuantType::PER_TENSOR);
    runParams.nd_flag = runParams.format_a_nd && runParams.format_b_nd;
    runParams.used_aligned_pattern = alignedMKN && runParams.nd_flag;
    runParams.bias_flag = params.hasBias;
    runParams.pattern_flag = compileParams.pattern_flag;
    runParams.unaligned_flag = !alignedMKN;
    runParams.zero_flag = compileParams.zero_flag;
    runParams.weight_nz_flag = weightNz;
    runParams.hf32_flag = 0;
    runParams.dtype_a = static_cast<int32_t>(params.aDtype);
    runParams.dtype_b = runParams.dtype_a;
    runParams.dtype_out = runParams.dtype_a;
    runParams.dtype_bias = ge::GetSizeByDataType(params.biasDtype);
    runParams.m = ops::CeilDiv(params.mSize, static_cast<uint64_t>(BLOCK_CUBE));
    runParams.k = ops::CeilDiv(params.kSize, static_cast<uint64_t>(reduceSize));
    runParams.n = ops::CeilDiv(params.nSize, static_cast<uint64_t>(BLOCK_CUBE));
    runParams.ori_shape_m = params.mSize;
    runParams.ori_shape_k = params.kSize;
    runParams.ori_shape_n = params.nSize;
    runParams.m_quant_check = params.transA && params.cDtype == ge::DT_INT8;
    runParams.n_quant_check = !params.transB;
    runParams.bias_dtype = params.biasDtype;
    runParams.vector_pre_conv_mode = params.cDtype == ge::DT_INT8;
    runParams.is_weight_quant_batch_matmul_v2 = true;
    Tiling tiling;
    tiling.tiling_id = std::numeric_limits<uint64_t>::max();
    bool ret = GenTiling("WeightQuantBatchMatmulV2", compileParams, runParams, tiling, context);
    if (tiling.tiling_id == std::numeric_limits<uint64_t>::max() || !ret) {
        return false;
    }
    Convert2AscendCTiling(tiling, matmulTiling, params, multiCoreResult);
    return true;
}

bool ComputeMatmulTiling::MsdA16W8CommonTiling(TCubeTiling &matmulTiling, MatmulMultiCoreResult &multiCoreResult,
                                               const MatmulParams &params, const AiCoreParams &aicoreParams)
{
    // weight为NZ-NK/ND-KN暂不支持
    if (((params.transB) && (params.format_b == ge::FORMAT_FRACTAL_NZ)) ||
        ((!params.transB) && (params.format_b == ge::FORMAT_ND))) {
        OPS_LOG_W("WeightQuantBatchMatmulV2",
                "MsdA16W8CommonTiling only support b is transposed and b is ND, "
                "or b is not transposed and b is NZ, current transB:[%s], formatB:[%s]",
                params.transB ? "true" : "false",
                (params.format_b == ge::FORMAT_ND) ? "FORMAT_ND" : "FORMAT_FRACTAL_NZ");
        return false;
    }

    uint64_t singleCoreN = ops::CeilAlign(ops::FloorDiv(params.nSize, static_cast<uint64_t>(aicoreParams.aicNum)),
                                          ALIGN_32);
    matmulTiling.set_singleCoreN(singleCoreN);

    // 根据A是否全载L1计算baseN，baseK，depthA1，depthB1，stepKa，stepKb
    CalcCommonTiling(matmulTiling, params, aicoreParams);

    matmulTiling.set_M(params.mSize);
    matmulTiling.set_N(params.nSize);
    matmulTiling.set_Ka(params.kSize);
    matmulTiling.set_Kb(params.kSize);
    matmulTiling.set_singleCoreM(params.mSize);
    matmulTiling.set_singleCoreK(params.kSize);
    matmulTiling.set_stepM(1);
    matmulTiling.set_stepN(1);
    matmulTiling.set_isBias(0);
    matmulTiling.set_iterateOrder(0);

    // 计算MSD场景transLen，L1Size和L0CSize
    CalcMsdBufferSize(matmulTiling, params);

    matmulTiling.set_dbL0A(DOUBLE_BUFFER_FACTOR);
    matmulTiling.set_dbL0B(DOUBLE_BUFFER_FACTOR);
    matmulTiling.set_dbL0C(1);
    matmulTiling.set_usedCoreNum(1);
    matmulTiling.set_batchM(1);
    matmulTiling.set_batchN(1);
    matmulTiling.set_singleBatchM(1);
    matmulTiling.set_singleBatchN(1);
    multiCoreResult.mDim = 1;
    multiCoreResult.nDim = std::min(
        aicoreParams.aicNum, ops::CeilDiv(params.nSize, static_cast<uint64_t>(matmulTiling.get_singleCoreN())));
    multiCoreResult.batchDim = 1;
    return true;
}

void ComputeMatmulTiling::CalcMsdBufferSize(TCubeTiling &matmulTiling, const MatmulParams &params)
{
    int32_t a1Length = static_cast<int32_t>(
        GetShapeSizeWithDataType(matmulTiling.get_baseM() * matmulTiling.get_baseK(), params.aDtype));
    int32_t b1Length = static_cast<int32_t>(
        GetShapeSizeWithDataType(matmulTiling.get_baseN() * matmulTiling.get_baseK(), params.bDtype));
    int32_t c1Length = matmulTiling.get_baseN() * matmulTiling.get_baseM() * sizeof(float);
    int32_t aL1Size = a1Length * matmulTiling.get_depthA1();
    int32_t bL1Size = b1Length * matmulTiling.get_depthB1();
    matmulTiling.set_transLength(std::max(std::max(a1Length, b1Length), c1Length));
    matmulTiling.set_shareL1Size(aL1Size + bL1Size);
    matmulTiling.set_shareL0CSize(c1Length);
}

void ComputeMatmulTiling::CalcCommonTiling(TCubeTiling &matmulTiling, const MatmulParams &params,
                                           const AiCoreParams &aicoreParams)
{
    uint64_t baseM = ops::CeilAlign(params.mSize, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t baseK = 128;
    // preload阶段要求baseN <= singleCoreN
    uint64_t baseN = std::min(256, matmulTiling.get_singleCoreN());
    uint64_t depthA1 = 8;
    uint64_t depthB1 = 8;
    uint64_t stepKa = 4;
    uint64_t stepKb = 4;
    uint64_t matASizeBound256 = 256 * 1024;
    uint64_t matASizeBound384 = 384 * 1024;
    uint64_t matASize = params.mSize * params.kSize;
    bool isAFullLoad = false;
    // A矩阵<=384KB时，考虑全载
    if (matASize <= matASizeBound384) {
        uint64_t depthA1Tmp = ops::CeilDiv(params.kSize, static_cast<uint64_t>(baseK));
        // A矩阵<=256KB时，A在L1全载, 由于depthA1做了关于baseK的上对齐，需要校验是否超L1 size
        if (matASize <=  matASizeBound256) {
            uint64_t l1SizeTmp = baseM * baseK * depthA1Tmp + baseN * baseK * depthB1;
            if (l1SizeTmp <= aicoreParams.l1Size) {
                isAFullLoad = true;
            }
        }
        // A矩阵(256KB, 384KB]时，减小stepKb至2，A在L1全载
        if ((matASize > matASizeBound256) || (!isAFullLoad)) {
            // 由于depthA1做了关于baseK的上对齐，需要校验是否超L1 size, baseK取256，baseN取128，depthB1取4
            uint64_t l1SizeTmp = baseM * 256 * depthA1Tmp + 128 * 256 * 4;
            if ( l1SizeTmp <= aicoreParams.l1Size) {
                // preload阶段要求baseN <= singleCoreN
                baseN = std::min(BASEN_128,  matmulTiling.get_singleCoreN());
                baseK = BASEK_256;
                depthB1 = DEPTHB1_4;
                stepKb = STEPKB_2;
                isAFullLoad = true;
            }
        }
        // A可以全载，设置depthA1和stepKa
        if (isAFullLoad) {
            depthA1 = depthA1Tmp;
            stepKa = depthA1Tmp;
        }
    }
    matmulTiling.set_baseM(baseM);
    matmulTiling.set_baseN(baseN);
    matmulTiling.set_baseK(baseK);
    matmulTiling.set_depthA1(depthA1);
    matmulTiling.set_depthB1(depthB1);
    matmulTiling.set_stepKa(stepKa);
    matmulTiling.set_stepKb(stepKb);
}

bool ComputeMatmulTiling::SimpleIncreTiling(TCubeTiling &matmulTiling, MatmulMultiCoreResult &multiCoreResult,
                                            const MatmulParams &params, const AiCoreParams &aicoreParams)
{
    if (params.aDtype != ge::DT_INT4 || params.bDtype != ge::DT_INT4) {
        OPS_LOG_W("WeightQuantBatchMatmulV2",
                "SimpleIncreTiling only support int4 matmul case. cur a dtype: [%s], b dtype: [%s]",
                ge::TypeUtils::DataTypeToAscendString(params.aDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(params.bDtype).GetString());
        return false;
    }

    if (params.transA || !params.transB) {
        OPS_LOG_W("WeightQuantBatchMatmulV2",
                "Int4IncreTiling only support a is not transposed, and b is transposed. transA:[%s], transB: [%s]",
                params.transA ? "true" : "false", params.transB ? "true" : "false");
        return false;
    }

    if (tryComputeSimpleTiling(matmulTiling, params, aicoreParams)) {
        // tiling的通用设置
        matmulTiling.set_usedCoreNum(1);
        matmulTiling.set_M(params.mSize);
        matmulTiling.set_N(params.nSize);
        matmulTiling.set_Ka(params.kSize);
        matmulTiling.set_Kb(params.kSize);
        matmulTiling.set_singleCoreM(params.mSize);
        matmulTiling.set_singleCoreK(params.kSize);
        matmulTiling.set_stepM(1);
        matmulTiling.set_isBias(0);
        matmulTiling.set_shareMode(0);
        matmulTiling.set_dbL0A(DOUBLE_BUFFER_FACTOR);  // db默认打开
        matmulTiling.set_dbL0B(DOUBLE_BUFFER_FACTOR);  // db默认打开
        matmulTiling.set_shareL1Size(0);
        matmulTiling.set_shareUbSize(0);
        matmulTiling.set_batchM(1);
        matmulTiling.set_batchN(1);
        matmulTiling.set_singleBatchM(1);
        matmulTiling.set_singleBatchN(1);

        multiCoreResult.mDim = 1;
        multiCoreResult.nDim = std::min(
            aicoreParams.aicNum, ops::CeilAlign(params.nSize, static_cast<uint64_t>(matmulTiling.get_singleCoreN())));
        multiCoreResult.batchDim = 1;
        return true;
    }
    return false;
}

void ComputeMatmulTiling::Convert2AscendCTiling(const Tiling &tbeTiling, TCubeTiling &matmulTiling,
                                                const MatmulParams &params, MatmulMultiCoreResult &multiCoreResult)
{
    auto mDim = ops::CeilDiv(params.mSize, ops::CeilDiv(params.mSize, static_cast<uint64_t>(tbeTiling.m_dim)));
    auto nDim = ops::CeilDiv(params.nSize, ops::CeilDiv(params.nSize, static_cast<uint64_t>(tbeTiling.n_dim)));
    matmulTiling.set_usedCoreNum(1);
    matmulTiling.set_M(params.mSize);
    matmulTiling.set_N(params.nSize);
    matmulTiling.set_Ka(params.kSize);
    matmulTiling.set_Kb(params.kSize);
    matmulTiling.set_Kb(params.kSize);
    if (params.transB && params.kbAlign) {
        // 转置且kbAlign为true场景，内轴需256对齐以提高nd2nz效率
        uint64_t kAlign = ops::CeilAlign(params.kSize, static_cast<uint64_t>(256));
        if (kAlign < MAX_SHAPE_DIM) {
            matmulTiling.set_Kb(kAlign);
        }
    }

    matmulTiling.set_singleCoreM(static_cast<int32_t>(ops::CeilDiv(params.mSize, mDim)));
    matmulTiling.set_singleCoreN(static_cast<int32_t>(ops::CeilDiv(params.nSize, nDim)));
    matmulTiling.set_baseN(tbeTiling.n_l0 * BLOCK_CUBE);
    matmulTiling.set_singleCoreK(params.kSize);
    matmulTiling.set_baseM(tbeTiling.m_l0 * BLOCK_CUBE);
    int32_t reduceSize = static_cast<int32_t>(GetBlockAlignSizeByDataType(params.aDtype));
    matmulTiling.set_baseK(tbeTiling.k_l0 * reduceSize);
    matmulTiling.set_depthA1(ops::CeilDiv(tbeTiling.kal1_16, tbeTiling.k_l0) * tbeTiling.m_al1 * tbeTiling.db_al1);
    matmulTiling.set_depthB1(ops::CeilDiv(tbeTiling.kbl1_16, tbeTiling.k_l0) * tbeTiling.n_bl1 * tbeTiling.db_bl1);
    matmulTiling.set_stepM(tbeTiling.m_al1);
    matmulTiling.set_stepN(tbeTiling.n_bl1);
    matmulTiling.set_stepKa(ops::CeilDiv(tbeTiling.kal1_16, tbeTiling.k_l0));
    matmulTiling.set_stepKb(ops::CeilDiv(tbeTiling.kbl1_16, tbeTiling.k_l0));
    int32_t a1Length = static_cast<int32_t>(
        GetShapeSizeWithDataType(matmulTiling.get_baseM() * matmulTiling.get_baseK(), params.aDtype));
    int32_t b1Length = static_cast<int32_t>(
        GetShapeSizeWithDataType(matmulTiling.get_baseN() * matmulTiling.get_baseK(), params.aDtype));
    int32_t c1Length = matmulTiling.get_baseN() * matmulTiling.get_baseM() * sizeof(float);  // L0C

    matmulTiling.set_isBias(params.hasBias ? 1 : 0);
    matmulTiling.set_transLength(std::max(std::max(a1Length, b1Length), c1Length));
    // MatrixTraverse枚举值和matmul api使用的枚举值相差1
    matmulTiling.set_iterateOrder(
        static_cast<int32_t>(GetIteratorOrder(tbeTiling, matmulTiling.get_singleCoreM(), matmulTiling.get_singleCoreN(),
                                              matmulTiling.get_singleCoreK(), params.aDtype)) -
        1);
    matmulTiling.set_shareMode(0);
    matmulTiling.set_dbL0A(2);  // db switch, 1: off, 2: on
    matmulTiling.set_dbL0B(2);  // db switch, 1: off, 2: on
    matmulTiling.set_dbL0C(tbeTiling.db_l0c);
    int32_t aL1Size = a1Length * matmulTiling.get_depthA1();
    int32_t biasL1Size =
        params.hasBias ? GetShapeSizeWithDataType(matmulTiling.get_baseN(), params.biasDtype) * tbeTiling.n_bl1 : 0;
    int32_t bL1Size = b1Length * matmulTiling.get_depthB1();
    int32_t quantScaleL1Size =
        params.cDtype == ge::DT_INT8 ? matmulTiling.get_baseN() * tbeTiling.n_bl1 * sizeof(uint64_t) : 0;
    matmulTiling.set_shareL1Size(aL1Size + bL1Size + biasL1Size + quantScaleL1Size);
    matmulTiling.set_shareL0CSize(c1Length);
    matmulTiling.set_shareUbSize(0);
    matmulTiling.set_batchM(1);
    matmulTiling.set_batchN(1);
    matmulTiling.set_singleBatchM(1);
    matmulTiling.set_singleBatchN(1);

    multiCoreResult.mDim = mDim;
    multiCoreResult.nDim = nDim;
    multiCoreResult.batchDim = tbeTiling.batch_dim;
}

MatrixTraverse ComputeMatmulTiling::GetIteratorOrder(const Tiling &tbeTiling, int32_t singleCoreM, int32_t singleCoreN,
                                                     int32_t singleCoreK, ge::DataType aDtype)
{
    int32_t reduceSize = static_cast<int32_t>(GetBlockAlignSizeByDataType(aDtype));
    bool fullkAL1Load = !((static_cast<float>(singleCoreK) / (tbeTiling.kal1_16 * reduceSize)) > 1.0);
    bool fullkBL1Load = !((static_cast<float>(singleCoreK) / (tbeTiling.kbl1_16 * reduceSize)) > 1.0);

    // if KAL1 and KBL1 both can not be full loaded, then select m or n which is no matter
    if (!fullkAL1Load && !fullkBL1Load) {
        return MatrixTraverse::FIRSTM;
    } else if (fullkAL1Load && !fullkBL1Load) {  // if KAL1 is full loaded, then select the order N first
        return MatrixTraverse::FIRSTN;
    } else if (!fullkAL1Load && fullkBL1Load) {  // if KBL1 is full loaded, then select the order M first
        return MatrixTraverse::FIRSTM;
    } else {
        // if AL1LoadSize less than BL1LoadSize, then select order N first, vice versa.
        int32_t mLoop = ops::CeilDiv(singleCoreM, static_cast<int32_t>(tbeTiling.m_al1 * tbeTiling.m_l0 * BLOCK_CUBE));
        int32_t nLoop = ops::CeilDiv(singleCoreN, static_cast<int32_t>(tbeTiling.n_bl1 * tbeTiling.n_l0 * BLOCK_CUBE));
        int32_t aL1LoadSize = singleCoreM + singleCoreN * mLoop;
        int32_t bL1LoadSize = singleCoreN + singleCoreM * nLoop;
        return aL1LoadSize < bL1LoadSize ? MatrixTraverse::FIRSTN : MatrixTraverse::FIRSTM;
    }
}

bool ComputeMatmulTiling::tryComputeSimpleTiling(TCubeTiling &matmulTiling, const MatmulParams &params,
                                                 const AiCoreParams &aicoreParams)
{
    // 当前只适配baseK大于等于256场景，baseK将划分成1024/512/256三档
    uint64_t baseKOption1024 = 1024;
    uint64_t baseKOption512 = 512;
    uint64_t baseKOption256 = 256;
    // a不转置场景，mAlign考虑16对齐，
    uint64_t mAlign = ops::CeilAlign(params.mSize, static_cast<uint64_t>(BLOCK_CUBE));
    matmulTiling.set_baseM(mAlign);

    // 考虑l0 开db和int4场景，l0a实际可用size即原始l0a大小
    uint64_t baseKMax = mAlign > 0 ? aicoreParams.l0aSize / mAlign : 0;
    if (baseKMax < baseKOption256) {
        OPS_LOG_W("WeightQuantBatchMatmulV2", "SimpleIncreTiling can not compute tiling. baseKMax:[%lu] ", baseKMax);
        return false;
    }
    uint64_t baseK = 0;
    if (baseKMax >= baseKOption1024) {
        baseK = baseKOption1024;
    } else if (baseKMax >= baseKOption512) {
        baseK = baseKOption512;
    } else {
        baseK = baseKOption256;
    }
    matmulTiling.set_baseK(baseK);
    if (!tryAFullLoad(matmulTiling, params, aicoreParams) &&
        !trySimpleTilingNormalLoad(matmulTiling, params, aicoreParams)) {
        return false;
    }
    uint64_t realL0cSize = static_cast<uint64_t>(matmulTiling.get_baseM()) * matmulTiling.get_baseN() * sizeof(int32_t);
    if (realL0cSize > aicoreParams.l0cSize) {
        OPS_LOG_W("WeightQuantBatchMatmulV2",
                "SimpleIncreTiling can not compute norm load tiling. realL0cSize:[%lu], l0cSize:[%lu]", realL0cSize,
                aicoreParams.l0cSize);
        return false;
    }
    int32_t a1Length = static_cast<int32_t>(
        GetShapeSizeWithDataType(matmulTiling.get_baseM() * matmulTiling.get_baseK(), params.aDtype));
    int32_t b1Length = static_cast<int32_t>(
        GetShapeSizeWithDataType(matmulTiling.get_baseN() * matmulTiling.get_baseK(), params.bDtype));
    int32_t c1Length = matmulTiling.get_baseN() * matmulTiling.get_baseM() * sizeof(float) * matmulTiling.get_stepN();
    matmulTiling.set_transLength(std::max(std::max(a1Length, b1Length), c1Length));
    // MatrixTraverse枚举值和matmul api使用的枚举值相差1
    matmulTiling.set_iterateOrder(static_cast<int32_t>(MatrixTraverse::FIRSTN) - 1);
    matmulTiling.set_shareL0CSize(c1Length);
    matmulTiling.set_singleCoreN(
        ops::CeilAlign(ops::CeilDiv(params.nSize, aicoreParams.aicNum), static_cast<uint64_t>(BLOCK_CUBE)));
    matmulTiling.set_dbL0C(std::min(DOUBLE_BUFFER_FACTOR, aicoreParams.l0cSize / realL0cSize));
    return true;
}

bool ComputeMatmulTiling::tryAFullLoad(TCubeTiling &matmulTiling, const MatmulParams &params,
                                       const AiCoreParams &aicoreParams)
{
    uint64_t kAlign = ops::CeilAlign(params.kSize, INT4_BLK_SIZE);
    uint64_t stepKa = ops::CeilDiv(kAlign, static_cast<uint64_t>(matmulTiling.get_baseK()));

    uint64_t aL1Size = matmulTiling.get_baseM() * (matmulTiling.get_baseK() >> 1) * stepKa;
    if (aL1Size >= aicoreParams.l1Size) {
        OPS_LOG_W("WeightQuantBatchMatmulV2",
                "SimpleIncreTiling can not compute A full load tiling. aL1Size:[%lu], l1Size:[%lu]", aL1Size,
                aicoreParams.l1Size);
        return false;
    }

    uint64_t stepKb = 4;
    uint64_t singleCoreKb = matmulTiling.get_baseK() * stepKb;
    // kb可全载场景，singleCoreKb需要进一步缩小
    if (singleCoreKb >= kAlign) {
        singleCoreKb = kAlign;
        if (static_cast<uint64_t>(matmulTiling.get_baseK()) > singleCoreKb) {
            matmulTiling.set_baseK(singleCoreKb);
        }
        stepKb = ops::CeilDiv(singleCoreKb, static_cast<uint64_t>(matmulTiling.get_baseK()));
    }
    uint64_t nAlign = ops::CeilAlign(params.nSize, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t l0bMaxBaseN =
        ops::FloorAlign(aicoreParams.l0bSize / matmulTiling.get_baseK(), static_cast<uint64_t>(BLOCK_CUBE));
    // weight默认开db，所以int4场景不需要除2再计算
    uint64_t l1bMaxBaseN =
        ops::FloorAlign((aicoreParams.l1Size - aL1Size) / singleCoreKb, static_cast<uint64_t>(BLOCK_CUBE));

    // stepN默认为1, 可以考虑适当调整stepN
    uint64_t stepN = 1;
    uint64_t l0cMaxBaseN = ops::FloorAlign(aicoreParams.l0cSize / (matmulTiling.get_baseM() * sizeof(int32_t) * stepN),
                                           static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t baseN = std::min(std::min(std::min(l1bMaxBaseN, l0bMaxBaseN), l0cMaxBaseN), nAlign);
    if (baseN < BLOCK_CUBE) {
        OPS_LOG_W("WeightQuantBatchMatmulV2", "SimpleIncreTiling can not compute A full load tiling. baseN:[%lu]", baseN);
        return false;
    }

    uint64_t bL1Size = baseN * stepN * (singleCoreKb >> 1);
    if (aL1Size + DOUBLE_BUFFER_FACTOR * bL1Size > aicoreParams.l1Size) {
        OPS_LOG_W("WeightQuantBatchMatmulV2",
                "SimpleIncreTiling can not compute A full load tiling. aL1Size:[%lu], bL1Size:[%lu], l1Size:[%lu]",
                aL1Size, bL1Size, aicoreParams.l1Size);
        return false;
    }

    matmulTiling.set_stepN(stepN);
    matmulTiling.set_stepKa(ops::CeilDiv(kAlign, static_cast<uint64_t>(matmulTiling.get_baseK())));
    matmulTiling.set_depthA1(matmulTiling.get_stepKa());
    matmulTiling.set_stepKb(stepKb);
    matmulTiling.set_baseN(baseN);
    matmulTiling.set_depthB1(DOUBLE_BUFFER_FACTOR * stepKb * stepN);
    return true;
}

bool ComputeMatmulTiling::trySimpleTilingNormalLoad(TCubeTiling &matmulTiling, const MatmulParams &params,
                                                    const AiCoreParams &aicoreParams)
{
    // a不能全载，尽量减少aL1的空间，扩大b的载入量，可以减少a的重复载入
    uint64_t kAlign = ops::CeilAlign(params.kSize, INT4_BLK_SIZE);
    // stepKb默认为4, 避免发生issue que阻塞
    uint64_t stepKb = 4;
    // stepKa最小应为stepKb。这样可以保证al1一次载入计算多次bl1
    uint64_t stepKaMin = stepKb;
    uint64_t aL1SizeMin =
        matmulTiling.get_baseM() * (std::min(matmulTiling.get_baseK() * stepKaMin, kAlign) >> 1) * DOUBLE_BUFFER_FACTOR;
    if (aL1SizeMin >= aicoreParams.l1Size) {
        OPS_LOG_W("WeightQuantBatchMatmulV2",
                "SimpleIncreTiling can not compute norm load tiling. aL1SizeMin:[%lu], l1Size:[%lu]", aL1SizeMin,
                aicoreParams.l1Size);
        return false;
    }
    uint64_t nAlign = ops::CeilAlign(params.nSize, static_cast<uint64_t>(BLOCK_CUBE));
    // n应尽量取大，减少a的重复载入
    uint64_t l0bMaxBaseN =
        ops::FloorAlign(aicoreParams.l0bSize / matmulTiling.get_baseK(), static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t l0cMaxBaseN = ops::FloorAlign(aicoreParams.l0cSize / (matmulTiling.get_baseM() * sizeof(int32_t)),
                                           static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t singleCoreKb = matmulTiling.get_baseK() * stepKb;
    uint64_t l1bMaxBaseN =
        ops::FloorAlign((aicoreParams.l1Size - aL1SizeMin) / singleCoreKb, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t baseN = std::min(std::min(std::min(l1bMaxBaseN, l0bMaxBaseN), l0cMaxBaseN), nAlign);
    if (baseN < ONE_BLK_SIZE) {
        OPS_LOG_W("WeightQuantBatchMatmulV2", "SimpleIncreTiling can not compute normal load tiling. baseN:[%lu]", baseN);
        return false;
    }
    // 根据bl1size来反推al1Size最大空间
    uint64_t bL1Size = baseN * (singleCoreKb >> 1);
    uint64_t singleCoreKaMax = (aicoreParams.l1Size - DOUBLE_BUFFER_FACTOR * bL1Size) / matmulTiling.get_baseM();
    uint64_t stepKa = ops::CeilDiv(singleCoreKaMax, static_cast<uint64_t>(matmulTiling.get_baseK()));
    if (stepKa < stepKb) {
        OPS_LOG_W("WeightQuantBatchMatmulV2",
                "SimpleIncreTiling can not compute norm load tiling. stepKa:[%lu], stepKb:[%lu]", stepKa, stepKb);
        return false;
    }

    uint64_t depthA1 = stepKa;
    // 修正实际的stepKa, 尽可能预留db
    if (stepKa / stepKb >= DOUBLE_BUFFER_FACTOR) {
        stepKa = stepKa >> 1;
        stepKa = ops::FloorAlign(stepKa, stepKb);
        depthA1 = DOUBLE_BUFFER_FACTOR * stepKa;
    }

    matmulTiling.set_stepKa(stepKa);
    matmulTiling.set_stepKb(stepKb);
    matmulTiling.set_baseN(baseN);
    matmulTiling.set_stepN(1);
    matmulTiling.set_depthA1(depthA1);
    matmulTiling.set_depthB1(DOUBLE_BUFFER_FACTOR * stepKb);
    return true;
}
}  // namespace optiling

