/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file swin_attention_score_quant_tiling.cc
 * \brief SwinAttentionScoreQuant tiling impl
 */

#include "register/op_def_registry.h"
#include "tiling/tiling_templates_registry.h"
#include "swin_attention_score_quant_tiling.h"

namespace optiling {
const uint32_t INT8_NOMASK_MODE = 0;
const uint32_t INT8_MASK_MODE = 1;
const uint32_t Q_IDX = 0;
const uint32_t K_IDX = 1;
const uint32_t V_IDX = 2;
const uint32_t SCALE_QUANT_IDX = 3;
const uint32_t SCALE_DEQUANT1_IDX = 4;
const uint32_t SCALE_DEQUANT2_IDX = 5;
const uint32_t BIAS_QUANT_IDX = 6;
const uint32_t BIAS_DEQUANT1_IDX = 7;
const uint32_t BIAS_DEQUANT2_IDX = 8;
const uint32_t MASK1_IDX = 9;
const uint32_t MASK2_IDX = 10;
const uint32_t QKV_DIM_SIZE = 4;
const uint32_t SMALL_S_DIM = 128;
const uint32_t CRITICAL_S_DIM = 970;
const uint32_t WORKSPACE_SIZE = 16777216;
const uint32_t BLOCK_NUM_PER_FRACTAL = 16;
const uint32_t NUM_PER_FRACTAL = 512;
const uint32_t BLOCK_SIZE_32 = 32;
const uint32_t BLOCK_SIZE_16 = 16;
const int64_t MIN_S_DIM = 1;
const int64_t MAX_S_DIM = 1024;
const int64_t MIN_H_DIM = 32;
const int64_t MAX_H_DIM = 64;

inline uint32_t DivUp(uint32_t num, uint32_t align)
{
    if (align > 0) {
        return (num + align - 1) / align;
    }
    return 0;
}

inline uint32_t RoundUp(uint32_t num, uint32_t align)
{
    return DivUp(num, align) * align;
}

struct SwinAttentionScoreQuantCompileInfo {};

inline bool CheckQTensor(gert::TilingContext *context, gert::Shape &qOriShape)
{
    OPS_CHECK(
        qOriShape.GetDimNum() != QKV_DIM_SIZE, OPS_LOG_E(context->GetNodeName(), "input tensor %lu's dim is not equal to %u\n", qOriShape.GetDimNum(), QKV_DIM_SIZE), return false);
    int64_t dimB = qOriShape.GetDim(0);
    int64_t dimN = qOriShape.GetDim(1);
    int64_t dimS = qOriShape.GetDim(2); // query的第2维是S
    int64_t dimH = qOriShape.GetDim(3); // query的第3维是H
    int64_t dimBN = dimB * dimN;
    OPS_CHECK(dimS < MIN_S_DIM || dimS > MAX_S_DIM, OPS_LOG_E(context->GetNodeName(),
        "input tensor query's s dim is %ld should great than 0 and less than 1024\n", dimS), return false);
    OPS_CHECK(dimH != MIN_H_DIM && dimH != MAX_H_DIM, OPS_LOG_E(context->GetNodeName(),
        "input tensor query's h dim is %ld should be 32 or 64\n", dimH), return false);
    OPS_CHECK(dimBN < 0 || dimBN > (int64_t)UINT32_MAX / dimS,
        OPS_LOG_E(context->GetNodeName(), "input tensor size exceeds the limit\n"), return false);
    return true;
}

inline bool CheckQKVDimSize(gert::TilingContext *context, gert::Shape &qOriShape, gert::Shape & kvOriShape)
{
    OPS_CHECK(kvOriShape.GetDimNum() != QKV_DIM_SIZE,
        OPS_LOG_E(context->GetNodeName(), "input tensor kv's dim num is not equal to q\n"), return false);
    for (uint32_t i = 0; i < qOriShape.GetDimNum(); i++) {
        OPS_CHECK(qOriShape.GetDim(i) != kvOriShape.GetDim(i),
            OPS_LOG_E(context->GetNodeName(), "input tensor kv's dim is not equal to q\n"), return false);
    }
    return true;
}

inline bool CheckKVTensor(gert::TilingContext *context, gert::Shape &qOriShape)
{
    for (uint32_t i = K_IDX; i <= V_IDX; i++) {
        auto kvShape = context->GetInputShape(i);
        if (kvShape != nullptr) {
            auto kvOriShape = kvShape->GetOriginShape();
            OPS_CHECK(!CheckQKVDimSize(context, qOriShape, kvOriShape),
                OPS_LOG_E(context->GetNodeName(), "input tensor kv's dim check failed\n"), return false);
        } else {
            OPS_LOG_E(context->GetNodeName(), "input tensor %u is nullptr\n", i);
            return false;
        }
    }
    return true;
}

inline bool CheckQuantTensor(gert::TilingContext *context)
{
    for (uint32_t i = SCALE_QUANT_IDX; i < BIAS_DEQUANT2_IDX; i++) {
        auto quantShape = context->GetOptionalInputShape(i);
        OPS_CHECK(quantShape == nullptr,
            OPS_LOG_E(context->GetNodeName(), "input tensor %u is nullptr\n", i), return false);
    }
    return true;
}

inline bool CheckMaskDimSize(gert::TilingContext *context, gert::Shape &qOriShape, gert::Shape & maskOriShape)
{
    OPS_CHECK(maskOriShape.GetDimNum() != QKV_DIM_SIZE,
        OPS_LOG_E(context->GetNodeName(), "input tensor mask's dim num not equal to q\n"), return false);
    OPS_CHECK(maskOriShape.GetDim(0) != 1,
        OPS_LOG_E(context->GetNodeName(), "input tensor mask[0] is not equal to 1\n"), return false);
    for (uint32_t i = 1; i < QKV_DIM_SIZE - 1; i++) {
        OPS_CHECK(qOriShape.GetDim(i) != maskOriShape.GetDim(i),
            OPS_LOG_E(context->GetNodeName(), "input dim[%u] of mask is wrong\n", i), return false);
    }
    OPS_CHECK(qOriShape.GetDim(QKV_DIM_SIZE - 2) != maskOriShape.GetDim(QKV_DIM_SIZE - 1),
        OPS_LOG_E(context->GetNodeName(), "input dim[3] of mask is wrong\n"), return false);  // Mask的最后2维大小相等
    return true;
}

inline bool CheckMaskTensor(gert::TilingContext *context, gert::Shape &qOriShape)
{
    auto maskShape1 = context->GetOptionalInputShape(MASK1_IDX);
    if (maskShape1 != nullptr) {
        auto maskOriShape = maskShape1->GetOriginShape();
        OPS_CHECK(!CheckMaskDimSize(context, qOriShape, maskOriShape),
            OPS_LOG_E(context->GetNodeName(), "input mask check failed\n"), return false);
    }
    return true;
}

inline bool CheckInTensor(gert::TilingContext *context)
{
    auto qShape = context->GetInputShape(Q_IDX);
    OPS_CHECK(qShape == nullptr,
        OPS_LOG_E(context->GetNodeName(), "input tensor %u is nullptr\n", Q_IDX), return false);
    auto qOriShape = qShape->GetOriginShape();
    OPS_CHECK(!CheckQTensor(context, qOriShape),
        OPS_LOG_E(context->GetNodeName(), "input q check failed\n"), return false);
    OPS_CHECK(!CheckKVTensor(context, qOriShape),
        OPS_LOG_E(context->GetNodeName(), "input kv check failed\n"), return false);
    OPS_CHECK(!CheckQuantTensor(context),
        OPS_LOG_E(context->GetNodeName(), "input quant check failed\n"), return false);
    OPS_CHECK(!CheckMaskTensor(context, qOriShape),
        OPS_LOG_E(context->GetNodeName(), "input mask check failed\n"), return false);
    
    return true;
}

inline void GetSingleCoreM(uint32_t s, uint32_t &singleCoreM)
{
    if (s > SMALL_S_DIM) {
        singleCoreM = BLOCK_NUM_PER_FRACTAL;
    } else {
        singleCoreM = RoundUp(s, BLOCK_NUM_PER_FRACTAL);
    }
}

inline ge::graphStatus SwinAttentionCfgTiling(SwinAttentionScoreQuantTilingData &tiling,
    uint32_t &coreNum, uint32_t &tmpSize, gert::TilingContext *context, bool hasMask)
{
    uint32_t cubeSharedUbSize;
    uint32_t vecSharedUbSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    coreNum = ascendcPlatform.GetCoreNum();
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    auto inputShape = context->GetInputShape(0);
    OPS_CHECK(inputShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context,
        "[SwinAttentionCfgTiling] inputShape is null."), return ge::GRAPH_FAILED);
    auto qOriShape = inputShape->GetOriginShape();
    uint32_t dimB = qOriShape.GetDim(0);
    uint32_t dimN = qOriShape.GetDim(1);
    uint32_t dimS = qOriShape.GetDim(2);    // query的第2维是S
    uint32_t dimH = qOriShape.GetDim(3);    // query的第3维是H
    uint32_t round16S = RoundUp(dimS, BLOCK_SIZE_16);
    uint32_t round32S = RoundUp(dimS, BLOCK_SIZE_32);
    tmpSize = 0;

    uint32_t singleCoreM;
    GetSingleCoreM(dimS, singleCoreM);

    uint32_t numPerS = DivUp(dimS, singleCoreM);
    uint32_t coreLoops = dimB;
    if (hasMask) {
        coreLoops = dimB;
    } else {
        coreLoops = dimB * dimN * numPerS;
    }
    tiling.set_coreLoops(coreLoops);
    coreNum = (coreNum < coreLoops) ? coreNum : coreLoops;

    // qk tiling
    matmul_tiling::MatmulApiTiling qkBmm(ascendcPlatform);
    matmul_tiling::MatmulApiTiling pvBmm(ascendcPlatform);
    qkBmm.SetAType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8,
        false);
    qkBmm.SetBType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8,
        true);
    qkBmm.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ,
        matmul_tiling::DataType::DT_FLOAT16);
    qkBmm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    qkBmm.SetShape(singleCoreM, dimS, dimH);
    qkBmm.SetOrgShape(singleCoreM, round32S, dimH);
    qkBmm.SetBias(true);
    qkBmm.SetBufferSpace(l1Size / 2, l0CSize);  // 使用1/2的l1空间
    OPS_CHECK(qkBmm.GetTiling(tiling.qkBmmTilingData) == -1,
        OPS_LOG_E(context->GetNodeName(), "get qk tiling failed\n"), return ge::GRAPH_FAILED);
    matmul_tiling::SysTilingTempBufSize qkBufSize;
    OPS_CHECK(MatmulGetTmpBufSize(tiling.qkBmmTilingData, qkBufSize) == -1,
        OPS_LOG_E(context->GetNodeName(), "get qk buf size failed\n"), return ge::GRAPH_FAILED);
    cubeSharedUbSize = qkBufSize.ubSize;

    // pv tiling
    pvBmm.SetAType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8,
        false);
    pvBmm.SetBType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8,
        true);
    pvBmm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16);
    pvBmm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    pvBmm.SetShape(singleCoreM, dimH, dimS);
    pvBmm.SetOrgShape(singleCoreM, dimH, round32S, round32S);
    pvBmm.SetBias(true);
    pvBmm.SetBufferSpace(l1Size / 2, l0CSize);  // 使用1/2的l1空间
    OPS_CHECK(pvBmm.GetTiling(tiling.pvBmmTilingData) == -1,
        OPS_LOG_E(context->GetNodeName(), "get pv tiling failed\n"), return ge::GRAPH_FAILED);
    uint32_t pvUbSize = tiling.pvBmmTilingData.get_transLength() + tiling.pvBmmTilingData.get_transLength();
    uint32_t pvUbSizeTmp = round32S * dimH + round32S * dimH;
    pvUbSize = std::max(pvUbSize, pvUbSizeTmp);
    cubeSharedUbSize = std::max(cubeSharedUbSize, pvUbSize);

    auto qkL0 = tiling.qkBmmTilingData.get_shareL0CSize();
    auto qkL1 = tiling.qkBmmTilingData.get_shareL1Size();
    auto pvL0 = tiling.pvBmmTilingData.get_shareL0CSize();
    auto pvL1 = tiling.pvBmmTilingData.get_shareL1Size();

    auto bmmMaxL0 = std::max(qkL0, pvL0);
    auto bmmMaxL1 = std::max(qkL1, pvL1);
    
    tiling.qkBmmTilingData.set_shareL0CSize(bmmMaxL0);
    tiling.qkBmmTilingData.set_shareL1Size(bmmMaxL1);
    tiling.pvBmmTilingData.set_shareL0CSize(bmmMaxL0);
    tiling.pvBmmTilingData.set_shareL1Size(bmmMaxL1);

    // softmax
    uint32_t softmaxTmpSize;
    const ge::Shape softmaxShape = ge::Shape({ singleCoreM, round16S });
    if (hasMask && dimS > CRITICAL_S_DIM) {
        softmaxTmpSize = 0;
    } else {
        softmaxTmpSize = AscendC::GetSoftMaxMaxTmpSize(softmaxShape, sizeof(uint16_t), true);
        AscendC::SoftMaxTilingFunc(softmaxShape, sizeof(uint16_t), softmaxTmpSize, tiling.softmaxTilingData);
    }
    vecSharedUbSize = softmaxTmpSize;

    uint32_t qSize = RoundUp(singleCoreM * dimH, NUM_PER_FRACTAL);
    uint32_t kSize = RoundUp(dimH * round32S, NUM_PER_FRACTAL);
    uint32_t pSize = RoundUp(singleCoreM * round32S, NUM_PER_FRACTAL);
    uint32_t vSize = RoundUp(round32S * dimH, NUM_PER_FRACTAL);

    tiling.set_dimB(dimB);
    tiling.set_dimN(dimN);
    tiling.set_dimS(dimS);
    tiling.set_dimH(dimH);
    tiling.set_qSize(qSize);
    tiling.set_kSize(kSize);
    tiling.set_pSize(pSize);
    tiling.set_vSize(vSize);
    tiling.set_cubeSharedUbSize(cubeSharedUbSize);
    tiling.set_vecSharedUbSize(vecSharedUbSize);
    
    return ge::GRAPH_SUCCESS;
}

void TilingLog(SwinAttentionScoreQuantTilingData &tiling) {
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[coreLoops]: %u", tiling.get_coreLoops());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[dimB]: %u", tiling.get_dimB());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[dimN]: %u", tiling.get_dimN());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[dimB]: %u", tiling.get_dimS());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[dimN]: %u", tiling.get_dimH());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[qSize]: %u", tiling.get_qSize());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[kSize]: %u", tiling.get_kSize());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[pSize]: %u", tiling.get_pSize());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[vSize]: %u", tiling.get_vSize());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[cubeSharedUbSize]: %u", tiling.get_cubeSharedUbSize());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[vecSharedUbSize]: %u", tiling.get_vecSharedUbSize());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[qkBmm.M]: %d", tiling.qkBmmTilingData.get_M());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[qkBmm.N]: %d", tiling.qkBmmTilingData.get_N());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[qkBmm.Ka]: %d", tiling.qkBmmTilingData.get_Ka());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[qkBmm.Kb]: %d", tiling.qkBmmTilingData.get_Kb());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[qkBmm.singleCoreM]: %d", tiling.qkBmmTilingData.get_singleCoreM());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[qkBmm.singleCoreN]: %d", tiling.qkBmmTilingData.get_singleCoreN());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[qkBmm.singleCoreK]: %d", tiling.qkBmmTilingData.get_singleCoreK());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[qkBmm.baseM]: %d", tiling.qkBmmTilingData.get_baseM());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[qkBmm.baseN]: %d", tiling.qkBmmTilingData.get_baseN());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[qkBmm.baseK]: %d", tiling.qkBmmTilingData.get_baseK());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[qkBmm.isBias]: %d", tiling.qkBmmTilingData.get_isBias());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[pvBmm.M]: %d", tiling.pvBmmTilingData.get_M());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[pvBmm.N]: %d", tiling.pvBmmTilingData.get_N());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[pvBmm.Ka]: %d", tiling.pvBmmTilingData.get_Ka());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[pvBmm.Kb]: %d", tiling.pvBmmTilingData.get_Kb());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[pvBmm.singleCoreM]: %d", tiling.pvBmmTilingData.get_singleCoreM());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[pvBmm.singleCoreN]: %d", tiling.pvBmmTilingData.get_singleCoreN());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[pvBmm.singleCoreK]: %d", tiling.pvBmmTilingData.get_singleCoreK());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[pvBmm.baseM]: %d", tiling.pvBmmTilingData.get_baseM());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[pvBmm.baseN]: %d", tiling.pvBmmTilingData.get_baseN());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[pvBmm.baseK]: %d", tiling.pvBmmTilingData.get_baseK());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[pvBmm.isBias]: %d", tiling.pvBmmTilingData.get_isBias());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.srcM]: %u", tiling.softmaxTilingData.get_srcM());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.srcK]: %u", tiling.softmaxTilingData.get_srcK());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.srcSize]: %u", tiling.softmaxTilingData.get_srcSize());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.outMaxM]: %u", tiling.softmaxTilingData.get_outMaxM());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.outMaxK]: %u", tiling.softmaxTilingData.get_outMaxK());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.outMaxSize]: %u", tiling.softmaxTilingData.get_outMaxSize());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.splitM]: %u", tiling.softmaxTilingData.get_splitM());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.splitK]: %u", tiling.softmaxTilingData.get_splitK());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.splitSize]: %u", tiling.softmaxTilingData.get_splitSize());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.reduceM]: %u", tiling.softmaxTilingData.get_reduceM());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.reduceK]: %u", tiling.softmaxTilingData.get_reduceK());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.reduceSize]: %u", tiling.softmaxTilingData.get_reduceSize());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.rangeM]: %u", tiling.softmaxTilingData.get_rangeM());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.tailM]: %u", tiling.softmaxTilingData.get_tailM());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.tailSplitSize]: %u", tiling.softmaxTilingData.get_tailSplitSize());
    OPS_LOG_I("[SwinAttentionScoreQuant]", "[softmaxTilingData.tailReduceSize]: %u", tiling.softmaxTilingData.get_tailReduceSize());
}

ASCENDC_EXTERN_C ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ge::graphStatus res = ge::GRAPH_SUCCESS;
    SwinAttentionScoreQuantTilingData tiling;
    uint32_t coreNum;
    uint32_t tmpSize;
    
    OPS_CHECK(!CheckInTensor(context), OPS_LOG_E(context->GetNodeName(), "input tensor check failed\n"),
        return ge::GRAPH_FAILED);

    auto maskShape = context->GetOptionalInputShape(MASK1_IDX);
    if (maskShape == nullptr) {
        OPS_LOG_I(context->GetNodeName(), "without mask\n");
        context->SetTilingKey(INT8_NOMASK_MODE);
        res = SwinAttentionCfgTiling(tiling, coreNum, tmpSize, context, false);
    } else {
        OPS_LOG_I(context->GetNodeName(), "with mask\n");
        context->SetTilingKey(INT8_MASK_MODE);
        res = SwinAttentionCfgTiling(tiling, coreNum, tmpSize, context, true);
    }

    OPS_CHECK(res == ge::GRAPH_FAILED,
        OPS_LOG_E(context->GetNodeName(), "get tiling failed\n"), return ge::GRAPH_FAILED);
    
    TilingLog(tiling);

    size_t *workspaces = context->GetWorkspaceSizes(1);
    OPS_CHECK(workspaces == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context,
                        "failed to get workspace size"),
                        return ge::GRAPH_FAILED);
    workspaces[0] = tmpSize + WORKSPACE_SIZE;
    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return res;
}

ge::graphStatus TilingPrepareForSwinAttentionScoreQuant(gert::TilingParseContext* context) {
    auto compileInfo = context->GetCompiledInfo<SwinAttentionScoreQuantCompileInfo>();
    OPS_LOG_E_IF_NULL(context, compileInfo, return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwinAttentionScoreQuant)
.Tiling(TilingFunc)
.TilingParse<SwinAttentionScoreQuantCompileInfo>(TilingPrepareForSwinAttentionScoreQuant);  // 向框架注册入口函数
}   // optiling