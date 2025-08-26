/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp
 * \brief
 */

#include "flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.h"
#include "tiling/tiling_type.h"
#include "tiling/tiling_templates_registry.h"

namespace optiling {

constexpr uint32_t INITIAL_S1_SPLIT_NUM = 128; // to avoid repeat max value 255
constexpr uint32_t INITIAL_S2_SPLIT_NUM = 64;
constexpr uint32_t MUL_CORE_SYNC_BUFFER = 64 * 1024;

constexpr uint32_t EMPTY_TENSOR = 0;
constexpr uint32_t NORMAL_TENSOR = 1;

constexpr uint32_t MAX_BASIC_BLOCK_SIZE = 1024;
constexpr uint32_t PSE_NORMAL_SHAPE_DIM = 4;
constexpr uint32_t TEMP_BUFFER_REMAIN_SIZE = 1024 * 2;

constexpr uint32_t ATTEN_MASK_SHAPE_DIMS_0 = 0;
constexpr uint32_t ATTEN_MASK_SHAPE_DIMS_1 = 1;
constexpr uint32_t ATTEN_MASK_DIM_LENGTH_2 = 2;
constexpr uint32_t ATTEN_MASK_DIM_LENGTH_4 = 4;

constexpr uint32_t PSE_DIM_NUM_1 = 1;
constexpr uint32_t PSE_DIM_NUM_2 = 2;

constexpr uint32_t INPUT_FORMAT_BN2GS2D = 0; // BNSD
constexpr uint32_t INPUT_FORMAT_S2BN2GD = 1; // SBH
constexpr uint32_t INPUT_FORMAT_BS2N2GD = 2; // BSH  BSND
constexpr uint32_t INPUT_FORMAT_TND = 3;     // TND

constexpr uint32_t CORE_INIT_NUM = 40;
constexpr uint32_t MATMUL_SIZE = 8 * 1024;

constexpr uint32_t INPUT_ALIGN = 16;
constexpr uint32_t WORKSPACE_NUM_ALIGN = 256;
constexpr int64_t GM_ALIGN = 512;

constexpr uint32_t CALCULATED_BLOCK_DIMENSION = 4;
constexpr uint32_t BASIC_BLOCK_MULTIPLE = 15;
constexpr uint32_t POST_NZ_RESERVED_N = 4;
constexpr uint32_t POST_NZ_COEX_NODE = 10;
constexpr uint32_t POST_COEX_NODE = 3;
constexpr uint32_t BUFFER_NUM = 1;

constexpr uint32_t FP16_BYTES = 2;
constexpr uint32_t FP16_BLOCK_NUMS = 16;
constexpr uint32_t FP32_BYTES = 4;
constexpr uint32_t FP32_BLOCK_NUMS = 8;
constexpr uint32_t SHAPE_INFO = 32;
constexpr uint32_t C0_SIZE = 16;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t DB_NUM = 2;

constexpr uint32_t SAMEAB_S1_BASE = 512;
constexpr uint32_t SAMEAB_S2_BASE = 512;
constexpr uint32_t MATMUL_INPUT_NUMS = 2;
constexpr uint32_t WORKSPACE_BUFFER = 20 * 1024 * 1024;
constexpr uint32_t PSE_ALIBI_S2_LIMIT_SIZE = 1024;
constexpr uint32_t BIT_NUMS = 8;
constexpr uint32_t S2_NZ_SIZE = 128;
constexpr uint32_t S1_NZ_SIZE = 128;
constexpr uint32_t MM12_ND2NZ_SIZE = 5000;
const char* TEMPLATE_NAME_SAME_AB = "FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb";

bool FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::IsCapable()
{
    // fp32不支持
    if (fBaseParams.queryType == ge::DT_FLOAT) {
        OPS_LOG_I(context_, "FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb not support fp32.");
        return false;
    }
    // TND不支持
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        OPS_LOG_I(context_, "FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb only support TND inputformat.");
        return false;
    }

    return true;
}

bool FlashAttentionScoreGradTilingSameABDeterministic::IsCapable()
{
    OPS_LOG_D(context_, "Get deterministic flag is %d", context_->GetDeterministic());
    if (context_->GetDeterministic() != 1) {
        return false;
    }

    // fp32不支持
    OPS_ERR_IF(context_->GetInputDesc(QUERY) == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "InputDesc of query is nullptr."),
               return false);
    if (context_->GetInputDesc(QUERY)->GetDataType() == ge::DT_FLOAT) {
        OPS_LOG_I(context_, "FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb not support fp32.");
        return false;
    }

    return true;
}

uint64_t FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::GetTilingKey() const
{
    auto dtypeValue = DtypeEnum::FLOAT32;
    if (fBaseParams.mode == BF16) {
        dtypeValue = DtypeEnum::BFLOAT16;
    } else if (fBaseParams.mode == FP32) {
        dtypeValue = DtypeEnum::FLOAT32;
    } else {
        dtypeValue = DtypeEnum::FLOAT16_PRECISION;
    }

    auto attenMaskCfg = fBaseParams.attenMaskOptional == EMPTY_TENSOR ? OptionEnum::DISABLE : OptionEnum::ENABLE;
    LayoutEnum inputLayout = LayoutEnum::BSND;
    if (fBaseParams.layoutType == INPUT_FORMAT_BN2GS2D) {        // 0
        inputLayout = LayoutEnum::BNSD;                          // 2
    } else if (fBaseParams.layoutType == INPUT_FORMAT_S2BN2GD) { // 1
        inputLayout = LayoutEnum::SBND;                          // 1
    } else if (fBaseParams.layoutType == INPUT_FORMAT_BS2N2GD) { // 2
        inputLayout = LayoutEnum::BSND;                          // 0
    } else if (fBaseParams.layoutType == INPUT_FORMAT_TND) {     // 3
        inputLayout = LayoutEnum::TND;                           // 3
    }

    auto pseValue = fBaseParams.pseOptional == NORMAL_TENSOR ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto dropValue = fBaseParams.keepProb < 1 ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto mm1IsNZOut = fBaseParams.mm1IsNZOut ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto mm2IsNZOut = fBaseParams.mm2IsNZOut ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto isDeterministic = 0;
    if (context_->GetDeterministic() == 1) {
        isDeterministic = 1;
        // s小于基本块大小时，可以走回非确定性模板
        if (fBaseParams.s1 <= fBaseParams.s1CvInner &&
            fBaseParams.s2 <= fBaseParams.s2CvInner && fBaseParams.g == 1) {
            isDeterministic = 0;
        }
    }
    auto isSameAb = 1;

    uint64_t tilingKey = GET_TILINGKEY(AxisEnum::S2, AxisEnum::S1, AxisEnum::S2, dtypeValue, inputLayout,
            SparseEnum::ALL, dropValue, pseValue, attenMaskCfg, mm1IsNZOut, mm2IsNZOut, isDeterministic, isSameAb);
    OPS_LOG_I(context_, "FAGTiling sameAB DoTiling success, tiling is %lu.", tilingKey);
    return tilingKey;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::GetPlatformInfo()
{
    // 待公共模板实现后，会删除该函数  直接继承基类
    uint32_t coreNum = CORE_INIT_NUM; // 40 is init core num

    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo *>(context_->GetCompileInfo());
        OPS_ERR_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_, "compile_info is null"),
                   return ge::GRAPH_FAILED);

        fBaseParams.coreNum = compileInfoPtr->aivNum;
        fBaseParams.aicNum = compileInfoPtr->aicNum;
        fBaseParams.ubSize = compileInfoPtr->ubSize;
        fBaseParams.l1Size = compileInfoPtr->l1Size;
        fBaseParams.l0aSize = compileInfoPtr->l0aSize;
        fBaseParams.l0cSize = compileInfoPtr->l0cSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();

        fBaseParams.coreNum = coreNum;
        fBaseParams.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, fBaseParams.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, fBaseParams.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, fBaseParams.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, fBaseParams.l0cSize);
    }
    OPS_ERR_IF((fBaseParams.coreNum == 0) || (fBaseParams.aicNum == 0),
                OPS_REPORT_VECTOR_INNER_ERR(context_, "num of coreNum(aivNum) is %ld, num of aicNum is %ld.",
                fBaseParams.coreNum, fBaseParams.aicNum),
                return ge::GRAPH_FAILED);

    fBaseParams.ubSize -= MATMUL_SIZE;
    OPS_ERR_IF(fBaseParams.ubSize <= 0,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "ubSize is invalid."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::SetQKVStartIdx() {
    tilingData.s1s2BNGS1S2BaseParams.set_qStartIdx(0);
    tilingData.s1s2BNGS1S2BaseParams.set_kvStartIdx(0);
    auto qStartIdxTensor = context_->GetOptionalInputTensor(Q_START_IDX);
    if (qStartIdxTensor == nullptr) {
        OPS_LOG_W(context_, "[%s]qStartIdxTensor is null pointer", TEMPLATE_NAME_SAME_AB);
        return;
    }
    auto &qStartIdxShape = qStartIdxTensor->GetShape().GetStorageShape();
    if (qStartIdxShape.GetDimNum() != 1) {
        OPS_LOG_W(context_, "[%s]qStartIdxShape is invalid %lu %ld", TEMPLATE_NAME_SAME_AB, qStartIdxShape.GetDimNum(),
                  qStartIdxShape.GetDim(0));
        return;
    }
    /* Get Data from tensor. */
    const int64_t *value = qStartIdxTensor->GetData<int64_t>();
    if (value == nullptr) {
        OPS_LOG_W(context_, "[%s]qStartIdxShape data is null pointer", TEMPLATE_NAME_SAME_AB);
        return;
    }
    fBaseParams.qStartIdx = value[0];

    auto kvStartIdxTensor = context_->GetOptionalInputTensor(KV_START_IDX);
    if (kvStartIdxTensor == nullptr) {
        OPS_LOG_W(context_, "[%s]kvStartIdxTensor is null pointer", TEMPLATE_NAME_SAME_AB);
        return;
    }
    auto &kvStartIdxShape = kvStartIdxTensor->GetShape().GetStorageShape();
    if (kvStartIdxShape.GetDimNum() != 1) {
        OPS_LOG_W(context_, "[%s]kvStartIdxShape is invalid %lu %ld", TEMPLATE_NAME_SAME_AB, kvStartIdxShape.GetDimNum(),
                  kvStartIdxShape.GetDim(0));
        return;
    }
    /* Get Data from tensor. */
    const int64_t *kvValue = kvStartIdxTensor->GetData<int64_t>();
    if (kvValue == nullptr) {
        OPS_LOG_W(context_, "[%s]qStartIdxShape data is null pointer", TEMPLATE_NAME_SAME_AB);
        return;
    }
    fBaseParams.kvStartIdx = kvValue[0];

    tilingData.s1s2BNGS1S2BaseParams.set_qStartIdx(fBaseParams.qStartIdx);
    tilingData.s1s2BNGS1S2BaseParams.set_kvStartIdx(fBaseParams.kvStartIdx);
}


bool FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::SetSparseParams()
{
    if (fBaseParams.sparseMode == PREFIX || fBaseParams.sparseMode == PREFIX_COMPRESS) {
        auto prefixNTensor = context_->GetOptionalInputTensor(PREFIX_N);
        if (prefixNTensor == nullptr) {
            OPS_LOG_W(context_, "FAG sameAB sparseMode is prefix, but prefixN tensor is null!");
            return false;
        }

        auto &prefixShape = prefixNTensor->GetShape().GetStorageShape();
        if (prefixShape.GetDimNum() != 1 || prefixShape.GetDim(0) != fBaseParams.b) {
            OPS_LOG_W(context_, "FAG sameAB sparseMode is prefix, but prefixshape size[%zu] or value is invalid!",
                      prefixShape.GetDimNum());
            return false;
        }

        const int64_t *value = prefixNTensor->GetData<int64_t>();
        if (value == nullptr) {
            OPS_LOG_W(context_, "FAG sameAB sparseMode is prefix, but prefixN data is null pointer!");
            return false;
        }
        const size_t shapeSize = prefixNTensor->GetShapeSize();
        for (size_t i = 0; i < shapeSize; i++) {
            fBaseParams.prefixN.push_back(value[i]);
        }

        if (fBaseParams.prefixN.size() == static_cast<uint64_t>(fBaseParams.b)) {
            return true;
        } else {
            OPS_LOG_W(context_, "FAG sameAB sparseMode is prefix, but prefixN size[%zu] or value is invalid!",
                      fBaseParams.prefixN.size());
            return false;
        }
    }

    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        OPS_LOG_D(context_, "in the TND scenario,isSparse is true by default");
        return true;
    }

    if (fBaseParams.sparseMode == ALL_MASK || fBaseParams.attenMaskOptional == EMPTY_TENSOR) {
        OPS_LOG_D(context_, "in the ALL_MASK or attenMask is none scenario, isSparse is false");
        return false;
    }

    // 兼容老版本，未配置sparseMode或配置sparseMode为0的处理
    if (fBaseParams.sparseMode == NO_MASK) {
        if (fBaseParams.s1 > fBaseParams.s1Token || fBaseParams.s2 > fBaseParams.s2Token) { // band场景，包含causal
            OPS_LOG_D(context_, "in the NONE_MASK  and token is band scenario,isSparse is true ");
            return true;
        } else {
            OPS_LOG_D(context_, "in the NONE_MASK  and token is not band scenario,isSparse is false");
            return false;
        }
    }

    if (fBaseParams.sparseMode == LEFT_UP_CAUSAL || fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL ||
        fBaseParams.sparseMode == RIGHT_DOWN_CASUAL_BAND || fBaseParams.sparseMode == BAND_LEFT_UP_CASUAL) {
        OPS_LOG_D(context_, "in the LEFT_UP_CAUSAL  or RIGHT_DOWN_CAUSAL scenario,isSparse is true");
        return true;
    }

    if (fBaseParams.sparseMode == BAND &&
        (fBaseParams.s1 > fBaseParams.s1Token || fBaseParams.s2 > fBaseParams.s2Token)) {
        OPS_LOG_D(context_, "in the BAND  and token is band scenario,isSparse is true ");
        return true;
    }

    OPS_LOG_D(context_, "no scenario is hit, isSparse is false ");
    return false;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::ProcessPseNormal(const char *inputLayout)
{
    auto pseShape = context_->GetOptionalInputShape(PSE_SHIFT);
    OPS_ERR_IF(pseShape == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "pseShape is nullptr."),
               return ge::GRAPH_FAILED);
    auto pseShapeDim = pseShape->GetStorageShape().GetDimNum();
    OPS_ERR_IF((pseShapeDim != PSE_NORMAL_SHAPE_DIM),
               OPS_REPORT_VECTOR_INNER_ERR(context_, "The shape of pse is not 4 dimensions, got %lu", pseShapeDim),
               return ge::GRAPH_FAILED);

    auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
    auto dim1 = pseShape->GetStorageShape().GetDim(DIM_1);
    auto dim2 = pseShape->GetStorageShape().GetDim(DIM_2);
    auto dim3 = pseShape->GetStorageShape().GetDim(DIM_3);

    bool isBN1S = (dim0 == fBaseParams.b && dim1 == fBaseParams.n1 && dim2 == 1 && dim3 == fBaseParams.s2);
    bool isBNSS = (dim0 == fBaseParams.b && dim1 == fBaseParams.n1 && dim2 == fBaseParams.s1 && dim3 == fBaseParams.s2);
    bool is1NSS = (dim0 == 1 && dim1 == fBaseParams.n1 && dim2 == fBaseParams.s1 && dim3 == fBaseParams.s2);
    bool isAlibiPse = (dim1 == fBaseParams.n1 && dim2 == MAX_BASIC_BLOCK_SIZE && dim3 == fBaseParams.s2);
    bool isPse = (fBaseParams.s1 == fBaseParams.s2 && fBaseParams.s1 >= MAX_BASIC_BLOCK_SIZE &&
                  fBaseParams.s1 <= fBaseParams.s1Token && fBaseParams.s2Token == 0);
    bool isTnd = strcmp(inputLayout, "TND") == 0;
    bool isTndPse = isTnd && fBaseParams.s1 <= fBaseParams.s1Token && fBaseParams.s2Token == 0;
    bool isAlibi1NHS = isPse && isAlibiPse && (dim0 == 1);
    bool isAlibiBNHS = isPse && isAlibiPse && (dim0 == fBaseParams.b);
    bool isTndAlibiPse1NHS = isTndPse && isAlibiPse && (dim0 == 1);
    bool isTndAlibiPseBNHS = isTndPse && isAlibiPse && (dim0 == fBaseParams.b);

    if (isTndAlibiPse1NHS) {
        fBaseParams.pseShapeType = PSE_SHAPE_TYPE_1NHS;
    } else if (isTndAlibiPseBNHS) {
        fBaseParams.pseShapeType = PSE_SHAPE_TYPE_BNHS;
    } else if (isBN1S && !isTnd) {
        fBaseParams.pseShapeType = PSE_SHAPE_TYPE_BN1S;
    } else if (isBNSS && !isTnd) {
        fBaseParams.pseShapeType = PSE_SHAPE_TYPE_BNSS;
    } else if (is1NSS && !isTnd) {
        fBaseParams.pseShapeType = PSE_SHAPE_TYPE_1NSS;
    } else if (isAlibi1NHS) {
        fBaseParams.pseShapeType = PSE_SHAPE_TYPE_1NHS;
    } else if (isAlibiBNHS) {
        fBaseParams.pseShapeType = PSE_SHAPE_TYPE_BNHS;
    } else {
        OPS_LOG_E(context_, "The shape of pse[%ld,%ld,%ld,%ld] is invalid or tocken[%ld,%ld] not casual", dim0, dim1,
                    dim2, dim3, fBaseParams.s1Token, fBaseParams.s2Token);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::ProcessPseInfo(const char *inputLayout)
{
    if (context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(PSETYPE)) {
        fBaseParams.pseType = *(context_->GetAttrs()->GetAttrPointer<int64_t>(PSETYPE)); //8
        if (fBaseParams.pseType < 0 || fBaseParams.pseType >= PSE_INVALID_TYPE) {
            OPS_LOG_E(context_, "FAG pseType %ld is invalid", fBaseParams.pseType);
            return ge::GRAPH_FAILED;
        }
    }

    auto pseShape = context_->GetOptionalInputShape(PSE_SHIFT);
    // 同样的疑问，这是标量tensor的判断方法
    if (pseShape == nullptr || pseShape->GetStorageShape().GetDimNum() == 0) {
        fBaseParams.pseOptional = EMPTY_TENSOR;
        return ge::GRAPH_SUCCESS;
    }

    fBaseParams.pseOptional = NORMAL_TENSOR;
    auto pse = context_->GetOptionalInputDesc(PSE_SHIFT);
    OPS_ERR_IF(pse == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "InputDesc of pse is nullptr."),
               return ge::GRAPH_FAILED);
    if (fBaseParams.pseType == PSE_OUTER_MUL_ADD_TYPE || fBaseParams.pseType == PSE_OUTER_ADD_MUL_TYPE) {
        OPS_ERR_IF(pse->GetDataType() != context_->GetInputDesc(QUERY)->GetDataType(),
                   OPS_REPORT_VECTOR_INNER_ERR(context_,
                                               "FAG invalid pse dtype[%s], should be same with query's dtype",
                                               ge::TypeUtils::DataTypeToSerialString(pse->GetDataType()).c_str()),
                   return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF(pse->GetDataType() != ge::DT_FLOAT,
                   OPS_REPORT_VECTOR_INNER_ERR(context_, "FAG invalid pse dtype[%s], should be ge::DT_FLOAT",
                                               ge::TypeUtils::DataTypeToSerialString(pse->GetDataType()).c_str()),
                   return ge::GRAPH_FAILED);
    }

    auto pseShapeDim = pseShape->GetStorageShape().GetDimNum();
    bool isTnd = strcmp(inputLayout, "TND") == 0;
    if (fBaseParams.pseType == PSE_INNER_MUL_ADD_TYPE ||
        fBaseParams.pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
        // 输入为[N1]或者[B, N1]
        if (pseShapeDim == PSE_DIM_NUM_1) {
            auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
            OPS_ERR_IF(dim0 != fBaseParams.n1,
                   OPS_REPORT_VECTOR_INNER_ERR(context_, "FAG invalid pse shape %ld, should be same with n1 %ld",
                   dim0, fBaseParams.n1),
                   return ge::GRAPH_FAILED);
            fBaseParams.pseShapeType = PSE_1_N2_G_SLOPE;
        } else if (pseShapeDim == PSE_DIM_NUM_2) {
            auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
            auto dim1 = pseShape->GetStorageShape().GetDim(DIM_1);
            OPS_ERR_IF(dim0 != fBaseParams.b || dim1 != fBaseParams.n1,
                   OPS_REPORT_VECTOR_INNER_ERR(context_,
                   "FAG invalid pse shape {%ld, %ld}, should be same with b n1 {%ld, %ld}",
                   dim0, dim1, fBaseParams.b, fBaseParams.n1),
                   return ge::GRAPH_FAILED);
            fBaseParams.pseShapeType = PSE_B_N2_G_SLOPE;
        } else {
            OPS_LOG_E(context_, "pse inner mode, unsupported pse shape");
            return ge::GRAPH_FAILED;
        }
    } else if (pseShapeDim == PSE_DIM_NUM_1 && isTnd) {
        auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
        bool isTndPseBN1S = isTnd && (dim0 == fBaseParams.t2 * fBaseParams.n1);
        bool isTndPseBNSS = isTnd && (dim0 == fBaseParams.sumS1S2Product * fBaseParams.n1);
        if (isTndPseBN1S) {
            fBaseParams.pseShapeType = PSE_SHAPE_TYPE_BN1S;
        } else if (isTndPseBNSS) {
            fBaseParams.pseShapeType = PSE_SHAPE_TYPE_BNSS;
        } else {
            OPS_LOG_E(context_, "pse outer mode, tnd, unsupported pse shape");
            return ge::GRAPH_FAILED;
        }
    } else {
        return ProcessPseNormal(inputLayout);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::CheckAttenMaskShape()
{
    auto attenMaskShape = context_->GetOptionalInputShape(ATTEN_MASK);
    OPS_ERR_IF(attenMaskShape == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "attenMaskShape is nullptr."),
               return ge::GRAPH_FAILED);
    auto storageShape = attenMaskShape->GetStorageShape();
    size_t dimNum = storageShape.GetDimNum();
    fBaseParams.attenMaskS2Size = storageShape.GetDim(dimNum - LAST_AXIS_IDX);
    fBaseParams.attenMaskS1Size = storageShape.GetDim(dimNum - SEC_LAST_AXIS_IDX);

    if (dimNum == ATTEN_MASK_DIM_LENGTH_2) {
        fBaseParams.attenMaskShapeType = ATTEN_MASK_SHAPE_TYPE_SS;
    } else if (dimNum == ATTEN_MASK_DIM_LENGTH_4) {
        auto dim0 = attenMaskShape->GetStorageShape().GetDim(ATTEN_MASK_SHAPE_DIMS_0);
        auto dim1 = attenMaskShape->GetStorageShape().GetDim(ATTEN_MASK_SHAPE_DIMS_1);
        if ((dim0 == fBaseParams.b) && (dim1 == fBaseParams.n2 * fBaseParams.g)) {
            fBaseParams.attenMaskShapeType = ATTEN_MASK_SHAPE_TYPE_BNSS;
        } else if ((dim0 == fBaseParams.b) && (dim1 == 1)) {
            fBaseParams.attenMaskShapeType = ATTEN_MASK_SHAPE_TYPE_B1SS;
        } else if ((dim0 == 1) && (dim1 == 1)) {
            fBaseParams.attenMaskShapeType = ATTEN_MASK_SHAPE_TYPE_SS;
        } else {
            OPS_LOG_E(context_, "dim value error, dim0 = %ld, dim1 = %ld", dim0, dim1);
            return ge::GRAPH_FAILED;
        }
    } else {
        OPS_LOG_E(context_, "dim num error, dimNum = %lu", dimNum);
        return ge::GRAPH_FAILED;
    }

    // check atten_mask shape when enable atten_mask_compress
    if (fBaseParams.attenMaskCompressMode == 0) {
        bool invalid = fBaseParams.attenMaskOptional != EMPTY_TENSOR &&
                       fBaseParams.layoutType != INPUT_FORMAT_TND &&
                       (fBaseParams.attenMaskS1Size != fBaseParams.s1 ||
                        fBaseParams.attenMaskS2Size != fBaseParams.s2);
        if (invalid) {
            OPS_LOG_E(context_, "atten mask shape [%ld,%ld] is invalid.", fBaseParams.attenMaskS1Size,
                      fBaseParams.attenMaskS2Size);
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (fBaseParams.attenMaskCompressMode == PREFIX_COMPRESS_MODE) {
        if (fBaseParams.attenMaskS1Size != PREFIX_COMPRESS_S1_SIZE ||
            fBaseParams.attenMaskS2Size != ATTEN_MASK_COMPRESS_LIMIT) {
            OPS_LOG_E(context_,
                      "atten mask shape for prefix compress mode is invalid, try setting it to [3072, 2048].");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (fBaseParams.attenMaskS1Size != fBaseParams.attenMaskS2Size) {
        OPS_LOG_E(context_, "atten mask shape is not square.");
        return ge::GRAPH_FAILED;
    }

    if (fBaseParams.attenMaskS2Size != ATTEN_MASK_COMPRESS_LIMIT) {
        OPS_LOG_E(context_, "atten mask shape is invalid, try setting it to [2048, 2048].");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::ProcessSparseModeInfo()
{
    // 新增SPARSE_MODE属性，上库兼容处理
    auto attrs = context_->GetAttrs();
    fBaseParams.sparseMode = NO_MASK;
    if (attrs->GetAttrNum() > static_cast<size_t>(SPARSE_MODE)) {
        fBaseParams.sparseMode = *(attrs->GetAttrPointer<int>(SPARSE_MODE)); // 7
        if (fBaseParams.sparseMode > BAND_LEFT_UP_CASUAL) {
            OPS_LOG_E(context_, "FAG sparseMode [%u] is invalid", fBaseParams.sparseMode);
            return ge::GRAPH_FAILED;
        }
    }

    if ((fBaseParams.layoutType != INPUT_FORMAT_TND) &&
        (fBaseParams.sparseMode == RIGHT_DOWN_CASUAL_BAND || fBaseParams.sparseMode == BAND_LEFT_UP_CASUAL)) {
        OPS_LOG_E(context_, " layout %u not support sparsemode %u", fBaseParams.layoutType, fBaseParams.sparseMode);
        return ge::GRAPH_FAILED;
    }

    fBaseParams.attenMaskCompressMode = 0;
    auto attenMaskShape = context_->GetOptionalInputShape(ATTEN_MASK);
    // 此处的GetDimNum是为了判断是否为空tensor？这个条件不能作为是否是空tensor的判断标准（这是标量tensor）
    if (attenMaskShape == nullptr || attenMaskShape->GetStorageShape().GetDimNum() == 0) {
        fBaseParams.attenMaskOptional = EMPTY_TENSOR;
        return ge::GRAPH_SUCCESS;
    }

    if (fBaseParams.sparseMode == LEFT_UP_CAUSAL) {
        fBaseParams.attenMaskCompressMode = LEFT_UP_CAUSAL_MODE;
    } else if (fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL) {
        fBaseParams.attenMaskCompressMode = RIGHT_DOWN_CAUSAL_MODE;
    } else if (fBaseParams.sparseMode == BAND) {
        fBaseParams.attenMaskCompressMode = BAND_EQUAL_S_MODE;
    } else if (fBaseParams.sparseMode == PREFIX_COMPRESS) {
        fBaseParams.attenMaskCompressMode = PREFIX_COMPRESS_MODE;
    }

    fBaseParams.attenMaskOptional = NORMAL_TENSOR;
    auto attenMask = context_->GetOptionalInputDesc(ATTEN_MASK);
    if (attenMask != nullptr) {
        auto attenMaskType = attenMask->GetDataType();
        OPS_ERR_IF(attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_UINT8,
                   OPS_REPORT_VECTOR_INNER_ERR(context_, "invalid attenMask dtype[%s], only support bool or uint8.",
                                               ge::TypeUtils::DataTypeToSerialString(attenMaskType).c_str()),
                   return ge::GRAPH_FAILED);
        fBaseParams.attenMaskDtype = ATTEN_MASK_TYPE_U8_BOOL;
    }

    if (CheckAttenMaskShape() != ge::GRAPH_SUCCESS) {
        PrintShapeInfo();
        return ge::GRAPH_FAILED;
    }

    fBaseParams.bandIdx = FindBandIdx();
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::PrintShapeInfo()
{
    OPS_LOG_I(context_,
              "FAG s1s2_bn2gs1s2 with shape b[%ld] n2[%ld] g[%ld] s1[%ld] s2[%ld] d[%ld] preToken[%ld] nextToken[%ld]!",
              fBaseParams.b, fBaseParams.n2, fBaseParams.g, fBaseParams.s1, fBaseParams.s2, fBaseParams.d,
              fBaseParams.s1Token, fBaseParams.s2Token);
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::GetBaseShapeInfo() {
    // 待公共模板实现后，会删除该函数  直接继承基类
    const gert::StorageShape *queryShape = context_->GetInputShape(QUERY); // [B, N2, G, S1, D]
    const gert::StorageShape *keyShape = context_->GetInputShape(KEY);     // [B, N2, 1, S2, D]
    const gert::StorageShape *valueShape = context_->GetInputShape(VALUE);
    const gert::StorageShape *dyShape = context_->GetInputShape(DY);

    int64_t headNum = *context_->GetAttrs()->GetAttrPointer<int>(HEAD_NUM);
    const char *inputLayout = context_->GetAttrs()->GetAttrPointer<char>(INPUT_LAYOUT);

    OPS_ERR_IF(headNum == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "headNum is 0."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(queryShape == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "queryShape is nullptr."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(keyShape == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "keyShape is nullptr."),
               return ge::GRAPH_FAILED);

    uint32_t dimSize = queryShape->GetStorageShape().GetDimNum();
    OPS_ERR_IF(static_cast<size_t>(dimSize) != strlen(inputLayout),
               OPS_REPORT_VECTOR_INNER_ERR(context_, "layout dims is not inputLayout's length."),
               return ge::GRAPH_FAILED);

    if (strcmp(inputLayout, "SBH") == 0) {
        OPS_LOG_D(context_, "inputLayout == SBH queryShape");
        fBaseParams.layoutType = INPUT_FORMAT_S2BN2GD;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(DIM_1);
        OPS_ERR_IF(keyShape->GetStorageShape().GetDim(DIM_2) == 0,
                  OPS_REPORT_VECTOR_INNER_ERR(context_, "dim N2 is 0."),
                  return ge::GRAPH_FAILED);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(DIM_2) / keyShape->GetStorageShape().GetDim(DIM_2);
        OPS_ERR_IF(fBaseParams.g == 0,
                   OPS_REPORT_VECTOR_INNER_ERR(context_, "g is 0"),
                   return ge::GRAPH_FAILED);
        fBaseParams.n2 = headNum / fBaseParams.g; // 跟se和mde讨论  按 headNum=n1 计算
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(DIM_0);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(DIM_2) / headNum; // H=N*D
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(DIM_0);
    } else if (strcmp(inputLayout, "BSH") == 0) {
        OPS_LOG_D(context_, "inputLayout == BSH queryShape");
        fBaseParams.layoutType = INPUT_FORMAT_BS2N2GD;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(DIM_0);
        OPS_ERR_IF(keyShape->GetStorageShape().GetDim(DIM_2) == 0,
                  OPS_REPORT_VECTOR_INNER_ERR(context_, "dim N2 is 0."),
                  return ge::GRAPH_FAILED);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(DIM_2) / keyShape->GetStorageShape().GetDim(DIM_2);
        OPS_ERR_IF(fBaseParams.g == 0,
                   OPS_REPORT_VECTOR_INNER_ERR(context_, "g is 0"),
                   return ge::GRAPH_FAILED);
        fBaseParams.n2 = headNum / fBaseParams.g;
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(DIM_1);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(DIM_2) / headNum; // H=N*D
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(DIM_1);
    } else if (strcmp(inputLayout, "BNSD") == 0) {
        OPS_LOG_D(context_, "inputLayout == BNSD queryShape");
        fBaseParams.layoutType = INPUT_FORMAT_BN2GS2D;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(DIM_0);
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(DIM_1);
        OPS_ERR_IF(keyShape->GetStorageShape().GetDim(DIM_1) == 0,
                  OPS_REPORT_VECTOR_INNER_ERR(context_, "dim N2 is 0."),
                  return ge::GRAPH_FAILED);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(DIM_1) / keyShape->GetStorageShape().GetDim(DIM_1);
        OPS_ERR_IF(fBaseParams.g == 0,
                   OPS_REPORT_VECTOR_INNER_ERR(context_, "g is 0"),
                   return ge::GRAPH_FAILED);
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(DIM_2);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(DIM_3);
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(DIM_2);
        OPS_LOG_D(context_, "inputLayout == BNSD queryShape" "%ld, %ld, %ld, %ld,",
                  queryShape->GetStorageShape().GetDim(DIM_0), queryShape->GetStorageShape().GetDim(DIM_1),
                  queryShape->GetStorageShape().GetDim(DIM_2), queryShape->GetStorageShape().GetDim(DIM_3));
    } else if (strcmp(inputLayout, "TND") == 0) {
        OPS_LOG_D(context_, "inputLayout is TND");
        fBaseParams.layoutType = INPUT_FORMAT_TND;
        auto actualSeqQlenTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_LEN);
        auto actualSeqKvlenTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_KV_LEN);
        if (actualSeqQlenTensor == nullptr || actualSeqKvlenTensor == nullptr) {
            OPS_LOG_E(context_, "actualSeqQlenTensor or actualSeqKvlenTensor is nullptr");
            return ge::GRAPH_FAILED;
        }
        const size_t seqQShapeSize = actualSeqQlenTensor->GetShapeSize();
        const size_t kvSeqShapeSize = actualSeqKvlenTensor->GetShapeSize();
        OPS_ERR_IF((seqQShapeSize != kvSeqShapeSize),
            OPS_REPORT_VECTOR_INNER_ERR(context_, "actualSeqQlenTensor shapeSize is not equal actualSeqKvlenTensor"),
            return ge::GRAPH_FAILED);

        const int64_t *qValue = actualSeqQlenTensor->GetData<int64_t>();
        const int64_t *kvValue = actualSeqKvlenTensor->GetData<int64_t>();
        OPS_ERR_IF(
            (qValue == nullptr || kvValue == nullptr),
            OPS_REPORT_VECTOR_INNER_ERR(context_, "qValue or kvValue is nullptr."),
            return ge::GRAPH_FAILED);
        for (size_t i = 0; i < seqQShapeSize; i++) {
            if (i == 0) {
                fBaseParams.actualSeqQlen.push_back(qValue[i]);
                fBaseParams.actualSeqKvlen.push_back(kvValue[i]);
            } else {
                fBaseParams.actualSeqQlen.push_back(qValue[i] - qValue[i - 1]);
                fBaseParams.actualSeqKvlen.push_back(kvValue[i] - kvValue[i - 1]);
            }
            fBaseParams.sumS1S2Product += fBaseParams.actualSeqQlen[i] * fBaseParams.actualSeqKvlen[i];
        }

        fBaseParams.b = seqQShapeSize;
        fBaseParams.t1 = qValue[seqQShapeSize - 1];
        fBaseParams.t2 = kvValue[seqQShapeSize - 1];
        fBaseParams.s1 = *std::max_element(fBaseParams.actualSeqQlen.begin(), fBaseParams.actualSeqQlen.end());
        fBaseParams.s2 = *std::max_element(fBaseParams.actualSeqKvlen.begin(), fBaseParams.actualSeqKvlen.end());
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(DIM_1);
        OPS_ERR_IF(keyShape->GetStorageShape().GetDim(DIM_1) == 0,
                  OPS_REPORT_VECTOR_INNER_ERR(context_, "dim N2 is 0."),
                  return ge::GRAPH_FAILED);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(DIM_1) / keyShape->GetStorageShape().GetDim(DIM_1);
        OPS_ERR_IF(fBaseParams.g == 0,
                   OPS_REPORT_VECTOR_INNER_ERR(context_, "g is 0"),
                   return ge::GRAPH_FAILED);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(DIM_2);
    } else if (strcmp(inputLayout, "BSND") == 0) {
        OPS_LOG_D(context_, "inputLayout == BSND queryShape");
        // inputLayout = "BSND"
        fBaseParams.layoutType = INPUT_FORMAT_BS2N2GD;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(DIM_0);
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(DIM_2);
        OPS_ERR_IF(keyShape->GetStorageShape().GetDim(DIM_2) == 0,
                  OPS_REPORT_VECTOR_INNER_ERR(context_, "dim N2 is 0."),
                  return ge::GRAPH_FAILED);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(DIM_2) / keyShape->GetStorageShape().GetDim(DIM_2);
        OPS_ERR_IF(fBaseParams.g == 0,
                   OPS_REPORT_VECTOR_INNER_ERR(context_, "g is 0"),
                   return ge::GRAPH_FAILED);
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(DIM_1);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(DIM_3);
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(DIM_1);
    } else {
        OPS_LOG_E(context_, "FAG inputLayout is invalid");
        return ge::GRAPH_FAILED;
    }
    OPS_ERR_IF(fBaseParams.n2 == 0,
            OPS_REPORT_VECTOR_INNER_ERR(context_, "n2 is 0."),
            return ge::GRAPH_FAILED);

    auto ret = IsSameShape(queryShape, dyShape);
    OPS_ERR_IF(!ret,
                OPS_REPORT_VECTOR_INNER_ERR(context_, "FAG different shape queryShape and dyShape"),
                return ge::GRAPH_FAILED);
    ret = IsSameShape(keyShape, valueShape);
    OPS_ERR_IF(!ret,
                OPS_REPORT_VECTOR_INNER_ERR(context_, "FAG different shape keyShape and valueShape"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::ProcessDropInfo(const char *inputLayout) {
    auto dropMask = context_->GetOptionalInputDesc(DROP_MASK);
    OPS_ERR_IF(dropMask == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_,
                "FAG keepProb [%f], dropMask can not be nullptr", fBaseParams.keepProb),
                return ge::GRAPH_FAILED);

    auto dropMaskType = dropMask->GetDataType();
    OPS_ERR_IF(dropMaskType != ge::DT_UINT8,
                OPS_REPORT_VECTOR_INNER_ERR(context_, "FAG invalid dropMask dtype[%s], only support uint8.",
                ge::TypeUtils::DataTypeToSerialString(dropMaskType).c_str()),
                return ge::GRAPH_FAILED);

    auto dropMaskShape = context_->GetOptionalInputShape(DROP_MASK);
    OPS_ERR_IF(dropMaskShape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_,
                "FAG keepProb [%f], dropMaskShape can not be nullptr", fBaseParams.keepProb),
                return ge::GRAPH_FAILED);

    int64_t dropMaskDim = dropMaskShape->GetStorageShape().GetDimNum();
    int64_t dropMaskShapeSize = 1;
    for (int64_t i = 0; i < dropMaskDim; ++i) {
        int64_t dimValue = dropMaskShape->GetStorageShape().GetDim(i);
        dropMaskShapeSize *= dimValue;
    }

    auto shapeSize = AlignUp(fBaseParams.dropMaskSize, static_cast<int64_t>(BIT_NUMS)) / BIT_NUMS;
    if (dropMaskShapeSize < shapeSize) {
        OPS_LOG_E(context_, "FAG Input dropMask shapeSize is invalid, it should not be less than %ld, but got %ld",
                  shapeSize, dropMaskShapeSize);
        return ge::GRAPH_FAILED;
    }

    if (strcmp(inputLayout, "TND") == 0) {
        for (int64_t i = 0; i < fBaseParams.b; i++) {
            if (fBaseParams.actualSeqKvlen[i] % BIT_NUMS != 0) {
                fBaseParams.dropoutIsDivisibleBy8 = 0;
                break;
            }
        }
    } else {
        if (fBaseParams.s2 % BIT_NUMS != 0) {
            fBaseParams.dropoutIsDivisibleBy8 = 0;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::GetShapeAttrsInfo()
{
    OPS_ERR_IF(context_ == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "context is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetAttrs() == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "GetAttrs is nullptr."),
               return ge::GRAPH_FAILED);

    auto ret = GetBaseShapeInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        PrintShapeInfo();
        return ret;
    }

    const char *inputLayout = context_->GetAttrs()->GetAttrPointer<char>(INPUT_LAYOUT);

    fBaseParams.n1 = fBaseParams.n2 * fBaseParams.g;
    fBaseParams.s1Align = (fBaseParams.s1 + INPUT_ALIGN - 1) / INPUT_ALIGN * INPUT_ALIGN;
    fBaseParams.s2Align = (fBaseParams.s2 + INPUT_ALIGN - 1) / INPUT_ALIGN * INPUT_ALIGN;

    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        fBaseParams.qSize = fBaseParams.t1 * fBaseParams.n2 * fBaseParams.g * fBaseParams.d;
        fBaseParams.kvSize = fBaseParams.t2 * fBaseParams.n2 * 1 * fBaseParams.d;
        fBaseParams.dropMaskSize = fBaseParams.n2 * fBaseParams.g * fBaseParams.sumS1S2Product;
    } else {
        fBaseParams.qSize = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.d;
        fBaseParams.kvSize = fBaseParams.b * fBaseParams.n2 * 1 * fBaseParams.s2 * fBaseParams.d;
        fBaseParams.dropMaskSize = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s2 * fBaseParams.s1;
    }

    // mBaseParams is used for matmal tiling module
    OPS_ERR_IF(context_->GetInputDesc(QUERY) == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "InputDesc of query is nullptr."),
               return ge::GRAPH_FAILED);
    auto queryType = context_->GetInputDesc(QUERY)->GetDataType();
    fBaseParams.queryType = queryType;
    fBaseParams.isBf16 = queryType == ge::DT_BF16 ? true : false;
    if (queryType == ge::DT_FLOAT) {
        fBaseParams.dataTypeSize = FP32_BYTES; // init date type fp32 is 4
        fBaseParams.dataBlockNum = FP32_BLOCK_NUMS;
        fBaseParams.calTypeSize = FP32_BYTES; // init cal type fp32 is 4
        fBaseParams.calBlockNum = FP32_BLOCK_NUMS;
    } else {
        fBaseParams.dataTypeSize = FP16_BYTES;
        fBaseParams.dataBlockNum = FP16_BLOCK_NUMS;
        fBaseParams.calTypeSize = FP32_BYTES;
        fBaseParams.calBlockNum = FP32_BLOCK_NUMS;
    }

    fBaseParams.mm1IsNZOut = false;
    fBaseParams.mm2IsNZOut = false;
    if (queryType != ge::DT_FLOAT) {
        bool mm1NzRangeS2 = (fBaseParams.s2 % S2_NZ_SIZE != 0 && fBaseParams.s2 < MM12_ND2NZ_SIZE);
        fBaseParams.mm1IsNZOut = (fBaseParams.layoutType != INPUT_FORMAT_TND) && mm1NzRangeS2;
        // d为72, 80, 88, 96时支持NZ输出
        bool mm2NzRangeD = ((fBaseParams.d == 72) || (fBaseParams.d == 80) || (fBaseParams.d == 88) || (fBaseParams.d == 96));
        fBaseParams.mm2IsNZOut = (mm2NzRangeD && fBaseParams.s1 >= S1_NZ_SIZE);
    }
    fBaseParams.dataBlockNum = BYTE_BLOCK / fBaseParams.dataTypeSize;
    fBaseParams.calBlockNum = BYTE_BLOCK / fBaseParams.calTypeSize;

    // nz out
    int64_t qSizeAlign = fBaseParams.qSize;
    int64_t kvSizeAlign = fBaseParams.kvSize;
    if (fBaseParams.mm2IsNZOut) {
        qSizeAlign = fBaseParams.qSize / fBaseParams.d * ((fBaseParams.d + C0_SIZE - 1) / C0_SIZE * C0_SIZE);
        kvSizeAlign = fBaseParams.kvSize / fBaseParams.d * ((fBaseParams.d + C0_SIZE - 1) / C0_SIZE * C0_SIZE);
    }
    fBaseParams.qSizeAlign = qSizeAlign;
    fBaseParams.kvSizeAlign = kvSizeAlign;

    fBaseParams.scaleValue = *(context_->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE));
    fBaseParams.keepProb = *(context_->GetAttrs()->GetAttrPointer<float>(KEEP_PROB));
    OPS_ERR_IF((fBaseParams.keepProb <= 0 || fBaseParams.keepProb > 1),
                OPS_REPORT_VECTOR_INNER_ERR(context_, "keepProb is illegal."),
                return ge::GRAPH_FAILED);

    fBaseParams.dropoutIsDivisibleBy8 = 1;

    if (fBaseParams.keepProb < 1) {
        ret = ProcessDropInfo(inputLayout);
        if (ret != ge::GRAPH_SUCCESS) {
            PrintShapeInfo();
            return ret;
        }
    }

    // token_info
    fBaseParams.s1Token = *(context_->GetAttrs()->GetAttrPointer<int64_t>(PRE_TOKENS));
    fBaseParams.s2Token = *(context_->GetAttrs()->GetAttrPointer<int64_t>(NEXT_TOKENS));

    ret = ProcessSparseModeInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        PrintShapeInfo();
        return ret;
    }

    ret = ProcessTokensInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        PrintShapeInfo();
        return ret;
    }

    ret = ProcessPseInfo(inputLayout);
    if (ret != ge::GRAPH_SUCCESS) {
        PrintShapeInfo();
        return ret;
    }

    ret = CheckDtypeValid(context_);
    OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "dtype is invalid."),
               return ge::GRAPH_FAILED);

    fBaseParams.isSparse = SetSparseParams();
    OPS_LOG_D(context_, "FAG sameAB sparse mode = %u, sparse %s.", fBaseParams.sparseMode,
              fBaseParams.isSparse ? "enable" : "disable");

    return fBaseParams.layoutType == INPUT_FORMAT_TND ?
               CheckTndShapeValid(context_, fBaseParams.t1, fBaseParams.n1, fBaseParams.d) :
               CheckShapeValid(context_, fBaseParams.b, fBaseParams.n1, fBaseParams.s1, fBaseParams.d);
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::DoOpTiling()
{
    auto ret = DoSplit();
    OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
               OPS_LOG_W(context_, "get DoSplit fail."),
               return ret);

    ret = DoPreTiling();
    OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
               OPS_LOG_W(context_, "get DoPreTiling fail."),
               return ret);

    SetQKVStartIdx();
    DoPreSfmgTiling();
    ret = DoPostTiling();
    OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
               OPS_LOG_W(context_, "get DoPostTiling fail."),
               return ret);
    DetermineMode();

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::AdjustCvInner()
{
    // calculate best cvInner
    if (fBaseParams.isSparse == false && fBaseParams.s2 < 4 * SAMEAB_S1_BASE) {
        if (fBaseParams.d == 64 && fBaseParams.s1 >= fBaseParams.s1CvInner && fBaseParams.s2 >= fBaseParams.s2CvInner) {
            // D=64时，mmbaseN=256, s1/s2足够大，不需要调整基本块
            return;
        }
        // fuzzy for base cvInner
        uint32_t baseS1;
        uint32_t baseS2;
        float threshold = 0.50;
        uint32_t step = 16;
        constexpr uint32_t maxBaseBlockSize = SAMEAB_S1_BASE * SAMEAB_S2_BASE;

        float s2FloatRatio = static_cast<float>(fBaseParams.s2) / static_cast<float>(SAMEAB_S2_BASE);
        uint32_t s2Ratio = static_cast<int>(s2FloatRatio);
        if (s2FloatRatio - s2Ratio > threshold) {
            s2Ratio += 1;
        }
        s2Ratio = s2Ratio < 1 ? 1 : s2Ratio;
        baseS2 = (fBaseParams.s2 + s2Ratio - 1) / s2Ratio;
        baseS2 = (baseS2 + step - 1) / step * step;
        baseS1 = (maxBaseBlockSize / baseS2) / step * step;
        baseS1 = baseS1 > 2 * SAMEAB_S1_BASE ? 2 * SAMEAB_S1_BASE : baseS1;

        int64_t alignS1 = (fBaseParams.s1 + step - 1) / step * step;
        if (static_cast<int64_t>(baseS1) > alignS1) {
            baseS1 = alignS1;
            uint32_t baseS2Resize = (maxBaseBlockSize / baseS1) / step * step;
            baseS2 = baseS2Resize > baseS2 ? baseS2Resize : baseS2;
        }
        baseS2 = baseS2 > 2 * SAMEAB_S2_BASE ? 2 * SAMEAB_S2_BASE : baseS2;

        fBaseParams.s1CvInner = baseS1;
        fBaseParams.s2CvInner = baseS2;
    }
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::DoSplit()
{
    fBaseParams.s1CvInner = SAMEAB_S1_BASE;
    fBaseParams.s2CvInner = SAMEAB_S2_BASE;

    AdjustCvInner();

    OPS_LOG_D(context_, "FAGTiling sameAB s1CvInner is %u s2CvInner is %u.", fBaseParams.s1CvInner, fBaseParams.s2CvInner);

    uint32_t s1Inner = INITIAL_S1_SPLIT_NUM;
    uint32_t s2Inner = INITIAL_S2_SPLIT_NUM;

    uint32_t tmpBufferSize =
        (fBaseParams.ubSize - s1Inner * s2Inner * BASIC_BLOCK_MULTIPLE - s1Inner * SHAPE_INFO * fBaseParams.calTypeSize) /
        BYTE_BLOCK * BYTE_BLOCK;
    if (fBaseParams.mm1IsNZOut) {
        tmpBufferSize = tmpBufferSize - TEMP_BUFFER_REMAIN_SIZE;
    }
    fBaseParams.tmpBufferSize = tmpBufferSize;

    uint32_t s1CvInner = fBaseParams.s1CvInner;
    OPS_ERR_IF(s1CvInner == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "divisor s1CvInner is 0."),
               return ge::GRAPH_FAILED);
    int64_t s1Outer = (fBaseParams.s1 + s1CvInner - 1) / s1CvInner;
    uint32_t s1TailTmp = fBaseParams.s1 % s1Inner;
    uint32_t s1CvTailTmp = fBaseParams.s1 % s1CvInner;
    fBaseParams.s1Tail = s1TailTmp == 0 ? s1Inner : s1TailTmp;
    fBaseParams.s1CvTail = s1CvTailTmp == 0 ? s1CvInner : s1CvTailTmp;
    fBaseParams.s1Inner = s1Inner;
    fBaseParams.s1Outer = s1Outer;

    uint32_t cvS2Inner = fBaseParams.s2CvInner;
    OPS_ERR_IF(cvS2Inner == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "divisor cvS2Inner is 0."),
               return ge::GRAPH_FAILED);
    int64_t s2Outer = (fBaseParams.s2 + cvS2Inner - 1) / cvS2Inner;
    uint32_t s2TailTmp = fBaseParams.s2 % s2Inner;
    uint32_t s2CvTailTmp = fBaseParams.s2 % cvS2Inner;
    fBaseParams.s2Tail = s2TailTmp == 0 ? s2Inner : s2TailTmp;
    fBaseParams.s2CvTail = s2CvTailTmp == 0 ? cvS2Inner : s2CvTailTmp;
    fBaseParams.s2Outer = s2Outer;
    fBaseParams.s2Inner = s2Inner;

    fBaseParams.baseMN = s1Inner * s2Inner;

    OPS_ERR_IF(
        (fBaseParams.baseMN == 0 || fBaseParams.s2Outer == 0 || fBaseParams.s1Outer == 0),
        OPS_REPORT_VECTOR_INNER_ERR(context_, "baseMN or s2Outer or s1Outer is 0."),
        return ge::GRAPH_FAILED);

    int64_t fusedOuter = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer;
    int64_t blockFactor = (fusedOuter + fBaseParams.coreNum - 1) / fBaseParams.coreNum;

    fBaseParams.blockOuter = fBaseParams.coreNum;
    fBaseParams.blockFactor = blockFactor;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::DoLibApiTiling()
{
    // mm tiling
    matmul_tiling::MatmulApiTiling mm1;
    matmul_tiling::MatmulApiTiling mm2;
    matmul_tiling::MatmulApiTiling mm3;
    matmul_tiling::DataType inputAType = matmul_tiling::DataType::DT_FLOAT;
    if(fBaseParams.mode == FP32) {
        inputAType = matmul_tiling::DataType::DT_FLOAT;
    } else {
        inputAType = matmul_tiling::DataType::DT_BFLOAT16;
    }
    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType,
                 false);
    mm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType,
                 true);
    mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    int64_t mmS1 = fBaseParams.s1;
    if (fBaseParams.mm1IsNZOut) {
        mmS1 = fBaseParams.s1CvInner;
    }
    // format left[B, N2, G, S1, D] right[B, N2, 1, S2, D] result[B, N2, G, S1, S2]
    if (fBaseParams.layoutType == INPUT_FORMAT_BN2GS2D) {
        mm1.SetOrgShape(mmS1, fBaseParams.s2, fBaseParams.d);
    } else if (fBaseParams.layoutType == INPUT_FORMAT_S2BN2GD) {
        mm1.SetOrgShape(mmS1, fBaseParams.s2,
                        static_cast<int64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.g * fBaseParams.d,
                        static_cast<int64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.d);
    } else if (fBaseParams.layoutType == INPUT_FORMAT_BS2N2GD || fBaseParams.layoutType == INPUT_FORMAT_TND) {
        mm1.SetOrgShape(mmS1, fBaseParams.s2, static_cast<int64_t>(fBaseParams.n2) * fBaseParams.g * fBaseParams.d,
                        static_cast<int64_t>(fBaseParams.n2) * fBaseParams.d);
    }

    mm1.SetShape(fBaseParams.s1CvInner, fBaseParams.s2CvInner, fBaseParams.d);
    mm1.SetBias(false);
    uint32_t baseBlockSize = 128;
    uint32_t baseM = std::min(fBaseParams.s1CvInner, baseBlockSize);
    uint32_t baseN = fBaseParams.d > 64 ? 128 : 256;
    baseN = std::min(fBaseParams.s2CvInner, baseN);
    mm1.SetFixSplit(baseM, baseN, -1);
    OPS_ERR_IF(mm1.GetTiling(tilingData.mm1TilingData) != 0,
              OPS_REPORT_VECTOR_INNER_ERR(context_, "matmul1 tilingData get fail."),
              return ge::GRAPH_FAILED);
    SetMatmulTilingBufferInfo(tilingData.mm1TilingData);

    // format left[B, N2, G, S1, S2] right[B, N2, G, S1, D] result[B, N2, G, S2, D]
    // matmal3/5
    mm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType,
                 true);
    mm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType,
                 false);
    auto outFormat = fBaseParams.mm2IsNZOut ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND;
    mm2.SetCType(matmul_tiling::TPosition::GM, outFormat, matmul_tiling::DataType::DT_FLOAT);
    if (fBaseParams.layoutType == INPUT_FORMAT_BN2GS2D) {
        // M/N/K
        mm2.SetOrgShape(fBaseParams.s2, fBaseParams.d, fBaseParams.s1);
    } else if (fBaseParams.layoutType == INPUT_FORMAT_S2BN2GD) {
        mm2.SetOrgShape(fBaseParams.s2,
                        static_cast<int64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.g * fBaseParams.d,
                        fBaseParams.s1, fBaseParams.s1);
    } else if (fBaseParams.layoutType == INPUT_FORMAT_BS2N2GD || fBaseParams.layoutType == INPUT_FORMAT_TND) {
        mm2.SetOrgShape(fBaseParams.s2, static_cast<int64_t>(fBaseParams.n2) * fBaseParams.g * fBaseParams.d,
                        fBaseParams.s1, fBaseParams.s1);
    }
    mm2.SetShape(fBaseParams.s2CvInner, fBaseParams.d, fBaseParams.s1CvInner);
    mm2.SetBias(false);
    uint32_t mm2BaseN = std::min(static_cast<uint32_t>(256), static_cast<uint32_t>((fBaseParams.d + 15) / 16 * 16));
    mm2.SetFixSplit(-1, mm2BaseN, -1);
    OPS_ERR_IF(mm2.GetTiling(tilingData.mm2TilingData) != 0,
              OPS_REPORT_VECTOR_INNER_ERR(context_, "matmul2 tilingData get fail."),
              return ge::GRAPH_FAILED);
    SetMatmulTilingBufferInfo(tilingData.mm2TilingData);

    // format left[B, N2, G, S1, S2] right[B, N2, 1, S2, D] result[B, N2, G, S1, D]
    // matmal4
    mm3.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType,
                 false);
    mm3.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType,
                 false);
    mm3.SetCType(matmul_tiling::TPosition::GM, outFormat, matmul_tiling::DataType::DT_FLOAT);
    if (fBaseParams.layoutType == INPUT_FORMAT_BN2GS2D) {
        // M/N/K
        mm3.SetOrgShape(fBaseParams.s1, fBaseParams.d, fBaseParams.s2);
    } else if (fBaseParams.layoutType == INPUT_FORMAT_S2BN2GD) {
        mm3.SetOrgShape(fBaseParams.s1, static_cast<int64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.d,
                        fBaseParams.s2, fBaseParams.s2);
    } else if (fBaseParams.layoutType == INPUT_FORMAT_BS2N2GD || fBaseParams.layoutType == INPUT_FORMAT_TND) {
        mm3.SetOrgShape(fBaseParams.s1, fBaseParams.n2 * fBaseParams.d, fBaseParams.s2, fBaseParams.s2);
    }
    mm3.SetShape(fBaseParams.s1CvInner, fBaseParams.d, fBaseParams.s2CvInner);
    mm3.SetBias(false);
    uint32_t mm3BaseN = std::min(baseBlockSize, static_cast<uint32_t>((fBaseParams.d + 15) / 16 * 16));
    mm3.SetFixSplit(baseM, mm3BaseN, -1);
    OPS_ERR_IF(mm3.GetTiling(tilingData.mm3TilingData) != 0,
              OPS_REPORT_VECTOR_INNER_ERR(context_, "matmul3 tilingData get fail."),
              return ge::GRAPH_FAILED);
    SetMatmulTilingBufferInfo(tilingData.mm3TilingData);

    // 这里是否需要修改为与kernel一致
    uint32_t cvS2Inner = fBaseParams.s2CvInner;
    uint32_t s2VSize = cvS2Inner > 256 ? 256 : cvS2Inner;
    uint32_t s1VecSize =
        std::min(((INITIAL_S1_SPLIT_NUM * INITIAL_S2_SPLIT_NUM + s2VSize - 1) / s2VSize), fBaseParams.s1Inner);

    auto softmaxShape = ge::Shape({s1VecSize, s2VSize});
    AscendC::SoftMaxTilingFunc(softmaxShape, fBaseParams.calTypeSize, fBaseParams.tmpBufferSize,
                               tilingData.softmaxTilingData);

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::SetMatmulTilingBufferInfo(TCubeTiling &mmTiling)
{
    mmTiling.set_shareMode(0);
    mmTiling.set_shareL1Size(fBaseParams.l1Size);
    mmTiling.set_shareL0CSize(fBaseParams.l0cSize);
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "GetWorkspaceSizes is nullptr."),
               return ge::GRAPH_FAILED);
    size_t workspaceSize = MUL_CORE_SYNC_BUFFER;
    uint32_t s1Inner = std::min(INITIAL_S1_SPLIT_NUM, fBaseParams.s1Align);

    // matmal3 q
    workspaceSize =
        (workspaceSize + static_cast<size_t>(fBaseParams.qSizeAlign) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 k
    workspaceSize =
        (workspaceSize + static_cast<size_t>(fBaseParams.kvSizeAlign) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 v
    workspaceSize =
        (workspaceSize + static_cast<size_t>(fBaseParams.kvSizeAlign) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // mask bool workspace size
    if (fBaseParams.dropoutIsDivisibleBy8 == 0) {
        workspaceSize =
            (workspaceSize + static_cast<size_t>(fBaseParams.dropMaskSize) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }
    // sfmg workspace
    workspaceSize =
        workspaceSize + static_cast<size_t>(AlignTo(fBaseParams.sfmgNormalAxisSize * 8 * FP32_BYTES,
                                                    static_cast<int64_t>(GM_ALIGN)));

    // matmal1/matmal2 workspace size
    size_t vectorCoreNum = fBaseParams.coreNum;
    workspaceSize = (workspaceSize +
                     vectorCoreNum * fBaseParams.s1CvInner * fBaseParams.s2CvInner * FP32_BYTES *
                         MATMUL_INPUT_NUMS +
                     GM_ALIGN) /
                    GM_ALIGN * GM_ALIGN;

    workspaceSize += WORKSPACE_BUFFER;
    workspaces[0] = workspaceSize;

    if (fBaseParams.pseType == PSE_INNER_MUL_ADD_TYPE ||
        fBaseParams.pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
        fBaseParams.pseAlibiBaseS2 = PSE_ALIBI_S2_LIMIT_SIZE;
        int64_t s2Tail = fBaseParams.s2 % PSE_ALIBI_S2_LIMIT_SIZE;
        if (s2Tail != 0) {
            fBaseParams.pseAlibiBaseS1 = std::min(static_cast<int64_t>(s1Inner),
                                                  UB_BASIC_LIMIT_SIZE / AlignUp(s2Tail, FRACTAL_NUM));
        } else {
            fBaseParams.pseAlibiBaseS1 = std::min(static_cast<int64_t>(s1Inner),
                                                  UB_BASIC_LIMIT_SIZE / fBaseParams.pseAlibiBaseS2);
        }
        fBaseParams.pseAlibiBaseS1 = std::max(fBaseParams.pseAlibiBaseS1, UB_BASIC_LIMIT_SIZE / s1Inner);
        if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
            if (fBaseParams.s2 > PSE_ALIBI_S2_LIMIT_SIZE) {
                fBaseParams.pseAlibiBaseS2 = PSE_ALIBI_S2_LIMIT_SIZE;
            } else {
                fBaseParams.pseAlibiBaseS2 = fBaseParams.s2Align;
            }
            fBaseParams.pseAlibiBaseS1 = s1Inner;
        }
        int64_t pseAlibiBytes = AlignUp(fBaseParams.pseAlibiBaseS2 * fBaseParams.pseAlibiBaseS1 * 2, GM_ALIGN) * fBaseParams.coreNum;
        workspaces[0] += pseAlibiBytes;
    }
    // 确定性计算
    if (context_->GetDeterministic() == 1) {
        int64_t dAlign = (fBaseParams.d + FP16_BLOCK_NUMS - 1) / FP16_BLOCK_NUMS * FP16_BLOCK_NUMS;
        int64_t cvS2Inner = fBaseParams.s2CvInner;
        workspaces[0] += AlignUp(fBaseParams.s1CvInner * dAlign * FP32_BYTES * fBaseParams.coreNum / 2 * DB_NUM, GM_ALIGN);
        workspaces[0] += AlignUp(cvS2Inner * dAlign * FP32_BYTES * fBaseParams.coreNum / 2 * DB_NUM, GM_ALIGN);
        workspaces[0] += AlignUp(cvS2Inner * dAlign * FP32_BYTES * fBaseParams.coreNum / 2 * DB_NUM, GM_ALIGN);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::PostTiling()
{
    SaveToTilingData();

    auto blockdim = CalcTschBlockDim(fBaseParams.coreNum, fBaseParams.aicNum,
                                     fBaseParams.coreNum);
    OPS_ERR_IF(blockdim == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context_,
                                           "blockdim is 0, aicNum is %ld, aivNum is %ld.", fBaseParams.aicNum,
                                           fBaseParams.coreNum),
               return ge::GRAPH_FAILED);
    context_->SetBlockDim(blockdim);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::SaveToTilingData()
{
    tilingData.s1s2BNGS1S2BaseParams.set_coreNum(fBaseParams.coreNum);

    // set tilingdata baseinfo
    tilingData.s1s2BNGS1S2BaseParams.set_b(fBaseParams.b);
    tilingData.s1s2BNGS1S2BaseParams.set_n2(fBaseParams.n2);
    tilingData.s1s2BNGS1S2BaseParams.set_g(fBaseParams.g);
    tilingData.s1s2BNGS1S2BaseParams.set_s1(fBaseParams.s1);
    tilingData.s1s2BNGS1S2BaseParams.set_d(fBaseParams.d);
    tilingData.s1s2BNGS1S2BaseParams.set_s2(fBaseParams.s2);

    tilingData.s1s2BNGS1S2BaseParams.set_pseOptional(fBaseParams.pseOptional);
    tilingData.s1s2BNGS1S2BaseParams.set_pseType(fBaseParams.pseType);
    tilingData.s1s2BNGS1S2BaseParams.set_pseShapeType(fBaseParams.pseShapeType);
    tilingData.s1s2BNGS1S2BaseParams.set_pseDtype(fBaseParams.pseDtype);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskOptional(fBaseParams.attenMaskOptional);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskShapeType(fBaseParams.attenMaskShapeType);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskDtype(fBaseParams.attenMaskDtype);
    tilingData.s1s2BNGS1S2BaseParams.set_scaleValue(fBaseParams.scaleValue);
    tilingData.s1s2BNGS1S2BaseParams.set_keepProb(fBaseParams.keepProb);

    // fBaseParams.s1Token int64_t类型   tilingData.s1s2BNGS1S2BaseParams.s1Token  int32_t类型 防止溢出
    tilingData.s1s2BNGS1S2BaseParams.set_s1Token(fBaseParams.s1Token > INT32_MAX ? INT32_MAX : fBaseParams.s1Token);
    tilingData.s1s2BNGS1S2BaseParams.set_s2Token(fBaseParams.s2Token > INT32_MAX ? INT32_MAX : fBaseParams.s2Token);

    tilingData.s1s2BNGS1S2BaseParams.set_sparseMode(fBaseParams.sparseMode);
    tilingData.s1s2BNGS1S2BaseParams.set_isSparse(fBaseParams.isSparse);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskS2Size(fBaseParams.attenMaskS2Size);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskCompressMode(fBaseParams.attenMaskCompressMode);

    // s1/s2 split
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1Outer(fBaseParams.s1Outer);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1Inner(fBaseParams.s1Inner);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1CvInner(fBaseParams.s1CvInner);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1Tail(fBaseParams.s1Tail);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1CvTail(fBaseParams.s1CvTail);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s2Outer(fBaseParams.s2Outer);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s2CvInner(fBaseParams.s2CvInner);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s2Inner(fBaseParams.s2Inner);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s2Tail(fBaseParams.s2Tail);

    tilingData.s1s2BNGS1S2SplitCoreParams.set_baseMN(fBaseParams.baseMN);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_bandIdx(fBaseParams.bandIdx);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_blockOuter(fBaseParams.blockOuter);

    if (fBaseParams.pseType == PSE_INNER_MUL_ADD_TYPE ||
        fBaseParams.pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
        tilingData.s1s2BNGS1S2BaseParams.set_pseAlibiBaseS1(fBaseParams.pseAlibiBaseS1);
        tilingData.s1s2BNGS1S2BaseParams.set_pseAlibiBaseS2(fBaseParams.pseAlibiBaseS2);
    }
    return ge::GRAPH_SUCCESS;
}

// 以下场景对外部输入token屏蔽，重新设置token值并做校验
ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::ProcessTokensInfo()
{
    OPS_LOG_D(context_, "Before correction ,the value of s1Token = %ld and the value of s2Token %ld.",
              fBaseParams.s1Token, fBaseParams.s2Token);

    // 自动校正left和right causal的token值，token信息仅用于sparse分核计算
    if (fBaseParams.sparseMode == LEFT_UP_CAUSAL || fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL ||
        fBaseParams.sparseMode == PREFIX || fBaseParams.sparseMode == PREFIX_COMPRESS) {
        fBaseParams.s1Token = INT32_MAX;
        fBaseParams.s2Token = 0;
    }

    // 对pad场景做校正
    // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
    if (fBaseParams.layoutType != INPUT_FORMAT_TND &&
        (fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL || fBaseParams.sparseMode == BAND ||
         fBaseParams.sparseMode == PREFIX || fBaseParams.sparseMode == PREFIX_COMPRESS)) {
        fBaseParams.s1Token = fBaseParams.s1Token + fBaseParams.s1 - fBaseParams.s2;
        fBaseParams.s2Token = fBaseParams.s2Token - fBaseParams.s1 + fBaseParams.s2;
    }

    if (fBaseParams.sparseMode == ALL_MASK || fBaseParams.attenMaskOptional == EMPTY_TENSOR) {
        fBaseParams.s1Token = INT32_MAX;
        fBaseParams.s2Token = INT32_MAX;
    }

    OPS_LOG_D(context_, "the corrected s1Token = %ld, s2Token %ld.", fBaseParams.s1Token,
              fBaseParams.s2Token);

    // 1  2  3  5  6  不校验
    if (fBaseParams.sparseMode == ALL_MASK || fBaseParams.sparseMode == LEFT_UP_CAUSAL ||
        fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL || fBaseParams.sparseMode == PREFIX ||
        fBaseParams.sparseMode == PREFIX_COMPRESS) {
        return ge::GRAPH_SUCCESS;
    }

    // 校验pad场景token是否合法
    if (fBaseParams.layoutType != INPUT_FORMAT_TND &&
        (-fBaseParams.s1Token > fBaseParams.s2 || -fBaseParams.s2Token > fBaseParams.s1 ||
         (fBaseParams.s1Token + fBaseParams.s2Token) < 0)) {
        OPS_LOG_E(context_,
                  "pre_token and next_token is invalid in the pad scene, got s1 %ld, s2 %ld,  "
                  "pre_token %ld, next_token %ld",
                  fBaseParams.s1, fBaseParams.s2, fBaseParams.s1Token, fBaseParams.s2Token);
        return ge::GRAPH_FAILED;
    }

    // 校验unpad场景token是否合法   0  4  7  8
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        // 7  8
        if (fBaseParams.sparseMode == RIGHT_DOWN_CASUAL_BAND || fBaseParams.sparseMode == BAND_LEFT_UP_CASUAL) {
            int64_t actualS1Len = fBaseParams.actualSeqQlen[fBaseParams.bandIdx];
            int64_t actualS2Len = fBaseParams.actualSeqKvlen[fBaseParams.bandIdx];
            if (-fBaseParams.s1Token > actualS1Len || -fBaseParams.s2Token > actualS2Len ||
                (fBaseParams.s1Token + fBaseParams.s2Token) <= 0) {
                OPS_LOG_E(context_,
                          "pre_token and next_token is invalid in the unpad scene, got b %ld, s1 %ld, s2 %ld,  "
                          "pre_token %ld, "
                          "next_token %ld, sparse_mode %u",
                          fBaseParams.bandIdx, actualS1Len, actualS2Len, fBaseParams.s1Token, fBaseParams.s2Token,
                          fBaseParams.sparseMode);
                return ge::GRAPH_FAILED;
            }
            return ge::GRAPH_SUCCESS;
        }

        // 0  4
        for (int64_t i = 0; i < fBaseParams.b; i++) {
            int64_t actualS1Len = fBaseParams.actualSeqQlen[i];
            int64_t actualS2Len = fBaseParams.actualSeqKvlen[i];
            if (fBaseParams.sparseMode == NO_MASK) {
                if (-fBaseParams.s1Token > actualS2Len || -fBaseParams.s2Token > actualS1Len ||
                    (fBaseParams.s1Token + fBaseParams.s2Token) <= 0) {
                    OPS_LOG_E(context_,
                              "pre_token and next_token is invalid in the unpad scene, got b %ld, s1 %ld, s2 %ld,  "
                              "pre_token %ld, "
                              "next_token %ld, sparse_mode %u",
                              i, actualS1Len, actualS2Len, fBaseParams.s1Token, fBaseParams.s2Token,
                              fBaseParams.sparseMode);
                    return ge::GRAPH_FAILED;
                }
            }
            if (fBaseParams.sparseMode == BAND) {
                if (-fBaseParams.s1Token > actualS1Len || -fBaseParams.s2Token > actualS2Len ||
                    (fBaseParams.s1Token + fBaseParams.s2Token) <= 0) {
                    OPS_LOG_E(context_,
                              "pre_token and next_token is invalid in the unpad scene, got b %ld, s1 %ld, s2 %ld,  "
                              "pre_token %ld, "
                              "next_token %ld, sparse_mode %u",
                              i, actualS1Len, actualS2Len, fBaseParams.s1Token, fBaseParams.s2Token,
                              fBaseParams.sparseMode);
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

int64_t FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::FindBandIdx()
{
    if (fBaseParams.sparseMode == RIGHT_DOWN_CASUAL_BAND) {
        for (int i = fBaseParams.b - 1; i >= 0; i--) {
            if (fBaseParams.actualSeqQlen[i] != 0) {
                return i;
            }
        }
    } else if (fBaseParams.sparseMode == BAND_LEFT_UP_CASUAL) {
        for (int64_t i = 0; i < fBaseParams.b; i++) {
            if (fBaseParams.actualSeqQlen[i] != 0) {
                return i;
            }
        }
    }
    return 0;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::DoPreTiling()
{
    uint32_t castBufferLen = 60 * 1024;
    uint32_t outputBufferLen = 30 * 1024;
    uint32_t inputBufferLen = 4 * 1024;
    int64_t singleUBProcessNum = castBufferLen / 2;

    int64_t maskSize = AlignTo(fBaseParams.dropMaskSize, static_cast<int64_t>(BOOL_BLOCK_NUMS));
    int64_t singleCoreNum = AlignTo(CeilCommon(maskSize, static_cast<int64_t>(fBaseParams.blockOuter)),
                                    static_cast<int64_t>(BOOL_BLOCK_NUMS));
    int64_t maskUsedCoreNum = static_cast<int64_t>(CeilCommon(maskSize, singleCoreNum));

    int64_t tailCoreNum = maskSize - (maskUsedCoreNum - 1) * singleCoreNum;
    tailCoreNum = AlignTo(tailCoreNum, static_cast<int64_t>(BOOL_BLOCK_NUMS));

    int64_t singleCoreUBLoop = static_cast<int64_t>(CeilCommon(singleCoreNum, singleUBProcessNum));
    int64_t tailCoreUBLoop = static_cast<int64_t>(CeilCommon(tailCoreNum, singleUBProcessNum));

    int64_t singleCoreUBLastLoopNum = static_cast<int64_t>(singleCoreNum - (singleCoreUBLoop - 1) * singleUBProcessNum);
    int64_t tailCoreUBLastLoopNum = static_cast<int64_t>(tailCoreNum - (tailCoreUBLoop - 1) * singleUBProcessNum);

    tilingData.preTilingData.set_maskCoreNum(maskUsedCoreNum);
    tilingData.preTilingData.set_castBufferLen(castBufferLen);
    tilingData.preTilingData.set_outputBufferLen(outputBufferLen);
    tilingData.preTilingData.set_inputBufferLen(inputBufferLen);
    tilingData.preTilingData.set_singleUBProcessNum(static_cast<int64_t>(singleUBProcessNum));
    tilingData.preTilingData.set_maskSingleCoreNum(singleCoreNum); // size == num
    tilingData.preTilingData.set_maskSingleCoreLoop(singleCoreUBLoop);
    tilingData.preTilingData.set_maskLastLoopNum(singleCoreUBLastLoopNum);
    tilingData.preTilingData.set_maskTailCoreLoop(tailCoreUBLoop);
    tilingData.preTilingData.set_maskTailCoreLastLoopNum(tailCoreUBLastLoopNum);

    OPS_ERR_IF(maskUsedCoreNum == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "divisor maskUsedCoreNumis 0."),
               return ge::GRAPH_FAILED);
    int64_t qPreBlockFactor = (fBaseParams.qSizeAlign + maskUsedCoreNum - 1) / maskUsedCoreNum;
    OPS_ERR_IF(qPreBlockFactor == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "divisor qPreBlockFactor is 0."),
               return ge::GRAPH_FAILED);
    int64_t qPreBlockTotal = (fBaseParams.qSizeAlign + qPreBlockFactor - 1) / qPreBlockFactor;
    int64_t qPreTailNumTmp = fBaseParams.qSizeAlign % qPreBlockFactor;
    int64_t qPreTailNum = qPreTailNumTmp == 0 ? qPreBlockFactor : qPreTailNumTmp;

    int64_t kvPreBlockFactor = (fBaseParams.kvSizeAlign + maskUsedCoreNum - 1) / maskUsedCoreNum;
    OPS_ERR_IF(kvPreBlockFactor == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "divisor kvPreBlockFactor is 0."),
               return ge::GRAPH_FAILED);
    int64_t kvPreBlockTotal = (fBaseParams.kvSizeAlign + kvPreBlockFactor - 1) / kvPreBlockFactor;
    int64_t kvPreTailNumTmp = fBaseParams.kvSizeAlign % kvPreBlockFactor;
    int64_t kvPreTailNum = kvPreTailNumTmp == 0 ? kvPreBlockFactor : kvPreTailNumTmp;

    int64_t maskPreBlockTotal = (fBaseParams.dropMaskSize);
    tilingData.preTilingData.set_qPreBlockFactor(qPreBlockFactor);
    tilingData.preTilingData.set_qPreBlockTotal(qPreBlockTotal);
    tilingData.preTilingData.set_qPreBlockTail(qPreTailNum);
    tilingData.preTilingData.set_kvPreBlockFactor(kvPreBlockFactor);
    tilingData.preTilingData.set_kvPreBlockTotal(kvPreBlockTotal);
    tilingData.preTilingData.set_kvPreBlockTail(kvPreTailNum);
    tilingData.preTilingData.set_dropoutIsDivisibleBy8(fBaseParams.dropoutIsDivisibleBy8);
    tilingData.preTilingData.set_maskPreBlockTotal(maskPreBlockTotal);

    int64_t dropBeginAddr = MUL_CORE_SYNC_BUFFER;

    int64_t qSizeReal = fBaseParams.qSize;
    int64_t kvSizeReal = fBaseParams.kvSize;
    if (fBaseParams.mm2IsNZOut) {
        qSizeReal = fBaseParams.qSizeAlign;
        kvSizeReal = fBaseParams.kvSizeAlign;
    }
    dropBeginAddr =
        (dropBeginAddr + (qSizeReal) * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    dropBeginAddr =
        (dropBeginAddr + (kvSizeReal) * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    dropBeginAddr =
        (dropBeginAddr + (kvSizeReal) * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    tilingData.preTilingData.set_dropBeginAddr(dropBeginAddr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::DoPostTiling()
{
    int64_t dAlign = (fBaseParams.d + FP16_BLOCK_NUMS - 1) / FP16_BLOCK_NUMS * FP16_BLOCK_NUMS;
    int64_t curPostCoexNode = fBaseParams.mm2IsNZOut ? POST_NZ_COEX_NODE : POST_COEX_NODE;
    int64_t nzReservedSize = fBaseParams.mm2IsNZOut ? dAlign / C0_SIZE * BLOCK_SIZE * POST_NZ_RESERVED_N : 0;
    int64_t postUbBaseSize = (fBaseParams.ubSize - 2 * nzReservedSize) / curPostCoexNode / BUFFER_NUM /
                             WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN;

    int64_t qPostBaseNum =
        fBaseParams.mm2IsNZOut ? (postUbBaseSize / fBaseParams.dataTypeSize / dAlign * fBaseParams.d)
        : (postUbBaseSize / fBaseParams.dataTypeSize);
    OPS_ERR_IF(qPostBaseNum == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "divisor qPostBaseNum is 0."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(fBaseParams.blockOuter == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "divisor fBaseParams.blockOuter is 0."),
               return ge::GRAPH_FAILED);
    int64_t qPostBlockTotal = fBaseParams.qSize;
    int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    int64_t qPostBlockFactor = (qPostBlockOuterTotal + fBaseParams.blockOuter - 1) / fBaseParams.blockOuter;

    int64_t kvPostBaseNum = qPostBaseNum;
    OPS_ERR_IF(kvPostBaseNum == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context_, "divisor kvPostBaseNum is 0."),
               return ge::GRAPH_FAILED);
    int64_t kvPostBlockTotal = fBaseParams.kvSize;
    int64_t kvPostTailNumTmp = kvPostBlockTotal % kvPostBaseNum;
    int64_t kvPostTailNum = kvPostTailNumTmp == 0 ? kvPostBaseNum : kvPostTailNumTmp;
    int64_t kvPostBlockOuterTotal = (kvPostBlockTotal + kvPostBaseNum - 1) / kvPostBaseNum;
    int64_t kvPostBlockFactor = (kvPostBlockOuterTotal + fBaseParams.blockOuter - 1) / fBaseParams.blockOuter;

    tilingData.postTilingData.set_scaleValue(fBaseParams.scaleValue);
    tilingData.postTilingData.set_coreNum(fBaseParams.coreNum);
    tilingData.postTilingData.set_postUbBaseSize(postUbBaseSize);
    tilingData.postTilingData.set_nzReservedSize(nzReservedSize);
    tilingData.postTilingData.set_qPostBlockFactor(qPostBlockFactor);
    tilingData.postTilingData.set_qPostBlockTotal(qPostBlockTotal);
    tilingData.postTilingData.set_qPostBaseNum(qPostBaseNum);
    tilingData.postTilingData.set_qPostTailNum(qPostTailNum);
    tilingData.postTilingData.set_qSizeAlign(fBaseParams.qSizeAlign);

    tilingData.postTilingData.set_kvPostBlockFactor(kvPostBlockFactor);
    tilingData.postTilingData.set_kvPostBlockTotal(kvPostBlockTotal);
    tilingData.postTilingData.set_kvPostBaseNum(kvPostBaseNum);
    tilingData.postTilingData.set_kvPostTailNum(kvPostTailNum);
    tilingData.postTilingData.set_kvSizeAlign(fBaseParams.kvSizeAlign);

    int64_t workspaceOffsets = MUL_CORE_SYNC_BUFFER;
    tilingData.postTilingData.set_dqWorkSpaceOffset(workspaceOffsets);

    workspaceOffsets = (workspaceOffsets + fBaseParams.qSizeAlign * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    tilingData.postTilingData.set_dkWorkSpaceOffset(workspaceOffsets);

    workspaceOffsets = (workspaceOffsets + fBaseParams.kvSizeAlign * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    tilingData.postTilingData.set_dvWorkSpaceOffset(workspaceOffsets);

    tilingData.postTilingData.set_b(fBaseParams.b);
    tilingData.postTilingData.set_n2(fBaseParams.n2);
    tilingData.postTilingData.set_g(fBaseParams.g);
    tilingData.postTilingData.set_s1(fBaseParams.s1);
    tilingData.postTilingData.set_s2(fBaseParams.s2);
    tilingData.postTilingData.set_d(fBaseParams.d);

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::DoPreSfmgTiling()
{
    int64_t dAlign = AlignTo(fBaseParams.d, static_cast<int64_t>(fBaseParams.dataBlockNum));
    uint32_t inputBufferLen = 24 * 1024; // castBuffer 24K*2=48K
    uint32_t castBufferLen = 48 * 1024; // castBuffer 48K*2=96K
    uint32_t outputBufferLen = CeilCommon(castBufferLen, dAlign) * 8; // 输出(s1,8)
    uint32_t tempBufferLen = 40 * 1024 - outputBufferLen;

    int64_t normalAxisSize = 0;
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        normalAxisSize = fBaseParams.t1 * fBaseParams.n2 * fBaseParams.g;
    } else {
        normalAxisSize = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1;
    }
    fBaseParams.sfmgNormalAxisSize = normalAxisSize;

    // 计算单核的计算量
    int64_t normalCoreSize = CeilCommon(normalAxisSize, static_cast<int64_t>(fBaseParams.coreNum));
    int64_t usedCoreNum = CeilCommon(normalAxisSize, normalCoreSize);
    int64_t tailCoreSize = normalAxisSize - (usedCoreNum - 1) * normalCoreSize;

    // 计算单loop的计算量及loop次数
    int64_t singleLoopNBurstNum = inputBufferLen / fBaseParams.dataTypeSize / dAlign;
    int64_t normalCoreLoopTimes = CeilCommon(normalCoreSize, singleLoopNBurstNum);
    int64_t normalCoreLastLoopNBurstNum = normalCoreSize - (normalCoreLoopTimes - 1) * singleLoopNBurstNum;
    int64_t tailCoreLoopTimes = CeilCommon(tailCoreSize, singleLoopNBurstNum);
    int64_t tailCoreLastLoopNBurstNum = tailCoreSize - (tailCoreLoopTimes - 1) * singleLoopNBurstNum;

    tilingData.preSfmgTilingData.set_usedCoreNum(usedCoreNum);
    tilingData.preSfmgTilingData.set_inputBufferLen(inputBufferLen);
    tilingData.preSfmgTilingData.set_castBufferLen(castBufferLen);
    tilingData.preSfmgTilingData.set_outputBufferLen(outputBufferLen);
    tilingData.preSfmgTilingData.set_tempBufferLen(tempBufferLen);

    tilingData.preSfmgTilingData.set_singleLoopNBurstNum(singleLoopNBurstNum);
    tilingData.preSfmgTilingData.set_normalCoreLoopTimes(normalCoreLoopTimes);
    tilingData.preSfmgTilingData.set_tailCoreLoopTimes(tailCoreLoopTimes);
    tilingData.preSfmgTilingData.set_normalCoreLastLoopNBurstNum(normalCoreLastLoopNBurstNum);
    tilingData.preSfmgTilingData.set_tailCoreLastLoopNBurstNum(tailCoreLastLoopNBurstNum);
    tilingData.preSfmgTilingData.set_normalCoreNBurstNums(normalCoreSize);

    int64_t dropMaskSize = 0;
    if (fBaseParams.dropoutIsDivisibleBy8 == 0) {
        dropMaskSize = (fBaseParams.dropMaskSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN; // 与主kernel的偏移计算保持一致
    }
    int64_t sfmgPreBeginAddr = tilingData.preTilingData.get_dropBeginAddr() + dropMaskSize;
    tilingData.preSfmgTilingData.set_sfmgPreBeginAddr(sfmgPreBeginAddr);
    tilingData.preSfmgTilingData.set_b(fBaseParams.b);
    tilingData.preSfmgTilingData.set_n2(fBaseParams.n2);
    tilingData.preSfmgTilingData.set_g(fBaseParams.g);
    tilingData.preSfmgTilingData.set_s1(fBaseParams.s1);
    tilingData.preSfmgTilingData.set_d(fBaseParams.d);

    auto softmaxGradShape = ge::Shape({singleLoopNBurstNum, dAlign});
    AscendC::SoftMaxGradTilingFunc(softmaxGradShape, fBaseParams.calTypeSize, tempBufferLen,
                                   tilingData.softmaxGradTilingData, true);
}

void FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb::DetermineMode()
{
    // 当前fp16都走高精度
    if (fBaseParams.queryType == ge::DT_FLOAT) {
        fBaseParams.mode = FP32;
    } else if (fBaseParams.queryType == ge::DT_BF16) {
        fBaseParams.mode = BF16;
    } else if (fBaseParams.queryType == ge::DT_FLOAT16) {
        fBaseParams.mode = INHP;
    } else {
        fBaseParams.mode = FP16;
    }
}


REGISTER_TILING_TEMPLATE("FlashAttentionScoreGrad", FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb, 15500);
REGISTER_TILING_TEMPLATE("FlashAttentionScoreGrad", FlashAttentionScoreGradTilingSameABDeterministic, 1100);

} // namespace optiling
