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
 * \file prompt_flash_attention_tiling.cpp
 * \brief
 */
#include <queue>
#include <vector>
#include <unordered_map>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/data_copy_transpose_tiling.h"
#include "error/ops_error.h"
#include "prompt_flash_attention_tiling.h"

#include <cstdint>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <unistd.h>
#include <stdio.h>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "register/tilingdata_base.h"

using namespace ge;
using namespace AscendC;
using namespace matmul_tiling;
namespace optiling {
constexpr uint32_t BYTE_BLOCK = 32; // The block size of datacopy, which moves data at the block granularity.
constexpr uint32_t SOFTMAX_BUFFER_NUM = 3;

constexpr uint32_t NUM_0 = 0;
constexpr uint32_t NUM_1 = 1;
constexpr uint32_t NUM_2 = 2;
constexpr uint32_t NUM_3 = 3;
constexpr uint32_t NUM_4 = 4;
constexpr uint32_t INDEX_2 = 2;
constexpr uint32_t INDEX_3 = 3;
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t ATTENTION_OUT_INDEX = 0;
constexpr uint32_t PSE_SHIFT_INDEX = 3;
constexpr uint32_t ATTEN_MASK_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX = 6;
constexpr uint32_t DEQ_SCALE1_INDEX = 7;
constexpr uint32_t QUANT_SCALE1_INDEX = 8;
constexpr uint32_t DEQ_SCALE2_INDEX = 9;
constexpr uint32_t QUANT_SCALE2_INDEX = 10;
constexpr uint32_t QUANT_OFFSET2_INDEX = 11;
constexpr uint32_t ANTIQUANT_SCALE_INDEX = 12;
constexpr uint32_t ANTIQUANT_OFFSET_INDEX = 13;

constexpr uint32_t INPUT_QKV_SHAPE_MIN_DIMS = 2;
constexpr uint32_t INPUT_QKV_SHAPE_MAX_DIMS = 4;

constexpr uint32_t ATTR_N_INDEX = 0;
constexpr uint32_t ATTR_SCALE_INDEX = 1;
constexpr uint32_t ATTR_PRE_TOKEN_INDEX = 2;
constexpr uint32_t ATTR_NEXT_TOKEN_INDEX = 3;
constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 4;
constexpr uint32_t ATTR_NUM_KV_HEADS_INDEX = 5;

constexpr uint64_t EMPTY_KV_TILING_KEY = 20;
constexpr uint32_t LOOP_BEGIN_NUM = 0;
constexpr uint32_t SPARSE_MODE_NO_MASK = 0;
constexpr uint32_t SPARSE_MODE_ALL_MASK = 1;
constexpr uint32_t SPARSE_MODE_LEFT_UP = 2;
constexpr uint32_t SPARSE_MODE_RIGHT_DOWN = 3;
constexpr uint32_t SPARSE_MODE_BAND = 4;
constexpr uint32_t SPARSE_MODE_INT_MAX = 214748647;
constexpr uint32_t ATTR_SPARSE_MODE = 6;
constexpr uint32_t ATTR_INNER_PRECISE = 7;
constexpr uint32_t SPARSE_OPTIMIZE_ATTENTION_SIZE = 2048;
constexpr uint32_t PSE_SHIFT_DIM = 4;
constexpr uint32_t ATTENTION_MASK_DIM2 = 2;
constexpr uint32_t ATTENTION_MASK_DIM3 = 3;
constexpr uint32_t ATTENTION_MASK_DIM4 = 4;
constexpr int32_t BLOCK_SIZE_BASE = 128;  // The current requirement is a multiple of 128, and to prevent cross block handling, the mm base is also set to 128.
constexpr int32_t BLOCK_SIZE_MAX = 512;

constexpr uint32_t CVDIFF_S2_THRESHOLDS = 1;
constexpr uint32_t CVDIFF_SMALL_QS_THRESHOLDS = 16;
constexpr uint32_t CVDIFF_MM1RES_UB_SIZE = 16384; // 128 * 128
constexpr uint32_t CVDIFF_SOUTER_FACTOR_DEFAULT = 128;
constexpr uint32_t CVDIFF_SMALL_KV_THRESHOLDS = 1024;
constexpr uint32_t CVDIFF_SINNER_FACTOR_SMALL_KVS = 512;   // kv_s <= 512 scene sinner slice size
constexpr uint32_t CVDIFF_SINNER_FACTOR_DEFAULT = 1024;    // CV diff general scene sinner slice size
constexpr uint32_t CVDIFF_SINNER_FACTOR_SMALL_QS = 2048;   // q_s <= 16 scene sinner slice size
constexpr uint32_t CVDIFF_MSD_BUFFER_SIZE_512B = 512; // 0.5k
constexpr uint32_t CVDIFF_MSD_BUFFER_SIZE_1024B = 1024; // 0.5k

constexpr uint32_t SPLIT_DOUBLE_UB = 2;
constexpr uint32_t DSPLIT_THRESHOLDS_512 = 512;
constexpr uint64_t DSPLIT_S2_D_TILING_KEY = 400;
constexpr uint64_t DSPLIT_S2_TILING_KEY = 300;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint64_t BENCHMARK_TILING_KEY = 1000000000000000000;
constexpr uint32_t THIRTY_ONE = 31;
constexpr uint32_t FROM_FUSED_FLAG = 71;
constexpr uint32_t MATMUL_NORM_MIN_SEQ = 128;
constexpr uint32_t MATMUL_NORM_MIN_HEADSIZE = 128;

constexpr uint32_t BLIMIT = 65536;
constexpr uint32_t NLIMIT = 256;  // n <= 256
constexpr uint32_t SLIMIT = 20971520;  // s、kvs <= 20M
constexpr uint32_t DLIMIT = 512; // D <= 512
constexpr uint32_t TLIMIT = 65536; // T <= 64K

constexpr uint32_t MSD_UB_BASE_WIDTH = 16;
constexpr uint32_t MSD_UB_HEGHT = 256;
constexpr uint32_t MSD_UB_INQUEUE = 8;
constexpr uint32_t MSD_UB_TMP_NM = 16;
constexpr uint32_t NO_MSD_UB_BMM2 = 64;
constexpr uint32_t NO_MSD_HIGH_PERFORMANCE = 2;
constexpr uint32_t NO_MSD_HIGH_PRECISION = 3;
constexpr uint32_t ONE_BLK_SIZE_PFA = 32;
constexpr uint32_t MAX_SOFTMAX_COMPUTE_LINES = 4;
constexpr uint32_t COMPUTELINE_FOR_BIG_D = 1;
constexpr uint32_t MAX_COMPUTELINES = 16;
constexpr uint32_t UB_SIZE_FOR_1_K = 1024;
constexpr uint32_t MSD_BIG_D = 256;
constexpr uint32_t CV_RATIO = 2;

constexpr int64_t SAMEAB_D_LIMIT_192 = 192L;
constexpr int64_t HIGH_PERF_BUFFER_NUM = 6L;
constexpr int64_t HIGH_PERF_API_BUFFER_MULTIPLE = 2L;
constexpr int64_t FRACTAL_NUM = 16;
constexpr int64_t AIV_AIC_NUM_RATIO = 2L;
constexpr int64_t S1_VEC2_BASE_SIZE_MAX = 512L;
constexpr int64_t BMM_BASICBLOCK_M_128 = 128L;
constexpr int64_t BMM_BASICBLOCK_N_256 = 256L;
constexpr int64_t BMM_BASICBLOCK_N_128 = 128L;
constexpr int64_t BMM1_DEPTH_A1_2 = 2L;
constexpr int64_t BMM1_DEPTH_A1_3 = 3L;
constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
constexpr int64_t MAX_AIC_NUM = 24L;
constexpr int64_t MAX_AIV_NUM = 48L;
constexpr int64_t INVALID_ROW_SPARSE_RATIO = 6L;
constexpr int64_t HIGH_PERF_BLOCK_SIZE = 128L;

enum AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_MODE,
    PREFIX_MODE,
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,
    BAND_LEFT_UP_CAUSAL_MODE
};

enum class SparseTypeEnum {
    ALL = 0,
    NONE = 1,
    ANY = 2,
    CAUSAL = 3,
    BAND = 4,
    PREFIX = 5,
    BAND_COMPRESS = 6,
    RIGHT_DOWN_CAUSAL = 7,
    RIGHT_DOWN_CAUSAL_BAND = 8,
    BAND_LEFT_UP_CAUSAL = 9
};

static const std::unordered_map<ge::DataType, string> g_strDataTypePfa = {
    {ge::DT_FLOAT, "DT_FLOAT"},
    {ge::DT_FLOAT16, "DT_FLOAT16"},
    {ge::DT_INT8, "DT_INT8"},
    {ge::DT_INT16, "DT_INT16"},
    {ge::DT_UINT16, "DT_UINT16"},
    {ge::DT_UINT8, "DT_UINT8"},
    {ge::DT_INT32, "DT_INT32"},
    {ge::DT_INT64, "DT_INT64"},
    {ge::DT_UINT32, "DT_UINT32"},
    {ge::DT_UINT64, "DT_UINT64"},
    {ge::DT_BOOL, "DT_BOOL"},
    {ge::DT_DOUBLE, "DT_DOUBLE"},
    {ge::DT_STRING, "DT_STRING"},
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8V"},
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},
    {ge::DT_QINT8, "DT_QINT8"},
    {ge::DT_QINT16, "DT_QINT16"},
    {ge::DT_QINT32, "DT_QINT32"},
    {ge::DT_QUINT8, "DT_QUINT8"},
    {ge::DT_QUINT16, "DT_QUINT16"},
    {ge::DT_RESOURCE, "DT_RESOURCE"},
    {ge::DT_STRING_REF, "DT_STRING_REF"},
    {ge::DT_DUAL, "DT_DUAL"},
    {ge::DT_VARIANT, "DT_VARIANT"},
    {ge::DT_BF16, "DT_BF16"},
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},
};

template <typename T> static T AlignUp(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T> static T AlignDown(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return num1 / num2 * num2;
}

template <typename T> static T CeilDivision(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T> static T CeilDiv(const T n1, const T n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

template <typename T> static T CalcTailSize(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

static uint32_t PromptGcd(uint32_t a, uint32_t b)
{
    if (a % b == 0) {
        return b;
	}
    return PromptGcd(b, a % b);
}

static ge::DataType ValidPfaDataType(ge::DataType type)
{
    return (g_strDataTypePfa.find(type) == g_strDataTypePfa.end()) ? ge::DT_UNDEFINED : type;
}

static ge::graphStatus ConvertContextToPFAParams(gert::TilingContext* context, ContextParamsForPFATiling& contextKeyParams)
{
    contextKeyParams.opName = context->GetNodeName();
    bool inputOutputIsNullPtr = (context->GetInputDesc(QUERY_INDEX) == nullptr) || (context->GetInputDesc(KEY_INDEX) == nullptr) ||
                                (context->GetInputDesc(VALUE_INDEX) == nullptr) || (context->GetOutputDesc(ATTENTION_OUT_INDEX) == nullptr) ||
                                (context->GetInputShape(QUERY_INDEX) == nullptr) || (context->GetInputShape(KEY_INDEX) == nullptr) ||
                                (context->GetInputShape(VALUE_INDEX) == nullptr) || (context->GetOutputShape(ATTENTION_OUT_INDEX) == nullptr);
    OPS_ERR_IF(inputOutputIsNullPtr,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "q, k, v or attenOut is nullptr!"),
                return ge::GRAPH_FAILED);

    contextKeyParams.isKvContinuous = 1;
    contextKeyParams.emptyTensor = 0;
    contextKeyParams.fromTilingSink = 0;
    contextKeyParams.pseShift = context->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMask = context->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    contextKeyParams.actualSeqenceLengthQ = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    contextKeyParams.actualSeqenceLengthKV = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    contextKeyParams.antiquantScale = context->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffset = context->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.inputDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    contextKeyParams.kDataType = context->GetInputDesc(KEY_INDEX)->GetDataType();
    contextKeyParams.vDataType = context->GetInputDesc(VALUE_INDEX)->GetDataType();
    contextKeyParams.blockTable = nullptr;
    contextKeyParams.keySharedPrefix = (nullptr);
    contextKeyParams.valueSharedPrefix = (nullptr);
    contextKeyParams.actualSharedPrefixLen = (nullptr);
    contextKeyParams.pseShiftDataType = (contextKeyParams.pseShift != nullptr) ?
    context->GetOptionalInputDesc(PSE_SHIFT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.maskDataType = (contextKeyParams.attentionMask != nullptr) ?
    context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.outputDataType = context->GetOutputDesc(0)->GetDataType();
    contextKeyParams.queryInputShape = context->GetInputShape(QUERY_INDEX);
    contextKeyParams.keyInputShape = context->GetInputShape(KEY_INDEX);
    contextKeyParams.valueInputShape = context->GetInputShape(VALUE_INDEX);
    contextKeyParams.pseShiftShape = context->GetOptionalInputShape(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INDEX);
    contextKeyParams.deqScale1Shape = context->GetOptionalInputShape(DEQ_SCALE1_INDEX);
    contextKeyParams.scale1Shape = context->GetOptionalInputShape(QUANT_SCALE1_INDEX);
    contextKeyParams.deqScale2Shape = context->GetOptionalInputShape(DEQ_SCALE2_INDEX);
    contextKeyParams.scale2Shape = context->GetOptionalInputShape(QUANT_SCALE2_INDEX);
    contextKeyParams.offset2Shape = context->GetOptionalInputShape(QUANT_OFFSET2_INDEX);
    contextKeyParams.antiquantScaleShape = context->GetOptionalInputShape(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffsetShape = context->GetOptionalInputShape(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.outputShape = context->GetOutputShape(0);
    auto attrs = context->GetAttrs();
    contextKeyParams.innerPrecisePtr = attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE);
    contextKeyParams.headsNumber = attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX);
    contextKeyParams.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE);
    contextKeyParams.preToken = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
    contextKeyParams.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
    contextKeyParams.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    contextKeyParams.layout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    contextKeyParams.numKeyValueHeads = attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
    contextKeyParams.workspaceSize = context->GetWorkspaceSizes(1);
    contextKeyParams.compileInfoPtr = reinterpret_cast<const PromptFlashAttentionCompileInfo *>(context->GetCompileInfo());
    contextKeyParams.isBSNDOut = (string(contextKeyParams.layout) == "BNSD_BSND") ? 1 : 0;

    contextKeyParams.deqScaleType = (context->GetOptionalInputDesc(DEQ_SCALE1_INDEX) != nullptr) ?
    context->GetOptionalInputDesc(DEQ_SCALE1_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.deqScale2Type = (context->GetOptionalInputDesc(DEQ_SCALE2_INDEX) != nullptr) ?
    context->GetOptionalInputDesc(DEQ_SCALE2_INDEX)->GetDataType() : contextKeyParams.inputDataType;

    contextKeyParams.quantScale2Type = (context->GetOptionalInputDesc(QUANT_SCALE2_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_SCALE2_INDEX)->GetDataType() : ge::DT_FLOAT;
    contextKeyParams.quantOffset2Type = (context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX)->GetDataType() : ge::DT_FLOAT;

    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::UpdateTilingKeyFlag(ContextParamsForPFATiling& contextKeyParams, uint64_t& tilingKey)
{
    uint64_t binaryFlag = 0;
    auto queryDtype = contextKeyParams.inputDataType;
    auto kvDtype = contextKeyParams.kDataType;

    if ((queryDtype == ge::DT_FLOAT16) && (kvDtype == ge::DT_INT8) && !(enableMsd)) {
        binaryFlag += 8;    // 4bit flag bit, the leftmost side indicates whether to perform inverse quantization operation, with a corresponding value of 2**3 = 8, and the remaining 3bit is reserved
    }
    tilingKey += (binaryFlag * 100000000000); // If inverse quantization is performed, tilingKey should increase by 8*100000000000.
    return;
}

bool PromptFlashAttentionTiling::GetApiTmpSize(const uint32_t sOuterFactor, const uint32_t sInnerFactor, const uint32_t typeByteSize)
{
    auto tmpShape = Shape({sOuterFactor, sInnerFactor});
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        apiTmpSize = GetSoftMaxFlashV2MinTmpSize(tmpShape, typeByteSize, typeByteSize, true, true);
        return true;
    }
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND910B) {
        uint32_t softmaxTmpSize = GetSoftMaxMinTmpSize(tmpShape, typeByteSize, true);
        uint32_t softmaxFlashTmpSize = GetSoftMaxFlashMinTmpSize(tmpShape, typeByteSize, true, true);
        if ((softmaxTmpSize == 0) || (softmaxFlashTmpSize == 0)) {
            return false;
        }
        apiTmpSize = std::max(softmaxTmpSize, softmaxFlashTmpSize);
    }
    return false;
}

size_t PromptFlashAttentionTiling::GetPFAWorkSpaceSize(PromptFlashAttentionTilingData& tilingData)
{
    size_t sysWorkspaceSize, workspaceSize;
    const uint64_t defaultSysWorkspaceSize910B = 16U * 1024U * 1024U;
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        sysWorkspaceSize = defaultSysWorkspaceSize;  // sys workspace size default value
        return sysWorkspaceSize;
    } else { // 910b
        uint64_t maxSpmSize = tilingData.promptAttentionTensorSizeRect.get_spmTmpSize();
        sysWorkspaceSize = defaultSysWorkspaceSize910B;  // sys workspace size default value
        if (tilingMod == TilingMod::CVDIFF) {
            int64_t mm1ResSize = tilingData.promptAttentionSingleCoreParams.get_singleProcessSOuterSize() * \
                                 tilingData.promptAttentionSingleCoreParams.get_singleProcessSInnerSize();
            int64_t mm2ResSize = tilingData.promptAttentionSingleCoreParams.get_singleProcessSOuterSize() * \
                                 tilingData.promptAttentionBaseParams.get_headSize();

            int64_t mdsExpandNumber = MSD_HIGH_PERFORMANCE_EXPEND_NUM;
            if (innerPrecise == HIGH_PRECISION) {
                mdsExpandNumber = MSD_HIGH_PRECISION_EXPEND_NUM;
            }
            if (enableMsd) {
                workspaceSize = sysWorkspaceSize + coreNum * softmaxDataTypeSize * (maxSpmSize + mm1ResSize * NUM_2 * mdsExpandNumber + mm2ResSize * NUM_2 * mdsExpandNumber); // 2:use 2mm ub
            } else {
                workspaceSize = sysWorkspaceSize + coreNum * softmaxDataTypeSize * (maxSpmSize + mm1ResSize * NUM_2 + mm2ResSize * NUM_2); // 2:use 2mm ub
            }

            if (enableKvAntiquant) {
                int32_t KvAntiquantSize = tilingData.promptAttentionSingleCoreParams.get_singleProcessSInnerSize() * \
                                 tilingData.promptAttentionBaseParams.get_alignedHeadSize();
                workspaceSize += coreNum * dataTypeSize * KvAntiquantSize * 2;  // key value, 2 is used to ensure alignment
            }
            if (enablePA) {
                workspaceSize += static_cast<uint64_t>(coreNum) * 2 * 2 * 64;  // 2 bmm, db, ensure alignment of each structure 64B, dcci cacheline needs to
            }

            if (enableMsd) {
                workspaceSize = workspaceSize + coreNum * mdsExpandNumber * tilingData.promptAttentionBaseParams.get_seqSize() * tilingData.promptAttentionBaseParams.get_headSize();
            }
         } else {
            if ((splitS2 == 1) && (splitD == 1)) {
                workspaceSize = sysWorkspaceSize + coreNum * softmaxDataTypeSize * (maxSpmSize + \
                    NUM_2 * tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() * // 2 : 2 mm ub
                    tilingData.promptAttentionSingleCoreParams.get_multiSmaxsInnerLoopTimes());
            } else {
                workspaceSize = sysWorkspaceSize + coreNum * softmaxDataTypeSize * (maxSpmSize + \
                    tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() + \
                    tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() * \
                    tilingData.promptAttentionSingleCoreParams.get_multiSmaxsInnerLoopTimes());
            }
        }
        return workspaceSize;
    }
}

ge::graphStatus PromptFlashAttentionTiling::TilingGetTilingKeyAttentionAscendC(uint64_t& tilingKey,
    ContextParamsForPFATiling& contextKeyParams, bool useNewTiling, PromptFlashAttentionTilingData &tilingData) {
    auto inputDataType = contextKeyParams.inputDataType;  // input q
    auto attenMaskElemType = contextKeyParams.maskDataType;
    auto outputDataType = contextKeyParams.outputDataType;  // output tensor
    tilingData.promptAttentionBaseParams.set_attenMaskElemType(attenMaskElemType);

    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        tilingKey = 12288; // 12288: 310p tiling
        tilingKey += (inputLayout == InputLayout::BNSD) ? 0 : 10000;  // 10000 : BSH/BSND 22288
        return ge::GRAPH_SUCCESS;
    }
    tilingKey = 0U;
    // If not in CV diff template,when there is a tail block, tilingKey should increase by 1.
    tilingKey += (tilingMod == TilingMod::CVDIFF) || (isSOuterNoTail && isSInnerNoTail && isDNoTail) ? 0U : 1U;
    tilingKey += inputDataType == ge::DT_BF16 ? 100U : 0U;     // When the input qkv is BF16, add 100.
    tilingKey += inputDataType == ge::DT_INT8 ? 200U : 0U;     // When the input qkv is INT8, add 200.
    tilingKey += tilingMod == TilingMod::CVDIFF ? 1002U : 0U;   // 1002: Add 1000 when using CV diff; Without distinguishing between tail and no tail, add 2 uniformly.
    tilingKey += outputDataType == ge::DT_INT8 ? 20000U : 0U;   // When output is INT8, add 20000.

    if (!useNewTiling) {
        return ge::GRAPH_SUCCESS; // The old template does not consider NSD differences, only 0, 1, 100, 101
    }

    tilingKey += 10U; // New Template 10、11、15、16、110、111、115、116.
    tilingKey += (inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD) ? 5U : 0U;

    // The KV cache inverse quantization for CV diff currently only handles the case where Q in the CV diff template is FP16.
    if ((inputDataType == ge::DT_FLOAT16 || inputDataType == ge::DT_BF16) && (tilingMod == TilingMod::CVDIFF)) {
        tilingKey = 1012;    // 1012：CV diff, +1000; new_tiling, +10; not distinguishing between tail and total, +2.
        tilingKey += ((inputDataType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) || (enableMsd && contextKeyParams.inputDataType == ge::DT_FLOAT16 && contextKeyParams.kDataType == ge::DT_INT8) ? 600 : 0;  // fp16 high precision mode, regarded as a type 600.
        tilingKey += (inputDataType == ge::DT_BF16) ? 100 : 0;    // 100: bf16
        tilingKey += (outputDataType == ge::DT_BF16) ? 10000 : 0; // When the output dtype is bf16, tilingKey should increase by 10000.
        tilingKey += (outputDataType == ge::DT_INT8) ? 20000 : 0; // 20000: The situation of outputDataType == ge::DT_INT8
        tilingKey += ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::SH) || (inputLayout == InputLayout::BSND)) ? 100000 : 0;   // When the inputLayout is BSH, SH or BSND, plus 100000.
        tilingKey += ((splitCoreMode != SplitCoreMode::SPLIT_NBS_CUBE && splitCoreMode != SplitCoreMode::SPLIT_ONEN_CUBE) && enableMatmulNorm ? 1000000 : 0);  // Only enable matmul tiling optimization, do not enable l1reuse, add 1000000, mutually exclusive with the following situation of 2000000.
        tilingKey += ((splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE || splitCoreMode == SplitCoreMode::SPLIT_ONEN_CUBE) ? 2000000 : 0);   // l1reuse defaults to enabling matmul tiling optimization, with an additional 2000000, which is mutually exclusive from the 1000000 situation mentioned above.
        UpdateTilingKeyFlag(contextKeyParams, tilingKey);         // Determine whether to perform inverse quantization and generate a binary number by combining it with the remaining reserved bits, and take its decimal representation.
    }

    if (enablePA) {
        tilingKey += 10000000;  // 10000000: the situation of PA
    }

    if (isKVHasPrefix) {
        tilingKey += 100000000;  // 100000000: the situation of prefix
    }

    if (enableMsd) {
        tilingKey += 400200000000;  // 400200000000: for msd
    }
 
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::PromptFlashAttentionSplitNS(ContextParamsForPFATiling& contextKeyParams,
                                            PromptFlashAttentionTilingData& tilingData,
                                            uint32_t curCoreNum, std::vector<int64_t>& actualSeqLengths) {
    if (contextKeyParams.fromTilingSink != 0) {
        return;
    }
    PromptAttentionSingleCoreParams* singleCoreParams = &tilingData.promptAttentionSingleCoreParams;
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    PromptAttentionSeqParams* seqParams = &tilingData.promptAttentionSeqParams;

    uint32_t arrayLen = baseParams->get_dimNumOfseq();

    std::vector<uint32_t> coreHeadNumTail(arrayLen, 0U);
    std::vector<uint32_t> actualS1(arrayLen, 0U);
    std::vector<uint32_t> singleCoreHeadNumSize(arrayLen, 0U);
    std::vector<uint32_t> actualCoreNums(arrayLen, 0U);

    uint32_t multiSmaxsInnerLoopTimes = 0;

    std::vector<uint32_t> sInnerLoopTimes(arrayLen, 0U);
    std::vector<uint32_t> sOuterBlockNums(arrayLen, 0U);

    for (uint32_t i = LOOP_BEGIN_NUM; i < arrayLen; i++) {
        int seqLen = actualSeqLengths[i];
        sOuterBlockNums[i] = (seqLen + singleCoreParams->get_singleProcessSOuterSize() - 1)
                                / (singleCoreParams->get_singleProcessSOuterSize());
        sInnerLoopTimes[i] = (seqLen + singleCoreParams->get_singleProcessSInnerSize() - 1)
                                / (singleCoreParams->get_singleProcessSInnerSize());

        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[i]);

        if ((seqLen % singleCoreParams->get_singleProcessSOuterSize()) != 0) {
            isSOuterNoTail = false;
        }
        if ((seqLen % singleCoreParams->get_singleProcessSInnerSize()) != 0) {
            isSInnerNoTail = false;
        }

        // Two strategies 1. gcd average core 2. Abandoning some core components
        uint32_t headNumSize = baseParams->get_headNumSize();
        uint32_t n1 = PromptGcd(curCoreNum, headNumSize);
        if (headNumSize / n1 > sOuterBlockNums[i]) {
            // Abandoning some cores or N-dimensional partitioning
            if (headNumSize > curCoreNum) {
                singleCoreHeadNumSize[i] = headNumSize / curCoreNum;
                coreHeadNumTail[i] = headNumSize % curCoreNum;
                actualS1[i] = 1;
                actualCoreNums[i] = curCoreNum;
            } else {
                singleCoreHeadNumSize[i] = 1;
                coreHeadNumTail[i] = 0;
                actualS1[i] = curCoreNum / headNumSize;
                actualCoreNums[i] = actualS1[i] * headNumSize;
            }
        } else { // gcd average core
            uint32_t s1 = (curCoreNum / n1);
            singleCoreHeadNumSize[i] = (headNumSize / n1);
            coreHeadNumTail[i] = 0;
            actualS1[i] = s1;
            actualCoreNums[i] = (n1 * actualS1[i]);
        }
    }
    seqParams->set_singleCoreHeadNumSize(singleCoreHeadNumSize.data());
    seqParams->set_actualS1(actualS1.data());
    seqParams->set_CoreHeadNumTail(coreHeadNumTail.data());
    seqParams->set_actualCoreNums(actualCoreNums.data());

    singleCoreParams->set_multiSmaxsInnerLoopTimes(multiSmaxsInnerLoopTimes);
}

void PromptFlashAttentionTiling::PromptFlashAttentionInitOutputSplit(uint64_t totalSize,
    PromptFlashAttentionTilingData &tilingData, uint32_t curCoreNum)
{
    PromptAttentionInitOutputParams *initParams = &tilingData.promptAttentionInitOutputParams;

    uint32_t singleCoreSize = (totalSize + curCoreNum - 1) / (curCoreNum); // Upward rounding, coreNum has been verified to be non-zero when obtained.

    if (outputType == ge::DT_INT8) {
        singleCoreSize = (singleCoreSize + 1) / 2 * 2;        // 2：In the int8 scenario, when initializing, fill in 0 according to the half type, requiring that the number of points allocated to each kernel must be even.
    }

    initParams->set_singleCoreSize(singleCoreSize);
    initParams->set_totalOutputSize(totalSize);
}

void PromptFlashAttentionTiling::PromptFlashAttentionInitSoftmaxLseOutputSplit(uint64_t totalSize,
    PromptFlashAttentionTilingData &tilingData)
{
    PromptAttentionInitOutputParams *initParams = &tilingData.promptAttentionInitOutputParams;
    initParams->set_totalSoftMaxLseOutputSize(totalSize);
}

void PromptFlashAttentionTiling::PromptFlashAttentionSplitNSNew(
    ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData,
    uint32_t curCoreNum, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV, int64_t actualSharedPrefixLen, bool useBalanceTiling) {
    if (contextKeyParams.fromTilingSink != 0) {
        return;
    }
    PromptAttentionSingleCoreParams* singleCoreParams = &tilingData.promptAttentionSingleCoreParams;
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    PromptAttentionSeqParams* seqParams = &tilingData.promptAttentionSeqParams;

    uint32_t arrayLen = baseParams->get_dimNumOfseq();
    uint32_t sOuterSize = singleCoreParams->get_singleProcessSOuterSize();
    uint32_t sInnerSize = singleCoreParams->get_singleProcessSInnerSize();
    if (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) {
        sOuterSize = sOuterSize * CV_RATIO;
        curCoreNum = curCoreNum / CV_RATIO;
    }

    std::vector<uint32_t> accumSOuterTilingNums(static_cast<size_t>(arrayLen), 0U);
    std::vector<uint32_t> sInnerLoopTimes(static_cast<size_t>(arrayLen), 0U);
    std::vector<uint32_t> sOuterBlockNums(static_cast<size_t>(arrayLen), 0U);

    // The tiling structure element needs to have a length greater than or equal to the length specified by TILING_DATA_FIELD_DEF_ARR.
    // If the tiling structure definition specifies a length of 50, the vector definition needs to compare its size with curCoreNum and take the larger value.
    const size_t tilingElementArrayLen = (static_cast<size_t>(curCoreNum) > 50UL) ? \
        static_cast<size_t>(curCoreNum) : 50UL;
    std::vector<uint32_t> coreSposEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSposStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidStart(tilingElementArrayLen, 0U);

    int64_t totalBlockWight = 0;
    int totalOuterBlockNum = 0;
    uint32_t preAccumSOuterNum = 0U;
    uint32_t multiSmaxsInnerLoopTimes = 0U;
    int nextTokensPerBatch = 0;
    uint32_t sInnerPrefixLoopTimes = (actualSharedPrefixLen + sInnerSize - 1) / sInnerSize;
    for (uint32_t i = LOOP_BEGIN_NUM; i < arrayLen; i++) {
        int seqLen = actualSeqLengths[i];
        int subSeqInnerLen = actualSeqLengthsKV[i];
        sOuterBlockNums[i] = (seqLen + sOuterSize - 1) / sOuterSize;
        sInnerLoopTimes[i] = (subSeqInnerLen + sInnerSize - 1) / sInnerSize + sInnerPrefixLoopTimes;
        accumSOuterTilingNums[i] = (sOuterBlockNums[i] * baseParams->get_headNumSize()) + preAccumSOuterNum;
        preAccumSOuterNum = accumSOuterTilingNums[i];

        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[i]);

        if (baseParams->get_sparseMode() == SPARSE_MODE_RIGHT_DOWN) {
            nextTokensPerBatch = subSeqInnerLen + actualSharedPrefixLen - seqLen;
        } else {
            nextTokensPerBatch = baseParams->get_nextTokens();
        }

        if (seqLen % sOuterSize != 0) {
            isSOuterNoTail = false;
        }
        if (subSeqInnerLen % sInnerSize != 0) {
            isSInnerNoTail = false;
        }
        totalOuterBlockNum += sOuterBlockNums[i];
        if (nextTokensPerBatch == 0) {
            totalBlockWight += (static_cast<int64_t>(sOuterBlockNums[i]) + 1) * static_cast<int64_t>(sOuterBlockNums[i]) / NUM_2;  // div 2
        } else {
            totalBlockWight += static_cast<int64_t>(sOuterBlockNums[i]) * static_cast<int64_t>(sInnerLoopTimes[i]);
        }
    }
    if ((!useBalanceTiling)) {
        accumSOuterTilingNums[0] = 0;
    }

    float coreWightTarget = (float(totalBlockWight * baseParams->get_headNumSize()) / float(curCoreNum));

    // The temporary algorithm needs to be optimized.
    int curWight = 0;
    int curCore = 0;
    coreSposStart[curCore] = 0;
    coreSidStart[curCore] = 0;
    coreNidStart[curCore] = 0;
    for (uint32_t i = LOOP_BEGIN_NUM; i < baseParams->get_headNumSize(); i++) {
        for (uint32_t j = 0; j < arrayLen; j++) {
            if (baseParams->get_sparseMode() == SPARSE_MODE_RIGHT_DOWN) {
                nextTokensPerBatch = actualSeqLengthsKV[j] + actualSharedPrefixLen - actualSeqLengths[j];
            } else {
                nextTokensPerBatch = baseParams->get_nextTokens();
            }
            for (uint32_t k = 0; k < sOuterBlockNums[j]; k++) {
                int64_t dif = int64_t(coreWightTarget * float(curCore + 1)) - curWight;
                int64_t curWightPlus;
                if (nextTokensPerBatch == 0) {
                    curWightPlus = k + 1;
                }else {
                    curWightPlus = sInnerLoopTimes[j];
                }
                if ((curWightPlus - dif) > dif) {
                    if (k == 0) {
                        if (j == 0) {
                            coreNidEnd[curCore] = i;
                            coreSidEnd[curCore] = arrayLen;
                            coreSposEnd[curCore] = sOuterBlockNums[arrayLen - 1];
                        } else {
                            coreNidEnd[curCore] = i + 1;
                            coreSidEnd[curCore] = j;
                            coreSposEnd[curCore] = sOuterBlockNums[j-1];
                        }
                    } else {
                        coreNidEnd[curCore] = i + 1;
                        coreSidEnd[curCore] = j + 1;
                        coreSposEnd[curCore] = k;
                    }
                    curCore += 1;
                    coreNidStart[curCore] = i;
                    coreSidStart[curCore] = j;
                    coreSposStart[curCore] = k;
                }
                curWight += curWightPlus;
            }
        }
    }
    coreNidEnd[curCore] = (baseParams->get_headNumSize());
    coreSidEnd[curCore] = arrayLen;
    coreSposEnd[curCore] = sOuterBlockNums[arrayLen-1];

    // Temporary reuse
    seqParams->set_CoreHeadNumTail(coreNidStart.data());
    seqParams->set_actualS1(coreNidEnd.data());
    seqParams->set_actualCoreNums(coreSidStart.data());
    seqParams->set_singleCoreHeadNumSize(coreSidEnd.data());
    seqParams->set_coreSeqPosStart(coreSposStart.data());
    seqParams->set_coreSeqPosEnd(coreSposEnd.data());

    singleCoreParams->set_multiSmaxsInnerLoopTimes(multiSmaxsInnerLoopTimes);
    uint32_t actualCoreNums = curCore + 1;
    if (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) {
        actualCoreNums = actualCoreNums * CV_RATIO;
    }
    singleCoreParams->set_actualCoreNums(actualCoreNums);
}

void PromptFlashAttentionTiling::GetPreNextTokensLeftUp(PromptFlashAttentionTilingData& tilingData,
    uint32_t actualSeqLength, uint32_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) {
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    int64_t sparsePreTokens = baseParams->get_preTokens();
    int64_t sparseNextTokens = baseParams->get_nextTokens();
    if (baseParams->get_sparseMode() == SPARSE_MODE_RIGHT_DOWN) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        nextTokensLeftUp = (int64_t)actualSeqLengthKV - (int64_t)actualSeqLength;
    } else if (baseParams->get_sparseMode() == SPARSE_MODE_BAND) {
        preTokensLeftUp = sparsePreTokens - (int64_t)actualSeqLengthKV + (int64_t)actualSeqLength;
        nextTokensLeftUp = sparseNextTokens + (int64_t)actualSeqLengthKV - (int64_t)actualSeqLength;
    } else {
        preTokensLeftUp = sparsePreTokens;
        nextTokensLeftUp = sparseNextTokens;
    }
}

void PromptFlashAttentionTiling::SetSplitCoreMode(PromptFlashAttentionTilingData& tilingData, uint32_t sOuterFactor) {
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;

    uint32_t actualSeqLength = baseParams->get_seqSize();
    uint32_t actualSeqLengthKV = baseParams->get_seqInnerSize();
    uint32_t b = baseParams->get_dimNumOfseq();
    uint32_t n = baseParams->get_headNumSize();
    uint32_t d = baseParams->get_headSize();
    uint32_t sOuterSizeByCube = sOuterFactor * CV_RATIO;
    uint32_t sOuterLoopByCube = (actualSeqLength + sOuterSizeByCube - 1) / sOuterSizeByCube;
    const int64_t seq3K = 3 * 1024;   // 3 * 1024 : 3K.
    const int64_t seq8K = 8 * 1024;   // 8 * 1024 : 8K.
    const int64_t seq16K = 16 * 1024;  // 16 * 1024 : 16K.
    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;

    bool enableLeftPadding = ((contextKeyParamsPtr->queryPaddingSize != nullptr) || (contextKeyParamsPtr->kvPaddingSize != nullptr));
    bool enableRingAttention = (contextKeyParamsPtr->isSoftMaxLseEnable == true);

    GetPreNextTokensLeftUp(tilingData, actualSeqLength, actualSeqLengthKV, preTokensLeftUp, nextTokensLeftUp);
    bool inputTypeFp16 = (inputType == ge::DT_FLOAT16) && (contextKeyParamsPtr->kDataType == ge::DT_FLOAT16) && (outputType == ge::DT_FLOAT16);
    bool inputTypeBf16 = (inputType == ge::DT_BF16) && (contextKeyParamsPtr->kDataType == ge::DT_BF16) && (outputType == ge::DT_BF16);
    bool baseCond = (d == MATMUL_NORM_MIN_HEADSIZE) && (inputTypeFp16 || inputTypeBf16) && (usePseShift == 0) && !isKVHasPrefix &&
        !enableLeftPadding && !enableRingAttention && (baseParams->get_isActualSeqLengthsNull() == 1) && (baseParams->get_isActualSeqLengthsKVNull() == 1) && 
        (contextKeyParamsPtr->isKvContinuous == 1) && (actualSeqLength == actualSeqLengthKV) && (tilingMod == TilingMod::CVDIFF);
    bool enableOneNByCubeToken = true;
    bool enableNBSByCubeToken = true;
    if (contextKeyParamsPtr->attentionMask != nullptr) {
        enableOneNByCubeToken = (preTokensLeftUp >= actualSeqLength && nextTokensLeftUp >= actualSeqLengthKV) ||
                                (nextTokensLeftUp == 0);  // When mask exists, only support nextTokens is 0 or all data are calculated.
        enableNBSByCubeToken = ((preTokensLeftUp >= actualSeqLength) &&
                                (nextTokensLeftUp >= actualSeqLengthKV || nextTokensLeftUp == 0));  // When mask exists, only support the triangle scene or all data are calculated.
    }
    bool enableOneNByCubeSeqMode = actualSeqLength >= seq16K && (b * n >= 12);  // 12 : b * n should be more than 12.
    bool enableNBSByCubeSeqMode = actualSeqLength >= seq3K && (b * n * sOuterLoopByCube >= coreNum);
    bool noBalance = (baseParams->get_headNumRatio() != 1 || b != 1 || tilingData.promptAttentionInitOutputParams.get_needInit()) || 
                     (actualSeqLength >= seq8K && contextKeyParamsPtr->attentionMask == nullptr);
    if (baseCond && enableOneNByCubeToken && enableOneNByCubeSeqMode) {
        splitCoreMode = SplitCoreMode::SPLIT_ONEN_CUBE;
    } else if (baseCond && enableNBSByCubeToken && enableNBSByCubeSeqMode && noBalance) {
        splitCoreMode = SplitCoreMode::SPLIT_NBS_CUBE;
    }
}

void PromptFlashAttentionTiling::PromptFlashAttentionSplitSeqOneN(PromptFlashAttentionTilingData& tilingData,
                                                                  uint32_t curCoreNum, bool isVectorCore) {
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    PromptAttentionSingleCoreParams* singleCoreParams = &tilingData.promptAttentionSingleCoreParams;
    PromptAttentionSeqParams* seqParams = &tilingData.promptAttentionSeqParams;

    uint32_t actualSeqLength = baseParams->get_seqSize();
    uint32_t actualSeqLengthKV = baseParams->get_seqInnerSize();
    int64_t preTokensLeftUp;
    int64_t nextTokensLeftUp;
    GetPreNextTokensLeftUp(tilingData, actualSeqLength, actualSeqLengthKV, preTokensLeftUp, nextTokensLeftUp);

    uint32_t sOuterSize = singleCoreParams->get_singleProcessSOuterSize();
    if (!isVectorCore) {   // When viewed from the perspective of a cube, sOuter * 2 is used for kernel partitioning, and within each cube kernel, 2 vector kernels still receive sOuter.
        sOuterSize = sOuterSize * 2; // 2 : sOuter * 2 is used for kernel partitioning.
        curCoreNum = curCoreNum / 2; // 2 : within each cube kernel, 2 vector kernels still receive sOuter.
    }

    int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
    int64_t outerBlockFirstColNums = (preTokensLeftUp < static_cast<int32_t>(actualSeqLength)) ?
                                     ((preTokensLeftUp + sOuterSize - 1) / sOuterSize + 1) : outerBlockNums;
    int64_t outerBlockLeftDownFirstColNums = outerBlockNums - outerBlockFirstColNums;
    int64_t leftDownBlockNums = (outerBlockLeftDownFirstColNums + 1) * outerBlockLeftDownFirstColNums / 2;

    int64_t innerBlockNums = (actualSeqLengthKV + sOuterSize - 1) / sOuterSize;
    int64_t innerBlockFirstRowNums = (nextTokensLeftUp < static_cast<int32_t>(actualSeqLengthKV)) ?
                                     ((nextTokensLeftUp + sOuterSize - 1) / sOuterSize + 1) : innerBlockNums;
    int64_t innerBlockRightUpFirstRowNums = innerBlockNums - innerBlockFirstRowNums;
    int64_t rightUpBlockNums = (innerBlockRightUpFirstRowNums + 1) * innerBlockRightUpFirstRowNums / 2;

    int64_t toCalcBlockNums = innerBlockNums * outerBlockNums - rightUpBlockNums - leftDownBlockNums;
    double perWeight = static_cast<double>(toCalcBlockNums) / static_cast<double>(curCoreNum);

    uint32_t coreSOuterIndexStart[50] = {0};
    uint32_t coreSOuterIndexEnd[50] = {0};
    int64_t curWeight = 0;
    uint32_t coreIndex = 0;
    int64_t sInnerBlockNums = 0;
    int64_t sInnerIndexStart = 0;
    int64_t sInnerIndexEnd = innerBlockFirstRowNums;

    for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
        sInnerIndexStart = (sOuterIndex < outerBlockFirstColNums) ? 0 : (sInnerIndexStart + 1);
        sInnerBlockNums = sInnerIndexEnd - sInnerIndexStart;
        curWeight += sInnerBlockNums;
        if (curWeight >= (perWeight * (coreIndex + 1))) {
            coreSOuterIndexEnd[coreIndex] = sOuterIndex;
            coreIndex++;
            coreSOuterIndexStart[coreIndex] = sOuterIndex;
            if (coreIndex >= curCoreNum - 1) {
                coreSOuterIndexEnd[coreIndex] = outerBlockNums;
                break;
            }
        }
        sInnerIndexEnd = std::min(innerBlockNums, sInnerIndexEnd + 1);
    }

    // The situation where the nuclear allocation is not full.
    coreSOuterIndexEnd[coreIndex] = outerBlockNums;
    seqParams->set_coreSeqPosStart(coreSOuterIndexStart);
    seqParams->set_coreSeqPosEnd(coreSOuterIndexEnd);
    uint32_t actualCoreNums = coreIndex + 1;
    if (!isVectorCore) {
        actualCoreNums = actualCoreNums * 2;  // 2 : Split core
    }
    singleCoreParams->set_actualCoreNums(actualCoreNums);
}

bool PromptFlashAttentionTiling::EnableMTE2BmmPipe(PromptFlashAttentionTilingData& tilingData,
                                                   matmul_tiling::MatmulApiTiling& bmm, TCubeTiling& bmmTilingData,
                                                   uint32_t sOuterFactor, uint32_t sInnerFactor) {
    if (tilingData.promptAttentionBaseParams.get_seqSize() > 16) { // When the size is greater than 16, use xiaoe speculative inference.
        return true;
    }
    uint32_t baseK = 32U;
    uint32_t head_size = tilingData.promptAttentionBaseParams.get_headSize();
    if(head_size%baseK != 0) {
        return true;
    }

    uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
    uint32_t baseN = std::min(uint32_t(512), sInnerFactor);
    if (enablePA) {
        baseN = BLOCK_SIZE_BASE;
    }
    int32_t ret = 0;
    ret = bmm.SetFixSplit(baseM, baseN, baseK);
    OPS_ERR_IF(ret != 0,
               OPS_REPORT_VECTOR_INNER_ERR("PromptFlashAttention", "bmm SetFixSplit failed, ret = %d!", ret),
               return false);
    bool res = bmm.GetTiling(bmmTilingData) != -1;
    return res;
}

void PromptFlashAttentionTiling::EnableBmmDoubleBuffer(TCubeTiling& bmmTilingData) {
    if ((bmmTilingData.get_depthA1() == 1) && (bmmTilingData.get_depthB1() == 1)) {
        bmmTilingData.set_depthA1(2); // 2 : depthA1
        bmmTilingData.set_depthB1(2); // 2 : depthB1
    }
}

void PromptFlashAttentionTiling::PromptFlashAttention310PSetBmm1(matmul_tiling::MatmulApiTiling& bmm1)
{
    // 310p mm1: A gm ND, B gm ND, C vec NZ
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT16, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT16, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ,
                  matmul_tiling::DataType::DT_FLOAT16);
}

void PromptFlashAttentionTiling::PromptFlashAttention310PSetBmm2(matmul_tiling::MatmulApiTiling& bmm2)
{
    // 310p mm2: A vec NZ, B gm ND, C vec ND
    bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ,
                  matmul_tiling::DataType::DT_FLOAT16, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT16, false);
    bmm2.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT16);
}

ge::graphStatus PromptFlashAttentionTiling::CheckKeyValueParamsConsistency(const ContextParamsForPFATiling& contextKeyParams) {
    if (!contextKeyParams.isKvContinuous) {
        return GRAPH_SUCCESS;
    }

    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    const uint32_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    const uint32_t valueDimNum = valueShape->GetStorageShape().GetDimNum();

    OPS_ERR_IF(contextKeyParams.kDataType != contextKeyParams.vDataType,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "tensor key dtype(%d) must be consistent with tensor value dtype(%d)!", contextKeyParams.kDataType, contextKeyParams.vDataType),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(keyDimNum != valueDimNum,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "tensor key shape dimNum(%u) must be consistent with tensor value shape dimNum(%u)!", keyDimNum, valueDimNum),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((keyDimNum < INPUT_QKV_SHAPE_MIN_DIMS) || (keyDimNum > INPUT_QKV_SHAPE_MAX_DIMS),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "tensor key shape dimNum(%u) is invalid! Only support range [%u, %u]", keyDimNum, INPUT_QKV_SHAPE_MIN_DIMS, INPUT_QKV_SHAPE_MAX_DIMS),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool PromptFlashAttentionTiling::PromptFlashAttentionCheckBmm1(PromptFlashAttentionTilingData& tilingData,
    TCubeTiling& bmm1TilingData,  int64_t l1SizeRemain, int64_t l0CSize,
    uint32_t sOuterFactor, uint32_t sInnerFactor, bool allGM, bool autoBaseMNK) {
    int32_t ret = 0;
    matmul_tiling::MatmulApiTiling bmm1(ascendPlatformInfo);
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        PromptFlashAttention310PSetBmm1(bmm1);
    } else { // 910b
        matmul_tiling::DataType bmm1InputType = matmul_tiling::DataType::DT_FLOAT16;
        matmul_tiling::DataType bmm1OutputType = matmul_tiling::DataType::DT_FLOAT16;
        GetMatMulType(bmm1InputType, bmm1OutputType);
        matmul_tiling::TPosition cPosition = allGM ? matmul_tiling::TPosition::GM : matmul_tiling::TPosition::VECCALC;
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, true);
        bmm1.SetCType(cPosition, matmul_tiling::CubeFormat::ND, bmm1OutputType);
    }
    ret = bmm1.SetShape(sOuterFactor, sInnerFactor, tilingData.promptAttentionBaseParams.get_headSize());
    OPS_ERR_IF(ret != 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm1 SetShape failed, ret = %d!", ret),
                    return false);
    int32_t ratio = tilingData.promptAttentionBaseParams.get_headNumRatio();
    int32_t strideQ = tilingData.promptAttentionBaseParams.get_headSize() *
                        tilingData.promptAttentionBaseParams.get_headNumSize();
    int32_t strideK = strideQ / ratio;
    if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::SH) ||
        (inputLayout == InputLayout::BSND)) {
        bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                    tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                    strideQ, strideK);

        if (enableKvAntiquant) {
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                             tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                             strideQ, tilingData.promptAttentionBaseParams.get_headSize());
        } else if (enableMsd) {
            // Left input BNSD, right input BSH
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                             tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                             tilingData.promptAttentionBaseParams.get_headSize(), strideK);
        }
    } else if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) {
        if (enablePA && PAlayoutType == 1) {  // The left matrix of PA is BNSD, and the right matrix is BSH.
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                     tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                     tilingData.promptAttentionBaseParams.get_headSize(), strideK);
        } else {
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                     tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                     tilingData.promptAttentionBaseParams.get_headSize());
        }
    }

    bmm1.SetBias(false);
    ret = bmm1.SetBufferSpace(l1SizeRemain, l0CSize);
    OPS_ERR_IF(ret != 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm1 SetBufferSpace failed, l1SizeRemain = %ld, l0CSize = %ld, ret = %d!",
                    l1SizeRemain, l0CSize, ret),
                    return false);
    if (enablePA) {
        ret = bmm1.SetFixSplit(sOuterFactor, BLOCK_SIZE_BASE);
    } else {
        ret = bmm1.SetFixSplit(sOuterFactor, sInnerFactor);
    }
    OPS_ERR_IF(ret != 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm1 SetFixSplit failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, ret = %d!",
                    l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, ret),
                    return false);
    if (inputType == ge::DT_INT8) {
        bmm1.SetDequantType(matmul_tiling::DequantType::SCALAR);
    }

    ret = bmm1.GetTiling(bmm1TilingData);
    if (autoBaseMNK) {
        if (enableMatmulNorm || splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE || splitCoreMode == SplitCoreMode::SPLIT_ONEN_CUBE) {
            uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
            uint32_t baseN = std::min(uint32_t(128), sInnerFactor);
            uint32_t baseK = 128U;
            ret = bmm1.SetFixSplit(baseM, baseN, baseK);
            OPS_ERR_IF(ret != 0,
                       OPS_REPORT_VECTOR_INNER_ERR("PromptFlashAttention", "bmm1 SetFixSplit failed, ret = %d!", ret),
                       return false);
            ret = bmm1.GetTiling(bmm1TilingData);
        } else {
            uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
            uint32_t baseN = std::min(uint32_t(256), sInnerFactor);
            uint32_t baseK = 64U;
            if (enablePA) {
                baseN = BLOCK_SIZE_BASE;
            }
            if (ret != 0) {
                ret = bmm1.SetFixSplit(baseM, baseN, baseK);
                OPS_ERR_IF(ret != 0,
                           OPS_REPORT_VECTOR_INNER_ERR("PromptFlashAttention", "bmm1 SetFixSplit failed, ret = %d!", ret),
                           return false);
                ret = bmm1.GetTiling(bmm1TilingData);
            }
        }
    }

    OPS_ERR_IF(ret != 0, // Get tiling fail for bmm1.
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm1 GetTiling failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, autoBaseMNK = %d, ret = %d!",
                    l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, autoBaseMNK, ret),
                    return false);

    bmm1TilingData.set_shareMode(0);
    bmm1TilingData.set_shareL1Size(l1SizeRemain);
    bmm1TilingData.set_shareL0CSize(l0CSize);
    if (curShortSocName != platform_ascendc::SocVersion::ASCEND310P) {
        bmm1TilingData.set_shareUbSize(0);
        EnableBmmDoubleBuffer(bmm1TilingData); // Open the double buffer for BMM1 calculation, and BMM1's MTE2 can be bound.
    }

    bool res = EnableMTE2BmmPipe(tilingData, bmm1, bmm1TilingData, sOuterFactor, sInnerFactor);  // Open MTE2 Matmul pipeline.

    OPS_ERR_IF(res == false,     // EnableMTE2BmmPipe fail.
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "EnableMTE2BmmPipe failed!"),
                    return false);

    return true;
}

void PromptFlashAttentionTiling::GetMatMulType(matmul_tiling::DataType &mmInputType,
    matmul_tiling::DataType &mmOutputType) {
    if (inputType == ge::DT_FLOAT16 && innerPrecise == HIGH_PRECISION) {
        mmInputType = matmul_tiling::DataType::DT_FLOAT16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (inputType == ge::DT_BF16) {
        mmInputType = matmul_tiling::DataType::DT_BF16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (inputType == ge::DT_INT8) {
        mmInputType = matmul_tiling::DataType::DT_INT8;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT16;
    }
}

bool PromptFlashAttentionTiling::PromptFlashAttentionCheckBmm2(PromptFlashAttentionTilingData& tilingData,
    TCubeTiling& bmm2TilingData,  int64_t l1SizeRemain, int64_t l0CSize,
    uint32_t sOuterFactor, uint32_t sInnerFactor, uint32_t dSplitFactor, bool allGM, bool autoBaseMNK) {
    int32_t ret = 0;
    matmul_tiling::MatmulApiTiling bmm2(ascendPlatformInfo);
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        PromptFlashAttention310PSetBmm2(bmm2);
        ret = bmm2.SetShape(sOuterFactor, tilingData.promptAttentionBaseParams.get_headSize(), sInnerFactor);
        if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND)) {
            int32_t ratio = tilingData.promptAttentionBaseParams.get_headNumRatio();
            int32_t strideQ = tilingData.promptAttentionBaseParams.get_headSize() *
                            tilingData.promptAttentionBaseParams.get_headNumSize();
            int32_t strideV = strideQ / ratio;
            bmm2.SetOrgShape(sOuterFactor, strideV, sInnerFactor,
                            tilingData.promptAttentionBaseParams.get_seqInnerSize());
        } else if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) { // M, N, KA, KB
            bmm2.SetOrgShape(sOuterFactor, tilingData.promptAttentionBaseParams.get_headSize(), sInnerFactor,
                            tilingData.promptAttentionBaseParams.get_seqInnerSize());
        }
    } else { // This is for 910B.
        matmul_tiling::DataType bmm2InputType = matmul_tiling::DataType::DT_FLOAT16;
        matmul_tiling::DataType bmm2OutputType = matmul_tiling::DataType::DT_FLOAT16;
        GetMatMulType(bmm2InputType, bmm2OutputType);
        if ((splitS2 == 1) && (splitD == 1)) {
            bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
            bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
            bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutputType);
            ret = bmm2.SetShape(sOuterFactor, tilingData.promptAttentionBaseParams.get_headSize(),
                            tilingData.promptAttentionBaseParams.get_seqInnerSize());
        } else {
            matmul_tiling::TPosition aPosition = allGM ? matmul_tiling::TPosition::GM : matmul_tiling::TPosition::TSCM;
            matmul_tiling::TPosition cPosition = allGM ? matmul_tiling::TPosition::GM : matmul_tiling::TPosition::VECCALC;
            bmm2.SetAType(aPosition, matmul_tiling::CubeFormat::NZ, bmm2InputType, false);
            bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
            bmm2.SetCType(cPosition, matmul_tiling::CubeFormat::ND_ALIGN, bmm2OutputType);
            ret = bmm2.SetShape(sOuterFactor, tilingData.promptAttentionBaseParams.get_headSize(), sInnerFactor);
        }
    OPS_ERR_IF(ret != 0,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm2 set SetShape failed, sOuterFactor = %u, sInnerFactor = %u, ret = %d!",
                sOuterFactor, sInnerFactor, ret),
                return false);
    int32_t ratio = tilingData.promptAttentionBaseParams.get_headNumRatio();
    int32_t strideQ = tilingData.promptAttentionBaseParams.get_headSize() *
                    tilingData.promptAttentionBaseParams.get_headNumSize();
    int32_t strideV = strideQ / ratio;
        if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) ||
            (inputLayout == InputLayout::SH)) {
            bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(), strideV,
                            tilingData.promptAttentionBaseParams.get_seqInnerSize());
            if (enableKvAntiquant) {
                bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                                 tilingData.promptAttentionBaseParams.get_headSize(),
                                 tilingData.promptAttentionBaseParams.get_seqInnerSize());
            }
        } else if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) {
            if (enablePA && PAlayoutType == 1) {  // The left matrix of PA is BNSD, and the right matrix is of PA is BSH.
                bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(), strideV,
                            tilingData.promptAttentionBaseParams.get_seqInnerSize());
            } else {
                bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                            tilingData.promptAttentionBaseParams.get_headSize(),
                            tilingData.promptAttentionBaseParams.get_seqInnerSize());
            }
        }
    }

    bmm2.SetBias(false);
    ret = bmm2.SetBufferSpace(l1SizeRemain, l0CSize);
    OPS_ERR_IF(ret != 0,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm2 set SetBufferSpace failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, ret = %d!",
                l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, ret),
                return false);
    if (inputType == ge::DT_INT8) {
        bmm2.SetDequantType(matmul_tiling::DequantType::SCALAR);
    }

    if (autoBaseMNK) {
        if (enableMatmulNorm || splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE || splitCoreMode == SplitCoreMode::SPLIT_ONEN_CUBE) {
            uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
            uint32_t baseN = std::min(uint32_t(128), tilingData.promptAttentionBaseParams.get_headSize());
            uint32_t baseK = 128U;
            ret = bmm2.SetFixSplit(baseM, baseN, baseK);
            OPS_ERR_IF(ret != 0,
                       OPS_REPORT_VECTOR_INNER_ERR("PromptFlashAttention", "bmm2 SetFixSplit failed, ret = %d!", ret),
                       return false);
        }
        ret = bmm2.GetTiling(bmm2TilingData);
    } else {
        if ((isDNoTail) || (splitS2 == 0) || (splitD == 1)) {
            ret = bmm2.SetFixSplit(sOuterFactor, dSplitFactor);
        } else {
            ret = bmm2.SetFixSplit(sOuterFactor, tilingData.promptAttentionBaseParams.get_alignedHeadSize());
        }
        OPS_ERR_IF(ret != 0,
                   OPS_REPORT_VECTOR_INNER_ERR("PromptFlashAttention", "bmm2 SetFixSplit failed, ret = %d!", ret),
                   return false);
        ret = bmm2.GetTiling(bmm2TilingData);
    }
    OPS_ERR_IF(ret != 0,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm2 set GetTiling failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, autoBaseMNK = %d, ret = %d!",
                l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, autoBaseMNK, ret),
                return false);
    bmm2TilingData.set_shareMode(0);
    bmm2TilingData.set_shareL1Size(l1SizeRemain);
    bmm2TilingData.set_shareL0CSize(l0CSize);
    OPS_ERR_IF(ret != 0,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm2 set shareL0CSize failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, autoBaseMNK = %d, ret = %d!",
                l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, autoBaseMNK, ret),
                return false);
    if (curShortSocName != platform_ascendc::SocVersion::ASCEND310P) {
         bmm2TilingData.set_shareUbSize(0);
    }
    return true;
}

void PromptFlashAttentionTiling::PromptFlashAttentionSetTensorSize(
    PromptFlashAttentionTilingData& tilingData,
    PromptAttentionSingleCoreTensorSize& tensorSize,
    uint32_t sOuterFactor, uint32_t sInnerFactor) {
    if (tilingData.promptAttentionBaseParams.get_useMask() == 0U && usePseShift == 0U) {
        // In scenarios where attentionMask is not configured and there is no pse, UB memory for attentionMask can be saved
        // But 2 BYTE_BLOCK (32BYTE) UB memory needs to be reserved for Bmm2UpdateDiv
        tensorSize.set_attenMaskUbSize(sOuterFactor * BYTE_BLOCK * NUM_2 / softmaxDataTypeSize);
    } else {
        tensorSize.set_attenMaskUbSize(sOuterFactor * sInnerFactor);
    }

    if (usePseShift == 0U) {
        tensorSize.set_pseShiftUbSize(0);
    } else {
        tensorSize.set_pseShiftUbSize(sOuterFactor * sInnerFactor);
    }

    if (enableMsd){
        if (tilingData.promptAttentionBaseParams.get_headSize() > MSD_BIG_D) {
            tensorSize.set_mmResUbSize(COMPUTELINE_FOR_BIG_D * sInnerFactor * 2); // 2:double buffer
        } else {
            tensorSize.set_mmResUbSize(CVDIFF_SMALL_QS_THRESHOLDS * CVDIFF_MSD_BUFFER_SIZE_1024B / sizeof(int32_t));
        }
    } else {
        tensorSize.set_mmResUbSize(sOuterFactor * sInnerFactor);
    }

    tensorSize.set_maskSize(tensorSize.get_mmResUbSize());
    tensorSize.set_softmaxSumSize(tensorSize.get_softmaxMaxSize());
    
    if (enableMsd) {
        tensorSize.set_softmaxExpSize(MSD_UB_BASE_WIDTH * ONE_BLK_SIZE_PFA);
    } else {
        tensorSize.set_softmaxExpSize(sOuterFactor * tilingData.promptAttentionBaseParams.get_softmaxTypeByteNum());
    }
    tensorSize.set_softmaxValueSize(sOuterFactor * sInnerFactor);
    if (enableMsd) {
        if (tilingData.promptAttentionBaseParams.get_headSize() > MSD_BIG_D) {
            tensorSize.set_bmm2ResUbSize(MAX_COMPUTELINES * tilingData.promptAttentionBaseParams.get_alignedHeadSize());
        } else {
             tensorSize.set_bmm2ResUbSize(MSD_UB_BASE_WIDTH * MSD_UB_HEGHT);
        }
     } else {
        tensorSize.set_bmm2ResUbSize(sOuterFactor * tilingData.promptAttentionBaseParams.get_alignedHeadSize());
    }
    tensorSize.set_tmpMMResBmm2PreUbSize(std::max(tensorSize.get_mmResUbSize(), tensorSize.get_bmm2ResUbSize()));
    tensorSize.set_tmpSoftmaxBmm2UbSize(SOFTMAX_BUFFER_NUM * tensorSize.get_softmaxMaxSize());
    if ((splitS2 == 1) && (splitD == 1)) {
        tensorSize.set_spmTmpSize(tensorSize.get_bmm2ResUbSize() + tensorSize.get_softmaxExpSize() * SPLIT_DOUBLE_UB);
    } else {
        tensorSize.set_spmTmpSize(tensorSize.get_bmm2ResUbSize());
    }
    // 310P needs tscm buf
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        tensorSize.set_scmTmpSize(tilingData.promptAttentionBaseParams.get_headSize() * std::max(sOuterFactor, sInnerFactor));
        tensorSize.set_softmaxMaxSize(sOuterFactor * (BYTE_BLOCK / softmaxDataTypeNZ_));
    } else {
        if (enableMsd) {
            tensorSize.set_softmaxMaxSize(MSD_UB_BASE_WIDTH * ONE_BLK_SIZE_PFA);
        } else {
            tensorSize.set_softmaxMaxSize(sOuterFactor * (BYTE_BLOCK / sizeof(float)));
        }
    }
    if (tilingData.promptAttentionBaseParams.get_maskTypeByteNum() == (BYTE_BLOCK / BOOLSIZE)) {
        tensorSize.set_selectSpaceUbSize(GetSelectWithBytesMaskMinTmpSize(Shape({sOuterFactor, sInnerFactor}), Shape({1}), 1,
            Shape({sOuterFactor, sInnerFactor}), 1, false));
    } else {
        tensorSize.set_selectSpaceUbSize(0);
    }
}

int64_t PromptFlashAttentionTiling::PromptFlashAttentionSetMsdUbSize(PromptFlashAttentionTilingData& tilingData, PromptAttentionSingleCoreTensorSize& tensorSize, int32_t sInnerFactorTmp) const
{
    int64_t msdUbSize =  0;
    if (enableMsd) {
        if (tilingData.promptAttentionBaseParams.get_headSize() > MSD_BIG_D) {
            int64_t msdTmpMmBufferSize = std::max(COMPUTELINE_FOR_BIG_D * sInnerFactorTmp * sizeof(float),
                                                  MAX_COMPUTELINES * tilingData.promptAttentionBaseParams.get_headSize() * sizeof(float)); // 2:double buffer
            tensorSize.set_msdInQueueSize(MAX_COMPUTELINES * tilingData.promptAttentionBaseParams.get_headSize() * FLOAT16SIZE);
            tensorSize.set_msdQRowSumBuffSize(MAX_COMPUTELINES * 8 * sizeof(float)); // 8:param of ub
            tensorSize.set_msdAMaxTmpBuffSize(MAX_COMPUTELINES * 8 * sizeof(float)); // 8:param of ub
            tensorSize.set_msdAMaxResBuffSize(MAX_COMPUTELINES * 8 * sizeof(float)); // 8:param of ub
            tensorSize.set_msdSoftmaxResAmaxBuffSize(MAX_COMPUTELINES * 8 * sizeof(float)); // 8:param of ub
            tensorSize.set_msdSoftmaxRowSumScaleBuffSize(MAX_COMPUTELINES * 8 * sizeof(float)); // 8:param of ub
            tensorSize.set_msdScaleBuffSize(tilingData.promptAttentionBaseParams.get_headSize() * sizeof(float));
            tensorSize.set_msdOffsetBuffSize(tilingData.promptAttentionBaseParams.get_headSize() * sizeof(float));
            tensorSize.set_msdTmpMm1BuffSize(msdTmpMmBufferSize);
            tensorSize.set_msdTmpMm2BuffSize(msdTmpMmBufferSize);
            tensorSize.set_msdOutQueueSize(msdTmpMmBufferSize / 2); // 2:half of msdOutQueueSize

            int64_t computeLines = COMPUTELINE_FOR_BIG_D;
            tilingData.promptAttentionTensorSizeRect.set_msdComputeLines(computeLines);
        } else {
            tensorSize.set_msdInQueueSize(CVDIFF_SINNER_FACTOR_DEFAULT * MSD_UB_INQUEUE);
            tensorSize.set_msdQRowSumBuffSize(CVDIFF_MSD_BUFFER_SIZE_512B);
            tensorSize.set_msdAMaxTmpBuffSize(CVDIFF_MSD_BUFFER_SIZE_512B);
            tensorSize.set_msdAMaxResBuffSize(CVDIFF_MSD_BUFFER_SIZE_512B);
            tensorSize.set_msdSoftmaxResAmaxBuffSize(CVDIFF_MSD_BUFFER_SIZE_512B);
            tensorSize.set_msdSoftmaxRowSumScaleBuffSize(CVDIFF_MSD_BUFFER_SIZE_512B);
            tensorSize.set_msdScaleBuffSize(CVDIFF_SINNER_FACTOR_DEFAULT);
            tensorSize.set_msdOffsetBuffSize(CVDIFF_SINNER_FACTOR_DEFAULT);
            tensorSize.set_msdTmpMm1BuffSize(CVDIFF_SINNER_FACTOR_DEFAULT * MSD_UB_TMP_NM);
            tensorSize.set_msdTmpMm2BuffSize(CVDIFF_SINNER_FACTOR_DEFAULT * MSD_UB_TMP_NM);
            tensorSize.set_msdOutQueueSize((CVDIFF_SINNER_FACTOR_DEFAULT * MSD_UB_TMP_NM) / 2); // 2:half of msdTmpMm1BuffSize
        }

        // msd UB size
        msdUbSize = static_cast<int64_t>(tensorSize.get_msdInQueueSize()) + static_cast<int64_t>(tensorSize.get_msdQRowSumBuffSize()) * NUM_2 + static_cast<int64_t>(tensorSize.get_msdAMaxResBuffSize()) * NUM_2 + 
                    static_cast<int64_t>(tensorSize.get_msdAMaxTmpBuffSize()) + static_cast<int64_t>(tensorSize.get_msdSoftmaxResAmaxBuffSize()) + 
                    static_cast<int64_t>(tensorSize.get_msdSoftmaxRowSumScaleBuffSize()) + static_cast<int64_t>(tensorSize.get_msdScaleBuffSize()) + static_cast<int64_t>(tensorSize.get_msdOffsetBuffSize()) + 
                    static_cast<int64_t>(tensorSize.get_msdTmpMm1BuffSize()) + static_cast<int64_t>(tensorSize.get_msdTmpMm2BuffSize()) + static_cast<int64_t>(tensorSize.get_msdOutQueueSize());
     }
 
    return msdUbSize;
}

uint32_t PromptFlashAttentionTiling::CalculateL1SizeUsed(PromptFlashAttentionTilingData& tilingData, const uint32_t typeByteSize)
{
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        return (typeByteSize * tilingData.promptAttentionTensorSizeRect.get_scmTmpSize() * 3); // 3：Two extra tscm buffers are needed for a1, b1 or b1, b2.
    }
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND910B) {
        return (typeByteSize * tilingData.promptAttentionTensorSizeRect.get_scmTmpSize());
    }
    return 0;
}

bool PromptFlashAttentionTiling::PromptFlashAttentionCheckArgsLegal(PromptFlashAttentionTilingData& tilingData,
    int64_t ubSize, int64_t l1Size, int64_t l0CSize, uint32_t typeByteSize,
    uint32_t& sOuterFactor, uint32_t sInnerFactor,
    bool& updateDiv, uint32_t maskTypeSize, uint32_t dSplitFactor) {
    // Adjusting basic blocks
    bool res = true;
    if (AdjustBasicBlock(tilingData, sOuterFactor) != ge::GRAPH_SUCCESS) {
            return false;
    }
    auto tmpShape = Shape({sOuterFactor, sInnerFactor});  // [S,s]
    int64_t softmaxTmpSize = GetSoftMaxMinTmpSize(tmpShape, typeByteSize, true);
    int64_t softmaxFlashTmpSize = GetSoftMaxFlashV2MinTmpSize(tmpShape, typeByteSize, softmaxDataTypeNZ_, true, true);
    if ((softmaxTmpSize == 0) || (softmaxFlashTmpSize == 0)) {
        return false;
    }

    int64_t queueBufferSize = 0;
    int64_t pseShiftBufferSize = 0;
    int64_t msdUbSize = 0;

    PromptFlashAttentionSetTensorSize(tilingData, tilingData.promptAttentionTensorSizeRect,
                                        sOuterFactor, sInnerFactor);
    msdUbSize = PromptFlashAttentionSetMsdUbSize(tilingData, tilingData.promptAttentionTensorSizeRect, static_cast<int32_t>(sInnerFactor));
    int32_t l1SizeRemain = l1Size - CalculateL1SizeUsed(tilingData, typeByteSize);
    if (l1SizeRemain < 0) {
        updateDiv = true;
        res = false;
        return res;
    }

    res = (PromptFlashAttentionCheckBmm1(tilingData, tilingData.bmm1TilingDataRect,
        l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor)) &&
        (PromptFlashAttentionCheckBmm2(tilingData, tilingData.bmm2TilingDataRect,
        l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, dSplitFactor));

    OPS_ERR_IF(res == false,
               OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "PromptFlashAttentionCheckBmm1 or PromptFlashAttentionCheckBmm2 failed."),
               return false);

    queueBufferSize = tilingData.promptAttentionTensorSizeRect.get_attenMaskUbSize();

    pseShiftBufferSize = tilingData.promptAttentionTensorSizeRect.get_pseShiftUbSize();

    pseMaskMaxSize = std::max(maskTypeSize, pseShiftElemSize);

    uint32_t pseShiftCastSize = 0U;
    if ((usePseShift == 1) && (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) || pseShiftElemType == ge::DT_BF16)) {
        pseShiftCastSize = FLOAT32SIZE;   // In the case of high-precision effectiveness or BF16, PSE needs to do a cast and apply for UB
    }

    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        matmul_tiling::SysTilingTempBufSize mm1bufSize, mm2bufSize;
        int32_t getBufRes;
        apiTmpSize = GetApiTmpSize(sOuterFactor, sInnerFactor, typeByteSize);
        getBufRes = MatmulGetTmpBufSize(tilingData.bmm1TilingDataRect, mm1bufSize);
        getBufRes += MatmulGetTmpBufSize(tilingData.bmm2TilingDataRect, mm2bufSize);
        if (getBufRes < 0) {
            updateDiv = true;
            res = false;
            return res;
        }
        ubSizeRemain = ubSize - (apiTmpSize +
                    tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() +
                    tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() * 2 + // 2:2 mm2 ub
                    SOFTMAX_BUFFER_NUM * tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize()) *
                    typeByteSize - tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize() * 4 - queueBufferSize * maskTypeSize * 2;    // 4: Multiply the obtained softmaxExpSize by 4, 2: Multiply maskTypeSize by 2
        tilingData.promptAttentionTensorSizeRect.set_tmpSoftMaxV2Size((ubSizeRemain + apiTmpSize) / UB_ALIGN * UB_ALIGN);
        tilingData.promptAttentionTensorSizeRect.set_mm1TmpUbSize(mm1bufSize.ubSize);
        tilingData.promptAttentionTensorSizeRect.set_mm2TmpUbSize(mm2bufSize.ubSize);
    } else {
        apiTmpSize = std::max(softmaxTmpSize, softmaxFlashTmpSize);
        if ((splitS2 == 1) && (splitD == 1)) {
            ubSizeRemain = ubSize - apiTmpSize - (tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() * SPLIT_DOUBLE_UB +
                                    tilingData.promptAttentionTensorSizeRect.get_softmaxValueSize() +
                                    SOFTMAX_BUFFER_NUM * tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize()) *
                                    typeByteSize - (queueBufferSize * pseMaskMaxSize) -
                                    tilingData.promptAttentionTensorSizeRect.get_selectSpaceUbSize() -
                                    pseShiftBufferSize * pseShiftCastSize - msdUbSize;
        } else if ((splitS2 == 1) && (splitD == 0)) {
            ubSizeRemain = ubSize - apiTmpSize - (tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() +
                                    tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() * NUM_2 + // 2:2 mm2 ub
                                    SOFTMAX_BUFFER_NUM * tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize()) *
                                    typeByteSize - (queueBufferSize * pseMaskMaxSize) -
                                    tilingData.promptAttentionTensorSizeRect.get_selectSpaceUbSize() -
                                    pseShiftBufferSize * pseShiftCastSize - msdUbSize;
        } else {
            ubSizeRemain = ubSize - apiTmpSize - (tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() +
                                    SPLIT_DOUBLE_UB * tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize()) *
                                    typeByteSize - (queueBufferSize * pseMaskMaxSize) -
                                    tilingData.promptAttentionTensorSizeRect.get_selectSpaceUbSize() -
                                    pseShiftBufferSize * pseShiftCastSize - msdUbSize;
        }
    }

    if (ubSizeRemain < 0) {
        updateDiv = true;
        res = false;
        return res;
    }
    updateDiv = (!res);
    return res;
}

ge::graphStatus PromptFlashAttentionTiling::PromptFlashAttentionApiTiling(PromptFlashAttentionTilingData& tilingData,
    uint32_t typeSize,  uint32_t sOuterFactor, uint32_t softmaxSInnerFactor, uint32_t softmaxSOuterFactor) {
    auto softmaxShapeRect = Shape({softmaxSOuterFactor, softmaxSInnerFactor});

    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        uint32_t sftV2Size = GetSoftMaxFlashV2MinTmpSize(softmaxShapeRect, softmaxDataTypeNZ_, softmaxDataTypeNZ_, true);
        if ((ubSizeRemain + apiTmpSize) < sftV2Size) {
            return ge::GRAPH_FAILED;
        }
        SoftMaxFlashV2TilingFunc(softmaxShapeRect, softmaxDataTypeNZ_, softmaxDataTypeNZ_, (ubSizeRemain + apiTmpSize) / UB_ALIGN * UB_ALIGN,
            tilingData.softmaxTilingDataRect, true);
    } else {
        SoftMaxTilingFunc(softmaxShapeRect, sizeof(float), ubSizeRemain + apiTmpSize, tilingData.softmaxTilingDataRect);
        SoftMaxFlashV2TilingFunc(softmaxShapeRect, softmaxDataTypeSize, sizeof(float), ubSizeRemain + apiTmpSize,
            tilingData.softmaxFlashTilingDataRect, true, true);
    }

    auto transposeSrcShapeRect = Shape({1, 1, sOuterFactor,
                                      tilingData.promptAttentionBaseParams.get_headSize()});
    auto transposeDstShape = Shape({tilingData.promptAttentionBaseParams.get_batchSize(),
                                      tilingData.promptAttentionBaseParams.get_headNumSize(),
                                      tilingData.promptAttentionBaseParams.get_seqSize(),
                                      tilingData.promptAttentionBaseParams.get_headSize() *
                                      tilingData.promptAttentionBaseParams.get_headNumSize()});

    GetDataCopyTransposeTiling(transposeDstShape, transposeSrcShapeRect, typeSize, tilingData.transposeTilingDataRect);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::PromptFlashAttentionSetTilingData(gert::TilingContext* context,
    PromptFlashAttentionTilingData& tilingData) {
    if (mlaRunFlag_) {
        mlaTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(mlaTilingData.GetDataSize());
    } else {
        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::GetRectangleFactor(uint32_t seqFactorThreshold,
    std::queue<uint32_t>& sQueue, int32_t threshold) {
    for (int i = seqFactorThreshold; i >= threshold ; i = (i - threshold)) { // threshold 16
        sQueue.push(i);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::SetInputLayout(const char* layout){
    if (layout == nullptr){
        inputLayout = InputLayout::BSH;
        return ge::GRAPH_SUCCESS;
    }

    std::string layoutStr(layout);
    if (layoutStr == "") {
        inputLayout = InputLayout::BSH;
    } else if (layoutStr == "SH") {
        inputLayout = InputLayout::SH;
    } else if (layoutStr == "BSH") {
        inputLayout = InputLayout::BSH;
    } else if (layoutStr == "NSD") {
        inputLayout = InputLayout::NSD;
    } else if (layoutStr == "BSND") {
        inputLayout = InputLayout::BSND;
    } else if (layoutStr == "BNSD") {
        inputLayout = InputLayout::BNSD;
    } else if (layoutStr == "BNSD_BSND") { // Reuse BNSD process for BNSD_BSND
        inputLayout = InputLayout::BNSD;
    } else if (layoutStr == "TND") {
        inputLayout = InputLayout::TND;
    } else {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool PromptFlashAttentionTiling::CheckInputDimAndHeadNum(ContextParamsForPFATiling& contextKeyParams, const uint32_t nQAttr, const uint32_t nKVAttr) {
    uint32_t nQ = nQAttr;
    uint32_t nKV = nKVAttr;
    if (nKVAttr == 0U) { // Detected that nKVAttr is the default value, which means that the customer did not pass in.
        nKV = nQAttr;
    }

    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    uint32_t queryShapeHeadNum = nQ;
    uint32_t keyShapeHeadNum = nKV;
    uint32_t valueShapeHeadNum = nKV;
    const uint32_t queryDim = queryShape->GetStorageShape().GetDimNum();
    const uint32_t keyDim = keyShape->GetStorageShape().GetDimNum();
    const uint32_t valueDim = valueShape->GetStorageShape().GetDimNum();
    const uint32_t nIdx = inputLayout == InputLayout::BNSD ? 1U : 2U; // BNSD: 1; BSND:2

    if (((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::BSND)) && (!enablePA)) {
        if ((queryDim == 4) && (keyDim == 4) && (valueDim == 4)) { // dim num: 4
            queryShapeHeadNum = queryShape->GetStorageShape().GetDim(nIdx);
            keyShapeHeadNum = keyShape->GetStorageShape().GetDim(nIdx);
            valueShapeHeadNum = valueShape->GetStorageShape().GetDim(nIdx);
        } else {
            OPS_LOG_E(contextKeyParams.opName, "input dim of q(%u), k(%u), v(%u) must be 4 for BNSD or BSND format!", queryDim, keyDim, valueDim);
            return false;
        }
    } else if ((inputLayout == InputLayout::NSD) && (!enablePA)) {
        if ((queryDim == 3) && (keyDim == 3) && (valueDim == 3)) { // dim num: 3
            queryShapeHeadNum = queryShape->GetStorageShape().GetDim(0);
            keyShapeHeadNum = keyShape->GetStorageShape().GetDim(0);
            valueShapeHeadNum = valueShape->GetStorageShape().GetDim(0);
        } else {
            OPS_LOG_E(contextKeyParams.opName, "input dim of q(%u), k(%u), v(%u) must be 3 for NSD format!", queryDim, keyDim, valueDim);
            return false;
        }
    }

    OPS_ERR_IF(nQ > 256U,   // The maximum limit for head is 256.
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "numHeads(%u) should not be more than 256!", nQ),
                    return false);

    OPS_ERR_IF(queryShapeHeadNum != nQ,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "numHeads(%u) in query shape must be equal to numHeads(%u) in attr!", queryShapeHeadNum, nQ),
                    return false);
    OPS_ERR_IF(keyShapeHeadNum != nKV,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "numHeads(%u) in key shape do not match numKeyValueHeads(%u) in attr!", keyShapeHeadNum, nKV),
                    return false);
    OPS_ERR_IF(valueShapeHeadNum != nKV,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "numHeads(%u) in value shape do not match numKeyValueHeads(%u) in attr!", valueShapeHeadNum, nKV),
                    return false);
    return true;
}

bool PromptFlashAttentionTiling::SetTilingHeadNumRatio(ContextParamsForPFATiling& contextKeyParams,
                                                       const int32_t* numQueryHeads, const int32_t* numKeyValueHeads,
                                                       PromptFlashAttentionTilingData& tilingData) {
    const int32_t nQ = *numQueryHeads;
    const int32_t nKV = *numKeyValueHeads;

    if ((nQ < 0) || (nKV < 0)) {
        OPS_LOG_E(contextKeyParams.opName, "numHeads(%d) or numKeyValueHeads(%d) is negative!", nQ, nKV);
        return false;
    }

    if (!CheckInputDimAndHeadNum(contextKeyParams, nQ, nKV)) {
        return false;
    }

    if (nKV == 0) { // Detected that nKV is the default value, which means that the customer did not pass in.
        tilingData.promptAttentionBaseParams.set_headNumRatio(1);
        return true;
    }

    if (nQ % nKV != 0) {
        OPS_LOG_E(contextKeyParams.opName, "numHeads(%d) must be divisible by numKeyValueHeads(%d)!", nQ, nKV);
        return false;
    } else {
        if (nQ / nKV > 64) {   // G cannot be greater than 64.
            OPS_LOG_E(contextKeyParams.opName, "numHeads / numKeyValueHeads = %d, cannot be larger than 64", nQ / nKV);
            return false;
        }
        tilingData.promptAttentionBaseParams.set_headNumRatio(nQ / nKV);
        return true;
    }
}

bool PromptFlashAttentionTiling::CheckNonEmptyShapeExceptions(ContextParamsForPFATiling& contextKeyParams,
                                                              const gert::StorageShape* shape,
                                                              const std::string &sName) {
    OPS_ERR_IF(shape == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "%s shape is null.", sName.c_str()),
                    return true);
    OPS_ERR_IF(shape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "Shape size of %s is overflow.", sName.c_str()),
                    return true);
    return false;
}

bool PromptFlashAttentionTiling::CheckActualSeqLength(ContextParamsForPFATiling& contextKeyParams, uint32_t b, uint32_t sQ, uint32_t sKV,
                                                      const gert::Tensor* actualSeqLenQ, const gert::Tensor* actualSeqLenKV,
                                                      InputLayout inLayout, PromptFlashAttentionTilingData& tilingData) {
    if (contextKeyParams.fromTilingSink != 0) {
        return true;
    }
    uint64_t actualLenDimsQ  = (actualSeqLenQ  != nullptr) ? actualSeqLenQ->GetShapeSize()  : 0;
    uint64_t actualLenDimsKV = (actualSeqLenKV != nullptr) ? actualSeqLenKV->GetShapeSize() : 0;
    bool inputActualSeqQ  = !((actualLenDimsQ  == 0) || (actualSeqLenQ  == nullptr) || (actualSeqLenQ->GetData<int64_t>()  == nullptr));
    bool inputActualSeqKV = !((actualLenDimsKV == 0) || (actualSeqLenKV == nullptr) || (actualSeqLenKV->GetData<int64_t>() == nullptr));
    int64_t actualSeqQSum = 0; 
    int64_t actualSeqTmp = 0; // The element of actualSeq.
    constexpr uint64_t actualLenDimsQMin = 1; // The length of actualSeqQ is 1
    constexpr uint64_t actualLenDimsKVMin = 1; // The length of actualSeqKV is 1

    // SH format verification separately.
    if (inLayout == InputLayout::SH) {
        if (inputActualSeqQ) {
            for (uint32_t i = LOOP_BEGIN_NUM; i < b; ++i) {
                actualSeqQSum = actualSeqQSum + static_cast<uint32_t>(actualSeqLenQ->GetData<int64_t>()[i]);
            }
            OPS_ERR_IF(actualSeqQSum != sQ,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "SH format sum of actual_seq_q(%ld) do not match s_q(%u)!", actualSeqQSum, sQ),
                            return false);
        }
        return true;
    }
    // Pass the length of actSeqlen to kernel.
    tilingData.promptAttentionBaseParams.set_actualSeqLengthsSize(actualLenDimsQ);
    tilingData.promptAttentionBaseParams.set_actualSeqLengthsKVSize(actualLenDimsKV);
    
    if (inputActualSeqQ) {   // check the length of actual_seq_lengthsQ, whether is 1 or batch size
        OPS_ERR_IF(actualLenDimsQ < b && actualLenDimsQ > actualLenDimsQMin,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Dim(%lu) of actual_seq_lengths must equal to 1 or greater than or equal to batch size(%u)!", actualLenDimsQ, b),
                        return false);
        uint32_t actualSeqQLength = std::min(static_cast<uint32_t>(actualLenDimsQ), b); // actual_seq_lengths is 1 or batch size
        for (uint32_t i = LOOP_BEGIN_NUM; i < actualSeqQLength; ++i) {
            actualSeqTmp = static_cast<int64_t>(actualSeqLenQ->GetData<int64_t>()[i]);
            OPS_ERR_IF(actualSeqTmp < 0 || actualSeqTmp > sQ,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Actual_seq_lengths[%u](%ld) must be in range[0, %u]!", i, actualSeqTmp, sQ),
                            return false);
        }
    }
    
    if (inputActualSeqKV) {  // check the length of actual_seq_lengthsKV,whether is 1 or batch size
        OPS_ERR_IF(actualLenDimsKV < b && actualLenDimsKV > actualLenDimsKVMin,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Dim(%lu) of actual_seq_lengths_kv must equal to 1 or greater than or equal to batch size(%u)!", actualLenDimsKV, b),
                        return false);
        uint32_t actualSeqKVLength = std::min(static_cast<uint32_t>(actualLenDimsKV), b); // actual_seq_lengths_KV is 1 or batch size
        for (uint32_t i = LOOP_BEGIN_NUM; i < actualSeqKVLength; ++i) {
            actualSeqTmp = static_cast<int64_t>(actualSeqLenKV->GetData<int64_t>()[i]);
            if (contextKeyParams.isKvContinuous == 1) {
                if (!enablePA) {
                    OPS_ERR_IF(actualSeqTmp < 0 || actualSeqTmp > sKV,
                                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Actual_seq_lengths_kv[%u](%ld) must be in range[0, %u]!", i, actualSeqTmp, sKV),
                                return false);
                } else {
                    OPS_ERR_IF(actualSeqTmp < 0,
                                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Actual_seq_lengths_kv[%u](%ld) must >= 0", i, actualSeqTmp),
                                return false);
                }
            } else {
                if ((inLayout == InputLayout::BSND) || (inLayout == InputLayout::BSH)) {
                    OPS_ERR_IF(actualSeqTmp < 0 || actualSeqTmp > contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1),
                                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Actual_seq_lengths_kv[%u](%ld) must be in range[0, %li]!", i, actualSeqTmp,
                                                                    contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1)),
                                    return false);
                } else {
                    OPS_ERR_IF(actualSeqTmp < 0 || actualSeqTmp > contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2),
                                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Actual_seq_lengths_kv[%u](%ld) must be in range[0, %li]!", i, actualSeqTmp,
                                                                    contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2)),
                                    return false);
                }
            }
        }
    }

    return true;
}

bool PromptFlashAttentionTiling::CheckPseShiftTypeAndShape(ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape *pseShiftShape, uint32_t b, uint32_t n, uint32_t s1, uint32_t s2) {
    if (contextKeyParams.fromTilingSink != 0) {
        return true;
    }
    pseShiftElemType = contextKeyParams.pseShiftDataType;

    OPS_ERR_IF((curShortSocName == platform_ascendc::SocVersion::ASCEND310P),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "not support 310P when pse is not null"),
                    return false);

    OPS_ERR_IF((inputType == ge::DT_FLOAT16 && pseShiftElemType != ge::DT_FLOAT16),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "q type is fp16, but pse shift type is not fp16, pse shift type = %s",
                    g_strDataTypePfa.at(ValidPfaDataType(pseShiftElemType)).c_str()),
                    return false);

    OPS_ERR_IF((inputType == ge::DT_BF16 && pseShiftElemType != ge::DT_BF16),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "q type is bf16, but pse shift type is not bf16, pse shift type = %s",
                    g_strDataTypePfa.at(ValidPfaDataType(pseShiftElemType)).c_str()),
                    return false);

    OPS_ERR_IF((inputType == ge::DT_INT8 && pseShiftElemType != ge::DT_FLOAT16),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "q type is int8, but pse shift type is not fp16, pse shift type = %s",
                    g_strDataTypePfa.at(ValidPfaDataType(pseShiftElemType)).c_str()),
                    return false);

    // Currently does not support D has super large size.
     OPS_ERR_IF((n == 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "num head is zero"),
                    return false);

    // If pse is empty, there is no need to perform PSE actions.
    if (((pseShiftShape != nullptr) && (pseShiftShape->GetStorageShape().GetShapeSize() == 0)) ||
        (pseShiftShape == nullptr)) {
            usePseShift = 0;
            return true;
    }

    if (pseShiftElemType == ge::DT_FLOAT16) {
        pseShiftElemSize = FLOAT16SIZE;
    } else if (pseShiftElemType == ge::DT_BF16) {
        pseShiftElemSize = BFLOAT16SIZE;
    }
    pseShiftTypeByteNum = BYTE_BLOCK / pseShiftElemSize;

    uint32_t pseShiftDim = pseShiftShape->GetStorageShape().GetDimNum();
    OPS_ERR_IF((pseShiftDim != PSE_SHIFT_DIM),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "pse shift shape must be 4 dimension, rather than %u dimension", pseShiftDim),
                    return false);

    pseShiftBatch = pseShiftShape->GetStorageShape().GetDim(0);
    uint32_t pseShiftN = pseShiftShape->GetStorageShape().GetDim(1);  // 1: The sirst dimension is N.
    pseShiftS1 = pseShiftShape->GetStorageShape().GetDim(2);          // 2: The second dimension is S1.
    pseShiftS2 = pseShiftShape->GetStorageShape().GetDim(3);          // 3: The third dimension is S2.
    OPS_ERR_IF(((pseShiftBatch != 1 && pseShiftBatch != b) || (pseShiftN != n) ||
                    (pseShiftS1 < s1) || (pseShiftS2 < s2)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "pse shift shape must be [1 or %u, %u, >=%u, >=%u], but now it is [%u, %u, %u, %u]",
                    b, n ,s1, s2, pseShiftBatch, pseShiftN, pseShiftS1, pseShiftS2),
                    return false);

    return true;
}

bool PromptFlashAttentionTiling::CheckPATypeAndShape(ContextParamsForPFATiling& contextKeyParams,
    const gert::Tensor* actualSeqLenKV, int32_t b, int32_t n, int32_t h, int32_t headNumRatio) {
    const int32_t* blockSize = contextKeyParams.blockSize;
    OPS_ERR_IF((*blockSize % BLOCK_SIZE_BASE != 0 || *blockSize < BLOCK_SIZE_BASE || *blockSize > BLOCK_SIZE_MAX),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "block size(%d) should be a multiple of %d, and can't greater than %d when PA enable",
                    *blockSize, BLOCK_SIZE_BASE, BLOCK_SIZE_MAX),
                    return false);

    const gert::StorageShape* blockTableShape = contextKeyParams.blockTableShape;
    OPS_ERR_IF((((blockTableShape != nullptr) && (blockTableShape->GetStorageShape().GetShapeSize() == 0)) ||
                (blockTableShape == nullptr)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "blockTable can't be empty when PA enable"),
                    return false);
    int32_t blockTableDim1 = static_cast<int32_t>(blockTableShape->GetStorageShape().GetDim(0));
    blockTableDim2 = static_cast<int32_t>(blockTableShape->GetStorageShape().GetDim(1));
    // When blockTableDim2>maxBlockNumPerBatch, the kernel should use blockTableDim2 as the second dimension when indexing block id in blockTable.
    // But for the verification of mask S2 axis, maxBlockNumPerBatch * tempBlockSize should still be used as the verification benchmark.

    if (contextKeyParams.fromTilingSink != 0) {
        tmpS2 = blockTableDim2 * (*blockSize); // Tiling sinking scene, workspace needs to be calculated, at this time, blockTableDim2 * blockSize is used as S2.
        return true;
    }
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    int32_t keyDim = keyShape->GetStorageShape().GetDimNum();
    int32_t valueDim = valueShape->GetStorageShape().GetDimNum();
    OPS_ERR_IF(keyDim != valueDim,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the dim num of key(%d) and value(%d) are inconsistent when PA enable", keyDim, valueDim),
                    return false);
    OPS_ERR_IF(((keyDim != 3) && (keyDim != 4)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the dim of key and value must be 3 or 4 when PA enable"),
                    return false);

    int32_t keyDim1 = keyShape->GetStorageShape().GetDim(0);  // block_num_sum
    int32_t keyDim2 = keyShape->GetStorageShape().GetDim(1);
    int32_t keyDim3 = keyShape->GetStorageShape().GetDim(2);
    int32_t keyDim4 = 0;
    int32_t valueDim1 = valueShape->GetStorageShape().GetDim(0);
    int32_t valueDim2 = valueShape->GetStorageShape().GetDim(1);
    int32_t valueDim3 = valueShape->GetStorageShape().GetDim(2);
    int32_t valueDim4 = 0;
    int32_t tempBlockSize = keyDim2;
    int32_t tempH = keyDim3;
    int32_t tempN = 0;
    int32_t tempD = 0;

    if (keyDim == 4) {  // dim num: 4
        keyDim4 = keyShape->GetStorageShape().GetDim(3);  // 3: The third dimension.
        valueDim4 = valueShape->GetStorageShape().GetDim(3);  // 3: The third dimension.
        tempN = keyDim2;
        tempBlockSize = keyDim3;
        tempD = keyDim4;
    }

    OPS_ERR_IF(((keyDim1 != valueDim1) || (keyDim2 != valueDim2) || (keyDim3 != valueDim3) || (keyDim4 != valueDim4)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the dim of key and value are inconsistent when PA enable"),
                    return false);
    
    int32_t actualSeqKVPerBatch = 0;
    int32_t blockNumPerBatch = 0;
    int64_t blockNumValid = 0;
    int32_t maxBlockNumPerBatch = 0;
    for (int32_t i = 0; i < b; i++) {
        actualSeqKVPerBatch = actualSeqLenKV->GetShapeSize() > 1 ? static_cast<int32_t>(actualSeqLenKV->GetData<int64_t>()[i]) :
                              static_cast<int32_t>(actualSeqLenKV->GetData<int64_t>()[0]);
        blockNumPerBatch = (actualSeqKVPerBatch + *blockSize - 1) / *blockSize;
        blockNumValid += blockNumPerBatch;
        if (blockNumPerBatch > maxBlockNumPerBatch) {
            maxBlockNumPerBatch = blockNumPerBatch;
        }
    }

    if (keyDim == 3) {  // dim num: 3
        PAlayoutType = 1;  // If it is three-dimensional, PAlayoutType = 1
        OPS_ERR_IF(((tempBlockSize != *blockSize) || (tempH * headNumRatio != h)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the dim of key [%d, %d, %d] is wrong, which should be [>=%ld, %d, %d] when PA enable", keyDim1,
                    keyDim2, keyDim3, blockNumValid, *blockSize, h / headNumRatio),  // When assigning headNumRatio, it is guaranteed that it will not be 0
                    return false);
        // In the BSH input of the PA scenario, it is required that the h of the KV matrix does not exceed 65535.  The dim and dim3 of the K/V have already been verified to be equal, so only the K matrix is verified here.
        OPS_ERR_IF(keyDim3 > 65535,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "layout of key/value is BSH, the h of key/value %d should not > 65535 when PA enable",
                    keyDim3),
                    return false);
    } else {
        PAlayoutType = 0;  // If it is four-dimensional, PAlayoutType = 0
        OPS_ERR_IF(((tempN * headNumRatio != n) || (tempBlockSize != *blockSize) || (tempD != (h / n))),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the dim of key [%d, %d, %d, %d] is wrong, which should be [>=%ld, %d, %d, %d] when PA enable",
                    keyDim1, keyDim2, keyDim3, keyDim4, blockNumValid, n / headNumRatio, *blockSize, (h / n)),
                    return false);
    }

    std::string layoutStr(contextKeyParams.layout);
    if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND" || layoutStr == "NSD") {
        OPS_ERR_IF(((keyDim != 3) && (keyDim != 4)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the layout of query is %s, key and value layout should be [>=%ld, %d, %d] or [>=%ld, %d, %d, %d] when PA enable",
                    layoutStr.c_str(), blockNumValid, *blockSize, h, blockNumValid, n, *blockSize, (h / n)),
                    return false);
    } else if (layoutStr == "BSH" || layoutStr == "BSND") {
        OPS_ERR_IF(keyDim != 3,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the layout of query is %s, key and value layout should be [>=%ld, %d, %d] when PA enable",
                    layoutStr.c_str(), blockNumValid, *blockSize, h),
                    return false);
    } else {
        OPS_LOG_E(contextKeyParams.opName, "unsupported input data layout when PA enable");
        return false;
    }

    ge::DataType blockTableType = contextKeyParams.blockTableType;
    OPS_ERR_IF((blockTableType != ge::DT_INT32),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "blockTable only support int32 when PA enable"),
                    return false);

    int32_t blockTableDim = static_cast<int32_t>(blockTableShape->GetStorageShape().GetDimNum());
    OPS_ERR_IF(blockTableDim != 2,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the dim of block table must be 2 when PA enable"),
                    return false);

    OPS_ERR_IF(((blockTableDim1 != b) || (blockTableDim2 < maxBlockNumPerBatch)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "block table shape should be [%d, >=%d], now is [%d, %d] when PA enable",
                    b, maxBlockNumPerBatch, blockTableDim1, blockTableDim2),
                    return false);

    OPS_ERR_IF((keyDim1 < blockNumValid),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the first dim of key(%d) should not less than valid block num(%ld) when PA enable",
                    keyDim1, blockNumValid),
                    return false);

    PABlockNumSum = keyDim1;
    tmpS2 = maxBlockNumPerBatch * tempBlockSize;
    return true;
}

bool PromptFlashAttentionTiling::CheckAttenMaskShape(ContextParamsForPFATiling& contextKeyParams,
                                                     const int32_t* sparseMode,
                                                     const gert::StorageShape* attenMaskShape,
                                                     const uint32_t sQ, const uint32_t sK, const uint32_t batchSize) {
    if (contextKeyParams.fromTilingSink != 0) {
        return true;
    }
    // Attention mask empty Tensor scene, no need to verify attention mask shape based on sparse mode value
    if (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0)) ||
        (attenMaskShape == nullptr)) {
        return true;
    }
    uint32_t attenMaskDim = attenMaskShape->GetStorageShape().GetDimNum();
    uint32_t attenMaskBatch = 1U;
    uint32_t attenMaskS1, attenMaskS2;
    int32_t checkShapeRet = 0;
    if (attenMaskDim == ATTENTION_MASK_DIM2) {
        attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(1);
        if ((sparseMode == nullptr) || (sparseMode != nullptr && *sparseMode == SPARSE_MODE_NO_MASK) ||
            (sparseMode != nullptr && *sparseMode == SPARSE_MODE_ALL_MASK)) {
            checkShapeRet = (attenMaskS1 >= sQ) && (attenMaskS2 >= sK) &&
                            (attenMaskBatch == 1 || attenMaskBatch == batchSize);
        }

        if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN || *sparseMode == SPARSE_MODE_BAND)) {
            checkShapeRet = attenMaskS1 == SPARSE_OPTIMIZE_ATTENTION_SIZE &&
                            attenMaskS2 == SPARSE_OPTIMIZE_ATTENTION_SIZE;
        }
    } else if (attenMaskDim == ATTENTION_MASK_DIM3) {
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(1);
        attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(2);  // 2: When the dim is 3, the second dimension is S2.
        if ((sparseMode == nullptr) || (sparseMode != nullptr && *sparseMode == SPARSE_MODE_NO_MASK) ||
            (sparseMode != nullptr && *sparseMode == SPARSE_MODE_ALL_MASK)) {
            checkShapeRet = (attenMaskS1 >= sQ) && (attenMaskS2 >= sK) &&
                            (attenMaskBatch == 1 || attenMaskBatch == batchSize);
        }
        if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN || *sparseMode == SPARSE_MODE_BAND)) {
            checkShapeRet = attenMaskBatch == 1 &&
                            attenMaskS1 == SPARSE_OPTIMIZE_ATTENTION_SIZE &&
                            attenMaskS2 == SPARSE_OPTIMIZE_ATTENTION_SIZE;
        }
    } else if (attenMaskDim == ATTENTION_MASK_DIM4) {
        uint32_t attenMaskN = 1U;
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskN = attenMaskShape->GetStorageShape().GetDim(1);
        attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(2);  // 2: When the dim is 4, the second dimension is S1.
        attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(3);  // 3: When the dim is 4, the third dimension is S2.
        if ((sparseMode == nullptr) || (sparseMode != nullptr && *sparseMode == SPARSE_MODE_NO_MASK) ||
            (sparseMode != nullptr && *sparseMode == SPARSE_MODE_ALL_MASK)) {
            checkShapeRet = (attenMaskS1 >= sQ) && (attenMaskS2 >= sK) &&
                            (attenMaskBatch == 1 || attenMaskBatch == batchSize);
        }
        if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN || *sparseMode == SPARSE_MODE_BAND)) {
            checkShapeRet = attenMaskBatch == 1 && attenMaskN == 1 &&
                            attenMaskS1 == SPARSE_OPTIMIZE_ATTENTION_SIZE &&
                            attenMaskS2 == SPARSE_OPTIMIZE_ATTENTION_SIZE;
        }
    } else {
        OPS_LOG_E(contextKeyParams.opName, "attenMask dim(%u) must be 2 or 3 or 4!", attenMaskDim);
        return false;
    }
    if ((sparseMode == nullptr) || ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_NO_MASK)) ||
        ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_ALL_MASK))) {
        OPS_ERR_IF(checkShapeRet != 1, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "attenMask batch(%u) must be 1 or %u, attenMask Q_S(%u) must be larger than sQ(%u), attenMask KV_S(%u) must be larger than sK(%u), please check",
            attenMaskBatch, batchSize, attenMaskS1, sQ, attenMaskS2, sK), return false);
    }
    if ((sparseMode != nullptr) && ((*sparseMode == SPARSE_MODE_LEFT_UP) || (*sparseMode == SPARSE_MODE_RIGHT_DOWN) || (*sparseMode == SPARSE_MODE_BAND))) {
        OPS_ERR_IF(checkShapeRet != 1, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "attenMask shape must be (2048, 2048) or (1, 2048, 2048) or (1, 1, 2048, 2048) when sparse mode = %d",
            *sparseMode), return false);
    }
    return true;
}

bool PromptFlashAttentionTiling::CheckAntiquantParamsShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* antiquantScaleShape,
                                                           const gert::StorageShape* antiquantOffsetShape, const uint32_t n, const uint32_t d, const uint32_t h,
                                                           PromptFlashAttentionTilingData& tilingData) {
    OPS_ERR_IF(contextKeyParams.antiquantScale == nullptr || antiquantScaleShape == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale is nullptr"),
                    return false);
    tilingData.promptAttentionBaseParams.set_isAntiPerchannel(1);
    if (antiquantScaleShape->GetStorageShape().GetDimNum() == 1) {
        tilingData.promptAttentionBaseParams.set_isAntiPerchannel(0);
        OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDim(0) != 2,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim[0] = %ld, but it should be 2 under Per-Tensor mode!", antiquantScaleShape->GetStorageShape().GetDim(0)),
                        return false);
        OPS_ERR_IF(antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetDim(0) != 2,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim[0] = %ld, but it should be 2 under Per-Tensor mode!", antiquantOffsetShape->GetStorageShape().GetDim(0)),
                        return false);
    } else {
        if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) {
            OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDimNum() != 4,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim num[%zu] should be 4 if layout is BNSD or NSD!", antiquantScaleShape->GetStorageShape().GetDimNum()),
                            return false);
            OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDim(0) != 2 || antiquantScaleShape->GetStorageShape().GetDim(1) != n ||
                            antiquantScaleShape->GetStorageShape().GetDim(2) != 1 || antiquantScaleShape->GetStorageShape().GetDim(3) != d,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim [%ld, %ld, %ld, %ld] is wrong!", antiquantScaleShape->GetStorageShape().GetDim(0),
                            antiquantScaleShape->GetStorageShape().GetDim(1), antiquantScaleShape->GetStorageShape().GetDim(2), antiquantScaleShape->GetStorageShape().GetDim(3)),
                            return false);
            OPS_ERR_IF(antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetDimNum() != 4,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim num[%zu] should be 4 if layout is BNSD or NSD!", antiquantOffsetShape->GetStorageShape().GetDimNum()),
                            return false);
            OPS_ERR_IF(antiquantOffsetShape != nullptr && (antiquantOffsetShape->GetStorageShape().GetDim(0) != 2 || antiquantOffsetShape->GetStorageShape().GetDim(1) != n ||
                            antiquantOffsetShape->GetStorageShape().GetDim(2) != 1 || antiquantOffsetShape->GetStorageShape().GetDim(3) != d),
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim [%ld, %ld, %ld, %ld] is wrong!", antiquantOffsetShape->GetStorageShape().GetDim(0),
                            antiquantOffsetShape->GetStorageShape().GetDim(1), antiquantOffsetShape->GetStorageShape().GetDim(2), antiquantOffsetShape->GetStorageShape().GetDim(3)),
                            return false);
        } else if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::SH)) {
            OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDimNum() != 2,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim num[%zu] should be 2 if layout is BSH or SH!", antiquantScaleShape->GetStorageShape().GetDimNum()),
                            return false);
            OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDim(0) != 2 || antiquantScaleShape->GetStorageShape().GetDim(1) != h,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim [%ld, %ld] is wrong!", antiquantScaleShape->GetStorageShape().GetDim(0),
                            antiquantScaleShape->GetStorageShape().GetDim(1)),
                            return false);
            OPS_ERR_IF(antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetDimNum() != 2,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim num[%zu] should be 2 if layout is BSH or SH!", antiquantOffsetShape->GetStorageShape().GetDimNum()),
                            return false);
            OPS_ERR_IF(antiquantOffsetShape != nullptr && (antiquantOffsetShape->GetStorageShape().GetDim(0) != 2 || antiquantOffsetShape->GetStorageShape().GetDim(1) != h),
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim [%ld, %ld] is wrong!", antiquantOffsetShape->GetStorageShape().GetDim(0),
                            antiquantOffsetShape->GetStorageShape().GetDim(1)),
                            return false);
        } else if (inputLayout == InputLayout::BSND) {
            OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDimNum() != 3,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim num[%zu] should be 3 if layout is BSND!", antiquantScaleShape->GetStorageShape().GetDimNum()),
                            return false);
            OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDim(0) != 2 || antiquantScaleShape->GetStorageShape().GetDim(1) != n ||
                            antiquantScaleShape->GetStorageShape().GetDim(2) != d,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim [%ld, %ld, %ld] is wrong!", antiquantScaleShape->GetStorageShape().GetDim(0),
                            antiquantScaleShape->GetStorageShape().GetDim(1), antiquantScaleShape->GetStorageShape().GetDim(2)),
                            return false);
            OPS_ERR_IF(antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetDimNum() != 3,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim num[%zu] should be 3 if layout is BSND!", antiquantOffsetShape->GetStorageShape().GetDimNum()),
                            return false);
            OPS_ERR_IF(antiquantOffsetShape != nullptr && (antiquantOffsetShape->GetStorageShape().GetDim(0) != 2 || antiquantOffsetShape->GetStorageShape().GetDim(1) != n ||
                            antiquantOffsetShape->GetStorageShape().GetDim(2) != d),
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim [%ld, %ld, %ld] is wrong!", antiquantOffsetShape->GetStorageShape().GetDim(0),
                            antiquantOffsetShape->GetStorageShape().GetDim(1), antiquantOffsetShape->GetStorageShape().GetDim(2)),
                            return false);
        }
    }

    return true;
}

ge::graphStatus PromptFlashAttentionTiling::CheckPostQuantParams(const ContextParamsForPFATiling& contextKeyParams, uint32_t h, uint32_t n) const {
    const gert::StorageShape* quantScale2Shape = contextKeyParams.scale2Shape;
    const gert::StorageShape* quantOffset2Shape = contextKeyParams.offset2Shape;
    const ge::DataType quantScale2Type = contextKeyParams.quantScale2Type;
    const ge::DataType quantOffset2Type = contextKeyParams.quantOffset2Type;
    int64_t quantScale2ShapeSize = 0;
    int64_t quantOffset2ShapeSize = 0;
    uint32_t quantD = 0;
    uint32_t queryD = h / n;

    if (outputType == ge::DT_INT8) {
        // Basic verification: quantScale2 must be inputted and not an empty tensor
        OPS_ERR_IF(quantScale2Shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale is nullptr when output type is int8."),
                return ge::GRAPH_FAILED);
        quantScale2ShapeSize = quantScale2Shape->GetStorageShape().GetShapeSize();
        quantD = quantScale2ShapeSize / n;
        OPS_ERR_IF(quantScale2ShapeSize == 0,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "quant_scale2 is empty tensor when output type is int8."),
                return ge::GRAPH_FAILED);

        // altert unsupported situation(post quant per-tensor + BF16 + BSH + D unalign)
        if ((contextKeyParams.inputDataType == ge::DT_BF16) && (quantScale2ShapeSize == 1) && (inputLayout == InputLayout::BSH) && (queryD % BYTE_BLOCK != 0)) {
            OPS_LOG_W(contextKeyParams.opName, "post quant per-tensor doesn't support D unaligned(%u), when qkv is bf16 and layout is BSH.", queryD);
        }

        // Cross characteristic verification: The After Quant per-channel does not currently support left padding, ring attention, and D non 32B alignment
        if (quantScale2ShapeSize != 1) {
            OPS_ERR_IF((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant per-channel do not support left padding."),
                    return ge::GRAPH_FAILED);
            OPS_ERR_IF(contextKeyParams.isSoftMaxLseEnable == true,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant per-channel do not support ring attention."),
                    return ge::GRAPH_FAILED);
            OPS_ERR_IF(quantD % BYTE_BLOCK != 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant per-channel do not support D(%u) non-32-byte aligned.", quantD),
                    return ge::GRAPH_FAILED);
        }

        // dtype verification
        OPS_ERR_IF((quantScale2Type != ge::DT_BF16) && (quantScale2Type != ge::DT_FLOAT) && (quantScale2Type != ge::DT_FLOAT16),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale dtype(%s) only support bf16, fp16 and fp32 .",
                g_strDataTypePfa.at(ValidPfaDataType(quantScale2Type)).c_str()),
                return ge::GRAPH_FAILED);
        OPS_ERR_IF((quantOffset2Shape != nullptr) && (quantScale2Type != quantOffset2Type),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale dtype(%s) and offset dtype(%s) must be consistent.",
                g_strDataTypePfa.at(ValidPfaDataType(quantScale2Type)).c_str(), g_strDataTypePfa.at(ValidPfaDataType(quantOffset2Type)).c_str()),
                return ge::GRAPH_FAILED);
        OPS_ERR_IF((inputType != ge::DT_BF16) && (quantScale2Type == ge::DT_BF16),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale and offset support bf16 only if input dtype(%s) is bf16.",
                g_strDataTypePfa.at(ValidPfaDataType(inputType)).c_str()),
                return ge::GRAPH_FAILED);
        OPS_ERR_IF((inputType != ge::DT_FLOAT16) && (quantScale2Type == ge::DT_FLOAT16),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale and offset support fp16 only if input dtype(%s) is fp16.",
                g_strDataTypePfa.at(ValidPfaDataType(inputType)).c_str()),
                return ge::GRAPH_FAILED);

        // shape verification
        if (quantOffset2Shape != nullptr) {
            quantOffset2ShapeSize = quantOffset2Shape->GetStorageShape().GetShapeSize();
            OPS_ERR_IF(quantScale2ShapeSize != quantOffset2ShapeSize,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "quant_scale2 dimension multiply result(%ld) do not equal quant_offset2 dimension multiply result(%ld).",
                    quantScale2ShapeSize, quantOffset2ShapeSize), return ge::GRAPH_FAILED);
        }
        OPS_ERR_IF((quantScale2ShapeSize != 1) && (quantScale2ShapeSize != h),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "post quant scale2/offset2 dimension multiply result only support 1 and H(%u), now is (%ld). "
                "Maybe the shape of scale2/offset2 do not match that of query, or D is not 32 Byte aligned, "
                "which post quant per-channel do not support.", h, quantScale2ShapeSize), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::AdjustBasicBlock(PromptFlashAttentionTilingData& tilingData,
                                                             uint32_t& sOuterFactor) {
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    uint32_t headNumSize = baseParams->get_headNumSize();
    uint32_t sCoreNum = (coreNum / headNumSize);
    uint32_t sOuterBlockNum = (maxQuerySeq + sOuterFactor - 1U) / sOuterFactor;
    if ((coreNum % headNumSize == 0) && (sCoreNum > 1) && (sOuterBlockNum % sCoreNum == 0) &&
        sOuterBlockNum / sCoreNum == 1) {
      // Open all core in the n direction; Multiple cores are opened in the s direction and each core only processes one Souter. At this point, the Souter is divided into two blocks for load balancing optimization.
      // To ensure that the basic block is an integer multiple of typeByteNum.
      sOuterFactor = (sOuterFactor / 2 + typeByteNum - 1) / typeByteNum * typeByteNum;  // split outer: 2
    }
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::Align(uint32_t &num) {
    num = (num + typeByteNum - 1) / typeByteNum * typeByteNum;
}

// Code for ut, no pratical to use.
ge::graphStatus PromptFlashAttentionTiling::GetBasicShape310P(uint32_t &b,
                                                              uint32_t &bKV,
                                                              uint32_t &s,
                                                              uint32_t &h,
                                                              uint32_t &seqInnerSize,
                                                              const gert::StorageShape *queryShape,
                                                              const gert::StorageShape *keyShape,
                                                              const uint32_t n,
                                                              size_t actualLenDims,
                                                              size_t actualLenDimsKV) {
    OPS_ERR_IF(queryShape == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "queryShape is null."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(keyShape == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "keyShape is null."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(n == 0,
                    OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "n is 0."),
                    return ge::GRAPH_FAILED);
    if (inputLayout == InputLayout::NSD) {
      uint32_t d;
      b = 1;
      bKV = 1;
      s = queryShape->GetStorageShape().GetDim(1);
      seqInnerSize = keyShape->GetStorageShape().GetDim(1);
      d = queryShape->GetStorageShape().GetDim(2); // dim num: 2
      Align(d);
      h = (d * n);
      return ge::GRAPH_SUCCESS;
    }

    if (inputLayout == InputLayout::BNSD) {
      uint32_t d;
      b = queryShape->GetStorageShape().GetDim(0);
      bKV = keyShape->GetStorageShape().GetDim(0);
      s = queryShape->GetStorageShape().GetDim(2); // dim num: 2
      seqInnerSize = keyShape->GetStorageShape().GetDim(2); // dim num: 2
      d = queryShape->GetStorageShape().GetDim(3); // dim num: 3
      Align(d);
      h = (queryShape->GetStorageShape().GetDim(1) * d);
      return ge::GRAPH_SUCCESS;
    }

    if (inputLayout == InputLayout::SH) {
      b = ((actualLenDims == 0) ? 1 : actualLenDims); // When the input layout is SH and actual_seq is not input, the batch of query is set to 1.
      bKV = ((actualLenDimsKV == 0) ? 1 : actualLenDimsKV); // When the input layout is SH and actual_seqkv is not input, the batch of key/value is set to 1.
      uint32_t d;
      s = queryShape->GetStorageShape().GetDim(0);
      h = queryShape->GetStorageShape().GetDim(1);
      seqInnerSize = keyShape->GetStorageShape().GetDim(0);

      Align(s);
      Align(seqInnerSize);
      d = (h / n);
      Align(d);
      h = (d * n);
      return ge::GRAPH_SUCCESS;
    }

    if (inputLayout == InputLayout::BSH) {
      uint32_t d;
      b = queryShape->GetStorageShape().GetDim(0); // dim num: 0, btach of query
      bKV = keyShape->GetStorageShape().GetDim(0); // dim num: 0, btach of kv
      s = queryShape->GetStorageShape().GetDim(1); // dim num: 0, s of query
      h = queryShape->GetStorageShape().GetDim(2); // dim num: 2
      seqInnerSize = keyShape->GetStorageShape().GetDim(1);
      d = h / n;
      Align(d);
      h = d * n;
      return ge::GRAPH_SUCCESS;
    }

    if (inputLayout == InputLayout::BSND) {
      uint32_t d;
      b = (queryShape->GetStorageShape().GetDim(0));
      bKV = (keyShape->GetStorageShape().GetDim(0));
      s = (queryShape->GetStorageShape().GetDim(1));
      d = (queryShape->GetStorageShape().GetDim(INDEX_3));
      seqInnerSize = (keyShape->GetStorageShape().GetDim(1));
      Align(d);
      h = (d * n);
      return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus PromptFlashAttentionTiling::GetAndCheckEmptyQueryShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape *queryShape) const {
    OPS_ERR_IF(queryShape == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "queryShape is null."),
               return ge::GRAPH_FAILED);
    uint32_t b = 0;
    uint32_t n = 0;
    uint32_t s = 0;
    uint32_t d = 0;
    uint32_t h = 0;
    if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) {
        if (queryShape->GetStorageShape().GetDimNum() == 3) { // dim num: 3
            b = 1;
            n = queryShape->GetStorageShape().GetDim(0);
            s = queryShape->GetStorageShape().GetDim(1);
            d = queryShape->GetStorageShape().GetDim(2); // dim num: 2
        } else {
            b = (queryShape->GetStorageShape().GetDim(0));
            n = (queryShape->GetStorageShape().GetDim(1));
            s = (queryShape->GetStorageShape().GetDim(2)); // dim num: 2
            d = (queryShape->GetStorageShape().GetDim(3)); // dim num: 3
        }
    } else if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) ||
        (inputLayout == InputLayout::SH)) {
        if (queryShape->GetStorageShape().GetDimNum() == NUM_2) { // dim num: 2
            b = 1; // Process according to batch = 1.
            s = (queryShape->GetStorageShape().GetDim(0));
            h = (queryShape->GetStorageShape().GetDim(1));
        } else if (queryShape->GetStorageShape().GetDimNum() == 3) { // 3 : BSH
            b = (queryShape->GetStorageShape().GetDim(0));
            s = (queryShape->GetStorageShape().GetDim(1));
            h = (queryShape->GetStorageShape().GetDim(2)); // dim num: 2
        } else { // BSND
            b = queryShape->GetStorageShape().GetDim(0);
            s = queryShape->GetStorageShape().GetDim(1);
            n = queryShape->GetStorageShape().GetDim(2); // dim num: 2
            d = queryShape->GetStorageShape().GetDim(3); // dim num: 3
        }
    } else if (inputLayout == InputLayout::TND) {
        b = 1u;
        n = queryShape->GetStorageShape().GetDim(SECOND_DIM);
        s = 1u;
        d = queryShape->GetStorageShape().GetDim(THIRD_DIM);
    } else {
        return ge::GRAPH_FAILED;
    }
    OPS_ERR_IF(b > BLIMIT, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
               "batch size should <= 65536, but batch size = %u", b), return ge::GRAPH_FAILED);
    if (s > SLIMIT) {
        OPS_LOG_W(contextKeyParams.opName,
                   "seq should <= 20m, but seq = %u", s);
    }
    if (inputLayout == InputLayout::BSH || inputLayout == InputLayout::SH) {
        OPS_ERR_IF(h > DLIMIT * NLIMIT, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                   "h should <= 512 * 256, but h = %u", h), return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF(n > NLIMIT, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "n should <= 256, but n = %u", n), return ge::GRAPH_FAILED);
        OPS_ERR_IF(d > DLIMIT, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "D should <= 512, but d = %u", d), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::SetMultiCoreParams()
{
    auto &multiCoreParams = mlaTilingData.PFAmultiCoreParams;
    auto &coreParams = mlaTilingData.PFAcoreParams;
    int64_t totalSize = coreParams.get_bOuterSize() * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize() *
                        coreParams.get_s1OuterSize();
    int64_t actualUsedAicNum = std::min(totalSize, static_cast<int64_t>(aicNum));
    multiCoreParams.set_totalSize(totalSize);
    multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAicNum));

    multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
    actualUsedAicNum = CeilDivision(totalSize, multiCoreParams.get_splitFactorSize());
    multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAicNum * AIV_AIC_NUM_RATIO));
}

void PromptFlashAttentionTiling::SetTensorSizeParams()
{
    auto &tensorSizeParams = mlaTilingData.PFAtensorSizeParams;
    auto &coreParams = mlaTilingData.PFAcoreParams;
    int64_t batchInnerSize = coreParams.get_bBaseSize() * coreParams.get_n2BaseSize() * coreParams.get_gBaseSize();
    tensorSizeParams.set_bmm1ResUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
    tensorSizeParams.set_attenMaskUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
    tensorSizeParams.set_castUbSize(batchInnerSize * s1BasicBlock * std::max(s2BasicBlock, dBasicBlock));
    tensorSizeParams.set_softmaxMaxUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxSumUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxExpUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / softmaxDataTypeSize));
    tensorSizeParams.set_apiTmpBufferBytes(apiMaxUBSize);

    tensorSizeParams.set_bmm2ResUbSize(batchInnerSize * s1BasicBlock * dBasicBlock);
}

bool PromptFlashAttentionTiling::InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx)
{
    int64_t s2BlkNum = CeilDivision(s2Size, s2BasicBlock);
    int64_t validS1Size = CeilDivision(s1SparseValidSize, s1BasicBlock);
    int64_t validS2Size = CeilDivision(s2SparseValidSize, s2BasicBlock);
    int64_t invalidRowSparseRatio = INVALID_ROW_SPARSE_RATIO;
    if (s2Size <= s2BasicBlock) {
        invalidRowSparseRatio = 1;
    }
    for (int64_t i = 0; i < static_cast<int64_t>(sparseValidArray.size()); i++) {
        int64_t reduceBlk =
            (i < validS1Size) ? 0 : (CeilDivision((i + 1) * s1BasicBlock - s1SparseValidSize, s2BasicBlock) - 1);
        int64_t addBlk =
            std::min(s2BlkNum - validS2Size,
                        CeilDivision((i + 1) * s1BasicBlock + s2SparseValidSize, s2BasicBlock) - validS2Size);
        int64_t validBlockNum = validS2Size - reduceBlk + addBlk;
        sparseValidArray[i] = validBlockNum > 0 ? validBlockNum : invalidRowSparseRatio;
    }
    return true;
}

bool PromptFlashAttentionTiling::PartitionSparseData(const std::vector<int64_t> &sparseRollingArray,
                                                        int64_t sparseRollingArraySum, int64_t sparseArraySize,
                                                        int64_t loadMaxEachCore, std::vector<int64_t> &partitionResult)
{
    int64_t s1OuterCutEachCore = loadMaxEachCore / sparseRollingArraySum;
    int64_t s1OuterLoadEachCore = s1OuterCutEachCore * sparseRollingArraySum;
    int64_t s1OuterNumEachCore = s1OuterCutEachCore * sparseRollingArray.size();

    int64_t targetCoreNum = partitionResult.size();
    int64_t coreIdx = 0;
    int64_t rollingIdx = 0;
    int64_t loadSize = s1OuterLoadEachCore;
    partitionResult[0] = 0;
    for (int64_t i = s1OuterNumEachCore; i < sparseArraySize; i++, rollingIdx++) {
        rollingIdx = (static_cast<uint64_t>(rollingIdx) >= sparseRollingArray.size()) ? 0 : rollingIdx;
        int64_t loadNext = sparseRollingArray[rollingIdx];
        bool needOneMoreCore = (loadSize + loadNext) > loadMaxEachCore;
        if (needOneMoreCore && coreIdx >= (targetCoreNum - 1)) {
            return false;
        }

        if (needOneMoreCore) {
            partitionResult[++coreIdx] = i;
            i += s1OuterNumEachCore;
            i--;
            rollingIdx--;
            loadSize = s1OuterLoadEachCore;
            continue;
        }
        loadSize += loadNext;
    }
    std::fill(partitionResult.begin() + coreIdx + 1, partitionResult.end(), sparseArraySize);
    return true;
}

bool PromptFlashAttentionTiling::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray,
                                                        PFAMultiCoreParams &multiCoreParams)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validCoreNum;
    validCoreNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum() / AIV_AIC_NUM_RATIO), MAX_AIC_NUM);
    int64_t totalSize = multiCoreParams.get_totalSize(); // BN2GS1.o
    int64_t *sparseStartIdx = multiCoreParams.get_sparseStartIdx();
    for (int64_t idx = 0; idx < MAX_AIV_NUM; ++idx) {
        sparseStartIdx[idx] = totalSize;
    }

    if (totalSize <= validCoreNum) {
        for (int64_t idx = 0; idx < totalSize; ++idx) {
            sparseStartIdx[idx] = idx;
        }

        return true;
    }

    // Minimize the max load each core to find a load balance result.
    // The range of max load each core is (loadEachCoreLowerBound, loadEachCoreUpperBound].
    std::vector<int64_t> partitionResult(validCoreNum, totalSize);
    std::vector<int64_t> lastValidPartitionResult(validCoreNum, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
    int64_t loadTotal = sparseArraySum * (totalSize / sparseValidArray.size());
    int64_t loadEachCoreLowerBound = loadTotal / validCoreNum - 1;
    int64_t loadEachCoreUpperBound =
        CeilDivision(loadTotal, validCoreNum) + (*std::max_element(sparseValidArray.begin(), sparseValidArray.end()));
    while (loadEachCoreLowerBound + 1 < loadEachCoreUpperBound) {
        int64_t loadMax = loadEachCoreLowerBound + (loadEachCoreUpperBound - loadEachCoreLowerBound) / 2;
        if ((loadMax * validCoreNum >= loadTotal) &&
            PartitionSparseData(sparseValidArray, sparseArraySum, totalSize, loadMax, partitionResult)) {
            loadEachCoreUpperBound = loadMax;
            lastValidPartitionResult.swap(partitionResult);
            continue;
        }
        loadEachCoreLowerBound = loadMax;
    }

    for (int64_t idx = 0; idx < validCoreNum; ++idx) {
        sparseStartIdx[idx] = lastValidPartitionResult[idx];
    }
    return true;
}

void PromptFlashAttentionTiling::SetSparseParams()
{
    auto &coreParams = mlaTilingData.PFAcoreParams;
    coreParams.set_s1SparseValidSize(s1SparseValidSize);
    coreParams.set_s2SparseValidSize(s2SparseValidSize);

    auto &multiCoreParams = mlaTilingData.PFAmultiCoreParams;
    std::vector<int64_t> sparseValidArray(coreParams.get_s1OuterSize(), 0);
    InitSparseValidArray(sparseValidArray, 0);
    SetSparseStartIdx(sparseValidArray, multiCoreParams);
}

// TND 新增
void PromptFlashAttentionTiling::SetMultiCoreParamsTND()
{
    auto &multiCoreParams = mlaTilingData.PFAmultiCoreParams;
    auto &coreParams = mlaTilingData.PFAcoreParams;
    accumS1BlockNum = 0;
    for (int64_t i = 0; i < bSize; i++) {
        OPS_LOG_D(contextKeyParamsPtr->opName, "[%s]actualSeqLenData data %ld is %ld.", "PFA_TND", i, actualSeqLenData[i]);
        OPS_LOG_D(contextKeyParamsPtr->opName, "[%s]actualSeqLenKvData data %ld is %ld.", "PFA_TND", i, actualSeqLenKvData[i]);
        accumS1BlockNum += CeilDivision(actualSeqLenData[i], s1BasicBlock);
    }
    int64_t totalSize = accumS1BlockNum * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize();
    int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
    int64_t actualUsedAicNum = std::min(totalSize, static_cast<int64_t>(aicNum));
    int64_t actualUsedAiCoreNum = isSameAB ? actualUsedAicNum * 2 : actualUsedAivNum;
    int64_t actualSplitAiCoreNum = isSameAB ? actualUsedAicNum : actualUsedAivNum;
    multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAiCoreNum));
    multiCoreParams.set_totalSize(totalSize);
    multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualSplitAiCoreNum));
    multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
}

void PromptFlashAttentionTiling::GetActualSeqLenData(int64_t inputIdx,
                                                     std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &res, int64_t &actualLen)
{
    auto actualSeqLenTensor = contextKeyParamsPtr->actualSeqenceLengthQ;
    if (inputIdx == ACTUAL_SEQ_KV_INDEX) {
        actualSeqLenTensor = contextKeyParamsPtr->actualSeqenceLengthKV;
    }
    if (actualSeqLenTensor == nullptr) {
        if (inputIdx == ACTUAL_SEQ_KV_INDEX) {
            OPS_LOG_E(contextKeyParamsPtr->opName, "[%s]actualSeqLengthKV can not be null pointer", "PFA_TND");
        } else {
            OPS_LOG_E(contextKeyParamsPtr->opName, "[%s]actualSeqLengthQ can not be null pointer", "PFA_TND");
        }
        return;
    }
    auto &actualSeqLenShape = actualSeqLenTensor->GetShape().GetStorageShape();
    if (actualSeqLenShape.GetDimNum() != 1) {
        OPS_LOG_W(contextKeyParamsPtr->opName, "[%s]actualSeqLenShape is invalid %lu %ld", "PFA_TND", actualSeqLenShape.GetDimNum(),
                  actualSeqLenShape.GetDim(0));
        return;
    }
    /* Get Data from tensor. */
    const int64_t *value = actualSeqLenTensor->GetData<int64_t>();
    if (value == nullptr) {
        OPS_LOG_E(contextKeyParamsPtr->opName, "[%s]actualSeqLenTensor data is null pointer", "PFA_TND");
        return;
    }
    res[0] = value[0];
    actualLen++;
    for (int64_t i = 1; i < actualSeqLenShape.GetDim(0); ++i) {
        auto qLen = value[i] - value[i - 1];
        res[i] = qLen < 0 ? 0 : qLen;
        actualLen++;
    }
}

bool PromptFlashAttentionTiling::BalanceLoad(const std::vector<int64_t> &sparseValidArray,
                                             PFAMultiCoreParams &multiCoreParams, std::vector<int64_t> &localValue,
                                             std::vector<int64_t> &sparseStartIdx)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAiCoreNum = isSameAB ? std::min(static_cast<int64_t>(multiCoreParams.get_coreNum() / 2), MAX_AIC_NUM)
                                        :std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
    int64_t totalSize = multiCoreParams.get_totalSize();
    int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
    int64_t tmpMaxVal = maxVal;

    // 从前往后遍历
    for (int64_t idx = 1; idx < validAiCoreNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
            localValue[idx - 1] += sparseValidArray[start];
            localValue[idx] -= sparseValidArray[start];
            sparseStartIdx[idx] += 1;
        } else if (start == totalSize) {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    // 从后往前遍历
    for (int64_t idx = validAiCoreNum - 1; idx > 0; --idx) {
        int64_t start = sparseStartIdx[idx];
        if (start == totalSize) {
            if (sparseStartIdx[idx - 1] == totalSize) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] = sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else if (start > 0) {
            if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] += sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    return (tmpMaxVal >= maxVal) ? false : true;
}

bool PromptFlashAttentionTiling::InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAivNum,
                                               int64_t totalSize, const std::vector<int64_t> &sparseStartIdx,
                                               std::vector<int64_t> &localValue)
{
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        int64_t end = ((idx + 1) < validAivNum) ? sparseStartIdx[idx + 1] : totalSize;
        if (start < totalSize) {
            localValue[idx] =
                std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0LL);
        } else {
            break;
        }
    }
    return true;
}

bool PromptFlashAttentionTiling::SetSparseStartIdxTND(const std::vector<int64_t> &sparseValidArray, PFAMultiCoreParams &multiCoreParams)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAiCoreNum = isSameAB ? std::min(static_cast<int64_t>(multiCoreParams.get_coreNum() / 2), MAX_AIC_NUM)
                                        :std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
    int64_t totalSize = multiCoreParams.get_totalSize(); // BN2GS1.o
    int64_t *sparseStartIdx = multiCoreParams.get_sparseStartIdx();
    int64_t maxAiCoreNum = isSameAB ? MAX_AIC_NUM : MAX_AIV_NUM;
    OPS_ERR_IF(totalSize <= 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "totalSize should be larger than 0."), return false);

    // initLoad: 使用均分策略, 保证后续不会比均分差
    int64_t splitFactorSize = multiCoreParams.get_splitFactorSize();
    std::vector<int64_t> localSparseStartIdx(maxAiCoreNum, totalSize);
    for (int64_t idx = 0; idx < maxAiCoreNum; ++idx) {
        localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
    }
    std::vector<int64_t> localValue(validAiCoreNum, 0);
    InitLoadValue(sparseValidArray, validAiCoreNum, totalSize, localSparseStartIdx, localValue);

    // 负载均衡粗调
    std::vector<int64_t> tmpLocalValue(validAiCoreNum, 0);
    std::vector<int64_t> tmpsparseStartIdx(maxAiCoreNum, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
    int64_t avgVal = CeilDivision(sparseArraySum, validAiCoreNum);

    tmpsparseStartIdx[0] = 0;
    for (int64_t idx = 1; idx < maxAiCoreNum; ++idx) {
        int64_t start = tmpsparseStartIdx[idx - 1];
        int64_t singleLoadValue = 0;
        tmpsparseStartIdx[idx] = start;
        while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
            singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
            tmpsparseStartIdx[idx] += 1;
        }

        if ((start + 1) < tmpsparseStartIdx[idx]) {
            int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
            bool loadValueFlag = (singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue);
            tmpsparseStartIdx[idx] = loadValueFlag ? (tmpsparseStartIdx[idx] - 1) : (tmpsparseStartIdx[idx]);
            singleLoadValue = loadValueFlag ? redoSingleLoadValue : singleLoadValue;
            sparseArraySum -= singleLoadValue;
            avgVal = CeilDivision(sparseArraySum, (validAiCoreNum - idx));
        }
    }

    InitLoadValue(sparseValidArray, validAiCoreNum, totalSize, tmpsparseStartIdx, tmpLocalValue);

    // 负载均衡精调
    while (BalanceLoad(sparseValidArray, multiCoreParams, tmpLocalValue, tmpsparseStartIdx)) {
        // 根据负载均衡是否能得到更好预测结果决定是否结束循环
    }

    // exchange initLoad and 负载均衡
    if ((*std::max_element(localValue.begin(), localValue.end())) >
        (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
        localSparseStartIdx.swap(tmpsparseStartIdx);
        localValue.swap(tmpLocalValue);
    }
    for (int64_t idx = 0; idx < maxAiCoreNum; ++idx) {
        sparseStartIdx[idx] = localSparseStartIdx[idx];
    }
    return true;
}

int64_t PromptFlashAttentionTiling::GetS2RealSize(uint8_t sparseType, int32_t bOutIdx, int64_t s1OutIdx)
{
    int64_t s2RealSize = s2Size;
    int64_t actualS1Len = actualSeqLenData[bOutIdx];
    int64_t actualS2Len = actualSeqLenKvData[bOutIdx];

    if (sparseType == static_cast<uint8_t>(SparseTypeEnum::CAUSAL)) {
        s2RealSize = s1BasicBlock * (s1OutIdx + 1);
    } else if (sparseType == static_cast<uint8_t>(SparseTypeEnum::RIGHT_DOWN_CAUSAL)) {
        s2RealSize = s1BasicBlock * (s1OutIdx + 1) + actualS2Len - actualS1Len;
        s2RealSize = std::max(s2RealSize, static_cast<int64_t>(0));
    }
    return std::min(s2RealSize, actualSeqLenKvData[bOutIdx]);
}

bool PromptFlashAttentionTiling::InitSparseValidArrayTND(std::vector<int64_t> &sparseValidArray, int64_t bIdx)
{
    uint8_t sparseType = mlaTilingData.PFAinputParams.get_sparseType();
    auto &coreParams = mlaTilingData.PFAcoreParams;
    for (int32_t i = 0; i < bSize; i++) {
        int64_t n2G = coreParams.get_n2OuterSize() * coreParams.get_gOuterSize();
        int64_t s1BlockNum = CeilDivision(actualSeqLenData[i], s1BasicBlock);
        // 每个s1方向上切分块的计算量
        for (int64_t k = 0; k < n2G; ++k) {
            for (int64_t j = 0; j < s1BlockNum; ++j) {
                // 此处暂时设置为1, 由于实测尾块1和128性能差距不大，理论上应该如下所示
                // 理论值: s1RealSize为std::min(s1BasicBlock, (actualSeqLenData[i] - s1BasicBlock * j))
                int64_t s1RealSize = 1;
                int64_t s2RealSize = GetS2RealSize(sparseType, i, j);
                // 新增一个系数, 解决理论和实际的差异
                int64_t s2RemainSize = s2RealSize % s2sizeLimitMax;
                s2RealSize = (s2RealSize / s2sizeLimitMax) * s2sizeLimitMax;
                s2RealSize += ((s2RemainSize > 0) ? COF[CeilDivision(s2RemainSize, 128L) - 1] : 0);
                sparseValidArray.emplace_back(s1RealSize * s2RealSize);
            }
        }
    }
    return true;
}

void PromptFlashAttentionTiling::SetSparseParamsTND()
{
    auto &coreParams = mlaTilingData.PFAcoreParams;
    auto &multiCoreParams = mlaTilingData.PFAmultiCoreParams;
    std::vector<int64_t> sparseValidArray;
    sparseValidArray.reserve(multiCoreParams.get_totalSize());
    InitSparseValidArrayTND(sparseValidArray, 0);
    SetSparseStartIdxTND(sparseValidArray, multiCoreParams);

    coreParams.set_s1SparseValidSize(s1SparseValidSize);
    coreParams.set_s2SparseValidSize(s2SparseValidSize);
}

uint32_t PromptFlashAttentionTiling::CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
{
    uint32_t ration;
    if (aicCoreNum == 0 || aivCoreNum == 0 || aicCoreNum > aivCoreNum) {
        return sliceNum;
    }
    ration = aivCoreNum / aicCoreNum;
    return (sliceNum + (ration - 1)) / ration;
}

bool PromptFlashAttentionTiling::CalcUBSize()
{
    apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * s1BasicBlock * s2BasicBlock * sizeof(float);
    return true;
}

void PromptFlashAttentionTiling::SetSoftMaxTiling()
{
    auto softmaxShape = ge::Shape({batchBasic, std::min(s1BasicBlock, alignedS1), std::min(s2BasicBlock, alignedS2)});

    AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, softmaxDataTypeSize, sizeof(float), apiMaxUBSize,
                                      mlaTilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
}

bool PromptFlashAttentionTiling::SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch,
    matmul_tiling::MatmulApiTiling &bmm1)
{
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, true);
    bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    // 分不满核，且稀疏场景，shape设置的较小能产生更好的tiling
    bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size),
                  std::min(tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), s2Size), dSize);
    if (inputLayout == InputLayout::TND) {
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(ascendPlatformInfo.l1Size, ascendPlatformInfo.l0CSize) != 0) {
            return false;
        }
        return true;
    } else {
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), dSize, dSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(ascendPlatformInfo.l1Size, ascendPlatformInfo.l0CSize) != 0) {
            return false;
        }

        int64_t baseM = std::min(BMM_BASICBLOCK_M_128, AlignUp(s1Size, FRACTAL_NUM));
        int64_t baseN = std::min(BMM_BASICBLOCK_N_128, AlignUp(s2Size, FRACTAL_NUM));
        bmm1.SetFixSplit(baseM, baseN, alignedD);

        return true;
    }
}

bool PromptFlashAttentionTiling::SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
    matmul_tiling::MatmulApiTiling &bmm2)
{
    int64_t singleM = std::min(tmpS1BasicBlock, s1Size);
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, false);
    bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    if (inputLayout == InputLayout::TND) {
        bmm2.SetShape(std::min(tmpS1BasicBlock, s1Size), dSize,
                      std::min(tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), s2Size));
        bmm2.SetOrgShape(s1Size, s2StrideSize, std::min(tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), s2Size),
                        s2StrideSize);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(ascendPlatformInfo.l1Size, ascendPlatformInfo.l0CSize) != 0) {
            return false;
        }
        return true;
    } else {
        bmm2.SetShape(singleM, valueDSize,
                        std::min(tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), s2Size));
        bmm2.SetOrgShape(s1Size, valueDSize, std::min(tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), s2Size),
                            valueDSize);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(ascendPlatformInfo.l1Size, ascendPlatformInfo.l0CSize) != 0) {
            return false;
        }
        if (valueDSize > 64 && valueDSize <= 128) {
            int64_t baseM = std::min(BMM_BASICBLOCK_M_128, AlignUp(s1Size, FRACTAL_NUM));
            int64_t baseN = std::min(BMM_BASICBLOCK_N_128, AlignUp(valueDSize, FRACTAL_NUM));
            bmm2.SetFixSplit(baseM, baseN);
        }
        int64_t baseM = std::min(BMM_BASICBLOCK_M_128, AlignUp(s1Size, FRACTAL_NUM));
        int64_t baseN = AlignUp(valueDSize, FRACTAL_NUM);
        bmm2.SetFixSplit(baseM, baseN);
        return true;
    }
}

bool PromptFlashAttentionTiling::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                 int64_t tmpDBasicBlock, int64_t batch,
                                                 matmul_tiling::MatmulApiTiling &bmm1,
                                                 matmul_tiling::MatmulApiTiling &bmm2)
{
    if (!SetBmm1TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, batch, bmm1) ||
        !SetBmm2TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm2)) {
        return false;
    }

    if (bmm1.GetTiling(mlaTilingData.bmm1TilingData) == -1) {
        return false;
    }

    if (mlaTilingData.bmm1TilingData.get_baseK() >= mlaTilingData.bmm1TilingData.get_singleCoreK() &&
        mlaTilingData.bmm1TilingData.get_depthA1() == BMM1_DEPTH_A1_2) {
        mlaTilingData.bmm1TilingData.set_depthA1(BMM1_DEPTH_A1_3);
    }

    // 当D > 128，由于L0 DB开启的限制，在设置tiling时把baseK切小，在MatmulPolicy中仍然按照不切K搬运
    if (inputLayout != InputLayout::TND) {
        mlaTilingData.bmm1TilingData.set_baseK(AlignUp(dSize / NUM_2, FRACTAL_NUM));
    }

    mlaTilingData.bmm1TilingData.set_shareMode(0);
    mlaTilingData.bmm1TilingData.set_shareL1Size(ascendPlatformInfo.l1Size);
    mlaTilingData.bmm1TilingData.set_shareL0CSize(ascendPlatformInfo.l0CSize);

    if (bmm2.GetTiling(mlaTilingData.bmm2TilingData) == -1) {
        return false;
    }

    mlaTilingData.bmm2TilingData.set_shareMode(0);
    mlaTilingData.bmm2TilingData.set_shareL1Size(ascendPlatformInfo.l1Size);
    mlaTilingData.bmm2TilingData.set_shareL0CSize(ascendPlatformInfo.l0CSize);
    return true;
}

bool PromptFlashAttentionTiling::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
    int64_t tmpDBasicBlock, int64_t batch)
{
    matmul_tiling::MatmulApiTiling bmm1(ascendPlatformInfo);
    matmul_tiling::MatmulApiTiling bmm2(ascendPlatformInfo);
    return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm1, bmm2);
}

void PromptFlashAttentionTiling::CalcS1S2BasicBlock(const BufferNum &bufferNum)
{
    s1BasicBlockBest = 256L;
    s1BasicBlock = std::min(s1BasicBlockBest, alignedS1);
    s2BasicBlock = std::min(128L, alignedS2);
    s1VecBasicBlock = s1BasicBlock / AIV_AIC_NUM_RATIO;
}

int64_t PromptFlashAttentionTiling::CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum)
{
    // if S2 basic block is min value 16, s1 basic block can reach max value, then we get:
    // s1 * 16 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp =>
    // s1 * (16 * X + D * Y + (expNum + 2) * (32 / sizeof(T))) * sizeof(T) + apiTmp
    // just ignore apiTmp now, consider it at last
    int64_t alignUnit = BYTE_BLOCK / dataTypeSize;
    int64_t maxS1BasicBlock = ascendPlatformInfo.ubSize / dataTypeSize /
        (FRACTAL_NUM * bufferNum.bufferS1S2Num + actualD * bufferNum.bufferS1DNum +
         (bufferNum.bufferExpNum + 2) * alignUnit); // here 2 means FlashSoftMax sum and max output
    return AlignDown(maxS1BasicBlock, FRACTAL_NUM);
}

int64_t PromptFlashAttentionTiling::CalcMaxS2BasicBlockSize(const BufferNum &bufferNum,
                                                                int64_t tmpS1BasicBlock)
{
    // used UB: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    // if D full load, use alignedD in above formula
    // if D not full load, use S2 basic block var in above formula
    // just ignore apiTmp now, consider it at last
    int64_t tmpS2BasicBlock;
    tmpS2BasicBlock = (ascendPlatformInfo.ubSize - tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK -
                        tmpS1BasicBlock * alignedD * bufferNum.bufferS1DNum * dataTypeSize) /
                        (tmpS1BasicBlock * bufferNum.bufferS1S2Num * dataTypeSize);
    return std::min(AlignDown(tmpS2BasicBlock, FRACTAL_NUM), alignedS2);
}

bool PromptFlashAttentionTiling::IsBasicBlockInSoftMax(const ge::Shape &shape)
{
    // 2 axes at least
    if (shape.GetDimNum() < 2) {
        return false;
    }

    int64_t lastAxis = shape.GetDim(shape.GetDimNum() - 1);
    // last axis should be less than 2048 and fullfil 64 times
    int64_t basicLastAxis = 64;
    int64_t lastAxisNum = 2048;
    if (lastAxis > lastAxisNum || lastAxis % basicLastAxis != 0) {
        return false;
    }

    int64_t preAxes = 1;
    for (size_t idx = 0; idx < shape.GetDimNum() - 1; ++idx) {
        preAxes *= shape.GetDim(idx);
    }

    // all axes except last one should be 8 times
    return preAxes % 8 == 0;
}

void PromptFlashAttentionTiling::GetBufferNum(BufferNum &bufferNum)
{
    bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
}

void PromptFlashAttentionTiling::MatchTemplate()
{
    // UB Size calc logic: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    BufferNum bufferNum;
    GetBufferNum(bufferNum);

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    CalcS1S2BasicBlock(bufferNum);
    nRatio = 8L;
    dBasicBlock = std::min(128L, alignedD);
    (void)CalcUBSize();
}

ge::graphStatus PromptFlashAttentionTiling::CheckInputShapeWhenLayoutIsTND(ContextParamsForPFATiling& contextKeyParams) {
    const gert::Tensor* actSeqLenData = contextKeyParams.actualSeqenceLengthQ;
    const gert::Tensor* actSeqLenDataKV = contextKeyParams.actualSeqenceLengthKV;
    int64_t actSeqLenDims = (actSeqLenData != nullptr) ? actSeqLenData->GetShapeSize() : 0;
    int64_t actSeqLenKVDims = (actSeqLenDataKV != nullptr) ? actSeqLenDataKV->GetShapeSize() : 0;
    OPS_ERR_IF((actSeqLenData == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, actualSeqenceLengthQ is required, but now is nullptr!"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((actSeqLenDataKV == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, actualSeqenceLengthKV is required, but now is nullptr!"),
        return ge::GRAPH_FAILED);
    // tiling下沉和acl graph当前不支持
    OPS_ERR_IF(((actSeqLenData->GetData<int64_t>() == nullptr) || (actSeqLenDataKV->GetData<int64_t>() == nullptr)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, not support tiling_schedule_optimize = True or config mode is reduce-overhead!"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((actSeqLenDims == 0),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, actualSeqenceLengthQ is required, but the number of element in it is 0!"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((actSeqLenKVDims == 0),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, actualSeqenceLengthKV is required, but the number of element in it is 0!"),
        return ge::GRAPH_FAILED);
    int64_t lastSeqLen = static_cast<int64_t>(actSeqLenData->GetData<int64_t>()[actSeqLenDims - 1]);
    int64_t lastSeqLenKV = static_cast<int64_t>(actSeqLenDataKV->GetData<int64_t>()[actSeqLenKVDims - 1]);

    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    uint32_t queryT = queryShape->GetStorageShape().GetDim(FIRST_DIM);
    uint32_t keyT = keyShape->GetStorageShape().GetDim(FIRST_DIM);
    uint32_t valueT = valueShape->GetStorageShape().GetDim(FIRST_DIM);

    OPS_ERR_IF((queryShape->GetStorageShape().GetDimNum() != DIM_NUM_3 || keyShape->GetStorageShape().GetDimNum() != DIM_NUM_3 ||
                valueShape->GetStorageShape().GetDimNum() != DIM_NUM_3),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, querDim(%zu) keyDim(%zu) valueDim(%zu) must be 3.",
                queryShape->GetStorageShape().GetDimNum(), keyShape->GetStorageShape().GetDimNum(), valueShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(queryT != lastSeqLen,
               OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, queryT(%u) must be equal to the last element of actualSeqenceLengthQ(%ld)", queryT, lastSeqLen),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF((keyT != lastSeqLenKV) || (valueT != lastSeqLenKV),
               OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, keyT(%u) and valueT(%u) must be equal to the last element of actualSeqenceLengthKV(%ld)", keyT, valueT, lastSeqLenKV),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF((queryT > TLIMIT) || (keyT > TLIMIT) || (valueT > TLIMIT),
               OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, T cannot be greater than 65536(64K), queryT=%u, keyT=%u, valueT=%u", queryT, keyT, valueT),
               return ge::GRAPH_FAILED);
    uint32_t queryN = queryShape->GetStorageShape().GetDim(SECOND_DIM);
    uint32_t keyN = keyShape->GetStorageShape().GetDim(SECOND_DIM);
    uint32_t valueN = valueShape->GetStorageShape().GetDim(SECOND_DIM);
    OPS_ERR_IF((queryN != keyN) || (keyN != valueN),
               OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, the values of queryN(%u), keyN(%u), valueN(%u) must be equal", queryN, keyN, valueN),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF((queryN != N_SIZE_8) && (queryN != N_SIZE_16) && (queryN != N_SIZE_32) && (queryN != N_SIZE_64) && (queryN != N_SIZE_128),
               OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, the value of N(%u) must be 8/16/32/64/128", queryN),
               return ge::GRAPH_FAILED);
    uint32_t queryD = queryShape->GetStorageShape().GetDim(THIRD_DIM);
    uint32_t keyD = keyShape->GetStorageShape().GetDim(THIRD_DIM);
    uint32_t valueD = valueShape->GetStorageShape().GetDim(THIRD_DIM);
    OPS_ERR_IF((queryD != D_SIZE_192) || (keyD != D_SIZE_192) || ((valueD != D_SIZE_192) && (valueD != D_SIZE_128)),
               OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, queryD(%u) or keyD(%u) must be 192, valueD(%u) must be 192/128!", queryD, keyD, valueD),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(contextKeyParams.inputDataType != ge::DT_BF16,
               OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is TND, inputDataType should be bf16"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckActSeqWhenLayoutIsTND(ContextParamsForPFATiling& contextKeyParams) {
    const gert::Tensor* actSeqLen = contextKeyParams.actualSeqenceLengthQ;
    const gert::Tensor* actSeqLenKV = contextKeyParams.actualSeqenceLengthKV;
    size_t batchOfQuery = actSeqLen->GetShapeSize();
    size_t batchOfKey = actSeqLenKV->GetShapeSize();
    OPS_ERR_IF(batchOfQuery != batchOfKey,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When layout is TND, the length of actualSeqenceLengthQ(%zu) and actualSeqenceLengthKV(%zu) must be equal.",
            batchOfQuery, batchOfKey),
        return ge::GRAPH_FAILED);
    for (uint32_t i = LOOP_BEGIN_NUM; i < batchOfQuery; ++i) {
        int64_t curActSeq = actSeqLen->GetData<int64_t>()[i];
        int64_t curActSeqKV = actSeqLenKV->GetData<int64_t>()[i];
        OPS_ERR_IF(curActSeq < NUM_0 || curActSeqKV < NUM_0,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When layout is TND, actualSeqenceLengthQ[%u]=%ld and actualSeqenceLengthKV[%u]=%ld must >= 0",
            i, curActSeq, i, curActSeqKV),
            return ge::GRAPH_FAILED);
    }
    int64_t lastActSeq = 0;
    int64_t lastActSeqKV = 0;
    for (uint32_t i = LOOP_BEGIN_NUM; i < batchOfQuery; ++i) {
        int64_t curActSeq = actSeqLen->GetData<int64_t>()[i];
        int64_t curActSeqKV = actSeqLenKV->GetData<int64_t>()[i];
        OPS_ERR_IF(curActSeq < lastActSeq,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When layout is TND, Actual_seq_lengths must be not decreasing, but it's not at %u, actSeqLen[%u]=%ld, actSeqLen[%u]=%ld",
                i, i, curActSeq, i-1, lastActSeq),
            return ge::GRAPH_FAILED);
        if (!enablePA) {
            OPS_ERR_IF(curActSeqKV < lastActSeqKV,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "When layout is TND, Actual_seq_lengths_kv must be not decreasing, but it's not at %u, actSeqLenKV[%u]=%ld, actSeqLenKV[%u]=%ld",
                    i, i, curActSeqKV, i-1, lastActSeqKV),
                return ge::GRAPH_FAILED);
        }
        lastActSeq = curActSeq;
        lastActSeqKV = curActSeqKV;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::RunBigKernelTilingWithParams(ContextParamsForPFATiling& contextKeyParams,
                                            uint64_t& tilingKey,
                                            uint32_t& blockDimToBeSet,
                                            PromptFlashAttentionTilingData& tilingData) {
    uint64_t l0CSize;
    uint64_t l1Size;
    uint64_t ubSize;
    auto compileInfoPtr = contextKeyParams.compileInfoPtr;
    contextKeyParamsPtr = &contextKeyParams;      // In subsequent rectification, contextKeyParams will be written as a member variable of the class.

    OPS_ERR_IF(compileInfoPtr == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);

    ubSize = compileInfoPtr->ubSize;
    l1Size = compileInfoPtr->l1Size;
    l0CSize = compileInfoPtr->l0CSize;

    coreNum = compileInfoPtr->aivNum;
    OPS_ERR_IF(coreNum == 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "coreNum is 0"),
                    return ge::GRAPH_FAILED);
    aivNum = compileInfoPtr->aivNum;
    OPS_ERR_IF(aivNum == 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "aivNum is 0"),
                    return ge::GRAPH_FAILED);
    aicNum = compileInfoPtr->aicNum;
    OPS_ERR_IF(aicNum == 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "aicNum is 0"),
                    return ge::GRAPH_FAILED);
    curShortSocName = compileInfoPtr->socShortName;
    defaultSysWorkspaceSize = compileInfoPtr->defaultSysWorkspaceSize;

    ascendPlatformInfo.socVersion = compileInfoPtr->socShortName;
    ascendPlatformInfo.l1Size = compileInfoPtr->l1Size;
    ascendPlatformInfo.l0CSize = compileInfoPtr->l0CSize;
    ascendPlatformInfo.l0ASize = compileInfoPtr->l0ASize;
    ascendPlatformInfo.l0BSize = compileInfoPtr->l0BSize;
    ascendPlatformInfo.ubSize = compileInfoPtr->ubSize;
    OPS_LOG_I(contextKeyParams.opName,
            "ascendPlatformInfo:aivNum = %u, aicNum = %u, l1Size = %lu, l0CSize = %lu, l0ASize = %lu, l0BSize = %lu, ubSize = %lu!",
            aivNum, aicNum, ascendPlatformInfo.l1Size, ascendPlatformInfo.l0CSize, ascendPlatformInfo.l0ASize, ascendPlatformInfo.l0BSize, ascendPlatformInfo.ubSize);
    
    int32_t outputDataTypeSize = FLOAT32SIZE;
    if(CheckIOType(contextKeyParams, tilingData, outputDataTypeSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OPS_ERR_IF(((inputType == ge::DT_FLOAT) || (outputType == ge::DT_FLOAT)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "inputType(%d) and outputType(%d) can not be DT_FLOAT", inputType, outputType),
                    return ge::GRAPH_FAILED);

    const int64_t* innerPrecisePtr = contextKeyParams.innerPrecisePtr;

    innerPrecise = innerPrecisePtr ? *innerPrecisePtr : HIGH_PERFORMANCE; // 910B defaults to high-performance, while 310P's high performance refers to high accuracy (without using approximate calculations).

    if (innerPrecise >= 4) { // 0: Invalid plural number; 4: Invalid if greater than or equal to 4; 0,1,2,3 are effective values for innerPrecise.
        OPS_LOG_W(contextKeyParams.opName, "innerPrecise [%lu] should be 0,1,2,3, please check.", innerPrecise);
    }
    // Determine if the bit1 bit of innerPrecise requires invalid correction.
    if ((innerPrecise >> 1) & 1) {
        tilingData.promptAttentionBaseParams.set_isRowInvalid(1U);
    } else {
        tilingData.promptAttentionBaseParams.set_isRowInvalid(0U);
    }
    // Determine the bit0 bit of innerPrecise, high-performance or high-precision mode.
    innerPrecise = ((innerPrecise >> 0) & 1) ? HIGH_PERFORMANCE : HIGH_PRECISION;
    OPS_ERR_IF(((innerPrecise != HIGH_PERFORMANCE) && (innerPrecise != HIGH_PRECISION)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "precision mode[%lu] should be 0 or 1", innerPrecise),
                    return ge::GRAPH_FAILED); // Currently only supports high-precision 0 and high-performance 1
    if (inputType != ge::DT_FLOAT16) {
        OPS_LOG_W(contextKeyParams.opName,
            "innerPrecise will not take effect when input type is %d!", inputType);
    }
    std::string tempLayoutStr(contextKeyParams.layout);
    OPS_ERR_IF((tempLayoutStr == "TND" && contextKeyParams.pseShift != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When Layout is TND, not support PSE"),
        return ge::GRAPH_FAILED);

    // FP16 pse is forced to enter high-precision mode.
    if ((contextKeyParams.pseShift != nullptr) && (inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PERFORMANCE)) {
        innerPrecise = HIGH_PRECISION;
        OPS_LOG_W(contextKeyParams.opName, "when the input is fp16, the mode is forcibly switched to high-precision!");
    }

    // blockTable is considered a PA scenario if it is not empty.
    if (contextKeyParams.blockTable != nullptr) {
        OPS_ERR_IF(tempLayoutStr == "TND",
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When layout is TND, not support Page Attention"),
            return ge::GRAPH_FAILED);
        enablePA = true;
        OPS_ERR_IF(contextKeyParams.blockSize == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "blockSize can't be null when PA enable"),
                    return ge::GRAPH_FAILED);
    }

    if (enablePA) {
        OPS_ERR_IF(inputType == ge::DT_INT8,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "Query DataType can't be INT8 when PA enable"),
                    return ge::GRAPH_FAILED);

        OPS_ERR_IF(curShortSocName == platform_ascendc::SocVersion::ASCEND310P,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "not support 310P when blockTable is not null"),
                    return ge::GRAPH_FAILED);

        OPS_ERR_IF(contextKeyParams.isKvContinuous == 0,   // The interception that is mutually exclusive with the left padding has been implemented in FIA.
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "not support tensorlist when blockTable is not null"),
                    return ge::GRAPH_FAILED);
        OPS_ERR_IF(enableKvAntiquant,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "not support antiquant when blockTable is not null"),
                    return ge::GRAPH_FAILED);
    }

    if ((curShortSocName == platform_ascendc::SocVersion::ASCEND310P) && innerPrecise == HIGH_PRECISION) {
        softmaxDataTypeNZ_ = FLOAT16SIZE;
        innerPrecise = HIGH_PERFORMANCE;
    }
    if (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PERFORMANCE)) ||
        (inputType == ge::DT_INT8)) {
        softmaxDataTypeSize = FLOAT16SIZE; // The default size is fp32.
    }

    uint32_t maskElemSize = dataTypeSize;
    if(CheckMaskType(contextKeyParams, tilingData, maskElemSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    typeByteNum = BYTE_BLOCK / dataTypeSize;
    outputTypeByteNum = BYTE_BLOCK / outputDataTypeSize;
    softmaxTypeByteNum = BYTE_BLOCK / softmaxDataTypeSize;
    maskTypeByteNum = BYTE_BLOCK / maskElemSize;

    tilingData.promptAttentionBaseParams.set_maskTypeByteNum(maskTypeByteNum);
    tilingData.promptAttentionBaseParams.set_softmaxTypeByteNum(softmaxTypeByteNum);
    tilingData.promptAttentionBaseParams.set_outputTypeByteNum(outputTypeByteNum);
    tilingData.promptAttentionBaseParams.set_typeByteNum(typeByteNum);
    // Get different shape.
    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    const gert::StorageShape* pseShiftShape = contextKeyParams.pseShiftShape;
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    const gert::StorageShape* deqScale1Shape = contextKeyParams.deqScale1Shape;
    const gert::StorageShape* quantScale1Shape = contextKeyParams.scale1Shape;
    const gert::StorageShape* deqScale2Shape = contextKeyParams.deqScale2Shape;
    const gert::StorageShape* quantScale2Shape = contextKeyParams.scale2Shape;
    const gert::StorageShape* quantOffset2Shape = contextKeyParams.offset2Shape;
    const gert::StorageShape* antiquantScaleShape = contextKeyParams.antiquantScaleShape;
    const gert::StorageShape* antiquantOffsetShape = contextKeyParams.antiquantOffsetShape;
    const gert::StorageShape* outShape = contextKeyParams.outputShape;
    const gert::StorageShape* SoftmaxLseOutShape = contextKeyParams.lseoutputShape;

    uint32_t deqScaleTypeFlag = (contextKeyParams.deqScaleType == DT_UINT64) ? 0U : 1U;
    uint32_t deqScale2TypeFlag = (contextKeyParams.deqScale2Type == DT_UINT64) ? 0U : 1U;

    tilingData.promptAttentionBaseParams.set_deqScaleFlag(deqScaleTypeFlag);
    tilingData.promptAttentionBaseParams.set_deqScale2Flag(deqScale2TypeFlag);

    OPS_ERR_IF(((contextKeyParams.inputDataType == ge::DT_INT8) && (contextKeyParams.outputDataType == ge::DT_FLOAT16) && ((contextKeyParams.scale2Shape != nullptr) || (contextKeyParams.offset2Shape != nullptr))),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "When query dtype is int8 and output dtype is fp16, quantScale2 and quantOffset2 should be null."),
                    return ge::GRAPH_FAILED);

    // KV prefix check.
    isKVHasPrefix = contextKeyParams.keySharedPrefix != nullptr && contextKeyParams.valueSharedPrefix != nullptr ? true : false;
    OPS_ERR_IF((!isKVHasPrefix && (contextKeyParams.keySharedPrefix != nullptr || contextKeyParams.valueSharedPrefix != nullptr)),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when system prefix is used, key_shared_prefix and value_shared_prefix are required!"),
                return ge::GRAPH_FAILED);
    if (isKVHasPrefix) {
        // The prefix does not support tensorlist, PA, or left padding
        OPS_ERR_IF((contextKeyParams.isKvContinuous == 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when tensorlist is used, system prefix is not supported!"),
                    return ge::GRAPH_FAILED);
        OPS_ERR_IF((enablePA),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when system prefix is used, page attention is not supported!"),
                    return ge::GRAPH_FAILED);
        OPS_ERR_IF(((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when system prefix is used, leftpadding is not supported!"),
                    return ge::GRAPH_FAILED);
        OPS_ERR_IF((contextKeyParams.inputDataType == ge::DT_INT8) && (contextKeyParams.kDataType == ge::DT_INT8),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when system prefix is used, query and key/value should not both be int8!"),
                    return ge::GRAPH_FAILED);

        uint32_t prefixKeyDim = contextKeyParams.keySharedPrefix->GetStorageShape().GetDimNum();
        uint32_t prefixValueDim = contextKeyParams.valueSharedPrefix->GetStorageShape().GetDimNum();
        uint32_t KVDim = keyShape->GetStorageShape().GetDimNum();
        OPS_ERR_IF(((prefixKeyDim != KVDim) || (prefixKeyDim != prefixValueDim)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "dim num of key_shared_prefix and value_shared_prefix should be same with KV, "
                    "but key_shared_prefix dim(%u), value_shared_prefix dim(%u), KV dim(%u)!", prefixKeyDim, prefixValueDim, KVDim),
                    return ge::GRAPH_FAILED);
        for (uint32_t i = 0; i < prefixKeyDim; i++) {
            uint32_t tmpPrefixKeyDim = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(i);
            uint32_t tmpPrefixValueDim = contextKeyParams.valueSharedPrefix->GetStorageShape().GetDim(i);
            OPS_ERR_IF(((tmpPrefixKeyDim == 0) || (tmpPrefixValueDim == 0)),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "key_shared_prefix and value_shared_prefix not support empty tensor, "
                        "but key_shared_prefix[%u]:%u, value_shared_prefix[%u]:%u!", i, tmpPrefixKeyDim, i, tmpPrefixValueDim),
                        return ge::GRAPH_FAILED);
            OPS_ERR_IF(((tmpPrefixKeyDim != tmpPrefixValueDim)),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of key_shared_prefix should be same with value_shared_prefix, "
                        "but key_shared_prefix[%u]:%u, value_shared_prefix[%u]:%u!", i, tmpPrefixKeyDim, i, tmpPrefixValueDim),
                        return ge::GRAPH_FAILED);
        }
    }

    // Set the last dim size of mask.
    SetMaskSize(attenMaskShape, tilingData);

    // Internal log printing, no need to print here, same below.
    if(CheckShape(contextKeyParams, queryShape, keyShape, valueShape, outShape, pseShiftShape, attenMaskShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // In the scene of entering the image, there may be a situation where out is an empty tensor. Here, out is empty and size 0 is processed, which is equivalent to doing nothing and returning directly.
    if ((keyShape->GetStorageShape().GetShapeSize() == 0) || (valueShape->GetStorageShape().GetShapeSize() == 0) ||
        (outShape->GetStorageShape().GetShapeSize() == 0) || (contextKeyParams.emptyTensor == 1)) {
        tilingKey = EMPTY_KV_TILING_KEY;
        OPS_ERR_IF(GetAndCheckEmptyQueryShape(contextKeyParams, queryShape) == ge::GRAPH_FAILED,
                   OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "GetAndCheckEmptyQueryShape failed."),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF((contextKeyParams.inputDataType != ge::DT_FLOAT16) || (contextKeyParams.kDataType != ge::DT_FLOAT16) || (contextKeyParams.vDataType != ge::DT_FLOAT16),
                   OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when input or output is empty tensor, input datatype should be float16."),
                   return ge::GRAPH_FAILED);
        PromptFlashAttentionInitOutputSplit(outShape->GetStorageShape().GetShapeSize(), tilingData, coreNum);
        tilingData.promptAttentionInitOutputParams.set_needInit(1);
        // core need to be full
        PromptAttentionInitOutputParams *initParams = &tilingData.promptAttentionInitOutputParams;
        uint32_t singleCoreSize = initParams->get_singleCoreSize();
        uint32_t actualCore = (singleCoreSize > 0) ? (outShape->GetStorageShape().GetShapeSize() + singleCoreSize - 1) / singleCoreSize : coreNum;
        blockDimToBeSet = ascendcPlatform.CalcTschBlockDim(actualCore, aicNum, coreNum);

        size_t* workspace = contextKeyParams.workspaceSize;
        const size_t sysWorkspaceSize = 16 * 1024 * 1024;  // workspace needs at least this much
        workspace[0] = sysWorkspaceSize;
        return ge::GRAPH_SUCCESS;
    }
    tilingData.promptAttentionBaseParams.set_useMask(1);
    if (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0))
        || (attenMaskShape == nullptr)) {
        tilingData.promptAttentionBaseParams.set_useMask(0);
    }

    if (inputType == ge::DT_INT8) {
        OPS_ERR_IF((deqScale1Shape == nullptr) || (quantScale1Shape == nullptr) || (deqScale2Shape == nullptr),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "dequant scale or first quant scale is nullptr when input type is int8."),
                    return ge::GRAPH_FAILED);
        OPS_ERR_IF((deqScale1Shape != nullptr && deqScale1Shape->GetStorageShape().GetShapeSize() == 0) ||
                        (quantScale1Shape != nullptr && quantScale1Shape->GetStorageShape().GetShapeSize() == 0) ||
                        (deqScale2Shape != nullptr && deqScale2Shape->GetStorageShape().GetShapeSize() == 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "dequant scale or first quant scale is empty tensor when input type is int8."),
                    return ge::GRAPH_FAILED);
    }

    const int32_t* n = contextKeyParams.headsNumber; // num_heads of q
    const int32_t* sparseMode = contextKeyParams.sparseMode;
    const int64_t* nextTokens = contextKeyParams.nextToken;
    const int64_t* preTokens = contextKeyParams.preToken;
    const float* scaleValue = contextKeyParams.scaleValue;
    const int32_t* blockSize = contextKeyParams.blockSize;

    int64_t sparsePreTokens;
    int64_t sparseNextTokens;
    int32_t sparseModeVal = 0;
    // KV consistency check.
    OPS_ERR_IF(CheckKeyValueParamsConsistency(contextKeyParams) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "key value consistency check failed!"),
                    return ge::GRAPH_FAILED);

    const int32_t* numKeyValueHeads = contextKeyParams.numKeyValueHeads;
    if (!SetTilingHeadNumRatio(contextKeyParams, n, numKeyValueHeads, tilingData)) {
        return ge::GRAPH_FAILED;
    }

    // Get different dims.
    uint32_t seqInnerSize = 0U; // kv_s
    uint32_t h = 0U;
    uint32_t valueD = 0U;
    uint32_t s = 0U;
    uint32_t b = 0U;
    uint32_t bKV = 0U;
    uint32_t prefixSeqInnerSize = 0;
    uint32_t bPreifx = 0U;
    uint32_t nPreifx = 0U;
    uint32_t hPreifx = 0U;
    uint32_t dPreifx = 0U;

    const gert::Tensor* tempData = contextKeyParams.actualSeqenceLengthQ;
    const gert::Tensor* tempDataKV = contextKeyParams.actualSeqenceLengthKV;
    size_t actualLenDims = (tempData != nullptr) ? tempData->GetShapeSize() : 0;
    size_t actualLenDimsKV = (tempDataKV != nullptr) ? tempDataKV->GetShapeSize() : 0;
    uint32_t isActualSeqLengthsNull = contextKeyParams.fromTilingSink == 0 ? (actualLenDims == 0 || tempData == nullptr || tempData->GetData<int64_t>() == nullptr) : 1;
    uint32_t isActualSeqLengthsKVNull = contextKeyParams.fromTilingSink == 0 ? (actualLenDimsKV == 0 || tempDataKV == nullptr || tempDataKV->GetData<int64_t>() == nullptr) : 1;
    OPS_ERR_IF(enablePA && (isActualSeqLengthsKVNull == 1) && (contextKeyParams.fromTilingSink == 0),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "actual seq length kv can't be null when blockTable is not null"),
                        return ge::GRAPH_FAILED);
    OPS_ERR_IF((inputLayout == InputLayout::TND) && ((actualLenDims > MAX_VAR_LEN_SEQ_LEN) || (actualLenDimsKV > MAX_VAR_LEN_SEQ_LEN)),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "array length of actual_seq_lengths_q(%lu) and actual_seq_lengths_kv(%lu) must be less than or equal to %ld when input layout is TND.",
                        actualLenDims, actualLenDimsKV,  MAX_VAR_LEN_SEQ_LEN), return ge::GRAPH_FAILED);
    if (inputLayout == (InputLayout::SH) && (actualLenDimsKV != 0)) {
        OPS_LOG_W(contextKeyParams.opName, "actual_seq_lengths_kv is useless for SH format!");
    }
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        unsigned int ret;
        ret = GetBasicShape310P(b, bKV, s, h, seqInnerSize, queryShape, keyShape, *n, actualLenDims, actualLenDimsKV);
        OPS_ERR_IF(ret == GRAPH_FAILED,
                        OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "execute is failed."),
                        return ge::GRAPH_FAILED);
        OPS_ERR_IF((s > 65536) || (seqInnerSize > 65536),
                        OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "310P not support Qs or KVs lager than 65536,Qs = %u, Kvs = %u", s, seqInnerSize),
                        return ge::GRAPH_FAILED);
        OPS_ERR_IF((tilingData.promptAttentionBaseParams.get_useMask()!= 0 && (s % 16 != 0 || seqInnerSize % 16 != 0 || s != seqInnerSize)),
                        OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "attention mask must be NULL, when Qs,Kvs is unAlign or Qs is not equal to Kvs, Qs = %u, Kvs = %u", s, seqInnerSize),
                        return ge::GRAPH_FAILED);
        OPS_ERR_IF(((*preTokens < static_cast<int32_t>(s)) || (*nextTokens < static_cast<int32_t>(seqInnerSize) && *nextTokens != 0)),
                        OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "pretokens should lager than Qs, nexttokens should be 0 or larger than Kvs, Qs = %u, Kvs = %u, preTokens = %ld, nextTokens = %ld", s, seqInnerSize, *preTokens, *nextTokens),
                        return ge::GRAPH_FAILED);
    } else {
        if (inputLayout == InputLayout::BNSD || inputLayout == InputLayout::NSD) {
            if (queryShape->GetStorageShape().GetDimNum() == 3) { // dim num: 3
                b = 1;
                bKV = 1;
                s = queryShape->GetStorageShape().GetDim(1);
                seqInnerSize = keyShape->GetStorageShape().GetDim(1);
                h = (*n) * queryShape->GetStorageShape().GetDim(2); // dim num: 2
                prefixSeqInnerSize = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(1) : 0;
            } else {
                b = queryShape->GetStorageShape().GetDim(0);
                bKV = keyShape->GetStorageShape().GetDim(0);
                s = queryShape->GetStorageShape().GetDim(2); // dim num: 2
                seqInnerSize = keyShape->GetStorageShape().GetDim(2); // dim num: 2
                h = queryShape->GetStorageShape().GetDim(1) * queryShape->GetStorageShape().GetDim(3);  // dim num: 3
                valueD = valueShape->GetStorageShape().GetDim(3);
                prefixSeqInnerSize = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(INDEX_2) : 0;
                bPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(0) : 0;
                nPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(1) : 0;
                dPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(INDEX_3) : 0;
            }
        } else if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) ||
            (inputLayout == InputLayout::SH)) {
            if (queryShape->GetStorageShape().GetDimNum() == NUM_2) { // dim num: 2
                b = actualLenDims == 0 ? 1 : actualLenDims; // When the input layout is SH and actual_seq is not input, the batch of query is set to 1.
                bKV = actualLenDimsKV == 0 ? 1 : actualLenDimsKV; // When the input layout is SH and actual_seqkv is not input, the batch of key/value is set to 1.
                s = queryShape->GetStorageShape().GetDim(0);
                h = queryShape->GetStorageShape().GetDim(1);
                seqInnerSize = keyShape->GetStorageShape().GetDim(0);
                prefixSeqInnerSize = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(0) : 0;
            } else if (queryShape->GetStorageShape().GetDimNum() == 3) { // 3 : BSH
                b = queryShape->GetStorageShape().GetDim(0);
                bKV = keyShape->GetStorageShape().GetDim(0);
                s = queryShape->GetStorageShape().GetDim(1);
                h = queryShape->GetStorageShape().GetDim(2); // dim num: 2
                seqInnerSize = keyShape->GetStorageShape().GetDim(1);
                prefixSeqInnerSize = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(1) : 0;
                bPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(0) : 0;
                hPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(INDEX_2) : 0;
            } else { // BSND
                b = queryShape->GetStorageShape().GetDim(0);
                bKV = keyShape->GetStorageShape().GetDim(0);
                s = queryShape->GetStorageShape().GetDim(1);
                h = queryShape->GetStorageShape().GetDim(INDEX_2) *
                    queryShape->GetStorageShape().GetDim(INDEX_3);
                seqInnerSize = keyShape->GetStorageShape().GetDim(1);
                prefixSeqInnerSize = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(1) : 0;
                bPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(0) : 0;
                nPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(INDEX_2) : 0;
                dPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(INDEX_3) : 0;
            }
        } else if (inputLayout == InputLayout::TND) {
            if (CheckInputShapeWhenLayoutIsTND(contextKeyParams) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED; 
            }
            if (CheckActSeqWhenLayoutIsTND(contextKeyParams) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            int64_t actualSeqQLen = 0;
            int64_t actualSeqKVLen = 0;
            int64_t t1Size = queryShape->GetStorageShape().GetDim(0);
            int64_t t2Size = keyShape->GetStorageShape().GetDim(0);
            valueD = valueShape->GetStorageShape().GetDim(2);
            realT1Size = t1Size;
            std::fill(actualSeqLenData.begin(), actualSeqLenData.end(), 0);
            std::fill(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0);
            GetActualSeqLenData(ACTUAL_SEQ_Q_INDEX, actualSeqLenData, actualSeqQLen);
            GetActualSeqLenData(ACTUAL_SEQ_KV_INDEX, actualSeqLenKvData, actualSeqKVLen);
            OPS_ERR_IF(actualSeqQLen != actualSeqKVLen,
                       OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "VarLen scene, q is not equal kv."), return false);
            bSize = actualSeqQLen;
            accumS1 = std::accumulate(actualSeqLenData.begin(), actualSeqLenData.end(), 0LL);
            accumS2 = std::accumulate(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0LL);
            OPS_ERR_IF(
                t1Size < accumS1 || t2Size < accumS2,
                OPS_REPORT_VECTOR_INNER_ERR(
                    contextKeyParams.opName,
                    "Query T(%ld) and key T(%ld) need larger than respectively sum of seqLen(%ld) and sekvLen(%ld).",
                    t1Size, t2Size, accumS1, accumS2),
                return false);
            maxS1Val = *std::max_element(actualSeqLenData.begin(), actualSeqLenData.end());
            maxS2Val = *std::max_element(actualSeqLenKvData.begin(), actualSeqLenKvData.end());
            s1Size = maxS1Val;
            s2Size = maxS2Val;
            n1Size = *n;
            OPS_ERR_IF(n1Size != queryShape->GetStorageShape().GetDim(1),
                       OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "head_num is [%ld], but got query dim1 [%ld].", n1Size,
                                                   queryShape->GetStorageShape().GetDim(1)),
                       return false);
            n2Size = keyShape->GetStorageShape().GetDim(1);
            OPS_ERR_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "N2 is zero."), return false);
            gSize = queryShape->GetStorageShape().GetDim(1) / n2Size;
            dSize = queryShape->GetStorageShape().GetDim(2);
            h1 = n1Size * dSize;
            h2 = n2Size * dSize;
            h = h1;
            s1StrideSize = gSize * n2Size * dSize;
            s2StrideSize = n2Size * dSize;
            int64_t seqQTotal = 0;
            for (int64_t i = 0; i < bSize; i++) {
                seqQTotal += actualSeqLenData[i];
            }
            s1BasicBlockBest = 128;
        } else {
            return ge::GRAPH_FAILED;
        }
        if (contextKeyParams.isKvContinuous == 0) {
            seqInnerSize = contextKeyParams.maxKVs;
        }
    }

    uint32_t actualSharedPrefixLen = 0U;
    if ((isKVHasPrefix) && (contextKeyParams.actualSharedPrefixLen != nullptr) &&
        (contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetShapeSize() > 0) && (contextKeyParams.fromTilingSink == 0U)) {
        uint32_t prefixDimNum = contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetDimNum();
        OPS_ERR_IF((prefixDimNum != 1),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "actualSharedPrefixLen dim num(%u) should be 1!", prefixDimNum),
                    return ge::GRAPH_FAILED);
        uint32_t prefixShapeSize = contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetShapeSize();
        OPS_ERR_IF((prefixShapeSize != 1),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "actualSharedPrefixLen length(%u) should be 1!", prefixShapeSize),
                    return ge::GRAPH_FAILED);
        OPS_ERR_IF((contextKeyParams.actualSharedPrefixLen->GetData<int64_t>() == nullptr),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "input actualSharedPrefixLen GetData is nullptr!"),
                    return ge::GRAPH_FAILED);
        actualSharedPrefixLen = static_cast<uint32_t>(contextKeyParams.actualSharedPrefixLen->GetData<int64_t>()[0]);
        OPS_ERR_IF((actualSharedPrefixLen > prefixSeqInnerSize),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "actualSharedPrefixLen(%u) must be in range[0, %u]!", actualSharedPrefixLen, prefixSeqInnerSize),
                    return ge::GRAPH_FAILED);
        tilingData.promptAttentionBaseParams.set_isActualSharedPrefixLenNull(0);
    } else {
        tilingData.promptAttentionBaseParams.set_isActualSharedPrefixLenNull(1);
        actualSharedPrefixLen = prefixSeqInnerSize;
    }
    tilingData.promptAttentionBaseParams.set_prefixSeqInnerSize(prefixSeqInnerSize);

    if (isKVHasPrefix) {
        OPS_ERR_IF((bPreifx != 1),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "prefix batch num(%u) only support 1!", bPreifx),
                    return ge::GRAPH_FAILED);
        if (inputLayout == InputLayout::BSH) {
            OPS_ERR_IF((hPreifx != h / tilingData.promptAttentionBaseParams.get_headNumRatio()),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "prefix H(%u) should be same with KV H(%u)!", hPreifx, h / tilingData.promptAttentionBaseParams.get_headNumRatio()),
                        return ge::GRAPH_FAILED);
        } else {
            OPS_ERR_IF((nPreifx != (*n) / tilingData.promptAttentionBaseParams.get_headNumRatio()) || (dPreifx != h / (*n)),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "prefix N(%u) and D(%u) should be same with KV N(%u) and D(%u)!", nPreifx, dPreifx, (*n) / tilingData.promptAttentionBaseParams.get_headNumRatio(), h / (*n)),
                        return ge::GRAPH_FAILED);
        }
    }

    if (((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::SH)) && (h > 65535)) {  // Moving into stride cannot exceed 65535
        OPS_LOG_W(contextKeyParams.opName, "h(%u) is larger than 65535, which may cause precision problem! Please use BNSD or BNSD_BSND instead.", h);
    }

    // PA scene does not have B-axis, no verification.
    OPS_ERR_IF((b != bKV) && (contextKeyParams.isKvContinuous == 1) && (!enablePA) && (contextKeyParams.fromTilingSink == 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "query batch must be equal to key/value batch, query batch = %u , key/value batch = %u .", b, bKV),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((b > BLIMIT),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "batch size(%u) should not be larger than %u!", b, BLIMIT),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((b > 128 && (inputLayout == InputLayout::SH)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "batch size(%u) should not be larger than 128 when input layout is SH!", b),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((curShortSocName == platform_ascendc::SocVersion::ASCEND310P && b > 128U),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "ascend310p platform do not support batch size(%u) more than 128.", b),
                    return ge::GRAPH_FAILED);

    bool iskvdiff = (seqInnerSize != s);
    OPS_ERR_IF((iskvdiff) && (inputLayout == InputLayout::SH) && (!enablePA) && (contextKeyParams.fromTilingSink == 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "SH format not support q kv diff, length of q = %u , length of kv = %u.", s, seqInnerSize),
                    return ge::GRAPH_FAILED);

    // Dims and length of actSeqLenQ & actSeqLenKV check.
    if (!CheckActualSeqLength(contextKeyParams, b, s, seqInnerSize, tempData, tempDataKV, inputLayout, tilingData)) {
        return ge::GRAPH_FAILED;
    }
    // When verifying the shape of the mask in PA scenarios, using maxBlockNumPerBatch * tempBlockSize as tmpS2.
    if (enablePA) { 
        if (!CheckPATypeAndShape(contextKeyParams, tempDataKV, (int32_t)b, (int32_t)(*n), (int32_t)h, (int32_t)tilingData.promptAttentionBaseParams.get_headNumRatio())) {
            return ge::GRAPH_FAILED;
        }
    } else {
        tmpS2 = seqInnerSize;
    }
    tilingData.promptAttentionBaseParams.set_PAlayoutType(PAlayoutType);

    if (!CheckAttenMaskShape(contextKeyParams, sparseMode, attenMaskShape, s, tmpS2 + actualSharedPrefixLen, b)) {
        return ge::GRAPH_FAILED;
    }
    // Data types and shapes for protecting pse.
    if (contextKeyParams.pseShift != nullptr) {
        usePseShift = 1;
        if (!CheckPseShiftTypeAndShape(contextKeyParams, pseShiftShape, b, *n, s, tmpS2)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        usePseShift = 0;
    }

    // Sparse mode check.
    int32_t sparseRet = 0;
    if (sparseMode != nullptr) {
        sparseRet = (*sparseMode != SPARSE_MODE_NO_MASK && *sparseMode != SPARSE_MODE_LEFT_UP &&
                     *sparseMode != SPARSE_MODE_RIGHT_DOWN && *sparseMode != SPARSE_MODE_ALL_MASK && *sparseMode != SPARSE_MODE_BAND);
        OPS_ERR_IF((sparseRet == 1),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "sparse_mode = %d is out of range.", *sparseMode),
                    return ge::GRAPH_FAILED);

        if (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0))
            || (attenMaskShape == nullptr)) {
            tilingData.promptAttentionBaseParams.set_useMask(0); // for sparse check rule 5
        }
    }

    sparsePreTokens = static_cast<int64_t>(*preTokens);
    sparseNextTokens = static_cast<int64_t>(*nextTokens);

    uint32_t attenMaskBatch = 1U;
    bool isBandMode = false;
    bool isDefaultMode = (sparseMode == nullptr) || ((sparseMode != nullptr) && *sparseMode == SPARSE_MODE_NO_MASK);
    if (attenMaskShape != nullptr) {
        uint32_t attenMaskDim = attenMaskShape->GetStorageShape().GetDimNum();
        if (attenMaskDim != NUM_2) { // 2: target dimension of attenMask
            attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        }
        if (sparseMode != nullptr) {
            if (*sparseMode == SPARSE_MODE_LEFT_UP) {
                sparsePreTokens = SPARSE_MODE_INT_MAX;
                sparseNextTokens = 0;
                sparseModeVal = *sparseMode;
            } else if (*sparseMode == SPARSE_MODE_RIGHT_DOWN) { // Right down tokens are calculated on the kernel side.
                sparsePreTokens = SPARSE_MODE_INT_MAX;
                sparseModeVal = *sparseMode;
            } else if (*sparseMode == SPARSE_MODE_ALL_MASK) {
                sparsePreTokens = SPARSE_MODE_INT_MAX;
                sparseNextTokens = SPARSE_MODE_INT_MAX;
                sparseModeVal = *sparseMode;
            } else if (*sparseMode == SPARSE_MODE_BAND) {
                sparseModeVal = *sparseMode;
                isBandMode = true;
                OPS_ERR_IF(*preTokens < 0 && *nextTokens < 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "preTokens and nextTokens must not be negative number in band mode, preTokens = %ld , nextTokens = %ld .", *preTokens, *nextTokens),
		            return ge::GRAPH_FAILED);
            }
            OPS_LOG_I(contextKeyParams.opName, "sparseMode is %d", *sparseMode);
        }
    }
    if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN ||
        *sparseMode == SPARSE_MODE_ALL_MASK || *sparseMode == SPARSE_MODE_BAND)) {
        sparseRet = (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0))
                    || (attenMaskShape == nullptr));

        OPS_ERR_IF((sparseRet == 1),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "attenMask should not be null when sparse_mode is %d.", *sparseMode),
                    return ge::GRAPH_FAILED);

        auto maskDataType = contextKeyParams.maskDataType;
        // When sparse=2, 3, 4, the mask type only supports bool, int8, uint8
        OPS_ERR_IF((*sparseMode != SPARSE_MODE_ALL_MASK) && (maskDataType != ge::DT_BOOL) &&
                       (maskDataType != ge::DT_INT8) && (maskDataType != ge::DT_UINT8),
                   OPS_REPORT_VECTOR_INNER_ERR(
                       contextKeyParams.opName,
                       "invalid maskType dtype[%s], maskType should be bool, int8 or uint8 when sparse mode is %d.",
                       g_strDataTypePfa.at(ValidPfaDataType(maskDataType)).c_str(), *sparseMode),
                   return ge::GRAPH_FAILED);
    }
    if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_NO_MASK)) {
        // sparse mode, We need to apply the same processing to two scenarios where the attention mask is empty tensor
        if (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0))
            || (attenMaskShape == nullptr)) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = SPARSE_MODE_INT_MAX;
            sparseModeVal = *sparseMode;
        }
    }

    if (isDefaultMode && ((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr))) {
        // For scenes with sparse mode=0 and left padding, the attention mask part is fully calculated
        sparsePreTokens = SPARSE_MODE_INT_MAX;
        sparseNextTokens = SPARSE_MODE_INT_MAX;
    }
    OPS_ERR_IF((sparsePreTokens < 0) && (sparseNextTokens < 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "preTokens and nextokens cannot neither be negative number, preTokens = %ld, nextTokens = %ld.", sparsePreTokens, sparseNextTokens),
		            return ge::GRAPH_FAILED);

    OPS_ERR_IF((sparseNextTokens * (-1)) > sparsePreTokens,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "nexttoken line should be higher than pretoken line."),
		            return ge::GRAPH_FAILED);

    OPS_ERR_IF(isDefaultMode && (sparseNextTokens < 0) && (sparseNextTokens * (-1)) >= (int32_t)s,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "nextTokens absolute value should be smaller than length of q, nextTokens = %ld, length of q = %u.", sparseNextTokens, s),
                    return ge::GRAPH_FAILED);

    OPS_ERR_IF(isDefaultMode && (sparsePreTokens < 0) && (sparsePreTokens * (-1) >= ((int32_t)tmpS2 + (int32_t)actualSharedPrefixLen)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "preToken absolute value should be smaller than length of k and v (length of k and v + length of prefix when enable prefix), "
                    "preTokens = %ld, seqLengthKV = %u, actualSharedPrefixLen = %u", sparsePreTokens, tmpS2, actualSharedPrefixLen),
                    return ge::GRAPH_FAILED);

    if (sparsePreTokens > SPARSE_MODE_INT_MAX) {
        sparsePreTokens = static_cast<int32_t>(SPARSE_MODE_INT_MAX);
    }
    if (sparseNextTokens > SPARSE_MODE_INT_MAX) {
        sparseNextTokens = static_cast<int32_t>(SPARSE_MODE_INT_MAX);
    }
    size_t lenDims = b; // The current length of the actSeqLen array is equal to batch size b.
    uint32_t isLayoutSH = (inputLayout == InputLayout::SH) ? 1U : 0U;

    std::vector<int64_t> actualSeqLengths(lenDims);
    int64_t middleActualSeqLengths = 0;
    std::vector<int64_t> actualSeqLengthsKV(lenDims);

    OPS_ERR_IF(((*n <= 0) || (*n > static_cast<int32_t>(h))),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "num heads is error."),
                    return ge::GRAPH_FAILED);
    uint32_t needInit = 0U;
    int64_t preTokensPerbatch = 0;
    int64_t nextTokensPerbatch = 0;
    bool checkQuantValue = (outputType == ge::DT_INT8) &&
                           (quantOffset2Shape != nullptr) &&
                           (quantOffset2Shape->GetStorageShape().GetShapeSize() != 0);

    OPS_ERR_IF((outputType == ge::DT_INT8 && isBandMode && ((sparsePreTokens < 0) || sparseNextTokens < 0)),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "When output type is int8, sparse mode = 4, preTokens (%ld) or nextTokens (%ld) cannot be negative.",  sparsePreTokens, sparseNextTokens),
                        return ge::GRAPH_FAILED);
    if (contextKeyParams.fromTilingSink == 0) {
        for (size_t i = LOOP_BEGIN_NUM; i < lenDims; i++) {
            if ((actualLenDims == 0) || (tempData == nullptr) || (tempData->GetData<int64_t>() == nullptr)) {
                actualSeqLengths[i] = s;
                middleActualSeqLengths += actualSeqLengths[i];
            } else {
                actualSeqLengths[i] = (actualLenDims > 1) ? static_cast<uint32_t>(tempData->GetData<int64_t>()[i]) :
                                      static_cast<uint32_t>(tempData->GetData<int64_t>()[0]);
                if (actualSeqLengths[i] != s) {
                    needInit = 1;
                    OPS_ERR_IF(isDefaultMode && sparseNextTokens < 0 && sparseNextTokens * (-1) >= (int32_t)actualSeqLengths[i],
                                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                                    "nexttoken absolute value should be smaller than actual length of q, "
                                    "nextTokens = %ld, actualSeqLengthsQ = %ld", sparseNextTokens, actualSeqLengths[i]),
                                    return ge::GRAPH_FAILED);
                }
                middleActualSeqLengths += actualSeqLengths[i];
            }
            if ((actualLenDimsKV == 0) || (tempDataKV == nullptr) || (tempDataKV->GetData<int64_t>() == nullptr)) {       // The user did not input act_seq_kv
                if (contextKeyParams.isKvContinuous == 1){
                    actualSeqLengthsKV[i] = tmpS2;
                } else {
                    if ((inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::BSH)) {
                        actualSeqLengthsKV[i] = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1);
                    } else {
                        actualSeqLengthsKV[i] = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2);   // 2: Obtain the second dimension
                    }
                }
            } else {
                actualSeqLengthsKV[i] = (actualLenDimsKV > 1) ? static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[i]) :
                                        static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[0]);
                if (actualSeqLengthsKV[i] != tmpS2) {
                    needInit = 1;
                }
            }
            OPS_ERR_IF(isDefaultMode && sparsePreTokens < 0 && \
                        (sparsePreTokens * (-1) >= (actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen)),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "preToken absolute value should be smaller than actual length of k and v "
                        "(actual length of k and v + length of prefix when enable prefix), preToken = %ld, actual length of k and v = %ld, actual prefix len = %u.",
                        sparsePreTokens, actualSeqLengthsKV[i], actualSharedPrefixLen),
                        return ge::GRAPH_FAILED);
            if (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) {
                preTokensPerbatch = SPARSE_MODE_INT_MAX;
                nextTokensPerbatch = actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen - actualSeqLengths[i];
            } else if (sparseModeVal == SPARSE_MODE_BAND) {
                preTokensPerbatch = sparsePreTokens - actualSeqLengthsKV[i] - (int64_t)actualSharedPrefixLen + actualSeqLengths[i];
                nextTokensPerbatch = sparseNextTokens + actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen - actualSeqLengths[i];
            } else {
                preTokensPerbatch = sparsePreTokens;
                nextTokensPerbatch = sparseNextTokens;
            }
            if ((nextTokensPerbatch < 0) ||
                (actualSeqLengths[i] > (actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen + preTokensPerbatch))) {
                needInit = 1;
            }
            // If (preTokensPerbatch + actualSeqLengthsKV[i] + actualSharedPrefixLen - actualSeqLengths[i]) < 0 or nextTokensPerbatch < 0,
            // the last few lines or the first few lines of the QKt matrix are not computed.
            OPS_ERR_IF((checkQuantValue && \
                ((preTokensPerbatch + actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen - actualSeqLengths[i] < 0) || (nextTokensPerbatch < 0))),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When sparse mode = %d, output dtype is int8, quantOffset2 is not null or empty tensor, "
                "preTokens = %ld and nextTokens = %ld, some rows of the matrix do not participate in the calculation, "
                "the accuracy of the final result will be incorrect. Please see the documentation for more details.",
                sparseModeVal, *preTokens, *nextTokens),
                return ge::GRAPH_FAILED);
            OPS_LOG_I(contextKeyParams.opName, "preTokensPerbatch[%lu] is %ld, nextTokensPerbatch[%lu] is %ld",
                    i, preTokensPerbatch, i, nextTokensPerbatch);
            if (!isBandMode && actualSeqLengths[i] > actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen + (int64_t)sparsePreTokens) {
                actualSeqLengths[i] = actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen + (int64_t)sparsePreTokens;
            }

            OPS_ERR_IF((isBandMode && (*nextTokens < 0) && (*nextTokens * (-1) >= actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen)),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "nextTokens absolute value should be smaller than actual length of k and v in band mode (actual length of k and v + length of "
                        "prefix when enable prefix), nextTokens = %ld, actual length of k and v = %ld, prefix length = %u",
                        *nextTokens, actualSeqLengthsKV[i], actualSharedPrefixLen),
                        return ge::GRAPH_FAILED);

            OPS_ERR_IF((isBandMode && (*preTokens < 0) && (*preTokens * (-1) >= actualSeqLengths[i])),
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                            "preTokens absolute value should be smaller than actual length of q in band mode, preTokens = %ld, actual length of q = %ld", *preTokens, actualSeqLengths[i]),
                            return ge::GRAPH_FAILED);

            if(isBandMode && actualSeqLengths[i] > actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen + preTokensPerbatch){
                actualSeqLengths[i] = actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen + preTokensPerbatch;
            }

            OPS_LOG_I(contextKeyParams.opName, "actualSeqLengths[%lu] is %ld, actualSeqLengthsKV[%lu] is %ld, actualSharedPrefixLen is %u, needInit is %u",
                    i, actualSeqLengths[i], i, actualSeqLengthsKV[i], actualSharedPrefixLen, needInit);
        }
    }
    uint32_t hDivN = h / *n; // dims: d = h / n
    // Intercepting high-precision mode does not support shape currently.
    const uint32_t precisionBlockEleCut = BYTE_BLOCK / FLOAT16SIZE; // High-precision currently only supports FP16, aligned at 32/2=16.
    OPS_ERR_IF((hDivN > DLIMIT),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "d should <= 512, but d = %u. When layout is BNSD, "
                    "d is query shape in dim 3, and layout is BSH, d = h / n", hDivN),
                    return ge::GRAPH_FAILED); // Both high-precision and high-performance d cannot exceed 512.
    if ((s > SLIMIT) || (tmpS2 > SLIMIT)) {
        OPS_LOG_W(contextKeyParams.opName,
                   "seq should <= 20M, qs = %u, kvs = %u", s, tmpS2);
    }
    OPS_ERR_IF(((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION) &&
                    (inputLayout == InputLayout::SH)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "do not support SH input format when high precision!"),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION) &&
                    (hDivN % precisionBlockEleCut) != 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "d should be align when high precision, d = %u", hDivN),
                    return ge::GRAPH_FAILED); // d will be padded here and the original value cannot be obtained, so it will not be printed
    if ((inputType == ge::DT_FLOAT16) && (outputType == ge::DT_INT8)) {
        OPS_ERR_IF((inputLayout == InputLayout::SH),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "When input dtype is fp16 and output dtype is int8, SH layout is not supported."),
                        return ge::GRAPH_FAILED);
        OPS_ERR_IF((deqScale1Shape != nullptr) || (quantScale1Shape != nullptr) || (deqScale2Shape != nullptr),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "When input dtype is fp16 and output dtype is int8, PFA inputs "
                        "dequantScale1, quantScale1 and dequantScale2 should be null."),
                        return ge::GRAPH_FAILED);
    }

    // Rear Quant parameter check.
    OPS_ERR_IF(CheckPostQuantParams(contextKeyParams, h, *n) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "post quant params check failed!"),
                    return ge::GRAPH_FAILED);

    // Perchannel judgment to be adapted, maintain the existing logic firstly.
    tilingData.promptAttentionBaseParams.set_isQuant2Perchannel(0);
    tilingData.promptAttentionBaseParams.set_isQuant2BF16(0);
    tilingData.promptAttentionBaseParams.set_isQuant2FP16(0);
    if (outputType == ge::DT_INT8) {
        if (quantScale2Shape->GetStorageShape().GetShapeSize() > 1) {
            tilingData.promptAttentionBaseParams.set_isQuant2Perchannel(1);
        }
        if (contextKeyParams.quantScale2Type == ge::DT_BF16) {
            tilingData.promptAttentionBaseParams.set_isQuant2BF16(1);
        }
        if (contextKeyParams.quantScale2Type == ge::DT_FLOAT16 && contextKeyParams.hasKeyAntiquantScale && contextKeyParams.hasValueAntiquantScale){
            tilingData.promptAttentionBaseParams.set_isQuant2FP16(1);
        }
    }

    if ((curShortSocName == platform_ascendc::SocVersion::ASCEND310P )&& softmaxDataTypeNZ_ == FLOAT16SIZE) {
        sparseModeVal = 99; // 99: 310p temporarily uses the sparse field to indicate whether to adopt an approximate calculation scheme
    }
    tilingData.promptAttentionBaseParams.set_dimNumOfseq(lenDims);
    tilingData.promptAttentionBaseParams.set_scaleValue(*scaleValue);
    tilingData.promptAttentionBaseParams.set_headSize(hDivN);
    if (enablePA) {
        tilingData.promptAttentionBaseParams.set_blockSize(*blockSize);
    } else {
        tilingData.promptAttentionBaseParams.set_blockSize(BLOCK_SIZE_BASE);
    }
    tilingData.promptAttentionBaseParams.set_blockTableDim2(blockTableDim2);
    tilingData.promptAttentionBaseParams.set_PABlockNumSum(PABlockNumSum);
    tilingData.promptAttentionBaseParams.set_seqInnerSize(tmpS2);
    tilingData.promptAttentionBaseParams.set_seqSize(s);
    tilingData.promptAttentionBaseParams.set_headNumSize(*n);
    tilingData.promptAttentionBaseParams.set_batchSize(lenDims);

    tilingData.promptAttentionBaseParams.set_preTokens(sparsePreTokens);
    tilingData.promptAttentionBaseParams.set_nextTokens(sparseNextTokens);
    tilingData.promptAttentionBaseParams.set_sparseMode(static_cast<uint32_t>(sparseModeVal));
    tilingData.promptAttentionBaseParams.set_isLayoutSH(isLayoutSH);
    tilingData.promptAttentionBaseParams.set_isActualSeqLengthsNull(isActualSeqLengthsNull);
    tilingData.promptAttentionBaseParams.set_isActualSeqLengthsKVNull(isActualSeqLengthsKVNull);
    tilingData.promptAttentionSingleCoreParams.set_attenMaskBatch(attenMaskBatch);
    tilingData.promptAttentionInitOutputParams.set_needInit(needInit);

    uint32_t originHeadSize = tilingData.promptAttentionBaseParams.get_headSize();
    uint32_t blockElementCnt = BYTE_BLOCK / dataTypeSize;
    if (originHeadSize % blockElementCnt != 0) { // Determine if D is aligned with 32B, using fp16 type with 16 elements.
        tilingData.promptAttentionBaseParams.set_alignedHeadSize(((
            originHeadSize + blockElementCnt - 1) / blockElementCnt) * blockElementCnt);
        isDNoTail = false;
    } else {
        tilingData.promptAttentionBaseParams.set_alignedHeadSize(originHeadSize);
    }

    // Check the kv antiquant parameters and the shapes of scale and offset.
    uint32_t nKV = *n / tilingData.promptAttentionBaseParams.get_headNumRatio();
    uint32_t hKV = h / tilingData.promptAttentionBaseParams.get_headNumRatio();
    if (enableKvAntiquant && !CheckAntiquantParamsShape(contextKeyParams, antiquantScaleShape, antiquantOffsetShape, nKV, hDivN, hKV, tilingData)) {
        return ge::GRAPH_FAILED;
    }

    // Determine whether to enter new tiling.
    bool useNewTiling = true;
    bool useBalanceTiling = true;
    bool noInputActualSeqKV = contextKeyParams.fromTilingSink == 0 ? ((actualLenDimsKV == 0) || (tempDataKV == nullptr) || (tempDataKV->GetData<int64_t>() == nullptr)) : true;
    if ((inputLayout != InputLayout::BNSD) && (inputLayout != InputLayout::NSD)
        && (tilingData.promptAttentionBaseParams.get_headNumRatio() == 1)
        && (lenDims == 1)
        && (!iskvdiff)
        && ((*n % coreNum == 0) && (tmpS2 < CVDIFF_S2_THRESHOLDS))
        && noInputActualSeqKV) {
        useNewTiling = false;
    }
    if (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) || (enablePA)) {
        useNewTiling = true; // High-precision mode does not follow the old template.
    }

    // Only applicable to scenarios where bs=1 currently, awaiting optimization.
    if ((needInit == 1) || (lenDims != 1)) {
        useBalanceTiling = false;
    }
    if (tilingData.promptAttentionBaseParams.get_headNumRatio() != 1) {
        useBalanceTiling = false;
    }
    OPS_LOG_I(contextKeyParams.opName,
              "Tiling Info: b is %u, bKV is %u, n is %d, numKeyValueHeads is %d, s1 is %u, s2 is %u, h is %u, d is %u, headNumRatio = %u",
              b, bKV, *n, *numKeyValueHeads, s, tmpS2, h, hDivN, tilingData.promptAttentionBaseParams.get_headNumRatio());
    OPS_LOG_I(contextKeyParams.opName,
              "inputLayout is %d, innerPrecise is %lu, "
              "scaleValue is %f, preTokens is %ld, nextTokens is %ld",
              static_cast<int>(inputLayout), innerPrecise, *scaleValue, *preTokens, *nextTokens);
    // Infering whether the tiling mode is D-axis split, S2 full load, CV diff, and whether to use the matmul norm template.
    InferTilingMod(contextKeyParams, actualSeqLengths, actualSeqLengthsKV, lenDims, hDivN, tmpS2, sparseModeVal);

    if (enableMsd) {
        if (s > CVDIFF_SMALL_QS_THRESHOLDS) {
            OPS_LOG_E("PromptFlashAttention", "S of query(%u) is larger than 16, when keyAntiquantScale or valueAntiquantScale is enabled.", s);
            return ge::GRAPH_FAILED;
        }
    }
    
    uint32_t sOuterFactor;
    uint32_t sInnerFactor;
    uint32_t softmaxSInnerFactor;
    uint32_t softmaxSOuterFactor;

    // Add TND Template there
    if(inputLayout == InputLayout::TND) {
        // sparseMode Check
        OPS_ERR_IF((sparseModeVal != SPARSE_MODE_NO_MASK && sparseModeVal != SPARSE_MODE_RIGHT_DOWN),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When Layout is TND, sparseMode only support 0 or 3, but sparseMode = %d",
            sparseModeVal),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF((sparseModeVal == SPARSE_MODE_NO_MASK && attenMaskShape != nullptr),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When Layout is TND, sparseMode = 0, not support attentionMask"),
            return ge::GRAPH_FAILED);

        // TND流程
        alignedS1 = AlignUp((uint32_t)s1Size, 16u);
        alignedS2 = AlignUp((uint32_t)s2Size, 16u);
        alignedD = AlignUp((uint32_t)dSize, 16u);
        MatchTemplate();

        auto &inputParams = mlaTilingData.PFAinputParams;
        inputParams.set_bSize(bSize);
        inputParams.set_n2Size(n2Size);
        inputParams.set_gSize(1);
        inputParams.set_s1Size(realT1Size);
        inputParams.set_s2Size(s2Size);
        inputParams.set_dSize(dSize);
        inputParams.set_valueDSize(valueD);
        inputParams.set_scaleValue(*scaleValue);
        inputParams.set_alignedS2(alignedS2);
        inputParams.set_layoutType(4);
        inputParams.set_qStartIdx(0);
        inputParams.set_kvStartIdx(0);
        inputParams.set_keepProb(1.0);
        if (fromPFA_) {
            inputParams.set_fromFused(0);
        } else {
            inputParams.set_fromFused(1);
        }
        inputParams.set_isSoftMaxLseEnable(contextKeyParams.isSoftMaxLseEnable);
        SparseTypeEnum sparseType = SparseTypeEnum::ALL;
        if (sparseModeVal == SPARSE_MODE_ALL_MASK) {
            if (sparsePreTokens < s1Size - 1 || sparseNextTokens < s2Size - 1) {
                OPS_LOG_W(contextKeyParams.opName,
                            "sparsePreTokens[%ld] and sparseNextTokens[%ld] not match sparseModeVal[%d], "
                            "sparsePreTokens and sparseNextTokens will be reset max int value.",
                            sparsePreTokens, sparseNextTokens, sparseModeVal);
                sparsePreTokens = std::numeric_limits<int32_t>::max();
                sparseNextTokens = std::numeric_limits<int32_t>::max();
            }
            sparseType = static_cast<SparseTypeEnum>(static_cast<uint8_t>(sparseModeVal));
        } else if (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) {
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                    OPS_LOG_E(contextKeyParams.opName, "When layout is TND and sparse = 3 ,Batch[%ld], need s1[%ld] <= s2[%ld].", i,
                                actualSeqLenData[i], actualSeqLenKvData[i]);
                    return ge::GRAPH_FAILED;
                }
            }
            sparsePreTokens = s2Size;
            sparseNextTokens = 0;
            sparseType = SparseTypeEnum::RIGHT_DOWN_CAUSAL;
        }
        inputParams.set_attenMaskDataType(1);
        inputParams.set_attenMaskShapeType(2);

        uint8_t attenMaskCompressMode = NO_COMPRESS_MODE;
        if (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) {
            attenMaskCompressMode = RIGHT_DOWN_CAUSAL_MODE;
        }
        inputParams.set_attenMaskCompressMode(attenMaskCompressMode);
        inputParams.set_attenMaskS2Size(2048);
        inputParams.set_sparseType(static_cast<uint8_t>(sparseType));
        inputParams.set_preTokens(sparsePreTokens);
        inputParams.set_nextTokens(sparseNextTokens);

        auto &coreParams = mlaTilingData.PFAcoreParams;
        coreParams.set_s1BaseSize(s1BasicBlock);
        coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock));
        coreParams.set_s1OuterSize(CeilDivision(s1Size, s1BasicBlock));
        coreParams.set_s2BaseSize(s2BasicBlock);
        coreParams.set_s2BaseTailSize(CalcTailSize(s2Size, s2BasicBlock));
        coreParams.set_s2OuterSize(CeilDivision(s2Size, s2BasicBlock));
        nRatio = std::min(nRatio, coreParams.get_s2OuterSize());
        coreParams.set_s2OuterSize(CeilDivision(coreParams.get_s2OuterSize(), nRatio));
        coreParams.set_nRatio(nRatio);

        coreParams.set_dBaseSize(dBasicBlock);
        coreParams.set_dBaseTailSize(CalcTailSize(static_cast<int64_t>(hDivN), dBasicBlock));
        coreParams.set_dOuterSize(CeilDivision(static_cast<int64_t>(hDivN), dBasicBlock));
        // 向下取整保证数据量不超32K
        int64_t s1Vec2BaseSize = 8 * 1024 * 2 / (alignedD * dataTypeSize);
        coreParams.set_s1Vec2BaseSize(std::min(s1Vec2BaseSize, S1_VEC2_BASE_SIZE_MAX));
        coreParams.set_s1Vec2BaseTailSize(s1Size % coreParams.get_s1Vec2BaseSize());
        coreParams.set_bBaseSize(1);
        coreParams.set_bBaseTailSize(1);
        coreParams.set_bOuterSize(bSize);

        coreParams.set_n2BaseSize(1);
        coreParams.set_n2BaseTailSize(1);
        coreParams.set_n2OuterSize(n2Size);

        coreParams.set_gBaseSize(1);
        coreParams.set_gBaseTailSize(1);
        coreParams.set_gOuterSize(1); // special
        
        s1SparseValidSize = sparsePreTokens;
        s2SparseValidSize = sparseNextTokens;

        SetMultiCoreParamsTND();
        SetTensorSizeParams();
        SetSparseParamsTND();

        if (!SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock, batchBasic)) {
            return ge::GRAPH_FAILED; 
        }
        SetSoftMaxTiling();

        auto transposeSrcShape = ge::Shape({coreParams.get_bBaseSize(), 1, std::min(s1BasicBlock, alignedS1),
                                            coreParams.get_gBaseSize() * std::min(dBasicBlock, alignedD)});
        auto transposeDstShape = ge::Shape({b, *n, s, *n * hDivN});
        GetDataCopyTransposeTiling(transposeDstShape, transposeSrcShape, dataTypeSize, mlaTilingData.transposeTilingData);

        auto &tensorSizeParams = mlaTilingData.PFAtensorSizeParams;
        size_t *workspaces = contextKeyParams.workspaceSize;
        int64_t bmm1Bytes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * softmaxDataTypeSize;
        // UB不常驻，stage1占用3倍的空间，stage2占用4倍空间
        workspaces[0] = static_cast<size_t>((bmm1Bytes * 3 +
                                             4 * coreParams.get_s1BaseSize() * alignedD * softmaxDataTypeSize) *
                                             aicNum) + WORK_SPACE_RESERVE_SIZE;
        mlaRunFlag_ = true;
        if (attenMaskShape == nullptr) {
            tilingKey = 3000000000000000001;
        } else {
            tilingKey = 3000000000000000000;
        }
        blockDimToBeSet = CalcTschBlockDim(mlaTilingData.PFAmultiCoreParams.get_coreNum(), aicNum, aivNum);
        return ge::GRAPH_SUCCESS;
    }

    // Currently, there will be no D splitting scenario, and split D = 0 is default when splitting.
    if (tilingMod == TilingMod::CVSAME) {
        OPS_ERR_IF(lenDims > 128,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "when D axis size(%u) is unaligend with 32 bytes, batch size(%zu) can not larger than 128.", hDivN, lenDims),
                        return ge::GRAPH_FAILED);
        auto ret = AdjustCVTiling(hDivN, *n, middleActualSeqLengths, ubSize, l1Size, l0CSize, maskElemSize,
                                    sOuterFactor, sInnerFactor, tilingData);
        OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "adjust tiling fail"),
                        return ret);
        softmaxSOuterFactor = sOuterFactor;
        softmaxSInnerFactor = sInnerFactor;
    } else {
        auto ret = AdjustCVTilingCVDiff(ubSize, l1Size, l0CSize, maskElemSize, sOuterFactor,
                                        sInnerFactor, softmaxSOuterFactor, tilingData);
        OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "adjust tiling cv diff fail"),
                        return ret);
        softmaxSInnerFactor = sInnerFactor;
    }

    uint32_t isKvContinuous = contextKeyParams.isKvContinuous;
    uint32_t fromFused = contextKeyParams.fromFused;
    tilingData.promptAttentionSingleCoreParams.set_singleProcessSOuterSize(sOuterFactor);
    tilingData.promptAttentionSingleCoreParams.set_singleProcessSInnerSize(sInnerFactor);
    tilingData.promptAttentionBaseParams.set_splitS2(splitS2);
    tilingData.promptAttentionBaseParams.set_splitD(splitD);
    tilingData.promptAttentionBaseParams.set_softmaxOuterSize(softmaxSOuterFactor);
    tilingData.promptAttentionBaseParams.set_usePseShift(usePseShift);
    tilingData.promptAttentionBaseParams.set_pseShiftTypeByteNum(pseShiftTypeByteNum);
    tilingData.promptAttentionBaseParams.set_pseMaskMaxSize(pseMaskMaxSize);
    tilingData.promptAttentionSingleCoreParams.set_pseShiftBatch(pseShiftBatch);
    tilingData.promptAttentionBaseParams.set_pseShiftS1Size(pseShiftS1);
    tilingData.promptAttentionBaseParams.set_pseShiftS2Size(pseShiftS2);
    tilingData.promptAttentionBaseParams.set_isKvContinuous(isKvContinuous);
    tilingData.promptAttentionBaseParams.set_isQHasLeftPadding(contextKeyParams.queryPaddingSize != nullptr ? 1 : 0);
    tilingData.promptAttentionBaseParams.set_isKVHasLeftPadding(contextKeyParams.kvPaddingSize != nullptr ? 1 : 0);
    tilingData.promptAttentionBaseParams.set_fromFused((fromFused == FROM_FUSED_FLAG) ? 1 : 0);
    tilingData.promptAttentionBaseParams.set_isBSNDOut(contextKeyParams.isBSNDOut);
    tilingData.promptAttentionBaseParams.set_isSoftMaxLseEnable(contextKeyParams.isSoftMaxLseEnable);

    // Compute tiling data.
    if (splitCoreMode == SplitCoreMode::SPLIT_ONEN_CUBE) {  // Enable N split kernel mode from the perspective of cube in long sequence scenes.
        tilingData.promptAttentionInitOutputParams.set_isOneN(1);
        PromptFlashAttentionSplitSeqOneN(tilingData, coreNum, false);
    } else {
        tilingData.promptAttentionInitOutputParams.set_isOneN(0);
        if (useNewTiling) {
            PromptFlashAttentionSplitNSNew(contextKeyParams, tilingData, coreNum, actualSeqLengths, actualSeqLengthsKV, actualSharedPrefixLen, useBalanceTiling);
        } else {
            PromptFlashAttentionSplitNS(contextKeyParams, tilingData, coreNum, actualSeqLengths);
        }
    }

    if (needInit == 1) {
        PromptFlashAttentionInitOutputSplit(outShape->GetStorageShape().GetShapeSize(), tilingData, coreNum);
    }

    if (contextKeyParams.isSoftMaxLseEnable) {
        PromptFlashAttentionInitSoftmaxLseOutputSplit(SoftmaxLseOutShape->GetStorageShape().GetShapeSize(), tilingData);
    }

    if (enableMsd) {
        int64_t keyAntiquantModeMsd = 0;
        int64_t valueAntiquantModeMsd = 0;
        if (contextKeyParams.keyAntiquantMode != nullptr) {
            keyAntiquantModeMsd = *contextKeyParams.keyAntiquantMode;
        }
        if (contextKeyParams.valueAntiquantMode != nullptr) {
            valueAntiquantModeMsd = *contextKeyParams.valueAntiquantMode;
        }
        bool isLeftPadding = ((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr));
        OPS_ERR_IF(((keyAntiquantModeMsd != 0 && keyAntiquantModeMsd != 1 && keyAntiquantModeMsd == valueAntiquantModeMsd) || keyAntiquantModeMsd != valueAntiquantModeMsd),
                         OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "keyAntiquantMode(%ld) or valueAntiquantMode(%ld) is not correct, keyAntiquantMode and valueAntiquantMode only support per-token and per-channel when keyAntiquantScale or valueAntiquantScale is enabled", 
                        keyAntiquantModeMsd, valueAntiquantModeMsd),
                        return ge::GRAPH_FAILED);
        OPS_ERR_IF((contextKeyParams.kDataType == ge::DT_INT4 || contextKeyParams.vDataType == ge::DT_INT4),
                         OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "int4 is not supported when keyAntiquantScale or valueAntiquantScale is enabled, date type of key = %d,  date type of value = %d", contextKeyParams.kDataType, contextKeyParams.vDataType),
                         return ge::GRAPH_FAILED);
        OPS_ERR_IF((contextKeyParams.isKvContinuous == 0),
                         OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "tensorlist is not supported when keyAntiquantScale or valueAntiquantScale is enabled"),
                         return ge::GRAPH_FAILED);
        OPS_ERR_IF((isLeftPadding == true),
                         OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "LeftPadding is not supported when keyAntiquantScale or valueAntiquantScale is enabled"),
                         return ge::GRAPH_FAILED);

        OPS_ERR_IF((contextKeyParams.inputDataType != ge::DT_BF16) && (contextKeyParams.inputDataType != ge::DT_FLOAT16),
                         OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "inputDataType is not bf16 or fp16"),
                         return ge::GRAPH_FAILED);
        OPS_ERR_IF((contextKeyParams.kDataType != contextKeyParams.vDataType),
                         OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "DataType of key(%d) not equal datatype of value(%d)", contextKeyParams.kDataType, contextKeyParams.vDataType),
                         return ge::GRAPH_FAILED);
        OPS_ERR_IF((contextKeyParams.kDataType != ge::DT_INT8) || (contextKeyParams.vDataType != ge::DT_INT8),
                         OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "DataType of key(%d) or datatype of value(%d) is not bf16", contextKeyParams.kDataType, contextKeyParams.vDataType),
                         return ge::GRAPH_FAILED);
        OPS_ERR_IF((contextKeyParams.outputDataType != ge::DT_BF16) && (contextKeyParams.outputDataType != ge::DT_INT8) && (contextKeyParams.inputDataType != ge::DT_FLOAT16),
                         OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "DataType of output(%d) is not bf16 or int8 or fp16", contextKeyParams.outputDataType),
                         return ge::GRAPH_FAILED);

        OPS_ERR_IF((contextKeyParams.KeyAntiquantScaleShape == nullptr) || (contextKeyParams.valueAntiquantScaleShape == nullptr) ,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Shape of KeyAntiquantScale or shape of valueAntiquantScaleShape is null") ,
                    return ge::GRAPH_FAILED); 
        OPS_ERR_IF((contextKeyParams.KeyAntiquantOffsetShape == nullptr) && (contextKeyParams.valueAntiquantOffsetShape != nullptr) ,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Shape of KeyAntiquantOffset is null, when shape of valueAntiquantOffsetShape is not null") ,
                    return ge::GRAPH_FAILED);        
        OPS_ERR_IF((contextKeyParams.KeyAntiquantOffsetShape != nullptr) && (contextKeyParams.valueAntiquantOffsetShape == nullptr) ,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Shape of KeyAntiquantOffset is not null, when shape of valueAntiquantOffsetShape is null") ,
                    return ge::GRAPH_FAILED); 

        if (keyAntiquantModeMsd == 1) {
            OPS_ERR_IF((contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDimNum() != NUM_2) || (contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDimNum() != NUM_2) ||
                      ((contextKeyParams.KeyAntiquantOffsetShape != nullptr) && contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDimNum() != NUM_2) 
                      || ((contextKeyParams.valueAntiquantOffsetShape != nullptr) && contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDimNum() != NUM_2),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Dimension number of KeyAntiquantScaleShape(%zu) or valueAntiquantScaleShape(%zu) or KeyAntiquantOffsetShape(%zu) or valueAntiquantOffsetShape(%zu) is not 2 in perToken mode", 
                                                contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDimNum(), contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDimNum(), 
                                                (contextKeyParams.KeyAntiquantOffsetShape != nullptr) ? contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDimNum() : 0, 
                                                (contextKeyParams.valueAntiquantOffsetShape != nullptr) ? contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDimNum() : 0),
                        return ge::GRAPH_FAILED);
 
            OPS_ERR_IF(((contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(0) != bKV) 
                        || (contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(1) != seqInnerSize)), 
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of KeyAntiquantScale(%ld, %ld)  is not same with BS(%u, %u) in perToken mode", 
                                                    contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(0), contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(1), bKV, seqInnerSize),
                        return ge::GRAPH_FAILED);
            OPS_ERR_IF(((contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(0) != bKV) 
                        || (contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(1) != seqInnerSize)), 
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of valueAntiquantScale(%ld, %ld)  is not same with BS(%u, %u) in perToken mode", 
                                                    contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(0), contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(1), bKV, seqInnerSize),
                        return ge::GRAPH_FAILED);
            OPS_ERR_IF(((contextKeyParams.KeyAntiquantOffsetShape != nullptr) && ((contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(0) != bKV) 
                        || (contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(1) != seqInnerSize))),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of KeyAntiquantOffset(%ld, %ld)  is not same with BS(%u, %u) in perToken mode", 
                                                    contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(0), contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(1), bKV, seqInnerSize),
                        return ge::GRAPH_FAILED);
            OPS_ERR_IF(((contextKeyParams.valueAntiquantOffsetShape != nullptr) && 
                        ((contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(0) != bKV) || 
                        (contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(1) != seqInnerSize))), 
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of valueAntiquantOffsetShape(%ld, %ld)  is not same with BS(%u, %u) in perToken mode", 
                                                    contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(0), contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(1), bKV, seqInnerSize),
                      return ge::GRAPH_FAILED);
        } else if (keyAntiquantModeMsd == 0) {
            if ((contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDimNum() == NUM_2) && (contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDimNum() == NUM_2)) {
                OPS_ERR_IF(((contextKeyParams.KeyAntiquantOffsetShape != nullptr) && contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDimNum() != NUM_2) 
                            || ((contextKeyParams.valueAntiquantOffsetShape != nullptr) && contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDimNum() != NUM_2), 
                           OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Dimension number of KeyAntiquantScaleShape(%zu) or valueAntiquantScaleShape(%zu) or KeyAntiquantOffsetShape(%zu) or valueAntiquantOffsetShape(%zu) is not same in perChannel mode", 
                                                      contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDimNum(), contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDimNum(), 
                                                      (contextKeyParams.KeyAntiquantOffsetShape != nullptr) ? contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDimNum() : 0, 
                                                      (contextKeyParams.valueAntiquantOffsetShape != nullptr) ? contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDimNum() : 0),
                           return ge::GRAPH_FAILED);
 
                OPS_ERR_IF(((contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(0) != nKV) || (contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(1) != hKV / nKV)), 
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of KeyAntiquantScale(%ld, %ld)  is not same with ND(%u, %u) in perChannel mode", 
                                                        contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(0), contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(1), nKV, hKV / nKV),
                            return ge::GRAPH_FAILED);
                OPS_ERR_IF(((contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(0) != nKV) || (contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(1) != hKV / nKV)), 
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of valueAntiquantScale(%ld, %ld)  is not same with ND(%u, %u) in perChannel mode", 
                                                        contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(0), contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(1), nKV, hKV / nKV),
                            return ge::GRAPH_FAILED);
                OPS_ERR_IF(((contextKeyParams.KeyAntiquantOffsetShape != nullptr) && ((contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(0) != nKV) 
                            || (contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(1) != hKV / nKV))), 
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of KeyAntiquantOffset(%ld, %ld)  is not same with ND(%u, %u) in perChannel mode", 
                                                        contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(0), contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(1), nKV, hKV / nKV),
                            return ge::GRAPH_FAILED);
                OPS_ERR_IF(((contextKeyParams.valueAntiquantOffsetShape != nullptr) && 
                            ((contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(0) != nKV) 
                            || (contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(1) != hKV / nKV))), 
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of valueAntiquantOffsetShape(%ld, %ld)  is not same with ND(%u, %u) in perChannel mode", 
                                                        contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(0), contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(1), nKV, hKV / nKV),
                            return ge::GRAPH_FAILED);                
            } else if ((contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDimNum() == INDEX_3) && (contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDimNum() == INDEX_3)) {
                OPS_ERR_IF(((contextKeyParams.KeyAntiquantOffsetShape != nullptr) && contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDimNum() != INDEX_3) 
                          || ((contextKeyParams.valueAntiquantOffsetShape != nullptr) && contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDimNum() != INDEX_3), 
                           OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Dimension number of KeyAntiquantScaleShape(%zu) or valueAntiquantScaleShape(%zu) or KeyAntiquantOffsetShape(%zu) or valueAntiquantOffsetShape(%zu) is not same in perChannel mode", 
                                                       contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDimNum(), contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDimNum(), 
                                                       (contextKeyParams.KeyAntiquantOffsetShape != nullptr) ? contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDimNum() : 0, 
                                                       (contextKeyParams.valueAntiquantOffsetShape != nullptr) ? contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDimNum() : 0),
                           return ge::GRAPH_FAILED);
 
                OPS_ERR_IF(((contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(0) != nKV) || (contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(1) != 1) 
                            || (contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(2) != hKV / nKV)), 
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of KeyAntiquantScale(%ld, %ld, %ld)  is not same with N1D(%u, 1, %u) in perChannel mode", 
                                                        contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(0), contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(1), contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(2), nKV, hKV / nKV),
                            return ge::GRAPH_FAILED);
                OPS_ERR_IF(((contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(0) != nKV) || (contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(1) != 1) 
                            || (contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(2) != hKV / nKV)), 
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of valueAntiquantScale(%ld, %ld, %ld)  is not same with N1D(%u, 1, %u) in perChannel mode", 
                                                        contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(0), contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(1), 
                                                        contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(2), nKV, hKV / nKV),
                            return ge::GRAPH_FAILED);
                OPS_ERR_IF(((contextKeyParams.KeyAntiquantOffsetShape != nullptr) && ((contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(0) != nKV) 
                            || (contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(1) != 1) || (contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(2) != hKV / nKV))), 
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of KeyAntiquantOffset(%ld, %ld, %ld)  is not same with N1D(%u, 1, %u) in perChannel mode", 
                                                        contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(0), contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(1), 
                                                        contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(2), nKV, hKV / nKV),
                            return ge::GRAPH_FAILED);
                OPS_ERR_IF(((contextKeyParams.valueAntiquantOffsetShape != nullptr) && 
                            ((contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(0) != nKV) ||
                            (contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(1) != 1) || 
                            (contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(2) != hKV / nKV))), 
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of valueAntiquantOffsetShape(%ld, %ld, %ld)  is not same with N1D(%u, 1, %u) in perChannel mode", 
                                                        contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(0), contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(1), 
                                                        contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(2), nKV, hKV / nKV),
                            return ge::GRAPH_FAILED); 
            } else if ((contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDimNum() == 1) && (contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDimNum() == 1)) {
                OPS_ERR_IF(((contextKeyParams.KeyAntiquantOffsetShape != nullptr) && contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDimNum() != 1) 
                           || ((contextKeyParams.valueAntiquantOffsetShape != nullptr) && contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDimNum() != 1), 
                           OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Dimension number of KeyAntiquantScaleShape(%zu) or valueAntiquantScaleShape(%zu) or KeyAntiquantOffsetShape(%zu) or valueAntiquantOffsetShape(%zu) is not same in perChannel mode", 
                                                       contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDimNum(), contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDimNum(), 
                                                       (contextKeyParams.KeyAntiquantOffsetShape != nullptr) ? contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDimNum() : 0, 
                                                       (contextKeyParams.valueAntiquantOffsetShape != nullptr) ? contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDimNum() : 0),
                           return ge::GRAPH_FAILED);
                
                OPS_ERR_IF((contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(0) != hKV) , 
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of KeyAntiquantScale(%ld)  is not same with H(%u) in perChannel mode", 
                                                        contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDim(0), hKV),
                            return ge::GRAPH_FAILED);
                OPS_ERR_IF(((contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(0) != hKV)), 
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of valueAntiquantScale(%ld)  is not same with H(%u) in perChannel mode", 
                                                        contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDim(0), hKV),
                            return ge::GRAPH_FAILED);
                OPS_ERR_IF(((contextKeyParams.KeyAntiquantOffsetShape != nullptr) && ((contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(0) != hKV))), 
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of KeyAntiquantOffset(%ld)  is not same with H(%u) in perChannel mode", 
                                                        contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDim(0), hKV),
                            return ge::GRAPH_FAILED);
                OPS_ERR_IF(((contextKeyParams.valueAntiquantOffsetShape != nullptr) && 
                            ((contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(0) != hKV))), 
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "shape of valueAntiquantOffsetShape(%ld)  is not same with H(%u) in perChannel mode", 
                                                        contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDim(0), hKV),
                            return ge::GRAPH_FAILED); 
            } else {
                OPS_LOG_E(contextKeyParams.opName, "Dimension number of KeyAntiquantScaleShape(%zu) or valueAntiquantScaleShape(%zu) or KeyAntiquantOffsetShape(%zu) or valueAntiquantOffsetShape(%zu) is not 1 or 2 or 3 in perChannel mode", 
                          contextKeyParams.KeyAntiquantScaleShape->GetStorageShape().GetDimNum(), contextKeyParams.valueAntiquantScaleShape->GetStorageShape().GetDimNum(), 
                          contextKeyParams.KeyAntiquantOffsetShape->GetStorageShape().GetDimNum(), contextKeyParams.valueAntiquantOffsetShape->GetStorageShape().GetDimNum());
                return ge::GRAPH_FAILED;
            }
        }
 
        OPS_ERR_IF(((keyAntiquantModeMsd == 0 && (contextKeyParams.KeyAntiquantScaleType != ge::DT_BF16 || contextKeyParams.valueAntiquantScaleType != ge::DT_BF16 
                    || (contextKeyParams.KeyAntiquantOffsetShape != nullptr && contextKeyParams.KeyAntiquantOffsetType != ge::DT_BF16) 
                    || (contextKeyParams.valueAntiquantOffsetShape != nullptr && contextKeyParams.valueAntiquantOffsetType != ge::DT_BF16)))),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "KeyAntiquantScaleType(%s) or valueAntiquantScaleType(%s) or KeyAntiquantOffsetType(%s) or valueAntiquantOffsetType(%s) is not bf16 in per-channel mode",
                                                g_strDataTypePfa.at(ValidPfaDataType(contextKeyParams.KeyAntiquantScaleType)).c_str(),
                                                g_strDataTypePfa.at(ValidPfaDataType(contextKeyParams.valueAntiquantScaleType)).c_str(),
                                                g_strDataTypePfa.at(ValidPfaDataType(contextKeyParams.KeyAntiquantOffsetType)).c_str(),
                                                g_strDataTypePfa.at(ValidPfaDataType(contextKeyParams.valueAntiquantOffsetType)).c_str()),
                    return ge::GRAPH_FAILED);
 
        OPS_ERR_IF(((keyAntiquantModeMsd == 1 && (contextKeyParams.KeyAntiquantScaleType != ge::DT_FLOAT || contextKeyParams.valueAntiquantScaleType != ge::DT_FLOAT 
                    || (contextKeyParams.KeyAntiquantOffsetShape != nullptr && contextKeyParams.KeyAntiquantOffsetType != ge::DT_FLOAT) 
                    || (contextKeyParams.valueAntiquantOffsetShape != nullptr && contextKeyParams.valueAntiquantOffsetType != ge::DT_FLOAT)))),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "KeyAntiquantScaleType(%s) or valueAntiquantScaleType(%s) or KeyAntiquantOffsetType(%s) or valueAntiquantOffsetType(%s) is not float32 in pertoken mode",
                                                g_strDataTypePfa.at(ValidPfaDataType(contextKeyParams.KeyAntiquantScaleType)).c_str(),
                                                g_strDataTypePfa.at(ValidPfaDataType(contextKeyParams.valueAntiquantScaleType)).c_str(),
                                                g_strDataTypePfa.at(ValidPfaDataType(contextKeyParams.KeyAntiquantOffsetType)).c_str(),
                                                g_strDataTypePfa.at(ValidPfaDataType(contextKeyParams.valueAntiquantOffsetType)).c_str()),
                    return ge::GRAPH_FAILED);
 
        if((contextKeyParams.KeyAntiquantOffsetShape != nullptr) && (contextKeyParams.valueAntiquantOffsetShape != nullptr)) {
            tilingData.promptAttentionBaseParams.set_hasKeyAntiquantOffset(1);
        } else {
            tilingData.promptAttentionBaseParams.set_hasKeyAntiquantOffset(0);
        }
 
        if(contextKeyParams.keyAntiquantMode != nullptr) {
            tilingData.promptAttentionBaseParams.set_keyAntiquantMode(keyAntiquantModeMsd);
        } else {
            tilingData.promptAttentionBaseParams.set_keyAntiquantMode(0);
        }
 
        if(contextKeyParams.valueAntiquantMode != nullptr) {
            tilingData.promptAttentionBaseParams.set_valueAntiquantMode(valueAntiquantModeMsd);
        } else {
            tilingData.promptAttentionBaseParams.set_valueAntiquantMode(0);
        }
 
        OPS_ERR_IF((tilingData.promptAttentionBaseParams.get_keyAntiquantMode() != tilingData.promptAttentionBaseParams.get_valueAntiquantMode()),
                         OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "keyAntiquantMode(%ld) != valueAntiquantMode(%ld) ", 
                         tilingData.promptAttentionBaseParams.get_keyAntiquantMode(), tilingData.promptAttentionBaseParams.get_valueAntiquantMode()),
                         return ge::GRAPH_FAILED);    
    }

    ge::graphStatus tilingRet = TilingGetTilingKeyAttentionAscendC(tilingKey, contextKeyParams, useNewTiling, tilingData);
    OPS_ERR_IF(tilingRet != ge::GRAPH_SUCCESS,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get tilingKey fail"),
                            return tilingRet);

    if ((splitS2 == 1) && (splitD == 1)) {
        tilingKey = DSPLIT_S2_D_TILING_KEY;
    }

    if ((splitS2 == 0) && (splitD == 1)) {
        tilingKey = DSPLIT_S2_TILING_KEY;
    }
    tilingRet = PromptFlashAttentionApiTiling(tilingData, outputDataTypeSize, sOuterFactor, softmaxSInnerFactor, softmaxSOuterFactor);
    OPS_ERR_IF(tilingRet != ge::GRAPH_SUCCESS,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get apiTiling fail"),
                            return tilingRet);

    blockDimToBeSet = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);

    size_t* workspaces = contextKeyParams.workspaceSize;
    workspaces[0] = GetPFAWorkSpaceSize(tilingData);
    OPS_LOG_I(contextKeyParams.opName, "The Tiling key is %lu", tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckIOType(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData, int32_t& outputDataTypeSize) {
    outputType = contextKeyParams.outputDataType;
    inputType = contextKeyParams.inputDataType;
    intputKeyType = contextKeyParams.kDataType;
    intputValueType = contextKeyParams.vDataType;
    std::string tempLayoutStr(contextKeyParams.layout);
    OPS_ERR_IF((tempLayoutStr == "TND") &&
        (inputType == ge::DT_INT8 && intputKeyType == ge::DT_INT8 && intputValueType == ge::DT_INT8),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When Layout is TND, not support QKV dataType is all int8!"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(tempLayoutStr == "TND" && outputType == ge::DT_INT8,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When Layout is TND, not support attention out dataType is int8!"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((tempLayoutStr == "TND") &&
        (inputType != ge::DT_INT8 && intputKeyType == ge::DT_INT8 && intputValueType == ge::DT_INT8),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When Layout is TND, not support KV Antiquant!"),
        return ge::GRAPH_FAILED);

    if (inputType == ge::DT_FLOAT16 && contextKeyParams.kDataType == ge::DT_INT8) {
        enableKvAntiquant = true;

        if (contextKeyParams.hasKeyAntiquantScale || contextKeyParams.hasValueAntiquantScale) {        
            enableKvAntiquant = false;
        }
    }

    if (contextKeyParams.hasKeyAntiquantScale || contextKeyParams.hasValueAntiquantScale) {        
        enableMsd = true;
        tilingData.promptAttentionBaseParams.set_isMsd(1);
    } else{
        enableMsd = false;
        tilingData.promptAttentionBaseParams.set_isMsd(0);
        OPS_ERR_IF(inputType == ge::DT_BF16 && contextKeyParams.kDataType == ge::DT_INT8,
                   OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "keyAntiquantScale and valueAntiquantScale should not be null, when data type of query is bf16 and data type of key/value is int8"),
                   return ge::GRAPH_FAILED);
    }

    if (inputType == ge::DT_FLOAT16) {
        dataTypeSize = FLOAT16SIZE;
    } else if (inputType == ge::DT_BF16) {
        dataTypeSize = BFLOAT16SIZE;
    } else if (inputType == ge::DT_INT8) {
        dataTypeSize = INT8SIZE;
    }
    if (outputType == ge::DT_FLOAT16) {
        outputDataTypeSize = FLOAT16SIZE;
    } else if (outputType == ge::DT_BF16) {
        outputDataTypeSize = BFLOAT16SIZE;
    } else if (outputType == ge::DT_INT8) {
        outputDataTypeSize = INT8SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckMaskType(ContextParamsForPFATiling &contextKeyParams,
                                                          PromptFlashAttentionTilingData &tilingData,
                                                          uint32_t &maskElemSize) {
    if (contextKeyParams.attentionMask != nullptr) {
        auto maskDataType = contextKeyParams.maskDataType;
        if (maskDataType == ge::DT_FLOAT16) {
            maskElemSize = FLOAT16SIZE;
        } else if (maskDataType == ge::DT_BOOL) {
            maskElemSize = BOOLSIZE;
        } else if (maskDataType ==
                   ge::DT_INT8) { // Adapt to static graph mode, bool type attentionmask is converted to int8.
            maskElemSize = INT8SIZE;
        } else if (maskDataType == ge::DT_UINT8) {
            maskElemSize = UINT8SIZE;
        }
        // FP32 mask type does not support.
        OPS_ERR_IF(maskDataType == ge::DT_FLOAT,
                   OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                                               "invalid maskType dtype[%s], maskType should not be float[%s]",
                                               g_strDataTypePfa.at(ValidPfaDataType(maskDataType)).c_str(),
                                               g_strDataTypePfa.at(ValidPfaDataType(ge::DT_FLOAT)).c_str()),
                   return ge::GRAPH_FAILED);
        // When in fp16 high-precision mode, the mask type only supports bool or int8.
        OPS_ERR_IF(((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) &&
                       (maskDataType != ge::DT_BOOL) && (maskDataType != ge::DT_INT8) && (maskDataType != ge::DT_UINT8),
                   OPS_REPORT_VECTOR_INNER_ERR(
                       contextKeyParams.opName,
                       "invalid maskType dtype[%s], maskType should be bool, int8 or uint8 when fp16 high-precision mode",
                       g_strDataTypePfa.at(ValidPfaDataType(maskDataType)).c_str()),
                   return ge::GRAPH_FAILED);
        // When bf16, the mask type only supports bool or int8.
        OPS_ERR_IF((inputType == ge::DT_BF16) && (maskDataType != ge::DT_BOOL) && (maskDataType != ge::DT_INT8) &&
                       (maskDataType != ge::DT_UINT8),
                   OPS_REPORT_VECTOR_INNER_ERR(
                       contextKeyParams.opName,
                       "invalid maskType dtype[%s], maskType should be bool, int8 or uint8 when input type is bfloat16",
                       g_strDataTypePfa.at(ValidPfaDataType(maskDataType)).c_str()),
                   return ge::GRAPH_FAILED);
        // FP16 mask type does not support invalid line correction.
        OPS_ERR_IF((maskDataType == ge::DT_FLOAT16 && tilingData.promptAttentionBaseParams.get_isRowInvalid()),
                   OPS_REPORT_VECTOR_INNER_ERR(
                       contextKeyParams.opName,
                       "invalid maskType dtype[%s], maskType should not be float16 when innerPrecise = 2 or 3",
                       g_strDataTypePfa.at(ValidPfaDataType(maskDataType)).c_str()),
                   return ge::GRAPH_FAILED);
        if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
            OPS_ERR_IF(maskDataType != ge::DT_BOOL,
                       OPS_REPORT_VECTOR_INNER_ERR(
                           contextKeyParams.opName,
                           "invalid maskType dtype[%s], maskType should be bool when socVersion is 310p",
                           g_strDataTypePfa.at(ValidPfaDataType(maskDataType)).c_str()),
                       return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::SetMaskSize(const gert::StorageShape* attenMaskShape, PromptFlashAttentionTilingData& tilingData) {
    auto maskKVsSize = 2048; // 2048 : default the last frist dim.
    auto maskQsSize = 2048; // 2048 : default the last second dim.
    if (attenMaskShape != nullptr) {
        maskKVsSize = attenMaskShape->GetStorageShape().GetDim(attenMaskShape->GetStorageShape().GetDimNum() - 1); // 1: last frist dim
        maskQsSize = attenMaskShape->GetStorageShape().GetDim(attenMaskShape->GetStorageShape().GetDimNum() - 2); // 2: last second dim
    }

    tilingData.promptAttentionBaseParams.set_maskKVsSize(maskKVsSize);
    tilingData.promptAttentionBaseParams.set_maskQsSize(maskQsSize);
}

ge::graphStatus PromptFlashAttentionTiling::CheckShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* queryShape, 
                                                       const gert::StorageShape* keyShape, const gert::StorageShape* valueShape, const gert::StorageShape* outShape, 
                                                       const gert::StorageShape* pseShiftShape, const gert::StorageShape* attenMaskShape) {
    if (CheckNonEmptyShapeExceptions(contextKeyParams, queryShape, "query")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonEmptyShapeExceptions(contextKeyParams, keyShape, "key")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonEmptyShapeExceptions(contextKeyParams, valueShape, "value")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonEmptyShapeExceptions(contextKeyParams, outShape, "out")) {
        return ge::GRAPH_FAILED;
    }
    // Optional input can be empty.
    OPS_ERR_IF((pseShiftShape != nullptr) &&
                    (pseShiftShape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "Shape size of pseShift is overflow."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((attenMaskShape != nullptr) &&
                    (attenMaskShape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "Shape size of attenMask is overflow."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((outShape->GetStorageShape().GetShapeSize() != 0) &&
                    (queryShape->GetStorageShape().GetShapeSize() == 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "query is empty tensor."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((queryShape->GetStorageShape().GetDimNum() < NUM_2) || (queryShape->GetStorageShape().GetDimNum() > 4), 
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "queryShape dim num is error, queryShape dim num = %lu", queryShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);

    OPS_ERR_IF(SetInputLayout(contextKeyParams.layout) == GRAPH_FAILED,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "invalid input layout:%s.", contextKeyParams.layout),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::InferTilingMod(const ContextParamsForPFATiling& contextKeyParams, const std::vector<int64_t>& actualSeqLengths, const std::vector<int64_t>& actualSeqLengthsKV,
                                                uint32_t actualSeqArrayLen, uint32_t hDivN, uint32_t seqInnerSize, int32_t sparseModeVal)
{
    if (hDivN > DSPLIT_THRESHOLDS_512) {   // D segmentation threshold // S1S2D splits into fp16 and int8 types
        splitD = 1;
    }

    if ((seqInnerSize <= DSPLIT_THRESHOLDS_512) && (splitD == 1)) {
        splitS2 = 0;
    }

    if ((curShortSocName != platform_ascendc::SocVersion::ASCEND310P) &&
        (splitD != 1) && (isDNoTail == true)) {
        tilingMod = TilingMod::CVDIFF;
    }

    // Determine whether to use the norm template
    if (curShortSocName != platform_ascendc::SocVersion::ASCEND310P) {
        int64_t minActualSeqLengths = INT64_MAX;
        int64_t minActualSeqLengthsKV = INT64_MAX;
        for (uint32_t i = 0; i < actualSeqArrayLen; ++i) {
            minActualSeqLengths = std::min(minActualSeqLengths, actualSeqLengths[i]);
            minActualSeqLengthsKV = std::min(minActualSeqLengthsKV, actualSeqLengthsKV[i]);
        }
        if (minActualSeqLengths >= MATMUL_NORM_MIN_SEQ && minActualSeqLengthsKV >= MATMUL_NORM_MIN_SEQ && hDivN == MATMUL_NORM_MIN_HEADSIZE &&
            inputType == ge::DT_FLOAT16 && contextKeyParams.kDataType == ge::DT_FLOAT16 &&
            contextKeyParams.maskDataType == ge::DT_BOOL && outputType == ge::DT_FLOAT16 && usePseShift == 0 &&
            inputLayout == InputLayout::BNSD && sparseModeVal == SPARSE_MODE_BAND && (!enablePA)) {     // Currently, only the matmul norm template is open for the X1 scenario
            enableMatmulNorm = true;
        }
    }
}

ge::graphStatus PromptFlashAttentionTiling::AdjustCVTiling(uint32_t hDivN, uint32_t n, int64_t middleActualSeqLengths,
                                                           int64_t ubSize, int64_t l1Size, int64_t l0CSize,
                                                           uint32_t maskElemSize, uint32_t& sOuterFactor,
                                                           uint32_t& sInnerFactor, PromptFlashAttentionTilingData& tilingData)
{
    // D is not split, S2 is fixed and cut into 128 sizes, S1 adjusts the size for splitting
    uint32_t minFactor = 128U;       // Souter
    uint32_t rectangleFactor = 128U; // Sinner
    uint32_t seqFactorThreshold = 128U;
    uint32_t dSplitFactor = hDivN;
    // 310P involves nz2nd conversion, and currently cannot arbitrarily increase the basic block size
    if (curShortSocName != platform_ascendc::SocVersion::ASCEND310P) {
        const uint32_t littleDLimit = 64;
        if ((tilingData.promptAttentionBaseParams.get_useMask() == 0) && (hDivN <= littleDLimit)) {
            // If attentionMask is not configured, it can save UB space for softmax calculation
            // In this scenario, when d is relatively small, the size of the basic block Sinner can be adjusted to 256 to improve computational performance
            rectangleFactor = 256;
        }
        // Strategy: When there are not enough sub cores, halve the initial value of the souter to a minimum of 32
        while (n * middleActualSeqLengths / seqFactorThreshold <= coreNum) {
            seqFactorThreshold = seqFactorThreshold / 2;  // div 2
            if (seqFactorThreshold <= 32) { // Minimum to 32
                break;
            }
        }
    }

    std::queue<uint32_t> rectangleQueue;
    if (GetRectangleFactor(seqFactorThreshold, rectangleQueue) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    minFactor = rectangleQueue.front();
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        minFactor = std::min(minFactor, (tilingData.promptAttentionBaseParams.get_seqSize() + 16 - 1) / 16 * 16); // Round up to an integer multiple of 16
        rectangleFactor = std::min(rectangleFactor, (tilingData.promptAttentionBaseParams.get_seqInnerSize() + 16 - 1) / 16 * 16); // Round up to an integer multiple of 16
    }

    while (true) {
        bool updateDivRect = false;
        if (PromptFlashAttentionCheckArgsLegal(tilingData, ubSize, l1Size, l0CSize,
            softmaxDataTypeSize, minFactor, rectangleFactor, updateDivRect, maskElemSize, dSplitFactor)) {
            break;
        }
        if (updateDivRect) {
            rectangleQueue.pop();
            if (rectangleQueue.size() == 0) {
                return ge::GRAPH_FAILED;
            }
            minFactor = (rectangleQueue.front());
        }
    }
    sOuterFactor = minFactor;
    sInnerFactor = rectangleFactor;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::PromptFlashAttentionCVDiffSetTensorSize(
    PromptFlashAttentionTilingData& tilingData,
    PromptAttentionSingleCoreTensorSize& tensorSize, uint32_t sOuterFactor,
    uint32_t sInnerFactor, uint32_t softmaxSOuterFactor)
{
    if (usePseShift == 0) {
        tensorSize.set_pseShiftUbSize(0);
    } else {
        tensorSize.set_pseShiftUbSize(softmaxSOuterFactor * sInnerFactor);
    }

    tensorSize.set_attenMaskUbSize(softmaxSOuterFactor * sInnerFactor);
    if(enableMsd){
        if (tilingData.promptAttentionBaseParams.get_headSize() > MSD_BIG_D) {
            tensorSize.set_mmResUbSize(COMPUTELINE_FOR_BIG_D * sInnerFactor * 2); // 2:double buffer
        } else {
            tensorSize.set_mmResUbSize(CVDIFF_SMALL_QS_THRESHOLDS * CVDIFF_MSD_BUFFER_SIZE_1024B / sizeof(int32_t)); // for msd
        }
    } else {
        tensorSize.set_mmResUbSize(tensorSize.get_attenMaskUbSize());
    }
    
    tensorSize.set_maskSize(tensorSize.get_mmResUbSize());
    
    if (enableMsd) {
        tensorSize.set_softmaxExpSize(MSD_UB_BASE_WIDTH * ONE_BLK_SIZE_PFA);
        tensorSize.set_softmaxMaxSize(MSD_UB_BASE_WIDTH * ONE_BLK_SIZE_PFA);
    } else {
        tensorSize.set_softmaxExpSize(sOuterFactor * tilingData.promptAttentionBaseParams.get_softmaxTypeByteNum());
        tensorSize.set_softmaxMaxSize(sOuterFactor * (BYTE_BLOCK / sizeof(float)));
    }

    tensorSize.set_softmaxSumSize(tensorSize.get_softmaxMaxSize());
    tensorSize.set_softmaxValueSize(sOuterFactor * sInnerFactor);
    if (enableMsd) {
        if (tilingData.promptAttentionBaseParams.get_headSize() > MSD_BIG_D) {
            tensorSize.set_bmm2ResUbSize(MAX_COMPUTELINES * tilingData.promptAttentionBaseParams.get_alignedHeadSize()); // for big d of msd
        } else {
            tensorSize.set_bmm2ResUbSize(MSD_UB_BASE_WIDTH * MSD_UB_HEGHT);
        }
    } else {
        tensorSize.set_bmm2ResUbSize(sOuterFactor * tilingData.promptAttentionBaseParams.get_alignedHeadSize());
    }
    tensorSize.set_tmpMMResBmm2PreUbSize(std::max(tensorSize.get_mmResUbSize(), tensorSize.get_bmm2ResUbSize()));
    tensorSize.set_tmpSoftmaxBmm2UbSize(SOFTMAX_BUFFER_NUM * tensorSize.get_softmaxMaxSize());

    if (tilingData.promptAttentionBaseParams.get_maskTypeByteNum() == (BYTE_BLOCK / BOOLSIZE)) {
        tensorSize.set_selectSpaceUbSize(
            GetSelectWithBytesMaskMinTmpSize(Shape({softmaxSOuterFactor, sInnerFactor}), Shape({1}), 1,
            Shape({softmaxSOuterFactor, sInnerFactor}), 1, false));
    } else {
        tensorSize.set_selectSpaceUbSize(0);
    }
    return ge::GRAPH_SUCCESS;
}

bool PromptFlashAttentionTiling::PromptFlashAttentionComputeCVDiffParams(PromptFlashAttentionTilingData& tilingData,
    int64_t ubSize, int64_t l1Size, int64_t l0CSize, uint32_t typeByteSize,
    uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t maskTypeSize, uint32_t &softmaxSOuterFactor)
{
    bool res = false;
    int32_t l1SizeRemain = l1Size;
    if (AdjustBasicBlock(tilingData, sOuterFactor) != ge::GRAPH_SUCCESS) {
            return false;
    }

    if (inputType == ge::DT_INT8) {
        res = FindOptimalTilingSouter(tilingData, sOuterFactor, sInnerFactor, softmaxSOuterFactor, ubSize, typeByteSize, maskTypeSize);
    } else {
        res = FindOptimalTilingBasicBLock(tilingData, sOuterFactor, sInnerFactor, softmaxSOuterFactor, ubSize, typeByteSize, maskTypeSize);
    }
    OPS_ERR_IF(res == false,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "FindOptimalTilingBasicBLock failed!"),
                    return false);

    // kvcache antiquant tiling
    if (enableKvAntiquant) {
        int32_t sKvAntiquantFactor = sInnerFactor;
        uint32_t kvAntiquantApiSizeMax = 0;
        uint32_t kvAntiquantApiSize = 0;
        auto srcShape = Shape({sKvAntiquantFactor, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
        auto scaleShape = Shape({1, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
        int64_t ubSizeRemainTmp = ubSizeRemain;
        do {
            srcShape = Shape({sKvAntiquantFactor, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
            GetAscendAntiQuantMaxMinTmpSize(srcShape, scaleShape, false, ge::DT_INT8, inputType, kvAntiquantApiSizeMax, kvAntiquantApiSize);
            ubSizeRemain = ubSizeRemainTmp - kvAntiquantApiSize - tilingData.promptAttentionBaseParams.get_alignedHeadSize() * 2 * FLOAT16SIZE - // scale offset fp16, 2 is used for alignment 
                (sKvAntiquantFactor * tilingData.promptAttentionBaseParams.get_alignedHeadSize() * (INT8SIZE + FLOAT16SIZE) * 1);   // Input/output
            if (ubSizeRemain < 0) {
                sKvAntiquantFactor -= 1;
            }
        } while (ubSizeRemain < 0 && sKvAntiquantFactor > 0);
        OPS_ERR_IF(sKvAntiquantFactor <= 0,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "cannot find valid sKvAntiquantFactor!"),
                        return false);
        tilingData.promptAttentionTensorSizeRect.set_kvAntiquantUbSize(sKvAntiquantFactor * tilingData.promptAttentionBaseParams.get_alignedHeadSize());
        tilingData.promptAttentionSingleCoreParams.set_kvAntiquantSInnerSize(sKvAntiquantFactor);
    }

    const uint32_t dSplitFactorBmm2 = 128U;
    SetSplitCoreMode(tilingData, sOuterFactor);
    res = PromptFlashAttentionCheckBmm1(tilingData, tilingData.bmm1TilingDataRect,
            l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, true, true);
    OPS_ERR_IF(res == false,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "PromptFlashAttentionCheckBmm1 failed!"),
                    return false);

    res = PromptFlashAttentionCheckBmm2(tilingData, tilingData.bmm2TilingDataRect,
            l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, dSplitFactorBmm2, true, true);
    OPS_ERR_IF(res == false,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "PromptFlashAttentionCheckBmm2 failed!"),
                    return false);

    return true;
}

bool PromptFlashAttentionTiling::FindOptimalTilingSouter(PromptFlashAttentionTilingData& tilingData,
    uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t &softmaxSOuterFactor,
    int64_t ubSize, uint32_t typeByteSize, uint32_t maskTypeSize)
{
    // This function has a fixed Sinner of 1024 or kvs, reducing Souter to make ub sufficient.
    // Currently, only Int8 is using it
    auto tmpShape = Shape({softmaxSOuterFactor, sInnerFactor});
    int64_t softmaxTmpSize = 0;
    int64_t softmaxFlashTmpSize = 0;
    int64_t queueBufferSize = 0;

    // Temporary solution, first calculate using the Tmp variable of type int32_t,
    // and then optimize by changing the input parameter to type int32_t
    int32_t sOuterFactorTmp = (int32_t)sOuterFactor;
    int32_t sInnerFactorTmp = (int32_t)sInnerFactor;
    int32_t softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
    const int32_t sOuterFactorStep = 16;
    const int32_t softmaxSOuterFactorStep = 8;

    int64_t pseShiftBufferSize = 0;
    pseMaskMaxSize = std::max(maskTypeSize, pseShiftElemSize);

    uint32_t pseShiftCastSize = 0U;
    if (usePseShift == 1 && (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) || pseShiftElemType == ge::DT_BF16)) {
        pseShiftCastSize = FLOAT32SIZE;   // In the case of high-precision effectiveness or bf16, pse needs to do a cast and apply for ub
    }

    uint32_t kvAntiquantApiSizeMax = 0U;
    uint32_t kvAntiquantApiSize = 0U;
    auto srcShape = Shape({1, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
    auto scaleShape = Shape({1, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
    GetAscendAntiQuantMaxMinTmpSize(srcShape, scaleShape, false, ge::DT_INT8, inputType, kvAntiquantApiSizeMax, kvAntiquantApiSize);
    // Minimum antiquant ub: api + scale offset + input/output only processes one line at a time
    int64_t minAntiquantUbSizeNeed = kvAntiquantApiSize + tilingData.promptAttentionBaseParams.get_alignedHeadSize() * 2 * FLOAT16SIZE + // scale offset fp16
                tilingData.promptAttentionBaseParams.get_alignedHeadSize() * (INT8SIZE + FLOAT16SIZE); // Input int8 and output fp16

    // lse extra ub size
    int64_t lseUbSize = contextKeyParamsPtr->isSoftMaxLseEnable ? 256 : 0;      // only the first 2 elements are valid

    ubSizeRemain = 0;
    while (ubSizeRemain <= 0 && sOuterFactorTmp > 0) {
        softmaxTmpSize = 0;
        softmaxFlashTmpSize = 0;
        while ((softmaxTmpSize == 0 || softmaxFlashTmpSize == 0) && (softmaxSOuterFactorTmp > 0)) {
            tmpShape = Shape({softmaxSOuterFactorTmp, sInnerFactorTmp});
            softmaxTmpSize = GetSoftMaxMinTmpSize(tmpShape, typeByteSize, true);
            softmaxFlashTmpSize = GetSoftMaxFlashV2MinTmpSize(tmpShape, typeByteSize, sizeof(float), true, true);
            if (softmaxTmpSize == 0 || softmaxFlashTmpSize == 0) {
                softmaxSOuterFactorTmp -= softmaxSOuterFactorStep;
            }
        }

        if (softmaxSOuterFactorTmp <= 0) {
            sOuterFactorTmp -= sOuterFactorStep;
            softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
            continue;
        }
        if (PromptFlashAttentionCVDiffSetTensorSize(tilingData, tilingData.promptAttentionTensorSizeRect,
                                                sOuterFactorTmp, sInnerFactorTmp, softmaxSOuterFactorTmp) != ge::GRAPH_SUCCESS) {
            return false;
        }

        int64_t msdUbSize = PromptFlashAttentionSetMsdUbSize(tilingData, tilingData.promptAttentionTensorSizeRect, sInnerFactorTmp);
        queueBufferSize = tilingData.promptAttentionTensorSizeRect.get_attenMaskUbSize();
        pseShiftBufferSize = tilingData.promptAttentionTensorSizeRect.get_pseShiftUbSize();
        apiTmpSize = std::max(softmaxTmpSize, softmaxFlashTmpSize);

        int64_t maskBmm2ShareSize = std::max(int64_t(queueBufferSize * pseMaskMaxSize),
            int64_t(tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() * typeByteSize));
        ubSizeRemain = ubSize - apiTmpSize - (tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() * NUM_2 + // 2:2 mm ub
                    tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() +       // bmm2ResPrev resident in UB
                    SOFTMAX_BUFFER_NUM * tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize()) *
                    typeByteSize - maskBmm2ShareSize - tilingData.promptAttentionTensorSizeRect.get_selectSpaceUbSize() -
                    pseShiftBufferSize * pseShiftCastSize - msdUbSize - lseUbSize;
        if ((ubSizeRemain <= 0) || (enableKvAntiquant && ubSizeRemain < minAntiquantUbSizeNeed)) {
            sOuterFactorTmp -= sOuterFactorStep;
            sInnerFactorTmp = (int32_t)sInnerFactor;
            softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
        }
    }

    OPS_ERR_IF((sOuterFactorTmp <= 0) || (sInnerFactorTmp <= 0) || (softmaxSOuterFactorTmp <= 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "cannot find valid sOuterFactor, sInnerFactor and softmaxSOuterFactor!"),
                    return false);
    sOuterFactor = (uint32_t)sOuterFactorTmp;
    sInnerFactor = (uint32_t)sInnerFactorTmp;
    softmaxSOuterFactor = (uint32_t)softmaxSOuterFactorTmp;
    return true;
}

bool PromptFlashAttentionTiling::FindOptimalTilingBasicBLock(PromptFlashAttentionTilingData& tilingData,
    uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t &softmaxSOuterFactor,
    int64_t ubSize, uint32_t typeByteSize, uint32_t maskTypeSize)
{
    auto tmpShape = Shape({softmaxSOuterFactor, sInnerFactor});
    int64_t softmaxTmpSize = 0;
    int64_t softmaxFlashTmpSize = 0;
    int64_t queueBufferSize = 0;

    // lse extra ub size
    int64_t lseUbSize = contextKeyParamsPtr->isSoftMaxLseEnable ? 256 : 0;      // only the first 2 elements are valid

    // Temporary solution, first calculate using the Tmp variable of type int32_t, and then optimize by changing the input parameter to type int32_t
    int32_t sOuterFactorTmp = (int32_t)sOuterFactor;
    int32_t sInnerFactorTmp = (int32_t)sInnerFactor;
    int32_t softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
    const int32_t sOuterFactorStep = 16;
    int32_t sInnerFactorStep = 64;
    const int32_t softmaxSOuterFactorStep = 8;

    int64_t pseShiftBufferSize = 0;
    pseMaskMaxSize = std::max(maskTypeSize, pseShiftElemSize);

    uint32_t pseShiftCastSize = 0U;
    if ((usePseShift == 1) && (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) || pseShiftElemType == ge::DT_BF16)) {
        pseShiftCastSize = FLOAT32SIZE;   // In the case of high-precision effectiveness or bf16, pse needs to do a cast and apply for ub
    }
    if (enablePA) {
        sInnerFactorStep = tilingData.promptAttentionBaseParams.get_blockSize();
    }
    uint32_t kvAntiquantApiSizeMax = 0U;
    uint32_t kvAntiquantApiSize = 0U;
    auto srcShape = Shape({1, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
    auto scaleShape = Shape({1, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
    GetAscendAntiQuantMaxMinTmpSize(srcShape, scaleShape, false, ge::DT_INT8, inputType, kvAntiquantApiSizeMax, kvAntiquantApiSize);
    // Minimum antiquant ub: api + scale offset + input/output only processes one line at a time
    int64_t minAntiquantUbSizeNeed = kvAntiquantApiSize + tilingData.promptAttentionBaseParams.get_alignedHeadSize() * 2 * FLOAT16SIZE + // scale offset fp16
                tilingData.promptAttentionBaseParams.get_alignedHeadSize() * (INT8SIZE + FLOAT16SIZE); // Input int8, Output fp16

    // post quant perchannel ub size
    int64_t postQuantUbSize = 0;
    if (tilingData.promptAttentionBaseParams.get_isQuant2Perchannel() == 1) {
        uint32_t floatSize = 4;
        uint32_t bf16Size = 2;
        postQuantUbSize = 2 * floatSize * tilingData.promptAttentionBaseParams.get_headSize();     // 2: scale2, offset2
        if (tilingData.promptAttentionBaseParams.get_isQuant2BF16() == 1 || tilingData.promptAttentionBaseParams.get_isQuant2FP16() == 1) {
            postQuantUbSize += 2 * bf16Size * tilingData.promptAttentionBaseParams.get_headSize(); // 2: scale2, offset2
        }
    }

    // AscendQuant reserves ub space
    auto postQuantSrcShape = Shape({sOuterFactor, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
    uint32_t bmm2ResTypeSize = (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) || (inputType == ge::DT_BF16)) ? FLOAT32SIZE : FLOAT16SIZE;
    uint32_t postQuantApiSizeMax = 0U;
    uint32_t postQuantApiSizeMin = 0U;

    ubSizeRemain = 0;
    int64_t msdUbSize =0;
    while (ubSizeRemain <= 0 && sOuterFactorTmp > 0) {
        while ((ubSizeRemain <= 0 && sInnerFactorTmp > 0) || (enableKvAntiquant && ubSizeRemain < minAntiquantUbSizeNeed && sInnerFactorTmp > 0)) {
            softmaxTmpSize = 0;
            softmaxFlashTmpSize = 0;
            while ((softmaxTmpSize == 0 || softmaxFlashTmpSize == 0) && (softmaxSOuterFactorTmp > 0)) {
                tmpShape = Shape({softmaxSOuterFactorTmp, sInnerFactorTmp});
                softmaxTmpSize = GetSoftMaxMinTmpSize(tmpShape, typeByteSize, true);
                softmaxFlashTmpSize = GetSoftMaxFlashV2MinTmpSize(tmpShape, typeByteSize, sizeof(float), true, true);
                if (softmaxTmpSize == 0 || softmaxFlashTmpSize == 0) {
                    softmaxSOuterFactorTmp -= softmaxSOuterFactorStep;
                }
            }

            if (softmaxSOuterFactorTmp <= 0) {
                sInnerFactorTmp -= sInnerFactorStep;
                softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
                continue;
            }

            if (PromptFlashAttentionCVDiffSetTensorSize(tilingData, tilingData.promptAttentionTensorSizeRect,
                                                    sOuterFactorTmp, sInnerFactorTmp, softmaxSOuterFactorTmp) != ge::GRAPH_SUCCESS) {
                return false;
            }

            msdUbSize = PromptFlashAttentionSetMsdUbSize(tilingData, tilingData.promptAttentionTensorSizeRect, sInnerFactorTmp);

            queueBufferSize = tilingData.promptAttentionTensorSizeRect.get_attenMaskUbSize();
            pseShiftBufferSize = tilingData.promptAttentionTensorSizeRect.get_pseShiftUbSize();
            apiTmpSize = std::max(softmaxTmpSize, softmaxFlashTmpSize);

            if (outputType == ge::DT_INT8) {
                postQuantSrcShape = Shape({sOuterFactorTmp, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
                GetAscendQuantMaxMinTmpSize(postQuantSrcShape, bmm2ResTypeSize, postQuantApiSizeMax, postQuantApiSizeMin);
            }

            int64_t maskBmm2ShareSize = std::max(int64_t(queueBufferSize * pseMaskMaxSize),
                int64_t(tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() * typeByteSize));
            ubSizeRemain = ubSize - apiTmpSize - (tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() * NUM_2 +  // 2:2 mm ub
                        tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() +       // bmm2ResPrev resident in UB
                        SOFTMAX_BUFFER_NUM * tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize()) *
                        typeByteSize - maskBmm2ShareSize - tilingData.promptAttentionTensorSizeRect.get_selectSpaceUbSize() -
                        pseShiftBufferSize * pseShiftCastSize - postQuantUbSize - postQuantApiSizeMin - msdUbSize - lseUbSize;
            if (ubSizeRemain <= 0 || (enableKvAntiquant && ubSizeRemain < minAntiquantUbSizeNeed)) {
                sInnerFactorTmp -= sInnerFactorStep;
                softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
            }
        }

        if ((ubSizeRemain <= 0) || (enableKvAntiquant && ubSizeRemain < minAntiquantUbSizeNeed)) {
            sOuterFactorTmp -= sOuterFactorStep;
            sInnerFactorTmp = (int32_t)sInnerFactor;
            softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
        }
    }

    OPS_ERR_IF((sOuterFactorTmp <= 0) || (sInnerFactorTmp <= 0) || (softmaxSOuterFactorTmp <= 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "cannot find valid sOuterFactor, sInnerFactor and softmaxSOuterFactor!"),
                    return false);
    sOuterFactor = (uint32_t)sOuterFactorTmp;
    sInnerFactor = (uint32_t)sInnerFactorTmp;
    softmaxSOuterFactor = (uint32_t)softmaxSOuterFactorTmp;
    return true;
}

ge::graphStatus PromptFlashAttentionTiling::AdjustCVTilingCVDiff(int64_t ubSize, int64_t l1Size, int64_t l0CSize,
    uint32_t maskElemSize, uint32_t& sOuterFactor, uint32_t& sInnerFactor, uint32_t& softmaxSOuterFactor,
    PromptFlashAttentionTilingData& tilingData)
{
    // New softmax tiling strategy, unified big tiling for mm1 mm2 (e.g. mm1=256x512, mm2=256xhead_size), softmax calculates multiple long tiling based on the UB space by horizontally cutting the big tiling into multiple long tiling (e.g. softmax=32x512).
    // Softmax calculates multiple long tiling based on the UB space by horizontally slicing big tiling (e.g. softmax=32x512).
    uint32_t minFactor = CVDIFF_SOUTER_FACTOR_DEFAULT;
    uint32_t rectangleFactor = CVDIFF_SINNER_FACTOR_DEFAULT;
    const uint32_t softmaxUbSize = CVDIFF_MM1RES_UB_SIZE;
    if ((tilingData.promptAttentionBaseParams.get_seqInnerSize() <= CVDIFF_SMALL_KV_THRESHOLDS) && (inputType != ge::DT_INT8)) {
        rectangleFactor = CVDIFF_SINNER_FACTOR_SMALL_KVS;
    }

    softmaxSOuterFactor = softmaxUbSize / rectangleFactor;

    if (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) ||
        (inputType == ge::DT_BF16)) {  // When high-precision mode or BF16 takes effect, adjust the starting tiling block.
        if (tilingData.promptAttentionBaseParams.get_alignedHeadSize() >= 200) {           // D: [200, ...)
            minFactor = 64;            // 64:  Adjust the size of the basic block Souter to 64.
            rectangleFactor = 512;     // 512: Adjust the size of the basic block Sinner to 512.
            softmaxSOuterFactor = 8;   // 8:   Adjust softmaxSOuter to 8.
        } else if (tilingData.promptAttentionBaseParams.get_alignedHeadSize() >= 128) {    // D: [128, 200)
            minFactor = 128;           // 128: Adjust the size of the basic block Souter to 128.
            rectangleFactor = 512;     // 512: Adjust the size of the basic block Sinner to 512.
            softmaxSOuterFactor = 8;   // 8:   Adjust softmaxSOuter to 8
        } else if (tilingData.promptAttentionBaseParams.get_alignedHeadSize() >= 32) {     // D: [32, 128)
            minFactor = 128;           // 128: Adjust the size of the basic block Souter to 128.
            rectangleFactor = 512;     // 512: Adjust the size of the basic block Sinner to 512.
            softmaxSOuterFactor = 16;  // 16:  Adjust softmaxSOuter to 16.
        } else {                                                                           // D: (0, 32)
            minFactor = 128;           // 128: Adjust the size of the basic block Souter to 128
            rectangleFactor = 512;     // 512: Adjust the size of the basic block Sinner to 512
            softmaxSOuterFactor = 32;  // 32:  Adjust softmaxSOuter to 32
        }
    }
    if (enablePA) {
        minFactor = 64;  // In the PA scenario, Souter starts cutting from 64 and tries to ensure that Sinner does not cut, so that Single is a multiple of blockSize
    }
    if (tilingData.promptAttentionBaseParams.get_seqSize() <= CVDIFF_SMALL_QS_THRESHOLDS) {   // Minimum basic block size.
        minFactor = CVDIFF_SMALL_QS_THRESHOLDS;  // Reduce S1 to avoid unnecessary calculation of mm1
        if ((tilingData.promptAttentionBaseParams.get_seqInnerSize() > CVDIFF_SINNER_FACTOR_SMALL_QS)
            && (tilingData.promptAttentionBaseParams.get_useMask() == 0)) {   // Only in scenes without masks can it be set to 2048.
            if (enableMsd) {
                rectangleFactor = CVDIFF_SINNER_FACTOR_DEFAULT; 
            } else {
                rectangleFactor = CVDIFF_SINNER_FACTOR_SMALL_QS;   // Increase S2 to improve softmax throughput.
            }
        }
        softmaxSOuterFactor = softmaxUbSize / rectangleFactor;

        // Reduce softmaxouter to the true souter.
        if ( tilingData.promptAttentionBaseParams.get_seqSize() < softmaxSOuterFactor) {
            softmaxSOuterFactor = tilingData.promptAttentionBaseParams.get_seqSize();
        }
    }

    if (enableKvAntiquant) {
        uint32_t sInnerMax = 1024 * 256 / tilingData.promptAttentionBaseParams.get_alignedHeadSize();   // The increase in workspace should not exceed 50M
        sInnerMax = (sInnerMax + THIRTY_ONE) / UB_ALIGN * UB_ALIGN;
        rectangleFactor = rectangleFactor > sInnerMax ? sInnerMax : rectangleFactor;
        softmaxSOuterFactor = softmaxUbSize / rectangleFactor;
    }

    bool res = PromptFlashAttentionComputeCVDiffParams(tilingData, ubSize, l1Size, l0CSize,
                    softmaxDataTypeSize, minFactor, rectangleFactor, maskElemSize, softmaxSOuterFactor);
    OPS_ERR_IF(res == false,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "PromptFlashAttentionComputeCVDiffParams failed!"),
                    return ge::GRAPH_FAILED);

    sOuterFactor = minFactor;
    sInnerFactor = rectangleFactor;

    return ge::GRAPH_SUCCESS;
}

PFA_EXTERN_C ge::graphStatus TilingPromptFlashAttention(gert::TilingContext* context) {
    if (context == nullptr) {
        OPS_LOG_E("PromptFlashAttention", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    if (context->GetRawTilingData() == nullptr) {
        OPS_LOG_E("PromptFlashAttention", "tiling context GetRawTilingData is nullptr!");
        return ge::GRAPH_FAILED;
    }
    PromptFlashAttentionTiling flashTiling(nullptr);
    PromptFlashAttentionTilingData tilingData;
    OPS_ERR_IF(memset_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(),
               0, context->GetRawTilingData()->GetCapacity()) != EOK,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "fail to memset tiling data"),
               return ge::GRAPH_FAILED);
    ContextParamsForPFATiling contextParamsForPFATiling = {
        .pseShift = nullptr,
        .attentionMask = nullptr,
        .actualSeqenceLengthQ = nullptr,
        .actualSeqenceLengthKV = nullptr,
        .antiquantScale = nullptr,
        .antiquantOffset = nullptr, // Initialize pfa context
        .queryPaddingSize = nullptr,
        .kvPaddingSize = nullptr,
        .blockTable = nullptr,
        .keySharedPrefix = nullptr,
        .valueSharedPrefix = nullptr,
        .actualSharedPrefixLen = nullptr,
        .KeyAntiquantScale = nullptr,
        .valueAntiquantScale = nullptr, // Initialize pfa context
        .KeyAntiquantOffset = nullptr,
        .valueAntiquantOffset = nullptr,
        .inputDataType = ge::DataType::DT_FLOAT16,
        .kDataType = ge::DataType::DT_FLOAT16,
        .vDataType = ge::DataType::DT_FLOAT16,
        .pseShiftDataType = ge::DataType::DT_FLOAT16,
        .maskDataType = ge::DataType::DT_FLOAT16,
        .blockTableType = ge::DataType::DT_FLOAT16, // Initialize pfa context
        .outputDataType = ge::DataType::DT_FLOAT16,
        .opName = nullptr,
        .queryInputShape = nullptr,
        .keyInputShape = nullptr,
        .valueInputShape = nullptr,
        .pseShiftShape = nullptr,
        .attentionMaskShape = nullptr,
        .deqScale1Shape = nullptr, // Initialize pfa context
        .scale1Shape = nullptr,
        .deqScale2Shape = nullptr,
        .scale2Shape = nullptr,
        .offset2Shape = nullptr,
        .antiquantScaleShape = nullptr,
        .antiquantOffsetShape = nullptr,
        .blockTableShape = nullptr,
        .outputShape = nullptr, // Initialize pfa context
        .lseoutputShape = nullptr,
        .KeyAntiquantScaleShape = nullptr,
        .valueAntiquantScaleShape = nullptr,
        .KeyAntiquantOffsetShape = nullptr,
        .valueAntiquantOffsetShape = nullptr,
        .KeyAntiquantScaleType = ge::DataType::DT_FLOAT16,
        .valueAntiquantScaleType = ge::DataType::DT_FLOAT16,
        .KeyAntiquantOffsetType = ge::DataType::DT_FLOAT16, // Initialize pfa context
        .valueAntiquantOffsetType = ge::DataType::DT_FLOAT16,
        .innerPrecisePtr = nullptr,
        .headsNumber = nullptr,
        .sparseMode = nullptr,
        .preToken = nullptr,
        .nextToken = nullptr,
        .scaleValue = nullptr,
        .blockSize = nullptr, // Initialize pfa context
        .layout = nullptr,
        .numKeyValueHeads = nullptr,
        .workspaceSize = nullptr,
        .compileInfoPtr = nullptr,
        .deqScaleType = ge::DataType::DT_FLOAT16,
        .deqScale2Type = ge::DataType::DT_FLOAT16,
        .quantScale2Type = ge::DataType::DT_FLOAT16,
        .quantOffset2Type = ge::DataType::DT_FLOAT16, // Initialize pfa context
        .isKvContinuous = 0,
        .kTensorList = {nullptr},
        .vTensorList = {nullptr},
        .maxKVs  =0,
        .fromFused = 0,
        .emptyTensor = 0,
        .isBSNDOut = 0,
        .softmaxLseFlag = nullptr, // Initialize pfa context
        .isSoftMaxLseEnable = false,
        .fromTilingSink = 0,
        .hasKeyAntiquantScale = 0,
        .hasValueAntiquantScale = 0,
        .isMsd = 0,
        .keyAntiquantMode = nullptr,
        .valueAntiquantMode = nullptr,
        .hasKeyAntiquantOffset = 0 // Initialize pfa context
    };
    auto ret = ConvertContextToPFAParams(context, contextParamsForPFATiling);
    uint64_t tilingKey = 7;  // 7: default tiling key
    uint32_t blockDimToBeSet;
    ret = flashTiling.RunBigKernelTilingWithParams(contextParamsForPFATiling, tilingKey, blockDimToBeSet, tilingData);
    tilingKey += BENCHMARK_TILING_KEY;
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDimToBeSet);
    flashTiling.PromptFlashAttentionSetTilingData(context, tilingData);
    return ret;
}
}
