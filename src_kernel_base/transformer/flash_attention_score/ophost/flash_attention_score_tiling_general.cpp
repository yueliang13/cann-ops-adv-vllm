/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_tiling_general.cpp
 * \brief
 */

#include <numeric>
#include "tiling/data_copy_transpose_tiling.h"
#include "tiling/tiling_base.h"
#include "tiling/tiling_templates_registry.h"
#include "tiling/tiling_type.h"
#include "flash_attention_score_tiling.h"
#include "flash_attention_score_tiling_common.h"

namespace optiling {
namespace FA {
const uint32_t BYTE_BLOCK = 32;
const int64_t GM_ALIGN = 512;
const int64_t FRACTAL_NUM = 16L;
const int64_t PSE_DIM_NUM = 4L;
const int64_t S1_VEC2_BASE_SIZE_MAX = 512L;

const int64_t BYTE_BIT_NUM = 8UL;
const size_t PSE_INPUT_INDEX = 3UL;
const size_t DROP_MASK_INPUT_INDEX = 4UL;
const size_t ATTENTION_MASK_INPUT_INDEX = 6UL;
const size_t PREFIX_INPUT_INDEX = 7UL;
const size_t ACTUAL_SEQ_LENGTH_INPUT_INDEX = 8UL;
const size_t ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX = 9UL;
const size_t Q_START_IDX_INPUT_INDEX = 10UL;
const size_t KV_START_IDX_INPUT_INDEX = 11UL;
const size_t ATTEN_OUT_INDEX = 3UL;
const size_t ATTENTION_MASK_DIM_NUM_4 = 4UL;
const size_t ATTENTION_MASK_DIM_NUM_2 = 2UL;
const int64_t BMM_SOFTMAX_RATIO = 4L;
const int64_t MAX_AIV_NUM = 48L;
const int64_t DROP_MASK_ALIGN_UNIT = 256L; // input bits, and align to 32B in UB
const int64_t HIGH_PERF_BUFFER_NUM = 6L;
const int64_t HIGH_PERF_SUPPORT_S2_BASIC = 128L;
const int64_t HIGH_PERF_API_BUFFER_MULTIPLE = 2L;
const int64_t HIGH_PERF_BLOCK_SIZE = 128L;
const uint32_t PSE_ALIBI_S_SIZE = 1024;

constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
const int64_t ATTEN_MASK_S1_REV_INDEX = 2L;
const int64_t ATTEN_MASK_COMPRESS_LIMIT = 2048L;
const int64_t ATTEN_MASK_COMPRESS_PREFIX_LIMIT = 3072L;
const int64_t MAX_VAR_LEN_SEQ_LEN = 4096L;
const int64_t S2_REUSE_SIZE_512 = 512L;
const int64_t S2_REUSE_SIZE_1024 = 1024L;
const int64_t S1_REUSE_SIZE_3840 = 3840L;
const int64_t D_SPECIFIC_SIZE = 64L;
const int64_t BALANCE_LOAD_LIST_SIZE = 8L;
constexpr int64_t COF[BALANCE_LOAD_LIST_SIZE] = {256, 384, 512, 640, 768, 896, 960, 1024};
const int64_t BMM1_BASICBLOCK_M_128 = 128L;
const int64_t BMM1_BASICBLOCK_N_256 = 256L;
const int64_t BMM1_BASICBLOCK_N_128 = 128L;
const int64_t BMM1_BASICBLOCK_K_64 = 64L;
const int64_t BMM1_BASICBLOCK_K_128 = 128L;
const int64_t S2_NZTOND_SIZE_64 = 64L;
const int64_t SPACE_NUM_2 = 2L;
const int64_t SPACE_NUM_3 = 3L;
const int64_t SPACE_NUM_4 = 4L;
const int64_t BMM2_BASICBLOCK_M_64 = 64L;
const int64_t BMM2_BASICBLOCK_N_64 = 64L;
const int64_t BMM2_BASICBLOCK_K_256 = 256L;
const int64_t S2_SPECIFIC_SIZE_928 = 928L;
const int64_t S2_NZTOND_SIZE_128 = 128L;
const int64_t UB_BASIC_LIMIT_SIZE = 8 * 1024;
const int64_t SLOPE_BN_DIM_NUM = 2L;
const int64_t SLOPE_N_DIM_NUM = 1L;
const int64_t L1REUSE_D_Limit = 128L;
const int64_t L1REUSE_BNG_Limit = 10L;
const int64_t L1REUSE_S2_Limit_1024 = 1024;
const int64_t L1REUSE_S2_Limit_2048 = 2048L;
const int64_t L1REUSE_S2_LIMIT_256 = 256;
const int64_t L1REUSE_S2_LIMIT_4032 = 4032;
const int64_t AICAIV_RATIO_2 = 2L;
const int64_t L1REUSE_S2_LIMIT_512 = 512;
const int64_t L1REUSE_BNG_LIMIT_64 = 64;
const int64_t L1REUSE_BNG_LIMIT_4800 = 4800L;
const int64_t L1REUSE_D_LIMIT_144 = 144L;
const int64_t INVALID_ROW_SPARSE_RATIO = 6L;
const int64_t HEAD_DIM_MAX_VALUE = 512L;
const int64_t DATA_TYPE_FP16 = 2L;
const int64_t DATA_TYPE_FP32 = 4L;
enum LayoutType : uint8_t {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
};

enum AttenMaskShapeType : uint8_t {
    ATTEN_B_N2_G_S1_S2 = 0,
    ATTEN_B_1_1_S1_S2 = 1,
    ATTEN_1_1_1_S1_S2 = 2,
    ATTEN_1_1_1_T_T = 99,
};

enum PseShapeType : uint8_t {
    PSE_B_N2_G_S1_S2 = 0,
    PSE_B_N2_G_1_S2 = 1,
    PSE_B_N2_G_SLOPE,
    PSE_1_N2_G_SLOPE
};

enum SparseMode : uint8_t {
    NO_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    PREFIX,
    PREFIX_COMPRESS,
    RIGHT_DOWN_CAUSAL_BAND,
    BAND_LEFT_UP_CAUSAL
};

enum AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_MODE,
    PREFIX_MODE,
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,
    BAND_LEFT_UP_CAUSAL_MODE
};

enum ImplMode : uint8_t {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};

enum PseType : uint8_t {
    PSE_OUTER_MUL_ADD_TYPE = 0, // v2 default
    PSE_OUTER_ADD_MUL_TYPE, // v1 current usage
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE
};


enum PseEncodeType : uint8_t {
    PES_ENCODE_NONE = 0,
    PSE_ENCODE_ALIBI_S2_FULL = 0x11, // shape: (1024, S2)
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

class TilingKey {
public:
    TilingKey() : splitS1(0), splitS2(0), splitD(0), dtype(0), layoutType(0), sparseType(0), reserved(0)
    {
    }

    void Reset()
    {
        splitS1 = 0;
        splitS2 = 0;
        splitD = 0;
        dtype = 0;
        layoutType = 0;
        sparseType = 0;
        reserved = 0;
    }

    uint32_t GetRawTilingKey() const
    {
        return *(reinterpret_cast<const uint32_t *>(this));
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << " splitS1: " << splitS1 << " splitS2: " << splitS2 << " splitD: " << splitD;
        return ss.str();
    }

    uint32_t splitS1    : 1;
    uint32_t splitS2    : 1;
    uint32_t splitD     : 1;
    uint32_t dtype      : 2;
    uint32_t layoutType : 2;
    uint32_t sparseType : 2;
    uint32_t reserved   : 23; // to fullfil 32 bit, if add new template bit then decrease this number
};

inline bool operator==(const TilingKey &left, const TilingKey &right)
{
    return left.GetRawTilingKey() == right.GetRawTilingKey();
}

using TemplateType = TilingKey;

class BufferNum {
public:
    // sum and max always use fp32, shape is (S1, 1), inner axis align 32B.
    size_t bufferS1S2Num; // unit: input dtype
    size_t bufferS1DNum;
    size_t bufferExpNum; // unit: input dtype, shape: [S1, 1], inner axis align 32B.
};

class FlashAttentionScoreTilingBase : public TilingBaseClass {
public:
    explicit FlashAttentionScoreTilingBase(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~FlashAttentionScoreTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    virtual bool GetSparseInfo(SparseEnum &sparseType);
    void EnableBandInvalidLineImplMode();
    bool SparseModeProcess(SparseEnum &sparseType);
    void SetSparseTilingInfo(SparseEnum &sparseType);
    bool SparseBandModeCheck(int64_t maxS1Value, int64_t maxS2Value, int64_t minS1Value, int64_t minS2Value,
                             SparseEnum &sparseType);
    bool PretokenAndNexttokenAdjustment(SparseEnum &sparseType);

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override = 0;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    virtual void GetBufferNum(BufferNum &bufferNum) const = 0;

    void Reset();

    void GetActualSeqLenData(int64_t inputIdx, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &res, int64_t &actualLen) const;

    virtual int64_t GetNRatio();

    virtual int64_t GetMinS1BasicBlock() const
    {
        return std::min(64L, alignedS1);
    }

    virtual bool IsTemplateMatched() const
    {
        return expectTemplate == actualTemplate;
    }

    ge::graphStatus CheckContext();
    virtual bool AnalyzeDtype();
    bool AnalyzeAttrs();
    bool AnalyzeLayout();
    bool Analyze3DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, size_t layoutLen);
    bool Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, size_t layoutLen);
    bool AnalyzeOptionalInput();
    bool MatchTemplate();
    virtual void CalcS1S2BasicBlock(const BufferNum &bufferNum);
    virtual void CalcDBasicBlock();
    int64_t CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum) const;
    int64_t CalcMaxS2BasicBlockSize(const BufferNum &bufferNum, int64_t tmpS1BasicBlock) const;
    virtual bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch = 1) = 0;
    bool IsBasicBlockInSoftMax(const ge::Shape &shape) const;
    virtual bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                    int64_t batch, matmul_tiling::MatmulApiTiling &bmm1);
    virtual bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock,
                                    int64_t batch, matmul_tiling::MatmulApiTiling &bmm2) = 0;
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                         matmul_tiling::MatmulApiTiling &bmm1, matmul_tiling::MatmulApiTiling &bmm2);
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch = 1);
    virtual void SetCoreParams();
    virtual void SetQKVStartIdx();
    virtual void SetMultiBatchCoreParams();
    virtual void SetMultiCoreParams();
    virtual void SetSoftMaxTiling();
    void SetDataCopyTransposeTiling();
    virtual void SetTensorSizeParams();
    virtual void SetSparseParams();
    virtual bool SetPseAlibiParams();
    virtual bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx);
    virtual bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, MultiCoreParams &multiCoreParams);
    void SetPrefixSparseStartIdx(const std::vector<std::vector<int64_t>> &sparseValidArray,
                                 MultiCoreParams &multiCoreParams);
    void PrintSparseMaxMinLoadPerCore(const std::vector<int64_t> &sparseValidArray, int64_t *sparseStartIdx,
                                      int32_t validAivNum, int64_t avgLoadSize);
    bool PartitionSparseData(const std::vector<int64_t> &sparseRollingArray, int64_t sparseRollingArraySum,
                             int64_t sparseArraySize, int64_t loadMaxEachCore, std::vector<int64_t> &partitionResult);
    SparseEnum GetPrefixNList(std::ostringstream &failReason);

    uint32_t aivNum;
    uint32_t aicNum;
    int64_t apiMaxUBSize = 0;

    matmul_tiling::DataType bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    ge::DataType inputDtype;
    int64_t inputDtypeBytes;
    int64_t calcTypeSize;

    bool isHighPercision; // fp16 high percision mode

    DtypeEnum tilingKeyDType;
    LayoutType tilingKeyLayout;
    ImplMode implMode;
    CubeFormatEnum tilingKeyBmm1Format = CubeFormatEnum::ND;
    CubeInputSourceEnum tilingKeyBmm1Source = CubeInputSourceEnum::GM;
    CubeInputSourceEnum tilingKeyBmm2Source = CubeInputSourceEnum::GM;

    int64_t bSize;
    int64_t gSize;
    int64_t dSize;
    int64_t n1Size;
    int64_t n2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t s1StrideSize; // query Shape S inner axes, for bmm1
    int64_t s2StrideSize; // key Shape S inner axes, for bmm1
    int64_t preTokens;
    int64_t nextTokens;
    int64_t s1SparseValidSize;
    int64_t s2SparseValidSize;
    int64_t sparseMode;
    int64_t pseType;
    int64_t pseAlibiBaseS1;
    int64_t pseAlibiBaseS2;
    int64_t qStartIdx;
    int64_t kvStartIdx;
    int64_t maxS1Val;
    int64_t minS1Val;
    int64_t accumS1;
    int64_t accumS1BlockNum;
    int64_t dropTotalSize;
    int64_t maxS2Val;
    int64_t minS2Val;
    int64_t accumS2;
    int64_t bandIndex;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenData;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenKvData;
    float keepProb;
    float scaleValue;
    uint8_t attenMaskCompressMode;
    uint8_t pseExistFlag;
    uint8_t attenMaskExistFlag;
    uint8_t dropMaskExistFlag;

    int64_t alignedS1;
    int64_t alignedS2;
    int64_t alignedD;

    int64_t s1BasicBlock;
    int64_t s2BasicBlock;
    int64_t dBasicBlock;
    int64_t batchBasic;
    int64_t nRatio;

    int64_t minUsedUBSize;
    int64_t maxValidS2Len;

    const char *templateName = "base";
    const char *opName;
    const char *inputLayout;
    const int64_t *prefixNData;
    TemplateType expectTemplate;
    TemplateType actualTemplate;

    bool isSparseValidSizeAligned = false;
    bool hasPse = false;
    bool hasAttenMask = false;
    bool hasDropOut = false;
    FlashAttentionScoreGeneralTilingData tilingData;
};

int64_t FlashAttentionScoreTilingBase::GetNRatio()
{
    return BMM_SOFTMAX_RATIO;
}

ge::graphStatus FlashAttentionScoreTilingBase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FlashAttentionScoreCompileInfo *>(context_->GetCompileInfo());
        OPS_ERR_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "compileInfoPtr is null."),
                   return ge::GRAPH_FAILED);
        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }
    OPS_LOG_I(context_, "get platform from compileInfo. aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
              aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTilingBase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED)
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto keepProbPtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    size_t *workspaces = context_->GetWorkspaceSizes(1);

    OPS_LOG_E_IF_NULL(context_, scaleValuePtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, keepProbPtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, preTokensPtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, nextTokensPtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, n1SizePtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, inputLayoutPtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, workspaces, return ge::GRAPH_FAILED)

    auto queryShape = context_->GetInputShape(0);
    auto queryDesc = context_->GetInputDesc(0);
    auto keyShape = context_->GetInputShape(1);
    auto attenOutShape = context_->GetOutputShape(ATTEN_OUT_INDEX);

    OPS_LOG_E_IF_NULL(context_, queryShape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, queryDesc, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, keyShape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, attenOutShape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, context_->GetRawTilingData(), return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context_, context_->GetRawTilingData()->GetData(), return ge::GRAPH_FAILED)
    OPS_ERR_IF(context_->GetRawTilingData()->GetCapacity() < tilingData.GetDataSize(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "context tiling data capacity %zu < actual tiling data size %zu.",
                                           context_->GetRawTilingData()->GetCapacity(), tilingData.GetDataSize()),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreTilingBase::SetSparseTilingInfo(SparseEnum &sparseType)
{
    auto &inputParams = tilingData.inputParams;
    inputParams.set_attenMaskCompressMode(attenMaskCompressMode);
    inputParams.set_sparseType(static_cast<uint8_t>(sparseType));
    inputParams.set_preTokens(preTokens);
    inputParams.set_nextTokens(nextTokens);
}

void FlashAttentionScoreTilingBase::EnableBandInvalidLineImplMode()
{
    if (implMode == AA_INVALID_LINE_HIGH_PRECISION) {
        return;
    }
    // pretoken and nexttoken are already valid values (leftup vertex) after adjusted
    if (preTokens < (s1Size - s2Size) || nextTokens < 0) {
        implMode = AA_INVALID_LINE_HIGH_PRECISION;
        OPS_LOG_I(context_, "Enable invalid line impl mode.");
        return;
    }
}

bool FlashAttentionScoreTilingBase::PretokenAndNexttokenAdjustment(SparseEnum &sparseType)
{
    if (sparseMode == ALL_MASK || sparseMode == PREFIX || sparseMode == PREFIX_COMPRESS) {
        if (preTokens < s1Size - 1 || nextTokens < s2Size - 1) {
            OPS_LOG_W(context_,
                      "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                      "preTokens and nextTokens will be reset max int value.",
                      preTokens, nextTokens, sparseMode);
            preTokens = std::numeric_limits<int32_t>::max();
            nextTokens = std::numeric_limits<int32_t>::max();
        }
        sparseType = (sparseMode == PREFIX_COMPRESS) ? static_cast<SparseEnum>(static_cast<uint8_t>(PREFIX)) :
                                                       static_cast<SparseEnum>(static_cast<uint8_t>(sparseMode));
    } else if (sparseMode == LEFT_UP_CAUSAL) {
        if (preTokens != s1Size || nextTokens != 0) {
            OPS_LOG_W(context_,
                      "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                      "preTokens will be reset max int value and nextTokens will be reset 0.",
                      preTokens, nextTokens, sparseMode);
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            nextTokens = 0;
        }
        sparseType = SparseEnum::CAUSAL;
    } else if (sparseMode == RIGHT_DOWN_CAUSAL) {
        if (s1Size == s2Size) {
            if (preTokens != s1Size || nextTokens != 0) {
                OPS_LOG_W(context_,
                          "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                          "preTokens will be reset max int value and nextTokens will be reset 0.",
                          preTokens, nextTokens, sparseMode);
                preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
                nextTokens = 0;
            }
            sparseType = SparseEnum::CAUSAL;
        } else {
            // unequal S, change to band
            preTokens = s1Size;
            nextTokens = s2Size - s1Size;
            OPS_LOG_D(context_,
                      "Unequal s, sparseType rightDownCasual reset to band, and reset preTokens[%ld] "
                      "and nextTokens[%ld].",
                      preTokens, nextTokens);
            sparseType = SparseEnum::BAND;
            // check need to enable AA_INVALID_LINE_HIGH_PRECISION
            EnableBandInvalidLineImplMode();
            s1SparseValidSize = preTokens;
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            isSparseValidSizeAligned = true;
        }
    } else if (sparseMode == BAND) {
        // unequal s, pretoken and nexttoken count from rigthDown vertex, need to change to leftUp vertex
        if (s1Size != s2Size) {
            preTokens = s1Size - s2Size + preTokens;
            nextTokens = s2Size - s1Size + nextTokens;
        }
        if (!SparseBandModeCheck(s1Size, s2Size, s1Size, s2Size, sparseType)) {
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingBase::SparseBandModeCheck(int64_t maxS1Value, int64_t maxS2Value, int64_t minS1Value,
                                                        int64_t minS2Value, SparseEnum &sparseType)
{
    int64_t oriPreTokens = (sparseMode == BAND) ? (preTokens + s2Size - s1Size) : preTokens;
    int64_t oriNextTokens = (sparseMode == BAND) ? (nextTokens + s1Size - s2Size) : nextTokens;
    if (preTokens >= 0 && nextTokens >= 0) {
        if (preTokens >= maxS1Value && nextTokens >= maxS2Value) {
            OPS_LOG_W(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error, should not both greater than maxS1Val[%ld] "
                      "maxS2Val[%ld].",
                      oriPreTokens, oriNextTokens, maxS1Value, maxS2Value);
            return true;
        }
        s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
        s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
        isSparseValidSizeAligned = true;
        sparseType = SparseEnum::BAND;
        // check need to enable AA_INVALID_LINE_HIGH_PRECISION
        EnableBandInvalidLineImplMode();
        return true;
    }

    if (preTokens < 0 && nextTokens < 0) {
        OPS_LOG_E(context_, "PreTokens[%ld] and nextTokens[%ld] config error, there is no valid data block.",
                  oriPreTokens, oriNextTokens);
        return false;
    }

    if (preTokens < 0 && nextTokens >= 0) {
        int64_t absPreTokens = std::abs(preTokens);
        if (nextTokens >= absPreTokens && absPreTokens < minS2Value) {
            // check need to enable AA_INVALID_LINE_HIGH_PRECISION
            EnableBandInvalidLineImplMode();
            s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), 0L);
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            isSparseValidSizeAligned = true;
            sparseType = SparseEnum::BAND;
            return true;
        } else {
            OPS_LOG_E(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error with S1[%ld], there is no valid data block.",
                      oriPreTokens, oriNextTokens, minS1Value);
            return false;
        }
    }

    if (preTokens >= 0 && nextTokens < 0) {
        int64_t absNextTokens = std::abs(nextTokens);
        if (absNextTokens <= preTokens && absNextTokens < minS1Value) {
            // check need to enable AA_INVALID_LINE_HIGH_PRECISION
            EnableBandInvalidLineImplMode();
            s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), 0L);
            isSparseValidSizeAligned = true;
            sparseType = SparseEnum::BAND;
            return true;
        } else {
            OPS_LOG_E(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error with S2[%ld], there is no valid data block.",
                      oriPreTokens, oriNextTokens, minS2Value);
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingBase::SparseModeProcess(SparseEnum &sparseType)
{
    if (sparseMode > PREFIX_COMPRESS) {
        OPS_LOG_E(context_, "Not support sparse mode of %ld.", sparseMode);
        return false;
    }

    if (!PretokenAndNexttokenAdjustment(sparseType)) {
        return false;
    }

    if (sparseMode == static_cast<int64_t>(SparseEnum::PREFIX) || sparseMode == static_cast<int64_t>(PREFIX_COMPRESS)) {
        std::ostringstream failReason;
        sparseType = GetPrefixNList(failReason);
        if (sparseType != SparseEnum::PREFIX && sparseMode == static_cast<int64_t>(PREFIX_COMPRESS)) {
            OPS_LOG_E(context_, "[%s] %s.", templateName, failReason.str().c_str());
            return false;
        }

        if (sparseType == SparseEnum::PREFIX && sparseMode == static_cast<int64_t>(PREFIX) &&
            tilingData.inputParams.get_attenMaskShapeType() != ATTEN_B_N2_G_S1_S2 &&
            tilingData.inputParams.get_attenMaskShapeType() != ATTEN_B_1_1_S1_S2 && bSize != 1) {
            OPS_LOG_E(context_, "Prefix mode get invalid atten_mask shape, should be [BNSS] or [B1SS].");
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingBase::GetSparseInfo(SparseEnum &sparseType)
{
    OPS_LOG_D(context_, "check sparse info: preTokens[%ld], nextTokens[%ld], s1[%ld], s2[%ld], attenMaskExistFlag[%d].",
              preTokens, nextTokens, s1Size, s2Size, attenMaskExistFlag);
    if (attenMaskExistFlag != 1) {
        return true;
    }

    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        return true;
    }

    if (sparseMode == NO_MASK) {
        if (preTokens >= s1Size && nextTokens == 0) {
            sparseType = SparseEnum::CAUSAL;
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
        } else {
            if (preTokens >= s1Size && nextTokens >= s2Size) {
                return true;
            }
            if (!SparseBandModeCheck(s1Size, s2Size, s1Size, s2Size, sparseType)) {
                return false;
            }
        }
    } else {
        if (!SparseModeProcess(sparseType)) {
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingBase::SetPseAlibiParams()
{
    auto &inputParams = tilingData.inputParams;
    inputParams.set_pseEncodeType(PES_ENCODE_NONE);
    auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
    if (pseShape == nullptr) {
        return true;
    }
    auto pseS1Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 2);
    auto pseS2Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 1);
    if (pseS1Size == PSE_ALIBI_S_SIZE && s1Size > PSE_ALIBI_S_SIZE && pseS2Size == s2Size) {
        if (s1Size != s2Size) {
            OPS_REPORT_VECTOR_INNER_ERR(opName, "Pse alibi only support same S1 S2 when S1 lager than 1024");
            return false;
        }
    }
    return true;
}

ge::graphStatus FlashAttentionScoreTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OPS_LOG_D_FULL(opName, "TilingContext: %s.", GetTilingContextDebugStr().c_str());
    OPS_ERR_IF(CheckContext() != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid context."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout() || !AnalyzeOptionalInput(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);

    alignedS1 = AlignUp(s1Size, FRACTAL_NUM);
    alignedS2 = AlignUp(s2Size, FRACTAL_NUM);
    alignedD = AlignUp(dSize, FRACTAL_NUM);

    OPS_ERR_IF(alignedS1 <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid alignedS1 %ld.", alignedS1),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(alignedS2 <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid alignedS2 %ld.", alignedS2),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(alignedD <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid alignedD %ld.", alignedD),
        return ge::GRAPH_FAILED);

    auto &inputParams = tilingData.inputParams;
    inputParams.set_bSize(bSize);
    inputParams.set_n2Size(n2Size);
    inputParams.set_gSize(gSize);
    inputParams.set_s1Size(s1Size);
    inputParams.set_s2Size(s2Size);
    inputParams.set_dSize(dSize);
    inputParams.set_keepProb(keepProb);
    inputParams.set_scaleValue(scaleValue);
    inputParams.set_alignedS2(alignedS2);
    inputParams.set_pseType(static_cast<uint32_t>(pseType));
    OPS_LOG_D(context_, "input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], keepProb[%f], scaleValue[%f],"
              "pseType:%ld.", bSize, n2Size, gSize, s1Size, s2Size, dSize, keepProb, scaleValue, pseType);

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreTilingBase::Reset()
{
    tilingData.SetDataPtr(context_->GetRawTilingData()->GetData());
    apiMaxUBSize = 0;

    bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    inputDtype = ge::DT_FLOAT16;
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    calcTypeSize = inputDtypeBytes;

    tilingKeyDType = DtypeEnum::FLOAT16;
    tilingKeyLayout = LayoutType::LAYOUT_BNSD;
    tilingKeyBmm1Format = CubeFormatEnum::ND;
    tilingKeyBmm1Source = CubeInputSourceEnum::GM;

    bSize = 0LL;
    gSize = 0LL;
    dSize = 0LL;
    n1Size = 0LL;
    n2Size = 0LL;
    s1Size = 0LL;
    s2Size = 0LL;
    maxS1Val = 0LL;
    minS1Val = 0LL;
    accumS1 = 0LL;
    accumS2 = 0LL;
    bandIndex = 0LL;
    dropTotalSize = 0LL;
    maxS2Val = 0LL;
    minS2Val = 0LL;

    s1StrideSize = 0LL;
    s2StrideSize = 0LL;
    preTokens = std::numeric_limits<int32_t>::max();
    nextTokens = std::numeric_limits<int32_t>::max();
    sparseMode = static_cast<int64_t>(NO_MASK);
    pseType = PSE_OUTER_ADD_MUL_TYPE;
    pseAlibiBaseS1 = 0;
    pseAlibiBaseS2 = 0;
    qStartIdx = 0;
    kvStartIdx = 0;
    keepProb = 1.0f;
    scaleValue = 1.0f;
    pseExistFlag = 0;
    attenMaskCompressMode = NO_COMPRESS_MODE;
    attenMaskExistFlag = 0;
    dropMaskExistFlag = 0;
    isHighPercision = true;

    alignedS1 = 0LL;
    alignedS2 = 0LL;
    alignedD = 0LL;

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    dBasicBlock = std::numeric_limits<int64_t>::max();
    nRatio = GetNRatio();

    minUsedUBSize = 0LL;
    maxValidS2Len = 0LL;
    batchBasic = 1LL;

    opName = nullptr;
    inputLayout = nullptr;

    actualTemplate.Reset();
}

bool FlashAttentionScoreTilingBase::AnalyzeDtype()
{
    inputDtype = context_->GetInputDesc(0)->GetDataType();
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    switch (inputDtype) {
        case ge::DT_FLOAT16:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT16;
            bmm1OutDtype = isHighPercision ? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;
            tilingKeyDType = isHighPercision ? DtypeEnum::FLOAT16_PRECISION : DtypeEnum::FLOAT16;
            calcTypeSize = isHighPercision ? ge::GetSizeByDataType(ge::DT_FLOAT) : ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_FLOAT:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::FLOAT32;
            isHighPercision = false;
            calcTypeSize = ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_BF16:
            bmmDtype = matmul_tiling::DataType::DT_BF16;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::BFLOAT16;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            isHighPercision = false;
            break;
        default:
            OPS_REPORT_VECTOR_INNER_ERR(opName, "not support input dtype: %s for now",
                                        ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
            return false;
    }

    bmm2OutDtype = bmm1OutDtype;
    OPS_LOG_D(context_, "Get high precision flag: %d.", isHighPercision);
    return true;
}

bool FlashAttentionScoreTilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto keepProbPtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<uint32_t>(idx++);
    inputLayout = attrs->GetAttrPointer<char>(idx++);

    preTokens = *preTokensPtr;
    nextTokens = *nextTokensPtr;
    keepProb = *keepProbPtr;
    scaleValue = *scaleValuePtr;
    n1Size = *n1SizePtr;
    OPS_ERR_IF(n1Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "Head num is zero."), return false);
    OPS_ERR_IF(keepProb <= 0.0 || keepProb > 1.0,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "keepProb value must be in range of (0, 1]."), return false);

    implMode = ImplMode::AA_HIGH_PRECISION;
    if (attrs->GetAttrNum() > idx) {
        auto implModePtr = attrs->GetAttrPointer<uint8_t>(idx++);
        if (static_cast<ImplMode>(*implModePtr) == ImplMode::AA_INVALID_LINE_HIGH_PRECISION) {
            implMode = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
        }
        isHighPercision = true; // use default value
    }

    if (attrs->GetAttrNum() > idx) {
        auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
        sparseMode = *sparseModePtr;
        if (sparseMode == LEFT_UP_CAUSAL) {
            attenMaskCompressMode = LEFT_UP_CAUSAL_MODE;
        } else if (sparseMode == RIGHT_DOWN_CAUSAL) {
            attenMaskCompressMode = RIGHT_DOWN_CAUSAL_MODE;
        } else if (sparseMode == BAND) {
            attenMaskCompressMode = BAND_MODE;
        } else if (sparseMode == RIGHT_DOWN_CAUSAL_BAND) {
            attenMaskCompressMode = RIGHT_DOWN_CAUSAL_BAND_MODE;
        } else if (sparseMode == BAND_LEFT_UP_CAUSAL) {
            attenMaskCompressMode = BAND_LEFT_UP_CAUSAL_MODE;
        } else if (sparseMode == PREFIX_COMPRESS) {
            attenMaskCompressMode = PREFIX_MODE;
        }
        OPS_LOG_D(context_, "The current value of attenMaskCompressMode is %u.", attenMaskCompressMode);
    }
    if (attrs->GetAttrNum() > idx) {
        auto pseTypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        pseType = *pseTypePtr;
        OPS_ERR_IF(pseType < 0 || pseType >= PSE_INVALID_TYPE,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "pseType value is out of range"), return false);
    }
    OPS_LOG_D(context_, "attrs: scale_value[%f] keep_prob[%f] pre_tockens[%ld] next_tockens[%ld] head_num[%ld]"
              "input_layout[%s] inner_precise[%d] sparse_mode[%ld] pseType[%ld].",
              scaleValue, keepProb, preTokens, nextTokens, n1Size, inputLayout, implMode, sparseMode, pseType);
    return true;
}

bool FlashAttentionScoreTilingBase::AnalyzeLayout()
{
    auto &queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto &keyShape = context_->GetInputShape(1)->GetStorageShape();

    size_t layoutLen = strlen(inputLayout);
    OPS_LOG_D(context_, "Get input_layout [%s].", inputLayout);
    OPS_ERR_IF(queryShape.GetDimNum() != layoutLen || keyShape.GetDimNum() != layoutLen,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "Invalid layout[%s].", inputLayout), return false);
    OPS_ERR_IF(!Analyze3DimLayout(queryShape, keyShape, layoutLen) ||
                   !Analyze4DimLayout(queryShape, keyShape, layoutLen),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "Get unsupported layout: %s", inputLayout), return false);
    OPS_ERR_IF(gSize == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "gSize is zero."), return false);
    OPS_ERR_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "n2Size is zero."), return false);
    OPS_ERR_IF(dSize > HEAD_DIM_MAX_VALUE || dSize <= 0L,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "dSize is not in range:(0, 512]."), return false);
    OPS_ERR_IF(n1Size % n2Size != 0,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "n1Size [%ld] should be a multiple of n2Size [%ld].", n1Size, n2Size),
               return false);
    return true;
}

void FlashAttentionScoreTilingBase::GetActualSeqLenData(int64_t inputIdx, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &res,
                                                        int64_t &actualLen) const
{
    auto actualSeqLenTensor = context_->GetOptionalInputTensor(inputIdx);
    if (actualSeqLenTensor == nullptr) {
        OPS_LOG_W(context_, "[%s]actualSeqLenTensor is null pointer", templateName);
        return;
    }
    auto &actualSeqLenShape = actualSeqLenTensor->GetShape().GetStorageShape();
    if (actualSeqLenShape.GetDimNum() != 1) {
        OPS_LOG_W(context_, "[%s]actualSeqLenShape is invalid %lu %ld", templateName, actualSeqLenShape.GetDimNum(),
                  actualSeqLenShape.GetDim(0));
        return;
    }
    /* Get Data from tensor. */
    const int64_t *value = actualSeqLenTensor->GetData<int64_t>();
    if (value == nullptr) {
        OPS_LOG_W(context_, "[%s]actualSeqLenTensor data is null pointer", templateName);
        return;
    }
    res[0] = value[0];
    actualLen++;
    for (int64_t i = 1; i < actualSeqLenShape.GetDim(0); ++i) {
        res[i] = value[i] - value[i - 1];
        actualLen++;
    }
}

bool FlashAttentionScoreTilingBase::Analyze3DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                                      size_t layoutLen)
{
    int64_t h1 = 0;
    int64_t h2 = 0;
    if (layoutLen == 3UL) {
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'H') { // 2: H idx
            s1Size = queryShape.GetDim(1);
            bSize = queryShape.GetDim(0);
            s2Size = keyShape.GetDim(1);
            h1 = queryShape.GetDim(2); // 2: H idx
            h2 = keyShape.GetDim(2);   // 2: H idx
            s1StrideSize = h1;
            s2StrideSize = h2;
            tilingData.inputParams.set_layoutType(LAYOUT_BSH);
            tilingKeyLayout = LayoutType::LAYOUT_BSH;
        } else if (inputLayout[0] == 'S' && inputLayout[1] == 'B' && inputLayout[2] == 'H') { // 2: H idx
            s1Size = queryShape.GetDim(0);
            s2Size = keyShape.GetDim(0);
            bSize = queryShape.GetDim(1);
            h1 = queryShape.GetDim(2); // 2: H idx
            h2 = keyShape.GetDim(2);   // 2: H idx
            s1StrideSize = h1 * bSize;
            s2StrideSize = h2 * bSize;
            tilingData.inputParams.set_layoutType(LAYOUT_SBH);
            tilingKeyLayout = LayoutType::LAYOUT_SBH;
        } else if (inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[2] == 'D') {
            int64_t actualSeqQLen = 0;
            int64_t actualSeqKVLen = 0;
            int64_t t1Size = queryShape.GetDim(0);
            int64_t t2Size = keyShape.GetDim(0);
            std::fill(actualSeqLenData.begin(), actualSeqLenData.end(), 0);
            std::fill(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0);
            GetActualSeqLenData(ACTUAL_SEQ_LENGTH_INPUT_INDEX, actualSeqLenData, actualSeqQLen);
            GetActualSeqLenData(ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX, actualSeqLenKvData, actualSeqKVLen);
            OPS_ERR_IF(actualSeqQLen != actualSeqKVLen,
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "VarLen scene, q is not equal kv."), return false);
            bSize = actualSeqQLen;
            accumS1 = std::accumulate(actualSeqLenData.begin(), actualSeqLenData.end(), 0LL);
            accumS2 = std::accumulate(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0LL);
            OPS_ERR_IF(
                t1Size != accumS1 || t2Size != accumS2,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName,
                    "Query T(%ld) and key T(%ld) need equal to respectively sum of seqLen(%ld) and sekvLen(%ld).",
                    t1Size, t2Size, accumS1, accumS2),
                return false);
            for (int64_t i = 0; i < bSize; ++i) {
                OPS_ERR_IF(actualSeqLenData[i] != 0 && actualSeqLenKvData[i] == 0,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName,
                        "Batch(%ld) seqLen is not 0 and sekvLen is 0, this scene is not supported.",
                        i),
                    return false);
            }
            uint32_t firstValidIndex = 0;
            uint32_t lastValidIndex = bSize - 1;
            for (int64_t i = 0; i < bSize; ++i) {
                if (actualSeqLenData[i] != 0) {
                    firstValidIndex = static_cast<uint32_t>(i);
                    break;
                }
            }
            for (auto i = bSize - 1; i >= 0; --i) {
                if (actualSeqLenData[i] != 0) {
                    lastValidIndex = static_cast<uint32_t>(i);
                    break;
                }
            }
            if (sparseMode == RIGHT_DOWN_CAUSAL_BAND) {
                bandIndex = lastValidIndex;
                tilingData.inputParams.set_bandIndex(lastValidIndex);
            }
            if (sparseMode == BAND_LEFT_UP_CAUSAL) {
                bandIndex = firstValidIndex;
                tilingData.inputParams.set_bandIndex(firstValidIndex);
            }
            maxS1Val = *std::max_element(actualSeqLenData.begin(), actualSeqLenData.end());
            maxS2Val = *std::max_element(actualSeqLenKvData.begin(), actualSeqLenKvData.end());
            s1Size = maxS1Val;
            s2Size = maxS2Val;
            OPS_ERR_IF(n1Size != queryShape.GetDim(1),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "head_num is [%ld], but got query dim1 [%ld].", n1Size,
                                                   queryShape.GetDim(1)),
                       return false);
            n2Size = keyShape.GetDim(1);
            OPS_ERR_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            gSize = queryShape.GetDim(1) / n2Size;
            dSize = queryShape.GetDim(2);
            h1 = n1Size * dSize;
            h2 = n2Size * dSize;
            s1StrideSize = gSize * n2Size * dSize;
            s2StrideSize = n2Size * dSize;
            tilingData.inputParams.set_layoutType(LAYOUT_TND);
            tilingKeyLayout = LayoutType::LAYOUT_TND;
        } else {
            return false;
        }
        OPS_ERR_IF(h1 == 0 || h2 == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "H is zero."), return false);
        OPS_ERR_IF(h1 % n1Size != 0,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "h1 [%ld] should be a multiple of n1Size [%ld].", h1, n1Size),
                   return false);
        dSize = h1 / n1Size;
        gSize = h1 / h2;
        n2Size = h2 / dSize;
    }

    return true;
}

bool FlashAttentionScoreTilingBase::Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                                      size_t layoutLen)
{
    if (layoutLen == 4UL) {
        // 2: N idx, 3: D idx
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            s1Size = queryShape.GetDim(1);
            s2Size = keyShape.GetDim(1);
            n2Size = keyShape.GetDim(2); // 2: N idx
            OPS_ERR_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            OPS_ERR_IF(n1Size != queryShape.GetDim(2),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "head_num is [%ld], but got query dim2 [%ld].", n1Size,
                                                   queryShape.GetDim(2)),
                       return false);
            gSize = queryShape.GetDim(2) / n2Size; // 2: N idx
            dSize = queryShape.GetDim(3);          // 3: D idx
            s1StrideSize = gSize * n2Size * dSize;
            s2StrideSize = n2Size * dSize;
            tilingData.inputParams.set_layoutType(LAYOUT_BSND);
            tilingKeyLayout = LayoutType::LAYOUT_BSND;
        } else if (inputLayout[0] == 'B' && inputLayout[1] == 'N' &&
                   // 2: S idx, 3: N idx
                   inputLayout[2] == 'S' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            n2Size = keyShape.GetDim(1); // 1: N idx
            OPS_ERR_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            OPS_ERR_IF(n1Size != queryShape.GetDim(1),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "head_num is [%ld], but got query dim1 [%ld].", n1Size,
                                                   queryShape.GetDim(1)),
                       return false);
            gSize = queryShape.GetDim(1) / n2Size;
            s1Size = queryShape.GetDim(2); // 2: S idx
            s2Size = keyShape.GetDim(2);   // 2: S idx
            dSize = queryShape.GetDim(3);  // 3: D idx
            s1StrideSize = dSize;
            s2StrideSize = dSize;
            tilingData.inputParams.set_layoutType(LAYOUT_BNSD);
            tilingKeyLayout = LayoutType::LAYOUT_BNSD;
        } else {
            return false;
        }
    }

    return true;
}

SparseEnum FlashAttentionScoreTilingBase::GetPrefixNList(std::ostringstream &failReason)
{
    auto prefixN = context_->GetOptionalInputTensor(PREFIX_INPUT_INDEX);
    if (prefixN == nullptr) {
        OPS_LOG_W(context_, "[%s]prefixN is null pointer while sparse mode is prefix", templateName);
        failReason << "prefixN is null pointer while sparse mode is prefix";
        return SparseEnum::ALL;
    }

    auto &prefixShape = prefixN->GetShape().GetStorageShape();
    if (prefixShape.GetDimNum() != 1) {
        OPS_LOG_W(context_, "[%s] prefixN shape is invalid, DimNum should be 1, but it is %lu.", templateName,
                  prefixShape.GetDimNum());
        failReason << "prefixN shape is invalid, DimNum should be 1, but it is " << prefixShape.GetDimNum();
        return SparseEnum::ALL;
    }
    if (prefixShape.GetDim(0) != bSize) {
        OPS_LOG_W(context_, "[%s] prefixN is invalid, it should be the same size as bSize[%ld], but it is %ld.",
                  templateName, bSize, prefixShape.GetDim(0));
        failReason << "prefixN is invalid, it should be the same size as bSize[" << bSize
                   << "], but it is " << prefixShape.GetDim(0);
        return SparseEnum::ALL;
    }
    /* Get Data from tensor. */
    prefixNData = prefixN->GetData<int64_t>();
    if (prefixNData == nullptr) {
        OPS_LOG_W(context_, "[%s]prefixN data is null pointer", templateName);
        failReason << "prefixN data is null pointer";
        return SparseEnum::ALL;
    }

    int64_t nMin = ((s2Size - s1Size) > 0) ? (s2Size - s1Size) : 0;
    for (int64_t i = 0; i < bSize; ++i) {
        if (prefixNData[i] < nMin || prefixNData[i] > s2Size) {
            OPS_LOG_W(context_, "[%s] batch[%ld] prefixN=%ld is invalid, should be in range of [%ld, %ld]",
                      templateName, i, prefixNData[i], nMin, s2Size);
            failReason << "batch[" << i << "] prefixN=" << prefixNData[i] << " is invalid, should be in range of ["
                       << nMin << ", " << s2Size << "]";
            return SparseEnum::ALL;
        }

        if (s1Size > s2Size && prefixNData[i] == 0) {
            implMode = AA_INVALID_LINE_HIGH_PRECISION;
            OPS_LOG_D(context_, "Enable invalid line impl mode.");
        }
    }

    return SparseEnum::PREFIX;
}
void FlashAttentionScoreTilingBase::SetQKVStartIdx() {
    tilingData.inputParams.set_qStartIdx(0);
    tilingData.inputParams.set_kvStartIdx(0);
    auto qStartIdxTensor = context_->GetOptionalInputTensor(Q_START_IDX_INPUT_INDEX);
    if (qStartIdxTensor == nullptr) {
        OPS_LOG_W(context_, "[%s]qStartIdxTensor is null pointer", templateName);
        return;
    }
    auto &qStartIdxShape = qStartIdxTensor->GetShape().GetStorageShape();
    if (qStartIdxShape.GetDimNum() != 1) {
        return;
    }
    /* Get Data from tensor. */
    const int64_t *value = qStartIdxTensor->GetData<int64_t>();
    if (value == nullptr) {
        return;
    }
    qStartIdx = value[0];

    auto kvStartIdxTensor = context_->GetOptionalInputTensor(KV_START_IDX_INPUT_INDEX);
    if (kvStartIdxTensor == nullptr) {
        OPS_LOG_W(context_, "[%s]kvStartIdxTensor is null pointer", templateName);
        return;
    }
    auto &kvStartIdxShape = kvStartIdxTensor->GetShape().GetStorageShape();
    if (kvStartIdxShape.GetDimNum() != 1) {
        return;
    }
    /* Get Data from tensor. */
    const int64_t *kvValue = kvStartIdxTensor->GetData<int64_t>();
    if (kvValue == nullptr) {
        return;
    }
    kvStartIdx = kvValue[0];

    tilingData.inputParams.set_qStartIdx(qStartIdx);
    tilingData.inputParams.set_kvStartIdx(kvStartIdx);
    OPS_LOG_D(context_, "[%s] SetQKVStartIdx qStartIdx:%ld, kvStartIdx:%ld", templateName, qStartIdx, kvStartIdx);
}
bool FlashAttentionScoreTilingBase::AnalyzeOptionalInput()
{
    // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    PseShapeType pseShapeType = PSE_B_N2_G_1_S2;
    auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
    if (pseShape != nullptr && pseShape->GetStorageShape().GetDimNum() != 0) {
        pseExistFlag = 1;
        auto &pseShapeDims = pseShape->GetStorageShape();
        size_t pseDimNum = pseShapeDims.GetDimNum();
        int64_t pseBSize = pseShapeDims.GetDim(0);
        if (pseType == PSE_INNER_MUL_ADD_TYPE || pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
            if (pseDimNum != SLOPE_BN_DIM_NUM && pseDimNum != SLOPE_N_DIM_NUM) {
                OPS_LOG_E(context_, "pse inner mode, unsupported pse shape");
                return false;
            }
            pseShapeType = PSE_B_N2_G_SLOPE;
            if (pseDimNum == 1) {
                pseShapeType = PSE_1_N2_G_SLOPE;
                pseBSize = 1;
            }
        } else if (tilingData.inputParams.get_layoutType() == LAYOUT_TND) {
            int64_t accumS1S2 = 0;
            for (int64_t i = 0; i < bSize; i++) {
                accumS1S2 += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            if (pseBSize == accumS2 * n1Size) {
                pseShapeType = PSE_B_N2_G_1_S2;
            } else if (pseBSize == accumS1S2 * n1Size) {
                pseShapeType = PSE_B_N2_G_S1_S2;
            } else if (pseDimNum == PSE_DIM_NUM && (pseShapeDims.GetDim(0) == 1 || pseShapeDims.GetDim(0) == bSize) &&
                       pseShapeDims.GetDim(1) == n1Size && pseShapeDims.GetDim(2) == PSE_ALIBI_S_SIZE &&
                       pseShapeDims.GetDim(3) == s2Size) {
                pseShapeType = PSE_B_N2_G_S1_S2;
            } else {
                OPS_LOG_E(context_, "get unsupported pse shape");
                return false;
            }
        } else {
            if (pseDimNum != PSE_DIM_NUM) {
                OPS_LOG_E(context_, "pse dim should be 4, but got %zu", pseDimNum);
                return false;
            }
            if (pseBSize != bSize && pseBSize != 1) {
                OPS_LOG_E(context_, "pse batchsize should be 1 or %ld, but got %ld", bSize, pseBSize);
                return false;
            }

            int64_t pseDim1Size = pseShapeDims.GetDim(1);
            int64_t pseDim2Size = pseShapeDims.GetDim(2);
            int64_t pseDim3Size = pseShapeDims.GetDim(3);
            if (pseDim1Size == n1Size && pseDim2Size == s1Size && pseDim3Size == s2Size) { // 2: pre last axiss
                pseShapeType = PSE_B_N2_G_S1_S2;
            } else if (pseDim1Size == n1Size && pseDim2Size == 1 && pseDim3Size == s2Size) {
                pseShapeType = PSE_B_N2_G_1_S2;
            } else if (pseDim1Size == n1Size && pseDim2Size == PSE_ALIBI_S_SIZE && pseDim3Size == s2Size) {
                if (s1Size < pseDim2Size) {
                    OPS_LOG_E(opName, "get unsupported pse shape, the shape is [%ld, %ld, %ld, %ld]", pseBSize, pseDim1Size,
                              pseDim2Size, pseDim3Size);
                    return false;
                }
                pseShapeType = PSE_B_N2_G_S1_S2;
            } else {
                OPS_LOG_E(opName, "get unsupported pse shape, the shape is [%ld, %ld, %ld, %ld]", pseBSize, pseDim1Size,
                          pseDim2Size, pseDim3Size);
                return false;
            }
        }
        tilingData.inputParams.set_pseBSize(static_cast<uint32_t>(pseBSize));
    }

    tilingData.inputParams.set_pseShapeType(pseShapeType);

    auto attenMaskInput = context_->GetOptionalInputDesc(ATTENTION_MASK_INPUT_INDEX);
    auto attenMaskShape = context_->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    if (attenMaskInput != nullptr && attenMaskShape != nullptr && attenMaskShape->GetStorageShape().GetDimNum() != 0) {
        attenMaskExistFlag = 1;
        auto attenMaskType = attenMaskInput->GetDataType();
        OPS_ERR_IF(attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_UINT8,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid attenMask dtype[%s], only support bool or uint8.",
                                               ge::TypeUtils::DataTypeToSerialString(attenMaskType).c_str()),
                   return false);

        tilingData.inputParams.set_attenMaskDataType(1);
        // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
        AttenMaskShapeType attenMaskShapeType = ATTEN_B_N2_G_S1_S2;
        auto &attenMaskStorageShape = attenMaskShape->GetStorageShape();
        size_t attenMaskDimNum = attenMaskStorageShape.GetDimNum();
        if (attenMaskDimNum == ATTENTION_MASK_DIM_NUM_4) {
            int64_t attenMaskDim0Size = attenMaskStorageShape.GetDim(0);
            int64_t attenMaskDim1Size = attenMaskStorageShape.GetDim(1);
            int64_t attenMaskDim2Size = attenMaskStorageShape.GetDim(2);
            int64_t attenMaskDim3Size = attenMaskStorageShape.GetDim(3);
            if (attenMaskDim0Size == 1 && attenMaskDim1Size == 1 && attenMaskDim2Size == s1Size &&
                attenMaskDim3Size == s2Size) {
                attenMaskShapeType = ATTEN_1_1_1_S1_S2;
            } else if (attenMaskDim0Size == bSize && attenMaskDim1Size == 1 && attenMaskDim2Size == s1Size &&
                       attenMaskDim3Size == s2Size) {
                attenMaskShapeType = ATTEN_B_1_1_S1_S2;
            } else if (attenMaskDim0Size == bSize && attenMaskDim1Size == n1Size && attenMaskDim2Size == s1Size &&
                       attenMaskDim3Size == s2Size) {
                attenMaskShapeType = ATTEN_B_N2_G_S1_S2;
            } else {
                OPS_LOG_E(context_, "get unsupported atten_mask shape, the shape is [%ld, %ld, %ld, %ld]",
                          attenMaskDim0Size, attenMaskDim1Size, attenMaskDim2Size, attenMaskDim3Size);
                return false;
            }
        } else if (attenMaskDimNum == ATTENTION_MASK_DIM_NUM_2) {
            int64_t attenMaskDim0Size = attenMaskStorageShape.GetDim(0);
            int64_t attenMaskDim1Size = attenMaskStorageShape.GetDim(1);
            if ((attenMaskDim0Size == s1Size && attenMaskDim1Size == s2Size) ||
                (attenMaskCompressMode != NO_COMPRESS_MODE)) {
                attenMaskShapeType = ATTEN_1_1_1_S1_S2; // maybe [S1, S2]
            } else if (attenMaskDim0Size == accumS1 && attenMaskDim1Size == accumS2) {
                attenMaskShapeType = ATTEN_1_1_1_T_T;
            } else {
                OPS_LOG_E(context_, "get unsupported atten_mask shape, the shape is [%ld, %ld]", attenMaskDim0Size,
                          attenMaskDim1Size);
                return false;
            }
        } else {
            OPS_LOG_E(context_, "atten mask dim should be 2 or 4, but got %zu", attenMaskDimNum);
            return false;
        }

        tilingData.inputParams.set_attenMaskShapeType(attenMaskShapeType);

        if ((attenMaskCompressMode != NO_COMPRESS_MODE && attenMaskCompressMode != PREFIX_MODE) &&
            ((attenMaskStorageShape.GetDim(attenMaskDimNum - ATTEN_MASK_S1_REV_INDEX) != ATTEN_MASK_COMPRESS_LIMIT) ||
             (attenMaskStorageShape.GetDim(attenMaskDimNum - 1) != ATTEN_MASK_COMPRESS_LIMIT))) {
            OPS_LOG_E(context_, "In the attenmask compression, please set the atten_mask_shape to [2048,2048].");
            return false;
        }
        if (attenMaskCompressMode == PREFIX_MODE &&
            ((attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - ATTEN_MASK_S1_REV_INDEX) !=
              ATTEN_MASK_COMPRESS_PREFIX_LIMIT) ||
             (attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - 1) != ATTEN_MASK_COMPRESS_LIMIT))) {
            OPS_LOG_E(context_, "In the prefix attenmask compression, please set the atten_mask_shape to [3072,2048].");
            return false;
        }
        tilingData.inputParams.set_attenMaskS2Size(attenMaskStorageShape.GetDim(attenMaskDimNum - 1));
    }

    auto dropMaskShape = context_->GetOptionalInputShape(DROP_MASK_INPUT_INDEX);
    auto dropMaskInput = context_->GetOptionalInputDesc(DROP_MASK_INPUT_INDEX);
    if (dropMaskInput != nullptr && dropMaskShape != nullptr && dropMaskShape->GetStorageShape().GetDimNum() != 0) {
        auto dropMaskDtype = dropMaskInput->GetDataType();
        OPS_ERR_IF(dropMaskDtype != ge::DT_UINT8,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dropMask dtype[%s], only support uint8.",
                                               ge::TypeUtils::DataTypeToSerialString(dropMaskDtype).c_str()),
                   return false);
        int64_t dimNum = dropMaskShape->GetStorageShape().GetDimNum();
        int64_t dropMaskShapeSize = 1;
        int64_t shapeSize = 0;
        for (int64_t i = 0; i < dimNum; ++i) {
            int64_t dimValue = dropMaskShape->GetStorageShape().GetDim(i);
            dropMaskShapeSize *= dimValue;
        }
        if (tilingData.inputParams.get_layoutType() == LAYOUT_TND) {
            int64_t accumS1S2 = 0;
            for (int64_t i = 0; i < bSize; i++) {
                accumS1S2 += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            shapeSize = accumS1S2 * n1Size;
        } else {
            shapeSize = bSize * n1Size * s1Size * s2Size;
        }
        shapeSize = AlignUp(shapeSize, BYTE_BIT_NUM) / BYTE_BIT_NUM;
        if (dropMaskShapeSize < shapeSize) {
            OPS_LOG_E(context_, "Input dropMask shapeSize is invalid, it should not be less than %ld, but got %ld",
                      shapeSize, dropMaskShapeSize);
            return false;
        }
        dropMaskExistFlag = 1;
    }

    // if s2Size algined to 8, then no need dropMaskOp to transfer dropMask from bit to byte format
    tilingData.inputParams.set_needDropMaskOp(static_cast<uint8_t>(dropMaskExistFlag == 1 && s2Size % 8 != 0));
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        auto needDropMaskOp = (dropMaskExistFlag == 1) && (s2Size % 8 != 0 || bSize > 1);
        tilingData.inputParams.set_needDropMaskOp(static_cast<uint8_t>(needDropMaskOp));
    }

    OPS_LOG_D(context_, "pseExistFlag: %d, attenMaskExistFlag: %d, dropMaskExistFlag: %d.", pseExistFlag,
              attenMaskExistFlag, dropMaskExistFlag);
    return true;
}

ge::graphStatus FlashAttentionScoreTilingBase::DoOpTiling()
{
    auto &inputParams = tilingData.inputParams;
    OPS_LOG_D(context_, "[%s]try template[%s]", templateName, expectTemplate.ToString().c_str());
    if (!MatchTemplate()) {
        OPS_LOG_I(context_,
                  "[%s]not match template[%s], input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], "
                  "keepProb[%f]",
                  templateName, expectTemplate.ToString().c_str(), inputParams.get_bSize(), inputParams.get_n2Size(),
                  inputParams.get_gSize(), inputParams.get_s1Size(), inputParams.get_s2Size(), inputParams.get_dSize(),
                  inputParams.get_keepProb());
        return ge::GRAPH_PARAM_INVALID;
    }

    SparseEnum sparseType = SparseEnum::ALL;
    OPS_ERR_IF(!GetSparseInfo(sparseType), OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to get sparse info."),
               return ge::GRAPH_FAILED);
    SetSparseTilingInfo(sparseType);
    inputParams.set_implMode(implMode);
    if (!isSparseValidSizeAligned) {
        s1SparseValidSize = preTokens;
        s2SparseValidSize = nextTokens;
    }
    SetQKVStartIdx();
    SetCoreParams();
    SetMultiCoreParams();
    SetTensorSizeParams();
    SetSparseParams();
    OPS_ERR_IF(!SetPseAlibiParams(), OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to set pse alibi info."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTilingBase::MatchTemplate()
{
    // UB Size calc logic: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    BufferNum bufferNum;
    GetBufferNum(bufferNum);

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    CalcS1S2BasicBlock(bufferNum);

    if (s2BasicBlock == std::numeric_limits<int64_t>::max()) {
        OPS_LOG_D(context_,
                  "[%s]can't find proper S1S2 basic block for shape: S1[%ld] S2[%ld], D[%ld], minS1BasicBlock[%ld], "
                  "dtype[%s], high precision[%d]",
                  templateName, s1Size, s2Size, dSize, GetMinS1BasicBlock(),
                  ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(), isHighPercision);
        return false;
    }

    CalcDBasicBlock();
    actualTemplate.splitS1 = s1BasicBlock < alignedS1 ? 1 : 0;
    actualTemplate.splitS2 = s2BasicBlock < alignedS2 ? 1 : 0;
    actualTemplate.splitD = dBasicBlock < alignedD ? 1 : 0;

    if (IsTemplateMatched()) {
        (void)CalcUBSize(s1BasicBlock, s2BasicBlock);
        OPS_LOG_D(context_, "[%s]final basic block: [%ld, %ld, %ld], match template[%s].", templateName, s1BasicBlock,
                  s2BasicBlock, dBasicBlock, actualTemplate.ToString().c_str());
        return true;
    }

    return false;
}

void FlashAttentionScoreTilingBase::CalcS1S2BasicBlock(const BufferNum &bufferNum)
{
    // calc s1 s2 first, we set d basic block as s2 now
    const int64_t actualD = expectTemplate.splitD == 0 ? alignedD : FRACTAL_NUM; // if split d we use min s2 16
    int64_t maxS1BasicBlock = CalcMaxS1BasicBlockSize(actualD, bufferNum);
    maxS1BasicBlock = std::min(maxS1BasicBlock, alignedS1);
    if (maxS1BasicBlock == 0) {
        return;
    }

    for (int64_t tmpS1BasicBlock = std::min(GetMinS1BasicBlock(), maxS1BasicBlock); tmpS1BasicBlock <= maxS1BasicBlock;
         tmpS1BasicBlock += FRACTAL_NUM) {
        int64_t tmpS2BasicBlock = CalcMaxS2BasicBlockSize(bufferNum, tmpS1BasicBlock);
        tmpS2BasicBlock = std::min(tmpS2BasicBlock, alignedS2);
        for (; tmpS2BasicBlock >= FRACTAL_NUM; tmpS2BasicBlock -= FRACTAL_NUM) {
            // drop mask bug workaround
            if (dropMaskExistFlag == 1 &&
                (tmpS2BasicBlock <= BYTE_BLOCK || CalcTailSize(alignedS2, tmpS2BasicBlock) <= BYTE_BLOCK)) {
                continue;
            }

            int64_t tmpDBasicBlock = expectTemplate.splitD == 1 ? std::min(tmpS2BasicBlock, alignedD) : alignedD;
            OPS_LOG_D(context_, "[%s]try basic block: [%ld, %ld]", templateName, tmpS1BasicBlock, tmpS2BasicBlock);
            if (CalcUBSize(tmpS1BasicBlock, tmpS2BasicBlock, 1) &&
                SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock)) {
                break;
            }
        }

        // check whether is valid, if tmpS1BasicBlock is too big, then there is no proper tmpS2BasicBlock
        if (tmpS2BasicBlock < FRACTAL_NUM) {
            break;
        }

        OPS_LOG_D(context_, "[%s]get candidate basic block: [%ld, %ld]", templateName, tmpS1BasicBlock,
                  tmpS2BasicBlock);
        if (s2BasicBlock == std::numeric_limits<int64_t>::max()) {
            s1BasicBlock = tmpS1BasicBlock;
            s2BasicBlock = tmpS2BasicBlock;
        } else if (s2BasicBlock == tmpS2BasicBlock && s1BasicBlock < tmpS1BasicBlock) {
            s1BasicBlock = tmpS1BasicBlock;
        } else {
            break;
        }
    }
}

void FlashAttentionScoreTilingBase::CalcDBasicBlock()
{
    return;
}

int64_t FlashAttentionScoreTilingBase::CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum) const
{
    // if S2 basic block is min value 16, s1 basic block can reach max value, then we get:
    // s1 * 16 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp =>
    // s1 * (16 * X + D * Y + (expNum + 2) * (32 / sizeof(T))) * sizeof(T) + apiTmp
    // just ignore apiTmp now, consider it at last
    int64_t alignUnit = BYTE_BLOCK / inputDtypeBytes;
    int64_t maxS1BasicBlock =
        aicoreParams_.ubSize / inputDtypeBytes /
        (FRACTAL_NUM * bufferNum.bufferS1S2Num + actualD * bufferNum.bufferS1DNum +
         (bufferNum.bufferExpNum + 2) * alignUnit); // here 2 means FlashSoftMax sum and max output
    return AlignDown(maxS1BasicBlock, FRACTAL_NUM);
}

int64_t FlashAttentionScoreTilingBase::CalcMaxS2BasicBlockSize(const BufferNum &bufferNum,
                                                               int64_t tmpS1BasicBlock) const
{
    // used UB: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    // if D full load, use alignedD in above formula
    // if D not full load, use S2 basic block var in above formula
    // just ignore apiTmp now, consider it at last
    int64_t tmpS2BasicBlock;
    if (expectTemplate.splitD == 0) {
        // here 2 means FlashSoftMax sum and max output
        tmpS2BasicBlock = (aicoreParams_.ubSize - tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK -
                           tmpS1BasicBlock * alignedD * bufferNum.bufferS1DNum * inputDtypeBytes) /
                          (tmpS1BasicBlock * bufferNum.bufferS1S2Num * inputDtypeBytes);
    } else {
        // here 2 means FlashSoftMax sum and max output
        tmpS2BasicBlock = (aicoreParams_.ubSize - tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK) /
                          (tmpS1BasicBlock * (bufferNum.bufferS1DNum + bufferNum.bufferS1S2Num) * inputDtypeBytes);
    }
    return std::min(AlignDown(tmpS2BasicBlock, FRACTAL_NUM), alignedS2);
}

bool FlashAttentionScoreTilingBase::IsBasicBlockInSoftMax(const ge::Shape &shape) const
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

void FlashAttentionScoreTilingBase::SetCoreParams()
{
    auto &coreParams = tilingData.coreParams;
    coreParams.set_s1BaseSize(s1BasicBlock);
    coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock));
    coreParams.set_s1OuterSize(CeilDivision(s1Size, s1BasicBlock));
    coreParams.set_s2BaseSize(s2BasicBlock);
    coreParams.set_s2BaseTailSize(CalcTailSize(s2Size, s2BasicBlock));
    coreParams.set_s2OuterSize(CeilDivision(s2Size, s2BasicBlock));
    if (expectTemplate.splitS2 == 1) {
        nRatio = std::min(GetNRatio(), coreParams.get_s2OuterSize());
        coreParams.set_s2OuterSize(CeilDivision(coreParams.get_s2OuterSize(), nRatio));
    } else if (expectTemplate.splitS1 == 1) {
        nRatio = std::min(GetNRatio(), coreParams.get_s1OuterSize());
    } else {
        nRatio = 1;
    }
    coreParams.set_nRatio(nRatio);

    coreParams.set_dBaseSize(dBasicBlock);
    coreParams.set_dBaseTailSize(CalcTailSize(dSize, dBasicBlock));
    coreParams.set_dOuterSize(CeilDivision(dSize, dBasicBlock));
    // 向下取整保证数据量不超32K
    int64_t s1Vec2BaseSize = 8 * 1024 * 2 / (alignedD * inputDtypeBytes);
    coreParams.set_s1Vec2BaseSize(std::min(s1Vec2BaseSize, S1_VEC2_BASE_SIZE_MAX));
    coreParams.set_s1Vec2BaseTailSize(s1Size % coreParams.get_s1Vec2BaseSize());
    SetMultiBatchCoreParams();
}

void FlashAttentionScoreTilingBase::SetMultiBatchCoreParams()
{
    auto &coreParams = tilingData.coreParams;
    coreParams.set_bBaseSize(1);
    coreParams.set_bBaseTailSize(1);
    coreParams.set_bOuterSize(bSize);

    coreParams.set_n2BaseSize(1);
    coreParams.set_n2BaseTailSize(1);
    coreParams.set_n2OuterSize(n2Size);

    coreParams.set_gBaseSize(1);
    coreParams.set_gBaseTailSize(1);
    coreParams.set_gOuterSize(gSize);
}

void FlashAttentionScoreTilingBase::SetMultiCoreParams()
{
    auto &multiCoreParams = tilingData.multiCoreParams;
    auto &coreParams = tilingData.coreParams;
    int64_t totalSize = coreParams.get_bOuterSize() * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize() *
                        coreParams.get_s1OuterSize();
    int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
    multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAivNum));
    multiCoreParams.set_totalSize(totalSize);
    multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum));
    multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
}

ge::graphStatus FlashAttentionScoreTilingBase::DoLibApiTiling()
{
    if (!SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock, batchBasic)) {
        return ge::GRAPH_FAILED;
    }
    SetSoftMaxTiling();
    SetDataCopyTransposeTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTilingBase::PostTiling()
{
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize()); // already check capcity in CheckContext
    auto blockDim = optiling::CalcTschBlockDim(tilingData.multiCoreParams.get_coreNum(), aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    auto &inputParams = tilingData.inputParams;
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    if (inputParams.get_needDropMaskOp() == 1) {
        blockDim = optiling::CalcTschBlockDim(aivNum, aicNum, aivNum);
        context_->SetBlockDim(blockDim);

        int64_t shapeTotalSize = inputParams.get_bSize() * inputParams.get_n2Size() * inputParams.get_gSize() *
                                 inputParams.get_s1Size() * inputParams.get_s2Size();
        auto layoutType = tilingData.inputParams.get_layoutType();
        if (layoutType == LAYOUT_TND) {
            for (int64_t i = 0; i < bSize; i++) {
                dropTotalSize += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            shapeTotalSize = inputParams.get_n2Size() * inputParams.get_gSize() * dropTotalSize;
        }
        shapeTotalSize = AlignUp(shapeTotalSize, GM_ALIGN);
        workspaces[0] += static_cast<size_t>(shapeTotalSize);
    }

    if (pseType == PSE_INNER_MUL_ADD_TYPE || pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
        tilingData.coreParams.set_pseAlibiBaseS1(pseAlibiBaseS1);
        tilingData.coreParams.set_pseAlibiBaseS2(pseAlibiBaseS2);
        int64_t pseAlibiBytes = AlignUp(pseAlibiBaseS2 * pseAlibiBaseS1 * 2, GM_ALIGN) *
                                tilingData.multiCoreParams.get_coreNum();
        workspaces[0] += pseAlibiBytes;
    }
    OPS_LOG_D(context_, "[%s] final workspace size:%zu, pseAlibiBaseS1:%ld, pseAlibiBaseS2:%ld.",
              templateName, workspaces[0], pseAlibiBaseS1, pseAlibiBaseS2);
    OPS_LOG_D_FULL(opName, "[%s] tiling data:%s", templateName, GetTilingDataDebugStr().c_str());
    OPS_LOG_D(context_, "[%s] tiling data size: %zu", templateName, tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTilingBase::SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch,
                                                       matmul_tiling::MatmulApiTiling &bmm1)
{
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
    bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
    bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
    bmm1.SetBias(false);
    if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
        return false;
    }
    if (bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock) != 0) {
        return false;
    }

    return true;
}

bool FlashAttentionScoreTilingBase::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                    int64_t tmpDBasicBlock, int64_t batch,
                                                    matmul_tiling::MatmulApiTiling &bmm1,
                                                    matmul_tiling::MatmulApiTiling &bmm2)
{
    if (!SetBmm1TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, batch, bmm1) ||
        !SetBmm2TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm2)) {
        return false;
    }

    if (bmm1.GetTiling(tilingData.bmm1TilingData) == -1) {
        OPS_LOG_E(context_, "BMM1 tiling failed.");
        return false;
    }
    tilingData.bmm1TilingData.set_shareMode(0);
    tilingData.bmm1TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm1TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    if (bmm2.GetTiling(tilingData.bmm2TilingData) == -1) {
        OPS_LOG_E(context_, "BMM2 tiling failed.");
        return false;
    }

    tilingData.bmm2TilingData.set_shareMode(0);
    tilingData.bmm2TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm2TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    return true;
}

bool FlashAttentionScoreTilingBase::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                    int64_t tmpDBasicBlock, int64_t batch)
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);
        matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);
        return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm1, bmm2);
    } else {
        OPS_LOG_D(context_, "platform info is null, use default info to generate matmul tiling.");
        matmul_tiling::MatmulApiTiling bmm1;
        matmul_tiling::MatmulApiTiling bmm2;
        return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm1, bmm2);
    }
}

void FlashAttentionScoreTilingBase::SetSoftMaxTiling()
{
    auto softmaxShape = ge::Shape({batchBasic, std::min(s1BasicBlock, alignedS1), std::min(s2BasicBlock, alignedS2)});

    AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                      tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
}

void FlashAttentionScoreTilingBase::SetDataCopyTransposeTiling()
{
    auto &coreParams = tilingData.coreParams;
    auto transposeSrcShape = ge::Shape({coreParams.get_bBaseSize(), 1, std::min(s1BasicBlock, alignedS1),
                                        coreParams.get_gBaseSize() * std::min(dBasicBlock, alignedD)});
    auto transposeDstShape = ge::Shape({bSize, n1Size, s1Size, n1Size * dSize});
    GetDataCopyTransposeTiling(transposeDstShape, transposeSrcShape, inputDtypeBytes, tilingData.transposeTilingData);
}

void FlashAttentionScoreTilingBase::SetTensorSizeParams()
{
    auto &tensorSizeParams = tilingData.tensorSizeParams;
    auto &coreParams = tilingData.coreParams;
    int64_t batchInnerSize = coreParams.get_bBaseSize() * coreParams.get_n2BaseSize() * coreParams.get_gBaseSize();
    tensorSizeParams.set_bmm1ResUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
    tensorSizeParams.set_attenMaskUbSize(attenMaskExistFlag * batchInnerSize * s1BasicBlock * s2BasicBlock);
    if (tilingData.inputParams.get_pseShapeType() == PSE_B_N2_G_S1_S2) {
        tensorSizeParams.set_pseUbSize(pseExistFlag * batchInnerSize * s1BasicBlock * s2BasicBlock);
    } else {
        tensorSizeParams.set_pseUbSize(pseExistFlag * batchInnerSize * s2BasicBlock); // PSE_B_N2_G_1_S2
    }

    tensorSizeParams.set_dropMaskUbSize(dropMaskExistFlag * batchInnerSize * s1BasicBlock *
                                        AlignUp(s2BasicBlock, DROP_MASK_ALIGN_UNIT) / BYTE_BIT_NUM / inputDtypeBytes);

    if (tensorSizeParams.get_pseUbSize() > 0) {
        hasPse = true;
    }
    if (tensorSizeParams.get_dropMaskUbSize() > 0) {
        hasDropOut = true;
    }
    if (tensorSizeParams.get_attenMaskUbSize() > 0) {
        hasAttenMask = true;
    }
    if (inputDtype == ge::DT_BF16 || isHighPercision) {
        if (expectTemplate.splitS2 == 1) {
            tensorSizeParams.set_castUbSize(batchInnerSize * s1BasicBlock * std::max(s2BasicBlock, dBasicBlock));
        } else {
            tensorSizeParams.set_castUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
        }
    }
    tensorSizeParams.set_softmaxMaxUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxSumUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxExpUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / calcTypeSize));
    tensorSizeParams.set_apiTmpBufferBytes(apiMaxUBSize);

    tensorSizeParams.set_bmm2ResUbSize(batchInnerSize * s1BasicBlock * dBasicBlock);
}

bool FlashAttentionScoreTilingBase::InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx)
{
    OPS_ERR_IF(sparseValidArray.size() == 0,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "Sparse valid array size should be larger than 0."), return false);
    uint8_t sparseType = tilingData.inputParams.get_sparseType();
    if (sparseType == static_cast<uint8_t>(SparseEnum::PREFIX)) {
        for (int64_t i = 0; i < static_cast<int64_t>(sparseValidArray.size()); i++) {
            int64_t s2IgnoredEndLen =
                tilingData.inputParams.get_s1Size() - tilingData.coreParams.get_s1BaseSize() * (i + 1);
            int64_t s2EndLen = 0;
            s2IgnoredEndLen = std::max(static_cast<int64_t>(0), s2IgnoredEndLen);
            if (tilingData.inputParams.get_s2Size() > s2IgnoredEndLen) {
                s2EndLen = tilingData.inputParams.get_s2Size() - s2IgnoredEndLen;
                s2EndLen = std::max(s2EndLen, prefixNData[bIdx]);
            } else {
                s2EndLen = tilingData.inputParams.get_s2Size();
                s2EndLen = std::min(s2EndLen, prefixNData[bIdx]);
            }

            s2EndLen = std::min(s2EndLen, tilingData.inputParams.get_s2Size());
            sparseValidArray[i] = CeilDivision(s2EndLen, s2BasicBlock);
        }
    } else {
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
            maxValidS2Len = std::max(sparseValidArray[i] * s1BasicBlock, maxValidS2Len);
        }
    }
    return true;
}

bool FlashAttentionScoreTilingBase::PartitionSparseData(const std::vector<int64_t> &sparseRollingArray,
                                                        int64_t sparseRollingArraySum, int64_t sparseArraySize,
                                                        int64_t loadMaxEachCore, std::vector<int64_t> &partitionResult)
{
    OPS_ERR_IF(partitionResult.size() == 0,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "partitionResult size should be larger than 0."), return false);

    OPS_ERR_IF(sparseRollingArraySum <= 0,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "sparseRollingArraySum should be larger than 0."), return false);
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

void FlashAttentionScoreTilingBase::SetPrefixSparseStartIdx(const std::vector<std::vector<int64_t>> &sparseValidArray,
                                                            MultiCoreParams &multiCoreParams)
{
    int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
    int64_t totalSize = multiCoreParams.get_totalSize(); // BN2GS1.o
    int64_t *sparseStartIdx = multiCoreParams.get_sparseStartIdx();
    for (int64_t idx = 0; idx < MAX_AIV_NUM; ++idx) {
        sparseStartIdx[idx] = totalSize;
    }
    if (totalSize <= validAivNum) {
        int64_t idx = 0;
        for (; idx < totalSize; ++idx) {
            sparseStartIdx[idx] = idx;
        }
        for (; idx < validAivNum; ++idx) {
            sparseStartIdx[idx] = totalSize;
        }
        return;
    }

    int64_t loadTotal = 0;
    /* Need to adapt when we split b. */
    for (int64_t i = 0; i < bSize; i++) {
        loadTotal += std::accumulate(sparseValidArray[i].begin(), sparseValidArray[i].end(), 0LL);
    }
    int64_t n2G = tilingData.coreParams.get_n2OuterSize() * tilingData.coreParams.get_gOuterSize();
    loadTotal *= n2G;

    auto loadEachCoreExpect = CeilDivision(loadTotal, validAivNum);
    int64_t s1OuterSize = tilingData.coreParams.get_s1OuterSize();
    int64_t tempBlock = 0;
    int64_t coreIdx = 0;
    int64_t loadStartIdx = 0;

    for (int64_t bNGS1Index = 0; bNGS1Index < bSize * n2G * s1OuterSize; ++bNGS1Index) {
        int64_t bIdx = bNGS1Index / (n2G * s1OuterSize);
        if (s1OuterSize == 0) {
            continue;
        }
        int64_t s1Idx = bNGS1Index % s1OuterSize;
        auto currBlockNum = sparseValidArray[bIdx][s1Idx];
        if (tempBlock >= loadEachCoreExpect) {
            if ((tempBlock + currBlockNum - loadEachCoreExpect) >= (loadEachCoreExpect - (tempBlock))) {
                /* 不累加当前block */
                sparseStartIdx[coreIdx++] = loadStartIdx;
                loadStartIdx = bNGS1Index;
                // 下一个核使用当前这个S1
                tempBlock = currBlockNum;
            } else {
                sparseStartIdx[coreIdx++] = loadStartIdx;
                loadStartIdx = (bNGS1Index + 1);
                tempBlock = 0;
            }
        } else {
            tempBlock += currBlockNum;
        }
    }

    if (tempBlock != 0) {
        sparseStartIdx[coreIdx++] = loadStartIdx;
    }

    /* 将没用到的核的start index置为最大值 */
    for (; coreIdx < MAX_AIV_NUM; ++coreIdx) {
        sparseStartIdx[coreIdx] = totalSize;
    }
}

bool FlashAttentionScoreTilingBase::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray,
                                                      MultiCoreParams &multiCoreParams)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
    int64_t totalSize = multiCoreParams.get_totalSize(); // BN2GS1.o
    int64_t *sparseStartIdx = multiCoreParams.get_sparseStartIdx();
    for (int64_t idx = 0; idx < MAX_AIV_NUM; ++idx) {
        sparseStartIdx[idx] = totalSize;
    }

    if (totalSize <= validAivNum) {
        for (int64_t idx = 0; idx < totalSize; ++idx) {
            sparseStartIdx[idx] = idx;
        }

        return true;
    }

    // Minimize the max load each core to find a load balance result.
    // The range of max load each core is (loadEachCoreLowerBound, loadEachCoreUpperBound].
    std::vector<int64_t> partitionResult(validAivNum, totalSize);
    std::vector<int64_t> lastValidPartitionResult(validAivNum, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
    int64_t loadTotal = sparseArraySum * (totalSize / sparseValidArray.size());
    OPS_ERR_IF(validAivNum <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "validAivNum should be larger than 0."),
               return false);
    int64_t loadEachCoreLowerBound = loadTotal / validAivNum - 1;
    int64_t loadEachCoreUpperBound =
        CeilDivision(loadTotal, validAivNum) + (*std::max_element(sparseValidArray.begin(), sparseValidArray.end()));
    while (loadEachCoreLowerBound + 1 < loadEachCoreUpperBound) {
        int64_t loadMax = loadEachCoreLowerBound + (loadEachCoreUpperBound - loadEachCoreLowerBound) / 2;
        if ((loadMax * validAivNum >= loadTotal) &&
            PartitionSparseData(sparseValidArray, sparseArraySum, totalSize, loadMax, partitionResult)) {
            loadEachCoreUpperBound = loadMax;
            lastValidPartitionResult.swap(partitionResult);
            continue;
        }
        loadEachCoreLowerBound = loadMax;
    }

    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        sparseStartIdx[idx] = lastValidPartitionResult[idx];
    }

    if (AlogCheckDebugLevel(OP, DLOG_DEBUG) == 1) {
        PrintSparseMaxMinLoadPerCore(sparseValidArray, sparseStartIdx, validAivNum,
                                     CeilDivision(loadTotal, validAivNum));
    }
    return true;
}

void FlashAttentionScoreTilingBase::PrintSparseMaxMinLoadPerCore(const std::vector<int64_t> &sparseValidArray,
                                                                 int64_t *sparseStartIdx, int32_t validAivNum,
                                                                 int64_t avgLoadSize)
{
    int64_t maxLoadSize = 0;
    int64_t minLoadSize = std::numeric_limits<int64_t>::max();
    int64_t totalSize = tilingData.multiCoreParams.get_totalSize();
    int64_t s1OuterSize = tilingData.coreParams.get_s1OuterSize();
    if (s1OuterSize == 0) {
        return;
    }
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t startIdx = sparseStartIdx[idx];
        int64_t endIdx = totalSize;
        if (idx + 1 < validAivNum) {
            endIdx = sparseStartIdx[idx + 1];
        }

        if (startIdx >= endIdx) {
            minLoadSize = 0;
            break;
        }

        int64_t s1OuterStartIdx = startIdx % s1OuterSize;
        int64_t s1OuterEndIdx = endIdx % s1OuterSize;
        int64_t loadSize = 0;
        if (s1OuterEndIdx > s1OuterStartIdx) {
            loadSize = std::accumulate(sparseValidArray.begin() + s1OuterStartIdx,
                                       sparseValidArray.begin() + s1OuterEndIdx, 0);
        } else {
            loadSize = std::accumulate(sparseValidArray.begin() + s1OuterStartIdx, sparseValidArray.end(), 0);
            loadSize = std::accumulate(sparseValidArray.begin(), sparseValidArray.begin() + s1OuterEndIdx, loadSize);
        }

        int64_t s1OuterLoop = (endIdx / s1OuterSize) - (startIdx / s1OuterSize);
        if (s1OuterLoop > 1) {
            if (s1OuterEndIdx > s1OuterStartIdx) {
                loadSize += s1OuterLoop * std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0);
            } else {
                loadSize += (s1OuterLoop - 1) * std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0);
            }
        }

        maxLoadSize = std::max(maxLoadSize, loadSize);
        minLoadSize = std::min(minLoadSize, loadSize);
    }

    OPS_LOG_D(context_, "[%s]each core load: max[%ld], min[%ld], avg[%ld]", templateName, maxLoadSize, minLoadSize,
              avgLoadSize);
}

void FlashAttentionScoreTilingBase::SetSparseParams()
{
    if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL)) {
        return;
    }

    if (expectTemplate.splitS2 == 0) {
        OPS_LOG_I(context_, "[%s]match not split S2 template, close sparse feature", templateName);
        tilingData.inputParams.set_sparseType(static_cast<uint8_t>(SparseEnum::ALL));
        return;
    }

    auto &coreParams = tilingData.coreParams;
    coreParams.set_s1SparseValidSize(s1SparseValidSize);
    coreParams.set_s2SparseValidSize(s2SparseValidSize);

    auto &multiCoreParams = tilingData.multiCoreParams;
    if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::PREFIX)) {
        std::vector<std::vector<int64_t>> sparseValidArray;
        for (int64_t bIdx = 0; bIdx < bSize; bIdx++) {
            sparseValidArray.emplace_back(std::vector<int64_t>(coreParams.get_s1OuterSize(), 0));
            InitSparseValidArray(sparseValidArray.back(), bIdx);
        }
        SetPrefixSparseStartIdx(sparseValidArray, multiCoreParams);
    } else {
        std::vector<int64_t> sparseValidArray(coreParams.get_s1OuterSize(), 0);
        InitSparseValidArray(sparseValidArray, 0);
        SetSparseStartIdx(sparseValidArray, multiCoreParams);
    }
}

ge::graphStatus FlashAttentionScoreTilingBase::GetWorkspaceSize()
{
    auto &tensorSizeParams = tilingData.tensorSizeParams;
    auto &coreParams = tilingData.coreParams;

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    int64_t bmm1Byetes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;
    int64_t bmm2Byetes = tensorSizeParams.get_bmm2ResUbSize() * calcTypeSize;
    workspaces[0] = static_cast<size_t>((bmm1Byetes + bmm2Byetes) * aivNum) + WORK_SPACE_RESERVE_SIZE;
    return ge::GRAPH_SUCCESS;
}

class FlashAttentionScoreTilingS1Bn2gs1 : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingS1Bn2gs1(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitD = 1;
        templateName = "FlashAttentionScoreS1Bn2gs1";
    }
    ~FlashAttentionScoreTilingS1Bn2gs1() override = default;

protected:
    int64_t s1Ratio = 1;
    int64_t workspaceLimit = 131072; // 8*128*128
    int64_t softmaxExtraSize = 512;
    int64_t s1dHighPerfBufferNum = 4;
    int64_t s2SizeLimitMax = 1024;
    int64_t s2SizeLimitMin = 128;
    int64_t isBasicBlockNum = 64;
    int64_t minSizeLimit = 65536; // 64 * 1024
    int64_t nRatioMax = 4;
    int64_t highPerfBlock = 128;
    int64_t l1SizeRemain = 0;
    int64_t elementSize = 4;
    int64_t nzndDataLimit = 20480; // 20 * 1024
    int64_t s2SizeNzndMinLimit = 704;
    int64_t dSizeLimit = 256;
    int64_t aicRatio = 1;
    int64_t aicRatioL1reuse = 2;
    bool enableL1Reuse = false;

    bool AnalyzeDtype() override
    {
        OPS_ERR_IF(!FlashAttentionScoreTilingBase::AnalyzeDtype(),
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze base dtype."), return false);
        bmm2OutDtype = bmmDtype;
        return true;
    }

    void SetEnableL1Reuse()
    {
        // FP32场景，不开启L1reuse
        if (inputDtypeBytes == DATA_TYPE_FP32) {
            enableL1Reuse = false;
            return;
        }
        // 使能增量L1reuse条件：s2>=512且BNG>64且D<=128
        // 增量L1reuse与原始L1reuse互斥
        if ((s2Size >= L1REUSE_S2_LIMIT_512 && bSize * n2Size * gSize > L1REUSE_BNG_LIMIT_64 &&
             dSize <= L1REUSE_D_Limit) ||
            (bSize * n2Size * gSize >= L1REUSE_BNG_LIMIT_4800 && s2Size == BMM2_BASICBLOCK_K_256 &&
             dSize == L1REUSE_D_LIMIT_144)) {
            enableL1Reuse = true;
            aicRatio = aicRatioL1reuse;
        }
        // 原始L1reuse说明
        // 因为一个Cube对应两个Vector, 一共需要两份L1空间存放Bmm2的右矩阵S2 * D
        // Nz的shape需要将s2Size对齐到16来计算剩余空间
        // 如果s2SizeALign16 * dSizeAlign16大于64K，则不使能该优化
        // L1reuse入口条件逻辑为：
        // 512<=S2<=1024且S1<=3840且D=64，并在非稀疏时开启
        if ((alignedS2 >= S2_REUSE_SIZE_512 && alignedS2 <= S2_REUSE_SIZE_1024) && s1Size <= S1_REUSE_SIZE_3840 &&
            alignedD == D_SPECIFIC_SIZE && (tilingData.inputParams.get_sparseType() == 0)) {
            tilingKeyBmm2Source = CubeInputSourceEnum::L1;
            enableL1Reuse = false;
            aicRatio = 1;
            if (alignedS2 == S2_REUSE_SIZE_512) {
                nRatioMax = 1;
            }
        }
    }

    void SetMultiCoreParams() override
    {
        auto &multiCoreParams = tilingData.multiCoreParams;
        auto &coreParams = tilingData.coreParams;
        int64_t totalSize = coreParams.get_bOuterSize() * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize() *
                            coreParams.get_s1OuterSize();
        int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
        multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAivNum));
        multiCoreParams.set_totalSize(totalSize);
        multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum) * aicRatio);
        multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
    }

    void SetCoreParams() override
    {
        SetEnableL1Reuse();
        // 稀疏场景不开启S1轴N:1配比
        FlashAttentionScoreTilingBase::SetCoreParams();
        if (tilingData.inputParams.get_sparseType() != static_cast<uint8_t>(SparseEnum::ALL)) {
            return;
        }
        SetMultiCoreParams();
        auto &coreParams = tilingData.coreParams;
        auto &multiCoreParams = tilingData.multiCoreParams;
        // 对于S2 < 128且S1 16对齐的场景，bmm1输出改成NZ格式，配比设为4，提升fix pipe效率
        if (alignedS2 < s2SizeLimitMin) {
            nRatioMax = 1;
        }
        // NZND入口条件逻辑为
        // 1、S2=64时开启；
        // 2、S2非64对齐时开启；
        // 3、S2大于s2SizeNzndMinLimit，且64对齐但非128对齐，BNGS1数据量大于nzndDataLimit时开启；
        // 4、满足以上条件且D>256时，开启NZND但不改变N配比；
        if ((s2Size % S2_NZTOND_SIZE_64 != 0 || s2Size == S2_NZTOND_SIZE_64) ||
            (s2Size >= s2SizeNzndMinLimit && s2Size % S2_NZTOND_SIZE_64 == 0 && s2Size % S2_NZTOND_SIZE_128 != 0 &&
             bSize * n2Size * gSize * s1Size > nzndDataLimit)) {
            if (dSize <= dSizeLimit) {
                nRatioMax = 4;
            }
            tilingKeyBmm1Format = CubeFormatEnum::NZ;
        }
        // 当前能分满核，考虑增大N
        while (s1Ratio < nRatioMax && multiCoreParams.get_totalSize() > ((GetNRatio() - 1) * aivNum / GetNRatio()) &&
               s1BasicBlock * GetNRatio() < alignedS1 && GetNRatio() * s1BasicBlock * alignedS2 <= workspaceLimit) {
            s1Ratio++;
            FlashAttentionScoreTilingBase::SetCoreParams();
            // S1轴N:1使能
            coreParams.set_s1OuterSize(CeilDivision(coreParams.get_s1OuterSize(), GetNRatio()));
            coreParams.set_s1BaseSize(s1BasicBlock * GetNRatio());
            coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock * GetNRatio()));
            SetMultiCoreParams();
        }
        // 分不满核，减小N
        while (multiCoreParams.get_totalSize() <= ((GetNRatio() - 1) * aivNum / GetNRatio()) ||
               s1BasicBlock * GetNRatio() > alignedS1 || GetNRatio() * s1BasicBlock * alignedS2 > workspaceLimit) {
            s1Ratio--;
            FlashAttentionScoreTilingBase::SetCoreParams();
            coreParams.set_s1OuterSize(CeilDivision(coreParams.get_s1OuterSize(), GetNRatio()));
            coreParams.set_s1BaseSize(s1BasicBlock * GetNRatio());
            coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock * GetNRatio()));
        }
    }

    void SetSparseParams() override
    {
        if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL)) {
            return;
        }
        auto &coreParams = tilingData.coreParams;
        coreParams.set_s1SparseValidSize(s1SparseValidSize);
        coreParams.set_s2SparseValidSize(s2SparseValidSize);
        auto &multiCoreParams = tilingData.multiCoreParams;
        if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::PREFIX)) {
            std::vector<std::vector<int64_t>> sparseValidArray;
            for (int64_t bIdx = 0; bIdx < bSize; bIdx++) {
                sparseValidArray.emplace_back(std::vector<int64_t>(coreParams.get_s1OuterSize(), 0));
                InitSparseValidArray(sparseValidArray.back(), bIdx);
            }
            SetPrefixSparseStartIdx(sparseValidArray, multiCoreParams);
        } else {
            std::vector<int64_t> sparseValidArray(coreParams.get_s1OuterSize(), 0);
            InitSparseValidArray(sparseValidArray, 0);
            SetSparseStartIdx(sparseValidArray, multiCoreParams);
        }
    }

    int64_t GetNRatio() override
    {
        return s1Ratio;
    }

    bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch) override
    {
        apiMaxUBSize = tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float) + softmaxExtraSize;
        return true;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = s1dHighPerfBufferNum;
    }

    void CalcS1S2BasicBlock(const BufferNum &bufferNum) override
    {
        s1BasicBlock = std::min(128L, alignedS1); // PERFORMANCE OPT
        s2BasicBlock = std::min(128L, alignedS2);
    }

    void CalcDBasicBlock() override
    {
        dBasicBlock = std::min(128L, alignedD);
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        bmm1.SetShape(std::min(static_cast<int64_t>(tilingData.coreParams.get_s1BaseSize()), s1Size), s2Size, dSize);
        bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if (bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock) != 0) {
            return false;
        }
        if (tilingKeyBmm2Source == CubeInputSourceEnum::L1) {
            l1SizeRemain = aicoreParams_.l1Size - alignedS2 * alignedD * elementSize;
        } else {
            l1SizeRemain = aicoreParams_.l1Size;
        }
        if (bmm1.SetBufferSpace(l1SizeRemain, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(std::min(static_cast<int64_t>(tilingData.coreParams.get_s1BaseSize()), s1Size), dSize, s2Size);
        bmm2.SetOrgShape(s1Size, s2StrideSize, s2Size, s2StrideSize);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(l1SizeRemain, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        // 在S1=S2，S2大于S2_SPECIFIC_SIZE_928且D=64时，使用性能更亲和的BMM2基本块
        if (s1Size == s2Size && s2Size >= S2_SPECIFIC_SIZE_928 && dSize == D_SPECIFIC_SIZE &&
            tilingData.inputParams.get_sparseType() == 0) {
            if (bmm2.SetFixSplit(BMM2_BASICBLOCK_M_64, BMM2_BASICBLOCK_N_64, BMM2_BASICBLOCK_K_256) != 0) {
                return false;
            }
        } else {
            if (bmm2.SetFixSplit(tmpS1BasicBlock, tmpDBasicBlock) != 0) {
                return false;
            }
        }
        return true;
    }

    bool IsTemplateMatched() const override
    {
        if (s2Size > s2SizeLimitMax) {
            return false;
        }
        if (s2Size > s2SizeLimitMin) {
            return true;
        }
        if (static_cast<uint64_t>(n2Size * gSize * ((alignedS1 + alignedS2) * dSize + alignedS2)
            * inputDtypeBytes) >= aicoreParams_.l1Size ||
            static_cast<uint64_t>(n2Size * gSize * (alignedS1 + dSize) * alignedS2
            * inputDtypeBytes) >= aicoreParams_.l1Size) {
            return true;
        }
        if (n2Size * gSize * alignedS1 * alignedS2 * inputDtypeBytes <= minSizeLimit * DATA_TYPE_FP16) {
            return false;
        }
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        return GET_TILINGKEY(AxisEnum::S1, AxisEnum::D, AxisEnum::NONE, implMode, tilingKeyDType, tilingKeyLayout,
                             tilingKeyBmm1Format, tilingKeyBmm2Source, SparseEnum::ANY,
                             PerformanceOrientedEnum::BIG_DOUBLE_BUFFER, hasDropOut, hasAttenMask, hasPse,
                             enableL1Reuse);
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        if (tilingData.inputParams.get_sparseType() == 0) {
            int32_t actualUsedAivNum = CeilDivision(tilingData.multiCoreParams.get_totalSize(),
                                                    (tilingData.multiCoreParams.get_splitFactorSize() / aicRatio));
            int32_t actualUsedAivNumMod2 = actualUsedAivNum % 2;
            if (enableL1Reuse && actualUsedAivNumMod2) {
                actualUsedAivNum++;
            }
            tilingData.multiCoreParams.set_coreNum(std::min(int32_t(aivNum), actualUsedAivNum));
        }
        tilingData.bmm1TilingData.set_shareL1Size(l1SizeRemain);
        tilingData.bmm2TilingData.set_shareL1Size(l1SizeRemain);
        auto &coreParams = tilingData.coreParams;
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Size = 0;
        int64_t bmm1AlignBytes = 0;
        int64_t s2SizeAlign16 = CeilDivision(s2Size, 16L) * 16L;
        bmm1Size = CeilDivision(coreParams.get_s1BaseSize() * s2SizeAlign16, 256L) * 256L;
        bmm1AlignBytes = bmm1Size * calcTypeSize * 2;
        // bmm1和stage1的workspace不能复用
        int64_t stage1AlignBytes = bmm1AlignBytes * inputDtypeBytes / DATA_TYPE_FP32;

        /* 计算bmm2需要用的workspace, 大小为CoreNum * s1BaseSize * alignedD (16对齐）,
         * bmm2计算完成后将数据输出在这块workspace上。
         * 这块workspace主要的作用是存放bmm2的后继输出，用来做div softmax sum和cast。 */
        int64_t bmm2AlignBytes = CeilDivision(coreParams.get_s1BaseSize() * alignedD, 256L) * 256L * calcTypeSize * 2;
        workspaces[0] = static_cast<size_t>((bmm1AlignBytes + stage1AlignBytes + bmm2AlignBytes) *
                                            tilingData.multiCoreParams.get_coreNum()) +
                        WORK_SPACE_RESERVE_SIZE;
        if (pseType == PSE_INNER_MUL_ADD_TYPE || pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
            pseAlibiBaseS2 = alignedS2;
            pseAlibiBaseS1 = std::min(static_cast<int64_t>(coreParams.get_s1BaseSize()),
                                      UB_BASIC_LIMIT_SIZE / pseAlibiBaseS2);
            pseAlibiBaseS1 = std::max(pseAlibiBaseS1, UB_BASIC_LIMIT_SIZE / coreParams.get_s1BaseSize());
        }
        return ge::GRAPH_SUCCESS;
    }
};

class FlashAttentionScoreTilingB : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingB(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        templateName = "FlashAttentionScoreB";
    }
    ~FlashAttentionScoreTilingB() override = default;

protected:
    int64_t blockBSizeLimit_ = 64 * 1024;
    int64_t blockBL2SizeLimit_ = 128 * 1024;
    int64_t blockBUBSizeLimit_ = 8 * 1024;
    int64_t maxS1BaseSize_ = 256;
    int64_t dVec2BasicBlock_ = 1;
    int64_t s1Vec2BasicBlock_ = 1;

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock(const BufferNum &bufferNum) override
    {
        s2BasicBlock = alignedS2;
        s1BasicBlock = blockBUBSizeLimit_ / s2BasicBlock / FRACTAL_NUM * FRACTAL_NUM;
        s1BasicBlock = std::min(s1BasicBlock, alignedS1);
        s1BasicBlock = std::min(maxS1BaseSize_, s1BasicBlock);
        dVec2BasicBlock_ = alignedD;
        s1Vec2BasicBlock_ = blockBUBSizeLimit_ / dVec2BasicBlock_ / FRACTAL_NUM * FRACTAL_NUM *
                            DATA_TYPE_FP16 / inputDtypeBytes;
        s1Vec2BasicBlock_ = std::min(s1Vec2BasicBlock_, alignedS1);
    }

    void SetCoreParams() override
    {
        auto &coreParams = tilingData.coreParams;
        auto &inputParams = tilingData.inputParams;
        int64_t n2 = inputParams.get_n2Size();
        int64_t g = inputParams.get_gSize();
        int64_t b = inputParams.get_bSize();
        int64_t s1 = inputParams.get_s1Size();
        int64_t bIn = 1;
        coreParams.set_bBaseSize(bIn);
        coreParams.set_bBaseTailSize(CalcTailSize(b, bIn));
        coreParams.set_bOuterSize(CeilDivision(b, bIn));
        coreParams.set_s1BaseSize(s1BasicBlock);
        coreParams.set_s1BaseTailSize(CalcTailSize(s1, s1BasicBlock));
        coreParams.set_s1OuterSize(CeilDivision(s1, s1BasicBlock));
        coreParams.set_s2BaseSize(s2BasicBlock);
        coreParams.set_s2BaseTailSize(CalcTailSize(s2Size, s2BasicBlock));
        coreParams.set_s2OuterSize(CeilDivision(s2Size, s2BasicBlock));
        coreParams.set_s1Vec2BaseSize(s1Vec2BasicBlock_);
        coreParams.set_s1Vec2BaseTailSize(CalcTailSize(s1, s1Vec2BasicBlock_));
        coreParams.set_s1Vec2OuterSize(CeilDivision(s1, s1Vec2BasicBlock_));
        coreParams.set_dBaseSize(dBasicBlock);
        coreParams.set_dBaseTailSize(CalcTailSize(dSize, dBasicBlock));
        coreParams.set_dOuterSize(CeilDivision(dSize, dBasicBlock));
        coreParams.set_s1SparseValidSize(s1SparseValidSize);
        coreParams.set_s2SparseValidSize(s2SparseValidSize);
        batchBasic = coreParams.get_bBaseSize() * n2 * g;
        OPS_LOG_D(context_, "[b:%ld, n2:%ld, g:%ld, s1:%ld, s2:%ld, batchBasic:%ld].", b, n2, g, s1,
                  inputParams.get_s2Size(), batchBasic);
        OPS_LOG_D(context_, "[bBaseSize:%d, bBaseTailSize:%d, bOuterSize:%ld].", coreParams.get_bBaseSize(),
                  coreParams.get_bBaseTailSize(), coreParams.get_bOuterSize());
        OPS_LOG_D(context_, "[s1BaseSize:%d, s1BaseTailSize:%d, s1OuterSize:%ld].", coreParams.get_s1BaseSize(),
                  coreParams.get_s1BaseTailSize(), coreParams.get_s1OuterSize());
        OPS_LOG_D(context_, "[s1Vec2BaseSize:%d, s1Vec2BaseTailSize:%d, s1Vec2OuterSize:%ld].",
                  coreParams.get_s1Vec2BaseSize(), coreParams.get_s1Vec2BaseTailSize(),
                  coreParams.get_s1Vec2OuterSize());
        OPS_LOG_D(context_, "[s2BaseSize:%d, s2BaseTailSize:%d, s2OuterSize: %ld].", coreParams.get_s2BaseSize(),
                  coreParams.get_s2BaseTailSize(), coreParams.get_s2OuterSize());
    }

    void SetMultiCoreParams() override
    {
        auto &multiCoreParams = tilingData.multiCoreParams;
        auto &coreParams = tilingData.coreParams;
        int64_t totalSize = coreParams.get_bOuterSize(); // 核间一共处理的Bo大小
        int64_t tempUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
        multiCoreParams.set_totalSize(totalSize);
        multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, tempUsedAivNum)); // 每个核处理的Bo大小
        multiCoreParams.set_splitFactorTailSize(
            CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize())); // 最后一个核处理的Bo大小
        multiCoreParams.set_coreNum(
            static_cast<int32_t>(CeilDivision(totalSize, multiCoreParams.get_splitFactorSize())));
        OPS_LOG_D(context_,
                  "[totalSize:%ld, tempUsedAivNum:%ld, splitFactorSize:%ld, splitFactorTailSize:%ld, coreNum:%d].",
                  totalSize, tempUsedAivNum, multiCoreParams.get_splitFactorSize(),
                  multiCoreParams.get_splitFactorTailSize(), multiCoreParams.get_coreNum());
    }

    void SetTensorSizeParams() override
    {
        auto &tensorSizeParams = tilingData.tensorSizeParams;
        auto &coreParams = tilingData.coreParams;
        tensorSizeParams.set_bmm1ResUbSize(s1BasicBlock * s2BasicBlock);
        tensorSizeParams.set_attenMaskUbSize(attenMaskExistFlag * s1BasicBlock * s2BasicBlock);
        tensorSizeParams.set_pseUbSize(pseExistFlag * s1BasicBlock * s2BasicBlock);
        tensorSizeParams.set_dropMaskUbSize(dropMaskExistFlag * s1BasicBlock *
                                            AlignUp(s2BasicBlock, DROP_MASK_ALIGN_UNIT) / BYTE_BIT_NUM /
                                            inputDtypeBytes);
        if (tensorSizeParams.get_pseUbSize() > 0) {
            hasPse = true;
        }
        if (tensorSizeParams.get_dropMaskUbSize() > 0) {
            hasDropOut = true;
        }
        if (tensorSizeParams.get_attenMaskUbSize() > 0) {
            hasAttenMask = true;
        }
        if (inputDtype == ge::DT_BF16 || isHighPercision) {
            if (expectTemplate.splitS2 == 1) {
                tensorSizeParams.set_castUbSize(s1BasicBlock * std::max(s2BasicBlock, dBasicBlock));
            } else {
                tensorSizeParams.set_castUbSize(s1BasicBlock * s2BasicBlock);
            }
        }
        tensorSizeParams.set_softmaxMaxUbSize(s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
        tensorSizeParams.set_softmaxSumUbSize(s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
        tensorSizeParams.set_softmaxExpUbSize(s1BasicBlock * (BYTE_BLOCK / calcTypeSize));
        tensorSizeParams.set_apiTmpBufferBytes(apiMaxUBSize);
        tensorSizeParams.set_bmm2ResUbSize(coreParams.get_s1Vec2BaseSize() * dBasicBlock);
    }

    void CalcDBasicBlock() override
    {
        dBasicBlock = alignedD;
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        bmm1.SetShape(s1Size, s2Size, dSize);
        bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
        bmm1.SetALayout(bSize, s1Size, n2Size, gSize, dSize);
        bmm1.SetBLayout(bSize, s2Size, n2Size, 1, dSize);
        bmm1.SetCLayout(bSize, s1Size, n2Size, gSize, s2Size);
        bmm1.SetBatchNum(batch);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if (bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock) != 0) {
            return false;
        }
        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(s1Size, dSize, s2Size);
        bmm2.SetOrgShape(s1Size, dSize, s2Size); // consider broadcst, N same as A tensor
        bmm2.SetALayout(bSize, s1Size, n2Size, gSize, s2Size);
        bmm2.SetBLayout(bSize, s2Size, n2Size, 1, dSize);
        bmm2.SetCLayout(bSize, s1Size, n2Size, gSize, dSize);
        bmm2.SetBatchNum(batch);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        int64_t maxDBasicBlock = AlignDown(aicoreParams_.l0cSize / (tmpS1BasicBlock * calcTypeSize), 16UL);
        if (bmm2.SetFixSplit(tmpS1BasicBlock, std::min(maxDBasicBlock, tmpDBasicBlock)) != 0) {
            return false;
        }
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        return GET_TILINGKEY(AxisEnum::NONE, AxisEnum::NONE, AxisEnum::B, implMode, tilingKeyDType, tilingKeyLayout,
                             SparseEnum::NONE, PerformanceOrientedEnum::BIG_DOUBLE_BUFFER, hasDropOut, hasAttenMask,
                             hasPse);
    }

    bool IsCapable() override
    {
        auto &inputParams = tilingData.inputParams;
        int64_t n2 = inputParams.get_n2Size();
        int64_t g = inputParams.get_gSize();
        bool notMatched = false;
        if (alignedS2 > HIGH_PERF_SUPPORT_S2_BASIC) {
            notMatched = true;
        }
        if (n2 * g * alignedS1 * alignedS2 * inputDtypeBytes > blockBSizeLimit_ * DATA_TYPE_FP16) {
            notMatched = true;
        }
        if (notMatched) {
            OPS_LOG_E(context_,
                      "[%s]not match template[%s], input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], "
                      "keepProb[%f]",
                      templateName, expectTemplate.ToString().c_str(), inputParams.get_bSize(),
                      inputParams.get_n2Size(), inputParams.get_gSize(), inputParams.get_s1Size(),
                      inputParams.get_s2Size(), inputParams.get_dSize(), inputParams.get_keepProb());
            return false;
        }
        return true;
    }

    bool IsTemplateMatched() const override
    {
        return true;
    }

    bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch) override
    {
        apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float);
        return true;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto &inputParams = tilingData.inputParams;
        auto &coreParams = tilingData.coreParams;
        auto &multiCoreParams = tilingData.multiCoreParams;
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Byetes = coreParams.get_bBaseSize() * inputParams.get_n2Size() * inputParams.get_gSize() *
                             inputParams.get_s1Size() * alignedS2 * calcTypeSize * inputDtypeBytes / DATA_TYPE_FP16;
        int64_t bmm2Byetes = coreParams.get_bBaseSize() * inputParams.get_n2Size() * inputParams.get_gSize() *
                             inputParams.get_s1Size() * alignedD * calcTypeSize;
        bmm1Byetes = AlignUp(bmm1Byetes, GM_ALIGN);
        bmm2Byetes = AlignUp(bmm2Byetes, GM_ALIGN);
        size_t pingPongNum = 2;
        workspaces[0] = WORK_SPACE_RESERVE_SIZE +
                        static_cast<size_t>((bmm1Byetes + bmm2Byetes) * pingPongNum * multiCoreParams.get_coreNum());

        if (pseType == PSE_INNER_MUL_ADD_TYPE || pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
            pseAlibiBaseS2 = alignedS2;
            pseAlibiBaseS1 = std::min(s1BasicBlock, UB_BASIC_LIMIT_SIZE / pseAlibiBaseS2);
            pseAlibiBaseS1 = std::max(pseAlibiBaseS1, UB_BASIC_LIMIT_SIZE / s1BasicBlock);
        }
        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = ge::Shape({s1BasicBlock, s2BasicBlock});
        AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                          tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
    }
};

class FlashAttentionScoreTilingS1s2Bn2gs1 : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingS1s2Bn2gs1(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitS2 = 1;
        templateName = "FlashAttentionScoreS1s2Bn2gs1";
    }
    ~FlashAttentionScoreTilingS1s2Bn2gs1() override = default;

protected:
    int64_t s2sizeLimitMin = 1024;
    int64_t dAlignSize = 16;
    bool enableL1Reuse = false;

    int64_t GetNRatio() override
    {
        return 8L;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock(const BufferNum &bufferNum) override
    {
        s1BasicBlock = std::min(64L, alignedS1);
        // d轴为64
        if (bSize * n1Size * gSize * CeilDiv(s1Size, s1BasicBlock) > aivNum) {
            s1BasicBlock = std::min(128L, alignedS1);
        }
        s2BasicBlock = std::min(128L, alignedS2);
        if (s2Size % S2_NZTOND_SIZE_64 != 0) {
            tilingKeyBmm1Format = CubeFormatEnum::NZ;
        }
    }

    void CalcDBasicBlock() override
    {
        dBasicBlock = std::min(128L, alignedD);
    }

    bool IsSpecialShape()
    {
        return bSize == 8 && n1Size == 32 && n2Size == 32 && s1Size == 2048 && s2Size == 2048 && dSize == 128 &&
               preTokens == 2048 && nextTokens == 0 && inputLayout[0] == 'S' && inputLayout[1] == 'B' &&
               inputLayout[2] == 'H' && pseExistFlag == 0 && attenMaskExistFlag == 1 &&
               tilingData.inputParams.get_attenMaskShapeType() == ATTEN_1_1_1_S1_S2;
    }

    void SetEnableL1Reuse()
    {
        // FP32场景，不开启L1reuse
        if (inputDtypeBytes == DATA_TYPE_FP32) {
            enableL1Reuse = false;
            return;
        }
        if (dSize > L1REUSE_D_Limit) {
            OPS_LOG_D(context_, "Current condition [dSize(%ld) > L1REUSE_D_Limit(%ld)] does not enable L1Reuse", dSize,
                      L1REUSE_D_Limit);
            return;
        }
        if (dSize == D_SPECIFIC_SIZE && tilingData.inputParams.get_layoutType() == LAYOUT_BNSD &&
            !(s2Size % L1REUSE_S2_LIMIT_256 == 0 || s2Size == L1REUSE_S2_LIMIT_4032)) {
            OPS_LOG_D(context_, "Current condition [dSize(%ld) && layout(BNSD)] does not enable L1Reuse", dSize);
            return;
        }
        if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL)) {
            enableL1Reuse = true;
            return;
        }

        if ((tilingData.inputParams.get_layoutType() == LAYOUT_BSND || tilingData.inputParams.get_layoutType() ==
            LAYOUT_BSH) && s2Size <= L1REUSE_S2_Limit_2048 && dSize <= D_SPECIFIC_SIZE &&
            bSize * n1Size <= L1REUSE_BNG_Limit) {
            OPS_LOG_D(context_, "Current condition [dSize(%ld) && layout(BSH/BSND) && BN(%ld)] does not enable L1Reuse",
                      dSize, bSize * n1Size);
            return;
        }

        if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::CAUSAL) ||
            tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::PREFIX)) {
            if (bSize * n1Size * gSize <= L1REUSE_BNG_Limit && s2Size <= L1REUSE_S2_Limit_2048) {
                OPS_LOG_D(context_, "Current condition [BNG(%ld) && s2Size(%ld)] does not enable L1Reuse.",
                          bSize * n1Size * gSize, s2Size);
                return;
            }
            enableL1Reuse = true;
            return;
        }

        if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::BAND)) {
            if ((bSize * n1Size * gSize <= L1REUSE_BNG_Limit && maxValidS2Len <= L1REUSE_S2_Limit_2048) ||
                (maxValidS2Len <= L1REUSE_S2_Limit_1024)) {
                OPS_LOG_D(context_, "Current condition [BNG(%ld) && maxValidS2Len(%ld)] does not enable L1Reuse.",
                          bSize * n1Size * gSize, maxValidS2Len);
                return;
            }
            enableL1Reuse = true;
        }
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        // 分不满核，且稀疏场景，shape设置的较小能产生更好的tiling
        bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size),
                      std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size), dSize);
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if (dSize > BMM1_BASICBLOCK_K_64 && dSize <= BMM1_BASICBLOCK_K_128 && inputDtypeBytes != DATA_TYPE_FP32) {
            int64_t baseM = std::min(tmpS1BasicBlock, AlignUp(s1Size, FRACTAL_NUM));
            bmm1.SetFixSplit(baseM, BMM1_BASICBLOCK_N_128, dSize);
        }

        if (IsSpecialShape()) {
            if (bmm1.SetFixSplit(BMM1_BASICBLOCK_M_128, BMM1_BASICBLOCK_N_256, BMM1_BASICBLOCK_K_64) != 0) {
                return false;
            }
        }
        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        int64_t singleM = std::min(tmpS1BasicBlock, s1Size);
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(singleM, dSize,
                      std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size));
        bmm2.SetOrgShape(s1Size, s2StrideSize, std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size),
                         s2StrideSize);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if (dSize == D_SPECIFIC_SIZE && tilingData.inputParams.get_layoutType() == LAYOUT_BNSD &&
            tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL) &&
            singleM >= BMM2_BASICBLOCK_M_64) {
            if (bmm2.SetFixSplit(BMM2_BASICBLOCK_M_64, BMM2_BASICBLOCK_M_64, BMM2_BASICBLOCK_K_256) != 0) {
                return false;
            }
        }
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        // not care about layout in tiling key, pass BSND(enum value is 0)
        return GET_TILINGKEY(AxisEnum::S1, AxisEnum::S2, AxisEnum::NONE, implMode, tilingKeyDType, tilingKeyLayout,
                             tilingKeyBmm1Format, SparseEnum::ANY, PerformanceOrientedEnum::BIG_DOUBLE_BUFFER,
                             hasDropOut, hasAttenMask, hasPse, enableL1Reuse);
    }

    bool IsCapable() override
    {
        if (s2Size > s2sizeLimitMin) {
            return true;
        }
        return false;
    }

    bool IsTemplateMatched() const override
    {
        return true;
    }

    bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch) override
    {
        apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float);
        return true;
    }

    void RefreshSplitFactor()
    {
        SetEnableL1Reuse();
        if (enableL1Reuse) {
            auto &multiCoreParams = tilingData.multiCoreParams;
            int64_t totalSize = multiCoreParams.get_totalSize();
            multiCoreParams.set_splitFactorSize(
                CeilDivision(totalSize, static_cast<int64_t>(multiCoreParams.get_coreNum())) * AICAIV_RATIO_2);
            multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
        }
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        RefreshSplitFactor();

        auto &tensorSizeParams = tilingData.tensorSizeParams;
        auto &coreParams = tilingData.coreParams;

        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Bytes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;

        // dSize小于64的场景，无需切D， workspace占用较小
        if (dSize <= D_SPECIFIC_SIZE) {
            // stage1占用2倍的空间，stage2占用2倍空间
            workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_2 +
                            SPACE_NUM_2 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                            WORK_SPACE_RESERVE_SIZE;
            // NZND场景，stage1占用3倍的空间，stage2占用2倍空间
            if (s2Size % S2_NZTOND_SIZE_64 != 0) {
                workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_3 +
                                SPACE_NUM_2 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                                WORK_SPACE_RESERVE_SIZE;
            }
            // FP32场景，stage1占用4倍的空间，stage2占用2倍空间
            if (inputDtypeBytes == DATA_TYPE_FP32) {
                workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_4 +
                                SPACE_NUM_2 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                                WORK_SPACE_RESERVE_SIZE;
            }
        } else {
            // 切D场景，stage1占用2倍的空间，stage2占用4倍空间
            workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_2 +
                            SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                            WORK_SPACE_RESERVE_SIZE;
            // NZND场景，stage1占用3倍的空间，stage2占用4倍空间
            if (s2Size % S2_NZTOND_SIZE_64 != 0) {
                workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_3 +
                                SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                                WORK_SPACE_RESERVE_SIZE;
            }
            // FP32场景，stage1占用4倍的空间，stage2占用4倍空间
            if (inputDtypeBytes == DATA_TYPE_FP32) {
                workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_4 +
                                SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                                WORK_SPACE_RESERVE_SIZE;
            }
        }
        if (pseType == PSE_INNER_MUL_ADD_TYPE || pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
            pseAlibiBaseS2 = s2sizeLimitMin;
            int64_t s2Tail = s2Size % s2sizeLimitMin;
            if (s2Tail != 0) {
                pseAlibiBaseS1 = std::min(s1BasicBlock, UB_BASIC_LIMIT_SIZE / AlignUp(s2Tail, FRACTAL_NUM));
            } else {
                pseAlibiBaseS1 = std::min(s1BasicBlock, UB_BASIC_LIMIT_SIZE / pseAlibiBaseS2);
            }
            pseAlibiBaseS1 = std::max(pseAlibiBaseS1, UB_BASIC_LIMIT_SIZE / s1BasicBlock);
        }

        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = ge::Shape({s1BasicBlock / GetNRatio(), s2BasicBlock * GetNRatio()});

        AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                          tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
    }

    bool SetPseAlibiParams() override
    {
        auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
        if (pseShape == nullptr) {
            return true;
        }
        if (pseType == PSE_INNER_MUL_ADD_TYPE || pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
            return true;
        }
        // 2: pre last axiss
        auto pseS1Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 2);
        auto pseS2Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 1);

        PseEncodeType pseEncodeType = PES_ENCODE_NONE;
        if (pseS1Size == PSE_ALIBI_S_SIZE && s1Size > PSE_ALIBI_S_SIZE) {
            if (s1Size == s2Size) {
                OPS_ERR_IF(tilingData.inputParams.get_sparseType() != static_cast<uint8_t>(SparseEnum::CAUSAL),
                           OPS_REPORT_VECTOR_INNER_ERR(opName, "Pse alibi only support causal sparse type."), return false);
                pseEncodeType = PSE_ENCODE_ALIBI_S2_FULL;
            } else {
                OPS_REPORT_VECTOR_INNER_ERR(opName, "Pse alibi only support same S1 S2 when S1 lager than 1024");
                return false;
            }
        }
        tilingData.inputParams.set_pseEncodeType(pseEncodeType);
        tilingData.inputParams.set_pseS1Size(pseS1Size);
        tilingData.inputParams.set_pseS2Size(pseS2Size);
        return true;
    }
};

class FlashAttentionVarLenScoreTiling : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionVarLenScoreTiling(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitS2 = 1;
        templateName = "FlashAttentionVarLenScore";
    }
    ~FlashAttentionVarLenScoreTiling() override = default;

protected:
    int64_t s2sizeLimitMax = 1024;

    int64_t GetNRatio() override
    {
        return 8L;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock(const BufferNum &bufferNum) override
    {
        s1BasicBlock = std::min(128L, alignedS1);
        s2BasicBlock = std::min(128L, alignedS2);
    }

    void CalcDBasicBlock() override
    {
        dBasicBlock = std::min(128L, alignedD);
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        // 分不满核，且稀疏场景，shape设置的较小能产生更好的tiling
        if (bSize * n1Size * gSize * CeilDiv(s1Size, s1BasicBlock) <= aivNum && attenMaskExistFlag == 1) {
            bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
        } else {
            bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size),
                          std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size), dSize);
        }
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }

        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(std::min(tmpS1BasicBlock, s1Size), dSize,
                      std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size));
        bmm2.SetOrgShape(s1Size, s2StrideSize, std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size),
                         s2StrideSize);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        // not care about layout in tiling key, pass BSND(enum value is 0)
        return GET_TILINGKEY(AxisEnum::S1, AxisEnum::S2, AxisEnum::NONE, implMode, tilingKeyDType, tilingKeyLayout,
                             SparseEnum::ANY, PerformanceOrientedEnum::BIG_DOUBLE_BUFFER, hasDropOut, hasAttenMask,
                             hasPse);
    }

    bool SetPseAlibiParams() override
    {
        OPS_LOG_D(context_, "Set pseAlibiParams begin.");
        auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
        if (pseShape == nullptr) {
            OPS_LOG_D(context_, "SetPseAlibiParams pseShape == nullptr.");
            return true;
        }
        if (pseType == PSE_INNER_MUL_ADD_TYPE || pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
            OPS_ERR_IF(tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::RIGHT_DOWN_CAUSAL_BAND),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "INNER Pse alibi only support BAND_LEFT_UP_CAUSAL sparse type."), return false);
            if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::BAND_LEFT_UP_CAUSAL)) {
                for (int64_t i = 0L; i < bSize; ++i) {
                    if (i == 0) {
                        if (actualSeqLenData[i] - actualSeqLenKvData[i] + qStartIdx - kvStartIdx == 0) {
                            continue;
                        } else {
                            OPS_LOG_E(context_, "INNER Pse alibi only support when actualSeqQLen and actualSeqKvLen are equal.");
                            return false;
                        }
                    }
                    if (actualSeqLenData[i] != actualSeqLenKvData[i]) {
                        OPS_LOG_E(context_, "INNER Pse alibi only support when actualSeqQLen and actualSeqKvLen are equal.");
                        return false;
                    }
                }
            }
            return true;
        }
        // 2: pre last axiss
        auto pseS1Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 2);
        auto pseS2Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 1);

        PseEncodeType pseEncodeType = PES_ENCODE_NONE;
        OPS_LOG_D(context_, "[%s] pseS1Size:%ld, pseS2Size:%ld.", templateName, pseS1Size, pseS2Size);
        if (pseS1Size == PSE_ALIBI_S_SIZE) {
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] != actualSeqLenKvData[i]) {
                    OPS_LOG_E(context_, "Pse alibi only support when actualSeqQLen and actualSeqKvLen are equal.");
                    return false;
                }
            }
            OPS_ERR_IF(tilingData.inputParams.get_sparseType() != static_cast<uint8_t>(SparseEnum::CAUSAL) &&
                           tilingData.inputParams.get_sparseType() !=
                               static_cast<uint8_t>(SparseEnum::RIGHT_DOWN_CAUSAL),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "Pse alibi only support causal sparse type."), return false);
            pseEncodeType = PSE_ENCODE_ALIBI_S2_FULL;
            OPS_LOG_D(context_, "[%s] PSE_ENCODE_ALIBI_S2_FULL.", templateName);
        }
        tilingData.inputParams.set_pseEncodeType(pseEncodeType);
        tilingData.inputParams.set_pseS1Size(pseS1Size);
        tilingData.inputParams.set_pseS2Size(pseS2Size);
        return true;
    }

    bool IsCapable() override
    {
        if (tilingKeyLayout != LayoutType::LAYOUT_TND) {
            return false;
        }
        return true;
    }
    void SetMultiCoreParams() override
    {
        auto &multiCoreParams = tilingData.multiCoreParams;
        auto &coreParams = tilingData.coreParams;
        accumS1BlockNum = 0;
        for (int64_t i = 0; i < bSize; i++) {
            OPS_LOG_D(context_, "[%s]actualSeqLenData data %ld is %ld.", templateName, i, actualSeqLenData[i]);
            OPS_LOG_D(context_, "[%s]actualSeqLenKvData data %ld is %ld.", templateName, i, actualSeqLenKvData[i]);
            accumS1BlockNum += CeilDivision(actualSeqLenData[i], s1BasicBlock);
        }
        int64_t totalSize = accumS1BlockNum * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize();
        int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
        multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAivNum));
        multiCoreParams.set_totalSize(totalSize);
        multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum));
        multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
    }
    bool IsTemplateMatched() const override
    {
        return true;
    }

    bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch) override
    {
        apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float);
        return true;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto &tensorSizeParams = tilingData.tensorSizeParams;
        auto &coreParams = tilingData.coreParams;

        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Byetes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;
        int64_t stage1Bytes = 0;
        // FP32场景，stage1需要再额外申请2倍的空间
        if (inputDtypeBytes == DATA_TYPE_FP32) {
            stage1Bytes = bmm1Byetes;
        }

        // dSize小于64的场景，无需切D， workspace占用较小
        if (dSize <= D_SPECIFIC_SIZE) {
            // 默认使能NZND, stage1占用3倍的空间，stage2占用2倍空间
            workspaces[0] = static_cast<size_t>(
                                (bmm1Byetes * SPACE_NUM_3 + stage1Bytes * SPACE_NUM_2 +
                                 SPACE_NUM_2 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                                 WORK_SPACE_RESERVE_SIZE;
        } else {
            // 切D场景，默认使能NZND，stage1占用3倍的空间，stage2占用4倍空间
            workspaces[0] = static_cast<size_t>(
                                (bmm1Byetes * SPACE_NUM_3 + stage1Bytes * SPACE_NUM_2 +
                                 SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                                 WORK_SPACE_RESERVE_SIZE;
        }
        if (pseType == PSE_INNER_MUL_ADD_TYPE || pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
            if (s2Size > s2sizeLimitMax) {
                pseAlibiBaseS2 = s2sizeLimitMax;
            } else {
                pseAlibiBaseS2 = alignedS2;
            }
            pseAlibiBaseS1 = s1BasicBlock;
        }
        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = ge::Shape({s1BasicBlock / GetNRatio(), s2BasicBlock * GetNRatio()});
        AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                          tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
    }

    bool CheckPretokenAndNexttoken(SparseEnum &sparseType)
    {
        if (sparseMode == ALL_MASK) {
            if (preTokens < s1Size - 1 || nextTokens < s2Size - 1) {
                OPS_LOG_W(context_,
                          "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                          "preTokens and nextTokens will be reset max int value.",
                          preTokens, nextTokens, sparseMode);
                preTokens = std::numeric_limits<int32_t>::max();
                nextTokens = std::numeric_limits<int32_t>::max();
            }
            sparseType = static_cast<SparseEnum>(static_cast<uint8_t>(sparseMode));
        } else if (sparseMode == LEFT_UP_CAUSAL) {
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            nextTokens = 0;
            sparseType = SparseEnum::CAUSAL;
        } else if (sparseMode == RIGHT_DOWN_CAUSAL) {
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                    OPS_LOG_E(context_, "Batch[%ld] s1[%ld] is larger than s2[%ld], exist invalid row.", i,
                              actualSeqLenData[i], actualSeqLenKvData[i]);
                    return false;
                }
            }
            preTokens = s2Size; // if sparse type is causal, template always need preTokens equal to s1Size
            nextTokens = 0;
            sparseType = SparseEnum::RIGHT_DOWN_CAUSAL;
        } else if (sparseMode == BAND) {
            if (preTokens < 0) {
                OPS_LOG_E(context_, "pre_tokens[%ld] config error, has invalid data block.", preTokens);
                return false;
            }
            if (nextTokens < 0 && preTokens + nextTokens < 0) {
                OPS_LOG_E(context_, "pre_tokens[%ld], next_tokens[%ld], invalid config.", preTokens, nextTokens);
                return false;
            }
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] == 0 || actualSeqLenKvData[i] == 0) {
                    continue;
                }
                if (actualSeqLenData[i] - nextTokens > actualSeqLenKvData[i]) {
                    OPS_LOG_E(context_, "Batch[%ld], s1[%ld], s2[%ld], next_tokens[%ld], has invalid row.", i,
                              actualSeqLenData[i], actualSeqLenKvData[i], nextTokens);
                    return false;
                }
            }
            if (s1Size == s2Size && preTokens >= s1Size && nextTokens == 0) {
                preTokens = s1Size;
                nextTokens = 0;
                sparseType = SparseEnum::CAUSAL;
                return true;
            }
            sparseType = SparseEnum::BAND_COMPRESS;
        } else if (sparseMode == RIGHT_DOWN_CAUSAL_BAND) {
            int64_t lastS2 = actualSeqLenKvData[bandIndex];
            if (preTokens < lastS2 || nextTokens > 0) {
                OPS_LOG_E(context_,
                          "RightDownCausal_Band mode: pre_tokens[%ld] is smaller than last valid s2[%ld]"
                          "or next_tokens[%ld] is larger than 0, wrong config.",
                          preTokens, lastS2, nextTokens);
                return false;
            }
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] == 0 || actualSeqLenKvData[i] == 0) {
                    continue;
                }
                if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                    OPS_LOG_E(context_, "Batch[%ld] s1[%ld] is larger than s2[%ld].", i, actualSeqLenData[i],
                              actualSeqLenKvData[i]);
                    return false;
                }
                if ((i == bandIndex) && (actualSeqLenData[i] - nextTokens > actualSeqLenKvData[i])) {
                    OPS_LOG_E(context_, "Batch[%ld], s1[%ld], s2[%ld], next_tokens[%ld], has invalid row.", i,
                              actualSeqLenData[i], actualSeqLenKvData[i], nextTokens);
                    return false;
                }
            }
            sparseType = SparseEnum::RIGHT_DOWN_CAUSAL_BAND;
        } else if (sparseMode == BAND_LEFT_UP_CAUSAL) {
            if (actualSeqLenData[bandIndex] - nextTokens > actualSeqLenKvData[bandIndex]) {
                OPS_LOG_E(context_, "Batch[%ld], s1[%ld], s2[%ld], next_tokens[%ld], has invalid row.", bandIndex,
                          actualSeqLenData[0], actualSeqLenKvData[0], nextTokens);
                return false;
            }
            int64_t firstS2 = actualSeqLenKvData[bandIndex];
            if (preTokens < firstS2) {
                OPS_LOG_E(context_, "Band_LeftUpCausal mode: pre_tokens[%ld] is smaller than first valid s2[%ld].",
                          preTokens, firstS2);
                return false;
            }
            sparseType = SparseEnum::BAND_LEFT_UP_CAUSAL;
        }
        return true;
    }

    bool SparseNoMaskModeCheck(int64_t maxS1Value, int64_t maxS2Value, int64_t minS2Value,
                               SparseEnum &sparseType)
    {
        if (nextTokens < 0) {
            OPS_LOG_E(context_, "nextTokens[%ld] config error, there is no valid data block.", nextTokens);
            return false;
        }
        if (preTokens >= maxS1Value && nextTokens >= maxS2Value) {
            return true;
        }
        for (int64_t i = 0L; i < bSize; ++i) {
            if (actualSeqLenData[i] == 0 || actualSeqLenKvData[i] == 0) {
                continue;
            }
            if (actualSeqLenKvData[i] + preTokens < actualSeqLenData[i]) {
                OPS_LOG_E(context_, "Batch[%ld] s1[%ld] s2[%ld] has invalid row, check pre_tokens and next_tokens.", i,
                          actualSeqLenData[i], actualSeqLenKvData[i]);
                return false;
            }
        }
        if (preTokens >= 0) {
            s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            sparseType = SparseEnum::BAND;
            isSparseValidSizeAligned = true;
            return true;
        }

        if (preTokens < 0) {
            int64_t absPreTokens = std::abs(preTokens);
            if (nextTokens >= absPreTokens) {
                s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), 0L);
                s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
                sparseType = SparseEnum::BAND;
                isSparseValidSizeAligned = true;
                return true;
            } else {
                OPS_LOG_E(context_,
                          "preTokens[%ld] and nextTokens[%ld] config error with S[%ld], has invalid data block.",
                          preTokens, nextTokens, minS2Value);
                return false;
            }
        }
        return true;
    }

    bool VarLenGetPrefixNList(SparseEnum &sparseType)
    {
        auto prefixN = context_->GetOptionalInputTensor(PREFIX_INPUT_INDEX);
        if (prefixN == nullptr) {
            OPS_LOG_E(context_, "[%s] prefixN is null pointer while sparse mode is prefix compress", templateName);
            return false;
        }

        auto &prefixShape = prefixN->GetShape().GetStorageShape();
        if (prefixShape.GetDimNum() != 1) {
            OPS_LOG_E(context_, "[%s] prefixN shape is invalid, DimNum should be 1, but it is %zu.", templateName,
                      prefixShape.GetDimNum());
            return false;
        }
        if (prefixShape.GetDim(0) != bSize) {
            OPS_LOG_E(context_,
                      "[%s] prefixN is invalid, it should be the same size as bSize[%ld], but it "
                      "is %ld.",
                      templateName, bSize, prefixShape.GetDim(0));
            return false;
        }

        /* Get Data from tensor. */
        prefixNData = prefixN->GetData<int64_t>();
        if (prefixNData == nullptr) {
            OPS_LOG_E(context_, "[%s] prefixN data is null pointer", templateName);
            return false;
        }

        for (int64_t i = 0; i < bSize; ++i) {
            if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                if (prefixNData[i] < 0 || prefixNData[i] > actualSeqLenKvData[i]) {
                    OPS_LOG_E(context_, "[%s] batch[%ld] prefixN=%ld is invalid, should be in range of [0, %ld]",
                              templateName, i, prefixNData[i], actualSeqLenKvData[i]);
                    return false;
                }
                if (prefixNData[i] == 0) {
                    implMode = AA_INVALID_LINE_HIGH_PRECISION;
                    OPS_LOG_D(context_, "Enable invalid line impl mode.");
                }
            } else {
                if (prefixNData[i] < actualSeqLenKvData[i] - actualSeqLenData[i] ||
                    prefixNData[i] > actualSeqLenKvData[i]) {
                    OPS_LOG_E(context_, "[%s] batch[%ld] prefixN=%ld is invalid, should be in range of [%ld, %ld]",
                              templateName, i, prefixNData[i], actualSeqLenKvData[i] - actualSeqLenData[i],
                              actualSeqLenKvData[i]);
                    return false;
                }
            }
        }

        sparseType = SparseEnum::PREFIX;
        return true;
    }

    bool VarLenSparseModeProcess(SparseEnum &sparseType)
    {
        if (sparseMode == static_cast<int64_t>(PREFIX) || sparseMode > static_cast<int64_t>(BAND_LEFT_UP_CAUSAL)) {
            OPS_LOG_E(context_, "Var len not support sparse mode %ld.", sparseMode);
            return false;
        }

        if (!CheckPretokenAndNexttoken(sparseType)) {
            OPS_LOG_E(context_, "Check pre_tokens and next_tokens failed.");
            return false;
        }

        if (sparseMode == static_cast<int64_t>(PREFIX_COMPRESS)) {
            if (!VarLenGetPrefixNList(sparseType)) {
                return false;
            }
        }
        return true;
    }

    bool GetSparseInfo(SparseEnum &sparseType) override
    {
        OPS_LOG_D(context_,
                  "check sparse feature: preTokens[%ld], nextTokens[%ld], s1[%ld], s2[%ld], attenMaskExist[%d].",
                  preTokens, nextTokens, s1Size, s2Size, attenMaskExistFlag);
        if (attenMaskExistFlag != 1 || tilingKeyLayout != LayoutType::LAYOUT_TND) {
            return true;
        }

        // if sparseMode is NoMask, preTokens and nextTokens start from top left vertex;
        // if sparseMode is Band, preTokens and nextTokens start from bottom right vertex.
        if (sparseMode == NO_MASK) {
            if (preTokens >= s1Size && nextTokens == 0) {
                sparseType = SparseEnum::CAUSAL;
                preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            } else {
                if (preTokens >= s1Size && nextTokens >= s2Size) {
                    return true;
                }
                int64_t minS2Value = *std::min_element(actualSeqLenKvData.begin(), actualSeqLenKvData.begin() + bSize);
                if (!SparseNoMaskModeCheck(s1Size, s2Size, minS2Value, sparseType)) {
                    return false;
                }
            }
        } else {
            if (!VarLenSparseModeProcess(sparseType)) {
                return false;
            }
        }
        return true;
    }

    int64_t GetS2RealSize(uint8_t sparseType, int32_t bOutIdx, int64_t s1OutIdx)
    {
        int64_t s2RealSize = s2Size;
        if (sparseType == static_cast<uint8_t>(SparseEnum::CAUSAL) && s1Size == s2Size) {
            s2RealSize = s1BasicBlock * (s1OutIdx + 1);
        } else if (sparseType == static_cast<uint8_t>(SparseEnum::PREFIX)) {
            s2RealSize = std::max(s1BasicBlock * (s1OutIdx + 1) - s1Size + s2Size, prefixNData[bOutIdx]);
        }
        return std::min(s2RealSize, actualSeqLenKvData[bOutIdx]);
    }

    bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx) override
    {
        uint8_t sparseType = tilingData.inputParams.get_sparseType();
        auto &coreParams = tilingData.coreParams;
        int64_t localAccumS1BlockNum = 0;
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
            localAccumS1BlockNum += (s1BlockNum * n2G);
        }

        return true;
    }

    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray, MultiCoreParams &multiCoreParams,
                     std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx)
    {
        // to avoid buffer overflow, or maybe sometimes we want to only verify single core
        int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
        int64_t totalSize = multiCoreParams.get_totalSize();
        int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
        int64_t tmpMaxVal = maxVal;

        // 从前往后遍历
        for (int64_t idx = 1; idx < validAivNum; ++idx) {
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
        for (int64_t idx = validAivNum - 1; idx > 0; --idx) {
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

    inline bool InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAivNum, int64_t totalSize,
                              const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue)
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

    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, MultiCoreParams &multiCoreParams) override
    {
        // to avoid buffer overflow, or maybe sometimes we want to only verify single core
        int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
        int64_t totalSize = multiCoreParams.get_totalSize(); // BN2GS1.o
        int64_t *sparseStartIdx = multiCoreParams.get_sparseStartIdx();

        OPS_ERR_IF(totalSize <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "totalSize should be larger than 0."),
                   return false);

        // initLoad: 使用均分策略, 保证后续不会比均分差
        int64_t splitFactorSize = multiCoreParams.get_splitFactorSize();
        std::vector<int64_t> localSparseStartIdx(MAX_AIV_NUM, totalSize);
        for (int64_t idx = 0; idx < MAX_AIV_NUM; ++idx) {
            localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
        }
        std::vector<int64_t> localValue(validAivNum, 0);
        InitLoadValue(sparseValidArray, validAivNum, totalSize, localSparseStartIdx, localValue);

        // 负载均衡粗调
        std::vector<int64_t> tmpLocalValue(validAivNum, 0);
        std::vector<int64_t> tmpsparseStartIdx(MAX_AIV_NUM, totalSize);
        int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
        int64_t avgVal = CeilDivision(sparseArraySum, validAivNum);

        tmpsparseStartIdx[0] = 0;
        for (int64_t idx = 1; idx < MAX_AIV_NUM; ++idx) {
            int64_t start = tmpsparseStartIdx[idx - 1];
            int64_t singleLoadValue = 0;
            tmpsparseStartIdx[idx] = start;
            while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
                singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
                tmpsparseStartIdx[idx] += 1;
            }

            if ((start + 1) < tmpsparseStartIdx[idx]) {
                int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
                tmpsparseStartIdx[idx] = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ?
                                             (tmpsparseStartIdx[idx] - 1) :
                                             (tmpsparseStartIdx[idx]);
                singleLoadValue = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ? redoSingleLoadValue :
                                                                                                  singleLoadValue;
                sparseArraySum -= singleLoadValue;
                avgVal = CeilDivision(sparseArraySum, (validAivNum - idx));
            }
        }

        InitLoadValue(sparseValidArray, validAivNum, totalSize, tmpsparseStartIdx, tmpLocalValue);

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
        for (int64_t idx = 0; idx < MAX_AIV_NUM; ++idx) {
            sparseStartIdx[idx] = localSparseStartIdx[idx];
        }

        return true;
    }

    void SetSparseParams() override
    {
        auto &coreParams = tilingData.coreParams;
        auto &multiCoreParams = tilingData.multiCoreParams;
        std::vector<int64_t> sparseValidArray;
        sparseValidArray.reserve(multiCoreParams.get_totalSize());
        InitSparseValidArray(sparseValidArray, 0);
        SetSparseStartIdx(sparseValidArray, multiCoreParams);

        coreParams.set_s1SparseValidSize(s1SparseValidSize);
        coreParams.set_s2SparseValidSize(s2SparseValidSize);
    }
};

class FlashAttentionScoreTilingDropMask : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingDropMask(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        if (context_->GetRawTilingData()->GetCapacity() >= tilingData.GetDataSize()) {
            OPS_ERR_IF(memset_s(context_->GetRawTilingData()->GetData(), tilingData.GetDataSize(), 0,
                                tilingData.GetDataSize()) != EOK,
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to memset tiling data"), return;);
        }
    }
    ~FlashAttentionScoreTilingDropMask() override = default;

protected:
    ge::graphStatus DoOpTiling() override
    {
        auto &inputParams = tilingData.inputParams;
        auto &dropmaskParams = tilingData.dropmaskParams;
        if (inputParams.get_needDropMaskOp() == 0) {
            return ge::GRAPH_PARAM_INVALID;
        }

        int64_t shapeTotalSize = inputParams.get_bSize() * inputParams.get_n2Size() * inputParams.get_gSize() *
                                 inputParams.get_s1Size() * inputParams.get_s2Size();
        auto layoutType = tilingData.inputParams.get_layoutType();
        if (layoutType == LAYOUT_TND) {
            for (int64_t i = 0; i < bSize; i++) {
                dropTotalSize += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            shapeTotalSize = inputParams.get_n2Size() * inputParams.get_gSize() * dropTotalSize;
            OPS_LOG_D(context_, "shapeTotalSize %ld dropTotalSize %ld.", shapeTotalSize, dropTotalSize);
        }
        // 保证每核计算数据量256倍数，2048表示bit位，256 * 8
        const int64_t ubCalFactor = 2048;
        int64_t shapeSplitCoreSize = CeilDivision(shapeTotalSize, ubCalFactor);
        int64_t shapeSingleCoreSize = CeilDivision(shapeSplitCoreSize, static_cast<int64_t>(aivNum));

        // ub能计算的最大元素数, 向下对齐
        // 单次ub计算量为x个元素,空间占用：x/8 * 1 [1个 uint8]+ 2x * 2[2个fp16,select的src和res] + x * 1 [1个uint8]共6份
        int64_t baseUbCalSize = AlignDown(CeilDivision(static_cast<int64_t>(aicoreParams_.ubSize), 6L), ubCalFactor);
        baseUbCalSize = std::min(baseUbCalSize, shapeSingleCoreSize * ubCalFactor);
        // ub 的外层循环次数
        int64_t multiCoreFactorSize = CeilDivision(shapeSingleCoreSize * ubCalFactor, baseUbCalSize);

        dropmaskParams.set_shapeTotalSize(shapeTotalSize);
        dropmaskParams.set_multiCoreFactorSize(static_cast<int32_t>(multiCoreFactorSize));
        dropmaskParams.set_multiCoreTotalSize(CeilDivision(shapeSplitCoreSize * ubCalFactor, baseUbCalSize));
        dropmaskParams.set_baseUbCalSize(static_cast<int32_t>(baseUbCalSize));
        return ge::GRAPH_PARAM_INVALID;
    }

    bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch) override
    {
        return true;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        return 0UL;
    }
};

// NOTE manually initialize tiling data in hostapi scenario in highest priority template
REGISTER_TILING_TEMPLATE("FlashAttentionScore", FlashAttentionScoreTilingDropMask, 0);
REGISTER_TILING_TEMPLATE("FlashAttentionScore", FlashAttentionVarLenScoreTiling, 94);
REGISTER_TILING_TEMPLATE("FlashAttentionScore", FlashAttentionScoreTilingS1s2Bn2gs1, 96);
REGISTER_TILING_TEMPLATE("FlashAttentionScore", FlashAttentionScoreTilingS1Bn2gs1, 97);
REGISTER_TILING_TEMPLATE("FlashAttentionScore", FlashAttentionScoreTilingB, 98);
} // namespace FA
} // namespace optiling
