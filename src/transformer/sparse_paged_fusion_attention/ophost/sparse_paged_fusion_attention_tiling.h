/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_paged_attention_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_NEW_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_NEW_H_

#include <cstdint>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/data_copy_transpose_tiling.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"

const uint32_t MAX_CORE_NUM = 50;
const uint32_t MAX_SIZE_BATCH = 256U;
#ifdef ASCENDC_OP_TEST
#define IFA_EXTERN_C extern "C"
#else
#define IFA_EXTERN_C
#endif
namespace optiling {

BEGIN_TILING_DATA_DEF(SparsePagedFusionAttentionInitOutputParams)
TILING_DATA_FIELD_DEF(uint32_t, isPerChnOut)
TILING_DATA_FIELD_DEF(uint32_t, isOutQuantTypeBf16)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttentionInitOutputParamsOp, SparsePagedFusionAttentionInitOutputParams)

BEGIN_TILING_DATA_DEF(SparsePagedFusionAttentionBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, headSize)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, maxPositionNumPerBatch)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, kvHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, qHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
TILING_DATA_FIELD_DEF(uint32_t, batchContinuousFlag)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftFlag)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftB)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftS)
TILING_DATA_FIELD_DEF(uint32_t, selectWithByteMaskTmpMinSize)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDims)
TILING_DATA_FIELD_DEF(uint32_t, kvPaddingFlag)
TILING_DATA_FIELD_DEF(uint32_t, msdIterNum)
TILING_DATA_FIELD_DEF(uint32_t, l2CacheOffFlag)
TILING_DATA_FIELD_DEF(uint32_t, antiquantPerTensorFlag)
TILING_DATA_FIELD_DEF(uint32_t, antiquantPerHeadFlag)
TILING_DATA_FIELD_DEF(uint32_t, antiquantParamsInPagedAttentionFlag)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskSize)
TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag)
TILING_DATA_FIELD_DEF(uint32_t, totalBlockNum)
TILING_DATA_FIELD_DEF(uint32_t, antiqSeqSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttentionBaseParamsOp, SparsePagedFusionAttentionBaseParams)

BEGIN_TILING_DATA_DEF(SparsePagedFusionAttentionCentSelectParams)
TILING_DATA_FIELD_DEF(int64_t, bSize);  // Batch size
TILING_DATA_FIELD_DEF(int64_t, n1Size); // Number of qheads
TILING_DATA_FIELD_DEF(int64_t, n2Size); // Number of qheads
TILING_DATA_FIELD_DEF(int64_t, gSize);  // Number of groups
// compute cent:
TILING_DATA_FIELD_DEF(int64_t, dSize);  // Dimension size of query and KV
TILING_DATA_FIELD_DEF(int64_t, cSize);  // Size of the cluster dimension
TILING_DATA_FIELD_DEF(int64_t, clusterBlockNum);  // Number of cluster blocks
TILING_DATA_FIELD_DEF(int64_t, clusterBlockSize);  // Size of the cluster block
// select position:
TILING_DATA_FIELD_DEF(int64_t, kvPageLen);  // Size of the sequence dimension of block_ids
TILING_DATA_FIELD_DEF(int64_t, maxBatch);  // Size of the sequence dimension of block_table
TILING_DATA_FIELD_DEF(int64_t, maxPage);  // Size of the sequence dimension of block_table
TILING_DATA_FIELD_DEF(int64_t, maxPageNum);  // Size of the sequence dimension of page_position
// tilling
TILING_DATA_FIELD_DEF(int32_t, blockSize);
TILING_DATA_FIELD_DEF(int32_t, usedCoreNum);
// TopK:
TILING_DATA_FIELD_DEF(int32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, tmpsize);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttentionCentSelectParamsOp, SparsePagedFusionAttentionCentSelectParams)

BEGIN_TILING_DATA_DEF(SparsePagedFusionAttentionCoreParams)
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, coreSidxEnd); // 50:MAX_CORE_NUM  coreSidxEnd数组首地址要保证8字节对齐
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttentionCoreParamsOp, SparsePagedFusionAttentionCoreParams);

BEGIN_TILING_DATA_DEF(SparsePagedFusionAttentionSingleCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, sInnerLoopTimes);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSize);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSizeTail);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, blockSplitBn2Range);
TILING_DATA_FIELD_DEF(uint32_t, tailSplitedBatchRange);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttentionSingleCoreParamsOp, SparsePagedFusionAttentionSingleCoreParams)

BEGIN_TILING_DATA_DEF(SparsePagedFusionAttentionSingleCoreTensorSize)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttentionSingleCoreTensorSizeOp, SparsePagedFusionAttentionSingleCoreTensorSize)

BEGIN_TILING_DATA_DEF(SparsePagedFusionAttentionSplitKVParams)
TILING_DATA_FIELD_DEF(uint32_t, s2)
TILING_DATA_FIELD_DEF(uint32_t, sInnerLoopSize)
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize)
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttentionSplitKVParamsOp, SparsePagedFusionAttentionSplitKVParams)

BEGIN_TILING_DATA_DEF(SparsePagedFusionAttentionTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedFusionAttentionBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedFusionAttentionCentSelectParams, centSelectParams);
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedFusionAttentionSplitKVParams, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedFusionAttentionCoreParams, sparsePagedFusionAttentionCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedFusionAttentionSingleCoreParams, sparsePagedFusionAttentionSingleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedFusionAttentionSingleCoreTensorSize, sparsePagedFusionAttentionSingleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedFusionAttentionInitOutputParams, outputParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttentionTilingDataOp, SparsePagedFusionAttentionTilingData)

BEGIN_TILING_DATA_DEF(SparsePagedFusionAttentionTilingDataPrefix)
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedFusionAttentionTilingData, base);
TILING_DATA_FIELD_DEF(uint64_t, prefixAttenOutOffset); // 临时输出偏移
TILING_DATA_FIELD_DEF(uint64_t, userPromptAttenOutOffset);
TILING_DATA_FIELD_DEF(uint64_t, tmpLseOffset);
TILING_DATA_FIELD_DEF(uint64_t, prefixLen); // prefix 长度
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum); // combine 分核参数，参考普通bn分核流程，总数不超过blockdim
TILING_DATA_FIELD_DEF(uint32_t, blockSplitBn2Range);
TILING_DATA_FIELD_DEF(uint32_t, tailSplitedBatchRange);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, batchSizeQ);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttentionTilingDataPrefixOp, SparsePagedFusionAttentionTilingDataPrefix)

BEGIN_TILING_DATA_DEF(SparsePagedFusionAttentionTilingDataV2)
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedFusionAttentionTilingData, tilingBase);
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedFusionAttentionTilingDataPrefix, tilingPrefix);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttention, SparsePagedFusionAttentionTilingDataV2)

BEGIN_TILING_DATA_DEF(SparsePagedFusionAttentionEmptyInputTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedFusionAttentionInitOutputParams, outputParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttention_13, SparsePagedFusionAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttention_14, SparsePagedFusionAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttention_27, SparsePagedFusionAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(SparsePagedFusionAttention_30, SparsePagedFusionAttentionEmptyInputTilingData)

struct SparsePagedFusionAttentionCompileInfo {
    int64_t core_num;
};

struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
    const gert::StorageShape *shape;
};

struct SparsePagedFusionAttentionContext {
    const char *opName;
    fe::PlatFormInfos *platformInfo;
    RequiredParaInfo query;
    RequiredParaInfo key;
    RequiredParaInfo value;
    OptionalParaInfo pseShift;
    OptionalParaInfo attenMask;
    OptionalParaInfo actualSeqLengths;
    OptionalParaInfo deqScale1;
    OptionalParaInfo quantScale1;
    OptionalParaInfo deqScale2;
    OptionalParaInfo quantScale2;
    OptionalParaInfo quantOffset2;
    OptionalParaInfo antiquantScale;
    OptionalParaInfo antiquantOffset;
    OptionalParaInfo blockTable;
    OptionalParaInfo kvPaddingSize;
    OptionalParaInfo keyAntiquantScale;
    OptionalParaInfo keyAntiquantOffset;
    OptionalParaInfo valueAntiquantScale;
    OptionalParaInfo valueAntiquantOffset;
    OptionalParaInfo keySharedPrefix;
    OptionalParaInfo valueSharedPrefix;
    OptionalParaInfo actualSharedPrefixLen;
    OptionalParaInfo queryRope;
    OptionalParaInfo keyRope;
    OptionalParaInfo keyRopeAntiquantScale;
    OptionalParaInfo l1Cent; // 
    OptionalParaInfo blockIds; // 
    OptionalParaInfo totalSeqlen; // 
    

    RequiredParaInfo blockPosition; //
    RequiredParaInfo pagePositionLength; // 
    RequiredParaInfo maxPagePositionLength; // 
    RequiredParaInfo attenOut;
    const uint32_t *numHeads;
    const float *scaleValue;
    const uint32_t *kvHeadNums;
    const char *layOut;
    const uint32_t *blockSize;
    const uint32_t *innerPrecise;
    const uint32_t *antiquantMode;
    const bool *softmaxLseFlag;
    const uint32_t *keyAntiquantMode;
    const uint32_t *valueAntiquantMode;
    const uint32_t *sparseMode;

    size_t *workSpaces;
    std::vector<gert::StorageShape *> kCache;
    std::vector<gert::StorageShape *> vCache;
    uint64_t tilingKey;
    uint32_t blockDim;
};

enum class TilingInOutMode : uint32_t {
    IO_INVALID = 0,
    INT8_INT8 = 1,
    FP16_INT8 = 2,
    INT8_FP16 = 3,
    FP16_FP16 = 4,
    BF16_BF16 = 5,
    FP32_FP32 = 6,
    FP16_FP16_SPLITKV = 7,
    BF16_INT8 = 8,
};

enum class IfaPerfMode : uint32_t {
    NORMAL = 0,
    BMM_ALL_BY_VEC,
    C1_V1
};

enum IfaSocVersion : uint32_t {
    SOC_ASCEND_910B = 0,
    SOC_ASCEND_310P = 1,
};

enum IfaLayout : uint32_t {
    BSH_BSND = 0,
    BNSD = 1,
};

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t KVINT4_BYTE_BLOCK = 64;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t NUM_BYTES_BOOL = 1;
constexpr uint32_t NUM_BYTES_INT8 = 1;
constexpr uint32_t MAX_MATMUL_BASE = 512;
constexpr uint32_t MATMUL_BASE_N = 256;
constexpr uint32_t MAX_MATMUL_BASE_M = 128;
constexpr uint32_t MAX_SPLIT_SIZE = 8192;
constexpr uint32_t L0B_SIZE = 64 * 1024;
constexpr uint32_t L0C_SIZE = 128 * 1024;
constexpr uint32_t DIM_BNSD = 4;
constexpr uint32_t DIM_BNSD_OR_BNSD = 4;
constexpr uint32_t DIM_BSH = 3;
constexpr uint32_t DIM_PER_TOKEN_KvSplit = 2;
constexpr uint32_t DIM_PER_TOKEN = 3;
constexpr uint32_t PER_CHANNEL_MODE = 0;
constexpr uint32_t PER_TOKEN_MODE = 1;
constexpr uint32_t PER_CHANNEL_TOKEN_MODE = 2;
constexpr uint32_t DEQUANT_PER_CHANNEL_MODE = 0; // 下面的0~4代表用户输入的数值，与上面的PER_CHANNEL_MODE、PER_TOKEN_MODE含义不同
constexpr uint32_t DEQUANT_PER_TOKEN_MODE = 1;
constexpr uint32_t DEQUANT_PER_TENSOR_HEAD_MODE = 2;
constexpr uint32_t DEQUANT_PER_TOKEN_HEAD_MODE = 3;
constexpr uint32_t DEQUANT_PER_TOKEN_PA_MODE = 4;
constexpr uint32_t DEQUANT_PER_TOKEN_HEAD_PA_MODE = 5;
constexpr uint32_t DIM_PER_CHANNEL_BNSD = 4;
constexpr uint32_t DIM_PER_CHANNEL_BSND = 3;
constexpr uint32_t DIM_PER_CHANNEL_BSH = 2;
constexpr uint32_t DIM_PER_TENSOR = 1;
constexpr uint32_t PER_TOKEN_N = 0;
constexpr uint32_t PER_TOKEN_B = 1;
constexpr uint32_t PER_TOKEN_S = 2;


class SparseFusionIFATiling {
public:
    SparseFusionIFATiling() = default;
    ~SparseFusionIFATiling() = default;

    ge::graphStatus DoTiling(gert::TilingContext &context);
    ge::graphStatus RunBigKernelTiling(SparsePagedFusionAttentionContext &context, SparsePagedFusionAttentionTilingDataV2 &tilingData,
                                       bool isWorkspace = false);
    ge::graphStatus SparsePagedFusionAttentionSetTilingData(gert::TilingContext &context,
                                                     SparsePagedFusionAttentionTilingDataV2 &tilingData);
    static ge::graphStatus ConvertContext(gert::TilingContext &context, SparsePagedFusionAttentionContext &ifaContext);
    bool NeedRollBack()
    {
        return passToOldTiling_;
    }

    uint32_t GetAntiquantSeqLength();

private:
    ge::graphStatus GetNpuInfo();
    ge::graphStatus PreProcess();
    ge::graphStatus ProcessBaseInputs();
    ge::graphStatus ProcessOptionalTensors();
    ge::graphStatus ProcessPseShift();
    ge::graphStatus ProcessAttenMask();
    ge::graphStatus ProcessActualSeqLen();
    ge::graphStatus ProcessQuant1();
    ge::graphStatus ProcessQuant2();
    ge::graphStatus ProcessDequant1();
    ge::graphStatus ProcessDequant2();
    ge::graphStatus ProcessAntiQuant();
    ge::graphStatus ProcessBlockTable();
    ge::graphStatus ProcessKVPaddingSize();
    ge::graphStatus ProcessSharedPrefix();
    ge::graphStatus ProcessSharedPrefixLen();
    void SetupPerfMode();
    bool EnableAllVec();
    bool EnableC1V1();
    void UpdatePerfMode();

    ge::graphStatus InitInOutMode();
    ge::graphStatus CheckBaseInputsNull();
    ge::graphStatus InputAttrsPreProcess();
    ge::graphStatus QKVPreProcess();
    ge::graphStatus ProcessPageAttentionFlag();
    ge::graphStatus KvShapePostProcess();
    ge::graphStatus CheckKVHeadNum(const gert::StorageShape *inputShape);
    ge::graphStatus CheckKVShape(const size_t &size, const gert::StorageShape *keyTensorInList, const gert::StorageShape *valueTensorInList);
    ge::graphStatus CheckQKOutShape();
    ge::graphStatus CheckKeyShapeTensor(const gert::Shape &aShape);
    ge::graphStatus ZeroTensorProcess();
    ge::graphStatus SharedPrefixTiling();
    ge::graphStatus SharedPrefixCheckBasic();
    ge::graphStatus SharedPrefixCheckShapes(const gert::Shape &keyShape, const gert::Shape &valueShape);

    ge::graphStatus CheckUbSpace();
    ge::graphStatus CheckPABlockSize();
    ge::graphStatus SetL2CacheFlag();
    ge::graphStatus SetQuantFlag();

    ge::graphStatus CheckQuant2Shape(const gert::Shape &inputParaShape);
    ge::graphStatus ProcessQuant2Dtype();
    ge::graphStatus CheckKVAntiQuantMode();
    ge::graphStatus CheckKVAntiQuantPerToken(const gert::Shape &inputParaShape);
    ge::graphStatus CheckKVAntiQuantPerHead(const gert::Shape &inputParaShape);
    ge::graphStatus CheckKVAntiQuantParamsInPagedAttention();
    ge::graphStatus CheckKVAntiQuantParamsShapeInPagedAttention(const gert::Shape &inputParaShape);
    ge::graphStatus CheckKVAntiQuantPerChannel(const gert::Shape &inputParaShape);
    ge::graphStatus CheckKVAntiQuantParaShapeLegal(const gert::Shape &inputParaShape);
    ge::graphStatus CheckAntiQuantParam(const gert::Tensor *antiquantScaleTensor,
                                        const gert::Tensor *antiquantOffsetTensor,
                                        const gert::CompileTimeTensorDesc *antiquantScaleDesc,
                                        const gert::CompileTimeTensorDesc *antiquantOffsetDesc);
    ge::graphStatus CheckAntiQuantParamKeyType(const gert::Tensor *antiquantOffsetTensor,
                                               const gert::CompileTimeTensorDesc *antiquantScaleDesc,
                                               const gert::CompileTimeTensorDesc *antiquantOffsetDesc);
    ge::graphStatus CheckAntiQuantParamValueType(const gert::Tensor *antiquantOffsetTensor,
                                                 const gert::CompileTimeTensorDesc *antiquantScaleDesc,
                                                 const gert::CompileTimeTensorDesc *antiquantOffsetDesc);
    ge::graphStatus CheckSupportKVLeftPadding();
    ge::graphStatus CheckInputFormatAndLimits();
    ge::graphStatus CheckInputParameterFormat();
    ge::graphStatus CheckInputAntiquantFormat();
    bool CalcUbBmm();
    bool CalcUbSoftMax();
    bool CalcUbAttenMask();
    bool CalcUbQuant();
    bool CalcUbDeQuant();
    bool CalcUbAntiQuant();
    bool CalcUbPageAttention();
    bool CalcUbKvSplit();

    bool CheckIfRollBack();
    bool CanChangeToNew();
    void AdjustPABmm1Tiling(uint32_t &bmm1BaseN);
    void AdjustPABmm2Tiling() const;
    bool ShapeEqual(const gert::Shape &aShape, const gert::Shape &bShape);

    ge::graphStatus Split();
    ge::graphStatus CalcInnerSize(uint32_t seqSize);
    ge::graphStatus SplitBN();

    std::vector<int64_t> InitSparseValidArray(const int64_t *actualLens);
    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                     std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx);
    void InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                       const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue);
    void SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                           uint32_t *sparseStartIdx, int64_t splitFactorSize);

    bool IsFlashDecode() const;
    ge::graphStatus SplitBN_V0();
    ge::graphStatus SplitBNS();

    bool CheckWorkSpace();
    bool GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType);

    ge::graphStatus CalcWorkSpace();
    ge::graphStatus CalcBlockDim();
    ge::graphStatus GenTilingKey();

    ge::graphStatus FillTiling();
    void FillTilingBaseParams();
    void FillTilingSplitKV();
    void FillTilingCentSelect();
    void FillTilingCoreParams();
    void FillTilingSingleCoreParams();
    void FillTilingSingleCoreTensorSize();
    void FillTilingSoftmax();
    void FillTilingSoftmaxFlashTiling();
    void FillTilingTranspose();
    void FillTilingOutputParams();
    bool GetBmm1Tiling(const matmul_tiling::DataType &qType, const matmul_tiling::DataType &kvType, const uint32_t M);
    bool GetBmm2Tiling(const matmul_tiling::DataType &qType, const matmul_tiling::DataType &kvType, const uint32_t M);
    bool FillTilingBmm(); // may fail

    ge::graphStatus CalcSysPrefixWorkSpace();
    ge::graphStatus FillSysPrefixTiling();
    ge::graphStatus CalcSysPrefixBlockDim();
    ge::graphStatus SplitForLseCombine();

private:
    bool passToOldTiling_ = false;
    uint32_t numHeads_ = 0;
    float scaleValue_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t innerPrecise_ = 0;
    uint32_t nNumOfQInOneGroup_ = 1;
    uint32_t msdIterNum_ = 1;
    uint32_t antiquantMode_ = 0;
    uint32_t antiquantPerTensorFlag_ = 0;
    uint32_t antiquantPerHeadFlag_ = 0;
    uint32_t antiquantParamsInPagedAttentionFlag_ = 0;
    uint32_t antiquantNum_ = 2;

    uint32_t headDim_ = 0;
    uint32_t seqSize_ = 0;
    uint32_t batchSize_ = 0;
    IfaLayout inputLayout_ = IfaLayout::BSH_BSND;
    uint32_t sMax_ = 0;
    uint32_t blockTypeSize_ = 0; // 计算中间量大小
    uint32_t kvSplitPart_ = 1;
    uint32_t clusterBlockSize_ = 256;

    uint32_t sMaxPrefix_ = 0;
    uint32_t maxActualPrefixLen_ = 0;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    size_t ubSize_ = 0;
    size_t l1Size_ = 0;
    size_t l0cSize_ = 0;
    size_t l0bSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    IfaSocVersion socVersion_ = IfaSocVersion::SOC_ASCEND_910B;
    size_t libapiSize_ = 0;

    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;

    size_t softmaxFlashTmpSize_ = 0;
    size_t softmaxTmpSize_ = 0;
    size_t softMaxSize_ = 0;

    size_t selectWithByteMaskTmpMinSize_ = 0;

    bool pseShiftFlag_ = false;
    uint32_t pseShiftTypeSize_ = NUM_BYTES_FLOAT16;
    uint32_t pseShiftBatch_ = 0U;
    uint32_t pseShiftS1_ = 0U;

    bool attenMaskFlag_ = false;
    uint32_t attenMaskSize_ = 0;
    uint32_t attenMaskTypeSize_ = 0;

    bool antiQuantFlag_ = false;
    size_t antiquantUb_ = 0;
    bool kvAntiParamSplitFlag_ = false;

    bool pageAttentionFlag_ = false;
    uint32_t maxBlockNumPerBatch_ = 0;
    uint32_t maxPositionNumPerBatch_ = 0;
    size_t kvPageResUbSize_ = 0;
    uint32_t totalBlockNum_ = 0;

    bool batchContinuousFlag_ = true;
    std::vector<int64_t> kvListSeqLens_;

    bool actualSeqLenFlag_ = false;
    bool kvPaddingSizeFlag_ = false;

    bool quantFlag_ = false;
    size_t quantUbSize_ = 0;

    bool isOutQuantPerChnOut_ = false;
    bool isOutQuantTypeBf16_ = false;

    uint32_t actualLenDims_ = 0;
    uint32_t maxActualseq_ = 0;

    // flash config
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 0; // flash attention
    uint32_t sInnerSizeTail_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t headDimAlign_ = 0;
    // uint32_t sOuterSize_;  // flash decode s2

    bool isSplitBPolicy_ = false;
    bool splitKVFlag_ = false;
    uint32_t kvSplit_ = 0;
    bool splitKVFlagPrefix_ = false;
    uint32_t antiqSeqSize_ = 0;

    IfaPerfMode perfMode_ = IfaPerfMode::NORMAL;
    TilingInOutMode inOutMode_ = TilingInOutMode::FP16_FP16;
    size_t workspaceSize_ = 0;

    uint32_t taskRation_ = 0;
    uint32_t usedCoreNum_ = 0;

    uint32_t startIdxEachCore_[MAX_CORE_NUM] = {};
    SparsePagedFusionAttentionContext *context_ = nullptr;
    SparsePagedFusionAttentionTilingData *tilingData_ = nullptr;
    SparsePagedFusionAttentionTilingDataPrefix *tilingDataPrefix_ = nullptr;
    bool isWorkspace_ = false;

    uint32_t formerCoreNum_ = 0;
    uint32_t blockSplitBn2Range_ = 0;
    uint32_t tailSplitedBatchRange_ = 0;

    uint32_t l2CacheOffFlag_ = 0;
    // softmaxLse
    bool softmaxLseFlag_ = false;
    bool tilingSinkFlag_ = false;

    bool sysPrefixFlag_ = false;
    bool isSysPrefixTiling_ = false;
    uint32_t batchSizeQ_ = 1;
    uint32_t actualLenDimsPrefix_ = 0;

    uint64_t prefixAttenOutOffset_ = 0;
    uint64_t userPromptAttenOutOffset_ = 0;
    uint64_t tmpLseOffset_ = 0;

    uint32_t formerCoreNumSp_ = 0;
    uint32_t blockSplitBn2RangeSp_ = 0;
    uint32_t tailSplitedBatchRangeSp_ = 0;
    uint32_t combinUsedCore_ = 0;
};

std::string SparseFusionDataTypeToSerialString(ge::DataType type);

ge::graphStatus TilingPrepareForSparsePagedFusionAttention(gert::TilingParseContext *context);
ge::graphStatus TilingSparsePagedFusionAttentionAdapter(gert::TilingContext *context, SparsePagedFusionAttentionContext &ifaContext,
                                                 SparsePagedFusionAttentionTilingDataV2 &ifaTilingData);

IFA_EXTERN_C ge::graphStatus TilingSparsePagedFusionAttention(gert::TilingContext *context);

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_H_
