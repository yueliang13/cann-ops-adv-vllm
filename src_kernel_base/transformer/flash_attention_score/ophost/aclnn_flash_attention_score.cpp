/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_flash_attention_score.h"
#include "flash_attention_score.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const int64_t PAD_BASIC_BLOCK = 16;
static const int64_t PAD_LOWER_BOUND_196 = 196;
static const int64_t PAD_ALIGN_128 = 128;
static const int64_t PAD_ALIGN_SPL_SHAPE = 448;
static const int64_t MAX_STRIDE_S1 = 65535;
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;
static const int64_t HEAD_DIM_MAX = 512;
static const int64_t PSE_TYPE_V1 = 1; // add and mul
static const int64_t PSE_INNER_MUL_ADD = 2;
static const int64_t PSE_INNER_MUL_ADD_SQRT = 3;
static const int64_t HEAD_DIM_72 = 72;
static const int64_t HEAD_DIM_88 = 88;
static const int64_t SEQ_LEN_1024 = 1024;
static const int64_t TND_UNPAD_MAX_S2 = 1024;
static const int64_t TND_UNPAD_MAX_S1_SUM = 160 * 1024;
static const int64_t TND_UNPAD_MAX_DDIM = 96;

struct AxesInfo {
    int64_t b;
    int64_t n1;
    int64_t n2;
    int64_t s1;
    int64_t s2;
    int64_t d;
};

enum class InputLayout {
    BSND,
    SBH,
    BNSD,
    BSH,
    TND
};

struct FaShapeInfo {
    AxesInfo axes;

    InputLayout inputLayout;
    string l0InputLayoutStr;

    uint64_t dimNum = 0;
    uint64_t padNum = 0;

    FVector<int64_t, DIM_NUM_4> perm_in;
    FVector<int64_t, DIM_NUM_4> perm_out;
    FVector<int64_t, DIM_NUM_4> reshapedQueryShape;
    FVector<int64_t, DIM_NUM_4> reshapedKeyValueShape;

    bool needPad = false;
    bool needTranspose = false;
    bool needReshape = false;
};

void AnalysisAxisForBsh(const Shape &qShape, const Shape &kShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BSH;
    shapeInfo.l0InputLayoutStr = "BSH";
    uint64_t dSize = qShape[2] / shapeInfo.axes.n1;
    shapeInfo.axes.d = dSize;
    if (dSize == 0) {
        return;
    }
    shapeInfo.axes.b = qShape[0];
    shapeInfo.axes.n2 = kShape[2] / dSize;
    shapeInfo.axes.s1 = qShape[1];
    shapeInfo.axes.s2 = kShape[1];
}

void AnalysisAxisForBsnd(const Shape &qShape, const Shape &kShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BSND;
    shapeInfo.l0InputLayoutStr = "BSND";
    shapeInfo.axes.b = qShape[0];
    shapeInfo.axes.n2 = kShape[2];
    shapeInfo.axes.s1 = qShape[1];
    shapeInfo.axes.s2 = kShape[1];
    shapeInfo.axes.d = qShape[3];
}

void AnalysisAxisForTnd(const Shape &qShape, const Shape &kShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::TND;
    shapeInfo.l0InputLayoutStr = "TND";
    shapeInfo.axes.n2 = kShape[1];
    shapeInfo.axes.d = qShape[DIM_NUM_2];
}

void AnalysisAxisForSbh(const Shape &qShape, const Shape &kShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::SBH;
    shapeInfo.l0InputLayoutStr = "SBH";
    uint64_t dSize = qShape[2] / shapeInfo.axes.n1;
    shapeInfo.axes.d = dSize;
    if (dSize == 0) {
        return;
    }
    shapeInfo.axes.b = qShape[1];
    shapeInfo.axes.n2 = kShape[2] / dSize;
    shapeInfo.axes.s1 = qShape[0];
    shapeInfo.axes.s2 = kShape[0];
}

void AnalysisAxisForBnsd(const Shape &qShape, const Shape &kShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BNSD;
    shapeInfo.l0InputLayoutStr = "BNSD";
    shapeInfo.axes.b = qShape[0];
    shapeInfo.axes.n2 = kShape[1];
    shapeInfo.axes.s1 = qShape[2];
    shapeInfo.axes.s2 = kShape[2];
    shapeInfo.axes.d = qShape[3];
}

aclnnStatus AnalysisAxis(const aclTensor *query, const aclTensor *key, const char *inputLayout, int64_t headNum,
                         FaShapeInfo &shapeInfo)
{
    Shape kShape = key->GetViewShape();
    Shape qShape = query->GetViewShape();
    shapeInfo.dimNum = qShape.GetDimNum();

    // 记录轴的长度 b, n2, g, s1, s2, d
    // H1等于N1*D, H2等于N2*D
    // N1等于g*N2
    shapeInfo.axes.n1 = headNum;
    std::string inputLayoutStr = op::ToString(inputLayout).GetString();
    if (shapeInfo.dimNum == DIM_NUM_3 && inputLayoutStr == "BSH") {
        // query: (B,S1,N1*D)
        // key/value: (B,S2,N2*D)
        AnalysisAxisForBsh(qShape, kShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_4 && inputLayoutStr == "BSND") {
        // query: (B,S1,N1,D)
        // key/value: (B,S2,N2,D)
        AnalysisAxisForBsnd(qShape, kShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_3 && inputLayoutStr == "SBH") {
        // query: (S1,B,N1*D)
        // key/value: (S2,B,N2*D)
        AnalysisAxisForSbh(qShape, kShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_4 && inputLayoutStr == "BNSD") {
        // query: (B,N1,S1,D)
        // key/value: (B,N2,S2,D)
        AnalysisAxisForBnsd(qShape, kShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_3 && inputLayoutStr == "TND") {
        // query: (T,N1,D)
        // key/value: (T,N2,D)
        AnalysisAxisForTnd(qShape, kShape, shapeInfo);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "not support input_layout %s with dim_num %lu", inputLayout, shapeInfo.dimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

void SetShapeInfoForBshBsnd(int64_t alignedH1Size, FaShapeInfo &shapeInfo)
{
    if (alignedH1Size > MAX_STRIDE_S1) {
        shapeInfo.needTranspose = true;
        shapeInfo.needReshape = true;
        shapeInfo.l0InputLayoutStr = "BNSD";

        // B,S,N,D -> B,N,S,D
        shapeInfo.perm_in.assign({0, 2, 1, 3});
        // B,N,S,D -> B,S,N,D
        shapeInfo.perm_out.assign(shapeInfo.perm_in.cbegin(), shapeInfo.perm_in.cend());
    }
    if (shapeInfo.needPad) {
        shapeInfo.needReshape = true;
    }

    if (shapeInfo.inputLayout == InputLayout::BSND) {
        shapeInfo.needReshape = false;
    }
    if (shapeInfo.needReshape) {
        if (!shapeInfo.needTranspose) {
            shapeInfo.l0InputLayoutStr = "BSND";
        }
        shapeInfo.reshapedQueryShape.assign({shapeInfo.axes.b, shapeInfo.axes.s1, shapeInfo.axes.n1, shapeInfo.axes.d});
        shapeInfo.reshapedKeyValueShape.assign(
            {shapeInfo.axes.b, shapeInfo.axes.s2, shapeInfo.axes.n2, shapeInfo.axes.d});
    }
}

void SetShapeInfoForSbh(int64_t alignedH1Size, FaShapeInfo &shapeInfo)
{
    if (shapeInfo.axes.b * alignedH1Size > MAX_STRIDE_S1) {
        shapeInfo.needTranspose = true;
        shapeInfo.needReshape = true;
        shapeInfo.l0InputLayoutStr = "BNSD";

        // S,B,N,D -> B,N,S,D
        shapeInfo.perm_in.assign({1, 2, 0, 3});
        // B,N,S,D -> S,B,N,D
        shapeInfo.perm_out.assign({2, 0, 1, 3});
    }
    if (shapeInfo.needPad) {
        shapeInfo.needReshape = true;
    }

    if (shapeInfo.needReshape) {
        if (!shapeInfo.needTranspose) {
            shapeInfo.l0InputLayoutStr = "SBH";
        }
        shapeInfo.reshapedQueryShape.assign({shapeInfo.axes.s1, shapeInfo.axes.b, shapeInfo.axes.n1, shapeInfo.axes.d});
        shapeInfo.reshapedKeyValueShape.assign(
            {shapeInfo.axes.s2, shapeInfo.axes.b, shapeInfo.axes.n2, shapeInfo.axes.d});
    }
}

static int64_t GetSumIntArrayMaxValue(const aclIntArray *intArrayValue)
{
    // 获取targetLengthsList中的最大值
    int64_t maxLength = 0;
    int64_t tmpMaxLength = 0;
    if (intArrayValue->Size() == 1) {
        maxLength = static_cast<int64_t>((*intArrayValue)[0]);
        return maxLength;
    }
    maxLength = static_cast<int64_t>((*intArrayValue)[0]);
    for (size_t i = 1; i < intArrayValue->Size(); ++i) {
        tmpMaxLength = static_cast<int64_t>((*intArrayValue)[i]) - static_cast<int64_t>((*intArrayValue)[i - 1]);
        if (tmpMaxLength > maxLength) {
            maxLength = tmpMaxLength;
        }
    }
    return maxLength;
}

bool IsNeedPad(const FaShapeInfo &shapeInfo, const aclIntArray *actualSeqQLenOptional,
               const aclIntArray *actualSeqKvLenOptional)
{
    if ((shapeInfo.axes.d == HEAD_DIM_72 || shapeInfo.axes.d == HEAD_DIM_88) &&
         shapeInfo.axes.s2 <= SEQ_LEN_1024 && shapeInfo.inputLayout != InputLayout::BNSD &&
         shapeInfo.inputLayout != InputLayout::TND && shapeInfo.axes.n1 == shapeInfo.axes.n2 &&
         shapeInfo.needTranspose == false) {
        return false;
    }

    if (shapeInfo.inputLayout == InputLayout::TND) {
        if (shapeInfo.axes.d >= TND_UNPAD_MAX_DDIM) {
            return true;
        }
        int64_t sKvLenMax = 0;
        int64_t sQLenSum = 0;
        if (actualSeqQLenOptional != nullptr && actualSeqKvLenOptional != nullptr &&
            actualSeqQLenOptional->Size() == actualSeqKvLenOptional->Size()) {
            sKvLenMax = GetSumIntArrayMaxValue(actualSeqKvLenOptional);
            sQLenSum = actualSeqQLenOptional->Size() >= 1 ?
                       static_cast<int64_t>((*actualSeqQLenOptional)[actualSeqQLenOptional->Size() - 1]) : 0;
        }

        if (sKvLenMax == 0 || sQLenSum == 0) {
            // 走原来逻辑是否pad
            OP_LOGD("Fa aclnn TND case sKvLenMax(%ld) or sQLenSum(%ld) is 0.", sKvLenMax, sQLenSum);
            return true;
        }

        if ((sKvLenMax <= TND_UNPAD_MAX_S2) && (sQLenSum < TND_UNPAD_MAX_S1_SUM)) {
            // 去除pad
            OP_LOGD("Fa aclnn TND case do not do pad dimD operation.");
            return false;
        }
    }
    return true;
}

aclnnStatus InputDtypeCheck(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                            const aclTensor *realShiftOptional, int64_t pseTypeOptional)
{
    auto vDtype = value->GetDataType();
    auto kDtype = key->GetDataType();
    auto qDtype = query->GetDataType();
    if (qDtype != kDtype || kDtype != vDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of query[%s], key[%s], value[%s] are not equal.",
                op::ToString(DataType(qDtype)).GetString(), op::ToString(DataType(kDtype)).GetString(),
                op::ToString(DataType(vDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (pseTypeOptional == PSE_INNER_MUL_ADD || pseTypeOptional == PSE_INNER_MUL_ADD_SQRT)
    { // Inner pse alibi, dtype must be fp32
        if (realShiftOptional == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When the pse type is 2 or 3, the pse input must be passed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        auto pseDtype = realShiftOptional->GetDataType();
        if (pseDtype != op::DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The data type %s of pse is not invalid in pse type 2 or 3 mode, It must be float32",
                    op::ToString(DataType(pseDtype)).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
    if (realShiftOptional) {
        auto pseDtype = realShiftOptional->GetDataType();
        if (pseDtype != qDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The data type %s of pse is not equal to the data type %s of query, key and value.",
                    op::ToString(DataType(pseDtype)).GetString(), op::ToString(DataType(qDtype)).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AnalysisInput(const aclTensor *query, const aclTensor *key, char *inputLayout, int64_t headNum,
                          FaShapeInfo &shapeInfo, const aclIntArray *actualSeqQLenOptional = nullptr,
                          const aclIntArray *actualSeqKvLenOptional = nullptr)
{
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "head_num must > 0, but got %ld", headNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(AnalysisAxis(query, key, inputLayout, headNum, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (shapeInfo.axes.d > HEAD_DIM_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Head dim must <= 512, but got %ld", shapeInfo.axes.d);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (shapeInfo.axes.n2 == 0 || shapeInfo.axes.d == 0) {
        return ACLNN_SUCCESS;
    }

    if (shapeInfo.inputLayout != InputLayout::TND &&
        (shapeInfo.axes.b == 0 || shapeInfo.axes.s1 == 0 || shapeInfo.axes.s2 == 0)) {
        return ACLNN_SUCCESS;
    }

    int64_t alignDim = (shapeInfo.axes.d < PAD_LOWER_BOUND_196 || shapeInfo.axes.d == PAD_ALIGN_SPL_SHAPE) ?
                           PAD_BASIC_BLOCK :
                           PAD_ALIGN_128;
    if (shapeInfo.axes.d % alignDim != 0) {
        shapeInfo.needPad = true;
        shapeInfo.padNum = (shapeInfo.axes.d + alignDim - 1) / alignDim * alignDim - shapeInfo.axes.d;
    }

    int64_t alignedH1Size = shapeInfo.axes.n1 * (shapeInfo.axes.d + shapeInfo.padNum);
    if (shapeInfo.inputLayout == InputLayout::BSH || shapeInfo.inputLayout == InputLayout::BSND) {
        SetShapeInfoForBshBsnd(alignedH1Size, shapeInfo);
    } else if (shapeInfo.inputLayout == InputLayout::SBH) {
        SetShapeInfoForSbh(alignedH1Size, shapeInfo);
    }

    if (!IsNeedPad(shapeInfo, actualSeqQLenOptional, actualSeqKvLenOptional)) {
        shapeInfo.needPad = false;
        shapeInfo.padNum = 0;
        shapeInfo.needReshape = false;
        if (shapeInfo.inputLayout == InputLayout::BSH) {
            shapeInfo.l0InputLayoutStr = "BSH";
        }
    }

    OP_LOGD("Analysis input success. The analysis result: [needReshape]: %d, [needPad]: %d, [padNum]: %lu,"
            "[needTranspose]: %d.",
            shapeInfo.needReshape, shapeInfo.needPad, shapeInfo.padNum, shapeInfo.needTranspose);
    return ACLNN_SUCCESS;
}

static inline const aclTensor *GeneratePaddings(int32_t dimNum, int32_t padNum, aclOpExecutor *executor)
{
    // 2代表每根轴的前后都可以补0
    FVector<int64_t> padVec(dimNum * 2, 0);
    padVec[padVec.size() - 1] = padNum;

    auto padArray = executor->AllocIntArray(padVec.data(), padVec.size());
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc padVec failed");
        return nullptr;
    }

    auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);
    return padTensor;
}

aclnnStatus Contiguous(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                       const aclTensor *&realShiftOptional, const aclTensor *&dropMaskOptional,
                       const aclTensor *&paddingMaskOptional, const aclTensor *&attenMaskOptional,
                       aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (realShiftOptional) {
        realShiftOptional = l0op::Contiguous(realShiftOptional, executor);
        CHECK_RET(realShiftOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (dropMaskOptional) {
        dropMaskOptional = l0op::Contiguous(dropMaskOptional, executor);
        CHECK_RET(dropMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (paddingMaskOptional) {
        paddingMaskOptional = l0op::Contiguous(paddingMaskOptional, executor);
        CHECK_RET(paddingMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (attenMaskOptional) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus PreprocessQKV(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                          const struct FaShapeInfo &shapeInfo, aclOpExecutor *executor)
{
    if (shapeInfo.needReshape) {
        query = l0op::Reshape(
            query, executor->AllocIntArray(shapeInfo.reshapedQueryShape.data(), shapeInfo.reshapedQueryShape.size()),
            executor);
        CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
        key = l0op::Reshape(
            key,
            executor->AllocIntArray(shapeInfo.reshapedKeyValueShape.data(), shapeInfo.reshapedKeyValueShape.size()),
            executor);
        CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
        value = l0op::Reshape(
            value,
            executor->AllocIntArray(shapeInfo.reshapedKeyValueShape.data(), shapeInfo.reshapedKeyValueShape.size()),
            executor);
        CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (shapeInfo.needPad) {
        int32_t dimNum = shapeInfo.inputLayout == InputLayout::TND ? DIM_NUM_3 : DIM_NUM_4;
        auto paddings = GeneratePaddings(dimNum, shapeInfo.padNum, executor);

        query = l0op::Pad(query, paddings, executor);
        CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
        key = l0op::Pad(key, paddings, executor);
        CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
        value = l0op::Pad(value, paddings, executor);
        CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (shapeInfo.needTranspose) {
        // B,S,N,D -> B,N,S,D
        // S,B,N,D -> B,N,S,D
        auto perm = executor->AllocIntArray(shapeInfo.perm_in.data(), shapeInfo.perm_in.size());
        query = l0op::Transpose(query, perm, executor);
        CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
        key = l0op::Transpose(key, perm, executor);
        CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
        value = l0op::Transpose(value, perm, executor);
        CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (shapeInfo.inputLayout == InputLayout::SBH && shapeInfo.needPad && !shapeInfo.needTranspose) {
        // (S,B,N,D) -> (S,B,N*D)
        FVector<int64_t, DIM_NUM_3> queryShape{shapeInfo.axes.s1, shapeInfo.axes.b,
                                               shapeInfo.axes.n1 * (shapeInfo.axes.d +
                                               static_cast<int64_t>(shapeInfo.padNum))};
        FVector<int64_t, DIM_NUM_3> keyValueShape{shapeInfo.axes.s2, shapeInfo.axes.b,
                                                  shapeInfo.axes.n2 * (shapeInfo.axes.d +
                                                  static_cast<int64_t>(shapeInfo.padNum))};

        query = l0op::Reshape(query, executor->AllocIntArray(queryShape.data(), queryShape.size()), executor);
        CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
        key = l0op::Reshape(key, executor->AllocIntArray(keyValueShape.data(), keyValueShape.size()), executor);
        CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
        value = l0op::Reshape(value, executor->AllocIntArray(keyValueShape.data(), keyValueShape.size()), executor);
        CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Postprocess(const aclTensor *&l0AttentionOutOut, const aclTensor *attentionOutOut,
                        struct FaShapeInfo &shapeInfo, aclOpExecutor *executor)
{
    if (shapeInfo.inputLayout == InputLayout::SBH && shapeInfo.needPad && !shapeInfo.needTranspose) {
        // (S,B,Hp) -> (S,B,N,Dp)
        FVector<int64_t, DIM_NUM_4> paddedSBNDShape{shapeInfo.axes.s1, shapeInfo.axes.b, shapeInfo.axes.n1,
                                                    shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum)};
        l0AttentionOutOut = l0op::Reshape(
            l0AttentionOutOut, executor->AllocIntArray(paddedSBNDShape.data(), paddedSBNDShape.size()), executor);
        CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (shapeInfo.needTranspose) {
        auto perm = executor->AllocIntArray(shapeInfo.perm_out.data(), shapeInfo.perm_out.size());
        l0AttentionOutOut = l0op::Transpose(l0AttentionOutOut, perm, executor);
        CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (shapeInfo.needPad) {
        // (B,S,N,D)
        // (S,B,N,D)
        // (B,N,S,D)
        // (T,N,D)
        FVector<int64_t, MAX_DIM_NUM> sizeVec = ToShapeVector(l0AttentionOutOut->GetViewShape());
        sizeVec.back() -= shapeInfo.padNum;
        if (shapeInfo.inputLayout == InputLayout::TND) {
            FVector<int64_t, DIM_NUM_3> offsetVec(DIM_NUM_3, 0);
            l0AttentionOutOut =
                l0op::Slice(l0AttentionOutOut, executor->AllocIntArray(offsetVec.data(), offsetVec.size()),
                            executor->AllocIntArray(sizeVec.data(), sizeVec.size()), executor);
        } else {
            FVector<int64_t, DIM_NUM_4> offsetVec(DIM_NUM_4, 0);
            l0AttentionOutOut =
                l0op::Slice(l0AttentionOutOut, executor->AllocIntArray(offsetVec.data(), offsetVec.size()),
                            executor->AllocIntArray(sizeVec.data(), sizeVec.size()), executor);
        }
        CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (shapeInfo.needReshape) {
        auto attentionOutOutShape = ToShapeVector(attentionOutOut->GetViewShape());
        l0AttentionOutOut =
            l0op::Reshape(l0AttentionOutOut,
                          executor->AllocIntArray(attentionOutOutShape.data(), attentionOutOutShape.size()), executor);
        CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckFaParam(const aclTensor *query, const aclTensor *key, const aclTensor *value, const char *inputLayout,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *attentionOutOut,
    const uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 必须的参数指针判空
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(inputLayout != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(softmaxMaxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(softmaxSumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(attentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, double scaleValueOptional, double keepProbOptional, int64_t preTokensOptional,
    int64_t nextTokensOptional, int64_t headNum, char *inputLayout, int64_t innerPreciseOptional,
    int64_t sparseModeOptional, const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut, const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckFaParam(query, key, value, inputLayout, softmaxMaxOut, softmaxSumOut, attentionOutOut,
        workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    L2_DFX_PHASE_1(aclnnFlashAttentionScore,
                   DFX_IN(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, prefixOptional, scaleValueOptional, keepProbOptional, preTokensOptional,
                          nextTokensOptional, headNum, inputLayout, innerPreciseOptional, sparseModeOptional),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CHECK_RET(InputDtypeCheck(query, key, value, realShiftOptional, PSE_TYPE_V1) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    FaShapeInfo shapeInfo;
    CHECK_RET(AnalysisInput(query, key, inputLayout, headNum, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    CHECK_RET(Contiguous(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
                         l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(PreprocessQKV(query, key, value, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto l0FlashAttentionScoreOuts = l0op::FlashAttentionScore(
        query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, prefixOptional,
        nullptr, nullptr, nullptr, nullptr, scaleValueOptional, keepProbOptional, preTokensOptional, nextTokensOptional, headNum,
        shapeInfo.l0InputLayoutStr.c_str(), innerPreciseOptional, sparseModeOptional, PSE_TYPE_V1, l0Executor);

    CHECK_RET(l0FlashAttentionScoreOuts[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0FlashAttentionScoreOuts[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0FlashAttentionScoreOuts[3] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto l0SoftmaxMaxOut = l0FlashAttentionScoreOuts[0];
    auto l0SoftmaxSumOut = l0FlashAttentionScoreOuts[1];
    // l0SoftmaxOutOut not used now
    auto l0AttentionOutOut = l0FlashAttentionScoreOuts[3];

    CHECK_RET(Postprocess(l0AttentionOutOut, attentionOutOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // l0SoftmaxOutOut not used now
    auto viewCopyResult3 = l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionScore);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionVarLenScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, double scaleValueOptional, double keepProbOptional,
    int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum, char *inputLayout,
    int64_t innerPreciseOptional, int64_t sparseModeOptional, const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut, const aclTensor *attentionOutOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckFaParam(query, key, value, inputLayout, softmaxMaxOut, softmaxSumOut, attentionOutOut,
        workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    L2_DFX_PHASE_1(aclnnFlashAttentionVarLenScore,
                   DFX_IN(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
                          scaleValueOptional, keepProbOptional, preTokensOptional, nextTokensOptional, headNum,
                          inputLayout, innerPreciseOptional, sparseModeOptional),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Layout %s is not TND, invalid shape, please check", inputLayout);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_INVALID;
    }
    FaShapeInfo shapeInfo;
    CHECK_RET(AnalysisInput(query, key, inputLayout, headNum, shapeInfo, actualSeqQLenOptional,
                            actualSeqKvLenOptional) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    CHECK_RET(Contiguous(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
                         l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(PreprocessQKV(query, key, value, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto l0FlashAttentionScoreOuts = l0op::FlashAttentionScore(
        query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, nullptr, nullptr, scaleValueOptional, keepProbOptional,
        preTokensOptional, nextTokensOptional, headNum, shapeInfo.l0InputLayoutStr.c_str(), innerPreciseOptional,
        sparseModeOptional, PSE_TYPE_V1, l0Executor);

    auto l0SoftmaxMaxOut = l0FlashAttentionScoreOuts[0];
    auto l0SoftmaxSumOut = l0FlashAttentionScoreOuts[1];
    // l0SoftmaxOutOut not used now
    auto l0AttentionOutOut = l0FlashAttentionScoreOuts[3];

    CHECK_RET(Postprocess(l0AttentionOutOut, attentionOutOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    if (l0SoftmaxMaxOut == nullptr || l0SoftmaxSumOut == nullptr || l0AttentionOutOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "l0SoftmaxMaxOut or l0SoftmaxSumOut or l0AttentionOutOut is null");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // l0SoftmaxOutOut not used now
    auto viewCopyResult3 = l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionVarLenScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionVarLenScore);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionScoreV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    double scaleValueOptional, double keepProbOptional,int64_t preTokensOptional,
    int64_t nextTokensOptional, int64_t headNum, char *inputLayout, int64_t innerPreciseOptional,
    int64_t sparseModeOptional, int64_t pseTypeOptional, const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut, const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckFaParam(query, key, value, inputLayout, softmaxMaxOut, softmaxSumOut, attentionOutOut,
        workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    L2_DFX_PHASE_1(aclnnFlashAttentionScoreV2,
                   DFX_IN(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, prefixOptional, qStartIdxOptional, kvStartIdxOptional, scaleValueOptional,
                          keepProbOptional, preTokensOptional, nextTokensOptional, headNum, inputLayout,
                          innerPreciseOptional, sparseModeOptional, pseTypeOptional),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CHECK_RET(InputDtypeCheck(query, key, value, realShiftOptional, pseTypeOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    FaShapeInfo shapeInfo;
    CHECK_RET(AnalysisInput(query, key, inputLayout, headNum, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    CHECK_RET(Contiguous(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
                         l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(PreprocessQKV(query, key, value, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto l0FlashAttentionScoreOuts = l0op::FlashAttentionScore(
        query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, prefixOptional,
        nullptr, nullptr, qStartIdxOptional, kvStartIdxOptional, scaleValueOptional, keepProbOptional,
        preTokensOptional, nextTokensOptional, headNum, shapeInfo.l0InputLayoutStr.c_str(), innerPreciseOptional,
        sparseModeOptional, pseTypeOptional, l0Executor);

    auto l0SoftmaxMaxOut = l0FlashAttentionScoreOuts[0];
    auto l0SoftmaxSumOut = l0FlashAttentionScoreOuts[1];
    // l0SoftmaxOutOut not used now
    auto l0AttentionOutOut = l0FlashAttentionScoreOuts[3];

    CHECK_RET(Postprocess(l0AttentionOutOut, attentionOutOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // l0SoftmaxOutOut not used now
    auto viewCopyResult3 = l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionScoreV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionVarLenScoreV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, double scaleValueOptional, double keepProbOptional,
    int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum, char *inputLayout,
    int64_t innerPreciseOptional, int64_t sparseModeOptional, int64_t pseTypeOptional,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckFaParam(query, key, value, inputLayout, softmaxMaxOut, softmaxSumOut, attentionOutOut,
        workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    L2_DFX_PHASE_1(aclnnFlashAttentionVarLenScoreV2,
                   DFX_IN(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
                          qStartIdxOptional, kvStartIdxOptional, scaleValueOptional, keepProbOptional,
                          preTokensOptional, nextTokensOptional, headNum, inputLayout,
                          innerPreciseOptional, sparseModeOptional, pseTypeOptional),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Layout %s is not TND, invalid shape, please check", inputLayout);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_INVALID;
    }
    FaShapeInfo shapeInfo;
    CHECK_RET(InputDtypeCheck(query, key, value, realShiftOptional, pseTypeOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(AnalysisInput(query, key, inputLayout, headNum, shapeInfo, actualSeqQLenOptional,
                            actualSeqKvLenOptional) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    CHECK_RET(Contiguous(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
                         l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(PreprocessQKV(query, key, value, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto l0FlashAttentionScoreOuts = l0op::FlashAttentionScore(
        query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional, scaleValueOptional,
        keepProbOptional, preTokensOptional, nextTokensOptional, headNum, shapeInfo.l0InputLayoutStr.c_str(),
        innerPreciseOptional, sparseModeOptional, pseTypeOptional, l0Executor);

    auto l0SoftmaxMaxOut = l0FlashAttentionScoreOuts[0];
    auto l0SoftmaxSumOut = l0FlashAttentionScoreOuts[1];
    // l0SoftmaxOutOut not used now
    auto l0AttentionOutOut = l0FlashAttentionScoreOuts[3];

    CHECK_RET(Postprocess(l0AttentionOutOut, attentionOutOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    if (l0SoftmaxMaxOut == nullptr || l0SoftmaxSumOut == nullptr || l0AttentionOutOut == nullptr) {
      OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "l0SoftmaxMaxOut or l0SoftmaxSumOut or l0AttentionOutOut is null");
      *workspaceSize = 0;
      uniqueExecutor.ReleaseTo(executor);
      return ACLNN_ERR_INNER_NULLPTR;
    }
    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // l0SoftmaxOutOut not used now
    auto viewCopyResult3 = l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionVarLenScoreV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionVarLenScoreV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}  // namespace

#ifdef __cplusplus
}
#endif
