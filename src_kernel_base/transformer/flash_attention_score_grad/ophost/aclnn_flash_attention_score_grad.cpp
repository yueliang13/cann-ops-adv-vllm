/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_flash_attention_score_grad.h"
#include "flash_attention_score_grad.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/fast_vector.h"
#include "runtime/context.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

#define CHECK_SCALAR_TENSOR(condition)                                                                                             \
    do {                                                                                                                           \
        if (condition) {                                                                                                           \
            OP_LOGW("There is a scalar tensor in the input optional parameters, and we will treat this input parameter as null."); \
        }                                                                                                                          \
    } while (0)

typedef struct FagInShapeInfoS {
    int64_t n1Dim;
    int64_t n2Dim;
    int64_t h1Dim;
    int64_t h2Dim;
    int64_t s1Dim;
    int64_t s2Dim;
    int64_t dDim;
    int64_t alignDim;

    int64_t querySDimStrideSize;
    int64_t kvSDimStrideSize;

    std::string inputLayoutStr;

    bool needPadDimD;
    bool needTranspose;
    bool passThrowInnerFag;
    bool needBackwordReshape;
} FagInShapeInfo;

typedef struct FagShapeArrayS {
    aclIntArray *queryShapeArray = nullptr;
    aclIntArray *keyShapeArray = nullptr;
    aclIntArray *dqShapeArray = nullptr;
    aclIntArray *dkShapeArray = nullptr;
    aclIntArray *queryBwShapeArray = nullptr;
    aclIntArray *keyBwShapeArray = nullptr;
    aclIntArray *dqBwShapeArray = nullptr;
    aclIntArray *dkBwShapeArray = nullptr;
} FagShapeArray;

static constexpr int64_t ALIGN_D_DIM_SIZE = 128;
static constexpr int64_t SPARE_ALIGN_D_DIM_SIZE = 16;
static constexpr int64_t MAX_BSN_DIMS_SIZE = 65535;
static constexpr int64_t MAX_LAYOUT_SIZE = 5;
static constexpr int64_t PSE_TYPE_V1 = 1; // add and mul
static const int64_t HEAD_DIM_72 = 72;
static const int64_t HEAD_DIM_88 = 88;
static const int64_t SEQ_LEN_4096 = 4096;
static constexpr size_t MIN_DIM = 3;
static const int64_t TND_MAX_S2 = 1024;
static const int64_t TND_MAX_S1_SUM = 160 * 1024;
static const int64_t TND_MAX_DDIM = 96;

bool CheckIsNeedPad(const FagInShapeInfo &fagShape)
{
    if ((fagShape.dDim == HEAD_DIM_72 || fagShape.dDim == HEAD_DIM_88) && fagShape.s1Dim <= SEQ_LEN_4096 &&
         fagShape.s2Dim <= SEQ_LEN_4096 &&  fagShape.inputLayoutStr != "BNSD" && fagShape.inputLayoutStr != "TND" &&
         fagShape.n1Dim == fagShape.n2Dim && fagShape.needTranspose == false) {
        OP_LOGD("Scenarios that do not require pad processing");
        return false;
    }
    return true;
}

static int64_t GetSumIntArrayMaxValue(const aclIntArray *intArrayValue) {
    // 获取targetLengthsList中的最大值
    int64_t maxLength = 0;
    int64_t tmpMaxLength = 0;
    if (intArrayValue->Size() == 1) {
      maxLength = static_cast<int64_t>((*intArrayValue)[0]);
      return maxLength;
    }
    maxLength = static_cast<int64_t>((*intArrayValue)[0]);
    for (size_t i = 1; i < intArrayValue->Size(); i++) {
        tmpMaxLength = static_cast<int64_t>((*intArrayValue)[i]) - static_cast<int64_t>((*intArrayValue)[i - 1]);
        if (tmpMaxLength > maxLength) {
            maxLength = tmpMaxLength;
        }
    }
    return maxLength;
}

bool CheckTndIsNeedPad(const FagInShapeInfo &fagShape, const aclIntArray *actualSeqQLenOptional,
                       const aclIntArray *actualSeqKvLenOptional, int64_t dDim)
{
    int64_t sKvLenMax = 0;
    int64_t sQLenSum = 0;
    int64_t deterministicValue = 0;
    rtError_t retRts = rtCtxGetSysParamOpt(SYS_OPT_DETERMINISTIC, &deterministicValue);
    if (retRts != RT_ERROR_NONE) {
        OP_LOGW("Fag aclnn unable to get system param determinstic.");
        // 如果determinstic参数获取失败，则不主动去除pad
        return true;
    }
    OP_LOGD("Fag aclnn deterministic is = %ld.", deterministicValue);
    // TND并且是非确定性计算
    if (fagShape.inputLayoutStr == "TND" && deterministicValue == 0 &&
        actualSeqQLenOptional != nullptr && actualSeqKvLenOptional != nullptr) {
        if (actualSeqQLenOptional->Size() == actualSeqKvLenOptional->Size()) {
            sKvLenMax = GetSumIntArrayMaxValue(actualSeqKvLenOptional);
            sQLenSum = actualSeqQLenOptional->Size() >= 1 ?
                       static_cast<int64_t>((*actualSeqQLenOptional)[actualSeqQLenOptional->Size() - 1]) : 0;
        }
    }
    if (sKvLenMax == 0 || sQLenSum == 0) {
        // 走原来逻辑是否pad
        OP_LOGD("Fag aclnn TND case sKvLenMax(%ld) or sQLenSum(%ld) is 0.", sKvLenMax, sQLenSum);
        return true;
    }

    OP_LOGD("Fag aclnn TND case deterministic: %ld, s2 max: %ld, dDim: %ld, s1 sum: %ld.", deterministicValue,
            sKvLenMax, dDim, sQLenSum);
    if ((sKvLenMax <= TND_MAX_S2) && (dDim < TND_MAX_DDIM) && (sQLenSum < TND_MAX_S1_SUM)) {
        // 去除pad
        OP_LOGD("Fag aclnn TND case do not do pad dimD operation.");
        return false;
    }
    return true;
}

static aclnnStatus InvalidTensorDimCheck(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                         const aclTensor *dy, const aclTensor *attentionIn, const aclTensor *dq,
                                         const aclTensor *dk, const aclTensor *dv)
{
    auto queryDimNum = query->GetViewShape().GetDimNum();
    auto keyDimNum = key->GetViewShape().GetDimNum();
    auto valueDimNum = value->GetViewShape().GetDimNum();
    auto dyDimNum = dy->GetViewShape().GetDimNum();
    auto attentionInDimNum = attentionIn->GetViewShape().GetDimNum();
    auto dqDimNum = dq->GetViewShape().GetDimNum();
    auto dkDimNum = dk->GetViewShape().GetDimNum();
    auto dvDimNum = dv->GetViewShape().GetDimNum();
    if (queryDimNum < MIN_DIM || keyDimNum < MIN_DIM || valueDimNum < MIN_DIM || dyDimNum < MIN_DIM ||
        attentionInDimNum < MIN_DIM || dqDimNum < MIN_DIM || dkDimNum < MIN_DIM || dvDimNum < MIN_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The input or output of FAG does not support tensors with dim less than 3.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus GetInputShapeInfo(const aclTensor *query, const aclTensor *key, int64_t headNum,
                                     const char *inputLayout, FagInShapeInfo &fagShape,
                                     const aclIntArray *actualSeqQLenOptional,
                                     const aclIntArray *actualSeqKvLenOptional)
{
    auto queryShape = query->GetViewShape();
    auto kvShape = key->GetViewShape();
    auto queryDimSize = query->Size();
    auto kvDimSize = key->Size();
    fagShape.inputLayoutStr = op::ToString(inputLayout).GetString();
    fagShape.n1Dim = (fagShape.inputLayoutStr == "BNSD") ? queryShape.GetDim(1) : queryShape.GetDim(2); // 1 or 2:n1
    fagShape.n2Dim = (fagShape.inputLayoutStr == "BNSD") ? kvShape.GetDim(1) : kvShape.GetDim(2);       // 1 or 2:n2
    fagShape.s1Dim = (fagShape.inputLayoutStr == "BNSD") ? queryShape.GetDim(2) : queryShape.GetDim(1); // 1 or 2:s1
    fagShape.s2Dim = (fagShape.inputLayoutStr == "BNSD") ? kvShape.GetDim(2) : kvShape.GetDim(1);       // 1 or 2:s2
    if (fagShape.inputLayoutStr == "BSH" || fagShape.inputLayoutStr == "SBH") {
        fagShape.h1Dim = queryShape.GetDim(2); // 2:h1
        fagShape.h2Dim = kvShape.GetDim(2);    // 2:h2
        fagShape.dDim = fagShape.h1Dim / headNum;
        CHECK_RET(fagShape.dDim != 0, ACLNN_ERR_PARAM_INVALID);
        fagShape.n1Dim = headNum;
        fagShape.n2Dim = fagShape.h2Dim / fagShape.dDim;
        fagShape.s1Dim = (fagShape.inputLayoutStr == "BSH") ? queryShape.GetDim(1) : queryShape.GetDim(0);
        fagShape.s2Dim = (fagShape.inputLayoutStr == "BSH") ? kvShape.GetDim(1) : kvShape.GetDim(0);
    } else if (fagShape.inputLayoutStr == "TND") {
        fagShape.dDim = queryShape.GetDim(2);  // 2:d
        fagShape.n1Dim = queryShape.GetDim(1); // 1:n1
        fagShape.n2Dim = kvShape.GetDim(1);    // 1:n2
    } else if (queryShape.GetDimNum() > MIN_DIM) {
        fagShape.dDim = queryShape.GetDim(3); // 3:d
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of the tensor whose input is BNSD/BSND is less than 4.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    fagShape.querySDimStrideSize = 0;
    fagShape.kvSDimStrideSize = 0;
    if (fagShape.inputLayoutStr == "BSND") { // stride is N * D
        fagShape.querySDimStrideSize = fagShape.n1Dim * fagShape.dDim;
        fagShape.kvSDimStrideSize = fagShape.n2Dim * fagShape.dDim;
    } else if (fagShape.inputLayoutStr == "BSH") {           // stride is H
        fagShape.querySDimStrideSize = queryShape.GetDim(2); // 2:dv
        fagShape.kvSDimStrideSize = kvShape.GetDim(2);       // 2:dv
    } else if (fagShape.inputLayoutStr == "SBH") {           // stride is B * H
        fagShape.querySDimStrideSize = fagShape.s1Dim == 0 ? 0 : (queryDimSize / fagShape.s1Dim);
        fagShape.kvSDimStrideSize = fagShape.s2Dim == 0 ? 0 : (kvDimSize / fagShape.s2Dim);
    }

    fagShape.alignDim = (fagShape.dDim < ALIGN_D_DIM_SIZE) ? SPARE_ALIGN_D_DIM_SIZE : ALIGN_D_DIM_SIZE;
    auto dDimAlignSize = (fagShape.dDim + fagShape.alignDim - 1) / fagShape.alignDim * fagShape.alignDim;

    // 判断是否需要PAD和transpose, 同时判断是否为如下特殊场景 (SBH下，只需要PAD不需要transpose)
    fagShape.needPadDimD =
        (fagShape.dDim % fagShape.alignDim != 0 && queryShape.GetShapeSize() != 0 && kvShape.GetShapeSize() != 0) ?
            true :
            false;

    // 计算是否超过65535时，应该使用对齐以后的D值
    if (fagShape.needPadDimD) {
        if (fagShape.inputLayoutStr == "BSND") { // stride is N * D
            fagShape.querySDimStrideSize = fagShape.n1Dim * dDimAlignSize;
            fagShape.kvSDimStrideSize = fagShape.n2Dim * dDimAlignSize;
        } else if (fagShape.inputLayoutStr == "BSH") {           // stride is H
            fagShape.querySDimStrideSize = fagShape.dDim == 0 ? 0 :
                (queryShape.GetDim(2) / fagShape.dDim * dDimAlignSize); // 2:dv
            fagShape.kvSDimStrideSize = fagShape.dDim == 0 ? 0 :
                (kvShape.GetDim(2) / fagShape.dDim * dDimAlignSize);       // 2:dv
        } else if (fagShape.inputLayoutStr == "SBH") {           // stride is B * H
            int64_t queryBHSize = fagShape.s1Dim == 0 ? 0 : (queryDimSize / fagShape.s1Dim);
            int64_t kvBHSize = fagShape.s2Dim == 0 ? 0 : (kvDimSize / fagShape.s2Dim);
            fagShape.querySDimStrideSize = fagShape.dDim == 0 ? 0 : (queryBHSize / fagShape.dDim * dDimAlignSize);
            fagShape.kvSDimStrideSize = fagShape.dDim == 0 ? 0 : (kvBHSize / fagShape.dDim * dDimAlignSize);
        }
    }

    bool needTranspose =
        queryShape.GetShapeSize() != 0 && kvShape.GetShapeSize() != 0 &&
        (fagShape.inputLayoutStr != "BNSD" && fagShape.inputLayoutStr != "TND" &&
         (fagShape.querySDimStrideSize > MAX_BSN_DIMS_SIZE || fagShape.kvSDimStrideSize > MAX_BSN_DIMS_SIZE));
    fagShape.needTranspose = needTranspose;

    if (!CheckIsNeedPad(fagShape) ||
        !CheckTndIsNeedPad(fagShape, actualSeqQLenOptional, actualSeqKvLenOptional, fagShape.dDim)) {
        fagShape.needPadDimD = false;
    }

    fagShape.passThrowInnerFag = (!(fagShape.needPadDimD) && !(fagShape.needTranspose));
    fagShape.needBackwordReshape =
        (fagShape.inputLayoutStr == "SBH" && fagShape.needPadDimD && !(fagShape.needTranspose));
    return ACLNN_SUCCESS;
}

static inline aclnnStatus ContiguousTensorWithCheck(const aclTensor *inputTensor, const aclTensor **outTensor,
                                                    aclOpExecutor *executor)
{
    if (inputTensor != nullptr && inputTensor->GetViewShape().GetDimNum() != 0) {
        *outTensor = l0op::Contiguous(inputTensor, executor);
        CHECK_RET(*outTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 输入入参如果是标量tensor，将会按照此可选输入为null处理 ;
    CHECK_SCALAR_TENSOR(inputTensor != nullptr && inputTensor->GetViewShape().GetDimNum() == 0);

    return ACLNN_SUCCESS;
}

static inline void ConvertInputLayout(FagInShapeInfo fagShape, const char *inputLayout, char *inputLayoutUnderTrans,
                                      size_t layoutUnderTransSize)
{
    if (fagShape.needTranspose) {                  // 1. 只要是需要transpose，输入FAG layout必然是BNSD
        inputLayoutUnderTrans[0] = 'B';            // 0: 'B'
        inputLayoutUnderTrans[1] = 'N';            // 1: 'N'
        inputLayoutUnderTrans[2] = 'S';            // 2: 'S'
        inputLayoutUnderTrans[3] = 'D';            // 3: 'D'
    } else if (fagShape.needBackwordReshape) {     // 2. 如果是SBH仅PAD场景，输入FAG layout必然还是SBH
        inputLayoutUnderTrans[0] = inputLayout[0]; // 0: 'S'
        inputLayoutUnderTrans[1] = inputLayout[1]; // 1: 'B'
        inputLayoutUnderTrans[2] = 'H';            // 2: 'H'
    } else if (fagShape.needPadDimD) { // 3. 如果是仅PAD场景，根据BSH/SBH/BNSD/BSND自适应reshape后的layout
        /* BSH  -> BSND
           SBH  -> SBND
           TND  -> TND
           BNSD -> BNSD
           BSND -> BSND */
        for (size_t i = 0; i < strlen(inputLayout) && i < layoutUnderTransSize - 1; i++) {
            if (inputLayout[i] == 'H') {
                inputLayoutUnderTrans[i] = 'N';
                inputLayoutUnderTrans[i + 1] = 'D';
                break;
            }
            inputLayoutUnderTrans[i] = inputLayout[i];
        }
    } else { // 4. 其他情况，保持原始layout
        for (size_t i = 0; i < strlen(inputLayout) && i < layoutUnderTransSize - 1; i++) {
            inputLayoutUnderTrans[i] = inputLayout[i];
        }
    }
}

static aclnnStatus ContiguousInputTensor(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                         const aclTensor *dy, const aclTensor *attentionInOptional,
                                         const aclTensor **queryCngs, const aclTensor **keyCngs,
                                         const aclTensor **valueCngs, const aclTensor **dyCngs,
                                         const aclTensor **attentionInOptionalCngs, aclOpExecutor *executor)
{
    auto ret = ACLNN_SUCCESS;

    // query如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(query, queryCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // key如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(key, keyCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // value如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(value, valueCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // dy如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(dy, dyCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // attentionInOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(attentionInOptional, attentionInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    return ret;
}

static aclnnStatus ContiguousOptionalInputTensor(
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor **pseShiftOptionalCngs, const aclTensor **dropMaskOptionalCngs,
    const aclTensor **paddingMaskOptionalCngs, const aclTensor **attenMaskOptionalCngs,
    const aclTensor **softmaxMaxOptionalCngs, const aclTensor **softmaxSumOptionalCngs,
    const aclTensor **softmaxInOptionalCngs, aclOpExecutor *executor)
{
    auto ret = ACLNN_SUCCESS;

    // pseShiftOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(pseShiftOptional, pseShiftOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // dropMaskOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(dropMaskOptional, dropMaskOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // paddingMaskOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(paddingMaskOptional, paddingMaskOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // attenMaskOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(attenMaskOptional, attenMaskOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // softmaxMaxOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(softmaxMaxOptional, softmaxMaxOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // softmaxSumOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(softmaxSumOptional, softmaxSumOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // softmaxInOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(softmaxInOptional, softmaxInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    return ret;
}

static void GetInputAndOutputReshapeArray(const aclTensor *query, const aclTensor *key, FagInShapeInfo fagShape,
                                          FagShapeArray &fagShapeArray, aclOpExecutor *executor)
{
    if (fagShape.passThrowInnerFag) {
        return;
    }

    if (fagShape.inputLayoutStr != "BSH" && fagShape.inputLayoutStr != "SBH") {
        return;
    }

    auto queryShape = query->GetViewShape();
    auto keyShape = key->GetViewShape();
    FVector<int64_t> queryReshapeList;
    FVector<int64_t> keyReshapeList;
    FVector<int64_t> dqReshapeList;
    FVector<int64_t> dkReshapeList;
    for (size_t i = 0; i < 3; i++) { // 3: sizeof("BSH")
        dqReshapeList.emplace_back(queryShape.GetDim(i));
        dkReshapeList.emplace_back(keyShape.GetDim(i));
        if (i < 2) { // 2: split last Dim
            queryReshapeList.emplace_back(queryShape.GetDim(i));
            keyReshapeList.emplace_back(keyShape.GetDim(i));
        }
    }

    queryReshapeList.emplace_back(fagShape.n1Dim);
    queryReshapeList.emplace_back(fagShape.dDim);
    keyReshapeList.emplace_back(fagShape.n2Dim);
    keyReshapeList.emplace_back(fagShape.dDim);

    // get shape array
    fagShapeArray.queryShapeArray = executor->AllocIntArray(queryReshapeList.data(), queryReshapeList.size());
    fagShapeArray.dqShapeArray = executor->AllocIntArray(dqReshapeList.data(), dqReshapeList.size());
    fagShapeArray.keyShapeArray = executor->AllocIntArray(keyReshapeList.data(), keyReshapeList.size());
    fagShapeArray.dkShapeArray = executor->AllocIntArray(dkReshapeList.data(), dkReshapeList.size());

    return;
}

static void GetInputAndOutputBackwordReshapeArrayForSBH(const aclTensor *query, const aclTensor *key,
                                                        FagInShapeInfo fagShape, FagShapeArray &fagShapeArray,
                                                        aclOpExecutor *executor)
{
    if (!(fagShape.needBackwordReshape)) {
        return;
    }

    if (query == nullptr || key == nullptr) {
        return;
    }
    auto queryShape = query->GetViewShape();
    auto keyShape = key->GetViewShape();
    FVector<int64_t> queryReshapeList;
    FVector<int64_t> keyReshapeList;
    FVector<int64_t> dqReshapeList;
    FVector<int64_t> dkReshapeList;
    for (size_t i = 0; i < 2; i++) { // 2: get SBH pre shape size 'SB'
        queryReshapeList.emplace_back(queryShape.GetDim(i));
        dqReshapeList.emplace_back(queryShape.GetDim(i));
        keyReshapeList.emplace_back(keyShape.GetDim(i));
        dkReshapeList.emplace_back(keyShape.GetDim(i));
    }

    auto dDimAlignSize = (fagShape.dDim + fagShape.alignDim - 1) / fagShape.alignDim * fagShape.alignDim;
    auto queryHDimAlignSize = fagShape.n1Dim * dDimAlignSize;
    auto keyHDimAlignSize = fagShape.n2Dim * dDimAlignSize;

    queryReshapeList.emplace_back(queryHDimAlignSize);
    keyReshapeList.emplace_back(keyHDimAlignSize);

    dqReshapeList.emplace_back(fagShape.n1Dim);
    dqReshapeList.emplace_back(dDimAlignSize);
    dkReshapeList.emplace_back(fagShape.n2Dim);
    dkReshapeList.emplace_back(dDimAlignSize);

    // get shape array
    fagShapeArray.queryBwShapeArray = executor->AllocIntArray(queryReshapeList.data(), queryReshapeList.size());
    fagShapeArray.dqBwShapeArray = executor->AllocIntArray(dqReshapeList.data(), dqReshapeList.size());
    fagShapeArray.keyBwShapeArray = executor->AllocIntArray(keyReshapeList.data(), keyReshapeList.size());
    fagShapeArray.dkBwShapeArray = executor->AllocIntArray(dkReshapeList.data(), dkReshapeList.size());

    return;
}

static aclnnStatus ReshapeInputTensor(const aclTensor **query, const aclTensor **key, const aclTensor **value,
                                      const aclTensor **dy, const aclTensor **attentionInOptional,
                                      FagInShapeInfo fagShape, FagShapeArray fagShapeArray, bool isBackWord,
                                      aclOpExecutor *executor)
{
    bool needReshape = isBackWord ? fagShape.needBackwordReshape : !(fagShape.passThrowInnerFag);
    if (!needReshape) {
        return ACLNN_SUCCESS;
    }

    if (fagShape.inputLayoutStr != "BSH" && fagShape.inputLayoutStr != "SBH") {
        return ACLNN_SUCCESS;
    }

    auto queryShapeArray = isBackWord ? fagShapeArray.queryBwShapeArray : fagShapeArray.queryShapeArray;
    auto keyShapeArray = isBackWord ? fagShapeArray.keyBwShapeArray : fagShapeArray.keyShapeArray;

    // reshape input
    *query = l0op::Reshape(*query, queryShapeArray, executor);
    CHECK_RET(*query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *key = l0op::Reshape(*key, keyShapeArray, executor);
    CHECK_RET(*key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *value = l0op::Reshape(*value, keyShapeArray, executor);
    CHECK_RET(*value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *dy = l0op::Reshape(*dy, queryShapeArray, executor);
    CHECK_RET(*dy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (*attentionInOptional != nullptr && (*attentionInOptional)->GetViewShape().GetDimNum() != 0) {
        *attentionInOptional = l0op::Reshape(*attentionInOptional, queryShapeArray, executor);
        CHECK_RET(*attentionInOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ReshapeOutputTensor(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut,
                                       FagInShapeInfo fagShape, FagShapeArray fagShapeArray, bool isBackWord,
                                       aclOpExecutor *executor)
{
    bool needReshape = isBackWord ? fagShape.needBackwordReshape : !(fagShape.passThrowInnerFag);
    if (!needReshape) {
        return ACLNN_SUCCESS;
    }

    if (fagShape.inputLayoutStr != "BSH" && fagShape.inputLayoutStr != "SBH") {
        return ACLNN_SUCCESS;
    }

    aclIntArray *dqShapeArray = isBackWord ? fagShapeArray.dqBwShapeArray : fagShapeArray.dqShapeArray;
    aclIntArray *dkShapeArray = isBackWord ? fagShapeArray.dkBwShapeArray : fagShapeArray.dkShapeArray;

    // reshape
    fagOut[0] = l0op::Reshape(fagOut[0], dqShapeArray, executor);
    CHECK_RET(fagOut[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    fagOut[1] = l0op::Reshape(fagOut[1], dkShapeArray, executor);
    CHECK_RET(fagOut[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    fagOut[2] = l0op::Reshape(fagOut[2], dkShapeArray, executor); // 2:dv
    CHECK_RET(fagOut[2] != nullptr, ACLNN_ERR_INNER_NULLPTR);     // 2:dv

    return ACLNN_SUCCESS;
}

static aclnnStatus PaddingInputTensorDdim(const aclTensor **query, const aclTensor **key, const aclTensor **value,
                                          const aclTensor **dy, const aclTensor **attentionInOptional,
                                          FagInShapeInfo fagShape, aclOpExecutor *executor)
{
    if (!(fagShape.needPadDimD)) {
        OP_LOGD("Fag aclnn case do not do pad dimD operation.");
        return ACLNN_SUCCESS;
    }
    OP_LOGD("Fag aclnn case do pad dimD operation.");

    // padding
    // query
    auto padSize = (fagShape.dDim + fagShape.alignDim - 1) / fagShape.alignDim * fagShape.alignDim - fagShape.dDim;
    aclIntArray *paddingArray = nullptr;
    if (fagShape.inputLayoutStr == "TND") {
        FVector<int64_t> padding = {0, 0, 0, 0, 0, padSize};
        paddingArray = executor->AllocIntArray(padding.data(), 6); // 6: TND 3dims, padding D dim
    } else {
        FVector<int64_t> padding = {0, 0, 0, 0, 0, 0, 0, padSize};
        paddingArray = executor->AllocIntArray(padding.data(), 8); // 8: BNSD 4dims, padding D dim
    }
    auto padTensor = executor->ConvertToTensor(paddingArray, DataType::DT_INT64);
    CHECK_RET(padTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *query = l0op::Pad(*query, padTensor, executor);
    CHECK_RET(*query != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // key
    *key = l0op::Pad(*key, padTensor, executor);
    CHECK_RET(*key != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // value
    *value = l0op::Pad(*value, padTensor, executor);
    CHECK_RET(*value != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // dy
    *dy = l0op::Pad(*dy, padTensor, executor);
    CHECK_RET(*dy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // attenmask_in
    if (*attentionInOptional != nullptr && (*attentionInOptional)->GetViewShape().GetDimNum() != 0) {
        *attentionInOptional = l0op::Pad(*attentionInOptional, padTensor, executor);
        CHECK_RET(*attentionInOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus SliceOutputTensorDdim(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut,
                                         FagInShapeInfo fagShape, aclOpExecutor *executor)
{
    if (!(fagShape.needPadDimD)) {
        return ACLNN_SUCCESS;
    }

    auto dqOutShape = (fagOut[0])->GetViewShape(); // 0: dq
    auto dkOutShape = (fagOut[1])->GetViewShape(); // 1: dk

    // slice
    FVector<int64_t> dqOutSizeVector;
    FVector<int64_t> dkOutSizeVector;
    for (size_t i = 0; i < dqOutShape.GetDimNum() - 1; i++) {
        dqOutSizeVector.emplace_back(dqOutShape.GetDim(i));
    }

    for (size_t i = 0; i < dkOutShape.GetDimNum() - 1; i++) {
        dkOutSizeVector.emplace_back(dkOutShape.GetDim(i));
    }

    aclIntArray *offsets = nullptr;
    if (fagShape.inputLayoutStr == "TND") {
        FVector<int64_t> offsetsVector = {0, 0, 0};
        offsets = executor->AllocIntArray(offsetsVector.data(), offsetsVector.size());
    } else {
        FVector<int64_t> offsetsVector = {0, 0, 0, 0};
        offsets = executor->AllocIntArray(offsetsVector.data(), offsetsVector.size());
    }

    dqOutSizeVector.emplace_back(fagShape.dDim);
    auto dqOutSize = executor->AllocIntArray(dqOutSizeVector.data(), dqOutSizeVector.size());
    fagOut[0] = l0op::Slice(fagOut[0], offsets, dqOutSize, executor); // 0: dq
    CHECK_RET(fagOut[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    dkOutSizeVector.emplace_back(fagShape.dDim);
    auto dkOutSize = executor->AllocIntArray(dkOutSizeVector.data(), dkOutSizeVector.size());
    fagOut[1] = l0op::Slice(fagOut[1], offsets, dkOutSize, executor); // 1: dk
    CHECK_RET(fagOut[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    fagOut[2] = l0op::Slice(fagOut[2], offsets, dkOutSize, executor); // 2: dv
    CHECK_RET(fagOut[2] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus TransposeInputTensor(const aclTensor **query, const aclTensor **key, const aclTensor **value,
                                        const aclTensor **dy, const aclTensor **attentionInOptional,
                                        FagInShapeInfo fagShape, aclOpExecutor *executor)
{
    if (!(fagShape.needTranspose)) {
        return ACLNN_SUCCESS;
    }

    if (fagShape.inputLayoutStr == "BNSD" || fagShape.inputLayoutStr == "TND") {
        return ACLNN_SUCCESS;
    }

    FVector<int64_t> transposeDim;
    if (fagShape.inputLayoutStr == "BSH" || fagShape.inputLayoutStr == "BSND") {
        transposeDim = {0, 2, 1, 3};
    } else {
        transposeDim = {1, 2, 0, 3};
    }

    auto perm = executor->AllocIntArray(transposeDim.data(), transposeDim.size());

    // query
    *query = l0op::Transpose(*query, perm, executor);
    CHECK_RET(*query != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // key
    *key = l0op::Transpose(*key, perm, executor);
    CHECK_RET(*key != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // value
    *value = l0op::Transpose(*value, perm, executor);
    CHECK_RET(*value != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // dy
    *dy = l0op::Transpose(*dy, perm, executor);
    CHECK_RET(*dy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // attentionInOptional
    if (*attentionInOptional != nullptr && (*attentionInOptional)->GetViewShape().GetDimNum() != 0) {
        *attentionInOptional = l0op::Transpose(*attentionInOptional, perm, executor);
        CHECK_RET(*attentionInOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus TransposeOutputTensor(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut,
                                         FagInShapeInfo fagShape, aclOpExecutor *executor)
{
    if (!(fagShape.needTranspose)) {
        return ACLNN_SUCCESS;
    }

    if (fagShape.inputLayoutStr == "BNSD" || fagShape.inputLayoutStr == "TND") {
        return ACLNN_SUCCESS;
    }

    FVector<int64_t> transposeDim;
    if (fagShape.inputLayoutStr == "BSH" || fagShape.inputLayoutStr == "BSND") {
        transposeDim = {0, 2, 1, 3};
    } else {
        transposeDim = {2, 0, 1, 3};
    }

    auto perm = executor->AllocIntArray(transposeDim.data(), transposeDim.size());

    // dqOut
    fagOut[0] = l0op::Transpose(fagOut[0], perm, executor);
    CHECK_RET(fagOut[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // dkOut
    fagOut[1] = l0op::Transpose(fagOut[1], perm, executor);
    CHECK_RET(fagOut[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // dvOut
    fagOut[2] = l0op::Transpose(fagOut[2], perm, executor);   // 2:dvOut
    CHECK_RET(fagOut[2] != nullptr, ACLNN_ERR_INNER_NULLPTR); // 2:dvOut

    // dpseOut
    return ACLNN_SUCCESS;
}

static aclnnStatus PreFlashAttentionScoreGrad(const aclTensor **query, const aclTensor **key, const aclTensor **value,
                                              const aclTensor **dy, const aclTensor **attentionInOptional,
                                              FagInShapeInfo fagShape, FagShapeArray &fagShapeArray,
                                              aclOpExecutor *executor)
{
    // 获取reshape array, SBH特殊场景下，需要提前获取调用FAG前反向reshape成SBH时所需的reshape array
    GetInputAndOutputReshapeArray(*query, *key, fagShape, fagShapeArray, executor);
    GetInputAndOutputBackwordReshapeArrayForSBH(*query, *key, fagShape, fagShapeArray, executor);

    // 将输入tensor从三维扩展成四维
    auto ret = ReshapeInputTensor(query, key, value, dy, attentionInOptional, fagShape, fagShapeArray, false, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 执行D轴Padding到对齐值
    ret = PaddingInputTensorDdim(query, key, value, dy, attentionInOptional, fagShape, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 执行输入transpose到BNSD
    ret = TransposeInputTensor(query, key, value, dy, attentionInOptional, fagShape, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果是SBH特殊场景，在调用FAG前，需要将SBND重新改成SBH，否则FAG将报错不支持layout
    ret = ReshapeInputTensor(query, key, value, dy, attentionInOptional, fagShape, fagShapeArray, true, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

static aclnnStatus PostFlashAttentionScoreGrad(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut,
                                               const aclTensor **dqOut, const aclTensor **dkOut,
                                               const aclTensor **dvOut, const aclTensor **dpseOut,
                                               FagInShapeInfo fagShape, FagShapeArray &fagShapeArray,
                                               aclOpExecutor *executor)
{
    // 如果是SBH特殊场景，在调用FAG后，需要将SBH重新改成SBND，以完成后续的slice等操作
    auto ret = ReshapeOutputTensor(fagOut, fagShape, fagShapeArray, true, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 将输出由BNSD转为原始shape
    ret = TransposeOutputTensor(fagOut, fagShape, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 将D轴padding脏数据切掉
    ret = SliceOutputTensorDdim(fagOut, fagShape, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 将输出tensor由四维还原成三维
    ret = ReshapeOutputTensor(fagOut, fagShape, fagShapeArray, false, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto dqOutViewCopyRes = l0op::ViewCopy(fagOut[0], *dqOut, executor);
    CHECK_RET(dqOutViewCopyRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dkOutViewCopyRes = l0op::ViewCopy(fagOut[1], *dkOut, executor);
    CHECK_RET(dkOutViewCopyRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dvOutViewCopyRes = l0op::ViewCopy(fagOut[2], *dvOut, executor);
    CHECK_RET(dvOutViewCopyRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (*dpseOut == nullptr || (*dpseOut)->GetDataType() == ge::DataType::DT_FLOAT) {
        return ACLNN_SUCCESS;
    }

    auto dpseOutViewCopyRes = l0op::ViewCopy(fagOut[3], *dpseOut, executor);
    CHECK_RET(dpseOutViewCopyRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus FlashAttentionScoreGradGetWorkspace(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional, double scaleValueOptional,
    double keepProbOptional, int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum,
    char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional, const aclTensor* dqOut,
    const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut, uint64_t *workspaceSize,
    aclOpExecutor *executor) {
    // 检查tensor维度是否大于2
    auto ret = InvalidTensorDimCheck(query, key, value, dy, attentionInOptional, dqOut, dkOut, dvOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 获取基本参数
    FagInShapeInfo fagShape;
    ret = GetInputShapeInfo(query, key, headNum, inputLayout, fagShape, actualSeqQLenOptional, actualSeqKvLenOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 输入连续性转换
    const aclTensor *queryCngs = nullptr;
    const aclTensor *keyCngs = nullptr;
    const aclTensor *valueCngs = nullptr;
    const aclTensor *dyCngs = nullptr;
    const aclTensor *attentionInOptionalCngs = nullptr;
    const aclTensor *pseShiftOptionalCngs = nullptr;
    const aclTensor *dropMaskOptionalCngs = nullptr;
    const aclTensor *paddingMaskOptionalCngs = nullptr;
    const aclTensor *attenMaskOptionalCngs = nullptr;
    const aclTensor *softmaxMaxOptionalCngs = nullptr;
    const aclTensor *softmaxSumOptionalCngs = nullptr;
    const aclTensor *softmaxInOptionalCngs = nullptr;
    ret = ContiguousInputTensor(query, key, value, dy, attentionInOptional, &queryCngs, &keyCngs, &valueCngs, &dyCngs,
                                &attentionInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = ContiguousOptionalInputTensor(
        pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional,
        softmaxSumOptional, softmaxInOptional, &pseShiftOptionalCngs, &dropMaskOptionalCngs, &paddingMaskOptionalCngs,
        &attenMaskOptionalCngs, &softmaxMaxOptionalCngs, &softmaxSumOptionalCngs, &softmaxInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // reshape + PAD + Transpose
    FagShapeArray fagShapeArray;
    ret = PreFlashAttentionScoreGrad(&queryCngs, &keyCngs, &valueCngs, &dyCngs, &attentionInOptionalCngs, fagShape,
                                     fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 调整input layout
    char inputLayoutUnderTrans[MAX_LAYOUT_SIZE] = {0};
    ConvertInputLayout(fagShape, inputLayout, inputLayoutUnderTrans, MAX_LAYOUT_SIZE);

    // 调用FAG ascendc接口
    auto fagRes = l0op::FlashAttentionScoreGrad(
        queryCngs, keyCngs, valueCngs, dyCngs, pseShiftOptionalCngs, dropMaskOptionalCngs, paddingMaskOptionalCngs,
        attenMaskOptionalCngs, softmaxMaxOptionalCngs, softmaxSumOptionalCngs, softmaxInOptionalCngs,
        attentionInOptionalCngs, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, nullptr, nullptr,
        scaleValueOptional, keepProbOptional, preTokensOptional, nextTokensOptional, headNum, inputLayoutUnderTrans,
        innerPreciseOptional, sparseModeOptional, PSE_TYPE_V1, executor);
    CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr,  // 0: dqOut 1: dkOut 2:dvOut
              ACLNN_ERR_INNER_NULLPTR);

    // transpose + slice + reshape + viewCopy
    ret = PostFlashAttentionScoreGrad(fagRes, &dqOut, &dkOut, &dvOut, &dpseOut, fagShape, fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    double scaleValueOptional, double keepProbOptional, int64_t preTokensOptional, int64_t nextTokensOptional,
    int64_t headNum, char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFlashAttentionScoreGrad,
                   DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional,
                          attentionInOptional, prefixOptional, scaleValueOptional, keepProbOptional, preTokensOptional,
                          nextTokensOptional, headNum, inputLayout, innerPreciseOptional, sparseModeOptional),
                   DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(keyIn != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(attentionInOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dqOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dkOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dvOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 空Tensor处理
    if (dqOut->IsEmpty() && dkOut->IsEmpty() && dvOut->IsEmpty()) {
        if (dpseOut == nullptr || dpseOut->IsEmpty()) {
            OP_LOGD("All out tensor is empty");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    // 异常防护
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid HeadNum, pls check input attr");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // calculate fag
    auto ret = FlashAttentionScoreGradGetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional, nullptr,
        nullptr, scaleValueOptional, keepProbOptional, preTokensOptional, nextTokensOptional, headNum, inputLayout,
        innerPreciseOptional, sparseModeOptional, dqOut, dkOut, dvOut, dpseOut, workspaceSize, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionScoreGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional, double scaleValueOptional,
    double keepProbOptional, int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum,
    char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional, const aclTensor *dqOut,
    const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFlashAttentionUnpaddingScoreGrad,
                   DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional,
                          attentionInOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
                          scaleValueOptional, keepProbOptional, preTokensOptional, nextTokensOptional, headNum,
                          inputLayout, innerPreciseOptional, sparseModeOptional),
                   DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    // layout检查
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout %s is not TND, invalid shape, pls check", inputLayout);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(keyIn != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(attentionInOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dqOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dkOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dvOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (dqOut->IsEmpty() && dkOut->IsEmpty() && dvOut->IsEmpty()) {
        if (dpseOut == nullptr || dpseOut->IsEmpty()) {
            OP_LOGD("All out tensor is empty");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    // 异常防护
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid HeadNum, pls check input attr");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // calculate fag
    auto ret = FlashAttentionScoreGradGetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, scaleValueOptional, keepProbOptional, preTokensOptional,
        nextTokensOptional, headNum, inputLayout, innerPreciseOptional, sparseModeOptional, dqOut, dkOut, dvOut,
        dpseOut, workspaceSize, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionUnpaddingScoreGrad);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}


static aclnnStatus FlashAttentionScoreGradV2GetWorkspace(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValueOptional,
    double keepProbOptional, int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum,
    char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional, int64_t pseTypeOptional,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor *executor) {
    // 检查tensor维度是否大于2
    auto ret = InvalidTensorDimCheck(query, key, value, dy, attentionInOptional, dqOut, dkOut, dvOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 获取基本参数
    FagInShapeInfo fagShape;
    ret = GetInputShapeInfo(query, key, headNum, inputLayout, fagShape, actualSeqQLenOptional, actualSeqKvLenOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 输入连续性转换
    const aclTensor *queryCngs = nullptr;
    const aclTensor *keyCngs = nullptr;
    const aclTensor *valueCngs = nullptr;
    const aclTensor *dyCngs = nullptr;
    const aclTensor *attentionInOptionalCngs = nullptr;
    const aclTensor *pseShiftOptionalCngs = nullptr;
    const aclTensor *dropMaskOptionalCngs = nullptr;
    const aclTensor *paddingMaskOptionalCngs = nullptr;
    const aclTensor *attenMaskOptionalCngs = nullptr;
    const aclTensor *softmaxMaxOptionalCngs = nullptr;
    const aclTensor *softmaxSumOptionalCngs = nullptr;
    const aclTensor *softmaxInOptionalCngs = nullptr;
    ret = ContiguousInputTensor(query, key, value, dy, attentionInOptional, &queryCngs, &keyCngs, &valueCngs, &dyCngs,
                                &attentionInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = ContiguousOptionalInputTensor(
        pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional,
        softmaxSumOptional, softmaxInOptional, &pseShiftOptionalCngs, &dropMaskOptionalCngs, &paddingMaskOptionalCngs,
        &attenMaskOptionalCngs, &softmaxMaxOptionalCngs, &softmaxSumOptionalCngs, &softmaxInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // reshape + PAD + Transpose
    FagShapeArray fagShapeArray;
    ret = PreFlashAttentionScoreGrad(&queryCngs, &keyCngs, &valueCngs, &dyCngs, &attentionInOptionalCngs, fagShape,
                                     fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 调整input layout
    char inputLayoutUnderTrans[MAX_LAYOUT_SIZE] = {0};
    ConvertInputLayout(fagShape, inputLayout, inputLayoutUnderTrans, MAX_LAYOUT_SIZE);

    // 调用FAG ascendc接口
    auto fagRes = l0op::FlashAttentionScoreGrad(
        queryCngs, keyCngs, valueCngs, dyCngs, pseShiftOptionalCngs, dropMaskOptionalCngs, paddingMaskOptionalCngs,
        attenMaskOptionalCngs, softmaxMaxOptionalCngs, softmaxSumOptionalCngs, softmaxInOptionalCngs,
        attentionInOptionalCngs, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional,
        kvStartIdxOptional, scaleValueOptional, keepProbOptional, preTokensOptional, nextTokensOptional, headNum,
        inputLayoutUnderTrans, innerPreciseOptional, sparseModeOptional, pseTypeOptional, executor);
    CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr,  // 0: dqOut 1: dkOut 2:dvOut
              ACLNN_ERR_INNER_NULLPTR);

    // transpose + slice + reshape + viewCopy
    ret = PostFlashAttentionScoreGrad(fagRes, &dqOut, &dkOut, &dvOut, &dpseOut, fagShape, fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValueOptional,
    double keepProbOptional, int64_t preTokensOptional, int64_t nextTokensOptional,
    int64_t headNum, char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional,
    int64_t pseTypeOptional, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnFlashAttentionScoreGradV2,
                   DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional,
                          attentionInOptional, prefixOptional, qStartIdxOptional, kvStartIdxOptional,
                          scaleValueOptional, keepProbOptional, preTokensOptional, nextTokensOptional, headNum,
                          inputLayout, innerPreciseOptional, sparseModeOptional, pseTypeOptional),
                   DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(keyIn != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(attentionInOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dqOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dkOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dvOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (dqOut->IsEmpty() && dkOut->IsEmpty() && dvOut->IsEmpty()) {
        if (dpseOut == nullptr || dpseOut->IsEmpty()) {
            OP_LOGD("All out tensor is empty");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    // 异常防护
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid HeadNum, pls check input attr");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // calculate fag
    auto ret = FlashAttentionScoreGradV2GetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional, nullptr,
        nullptr, qStartIdxOptional, kvStartIdxOptional, scaleValueOptional, keepProbOptional, preTokensOptional,
        nextTokensOptional, headNum, inputLayout, innerPreciseOptional, sparseModeOptional, pseTypeOptional, dqOut,
        dkOut, dvOut, dpseOut, workspaceSize, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnFlashAttentionScoreGradV2);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValueOptional,
    double keepProbOptional, int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum,
    char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional, int64_t pseTypeOptional,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnFlashAttentionUnpaddingScoreGradV2,
        DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
               softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
               actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional, scaleValueOptional,
               keepProbOptional, preTokensOptional, nextTokensOptional, headNum, inputLayout, innerPreciseOptional,
               sparseModeOptional, pseTypeOptional),
        DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    // layout检查
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout %s is not TND, invalid shape, pls check", inputLayout);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(keyIn != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(attentionInOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dqOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dkOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dvOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (dqOut->IsEmpty() && dkOut->IsEmpty() && dvOut->IsEmpty()) {
        if (dpseOut == nullptr || dpseOut->IsEmpty()) {
            OP_LOGD("All out tensor is empty");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    // 异常防护
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid HeadNum, pls check input attr");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // calculate fag
    auto ret = FlashAttentionScoreGradV2GetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional, scaleValueOptional,
        keepProbOptional, preTokensOptional, nextTokensOptional, headNum, inputLayout, innerPreciseOptional,
        sparseModeOptional, pseTypeOptional, dqOut, dkOut, dvOut, dpseOut, workspaceSize, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    const aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnFlashAttentionUnpaddingScoreGradV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
