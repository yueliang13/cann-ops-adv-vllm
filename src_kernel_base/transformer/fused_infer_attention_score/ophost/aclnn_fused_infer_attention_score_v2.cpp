/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "graph/types.h"
#include "aclnn_fused_infer_attention_score_v2.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
const uint64_t INT4_NUMS_IN_INT32 = 8;

extern aclnnStatus aclnnInnerFusedInferAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1, const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blockTable, const aclTensor *queryPaddingSize, const aclTensor *kvPaddingSize,
    const aclTensor *keyAntiquantScale, const aclTensor *keyAntiquantOffset, const aclTensor *valueAntiquantScale,
    const aclTensor *valueAntiquantOffset, const aclTensor *keySharedPrefix, const aclTensor *valueSharedPrefix,
    const aclIntArray *actualSharedPrefixLen, const aclTensor *query_rope, 
    const aclTensor *key_rope, const aclTensor *keyRopeAntiquantScale,
    int64_t numHeads, double scaleValue, int64_t preTokens,
    int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode, int64_t valueAntiquantMode,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerFusedInferAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                      const aclrtStream stream);
extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void *executor, const size_t irIndex);

void TensorPreProcess(const aclTensorList *key, const aclTensorList *value, const aclTensorList *&tensorListKey,
                      const aclTensorList *&tensorListValue)
{
    if (tensorListKey == nullptr) {
        OP_LOGD("TensorListKey is nullptr,TensorPreProcess exit.");
        return;
    }
    if (tensorListValue == nullptr) {
        OP_LOGD("tensorListValue is nullptr,TensorPreProcess exit..");
        return;
    }
    if ((*tensorListKey)[0]->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kv's from OriginalShape is completed.");
        return;
    }
    if ((*tensorListValue)[0]->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kv's from OriginalShape is completed.");
        return;
    }
    auto tempKey = const_cast<aclTensorList *>(tensorListKey);
    for (uint64_t i = 0; i < tempKey->Size(); i++) {
        op::Shape viewShape = (*tempKey)[i]->GetViewShape();
        auto viewShapeDim = viewShape.GetDimNum();
        viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * INT4_NUMS_IN_INT32;
        (*tempKey)[i]->SetViewShape(viewShape);
        (*tempKey)[i]->SetDataType(DataType::DT_INT4);
    }

    auto tempValue = const_cast<aclTensorList *>(tensorListValue);
    for (uint64_t i = 0; i < tempValue->Size(); i++) {
        op::Shape viewShape = (*tempValue)[i]->GetViewShape();
        auto viewShapeDim = viewShape.GetDimNum();
        viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * INT4_NUMS_IN_INT32;
        (*tempValue)[i]->SetViewShape(viewShape);
        (*tempValue)[i]->SetDataType(DataType::DT_INT4);
    }

    OP_LOGD("The conversion of kv from int32 to int4 is completed.");
}

void PrefixTensorPreProcess(const aclTensor *keyPrefix, const aclTensor *valuePrefix, const aclTensor *&tensorKey,
                            const aclTensor *&tensorValue)
{
    if (tensorKey == nullptr) {
        OP_LOGD("TensorListKey is nullptr,TensorPreProcess exit.");
        return;
    }
    if (tensorValue == nullptr) {
        OP_LOGD("tensorListValue is nullptr,TensorPreProcess exit..");
        return;
    }
    if (tensorKey->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kvPrefix's from OriginalShape is completed.");
        return;
    }
    if (tensorValue->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kvPrefix's from OriginalShape is completed.");
        return;
    }
    auto tempKey = const_cast<aclTensor *>(tensorKey);
    op::Shape viewKeyShape = tempKey->GetViewShape();
    auto viewKeyShapeDim = viewKeyShape.GetDimNum();
    viewKeyShape[viewKeyShapeDim - 1] = viewKeyShape[viewKeyShapeDim - 1] * INT4_NUMS_IN_INT32;
    tempKey->SetViewShape(viewKeyShape);
    tempKey->SetDataType(DataType::DT_INT4);

    auto tempValue = const_cast<aclTensor *>(tensorValue);
    op::Shape viewValueShape = tempValue->GetViewShape();
    auto viewValueShapeDim = viewValueShape.GetDimNum();
    viewValueShape[viewValueShapeDim - 1] = viewValueShape[viewValueShapeDim - 1] * INT4_NUMS_IN_INT32;
    tempValue->SetViewShape(viewValueShape);
    tempValue->SetDataType(DataType::DT_INT4);

    OP_LOGD("The conversion of kvPrefix from int32 to int4 is completed.");
}


aclnnStatus aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
    const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional,
    const aclTensor *deqScale2Optional,
    const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional,
    const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional,
    const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional,
    const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional,
    const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional,
    const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional,
    int64_t numHeads, double scaleValue, int64_t preTokens,
    int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize,
    int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode, int64_t valueAntiquantMode,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const aclTensorList *tensorListKey = key;
    const aclTensorList *tensorListValue = value;
    TensorPreProcess(key, value, tensorListKey, tensorListValue);

    const aclTensor *tensorKeySharedPrefixOptional = keySharedPrefixOptional;
    const aclTensor *tensorValueSharedPrefixOptional = valueSharedPrefixOptional;
    PrefixTensorPreProcess(keySharedPrefixOptional, valueSharedPrefixOptional, tensorKeySharedPrefixOptional,
                           tensorValueSharedPrefixOptional);

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;
    if (softmaxLseFlag == false) {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), (void *)&addr);
        placeHolder = tempTensor;
    } else {
        placeHolder = softmaxLse;
    }
    aclnnStatus ret = aclnnInnerFusedInferAttentionScoreGetWorkspaceSize(
        query, tensorListKey, tensorListValue, pseShiftOptional, attenMaskOptional, actualSeqLengthsOptional,
        actualSeqLengthsKvOptional, deqScale1Optional, quantScale1Optional, deqScale2Optional, quantScale2Optional,
        quantOffset2Optional, antiquantScaleOptional, antiquantOffsetOptional, blockTableOptional,
        queryPaddingSizeOptional, kvPaddingSizeOptional, keyAntiquantScaleOptional, keyAntiquantOffsetOptional,
        valueAntiquantScaleOptional, valueAntiquantOffsetOptional, tensorKeySharedPrefixOptional,
        tensorValueSharedPrefixOptional, actualSharedPrefixLenOptional, nullptr, nullptr, nullptr,
        numHeads, scaleValue, preTokens, nextTokens,
        inputLayout, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag,
        keyAntiquantMode, valueAntiquantMode, attentionOut, placeHolder, workspaceSize, executor);
    if (ret == 0) {
        if (NnopbaseDisableOptionalInput != nullptr) {
            NnopbaseDisableOptionalInput(*executor, 24U); // 24 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 25U); // 25 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 26U); // 26 is input irIndex
        }
    }
    if (softmaxLseFlag == false) {
        aclDestroyTensor(tempTensor);
    }
    return ret;
}

aclnnStatus aclnnFusedInferAttentionScoreV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            const aclrtStream stream)
{
    return aclnnInnerFusedInferAttentionScore(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif