/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FlashAttentionScore);

const std::array<const aclTensor *, 4>
FlashAttentionScore(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                    const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,
                    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
                    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
                    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
                    const aclIntArray *kvStartIdxOptional, double scaleValueOptional, double keepProbOptional,
                    int64_t preTockensOptional, int64_t nextTockensOptional, int64_t headNum, const char *inputLayout,
                    int64_t innerPreciseOptional, int64_t sparseModeOptional, int64_t pseTypeOptional,
                    aclOpExecutor *executor)
{
    L0_DFX(FlashAttentionScore, query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
           attenMaskOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional,
           kvStartIdxOptional, scaleValueOptional, keepProbOptional, preTockensOptional, nextTockensOptional,
           headNum, inputLayout, innerPreciseOptional, sparseModeOptional, pseTypeOptional);

    if (realShiftOptional == nullptr) {
        realShiftOptional = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (dropMaskOptional == nullptr) {
        dropMaskOptional = executor->AllocTensor(DataType::DT_UINT8, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (paddingMaskOptional == nullptr) {
        paddingMaskOptional = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (attenMaskOptional == nullptr) {
        attenMaskOptional = executor->AllocTensor(DataType::DT_BOOL, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *prefixOptionalTensor = nullptr;
    if (prefixOptional) {
        prefixOptionalTensor = executor->ConvertToTensor(prefixOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(prefixOptionalTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(prefixOptionalTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(prefixOptionalTensor)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        prefixOptionalTensor = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *actualSeqQLen = nullptr;
    if (actualSeqQLenOptional) {
        actualSeqQLen = executor->ConvertToTensor(actualSeqQLenOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqQLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqQLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqQLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualSeqQLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *actualSeqKvLen = nullptr;
    if (actualSeqKvLenOptional) {
        actualSeqKvLen = executor->ConvertToTensor(actualSeqKvLenOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqKvLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqKvLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqKvLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualSeqKvLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *qStartIdxOptionalTensor = nullptr;
    if (qStartIdxOptional) {
        qStartIdxOptionalTensor = executor->ConvertToTensor(qStartIdxOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(qStartIdxOptionalTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(qStartIdxOptionalTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(qStartIdxOptionalTensor)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        qStartIdxOptionalTensor = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *kvStartIdxOptionalTensor = nullptr;
    if (kvStartIdxOptional) {
        kvStartIdxOptionalTensor = executor->ConvertToTensor(kvStartIdxOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(kvStartIdxOptionalTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(kvStartIdxOptionalTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(kvStartIdxOptionalTensor)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        kvStartIdxOptionalTensor = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto softmaxMaxOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxSumOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxOutOut = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto attentionOutOut = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(FlashAttentionScore,
                           OP_INPUT(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                                    attenMaskOptional, prefixOptionalTensor, actualSeqQLen, actualSeqKvLen,
                                    qStartIdxOptionalTensor, kvStartIdxOptionalTensor),
                           OP_OUTPUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut),
                           OP_ATTR(static_cast<float>(scaleValueOptional), static_cast<float>(keepProbOptional),
                                   preTockensOptional, nextTockensOptional, headNum, inputLayout, innerPreciseOptional,
                                   sparseModeOptional, pseTypeOptional));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FlashAttentionScore InferShape failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(FlashAttentionScore,
                                OP_INPUT(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                                         attenMaskOptional, prefixOptionalTensor, actualSeqQLen, actualSeqKvLen,
                                         qStartIdxOptionalTensor, kvStartIdxOptionalTensor),
                                OP_OUTPUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut),
                                OP_ATTR(static_cast<float>(scaleValueOptional), static_cast<float>(keepProbOptional),
                                        preTockensOptional, nextTockensOptional, headNum, inputLayout,
                                        innerPreciseOptional, sparseModeOptional, pseTypeOptional));
    return {softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut};
}

} // namespace l0op
