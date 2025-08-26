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
 * \file rotary_pos_emb_infer.cc
 * \brief
 */

#include "local_infer_context.h"
namespace ops {
namespace {
constexpr size_t INPUT_Q_INDEX = 0;
constexpr size_t INPUT_K_INDEX = 1;
constexpr size_t INPUT_COS_INDEX = 2;
constexpr size_t INPUT_SIN_INDEX = 3;
constexpr size_t OUTPUT_Q_INDEX = 0;
constexpr size_t OUTPUT_K_INDEX = 1;

uint32_t InferShapeForRotaryPosEmbInfer(LocalInferShapeParams &params)
{
    if (params.GetInputDimNum(INPUT_Q_INDEX) != params.GetInputDimNum(INPUT_K_INDEX) ||
        params.GetInputDimNum(INPUT_Q_INDEX) == 0) {
        return INFER_FAILED;
    }

    for (size_t dimIdx = 0; dimIdx < params.GetInputDimNum(INPUT_Q_INDEX) - 1; dimIdx++) {
        if (params.GetInputDim(INPUT_Q_INDEX, dimIdx) != params.GetInputDim(INPUT_K_INDEX, dimIdx)) {
            return INFER_FAILED;
        }
    }

    if (!params.InputDimsEqual(INPUT_COS_INDEX, INPUT_SIN_INDEX)) {
        return INFER_FAILED;
    }

    size_t dimNum = params.GetInputDimNum(INPUT_Q_INDEX);
    params.SetOutputDimNum(OUTPUT_Q_INDEX, dimNum);
    params.SetOutputDimNum(OUTPUT_K_INDEX, dimNum);
    for (size_t dimIdx = 0; dimIdx < dimNum; dimIdx++) {
        params.SetOutputDim(OUTPUT_Q_INDEX, dimIdx, params.GetInputDim(INPUT_Q_INDEX, dimIdx));
        params.SetOutputDim(OUTPUT_K_INDEX, dimIdx, params.GetInputDim(INPUT_K_INDEX, dimIdx));
    }

    return INFER_SUCCESS;
}

uint32_t InferDataTypeForRotaryPosEmbInfer(LocalInferDataTypeParams &params)
{
    params.SetOutputFormat(OUTPUT_Q_INDEX, params.GetInputFormat(INPUT_Q_INDEX));
    params.SetOutputDType(OUTPUT_K_INDEX, params.GetInputDType(INPUT_K_INDEX));
    return INFER_SUCCESS;
}
}  // namespace

REG_OP_INFERSHAPE(RotaryPosEmbInfer, InferShapeForRotaryPosEmbInfer, InferDataTypeForRotaryPosEmbInfer);
}  // namespace ops
