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
 * \file ffn_param.cpp
 * \brief FFN 参数信息.
 */

#include "ffn_param.h"

using namespace ops::adv::tests::ffn;

Param::Param(std::vector<Tensor> inputs, std::vector<int64_t> expertTokensData, std::string activation,
             int32_t innerPrecise, int32_t outputDtype, bool tokensIndexFlag)
    : mExpertTokensData(std::move(expertTokensData)), mActivation(activation), mInnerPrecise(innerPrecise),
      mOutputDtype(outputDtype), mTokensIndexFlag(tokensIndexFlag)
{
    for (auto &tensor : inputs) {
        mTensors[tensor.Name()] = tensor;
    }
}