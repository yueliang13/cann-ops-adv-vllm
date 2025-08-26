/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swin_attention_score_quant_case.h
 * \brief SwinAttentionScoreQuant 测试用例.
 */

#pragma once
#include <register/op_impl_registry.h>
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"

namespace ops::adv::tests::SwinAttentionScoreQuant {
class SwinAttentionScoreQuantCase : public ops::adv::tests::utils::Case {
using OpInfo = ops::adv::tests::utils::OpInfo;
using Context = ops::adv::tests::utils::Context;
using Tensor = ops::adv::tests::utils::Tensor;
using TensorList = ops::adv::tests::utils::TensorList;

public:
class Param {
public:
    int64_t b = 0;
    int64_t n = 0;
    int64_t s = 0;
    int64_t h = 0;

    bool qTrans = true;
    bool kTrans = true;
    bool vTrans = true;
    int softmaxAxes = -1;


    Param();
    Param(int64_t pB, int64_t pN, int64_t pS, int64_t pH, bool qTranspose, bool kTranspose, bool vTranspose, int pSoftmaxAxes);
};
class DoTilingParam {
public:
    gert::TilingContext *ctx = nullptr;
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
};
OpInfo mOpInfo;
Context mCtx;
Param mParam;
Tensor query, key, value, scale_quant, scale_dequant1, scale_dequant2, bias_quant, bias_dequant1,bias_dequant2, padding_mask1, padding_mask2, attention_score;
gert::OpImplRegisterV2::TilingKernelFunc swinAttentionScoreQuantTilingFunc = nullptr;
SwinAttentionScoreQuantCase();
SwinAttentionScoreQuantCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
bool Run() override;
bool InitParam() override;
bool InitOpInfo() override;
bool InitCurrentCasePtr() override;
bool DoOpTiling(DoTilingParam &tilingParam);
};

} // namespace ops::adv::tests::MoeGatingTopKSoftmaxV2