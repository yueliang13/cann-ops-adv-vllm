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
 * \file swin_transformer_ln_qkv_quant_case.h
 * \brief SwinTransformerLnQkvQuant 测试用例.
 */

#pragma once
#include <register/op_impl_registry.h>
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"

namespace ops::adv::tests::SwinTransformerLnQkvQuant {
class SwinTransformerLnQkvQuantCase : public ops::adv::tests::utils::Case {
using OpInfo = ops::adv::tests::utils::OpInfo;
using Context = ops::adv::tests::utils::Context;
using Tensor = ops::adv::tests::utils::Tensor;
using TensorList = ops::adv::tests::utils::TensorList;

public:
class Param {
public:
int64_t b = 0;
int64_t s = 0;
int64_t h = 0;
int64_t ori_height = 0;
int64_t ori_weight = 0;
int64_t headNum = 0;
int64_t hWin = 0;
int64_t wWin = 0;
int64_t sizePerhead = 0;
bool bTrans = true;
float epslion = 0.0001f;
ge::DataType indexDtype = ge::DataType::DT_FLOAT16;
ge::DataType xDtype = ge::DataType::DT_FLOAT16;
ge::DataType finishDtype = ge::DataType::DT_FLOAT16;

    Param();
    Param(int64_t B, int64_t S, int64_t H, int64_t ori_height, int64_t ori_weight, 
 int64_t headNum,  int64_t hWin, int64_t wWin, int64_t sizePerhead, bool bTrans, float epslion);
};
class DoTilingParam {
public:
    gert::TilingContext *ctx = nullptr;
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
};
OpInfo mOpInfo;
Context mCtx;
Param mParam;
Tensor x, gamma, beta, weight, bias, quant_scale, quant_offset, dequant_scale,query_output, key_output, value_output;
gert::OpImplRegisterV2::TilingKernelFunc swinTransformerLnQkvQuantTilingFunc = nullptr;
SwinTransformerLnQkvQuantCase();
SwinTransformerLnQkvQuantCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
bool Run() override;
bool InitParam() override;
bool InitOpInfo() override;
bool InitCurrentCasePtr() override;
bool DoOpTiling(DoTilingParam &tilingParam);
};

} // namespace ops::adv::tests::MoeGatingTopKSoftmaxV2