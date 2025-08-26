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
 * \file moe_gating_top_k_softmax_v2_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class MoeGatingTopKSoftmaxV2 : public OpDef {
 public:
  explicit MoeGatingTopKSoftmaxV2(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("finished")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("expert_idx")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("softmax_result")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("k").AttrType(REQUIRED).Int();
    this->Attr("renorm").AttrType(OPTIONAL).Int(0);
    this->Attr("output_softmax_result_flag").AttrType(OPTIONAL).Bool(false);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(MoeGatingTopKSoftmaxV2);
}  // namespace ops