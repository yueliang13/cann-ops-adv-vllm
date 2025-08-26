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
 * \file moe_token_permute.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class MoeTokenPermute : public OpDef {
 public:
  explicit MoeTokenPermute(const char* name) : OpDef(name) {
    this->Input("tokens")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16,  ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("permute_tokens")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("sorted_indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("num_out_tokens").AttrType(OPTIONAL).Int(0);
    this->Attr("padded_mode").AttrType(OPTIONAL).Bool(false);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(MoeTokenPermute);
}  // namespace ops

