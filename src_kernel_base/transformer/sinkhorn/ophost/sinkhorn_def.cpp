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
 * \file sinkhorn_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"


namespace ops {
class Sinkhorn : public OpDef {
public:
    explicit Sinkhorn(const char* name) : OpDef(name) {
        this->Input("cost")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("p")
            .ParamType(REQUIRED)
            .Follow("cost");
        this->Attr("tol")
            .AttrType(OPTIONAL)
            .Float(0.0001f);

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Sinkhorn);
}  // namespace ops
