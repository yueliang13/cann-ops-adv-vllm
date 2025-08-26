/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sinkhorn.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Sinkhorn);

const aclTensor *Sinkhorn(const aclTensor *cost, const aclScalar *tol, aclTensor *p, aclOpExecutor *executor)
{
    L0_DFX(Sinkhorn, cost, p);

    float fTol = 0.0001f;
    if (tol != nullptr) {
        fTol = tol->ToFloat();
    }

    auto out = executor->AllocTensor(cost->GetStorageShape(), cost->GetViewShape(), cost->GetDataType(),
                                     cost->GetStorageFormat(), cost->GetOriginalFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Sinkhorn, OP_INPUT(cost), OP_OUTPUT(p), OP_ATTR(fTol));

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Sinkhorn launch kernel failed!");
        return nullptr;
    }
    return out;
}

} // namespace l0op
