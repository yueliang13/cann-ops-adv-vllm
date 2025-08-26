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
 * \file arope_case.h
 * \brief ApplyRotaryPosEmb 测试用例基类.
 */

#pragma once
#include <cstdint>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>

namespace ops::adv::tests::arope {
class ARopeCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;

public:
    class Param {
    public:
        int64_t b = 1;
        int64_t s = 1;
        int64_t qn = 1;
        int64_t kn = 1;
        int64_t d = 1;
        ge::DataType dataType = ge::DataType::DT_FLOAT16;

        Param();
        Param(int64_t pb, int64_t ps, int64_t pqn, int64_t pkn, int64_t pd, 
              ge::DataType pDataType);
    };

    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
    };

    Tensor q, k, cos, sin, q_out, k_out;
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    gert::OpImplRegisterV2::TilingKernelFunc aropeTilingFunc = nullptr;
    ARopeCase();
    ARopeCase(const char *name, bool enable, const char *dbgInfo, OpInfo rope, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOptiling(DoTilingParam &tilingParam);
};

} // namespace ops::adv::tests::arope
