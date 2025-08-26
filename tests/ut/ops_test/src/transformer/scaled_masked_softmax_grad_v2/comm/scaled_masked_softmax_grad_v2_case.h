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
 * \file scaled_masked_softmax_grad_v2_case.h
 * \brief ScaledMaskedSoftmaxGradV2 测试用例基类.
 */

#pragma once
#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"

namespace ops::adv::tests::ScaledMaskedSoftmaxGradV2 {
class ScaledMaskedSoftmaxGradV2Case : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;

public:
    class Param {
    public:
        int64_t b = 1;
        int64_t n = 1;
        int64_t s = 1;
        int64_t d = 1;
        int64_t maskB = 1;
        int64_t maskN = 1;
        float scaleValue = 1.0;
        bool fixedTriuMask = false;
        ge::DataType dataType = ge::DataType::DT_FLOAT16;
        ge::DataType maskDataType = ge::DataType::DT_BOOL;

        Param();
        Param(int64_t pb, int64_t pn, int64_t ps, int64_t pd, int64_t pmaskB, int64_t pmaskN, float pscaleValue,
            bool pfixedTriuMask, ge::DataType pDataType, ge::DataType pmaskDataType);
    };

    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
    };

    Tensor yGrad;
    Tensor y;
    Tensor mask;
    Tensor xGrad;
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    gert::OpImplRegisterV2::TilingKernelFunc scaledMaskedSoftmaxGradV2TilingFunc = nullptr;
    ScaledMaskedSoftmaxGradV2Case();
    ScaledMaskedSoftmaxGradV2Case(const char *name, bool enable, const char *dbgInfo, OpInfo softmax, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOptiling(DoTilingParam &tilingParam);
};

} // namespace ops::adv::tests::ScaledMaskedSoftmaxGradV2
