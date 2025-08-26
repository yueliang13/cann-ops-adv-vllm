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
 * \file scaled_masked_softmax_v2_case.h
 * \brief
 */

#pragma once
#include <vector>
#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"

namespace ops::adv::tests::ScaledMaskedSoftmaxV2 {
class ScaledMaskedSoftmaxV2Case : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;
    using TensorList = ops::adv::tests::utils::TensorList;
public:
    enum class MaskType
    {
        SameShape = 0x0000,         // x和mask shape B N 相同 
        BroadCastB = 0x0001,        // x和mask shape 需要broadcast B
        BroadCastN = 0x0010,        // x和mask shape 需要broadcast N
        ReshapeBN = 0x0100          // mask shape 为2维 reshape成[1,1,s1,s2]
    };
    class Param {
    public:
        int64_t b = 0;
        int64_t n = 0;
        int64_t s1 = 0;
        int64_t s2 = 0;
        float scale = 1.0;
        bool genMask = false;
        MaskType maskType = MaskType::SameShape;
        ge::DataType xDtype = ge::DataType::DT_FLOAT;

        Param();
        Param(int64_t pb, int64_t pn, int64_t ps1, int64_t ps2, float pscale, bool pgenMask, MaskType pmaskType, ge::DataType pxDtypeIn);
    };
    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
    };
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    Tensor x, mask, y;
    gert::OpImplRegisterV2::TilingKernelFunc scaledMaskedSoftmaxV2TilingFunc = nullptr;
    ScaledMaskedSoftmaxV2Case();
    ScaledMaskedSoftmaxV2Case(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOpTiling(DoTilingParam &tilingParam);
};

} // namespace ops::adv::tests::ScaledMaskedSoftmaxV2