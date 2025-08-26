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
 * \file moe_token_unpermute_case.h
 * \brief MoeTokenUnpermute 测试用例.
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

namespace ops::adv::tests::MoeTokenUnpermute {
class MoeTokenUnpermuteCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;
    using TensorList = ops::adv::tests::utils::TensorList;

public:
    class Param {
    public:
        int64_t n = 0;
        int64_t h = 0;
        int64_t k = 0;
        bool paddedMode = false;
        std::vector<int64_t> restoreShape = {1, 1};
        ge::DataType permutedTokensDtype = ge::DataType::DT_FLOAT;
        ge::DataType sortedIndicesDtype = ge::DataType::DT_INT32;
        ge::DataType probsDtype = ge::DataType::DT_FLOAT;

        Param();
        Param(int64_t pN, int64_t pH, int64_t pK, bool pPaddedMode, ge::DataType permutedTokensDtypeIn,
              ge::DataType sortedIndicesDtypeIn, ge::DataType probsDtypeIn);
    };
    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
    };
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    // input
    Tensor permutedTokens;
    Tensor sortedIndices;
    Tensor probs;
    // output
    Tensor unpermutedTokens;
    gert::OpImplRegisterV2::TilingKernelFunc moeTokenUnpermuteTilingFunc = nullptr;
    MoeTokenUnpermuteCase();
    MoeTokenUnpermuteCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::MoeTokenUnpermute