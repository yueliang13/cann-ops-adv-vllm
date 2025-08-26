/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_v2_case.h
 * \brief MoeInitRoutingV2 测试用例.
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

namespace ops::adv::tests::MoeInitRoutingV2 {
class MoeInitRoutingV2Case : public ops::adv::tests::utils::Case {
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
        int64_t activeNum = 0;
        int64_t c = 0;
        int64_t e = 0;
        int64_t dropPadMode = 0;
        int64_t countFlag = 0;
        bool tokenFlag = false;
        ge::DataType optionalOutputDt = ge::DataType::DT_INT32;
        Param();
        Param(int64_t n, int64_t h, int64_t k, int64_t activeNum, int64_t c, int64_t e, int64_t dropPadMode,
              int64_t countFlag, bool tokenFlag, ge::DataType optionalOutputDt);
    };
    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
    };
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    Tensor x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity;
    gert::OpImplRegisterV2::TilingKernelFunc moeInitRoutingV2TilingFunc = nullptr;
    MoeInitRoutingV2Case();
    MoeInitRoutingV2Case(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOpTiling(DoTilingParam &tilingParam);
};

} // namespace ops::adv::tests::MoeInitRoutingV2
