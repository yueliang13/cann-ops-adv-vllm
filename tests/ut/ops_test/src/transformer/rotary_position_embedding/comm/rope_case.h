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
 * \file rope_case.h
 * \brief RotaryPositionEmbedding 测试用例基类.
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

namespace ops::adv::tests::rope {
class RopeCase : public ops::adv::tests::utils::Case {
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
        int64_t triB = 1;
        int64_t triN = 1;
        int64_t mode = 0;
        std::string layout = "BNSD";
        ge::DataType dataType = ge::DataType::DT_FLOAT16;

        Param();
        Param(int64_t pb, int64_t pn, int64_t ps, int64_t pd, int64_t pTriB, int64_t pTriN, 
              int64_t pMode, std::string pLayout, ge::DataType pDataType);
    };

    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
    };

    int64_t h;
    Tensor x, cos, sin, y;
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    gert::OpImplRegisterV2::TilingKernelFunc ropeTilingFunc = nullptr;
    RopeCase();
    RopeCase(const char *name, bool enable, const char *dbgInfo, OpInfo rope, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOptiling(DoTilingParam &tilingParam);
};

} // namespace ops::adv::tests::rope
