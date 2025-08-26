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

#ifndef ROPE_INFER_CASE_H
#define ROPE_INFER_CASE_H

#pragma once
#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"

namespace ops::adv::tests::rope {
class RopeInferCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;

public:
    class Param {
    public:
        int64_t batch = 1;
        int64_t hiddensizeQ = 1;
        int64_t hiddensizeK = 1;
        int64_t headDim = 1;
        int64_t rotaryCoeff = 2;
        int64_t cosFormat = 0;
        bool largeDim = false; // cos 和sin 是否输入 3维
        int64_t rotaryCoefficiency = 2;
        std::string layout = "ND";
        ge::DataType dataType = ge::DataType::DT_FLOAT16;
        ge::DataType dataTypeCos = ge::DataType::DT_FLOAT16;
        
        Param();
        Param(int64_t pBatch, int64_t pHiddensizeQ, int64_t pHiddensizeK, int64_t pHeadDim, bool pLargeDim, int64_t pRotaryCoefficiency,
            std::string pLayout, ge::DataType pDataType, ge::DataType pDataTypeCos);
    };

    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
    };

    int64_t h;
    Tensor q, k, cos, sin, seqlen, q_out, k_out;
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    gert::OpImplRegisterV2::TilingKernelFunc ropeTilingFunc = nullptr;
    RopeInferCase();
    RopeInferCase(const char *name, bool enable, const char *dbgInfo, OpInfo rope, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOptiling(DoTilingParam &tilingParam);
};

} // namespace ops::adv::tests::rope

#endif