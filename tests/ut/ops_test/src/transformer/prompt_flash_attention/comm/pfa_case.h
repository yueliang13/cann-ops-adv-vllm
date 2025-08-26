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
 * \file pfa_case.h
 * \brief PromptFlashAttention 测试用例.
 */

#pragma once

#include <vector>
#include <cstdint>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>

namespace ops::adv::tests::pfa {
class PfaCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;

public:
    enum class PseShiftShapeType {
        NONE,
        B_1_N_S,
        _1_N_1_S
    };
    enum class AttenMaskShapeType {
        NONE,
        B_N_1_S,
        B_1_S,
    };

    enum class QuantShapeType {
        NONE,
        PER_1,
        POST_1,
        ALL_1,
    };

    class Param {
    public:
        int64_t b = 0;
        int64_t n = 0;
        int64_t s = 0;
        int64_t d = 0;
        std::string layout = "BSH";
        int64_t numHeads = 1;
        int64_t kvNumHeads = 0;
        float scaleValue = 1.0f;
        int64_t blockSize = 0;
        int64_t innerPrecise = 1;
        int64_t sparseMode = 0;
        int64_t preTokens = 524288;
        int64_t nextTokens = 0;
        ge::DataType qDataType = ge::DataType::DT_FLOAT16;
        ge::DataType kvDataType = ge::DataType::DT_FLOAT16;
        ge::DataType outDataType = ge::DataType::DT_FLOAT16;
        PseShiftShapeType pseShiftType = PseShiftShapeType::NONE;
        AttenMaskShapeType attenMaskType = AttenMaskShapeType::NONE;
        QuantShapeType quantType = QuantShapeType::NONE;
        std::vector<int64_t> actualSeqLength = {};
        std::vector<int64_t> actualSeqLengthKV = {};
        Param();
        Param(int64_t pB, int64_t pN, int64_t pS, int64_t pD, std::string pLayout, int64_t pNumHeads,
              int64_t pKvNumHeads, float pScaleValue, int64_t pBlockSize, int64_t pInnerPrecise, int64_t pSparseMode,
              int64_t pPreTokens, int64_t pNextTokens);
    };

    class DoTilingParam {
        public:
        gert::TilingContext* ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor* actualSeqLengthsTensor = nullptr;
        gert::Tensor* actualSeqLengthsKVTensor = nullptr;
    };

    int64_t h;
    Tensor query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, deqScale1, quantScale1,
        deqScale2, quantScale2, quantOffset2, attentionOut;
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    gert::OpImplRegisterV2::TilingKernelFunc pfaTilingFunc = nullptr;
    PfaCase();
    PfaCase(const char *name, bool enable, const char *dbgInfo, OpInfo mOpInfo, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOpTiling(DoTilingParam& tilingParam);
};

} // namespace ops::adv::tests::pfa
