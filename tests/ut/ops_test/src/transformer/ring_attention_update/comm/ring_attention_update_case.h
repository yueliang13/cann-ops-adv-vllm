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
 * \file ring_attention_update_case.h
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

namespace ops::adv::tests::RingAttentionUpdate {
class RingAttentionUpdateCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;
    using TensorList = ops::adv::tests::utils::TensorList;

public:
    class Param {
    public:
        int64_t s = 0;
        int64_t b = 0;
        int64_t h = 0;
        int64_t n = 0;
        int64_t t = 0;
        int64_t d = 0;
        std::string layout = "BSH";
        ge::DataType attnDataType = ge::DataType::DT_FLOAT16;
        ge::DataType softmaxDataType = ge::DataType::DT_FLOAT;
        ge::DataType seqLenDataType = ge::DataType::DT_INT64;
    
        Param();
        Param(int64_t pS, int64_t pB, int64_t pH, int64_t pN, int64_t pT, int64_t pD, std::string pLayout,
              ge::DataType attnDataTypeIn, ge::DataType softmaxDataTypeIn, ge::DataType seqLenDataTypeIn);
    };
    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
    };
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    Tensor prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen;
    Tensor attn_out, softmax_max, softmax_sum;

    gert::OpImplRegisterV2::TilingKernelFunc ringAttentionUpdateTilingFunc = nullptr;
    RingAttentionUpdateCase();
    RingAttentionUpdateCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOpTiling(DoTilingParam &tilingParam);
};

} // namespace ops::adv::tests::RingAttentionUpdate

