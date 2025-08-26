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
 * \file fia_case.h
 * \brief FusedInferAttentionScore 测试用例.
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

namespace ops::adv::tests::fia {
class FiaCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;

public:
    class Param {
    public:
        int64_t b = 0;
        int64_t n = 0;
        int64_t s = 0;
        int64_t d = 0;
        int64_t numHeads = 1;
        float scaleValue = 1.0f;
        int64_t pre_tokens = 2147483647;
        int64_t next_tokens = 0;
        std::string layout = "BSH";
        int64_t kvNumHeads = 0;
        int64_t sparse_mode = 0;
        int64_t innerPrecise = 1;
        int64_t blockSize = 0;
        int64_t antiquant_mode = 0;
        int64_t softmax_lse_flag = 0;
        int64_t key_antiquant_mode = 0;
        int64_t value_antiquant_mode = 0;
        ge::DataType qDataType = ge::DataType::DT_FLOAT16;
        ge::DataType kvDataType = ge::DataType::DT_FLOAT16;
        ge::DataType outDataType = ge::DataType::DT_FLOAT16;

        Param();
    };


    int64_t h;
    Tensor query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, deqScale1, quantScale1,
        deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, blocktable, queryPaddinSize,
        kvPaddingSize, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
        keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse;
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    FiaCase();
    FiaCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::fia
