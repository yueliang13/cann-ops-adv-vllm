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
 * \file fa_case.h
 * \brief FlashAttentionScore / FlashAttentionScoreGrad 测试用例.
 */

#pragma once

#include <vector>
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "fa_param.h"
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>

namespace ops::adv::tests::fa {

/**
 * 算子 FlashAttentionScore / FlashAttentionScoreGrad 参数
 */
class FaCase : public ops::adv::tests::utils::Case {
public:
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using FaParam = ops::adv::tests::fa::FaParam;

    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor *prefixTensor = nullptr;
        gert::Tensor *actSeqQLenTensor = nullptr;
        gert::Tensor *actSeqKVLenTensor = nullptr;
        gert::Tensor *qStartIdxTensor = nullptr;
        gert::Tensor *kvStartIdxTensor = nullptr;
    };

    /**
     * 执行 Tiling 前, 修改 TilingContext 回调函数.
     *
     * \attention: 一般用于异常用例.
     */
    typedef void (*PreTilingRunCbf)(DoTilingParam &tilingParam);

public:
    /* 算子控制信息 */
    OpInfo mForward;
    OpInfo mReverse;
    Context mForwardCtx;
    Context mReverseCtx;

    /* 输入/输出 参数 */
    FaParam mParam;

    gert::OpImplRegisterV2::TilingKernelFunc mFasOriginTilingFunc = nullptr;
    gert::OpImplRegisterV2::TilingKernelFunc mFagOriginTilingFunc = nullptr;
    PreTilingRunCbf mPreTilingRunCbf = nullptr;

public:
    FaCase();
    FaCase(const char *name, bool enable, const char *dbgInfo, OpInfo forward, OpInfo reverse, FaParam param,
           int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);

    bool Run() override;
    bool DoOpTiling(DoTilingParam &tilingParam);

    static void PreTilingRunCbf_SetPlatformInfoNull(FaCase::DoTilingParam &tilingParam);

protected:
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitOriginTilingFunc();
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::fa
