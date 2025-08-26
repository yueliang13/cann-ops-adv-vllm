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
 * \file fag_case.cpp
 * \brief FlashAttentionScoreGrad 测试用例.
 */

#include "fag_case.h"
#include <utility>

using namespace ops::adv::tests::fag;

FagCase::FagCase() : FagCase("Undefined", true, "", OpInfo(), FaParam(), kTilingTemplatePriority_Invalid)
{
}

FagCase::FagCase(const char *name, bool enable, const char *dbgInfo, OpInfo reverse, FaParam param,
                 int32_t tilingTemplatePriority)
    : FaCase(name, enable, dbgInfo, OpInfo(), std::move(reverse), std::move(param), tilingTemplatePriority)
{
}

bool FagCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mReverse.ProcessTiling(mName)) {
        return false;
    }
    if (!mReverse.ProcessKernel(mName)) {
        return false;
    }
    return true;
}
