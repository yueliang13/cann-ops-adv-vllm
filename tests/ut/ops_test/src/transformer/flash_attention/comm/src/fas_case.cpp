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
 * \file fas_case.cpp
 * \brief
 */

#include "fas_case.h"
#include <utility>

using namespace ops::adv::tests::fas;

FasCase::FasCase() : FasCase("Undefined", true, "", OpInfo(), FaParam())
{
}

FasCase::FasCase(const char *name, bool enable, const char *dbgInfo, OpInfo forward, FaParam param)
    : FaCase(name, enable, dbgInfo, std::move(forward), OpInfo(), std::move(param), kTilingTemplatePriority_Invalid)
{
}

bool FasCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mForward.ProcessTiling(mName)) {
        return false;
    }
    if (!mForward.ProcessKernel(mName)) {
        return false;
    }
    return true;
}
