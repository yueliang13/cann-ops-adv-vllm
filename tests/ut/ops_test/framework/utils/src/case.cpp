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
 * \file case.cpp
 * \brief 测试用例.
 */

#include "tests/utils/case.h"
#include <utility>
#include "tests/utils/log.h"

using namespace ops::adv::tests::utils;

void *Case::mCurrentCasePtr = nullptr;

Case::Case() : Case("Undefined", true, "", kTilingTemplatePriority_Invalid)
{
}

Case::Case(const char *name, bool enable, const char *dbgInfo, int32_t tilingTemplatePriority)
    : mName(name), mEnable(enable), mDbgInfo(dbgInfo), mTilingTemplatePriority(tilingTemplatePriority)
{
}

bool Case::Init()
{
    bool rst = this->InitCurrentCasePtr();
    rst = rst && this->InitParam();
    rst = rst && this->InitOpInfo();
    LOG_IF(!rst, LOG_ERR("Case(%s, %s) Init failed", mName.c_str(), mDbgInfo.c_str()));
    return rst;
}

void *Case::GetCurrentCase()
{
    return Case::mCurrentCasePtr;
}

const char *Case::GetRootPath()
{
    return mRootPath.c_str();
}

bool Case::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}
