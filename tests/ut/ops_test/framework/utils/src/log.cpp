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
 * \file log.cpp
 * \brief
 */

#include "tests/utils/log.h"

namespace {
uint32_t g_LogErrCnt = 0;
}

void ops::adv::tests::utils::AddLogErrCnt()
{
    g_LogErrCnt++;
}

bool ops::adv::tests::utils::ChkLogErrCnt()
{
    auto rst = g_LogErrCnt == 0;
    if (!rst) {
        fprintf(stdout, "%s:%d [ERROR] There have %u error log in current case.\n", __FILE__, __LINE__, g_LogErrCnt);
        g_LogErrCnt = 0;
    }
    return rst;
}
