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
 * \file log.h
 * \brief
 */

#pragma once

#include <cstdio>
#include <type_traits>
#include <base/alog_pub.h>

namespace ops::adv::tests::utils {

void AddLogErrCnt();

bool ChkLogErrCnt();

#define LOG_ERR(fmt, args...)                                                                                          \
    do {                                                                                                               \
        fprintf(stdout, "%s:%d [ERROR] " fmt "\n", __FILE__, __LINE__, ##args);                                        \
        ops::adv::tests::utils::AddLogErrCnt();                                                                        \
    } while (0)

#define LOG_DBG(fmt, args...)                                                                                          \
    do {                                                                                                               \
        if (AlogCheckDebugLevel(OP, DLOG_DEBUG) == 1) {                                                                      \
            fprintf(stdout, "%s:%d [DEBUG] " fmt "\n", __FILE__, __LINE__, ##args);                                    \
        }                                                                                                              \
    } while (0)

#define LOG_IF(COND, LOG_FUNC)                                                                                         \
    static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, "condition should be bool");            \
    do {                                                                                                               \
        if (__builtin_expect((COND), 0)) {                                                                             \
            LOG_FUNC;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define LOG_IF_EXPR(COND, LOG_FUNC, EXPR)                                                                              \
    static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, "condition should be bool");            \
    do {                                                                                                               \
        if (__builtin_expect((COND), 0)) {                                                                             \
            LOG_FUNC;                                                                                                  \
            EXPR;                                                                                                      \
        }                                                                                                              \
    } while (0)

#define IF_EXPR(COND, EXPR)                                                                                            \
    static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, "condition should be bool");            \
    do {                                                                                                               \
        if (__builtin_expect((COND), 0)) {                                                                             \
            EXPR;                                                                                                      \
        }                                                                                                              \
    } while (0)

} // namespace ops::adv::tests::utils
