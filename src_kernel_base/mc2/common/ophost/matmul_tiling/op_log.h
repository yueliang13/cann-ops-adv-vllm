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
 * \file op_log.h
 * \brief
 */
#ifndef OPS_COMMON_INC_OP_LOG_H_
#define OPS_COMMON_INC_OP_LOG_H_
#define unlikely(x) __builtin_expect((x), 0)
#define OPS_LOG_I(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
#define OPS_LOG_E(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
#define OPS_LOG_D(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
#define OPS_LOG_E_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do                                                                                                           \
  {                                                                                                            \
    if (unlikely(condition))                                                                                   \
    {                                                                                                          \
      OPS_LOG_E(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)            \
    if ((ptr) == nullptr)                                    \
    {                                                        \
        std::printf("nullptr error!");                       \
        return ge::GRAPH_SUCCESS;                            \
    }

#define OPS_LOG_D_FULL(opname, ...) OP_LOG_FULL(DLOG_DEBUG, get_op_info(opname), __VA_ARGS__)


#define OPS_LOG_I_IF_RETURN(condition, return_value, op_name, fmt, ...)                                          \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OPS_LOG_I(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)

#define OPS_LOG_E_WITHOUT_REPORT(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)

#endif  // OPS_COMMON_INC_OP_LOG_H_
