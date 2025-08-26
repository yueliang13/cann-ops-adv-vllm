/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file matmul_v3_tiling_strategy.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_STRATEGY_H__
#define __OP_HOST_MATMUL_V3_STRATEGY_H__

#include <map>
#include <vector>
#include <cstdint>

#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace matmul_v3_advanced {
namespace strategy {
constexpr int32_t STREAM_K = 0;
constexpr int32_t FULL_LOAD_BASE = 1;
constexpr int32_t BASE = 999;
}
}
}

#endif // __OP_HOST_MATMUL_V3_STRATEGY_H__
