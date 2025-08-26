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
 * \file mc2_tiling_struct.h
 * \brief
*/

#ifndef __MC2_TILING_STRUCT_H__
#define __MC2_TILING_STRUCT_H__

#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint8_t COMM_ALG_DEFAULT = 0;
constexpr uint8_t COMM_ALG_FULL_MESH = 1;
constexpr uint8_t COMM_ALG_DOUBLE_RING = 2;
constexpr uint8_t COMM_ALG_SWITCH_WING = 3;
constexpr uint32_t DOUBLE_RING_FACTOR = 2;
constexpr int64_t KVALUE_MIN = 256;
constexpr int64_t KVALUE_MAX = 65535;
} // namespace optiling

#endif