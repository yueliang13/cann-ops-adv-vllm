/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mat_mul_v3_compile_info.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_COMPILE_INFO_H__
#define __OP_HOST_MATMUL_V3_COMPILE_INFO_H__
#include <cstdint>
#include <string>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

struct MatmulV3CompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0CSize{0UL};
    uint64_t l0ASize{0UL};
    uint64_t l0BSize{0UL};
    uint64_t btSize{0UL};
    float cubeFreq{0};
    platform_ascendc::SocVersion socVersion;
    std::string socVersionStr = "";
    bool supportL0c2out = false;
    bool supportL12BtBf16 = false;
};
}
#endif // __OP_HOST_MATMUL_V3_COMPILE_INFO_H__