/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

/*!
 * \file prompt_flash_attention_tiling_compile_info.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_TILING_STRUCT_COMPILE_INFO_H
#define PROMPT_FLASH_ATTENTION_TILING_STRUCT_COMPILE_INFO_H
#include <cstdint>
#include <vector>
#include <queue>
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/data_copy_transpose_tiling_def.h"
#include "tiling/data_copy_transpose_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "register/op_def_registry.h"

namespace optiling { 

struct PromptFlashAttentionCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    size_t defaultSysWorkspaceSize;
    platform_ascendc::SocVersion socShortName;
};

}

#endif  // PROMPT_FLASH_ATTENTION_TILING_STRUCT_COMPILE_INFO_H