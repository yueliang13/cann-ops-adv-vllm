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
 * \file prompt_flash_attention_tiling.cc
 * \brief
 */

#include "prompt_flash_attention_tiling.h"
#include "register/op_def_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
constexpr uint32_t ACTUAL_SEQ_Q_INDEX_PFA = 5;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX_PFA = 6;
static ge::graphStatus TilingPrepareForPromptFlashAttention(gert::TilingParseContext* context) {
    auto platformInfoPtr = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfoPtr == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null!"),
                    return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<PromptFlashAttentionCompileInfo>();
    OPS_ERR_IF(compileInfoPtr == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null!"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);

    compileInfoPtr->socShortName = ascendcPlatform.GetSocVersion();
    if (compileInfoPtr->socShortName == platform_ascendc::SocVersion::ASCEND310P) {
        // sys workspace size default value
        compileInfoPtr->defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    } else {
        compileInfoPtr->defaultSysWorkspaceSize = 0;
    }

    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(PromptFlashAttention)
    .TilingInputsDataDependency({ACTUAL_SEQ_Q_INDEX_PFA, ACTUAL_SEQ_KV_INDEX_PFA})
    .Tiling(TilingPromptFlashAttention)
    .TilingParse<PromptFlashAttentionCompileInfo>(TilingPrepareForPromptFlashAttention); // Register entrance functions to the framework
}  // namespace optiling
