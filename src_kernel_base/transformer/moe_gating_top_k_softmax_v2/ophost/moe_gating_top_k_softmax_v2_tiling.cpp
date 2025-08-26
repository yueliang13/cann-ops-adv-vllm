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
 * \file moe_gating_top_k_softmax_v2_tiling.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "tiling/tiling_templates_registry.h"
#include "moe_gating_top_k_softmax_v2_tiling.h"

using namespace AscendC;
namespace optiling {

ASCENDC_EXTERN_C ge::graphStatus TilingMoeGatingTopKSoftmaxV2(gert::TilingContext *context)
{
    // 初始化算子Tiling类
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepare4MoeGatingTopKSoftmaxV2(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeGatingTopKSoftmaxV2)
    .Tiling(TilingMoeGatingTopKSoftmaxV2)
    .TilingParse<MoeGatingTopKSoftmaxV2CompileInfo>(TilingPrepare4MoeGatingTopKSoftmaxV2);
} // namespace optiling