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
 * \file sparse_paged_attention_tiling.cc
 * \brief
 */

#include "sparse_paged_fusion_attention_tiling.h"
#include "register/op_def_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
ge::graphStatus TilingPrepareForSparsePagedFusionAttention(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(SparsePagedFusionAttention)
    .Tiling(TilingSparsePagedFusionAttention)
    .TilingParse<SparsePagedFusionAttentionCompileInfo>(TilingPrepareForSparsePagedFusionAttention)
    .TilingInputsDataDependency({5}, {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU});
} // namespace optiling
