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
 * \file incre_flash_attention_tiling.cc
 * \brief
 */

 #include "compute_cent_tiling.cpp"
 #include "register/op_def_registry.h"
 
 using namespace ge;
 using namespace AscendC;
 namespace optiling {
 ge::graphStatus TilingPrepareForComputeCent(gert::TilingParseContext *context)
 {
     (void)context;
     return ge::GRAPH_SUCCESS;
 }
 IMPL_OP_OPTILING(ComputeCent)
     .Tiling(TilingComputeCent)
     ;
 } // namespace optiling
 