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
 * \file mirg_case.h
 * \brief MoeFinalizeRoutingV2Grad 测试用例.
 */

 #pragma once
 #include "graph/types.h"

 #include <vector>
 #include <cstdint>

 #include "tests/utils/case.h"
 #include "tests/utils/op_info.h"
 #include "tests/utils/context.h"
 #include "tests/utils/tensor.h"
 #include "tests/utils/tensor_list.h"
 #include <exe_graph/runtime/tiling_context.h>
 #include <register/op_impl_registry.h>
 
 namespace ops::adv::tests::mirg {
 class MirgCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;
    using TensorList = ops::adv::tests::utils::TensorList;
 public:
 
     class Param {
     public:
        int64_t num_rows, k, hidden_size, e, c, dropPadMode, activeNum;
        ge::DataType dx;
        Param();
        Param(int64_t num_rows_, int64_t k_, int64_t hidden_size_, int64_t e_, int64_t c_, int64_t drop_pad_mode_, int64_t active_num_, ge::DataType dx_);
     };
     class DoTilingParam {
     public:
         gert::TilingContext *ctx = nullptr;
         ge::graphStatus ret = ge::GRAPH_SUCCESS;
     };
 
     int64_t h;
     Tensor gradExpandedX;
     Tensor expandedRowIdx;
     Tensor out;
     OpInfo mOpInfo;
     Context mCtx;
     Param mParam;
     gert::OpImplRegisterV2::TilingKernelFunc mirgTilingFunc = nullptr;
     MirgCase();
     MirgCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
     bool Run() override;
     bool InitParam() override;
     bool InitOpInfo() override;
     bool InitCurrentCasePtr() override;
     bool DoOpTiling(DoTilingParam& tilingParam);
 };
 
 } // namespace ops::adv::tests::mirg