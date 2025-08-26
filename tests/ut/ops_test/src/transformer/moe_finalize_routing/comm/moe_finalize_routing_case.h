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
 * \file moe_finalize_routing_case.h
 * \brief moe_finalize_routing 测试用例.
 */

#pragma once

#include <vector>
#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"

namespace ops::adv::tests::MoeFinalizeRoutingCase {

/**

算子 moe_finalize_routing 参数
*/
class MoeFinalizeRoutingCase : public ops::adv::tests::utils::Case {
  using OpInfo = ops::adv::tests::utils::OpInfo;
  using Context = ops::adv::tests::utils::Context;
  using Tensor = ops::adv::tests::utils::Tensor;
  using TensorList = ops::adv::tests::utils::TensorList;
public:
  class Param {
    public:
      int expert_num = 0;
      int token_len = 0;
      int top_k = 0;
      int num_rows = 0;
      std::string layout = "BSH";
      ge::DataType dType = ge::DT_FLOAT;
      Param();
      Param(int expert_num, int token_len, int top_k, int num_rows, std::string layout, ge::DataType dType);
  };

  class DoTilingParam {
  public:
    gert::TilingContext* ctx = nullptr;
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
  };

  OpInfo mOpInfo;
  Context mCtx;
  Param mParam;
  Tensor expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out;
  gert::OpImplRegisterV2::TilingKernelFunc moeFinalizeRoutingTilingFunc = nullptr; 
  MoeFinalizeRoutingCase();
  MoeFinalizeRoutingCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
  bool Run() override;
  bool InitParam() override;
  bool InitOpInfo() override;
  bool InitCurrentCasePtr() override;
  bool DoOpTiling(DoTilingParam &tilingParam);
};

} // namespace ascendc::ops::adv::tests::moefinalizerouting
