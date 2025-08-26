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
 * \file moe_finalize_routing_case.cpp
 * \brief moe_finalize_routing 测试用例.
 */

#include "moe_finalize_routing_case.h"
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/tiling_base.h"

  /**
   *以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
   *参数所控制的 Kernel 入口一致.
   */
#define Moe_Finalize_Routing_KERNEL_PARAM                                                                               \
    (GM_ADDR expandedPermutedRows, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias, GM_ADDR scales, GM_ADDR expandedSrcToDstRow, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)

using MoeFinalizeRoutingKernelFunc = void(*) Moe_Finalize_Routing_KERNEL_PARAM;

extern "C" __global__ __aicore__ void moe_finalize_routing Moe_Finalize_Routing_KERNEL_PARAM;

using namespace ops::adv::tests::MoeFinalizeRoutingCase;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunMoeFinalizeRouting(void* func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf*>& inputs,
                          std::vector<TensorIntf*>& outputs, uint8_t* workspace, uint8_t* tilingData)
{
// Kernel 运行
  auto kernelFunc = (MoeFinalizeRoutingKernelFunc)func;
  ICPU_SET_TILING_KEY(tilingKey);
  ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
              inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(),
              outputs[0]->GetDevData(), workspace, tilingData);
  return true;
}

extern "C" ge::graphStatus TilingMoeFinalizeRouting(gert::TilingContext *context)
{
  auto *moeFinalizeRoutingCase = static_cast<MoeFinalizeRoutingCase *>(Case::GetCurrentCase());
  if (moeFinalizeRoutingCase != nullptr) {
    MoeFinalizeRoutingCase::DoTilingParam p;
    p.ctx = context;
    p.ret = ge::GRAPH_SUCCESS;
  if (!moeFinalizeRoutingCase->DoOpTiling(p)) {
    return p.ret;
  }
  return moeFinalizeRoutingCase->moeFinalizeRoutingTilingFunc(context);
}
return ge::GRAPH_FAILED;
}

bool MoeFinalizeRoutingCase::InitParam()
{
  expandedPermutedRows = Tensor("expandedPermutedRows", {mParam.num_rows * mParam.top_k, mParam.token_len}, "2", mParam.dType, ge::FORMAT_ND);
  skip1 = Tensor("skip1", {mParam.num_rows, mParam.token_len},"2", mParam.dType, ge::FORMAT_ND);
  skip2 = Tensor("skip2", {mParam.num_rows, mParam.token_len},"2", mParam.dType, ge::FORMAT_ND);
  bias = Tensor("bias", {mParam.expert_num, mParam.token_len},"2", mParam.dType, ge::FORMAT_ND);
  scales = Tensor("scales", {mParam.num_rows, mParam.top_k},"2", mParam.dType, ge::FORMAT_ND);
  expandedSrcToDstRow =
  Tensor("expandedSrcToDstRow", {mParam.num_rows * mParam.top_k},"1", ge::DT_INT32, ge::FORMAT_ND);
  expertForSourceRow =
  Tensor("expertForSourceRow", {mParam.num_rows, mParam.top_k},"2", ge::DT_INT32, ge::FORMAT_ND);

  out = Tensor("out", {mParam.num_rows, mParam.token_len},"2", mParam.dType, ge::FORMAT_ND);

  return true;
}

bool MoeFinalizeRoutingCase::InitOpInfo(){
  bool rst = mCtx.SetOpName("MoeFinalizeRouting");
  rst = rst && mCtx.SetDeterministic(false);
  rst = rst && mCtx.SetInputs(
  {&expandedPermutedRows, &skip1, &skip2, &bias, &scales, &expandedSrcToDstRow, &expertForSourceRow});
  rst = rst && mCtx.SetOutputs({&out});
  rst = rst && mCtx.SetKernelRunCbf(RunMoeFinalizeRouting);
  rst = rst && mCtx.SetKernelMainFunc((void *)moe_finalize_routing);
  rst = rst && mOpInfo.SetContext(&mCtx);

  auto *platform = Platform::GetGlobalPlatform();
  if (platform == nullptr) {
      LOG_ERR("Global Platform is null");
      return false;
  }

 moeFinalizeRoutingTilingFunc =
     (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("Tiling4MoeFinalizeRouting");
  if (moeFinalizeRoutingTilingFunc == nullptr) {
      LOG_ERR("Can't get origin tiling func, moeFinalizeRouting(%p)", moeFinalizeRoutingTilingFunc);
      return false;
  }
  IMPL_OP(MoeFinalizeRouting).Tiling(TilingMoeFinalizeRouting);
  return rst;
}

bool MoeFinalizeRoutingCase::InitCurrentCasePtr()
{
  Case::mCurrentCasePtr = this;
  return true;
}

bool MoeFinalizeRoutingCase::Run()
{
  if (!mEnable) {
    return true;
  }
  if (!mOpInfo.ProcessTiling(mName)) {
    return false;
  }
  if (!mOpInfo.ProcessKernel(mName)) {
    return false;
  }
  return true;
}

MoeFinalizeRoutingCase::MoeFinalizeRoutingCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param)
  : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
  this->mOpInfo.mName = "MoeFinalizeRouting";
}

MoeFinalizeRoutingCase::MoeFinalizeRoutingCase()
{
}
MoeFinalizeRoutingCase::Param::Param()
{
}
MoeFinalizeRoutingCase::Param::Param(int pExpert_num, int pToken_len, int pTop_k, int pNum_rows, std::string pLayout, ge::DataType pDataType)
  : expert_num(pExpert_num), token_len(pToken_len), top_k(pTop_k), num_rows(pNum_rows), layout(pLayout), dType(pDataType)
{
}

bool MoeFinalizeRoutingCase::DoOpTiling(DoTilingParam& tilingParam) {
  if (tilingParam.ctx == nullptr) {
    return false;
  }
  return true;
}