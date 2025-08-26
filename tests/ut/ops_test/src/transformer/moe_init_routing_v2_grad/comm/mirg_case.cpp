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
 * \file mirg_case.cpp
 * \brief MoeInitRoutingV2Grad 测试用例.
 */
 #include "mirg_case.h"
 
 #include <utility>
 #include <tikicpulib.h>
 #include <graph/utils/type_utils.h>
 #include <register/op_impl_registry.h>

 #include "tests/utils/log.h"
 #include "tests/utils/platform.h"
 #include "tiling/mirg/tiling_data.h"
 #include "tiling/tiling_base.h"
 
 /**
  * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
  * 参数所控制的 Kernel 入口一致.
  */
 
 #define MIRG_KERNEL_PARAM                                                                                               \
     (GM_ADDR gradExpandedX, \
         GM_ADDR expandedRowIdx,\
         GM_ADDR out,\
         GM_ADDR workspace, GM_ADDR tiling)
 
 typedef void(*MirgKernelFunc) MIRG_KERNEL_PARAM;
 
 extern "C" __global__ __aicore__ void moe_init_routing_v2_grad MIRG_KERNEL_PARAM;

 using namespace ops::adv::tests::mirg;
 using TensorIntf = ops::adv::tests::utils::TensorIntf;
 using Case = ops::adv::tests::utils::Case;
 using Platform = ops::adv::tests::utils::Platform;
 
 bool RunMoeInitRoutingV2Grad(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                             std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
 {
     // Kernel 运行
     auto kernelFunc = (MirgKernelFunc)func;
     ICPU_SET_TILING_KEY(tilingKey);
     ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), outputs[0]->GetDevData(),
                 workspace, tilingData);
     return true;
 }
 
 extern "C" ge::graphStatus TilingMoeInitRoutingV2Grad(gert::TilingContext *context)
 {
     auto *mirgCase = static_cast<MirgCase *>(Case::GetCurrentCase());
     if (mirgCase != nullptr) {
         MirgCase::DoTilingParam p;
         p.ctx = context;
         p.ret = ge::GRAPH_SUCCESS;
         if (!mirgCase->DoOpTiling(p)) {
             return p.ret;
         }
         return mirgCase->mirgTilingFunc(context);
     }
     return ge::GRAPH_FAILED;
 }
 
 bool MirgCase::InitParam()
 {
    int expand_n = mParam.activeNum > 0 ? mParam.activeNum : mParam.num_rows * mParam.k;
    if(mParam.dropPadMode == 1) {
        gradExpandedX = Tensor("gradExpandX", {mParam.e, mParam.c, mParam.hidden_size}, "3", mParam.dx, ge::FORMAT_ND);
    } else {
        gradExpandedX = Tensor("gradExpandX", {expand_n, mParam.hidden_size}, "2", mParam.dx, ge::FORMAT_ND);
    }

     expandedRowIdx = Tensor("expandedRowIdx", {mParam.num_rows * mParam.k}, "1", ge::DataType::DT_INT32, ge::FORMAT_ND);
     out = Tensor("out", {mParam.num_rows, mParam.hidden_size}, "2", mParam.dx, ge::FORMAT_ND);
 
     return true;
 }
 
 bool MirgCase::InitOpInfo()
 {
     auto *mirgKernelFunc = (void *)moe_init_routing_v2_grad;
 
     bool rst = mCtx.SetOpName("MoeInitRoutingV2Grad");
     rst = rst && mCtx.SetDeterministic(false);
     rst = rst && mCtx.SetInputs({&gradExpandedX, &expandedRowIdx});
     rst = rst && mCtx.SetOutputs({&out});
     rst = rst && mCtx.SetAttrs({{"top_k", mParam.k}, {"drop_pad_mode", mParam.dropPadMode}, {"active_num", mParam.activeNum}});
     rst = rst && mCtx.SetKernelRunCbf(RunMoeInitRoutingV2Grad);
     rst = rst && mCtx.SetKernelMainFunc(mirgKernelFunc);
     rst = rst && mOpInfo.SetContext(&mCtx);
 
     auto *platform = Platform::GetGlobalPlatform();
     if (platform == nullptr) {
         LOG_ERR("Global Platform is null");
         return false;
     }
 
     mirgTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingForMoeInitRoutingV2Grad");
     if (mirgTilingFunc == nullptr) {
         LOG_ERR("Can't get origin tiling func, mirg(%p)", mirgTilingFunc);
         return false;
     }
     IMPL_OP(MoeInitRoutingV2Grad).Tiling(TilingMoeInitRoutingV2Grad);
     return true;
 }
 
 bool MirgCase::InitCurrentCasePtr()
 {
     Case::mCurrentCasePtr = this;
     return true;
 }
 
 bool MirgCase::Run()
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
 
 MirgCase::MirgCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param)
     : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
 {
     this->mOpInfo.mName = "MoeInitRoutingV2Grad";
 }
 
 MirgCase::MirgCase()
 {
 }
 MirgCase::Param::Param()
 {
 }
 MirgCase::Param::Param(int64_t num_rows_, int64_t k_, int64_t hidden_size_, int64_t e_, int64_t c_, int64_t drop_pad_mode_, int64_t active_num_, ge::DataType dx_)
     : num_rows(num_rows_), k(k_), hidden_size(hidden_size_), e(e_), c(c_), dropPadMode(drop_pad_mode_), activeNum(active_num_), dx(dx_)
 {
 }
 
 
 bool MirgCase::DoOpTiling(DoTilingParam& tilingParam) {
   if (tilingParam.ctx == nullptr) {
     return false;
   }
   return true;
 }