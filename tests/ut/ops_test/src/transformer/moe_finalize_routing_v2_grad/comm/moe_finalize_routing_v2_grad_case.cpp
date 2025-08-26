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
 * \file moe_finalize_routing_v2_grad_case.cpp
 * \brief MoeFinalizeRoutingV2Grad 测试用例.
 */
#include "moe_finalize_routing_v2_grad_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/tiling_base.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define Moe_Finalize_Routing_V2_Grad_KERNEL_PARAM                                                                               \
    (GM_ADDR gradY, GM_ADDR expandedRowIdx,GM_ADDR expandedX, GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR bias, GM_ADDR gradExpandedX, GM_ADDR gradScales, GM_ADDR workspace, GM_ADDR tiling)

using MoeFinalizeRoutingV2GradKernelFunc = void(*) Moe_Finalize_Routing_V2_Grad_KERNEL_PARAM;

extern "C" __global__ __aicore__ void moe_finalize_routing_v2_grad Moe_Finalize_Routing_V2_Grad_KERNEL_PARAM;

using namespace ops::adv::tests::MoeFinalizeRoutingV2Grad;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunMoeFinalizeRoutingV2Grad(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                         std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (MoeFinalizeRoutingV2GradKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(),inputs[2]->GetDevData(),inputs[3]->GetDevData(),inputs[4]->GetDevData(),
    inputs[5]->GetDevData(), outputs[0]->GetDevData(), outputs[1]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForMoeFinalizeRoutingV2GradStub(gert::TilingContext *context)
{
    auto *moeFinalizeRoutingV2GradCase = static_cast<MoeFinalizeRoutingV2GradCase *>(Case::GetCurrentCase());
    if (moeFinalizeRoutingV2GradCase != nullptr) {
        MoeFinalizeRoutingV2GradCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!moeFinalizeRoutingV2GradCase->DoOpTiling(p)) {
            return p.ret;
        }
        return moeFinalizeRoutingV2GradCase->moeFinalizeRoutingV2GradTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool MoeFinalizeRoutingV2GradCase::InitParam()
{
    gradY = Tensor("gradY", {mParam.numRows, mParam.h}, "2", mParam.dataType, ge::FORMAT_ND);
    expertIdx = Tensor("expertIdx", {mParam.numRows, mParam.k}, "2", ge::DataType::DT_INT32, ge::FORMAT_ND);
    expandedRowIdx = Tensor("expandedRowIdx", {mParam.numRows * mParam.k}, "1", ge::DataType::DT_INT32, ge::FORMAT_ND);
    if(mParam.dropPadMode == 0 || mParam.dropPadMode == 2) {
        expandedX = Tensor("expandedX", {mParam.numRows * mParam.k, mParam.h}, "2", mParam.dataType, ge::FORMAT_ND);
        gradExpandedX = Tensor("gradExpandedX", {mParam.numRows * mParam.k, mParam.h}, "2", mParam.dataType, ge::FORMAT_ND);
    } else {
        expandedX = Tensor("expandedX", {mParam.e, mParam.c, mParam.h}, "3", mParam.dataType, ge::FORMAT_ND);
        gradExpandedX = Tensor("gradExpandedX", {mParam.e, mParam.c, mParam.h}, "3", mParam.dataType, ge::FORMAT_ND);
    }
    scales = Tensor("scales", {mParam.numRows, mParam.k}, "2", mParam.dataType, ge::FORMAT_ND);
    bias = Tensor("bias", {mParam.e, mParam.h}, "2", mParam.dataType, ge::FORMAT_ND);
    gradScales = Tensor("gradScales", {mParam.numRows, mParam.k}, "1", mParam.dataType, ge::FORMAT_ND);
    return true;
}

bool MoeFinalizeRoutingV2GradCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("MoeFinalizeRoutingV2Grad");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&gradY, &expandedRowIdx, &expandedX, &scales, &expertIdx, &bias});
    rst =
        rst && mCtx.SetOutputs({&gradExpandedX, &gradScales});
    rst = rst && mCtx.SetAttrs({{"drop_pad_mode", mParam.dropPadMode}});
    rst = rst && mCtx.SetKernelRunCbf(RunMoeFinalizeRoutingV2Grad);
    rst = rst && mCtx.SetKernelMainFunc((void *)moe_finalize_routing_v2_grad);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    moeFinalizeRoutingV2GradTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("Tiling4MoeFinalizeRoutingV2Grad");
    if (moeFinalizeRoutingV2GradTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, moeFinalizeRoutingV2Grad(%p)", moeFinalizeRoutingV2GradTilingFunc);
        return false;
    }
    IMPL_OP(MoeFinalizeRoutingV2Grad).Tiling(TilingForMoeFinalizeRoutingV2GradStub);
    return rst;
}

bool MoeFinalizeRoutingV2GradCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool MoeFinalizeRoutingV2GradCase::Run()
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

MoeFinalizeRoutingV2GradCase::MoeFinalizeRoutingV2GradCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                           Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "MoeFinalizeRoutingV2Grad";
}

MoeFinalizeRoutingV2GradCase::MoeFinalizeRoutingV2GradCase()
{
}
MoeFinalizeRoutingV2GradCase::Param::Param()
{
}
MoeFinalizeRoutingV2GradCase::Param::Param(int64_t pE, int64_t pC,int64_t pNumRows, int64_t pH, int64_t pK, int64_t pDropPadMode, ge::DataType pdataType)

    : e(pE), c(pC), numRows(pNumRows), h(pH), k(pK), dropPadMode(pDropPadMode), dataType(pdataType)
{
}


bool MoeFinalizeRoutingV2GradCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}