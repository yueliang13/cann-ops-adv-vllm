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
 * \file grouped_bias_add_grad_case.cpp
 * \brief GroupedBiasAddGrad 测试用例.
 */
#include "grouped_bias_add_grad_case.h"
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

#define GROUPED_BIAS_ADD_GRAD_KERNEL_PARAM                                                                               \
    (GM_ADDR grad_y, GM_ADDR group_idx, GM_ADDR grad_bias,GM_ADDR workspace, GM_ADDR tiling_data)

using GroupedBiasAddGradKernelFunc = void(*) GROUPED_BIAS_ADD_GRAD_KERNEL_PARAM;

extern "C" __global__ __aicore__ void grouped_bias_add_grad GROUPED_BIAS_ADD_GRAD_KERNEL_PARAM;

using namespace ops::adv::tests::GroupedBiasAddGrad;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunGroupedBiasAddGrad(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                         std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (GroupedBiasAddGradKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), outputs[0]->GetDevData(),
                workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForGroupedBiasAddGradStub(gert::TilingContext *context)
{
    auto *groupedBiasAddGradCase = static_cast<GroupedBiasAddGradCase *>(Case::GetCurrentCase());
    if (groupedBiasAddGradCase != nullptr) {
        GroupedBiasAddGradCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!groupedBiasAddGradCase->DoOpTiling(p)) {
            return p.ret;
        }
        return groupedBiasAddGradCase->groupedBiasAddGradTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool GroupedBiasAddGradCase::InitParam()
{
    if (mParam.inputOpt) {
        x = Tensor("x", {mParam.n, mParam.h}, "2", mParam.xDtype, ge::FORMAT_ND);
        groupIdx = Tensor("groupIdx", {mParam.k}, "1", mParam.indexDtype, ge::FORMAT_ND);
    } else {
        x = Tensor("x", {mParam.n,3, mParam.h}, "3", mParam.xDtype, ge::FORMAT_ND);
        groupIdx = Tensor("groupIdx", {}, "None", mParam.indexDtype, ge::FORMAT_ND);
    }
    // expandX的shape由参数决定

    gradBias = Tensor("gradBias", {mParam.k, mParam.h}, "2", mParam.xDtype, ge::FORMAT_ND);

    return true;
}

bool GroupedBiasAddGradCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("GroupedBiasAddGrad");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&x, &groupIdx});
    rst = rst && mCtx.SetOutputs({&gradBias});
    rst = rst && mCtx.SetAttrs({{"group_idx_type", 0}});
    rst = rst && mCtx.SetKernelRunCbf(RunGroupedBiasAddGrad);
    rst = rst && mCtx.SetKernelMainFunc((void *)grouped_bias_add_grad);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    groupedBiasAddGradTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingForGroupedBiasAddGrad");
    if (groupedBiasAddGradTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, groupedBiasAddGrad(%p)", groupedBiasAddGradTilingFunc);
        return false;
    }
    IMPL_OP(GroupedBiasAddGrad).Tiling(TilingForGroupedBiasAddGradStub);
    return rst;
}

bool GroupedBiasAddGradCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool GroupedBiasAddGradCase::Run()
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

GroupedBiasAddGradCase::GroupedBiasAddGradCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                           Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "GroupedBiasAddGrad";
}

GroupedBiasAddGradCase::GroupedBiasAddGradCase()
{
}
GroupedBiasAddGradCase::Param::Param()
{
}

GroupedBiasAddGradCase::Param::Param(int64_t pN, int64_t pH, int64_t pK, bool pInputOpt, ge::DataType indexDtypeIn, ge::DataType xDtypeIn)

    : n(pN), h(pH), k(pK), inputOpt(pInputOpt), indexDtype(indexDtypeIn),  xDtype(xDtypeIn)
{
}


bool GroupedBiasAddGradCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}