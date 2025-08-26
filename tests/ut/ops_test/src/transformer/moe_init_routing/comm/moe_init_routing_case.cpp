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
 * \file moe_init_routing_case.cpp
 * \brief MoeInitRouting 测试用例.
 */
#include "moe_init_routing_case.h"
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

#define MOE_INIT_ROUTING_KERNEL_PARAM                                                                               \
    (GM_ADDR x, GM_ADDR rowIdx, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR expandedExpertIdx, GM_ADDR workspace, GM_ADDR tiling)

using MoeInitRoutingKernelFunc = void(*) MOE_INIT_ROUTING_KERNEL_PARAM;

extern "C" __global__ __aicore__ void moe_init_routing MOE_INIT_ROUTING_KERNEL_PARAM;

using namespace ops::adv::tests::MoeInitRouting;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunMoeInitRouting(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                         std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (MoeInitRoutingKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(), outputs[0]->GetDevData(),
                outputs[1]->GetDevData(), outputs[2]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForMoeInitRoutingStub(gert::TilingContext *context)
{
    auto *moeInitRoutingCase = static_cast<MoeInitRoutingCase *>(Case::GetCurrentCase());
    if (moeInitRoutingCase != nullptr) {
        MoeInitRoutingCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!moeInitRoutingCase->DoOpTiling(p)) {
            return p.ret;
        }
        return moeInitRoutingCase->moeInitRoutingTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool MoeInitRoutingCase::InitParam()
{
    x = Tensor("x", {mParam.n, mParam.h}, "2", mParam.xDtype, ge::FORMAT_ND);
    rowIdx = Tensor("rowIdx", {mParam.n, mParam.k}, "2", mParam.yDtype, ge::FORMAT_ND);
    expertIdx = Tensor("expertIdx", {mParam.n, mParam.k}, "2", mParam.xDtype, ge::FORMAT_ND);
    int64_t firstDim = std::min(mParam.n, mParam.activeNum);
    expandedX = Tensor("expandedX", {firstDim * mParam.k, mParam.h}, "2", mParam.yDtype, ge::FORMAT_ND);
    expandedRowIdx = Tensor("expandedRowIdx", {mParam.n * mParam.k}, "1", mParam.xDtype, ge::FORMAT_ND);
    expandedExpertIdx = Tensor("expandedExpertIdx", {mParam.n * mParam.k}, "1", mParam.xDtype, ge::FORMAT_ND);
    return true;
}

bool MoeInitRoutingCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("MoeInitRouting");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&x, &rowIdx, &expertIdx});
    rst =
        rst && mCtx.SetOutputs({&expandedX, &expandedRowIdx, &expandedExpertIdx});
    rst = rst && mCtx.SetAttrs({{"active_num", mParam.activeNum}});
    rst = rst && mCtx.SetKernelRunCbf(RunMoeInitRouting);
    rst = rst && mCtx.SetKernelMainFunc((void *)moe_init_routing);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    moeInitRoutingTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingForMoeInitRouting");
    if (moeInitRoutingTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, moeInitRouting(%p)", moeInitRoutingTilingFunc);
        return false;
    }
    IMPL_OP(MoeInitRouting).Tiling(TilingForMoeInitRoutingStub);
    return rst;
}

bool MoeInitRoutingCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool MoeInitRoutingCase::Run()
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

MoeInitRoutingCase::MoeInitRoutingCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                           Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "MoeInitRouting";
}

MoeInitRoutingCase::MoeInitRoutingCase()
{
}
MoeInitRoutingCase::Param::Param()
{
}
MoeInitRoutingCase::Param::Param(int64_t pN, int64_t pH, int64_t pK, int64_t pActiveNum, ge::DataType pxDtype, ge::DataType pyDtype)

    : n(pN), h(pH), k(pK), activeNum(pActiveNum), xDtype(pxDtype), yDtype(pyDtype)
{
}


bool MoeInitRoutingCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}