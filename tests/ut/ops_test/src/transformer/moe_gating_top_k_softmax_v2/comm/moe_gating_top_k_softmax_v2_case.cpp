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
 * \file moe_gating_top_k_softmax_v2_case.cpp
 * \brief MoeGatingTopKSoftmaxV2 测试用例.
 */
#include "moe_gating_top_k_softmax_v2_case.h"
#include <tikicpulib.h>
#include "tests/utils/log.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define MOE_GATING_TOP_K_SOFTMAX_V2_KERNEL_PARAM                                                                                               \
    (GM_ADDR gating, GM_ADDR finished, GM_ADDR out, GM_ADDR indicesOut, GM_ADDR softmaxOut, GM_ADDR workspace, GM_ADDR tiling)

using MoeGatingTopKSoftmaxV2KernelFunc = void(*) MOE_GATING_TOP_K_SOFTMAX_V2_KERNEL_PARAM;

extern "C" __global__ __aicore__ void moe_gating_top_k_softmax_v2 MOE_GATING_TOP_K_SOFTMAX_V2_KERNEL_PARAM;

using namespace ops::adv::tests::MoeGatingTopKSoftmaxV2;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunMoeGatingTopKSoftmaxV2(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                               std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (MoeGatingTopKSoftmaxV2KernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), outputs[0]->GetDevData(),
                outputs[1]->GetDevData(), outputs[2]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForMoeGatingTopKSoftmaxV2Stub(gert::TilingContext *context)
{
    auto *moeGatingTopKSoftmaxV2Case = static_cast<MoeGatingTopKSoftmaxV2Case *>(Case::GetCurrentCase());
    if (moeGatingTopKSoftmaxV2Case != nullptr) {
        MoeGatingTopKSoftmaxV2Case::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!moeGatingTopKSoftmaxV2Case->DoOpTiling(p)) {
            return p.ret;
        }
        return moeGatingTopKSoftmaxV2Case->moeGatingTopKSoftmaxV2TilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool MoeGatingTopKSoftmaxV2Case::InitParam()
{
    gating = Tensor("gating", {mParam.n, mParam.h}, "2", mParam.xDtype, ge::FORMAT_ND);
    finished = Tensor("finished", {mParam.n}, "1", mParam.finishDtype, ge::FORMAT_ND);
    out = Tensor("out", {mParam.n, mParam.k}, "2", mParam.xDtype, ge::FORMAT_ND);
    indicesOut = Tensor("indicesOut", {mParam.n, mParam.k}, "2", mParam.indexDtype, ge::FORMAT_ND);
    softmaxOut = Tensor("softmaxOut", {mParam.n, mParam.h}, "2", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    return true;
}

bool MoeGatingTopKSoftmaxV2Case::InitOpInfo()
{
    auto *moeKernelFunc = (void *)moe_gating_top_k_softmax_v2;
    bool rst = mCtx.SetOpName("MoeGatingTopKSoftmaxV2");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&gating, &finished});
    rst = rst && mCtx.SetOutputs({&out, &indicesOut, &softmaxOut});
    rst = rst && mCtx.SetAttrs({{"k", mParam.k}, {"renorm", mParam.renorm}});
    rst = rst && mCtx.SetKernelRunCbf(RunMoeGatingTopKSoftmaxV2);
    rst = rst && mCtx.SetKernelMainFunc(moeKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    moeGatingTopKSoftmaxV2TilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingMoeGatingTopKSoftmaxV2");
    if (moeGatingTopKSoftmaxV2TilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, moeGatingTopKSoftmaxV2(%p)", moeGatingTopKSoftmaxV2TilingFunc);
        return false;
    }
    IMPL_OP(MoeGatingTopKSoftmaxV2).Tiling(TilingForMoeGatingTopKSoftmaxV2Stub);
    return rst;
}

bool MoeGatingTopKSoftmaxV2Case::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool MoeGatingTopKSoftmaxV2Case::Run()
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

MoeGatingTopKSoftmaxV2Case::MoeGatingTopKSoftmaxV2Case(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                                       Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "MoeGatingTopKSoftmaxV2";
}

MoeGatingTopKSoftmaxV2Case::MoeGatingTopKSoftmaxV2Case()
{
}
MoeGatingTopKSoftmaxV2Case::Param::Param()
{
}
MoeGatingTopKSoftmaxV2Case::Param::Param(int64_t pN, int64_t pH, int64_t pK, int64_t pRenorm, ge::DataType xDtypeIn)

    : n(pN), h(pH), k(pK), renorm(pRenorm), xDtype(xDtypeIn)
{
}

bool MoeGatingTopKSoftmaxV2Case::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}