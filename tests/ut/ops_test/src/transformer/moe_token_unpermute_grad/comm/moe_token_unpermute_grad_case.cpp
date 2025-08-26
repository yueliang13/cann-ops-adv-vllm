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
 * \file moe_token_unpermute_grad_case.cpp
 * \brief MoeTokenUnpermuteGrad 测试用例.
 */
#include "moe_token_unpermute_grad_case.h"
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

#define MOE_TOKEN_UNPERMUTE_GRAD_KERNEL_PARAM                                                                               \
    (GM_ADDR permutedTokens, GM_ADDR unpermutedTokensGrad, GM_ADDR sortedIndices, GM_ADDR probs, GM_ADDR permutedTokensGrad, GM_ADDR probsGrad, GM_ADDR workspace, GM_ADDR tiling)

using MoeTokenUnpermuteGradKernelFunc = void(*) MOE_TOKEN_UNPERMUTE_GRAD_KERNEL_PARAM;

extern "C" __global__ __aicore__ void moe_token_unpermute_grad
    MOE_TOKEN_UNPERMUTE_GRAD_KERNEL_PARAM;

using namespace ops::adv::tests::MoeTokenUnpermuteGrad;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunMoeTokenUnpermuteGrad(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                         std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (MoeTokenUnpermuteGradKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(), inputs[3]->GetDevData(), outputs[0]->GetDevData(),
                outputs[1]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForMoeTokenUnpermuteGradStub(gert::TilingContext *context)
{
    auto *moeTokenUnpermuteGradCase = static_cast<MoeTokenUnpermuteGradCase *>(Case::GetCurrentCase());
    if (moeTokenUnpermuteGradCase != nullptr) {
        MoeTokenUnpermuteGradCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!moeTokenUnpermuteGradCase->DoOpTiling(p)) {
            return p.ret;
        }
        return moeTokenUnpermuteGradCase->moeTokenUnpermuteGradTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool MoeTokenUnpermuteGradCase::InitParam()
{
    permutedTokens = Tensor("permutedTokens", {mParam.n * mParam.k, mParam.h}, "2", mParam.permutedTokensDtype, ge::FORMAT_ND);
    unpermutedTokensGrad = Tensor("unpermutedTokensGrad", {mParam.n, mParam.h}, "2", mParam.unpermutedTokensGradDtype, ge::FORMAT_ND);
    sortedIndices = Tensor("sortedIndices", {mParam.n * mParam.k,}, "1", mParam.sortedIndicesDtype, ge::FORMAT_ND);
    probs = Tensor("probs", {mParam.n, mParam.k}, "1", mParam.probsDtype, ge::FORMAT_ND);
    permutedTokensGrad = Tensor("permutedTokensGrad", {mParam.n * mParam.k, mParam.h}, "2", mParam.permutedTokensDtype, ge::FORMAT_ND);
    probsGrad = Tensor("probsGrad", {mParam.n, mParam.k}, "2", mParam.probsDtype, ge::FORMAT_ND);
    return true;
}

bool MoeTokenUnpermuteGradCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("MoeTokenUnpermuteGrad");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&permutedTokens, &unpermutedTokensGrad, &sortedIndices, &probs});
    rst = rst && mCtx.SetOutputs({&permutedTokensGrad, &probsGrad});
    rst = rst && mCtx.SetAttrs({{"padded_mode", mParam.paddedMode}});
    rst = rst && mCtx.SetKernelRunCbf(RunMoeTokenUnpermuteGrad);
    rst = rst && mCtx.SetKernelMainFunc((void *)moe_token_unpermute_grad);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    moeTokenUnpermuteGradTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("Tiling4MoeTokenUnpermuteGrad");
    if (moeTokenUnpermuteGradTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, moeTokenUnpermuteGrad(%p)", moeTokenUnpermuteGradTilingFunc);
        return false;
    }
    IMPL_OP(MoeTokenUnpermuteGrad).Tiling(TilingForMoeTokenUnpermuteGradStub);
    return rst;
}

bool MoeTokenUnpermuteGradCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool MoeTokenUnpermuteGradCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    mOpInfo.ProcessKernel(mName);
    return true;
}

MoeTokenUnpermuteGradCase::MoeTokenUnpermuteGradCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                           Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "MoeTokenUnpermuteGrad";
}

MoeTokenUnpermuteGradCase::MoeTokenUnpermuteGradCase()
{
}
MoeTokenUnpermuteGradCase::Param::Param()
{
}
MoeTokenUnpermuteGradCase::Param::Param(int64_t pN, int64_t pH, int64_t pK, bool pPaddedMode,
                                        ge::DataType permutedTokensDtypeIn, ge::DataType unpermutedTokensGradDtypeIn,
                                        ge::DataType sortedIndicesDtypeIn, ge::DataType probsDtypeIn)

    : n(pN), h(pH), k(pK), paddedMode(pPaddedMode), permutedTokensDtype(permutedTokensDtypeIn),  unpermutedTokensGradDtype(unpermutedTokensGradDtypeIn),
    sortedIndicesDtype(sortedIndicesDtypeIn), probsDtype(probsDtypeIn)
{
}


bool MoeTokenUnpermuteGradCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}