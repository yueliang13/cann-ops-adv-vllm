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
 * \file moe_compute_expert_tokens_case.cpp
 * \brief MoeComputeExpertTokens 测试用例.
 */
#include "moe_compute_expert_tokens_case.h"
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

#define Moe_Compute_Expert_Tokens_KERNEL_PARAM                                                                               \
    (GM_ADDR sortExperts, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)

using MoeComputeExpertTokensKernelFunc = void(*) Moe_Compute_Expert_Tokens_KERNEL_PARAM;

extern "C" __global__ __aicore__ void moe_compute_expert_tokens Moe_Compute_Expert_Tokens_KERNEL_PARAM;

using namespace ops::adv::tests::MoeComputeExpertTokens;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunMoeComputeExpertTokens(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                         std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (MoeComputeExpertTokensKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), outputs[0]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForMoeComputeExpertTokensStub(gert::TilingContext *context)
{
    auto *moeComputeExpertTokensCase = static_cast<MoeComputeExpertTokensCase *>(Case::GetCurrentCase());
    if (moeComputeExpertTokensCase != nullptr) {
        MoeComputeExpertTokensCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!moeComputeExpertTokensCase->DoOpTiling(p)) {
            return p.ret;
        }
        return moeComputeExpertTokensCase->moeComputeExpertTokensTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool MoeComputeExpertTokensCase::InitParam()
{
    sorted_experts = Tensor("sorted_experts", {mParam.n}, "1", ge::DataType::DT_INT32, ge::FORMAT_ND);
    out = Tensor("out", {mParam.e}, "1", ge::DataType::DT_INT32, ge::FORMAT_ND);
    return true;
}

bool MoeComputeExpertTokensCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("MoeComputeExpertTokens");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&sorted_experts});
    rst =
        rst && mCtx.SetOutputs({&out});
    rst = rst && mCtx.SetAttrs({{"num_experts", mParam.numExperts}});
    rst = rst && mCtx.SetKernelRunCbf(RunMoeComputeExpertTokens);
    rst = rst && mCtx.SetKernelMainFunc((void *)moe_compute_expert_tokens);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    moeComputeExpertTokensTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("Tiling4MoeComputeExpertTokens");
    if (moeComputeExpertTokensTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, moeComputeExpertTokens(%p)", moeComputeExpertTokensTilingFunc);
        return false;
    }
    IMPL_OP(MoeComputeExpertTokens).Tiling(TilingForMoeComputeExpertTokensStub);
    return rst;
}

bool MoeComputeExpertTokensCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool MoeComputeExpertTokensCase::Run()
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

MoeComputeExpertTokensCase::MoeComputeExpertTokensCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                           Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "MoeComputeExpertTokens";
}

MoeComputeExpertTokensCase::MoeComputeExpertTokensCase()
{
}
MoeComputeExpertTokensCase::Param::Param()
{
}
MoeComputeExpertTokensCase::Param::Param(int64_t pN, int64_t pE, int64_t pNumExperts, ge::DataType pDataType)

    : n(pN), e(pE), numExperts(pNumExperts), dataType(pDataType)
{
}


bool MoeComputeExpertTokensCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}