/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_token_permute_case.cpp
 * \brief MoeTokenPermute 测试用例.
 */
#include "moe_token_permute_case.h"
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

#define MOE_TOKEN_PERMUTE_KERNEL_PARAM                                                                               \
    (GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR workspace, GM_ADDR tiling)

using MoeTokenPermuteKernelFunc = void(*) MOE_TOKEN_PERMUTE_KERNEL_PARAM;

extern "C" __global__ __aicore__ void moe_token_permute MOE_TOKEN_PERMUTE_KERNEL_PARAM;

using namespace ops::adv::tests::MoeTokenPermute;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunMoeTokenPermute(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                         std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (MoeTokenPermuteKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), outputs[0]->GetDevData(),
                outputs[1]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForMoeTokenPermuteStub(gert::TilingContext *context)
{
    auto *moeTokenPermuteCase = static_cast<MoeTokenPermuteCase *>(Case::GetCurrentCase());
    if (moeTokenPermuteCase != nullptr) {
        MoeTokenPermuteCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!moeTokenPermuteCase->DoOpTiling(p)) {
            return p.ret;
        }
        return moeTokenPermuteCase->moeTokenPermuteTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool MoeTokenPermuteCase::InitParam()
{
    x = Tensor("x", {mParam.n, mParam.h}, "2", mParam.xDtype, ge::FORMAT_ND);
    expertIdx = Tensor("expertIdx", {mParam.n, mParam.k}, "2", mParam.indexDtype, ge::FORMAT_ND);
    // expandX的shape由参数决定
    int64_t firstDim = mParam.n * mParam.k;
    if (mParam.activeNum > 0 && mParam.activeNum < firstDim) {
        firstDim = mParam.activeNum;
    }
    expandedX = Tensor("expandedX", {firstDim, mParam.h}, "2", mParam.xDtype, ge::FORMAT_ND);
    expandedRowIdx = Tensor("expandedRowIdx", {mParam.n * mParam.k}, "1", ge::DataType::DT_INT32, ge::FORMAT_ND);

    return true;
}

bool MoeTokenPermuteCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("MoeTokenPermute");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&x, &expertIdx});
    rst = rst && mCtx.SetOutputs({&expandedX, &expandedRowIdx});
    rst = rst && mCtx.SetAttrs({{"num_out_tokens", mParam.activeNum},
                                {"padded_mode", mParam.c}});
    rst = rst && mCtx.SetKernelRunCbf(RunMoeTokenPermute);
    rst = rst && mCtx.SetKernelMainFunc((void *)moe_token_permute);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    moeTokenPermuteTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingForMoeTokenPermute");
    if (moeTokenPermuteTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, moeTokenPermute(%p)", moeTokenPermuteTilingFunc);
        return false;
    }
    IMPL_OP(MoeTokenPermute).Tiling(TilingForMoeTokenPermuteStub);
    return rst;
}

bool MoeTokenPermuteCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool MoeTokenPermuteCase::Run()
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

MoeTokenPermuteCase::MoeTokenPermuteCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                           Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "MoeTokenPermute";
}

MoeTokenPermuteCase::MoeTokenPermuteCase()
{
}
MoeTokenPermuteCase::Param::Param()
{
}
MoeTokenPermuteCase::Param::Param(int64_t pN, int64_t pH, int64_t pK, int64_t pActiveNum, int64_t pC, int64_t pE,
                                  int64_t pDropPadMode, ge::DataType indexDtypeIn, ge::DataType xDtypeIn)

    : n(pN), h(pH), k(pK), activeNum(pActiveNum), c(pC), e(pE), dropPadMode(pDropPadMode), indexDtype(indexDtypeIn),  xDtype(xDtypeIn)
{
}


bool MoeTokenPermuteCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}