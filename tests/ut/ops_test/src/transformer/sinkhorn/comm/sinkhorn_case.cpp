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
 * \file sinkhorn_case.cpp
 * \brief Sinkhorn 测试用例.
 */
#include "sinkhorn_case.h"
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
#define SINKHORN_KERNEL_PARAM (GM_ADDR cost, GM_ADDR p, GM_ADDR workspace, GM_ADDR tiling)

typedef void(*SinkhornKernelFunc) SINKHORN_KERNEL_PARAM;

extern "C" __global__ __aicore__ void sinkhorn SINKHORN_KERNEL_PARAM;

using namespace ops::adv::tests::sinkhornns;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunSinkhorn(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                 std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (SinkhornKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), outputs[0]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingSinkhornStub(gert::TilingContext *ctx)
{
    auto *sinkhornCase = static_cast<SinkhornCase *>(Case::GetCurrentCase());
    if (sinkhornCase != nullptr) {
        SinkhornCase::DoTilingParam tilingParam;
        tilingParam.ctx = ctx;
        tilingParam.ret = ge::GRAPH_SUCCESS;
        if (!sinkhornCase->DoOptiling(tilingParam)) {
            return tilingParam.ret;
        }
        return sinkhornCase->sinkhornTilingFunc(ctx);
    }
    return ge::GRAPH_FAILED;
}

bool SinkhornCase::InitParam()
{
    cost = Tensor("cost", {mParam.tokens, mParam.experts}, "ND", mParam.dataType, ge::FORMAT_ND);
    p = Tensor("p", {mParam.tokens, mParam.experts}, "ND", mParam.dataType, ge::FORMAT_ND);
    return true;
}

bool SinkhornCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("Sinkhorn");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&cost});
    rst = rst && mCtx.SetOutputs({&p});
    rst = rst && mCtx.SetAttrs({{"tol", mParam.tol}});
    rst = rst && mCtx.SetKernelRunCbf(RunSinkhorn);
    rst = rst && mCtx.SetKernelMainFunc((void *)sinkhorn);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    sinkhornTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingForSinkhorn");
    if (sinkhornTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, sinkhorn(%p)", sinkhornTilingFunc);
        return false;
    }
    IMPL_OP(Sinkhorn).Tiling(TilingSinkhornStub);

    return rst;
}

bool SinkhornCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool SinkhornCase::Run()
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

SinkhornCase::SinkhornCase(const char *name, bool enable, const char *dbgInfo, OpInfo sinkhorn, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(sinkhorn)), mParam(std::move(param))
{
    this->mOpInfo.mName = "Sinkhorn";
}

SinkhornCase::SinkhornCase()
{
    this->mName = "SinkhornCase";
    this->mOpInfo.mName = "Sinkhorn";
}

SinkhornCase::Param::Param()
{
}

SinkhornCase::Param::Param(int64_t pTokens, int64_t pExperts, float pTol, ge::DataType pDataType)
    : tokens(pTokens), experts(pExperts), tol(pTol), dataType(pDataType)
{
}

bool SinkhornCase::DoOptiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}