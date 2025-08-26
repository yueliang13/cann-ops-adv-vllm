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
 * \file rope_grad_case.cpp
 * \brief RotaryPositionEmbeddingGrad 测试用例.
 */
#include "rope_grad_case.h"
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
#define ROPE_GRAD_KERNEL_PARAM                                                                                          \
    (GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR x, GM_ADDR xGrad, GM_ADDR cosGrad, GM_ADDR sinGrad,               \
     GM_ADDR workspace, GM_ADDR tiling)

typedef void(*RopeGradKernelFunc) ROPE_GRAD_KERNEL_PARAM;

extern "C" __global__ __aicore__ void rotary_position_embedding_grad ROPE_GRAD_KERNEL_PARAM;

using namespace ops::adv::tests::rope_grad;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunRotaryPositionEmbeddingGrad(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                                    std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (RopeGradKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), outputs[0]->GetDevData(), outputs[1]->GetDevData(), outputs[2]->GetDevData(),
                workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingRotaryPositionEmbeddingGradStub(gert::TilingContext *ctx)
{
    auto *ropeGradCase = static_cast<RopeGradCase *>(Case::GetCurrentCase());
    if (ropeGradCase != nullptr) {
        RopeGradCase::DoTilingParam tilingParam;
        tilingParam.ctx = ctx;
        tilingParam.ret = ge::GRAPH_SUCCESS;
        if (!ropeGradCase->DoOptiling(tilingParam)) {
            return tilingParam.ret;
        }
        return ropeGradCase->ropeGradTilingFunc(ctx);
    }
    return ge::GRAPH_FAILED;
}

bool RopeGradCase::InitParam()
{
    if (mParam.layout == "BNSD") {
        grad = Tensor("grad", {mParam.b, mParam.n, mParam.s, mParam.d}, "BNSD", mParam.dataType, ge::FORMAT_ND);
        cos = Tensor("cos", {mParam.triB, mParam.triN, mParam.s, mParam.d}, "BNSD", mParam.dataType, ge::FORMAT_ND);
        sin = Tensor("sin", {mParam.triB, mParam.triN, mParam.s, mParam.d}, "BNSD", mParam.dataType, ge::FORMAT_ND);
        xGrad = Tensor("xGrad", {mParam.b, mParam.n, mParam.s, mParam.d}, "BNSD", mParam.dataType, ge::FORMAT_ND);
        cosGrad =
            Tensor("cosGrad", {mParam.triB, mParam.triN, mParam.s, mParam.d}, "BNSD", mParam.dataType, ge::FORMAT_ND);
        sinGrad =
            Tensor("sinGrad", {mParam.triB, mParam.triN, mParam.s, mParam.d}, "BNSD", mParam.dataType, ge::FORMAT_ND);
    } else if (mParam.layout == "BSND") {
        grad = Tensor("grad", {mParam.b, mParam.s, mParam.n, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
        cos = Tensor("cos", {mParam.triB, mParam.s, mParam.triN, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
        sin = Tensor("sin", {mParam.triB, mParam.s, mParam.triN, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
        xGrad = Tensor("xGrad", {mParam.b, mParam.s, mParam.n, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
        cosGrad =
            Tensor("cosGrad", {mParam.triB, mParam.s, mParam.triN, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
        sinGrad =
            Tensor("sinGrad", {mParam.triB, mParam.s, mParam.triN, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
    } else if (mParam.layout == "SBND") {
        grad = Tensor("grad", {mParam.s, mParam.b, mParam.n, mParam.d}, "SBND", mParam.dataType, ge::FORMAT_ND);
        cos = Tensor("cos", {mParam.s, mParam.triB, mParam.triN, mParam.d}, "SBND", mParam.dataType, ge::FORMAT_ND);
        sin = Tensor("sin", {mParam.s, mParam.triB, mParam.triN, mParam.d}, "SBND", mParam.dataType, ge::FORMAT_ND);
        xGrad = Tensor("xGrad", {mParam.s, mParam.b, mParam.n, mParam.d}, "SBND", mParam.dataType, ge::FORMAT_ND);
        cosGrad =
            Tensor("cosGrad", {mParam.s, mParam.triB, mParam.triN, mParam.d}, "SBND", mParam.dataType, ge::FORMAT_ND);
        sinGrad =
            Tensor("sinGrad", {mParam.s, mParam.triB, mParam.triN, mParam.d}, "SBND", mParam.dataType, ge::FORMAT_ND);
    }
    return true;
}

bool RopeGradCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("RotaryPositionEmbeddingGrad");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&grad, &cos, &sin, &x});
    rst = rst && mCtx.SetOutputs({&xGrad, &sinGrad, &cosGrad});
    rst = rst && mCtx.SetAttrs({{"mode", mParam.mode}});

    rst = rst && mCtx.SetKernelRunCbf(RunRotaryPositionEmbeddingGrad);
    rst = rst && mCtx.SetKernelMainFunc((void *)rotary_position_embedding_grad);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    ropeGradTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingRotaryPositionEmbeddingGrad");
    if (ropeGradTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, rope(%p)", ropeGradTilingFunc);
        return false;
    }
    IMPL_OP(RotaryPositionEmbeddingGrad).Tiling(TilingRotaryPositionEmbeddingGradStub);

    return rst;
}

bool RopeGradCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool RopeGradCase::Run()
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

RopeGradCase::RopeGradCase(const char *name, bool enable, const char *dbgInfo, OpInfo rope, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(rope)), mParam(std::move(param))
{
    this->mOpInfo.mName = "RotaryPositionEmbeddingGrad";
}

RopeGradCase::RopeGradCase()
{
    this->mName = "RopeGradCase";
    this->mOpInfo.mName = "RotaryPositionEmbeddingGrad";
}

RopeGradCase::Param::Param()
{
}

RopeGradCase::Param::Param(int64_t pb, int64_t pn, int64_t ps, int64_t pd, int64_t pTriB, int64_t pTriN, int64_t pMode,
                           std::string pLayout, ge::DataType pDataType)
    : b(pb), n(pn), s(ps), d(pd), triB(pTriB), triN(pTriN), mode(pMode), layout(pLayout), dataType(pDataType)
{
}

bool RopeGradCase::DoOptiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}