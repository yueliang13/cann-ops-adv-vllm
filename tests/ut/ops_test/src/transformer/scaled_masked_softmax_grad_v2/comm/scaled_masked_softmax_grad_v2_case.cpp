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
 * \file scaled_masked_softmax_grad_v2_case.cpp
 * \brief ScaledMaskedSoftmaxGradV2 测试用例.
 */
#include "scaled_masked_softmax_grad_v2_case.h"
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
#define SCALED_MASKED_SOFTMAX_GRAD_V2_KERNEL_PARAM                                                                                          \
    (const GM_ADDR yGrad, const GM_ADDR y, const GM_ADDR mask, const GM_ADDR xGrad, GM_ADDR workspace, GM_ADDR tiling)

using ScaledMaskedSoftmaxGradV2KernelFunc = void(*) SCALED_MASKED_SOFTMAX_GRAD_V2_KERNEL_PARAM;

extern "C" __global__ __aicore__ void scaled_masked_softmax_grad_v2 SCALED_MASKED_SOFTMAX_GRAD_V2_KERNEL_PARAM;

using namespace ops::adv::tests::ScaledMaskedSoftmaxGradV2;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunScaledMaskedSoftmaxGradV2(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                                    std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    int32_t inputYGradIdx = 0;
    int32_t inputYIdx = 1;
    int32_t inputMaskIdx = 2;
    int32_t outputXGradIdx = 0;
    // Kernel 运行
    auto kernelFunc = (ScaledMaskedSoftmaxGradV2KernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[inputYGradIdx]->GetDevData(), inputs[inputYIdx]->GetDevData(),
        inputs[inputMaskIdx]->GetDevData(), outputs[outputXGradIdx]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingScaledMaskedSoftmaxGradV2Stub(gert::TilingContext *ctx)
{
    auto *scaledMaskedSoftmaxGradV2Case = static_cast<ScaledMaskedSoftmaxGradV2Case *>(Case::GetCurrentCase());
    if (scaledMaskedSoftmaxGradV2Case != nullptr) {
        ScaledMaskedSoftmaxGradV2Case::DoTilingParam tilingParam;
        tilingParam.ctx = ctx;
        tilingParam.ret = ge::GRAPH_SUCCESS;
        if (!scaledMaskedSoftmaxGradV2Case->DoOptiling(tilingParam)) {
            return tilingParam.ret;
        }
        return scaledMaskedSoftmaxGradV2Case->scaledMaskedSoftmaxGradV2TilingFunc(ctx);
    }
    return ge::GRAPH_FAILED;
}

bool ScaledMaskedSoftmaxGradV2Case::InitParam()
{
    yGrad = Tensor("yGrad", {mParam.b, mParam.n, mParam.s, mParam.d}, "BNSD", mParam.dataType, ge::FORMAT_ND);
    y = Tensor("y", {mParam.b, mParam.n, mParam.s, mParam.d}, "BNSD", mParam.dataType, ge::FORMAT_ND);
    mask = Tensor("mask", {mParam.maskB, mParam.maskN, mParam.s, mParam.d}, "BNSD", mParam.maskDataType, ge::FORMAT_ND);
    xGrad = Tensor("xGrad", {mParam.b, mParam.n, mParam.s, mParam.d}, "BNSD", mParam.dataType, ge::FORMAT_ND);
    return true;
}

bool ScaledMaskedSoftmaxGradV2Case::InitOpInfo()
{
    bool rst = mCtx.SetOpName("ScaledMaskedSoftmaxGradV2");
    rst = rst && mCtx.SetDeterministic(true);
    rst = rst && mCtx.SetInputs({&yGrad, &y, &mask});
    rst = rst && mCtx.SetOutputs({&xGrad});
    rst = rst && mCtx.SetAttrs({{"scale", mParam.scaleValue},
                                {"fixedTriuMask", mParam.fixedTriuMask}});

    rst = rst && mCtx.SetKernelRunCbf(RunScaledMaskedSoftmaxGradV2);
    rst = rst && mCtx.SetKernelMainFunc((void *)scaled_masked_softmax_grad_v2);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    scaledMaskedSoftmaxGradV2TilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("Tiling4ScaledMaskedSoftmaxGradV2");
    if (scaledMaskedSoftmaxGradV2TilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, rope(%p)", scaledMaskedSoftmaxGradV2TilingFunc);
        return false;
    }
    IMPL_OP(ScaledMaskedSoftmaxGradV2).Tiling(TilingScaledMaskedSoftmaxGradV2Stub);

    return rst;
}

bool ScaledMaskedSoftmaxGradV2Case::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool ScaledMaskedSoftmaxGradV2Case::Run()
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

ScaledMaskedSoftmaxGradV2Case::ScaledMaskedSoftmaxGradV2Case(const char *name, bool enable, const char *dbgInfo, OpInfo rope, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(rope)), mParam(std::move(param))
{
    this->mOpInfo.mName = "ScaledMaskedSoftmaxGradV2";
}

ScaledMaskedSoftmaxGradV2Case::ScaledMaskedSoftmaxGradV2Case()
{
    this->mName = "ScaledMaskedSoftmaxGradV2Case";
    this->mOpInfo.mName = "ScaledMaskedSoftmaxGradV2";
}

ScaledMaskedSoftmaxGradV2Case::Param::Param()
{
}

ScaledMaskedSoftmaxGradV2Case::Param::Param(int64_t pb, int64_t pn, int64_t ps, int64_t pd, int64_t pmaskB,
    int64_t pmaskN, float pscaleValue, bool pfixedTriuMask, ge::DataType pDataType, ge::DataType pmaskDataType)
    : b(pb), n(pn), s(ps), d(pd), maskB(pmaskB), maskN(pmaskN), scaleValue(pscaleValue), fixedTriuMask(pfixedTriuMask),
    dataType(pDataType), maskDataType(pmaskDataType)
{
}

bool ScaledMaskedSoftmaxGradV2Case::DoOptiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}