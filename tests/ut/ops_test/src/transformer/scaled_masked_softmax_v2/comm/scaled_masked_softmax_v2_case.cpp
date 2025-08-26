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
 * \file scaled_masked_softmax_v2_case.cpp
 * \brief
 */

#include "scaled_masked_softmax_v2_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/tiling_base.h"

#define SCALED_MASKED_SOFTMAX_V2_KERNEL_PARAM                                                                               \
    (GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)

using ScaledMaskedSoftmaxV2KernelFunc = void(*) SCALED_MASKED_SOFTMAX_V2_KERNEL_PARAM;

extern "C" __global__ __aicore__ void scaled_masked_softmax_v2 SCALED_MASKED_SOFTMAX_V2_KERNEL_PARAM;

using namespace ops::adv::tests::ScaledMaskedSoftmaxV2;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunScaledMaskedSoftmaxV2(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                         std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (ScaledMaskedSoftmaxV2KernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), outputs[0]->GetDevData(),
                workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForScaledMaskedSoftmaxV2Stub(gert::TilingContext *context)
{
    auto *scaledMaskedSoftmaxV2Case = static_cast<ScaledMaskedSoftmaxV2Case *>(Case::GetCurrentCase());
    if (scaledMaskedSoftmaxV2Case != nullptr) {
        ScaledMaskedSoftmaxV2Case::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!scaledMaskedSoftmaxV2Case->DoOpTiling(p)) {
            return p.ret;
        }
        return scaledMaskedSoftmaxV2Case->scaledMaskedSoftmaxV2TilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool ScaledMaskedSoftmaxV2Case::InitParam()
{
    x = Tensor("x", {mParam.b, mParam.n, mParam.s1, mParam.s2}, "4", mParam.xDtype, ge::FORMAT_ND);
    int64_t maskBatch = mParam.b;
    int64_t maskHeadNum = mParam.n;
    if((mParam.maskType == MaskType::BroadCastB) || (mParam.maskType == MaskType::ReshapeBN)) {
        maskBatch = 1;
    }
    if ((mParam.maskType == MaskType::BroadCastN) || (mParam.maskType == MaskType::ReshapeBN)) {
        maskHeadNum = 1;
    }
    mask = Tensor("mask", {maskBatch, maskHeadNum, mParam.s1, mParam.s2}, "4", ge::DataType::DT_BOOL, ge::FORMAT_ND);
    y = Tensor("y", {mParam.b, mParam.n, mParam.s1, mParam.s2}, "4", mParam.xDtype, ge::FORMAT_ND);
    return true;
}

bool ScaledMaskedSoftmaxV2Case::InitOpInfo()
{
    bool rst = mCtx.SetOpName("ScaledMaskedSoftmaxV2");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&x, &mask});
    rst = rst && mCtx.SetOutputs({&y});
    rst = rst && mCtx.SetAttrs({{"scale", mParam.scale},
                                {"fixed_triu_mask", mParam.genMask}});
    rst = rst && mCtx.SetKernelRunCbf(RunScaledMaskedSoftmaxV2);
    rst = rst && mCtx.SetKernelMainFunc((void *)scaled_masked_softmax_v2);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    scaledMaskedSoftmaxV2TilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingScaledMaskedSoftmaxV2");
    if (scaledMaskedSoftmaxV2TilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, scaledMaskedSoftmaxV2(%p)", scaledMaskedSoftmaxV2TilingFunc);
        return false;
    }
    IMPL_OP(ScaledMaskedSoftmaxV2).Tiling(TilingForScaledMaskedSoftmaxV2Stub);
    return rst;
}

bool ScaledMaskedSoftmaxV2Case::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool ScaledMaskedSoftmaxV2Case::Run()
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

ScaledMaskedSoftmaxV2Case::ScaledMaskedSoftmaxV2Case(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                           Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "ScaledMaskedSoftmaxV2";
}

ScaledMaskedSoftmaxV2Case::ScaledMaskedSoftmaxV2Case()
{
}
ScaledMaskedSoftmaxV2Case::Param::Param()
{
}
ScaledMaskedSoftmaxV2Case::Param::Param(int64_t pb, int64_t pn, int64_t ps1, int64_t ps2, float pscale, bool pgenMask, MaskType pmaskType, ge::DataType pxDtypeIn)

    : b(pb), n(pn), s1(ps1), s2(ps2), scale(pscale), genMask(pgenMask), maskType(pmaskType), xDtype(pxDtypeIn)
{
}


bool ScaledMaskedSoftmaxV2Case::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}