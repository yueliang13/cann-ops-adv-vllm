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
 * \file ring_attention_update_case.cpp
 * \brief
 */

#include "ring_attention_update_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/tiling_base.h"

#define RING_ATTENTION_UPDATE_KERNEL_PARAM \
    (GM_ADDR prev_attn_out, GM_ADDR prev_softmax_max, GM_ADDR prev_softmax_sum, \
    GM_ADDR cur_attn_out, GM_ADDR cur_softmax_max, GM_ADDR cur_softmax_sum, \
    GM_ADDR actual_seq_qlen, \
    GM_ADDR attn_out, GM_ADDR softmax_max, GM_ADDR softmax_sum, \
    GM_ADDR workspace, GM_ADDR tiling)

using RingAttentionUpdateKernelFunc = void(*) RING_ATTENTION_UPDATE_KERNEL_PARAM;

extern "C" __global__ __aicore__ void ring_attention_update RING_ATTENTION_UPDATE_KERNEL_PARAM;

using namespace ops::adv::tests::RingAttentionUpdate;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunRingAttentionUpdate(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
    std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (RingAttentionUpdateKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim,
                inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(),
                inputs[6]->GetDevData(),
                outputs[0]->GetDevData(), outputs[1]->GetDevData(), outputs[2]->GetDevData(),
                workspace, tilingData);
    return true;
}


extern "C" ge::graphStatus TilingForRingAttentionUpdateStub(gert::TilingContext *context)
{
    auto *ringAttentionUpdateCase = static_cast<RingAttentionUpdateCase *>(Case::GetCurrentCase());
    if (ringAttentionUpdateCase != nullptr) {
        RingAttentionUpdateCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!ringAttentionUpdateCase->DoOpTiling(p)) {
            return p.ret;
        }
        return ringAttentionUpdateCase->ringAttentionUpdateTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool RingAttentionUpdateCase::InitParam()
{
    if (mParam.layout == "SBH") {
        prev_attn_out = Tensor("prev_attn_out", {mParam.s, mParam.b, mParam.h}, "3", mParam.attnDataType, ge::FORMAT_ND);
        prev_softmax_max = Tensor("prev_softmax_max", {mParam.b, mParam.n, mParam.s, 8}, "4", mParam.softmaxDataType, ge::FORMAT_ND);
        prev_softmax_sum = Tensor("prev_softmax_sum", {mParam.b, mParam.n, mParam.s, 8}, "4", mParam.softmaxDataType, ge::FORMAT_ND);
        cur_attn_out = Tensor("cur_attn_out", {mParam.s, mParam.b, mParam.h}, "3", mParam.attnDataType, ge::FORMAT_ND);
        cur_softmax_max = Tensor("cur_softmax_max", {mParam.b, mParam.n, mParam.s, 8}, "4", mParam.softmaxDataType, ge::FORMAT_ND);
        cur_softmax_sum = Tensor("cur_softmax_sum", {mParam.b, mParam.n, mParam.s, 8}, "4", mParam.softmaxDataType, ge::FORMAT_ND);
        actual_seq_qlen = Tensor("actual_seq_qlen", {mParam.b + 1}, "1", mParam.seqLenDataType, ge::FORMAT_ND);
    
        attn_out = Tensor("attn_out", {mParam.s, mParam.b, mParam.h}, "3", mParam.attnDataType, ge::FORMAT_ND);
        softmax_max = Tensor("softmax_max", {mParam.b, mParam.n, mParam.s, 8}, "4", mParam.softmaxDataType, ge::FORMAT_ND);
        softmax_sum = Tensor("softmax_sum", {mParam.b, mParam.n, mParam.s, 8}, "4", mParam.softmaxDataType, ge::FORMAT_ND);
    } else {
        prev_attn_out = Tensor("prev_attn_out", {mParam.t, mParam.n, mParam.d}, "3", mParam.attnDataType, ge::FORMAT_ND);
        prev_softmax_max = Tensor("prev_softmax_max", {mParam.t, mParam.n, 8}, "3", mParam.softmaxDataType, ge::FORMAT_ND);
        prev_softmax_sum = Tensor("prev_softmax_sum", {mParam.t, mParam.n, 8}, "3", mParam.softmaxDataType, ge::FORMAT_ND);
        cur_attn_out = Tensor("cur_attn_out", {mParam.t, mParam.n, mParam.d}, "3", mParam.attnDataType, ge::FORMAT_ND);
        cur_softmax_max = Tensor("cur_softmax_max", {mParam.t, mParam.n, 8}, "3", mParam.softmaxDataType, ge::FORMAT_ND);
        cur_softmax_sum = Tensor("cur_softmax_sum", {mParam.t, mParam.n, 8}, "3", mParam.softmaxDataType, ge::FORMAT_ND);
        actual_seq_qlen = Tensor("actual_seq_qlen", {mParam.b + 1}, "1", mParam.seqLenDataType, ge::FORMAT_ND);
    
        attn_out = Tensor("attn_out", {mParam.t, mParam.n, mParam.d}, "3", mParam.attnDataType, ge::FORMAT_ND);
        softmax_max = Tensor("softmax_max", {mParam.t, mParam.n, 8}, "3", mParam.softmaxDataType, ge::FORMAT_ND);
        softmax_sum = Tensor("softmax_sum", {mParam.t, mParam.n, 8}, "3", mParam.softmaxDataType, ge::FORMAT_ND);
    }
    return true;
}

bool RingAttentionUpdateCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("RingAttentionUpdate");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&prev_attn_out, &prev_softmax_max, &prev_softmax_sum,
                                 &cur_attn_out, &cur_softmax_max, &cur_softmax_sum, &actual_seq_qlen});
    rst = rst && mCtx.SetOutputs({&attn_out, &softmax_max, &softmax_sum});
    rst = rst && mCtx.SetAttrs({{"input_layout", mParam.layout}});
    rst = rst && mCtx.SetKernelRunCbf(RunRingAttentionUpdate);
    rst = rst && mCtx.SetKernelMainFunc((void *)ring_attention_update);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    ringAttentionUpdateTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("Tiling4RingAttentionUpdate");
    if (ringAttentionUpdateTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, ringAttentionUpdate(%p)", ringAttentionUpdateTilingFunc);
        return false;
    }
    IMPL_OP(RingAttentionUpdate).Tiling(TilingForRingAttentionUpdateStub);
    return rst;
}

bool RingAttentionUpdateCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool RingAttentionUpdateCase::Run()
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

RingAttentionUpdateCase::RingAttentionUpdateCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                           Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "RingAttentionUpdate";
}

RingAttentionUpdateCase::RingAttentionUpdateCase()
{
}
RingAttentionUpdateCase::Param::Param()
{
}
RingAttentionUpdateCase::Param::Param(int64_t pS, int64_t pB, int64_t pH, int64_t pN, int64_t pT, int64_t pD, std::string pLayout,
                                      ge::DataType attnDataTypeIn, ge::DataType softmaxDataTypeIn, ge::DataType seqLenDataTypeIn)

    : s(pS), b(pB), h(pH), n(pN), t(pT), d(pD), layout(pLayout), attnDataType(attnDataTypeIn),  softmaxDataType(softmaxDataTypeIn), seqLenDataType(seqLenDataTypeIn)
{
}


bool RingAttentionUpdateCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}