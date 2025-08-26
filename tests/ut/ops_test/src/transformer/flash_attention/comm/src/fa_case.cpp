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
 * \file fa_case.cpp
 * \brief FlashAttentionScore / FlashAttentionScoreGrad 测试用例.
 */

#include "fa_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/fa/tiling_data.h"
#include "tiling/tiling_templates_registry.h"

using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;
using FaCase = ops::adv::tests::fa::FaCase;

namespace optiling {
ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionScore(gert::TilingContext *context);
ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionGradScore(gert::TilingContext *context);
} // namespace optiling

namespace {

const size_t FAS_PREFIX_INPUT_INDEX = 7UL;
const size_t FAS_ACTUAL_SEQ_LENGTH_INPUT_INDEX = 8UL;
const size_t FAS_ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX = 9UL;
const size_t FAS_Q_START_IDX_INPUT_INDEX = 10UL;
const size_t FAS_KV_START_IDX_INPUT_INDEX = 11UL;
const size_t FAG_PREFIX_INPUT_INDEX = 12UL;
const size_t FAG_ACTUAL_SEQ_LENGTH_INPUT_INDEX = 13UL;
const size_t FAG_ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX = 14UL;

ASCENDC_EXTERN_C ge::graphStatus FlashAttentionScoreTilingFuncStub(gert::TilingContext *context)
{
    auto *faCase = static_cast<FaCase *>(Case::GetCurrentCase());
    if (faCase != nullptr) {
        FaCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        p.prefixTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAS_PREFIX_INPUT_INDEX));
        p.actSeqQLenTensor =
            const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAS_ACTUAL_SEQ_LENGTH_INPUT_INDEX));
        p.actSeqKVLenTensor =
            const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAS_ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX));
        p.qStartIdxTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAS_Q_START_IDX_INPUT_INDEX));
        p.kvStartIdxTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAS_KV_START_IDX_INPUT_INDEX));

        if (faCase->DoOpTiling(p)) {
            return p.ret;
        }
        return faCase->mFasOriginTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

ASCENDC_EXTERN_C ge::graphStatus FlashAttentionScoreGradTilingFuncStub(gert::TilingContext *context)
{
    auto *faCase = static_cast<FaCase *>(Case::GetCurrentCase());
    if (faCase != nullptr) {
        FaCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        p.prefixTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAG_PREFIX_INPUT_INDEX));
        p.actSeqQLenTensor =
            const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAG_ACTUAL_SEQ_LENGTH_INPUT_INDEX));
        p.actSeqKVLenTensor =
            const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAG_ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX));
        if (faCase->DoOpTiling(p)) {
            return p.ret;
        }
        return faCase->mFagOriginTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

} // namespace

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */
#define FAS_KERNEL_PARAM                                                                                               \
    (__gm__ uint8_t * query, __gm__ uint8_t * key, __gm__ uint8_t * value, __gm__ uint8_t * pse,                       \
     __gm__ uint8_t * dropMask, __gm__ uint8_t * paddingMask, __gm__ uint8_t * attenMask, __gm__ uint8_t * prefix,     \
     __gm__ uint8_t * actualSeqLengths, __gm__ uint8_t * actualSeqLengthsKv, __gm__ uint8_t * qStartIdx,               \
     __gm__ uint8_t * kvStartIdx, __gm__ uint8_t * softmaxMax, __gm__ uint8_t * softmaxSum,                            \
     __gm__ uint8_t * softmaxOut, __gm__ uint8_t * attentionOut, __gm__ uint8_t * workspace, __gm__ uint8_t * tiling)

#define FAG_KERNEL_PARAM                                                                                               \
    (__gm__ uint8_t * query, __gm__ uint8_t * key, __gm__ uint8_t * value, __gm__ uint8_t * dy,                        \
     __gm__ uint8_t * pse_shift, __gm__ uint8_t * drop_mask, __gm__ uint8_t * padding_mask,                            \
     __gm__ uint8_t * atten_mask, __gm__ uint8_t * softmax_max, __gm__ uint8_t * softmax_sum,                          \
     __gm__ uint8_t * softmax_in, __gm__ uint8_t * attention_in, __gm__ uint8_t * prefix,                              \
     __gm__ uint8_t * actual_seq_qlen, __gm__ uint8_t * actual_seq_kvlen, __gm__ uint8_t * q_start_idx,                \
     __gm__ uint8_t * kv_start_idx, __gm__ uint8_t * dq, __gm__ uint8_t * dk, __gm__ uint8_t * dv,                     \
     __gm__ uint8_t * dpse, __gm__ uint8_t * workspace, __gm__ uint8_t * tiling_data)

typedef void(*FasKernelFunc) FAS_KERNEL_PARAM;
typedef void(*FagKernelFunc) FAG_KERNEL_PARAM;

extern "C" __global__ __aicore__ void flash_attention_score_fp16 FAS_KERNEL_PARAM;
extern "C" __global__ __aicore__ void flash_attention_score_fp32 FAS_KERNEL_PARAM;
extern "C" __global__ __aicore__ void flash_attention_score_bf16 FAS_KERNEL_PARAM;
extern "C" __global__ __aicore__ void flash_attention_score_grad_fp16 FAG_KERNEL_PARAM;
extern "C" __global__ __aicore__ void flash_attention_score_grad_fp32 FAG_KERNEL_PARAM;
extern "C" __global__ __aicore__ void flash_attention_score_grad_bf16 FAG_KERNEL_PARAM;

using namespace ops::adv::tests::fa;
using TensorIntf = ops::adv::tests::utils::TensorIntf;

bool RunFlashAttention(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                       std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (FasKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim,
                inputs[0]->GetDevData(),  // query
                inputs[1]->GetDevData(),  // key
                inputs[2]->GetDevData(),  // value
                inputs[3]->GetDevData(),  // pse
                inputs[4]->GetDevData(),  // dropMask
                inputs[5]->GetDevData(),  // paddingMask
                inputs[6]->GetDevData(),  // attenMask
                inputs[7]->GetDevData(),  // prefix
                inputs[8]->GetDevData(),  // actSeqQLens
                inputs[9]->GetDevData(),  // actSeqKVLens
                inputs[10]->GetDevData(), // qStartIdx
                inputs[11]->GetDevData(), // kvStartIdx
                outputs[0]->GetDevData(), // softmaxMax
                outputs[1]->GetDevData(), // softmaxSum
                outputs[2]->GetDevData(), // softmaxRes
                outputs[3]->GetDevData(), // attenRes
                workspace, tilingData);
    return true;
}

bool RunFlashAttentionGrad(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                           std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (FagKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim,
                inputs[0]->GetDevData(),  // query
                inputs[1]->GetDevData(),  // key
                inputs[2]->GetDevData(),  // value
                inputs[3]->GetDevData(),  // dy
                inputs[4]->GetDevData(),  // pse
                inputs[5]->GetDevData(),  // dropMask
                inputs[6]->GetDevData(),  // paddingMask
                inputs[7]->GetDevData(),  // attenMask
                inputs[8]->GetDevData(),  // softmaxMax
                inputs[9]->GetDevData(),  // softmaxSum
                inputs[10]->GetDevData(), // softMaxRes
                inputs[11]->GetDevData(), // attenRes
                inputs[12]->GetDevData(), // prefix
                inputs[13]->GetDevData(), // actualSeqQLen
                inputs[14]->GetDevData(), // actualSeqKvLen
                inputs[15]->GetDevData(), // qStartIdx
                inputs[16]->GetDevData(), // kvStartIdx
                outputs[0]->GetDevData(), // dq
                outputs[1]->GetDevData(), // dk
                outputs[2]->GetDevData(), // dv
                outputs[3]->GetDevData(), // dpse
                workspace, tilingData);
    return true;
}

FaCase::FaCase() : FaCase("Undefined", true, "", OpInfo(), OpInfo(), FaParam(), kTilingTemplatePriority_Invalid)
{
}

FaCase::FaCase(const char *name, bool enable, const char *dbgInfo, OpInfo forward, OpInfo reverse, FaParam param,
               int32_t tilingTemplatePriority)
    : Case(name, enable, dbgInfo, tilingTemplatePriority), mForward(std::move(forward)), mReverse(std::move(reverse)),
      mParam(std::move(param))
{
    mForward.mName = "FlashAttentionScore";
    mReverse.mName = "FlashAttentionScoreGrad";
}

bool FaCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mForward.ProcessTiling(mName)) {
        return false;
    }
    if (!mForward.ProcessKernel(mName)) {
        return false;
    }
    if (!mReverse.ProcessTiling(mName)) {
        return false;
    }
    if (!mReverse.ProcessKernel(mName)) {
        return false;
    }
    return true;
}

bool FaCase::InitParam()
{
    if (!mParam.Init()) {
        return false;
    }
    return true;
}

bool FaCase::InitOpInfo()
{
    auto *fasKernelFunc = (void *)flash_attention_score_bf16;
    auto *fagKernelFunc = (void *)flash_attention_score_grad_bf16;
    if (mParam.dtype == ge::DataType::DT_FLOAT16) {
        fasKernelFunc = (void *)flash_attention_score_fp16;
        fagKernelFunc = (void *)flash_attention_score_grad_fp16;
    } else if (mParam.dtype == ge::DataType::DT_FLOAT) {
        fasKernelFunc = (void *)flash_attention_score_fp32;
        fagKernelFunc = (void *)flash_attention_score_grad_fp32;
    }

    bool rst = mForwardCtx.SetOpName(mForward.mName.c_str());
    rst = rst && mForwardCtx.SetDeterministic(mForward.mCtr.mDeterministic);
    rst = rst && mForwardCtx.SetInputs({&mParam.query, &mParam.key, &mParam.value, &mParam.pse, &mParam.dropMask,
                                        &mParam.paddingMask, &mParam.attenMask, &mParam.prefix, &mParam.actualSeqQLen,
                                        &mParam.actualSeqKvLen, &mParam.qStartIdx, &mParam.kvStartIdx});
    rst = rst && mForwardCtx.SetOutputs({&mParam.softmaxMax, &mParam.softmaxSum, &mParam.softmaxRes, &mParam.attenRes});
    rst = rst && mForwardCtx.SetAttrs({{"scale_value", mParam.scale},
                                       {"keep_prob", mParam.keepProb},
                                       {"pre_tockens", mParam.preTokens},
                                       {"next_tockens", mParam.nxtTokens},
                                       {"head_num", mParam.n1},
                                       {"input_layout", mParam.layout},
                                       {"inner_precise", mParam.innerPrecise},
                                       {"sparse_mode", mParam.sparseMode},
                                       {"pse_type", mParam.pseType}});
    rst = rst && mForwardCtx.SetTilingDataMaxSize(2456);    /* 2456 FlashAttentionScore 最大 TilingData 大小 */
    rst = rst && mForwardCtx.SetKernelRunCbf(RunFlashAttention);
    rst = rst && mForwardCtx.SetKernelMainFunc((void *)fasKernelFunc);
    rst = rst && mForward.SetContext(&mForwardCtx);
    rst = rst && mReverseCtx.SetOpName(mReverse.mName.c_str());
    rst = rst && mReverseCtx.SetDeterministic(mReverse.mCtr.mDeterministic);
    rst = rst &&
          mReverseCtx.SetInputs({&mParam.query, &mParam.key, &mParam.value, &mParam.dy, &mParam.pse, &mParam.dropMask,
                                 &mParam.paddingMask, &mParam.attenMask, &mParam.softmaxMax, &mParam.softmaxSum,
                                 &mParam.softmaxRes, &mParam.attenRes, &mParam.prefix, &mParam.actualSeqQLen,
                                 &mParam.actualSeqKvLen, &mParam.qStartIdx, &mParam.kvStartIdx});
    rst = rst && mReverseCtx.SetOutputs({&mParam.dq, &mParam.dk, &mParam.dv, &mParam.dPse});
    rst = rst && mReverseCtx.SetAttrs({{"scale_value", mParam.scale},
                                       {"keep_prob", mParam.keepProb},
                                       {"pre_tockens", mParam.preTokens},
                                       {"next_tockens", mParam.nxtTokens},
                                       {"head_num", mParam.n1},
                                       {"input_layout", mParam.layout},
                                       {"inner_precise", mParam.innerPrecise},
                                       {"sparse_mode", mParam.sparseMode},
                                       {"pse_type", mParam.pseType}});
    rst = rst && mReverseCtx.SetTilingDataMaxSize(2560);    /* 2560 FlashAttentionScoreGrad 最大 TilingData 大小 */
    rst = rst && mReverseCtx.SetKernelRunCbf(RunFlashAttentionGrad);
    rst = rst && mReverseCtx.SetKernelMainFunc((void *)fagKernelFunc);
    rst = rst && mReverse.SetContext(&mReverseCtx);
    if (!rst) {
        return rst;
    }

    if (!this->InitOriginTilingFunc()) {
        return false;
    }
    IMPL_OP(FlashAttentionScore).Tiling(FlashAttentionScoreTilingFuncStub);
    IMPL_OP(FlashAttentionScoreGrad).Tiling(FlashAttentionScoreGradTilingFuncStub);

    return true;
}

bool FaCase::InitOriginTilingFunc()
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    /* FlashAttentionScore FlashAttentionScoreGrad 需提供修改 TilingContext 功能 */
    /* FlashAttentionScoreGrad 需提供按指定优先级调用 Tiling 模板功能 */
    mFasOriginTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingFlashAttentionScore");
    mFagOriginTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingFlashAttentionGradScore");
    if (mFasOriginTilingFunc == nullptr || mFagOriginTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, Fas(%p), Fag(%p)", mFasOriginTilingFunc, mFagOriginTilingFunc);
        return false;
    }
    return true;
}

bool FaCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool FaCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    if (mPreTilingRunCbf != nullptr) {
        mPreTilingRunCbf(tilingParam);
    }
    /* 外部无法构造 Tensor 的数据, 此处进行处理 */
    if (tilingParam.prefixTensor != nullptr) {
        tilingParam.prefixTensor->SetData(gert::TensorData{mParam.prefixTensorData.data()});
    }
    if (tilingParam.actSeqQLenTensor != nullptr) {
        tilingParam.actSeqQLenTensor->SetData(gert::TensorData{mParam.actualSeqQLenTensorData.data()});
    }
    if (tilingParam.actSeqKVLenTensor != nullptr) {
        tilingParam.actSeqKVLenTensor->SetData(gert::TensorData{mParam.actualSeqKVLenTensorData.data()});
    }
    if (tilingParam.qStartIdxTensor != nullptr) {
        tilingParam.qStartIdxTensor->SetData(gert::TensorData{mParam.qStartIdxTensorData.data()});
    }
    if (tilingParam.kvStartIdxTensor != nullptr) {
        tilingParam.kvStartIdxTensor->SetData(gert::TensorData{mParam.kvStartIdxTensorData.data()});
    }
    /* 按优先级 Tiling */
    auto priority = mTilingTemplatePriority;
    if (priority == Case::kTilingTemplatePriority_Invalid) {
        return false;
    }
    tilingParam.ret = optiling::TilingRegistry::GetInstance().DoTilingImpl(tilingParam.ctx, {priority});
    return true;
}

void FaCase::PreTilingRunCbf_SetPlatformInfoNull(FaCase::DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return;
    }
    const auto compute_node_info = tilingParam.ctx->GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
        return;
    }
    /* PlatformInfo 位于 Inputs 和 Outputs 之后 */
    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum() + 1U;
    auto kernelContext = (gert::KernelContext *)tilingParam.ctx;
    kernelContext->GetContext()->values[index] = nullptr;
}
