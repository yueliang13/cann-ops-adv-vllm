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
 * \file swin_attention_score_quant_case.cpp
 * \brief swin_attention_score_quant 测试用例.
 */
#include "swin_attention_score_quant_case.h"
#include <utility>
#include <tikicpulib.h>
#include "tests/utils/log.h"
#include <register/op_impl_registry.h>
/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define SWIN_ATTENTION_SCORE_QUANT_KERNEL_PARAM                                                                                               \
    (GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR scale_quant, GM_ADDR scale_dequant1, GM_ADDR scale_dequant2, GM_ADDR bias_quant, \
    GM_ADDR bias_dequant1, GM_ADDR bias_dequant2, GM_ADDR padding_mask1, GM_ADDR padding_mask2, GM_ADDR attention_score, GM_ADDR workspace, GM_ADDR tiling)

using SwinAttentionScoreQuantKernelFunc = void(*) SWIN_ATTENTION_SCORE_QUANT_KERNEL_PARAM;

extern "C" __global__ __aicore__ void swin_attention_score_quant SWIN_ATTENTION_SCORE_QUANT_KERNEL_PARAM;

using namespace ops::adv::tests::SwinAttentionScoreQuant;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunSwinAttentionScoreQuant(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                               std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (SwinAttentionScoreQuantKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
    inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(), inputs[7]->GetDevData(),
    inputs[8]->GetDevData(), inputs[9]->GetDevData(), inputs[10]->GetDevData(), outputs[0]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForSwinAttentionScoreQuantStub(gert::TilingContext *context)
{
    auto *swinAttentionScoreQuantCase = static_cast<SwinAttentionScoreQuantCase *>(Case::GetCurrentCase());
    if (swinAttentionScoreQuantCase != nullptr) {  
        SwinAttentionScoreQuantCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!swinAttentionScoreQuantCase->DoOpTiling(p)) {
            return p.ret;
        }
        return swinAttentionScoreQuantCase->swinAttentionScoreQuantTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool SwinAttentionScoreQuantCase::InitParam()
{
    query = Tensor("query", {mParam.b, mParam.n, mParam.s, mParam.h}, "4", ge::DataType::DT_INT8, ge::FORMAT_ND);
    key = Tensor("key", {mParam.b, mParam.n, mParam.s, mParam.h}, "4", ge::DataType::DT_INT8, ge::FORMAT_ND);
    value = Tensor("value", {mParam.b, mParam.n, mParam.s, mParam.h}, "4", ge::DataType::DT_INT8, ge::FORMAT_ND);
    scale_quant = Tensor("scale_quant", {1, mParam.s}, "2", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    scale_dequant1 = Tensor("scale_dequant1", {1, mParam.s}, "2", ge::DataType::DT_UINT64, ge::FORMAT_ND);
    scale_dequant2 = Tensor("scale_dequant2", {1, mParam.h}, "2", ge::DataType::DT_UINT64, ge::FORMAT_ND);
    bias_quant = Tensor("bias_quant", {1, mParam.s}, "2", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    bias_dequant1 = Tensor("bias_dequant1", {1, mParam.s}, "2", ge::DataType::DT_INT32, ge::FORMAT_ND);
    bias_dequant2 = Tensor("bias_dequant2", {1, mParam.h}, "2", ge::DataType::DT_INT32, ge::FORMAT_ND);

    padding_mask1 = Tensor("padding_mask1", {1, mParam.n, mParam.s, mParam.s}, "4", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);

    attention_score = Tensor("attention_score", {mParam.b, mParam.n, mParam.s, mParam.h}, "4", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    return true;
}

bool SwinAttentionScoreQuantCase::InitOpInfo()
{
    auto *moeKernelFunc = (void *)swin_attention_score_quant;
    bool rst = mCtx.SetOpName("SwinAttentionScoreQuant");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&query, &key, &value, &scale_quant, &scale_dequant1, &scale_dequant2, &bias_quant, &bias_dequant1, &bias_dequant2,
                                &padding_mask1, &padding_mask2});
    rst = rst && mCtx.SetOutputs({&attention_score});
    rst = rst && mCtx.SetAttrs({{"query_transpose", mParam.qTrans}, 
                                {"key_transpose", mParam.kTrans}, 
                                {"value_transpose", mParam.vTrans},
                                {"softmax_axes", mParam.softmaxAxes}});
    rst = rst && mCtx.SetKernelRunCbf(RunSwinAttentionScoreQuant);
    rst = rst && mCtx.SetKernelMainFunc(moeKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    swinAttentionScoreQuantTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingFunc");
    if (swinAttentionScoreQuantTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, swinAttentionScoreQuant(%p)", swinAttentionScoreQuantTilingFunc);
        return false;
    }
    IMPL_OP(SwinAttentionScoreQuant).Tiling(TilingForSwinAttentionScoreQuantStub);
    return rst;
}

bool SwinAttentionScoreQuantCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool SwinAttentionScoreQuantCase::Run()
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

SwinAttentionScoreQuantCase::SwinAttentionScoreQuantCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                                       Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "SwinAttentionScoreQuant";
}

SwinAttentionScoreQuantCase::SwinAttentionScoreQuantCase()
{
}

SwinAttentionScoreQuantCase::Param::Param()
{
}
SwinAttentionScoreQuantCase::Param::Param(int64_t pB, int64_t pN, int64_t pS, int64_t pH,
                                        bool qTranspose, bool kTranspose, bool vTranspose, int pSoftmaxAxes)

    : b(pB), n(pN), s(pS), h(pH), qTrans(qTranspose), kTrans(kTranspose), vTrans(vTranspose), softmaxAxes(pSoftmaxAxes)
{
}

bool SwinAttentionScoreQuantCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}