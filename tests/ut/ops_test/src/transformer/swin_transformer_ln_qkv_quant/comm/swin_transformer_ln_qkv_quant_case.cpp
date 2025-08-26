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
 * \file swin_transformer_ln_qkv_quant_case.cpp
 * \brief swin_transformer_ln_qkv_quant 测试用例.
 */
#include "swin_transformer_ln_qkv_quant_case.h"
#include <utility>
#include <tikicpulib.h>
#include "tests/utils/log.h"
#include <register/op_impl_registry.h>
/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define SWIN_TRANSFORMER_LN_QKV_QUANT_KERNEL_PARAM                                                                                               \
    (GM_ADDR x, GM_ADDR gamma, GM_ADDR beta,GM_ADDR weight, GM_ADDR bias, GM_ADDR quant_scale, GM_ADDR quant_offset, \
    GM_ADDR dequant_scale, GM_ADDR query_output, GM_ADDR key_output, GM_ADDR value_output, GM_ADDR workspace, GM_ADDR tiling)

using SwinTransformerLnQkvQuantKernelFunc = void(*) SWIN_TRANSFORMER_LN_QKV_QUANT_KERNEL_PARAM;

extern "C" __global__ __aicore__ void swin_transformer_ln_qkv_quant SWIN_TRANSFORMER_LN_QKV_QUANT_KERNEL_PARAM;

using namespace ops::adv::tests::SwinTransformerLnQkvQuant;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunSwinTransformerLnQkvQuant(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                               std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (SwinTransformerLnQkvQuantKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
    inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(), inputs[7]->GetDevData(),
    outputs[0]->GetDevData(), outputs[1]->GetDevData(), outputs[2]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForSwinTransformerLnQkvQuantStub(gert::TilingContext *context)
{
    auto *swinTransformerLnQkvQuantCase = static_cast<SwinTransformerLnQkvQuantCase *>(Case::GetCurrentCase());
    if (swinTransformerLnQkvQuantCase != nullptr) {  
        SwinTransformerLnQkvQuantCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!swinTransformerLnQkvQuantCase->DoOpTiling(p)) {
            return p.ret;
        }
        return swinTransformerLnQkvQuantCase->swinTransformerLnQkvQuantTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool SwinTransformerLnQkvQuantCase::InitParam()
{
    x = Tensor("x", {mParam.b, mParam.s, mParam.h}, "2", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    gamma = Tensor("gamma", {mParam.h}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    beta = Tensor("beta", {mParam.h}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    weight = Tensor("weight", {mParam.h * 3, mParam.h}, "2", ge::DataType::DT_INT8, ge::FORMAT_ND);
    bias = Tensor("bias", {mParam.h * 3, mParam.h}, "1", ge::DataType::DT_INT32, ge::FORMAT_ND);
    quant_scale = Tensor("quant_scale", {mParam.h}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    quant_offset = Tensor("quant_offset", {mParam.h}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    dequant_scale = Tensor("dequant_scale", {mParam.h * 3}, "1", ge::DataType::DT_UINT64, ge::FORMAT_ND);

    query_output = Tensor("query_output", {mParam.b * (mParam.s / mParam.hWin / mParam.wWin), mParam.headNum, mParam.hWin * mParam.wWin, 
                                            mParam.sizePerhead}, "4", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    key_output = Tensor("key_output", {mParam.b * (mParam.s / mParam.hWin / mParam.wWin), mParam.headNum, mParam.hWin * mParam.wWin, 
                                            mParam.sizePerhead}, "4", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    value_output = Tensor("value_output", {mParam.b * (mParam.s / mParam.hWin / mParam.wWin), mParam.headNum, mParam.hWin * mParam.wWin, 
                                            mParam.sizePerhead}, "4", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    return true;
}

bool SwinTransformerLnQkvQuantCase::InitOpInfo()
{
    auto *moeKernelFunc = (void *)swin_transformer_ln_qkv_quant;
    bool rst = mCtx.SetOpName("SwinTransformerLnQkvQuant");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&x, &gamma, &beta, &weight, &bias, &quant_scale, &quant_offset, &dequant_scale});
    rst = rst && mCtx.SetOutputs({&query_output, &key_output, &value_output});
    rst = rst && mCtx.SetAttrs({{"head_num", mParam.headNum}, {"seq_length", mParam.sizePerhead}, {"epsilon", mParam.epslion}, {"ori_height", mParam.ori_height},
                                {"ori_weight", mParam.ori_weight},
                                {"h_win_size", mParam.hWin}, {"w_win_size", mParam.wWin}, {"weight_transpose", mParam.bTrans},});
    rst = rst && mCtx.SetKernelRunCbf(RunSwinTransformerLnQkvQuant);
    rst = rst && mCtx.SetKernelMainFunc(moeKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    swinTransformerLnQkvQuantTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingFuncForSwinTransformerLnQkvQuant");
    if (swinTransformerLnQkvQuantTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, swinTransformerLnQkvQuant(%p)", swinTransformerLnQkvQuantTilingFunc);
        return false;
    }
    IMPL_OP(SwinTransformerLnQkvQuant).Tiling(TilingForSwinTransformerLnQkvQuantStub);
    return rst;
}

bool SwinTransformerLnQkvQuantCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool SwinTransformerLnQkvQuantCase::Run()
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

SwinTransformerLnQkvQuantCase::SwinTransformerLnQkvQuantCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                                       Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "SwinTransformerLnQkvQuant";
}

SwinTransformerLnQkvQuantCase::SwinTransformerLnQkvQuantCase()
{
}

SwinTransformerLnQkvQuantCase::Param::Param()
{
}
SwinTransformerLnQkvQuantCase::Param::Param(int64_t B, int64_t S, int64_t H, int64_t oriHeight, int64_t oriWeight, 
 int64_t pHeadNum,  int64_t pHWin, int64_t pwWin, int64_t psizePerhead, bool pbTrans, float pepslion)

    : b(B), s(S), h(H), ori_height(oriHeight), ori_weight(oriWeight), headNum(pHeadNum),\
    hWin(pHWin), wWin(pwWin), sizePerhead(psizePerhead), bTrans(pbTrans), epslion(pepslion)
{
}

bool SwinTransformerLnQkvQuantCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}