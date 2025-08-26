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
 * \file rope_case.cpp
 * \brief RotaryPosEmbInfer 测试用例.
 */
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/tiling_base.h"
#include "rope_infer_case.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */
#define ROPE_KERNEL_PARAM (GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR seqLen, GM_ADDR outQ, GM_ADDR outK, GM_ADDR workspace, GM_ADDR tiling)

typedef void(*RopeKernelFunc) ROPE_KERNEL_PARAM;

extern "C" __global__ __aicore__ void rotary_pos_emb_infer ROPE_KERNEL_PARAM;

using namespace ops::adv::tests::rope;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunRotaryPosEmbInfer(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
    std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (RopeKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim,
        inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(), inputs[3]->GetDevData(),
        inputs[4]->GetDevData(), outputs[0]->GetDevData(), outputs[1]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingRotaryPosEmbInferStub(gert::TilingContext *ctx)
{
    auto *ropeInferCase = static_cast<RopeInferCase *>(Case::GetCurrentCase());
    if (ropeInferCase !=nullptr) {
        RopeInferCase::DoTilingParam tilingParam;
        tilingParam.ctx = ctx;
        tilingParam.ret = ge::GRAPH_SUCCESS;
        if (!ropeInferCase->DoOptiling(tilingParam)) {
            return tilingParam.ret;
        }
        return ropeInferCase->ropeTilingFunc(ctx);
    }
    return ge::GRAPH_FAILED;
}

bool RopeInferCase::InitParam()
{
    q = Tensor("q", {mParam.batch, mParam.hiddensizeQ}, "ND", mParam.dataType, ge::FORMAT_ND);
    k = Tensor("k", {mParam.batch, mParam.hiddensizeK}, "ND", mParam.dataType, ge::FORMAT_ND);
    seqlen = Tensor("seqlen", {mParam.batch, 1}, "ND", ge::DataType::DT_INT32, ge::FORMAT_ND);
    q_out = Tensor("q_out", {mParam.batch, mParam.hiddensizeQ}, "ND", mParam.dataType, ge::FORMAT_ND);
    k_out = Tensor("k_out", {mParam.batch, mParam.hiddensizeK}, "ND", mParam.dataType, ge::FORMAT_ND);
    
    if(mParam.largeDim){
        cos = Tensor("cos", {mParam.batch, mParam.hiddensizeQ / mParam.headDim, mParam.headDim}, "ND", mParam.dataTypeCos, ge::FORMAT_ND);
        sin = Tensor("sin", {mParam.batch, mParam.hiddensizeQ / mParam.headDim, mParam.headDim}, "ND", mParam.dataTypeCos, ge::FORMAT_ND);
    } else{
        cos = Tensor("cos", {mParam.batch, mParam.headDim}, "ND", mParam.dataTypeCos, ge::FORMAT_ND);
        sin = Tensor("sin", {mParam.batch, mParam.headDim}, "ND", mParam.dataTypeCos, ge::FORMAT_ND);
    }

    return true;
}

bool RopeInferCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("RotaryPosEmbInfer");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&q, &k, &cos, &sin, &seqlen});
    rst = rst && mCtx.SetOutputs({&q_out, &k_out});
    rst = rst && mCtx.SetAttrs({{"rotaryCoeff", mParam.rotaryCoefficiency}, {"cosFormat", 0}});

    rst = rst && mCtx.SetKernelRunCbf(RunRotaryPosEmbInfer);
    rst = rst && mCtx.SetKernelMainFunc((void *)rotary_pos_emb_infer);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    ropeTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("Tiling4RotaryPosEmbInfer");
    if (ropeTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, rope(%p)", ropeTilingFunc);
        return false;
    }
    IMPL_OP(RotaryPosEmbInfer).Tiling(TilingRotaryPosEmbInferStub);

    return rst;
}

bool RopeInferCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool RopeInferCase::Run()
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

RopeInferCase::RopeInferCase(const char *name, bool enable, const char *dbgInfo, OpInfo rope, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(rope)), mParam(std::move(param))
{
    this->mOpInfo.mName = "RotaryPosEmbInfer";
}

RopeInferCase::RopeInferCase()
{
    this->mName = "RopeInferCase";
    this->mOpInfo.mName = "RotaryPosEmbInfer";
}

RopeInferCase::Param::Param()
{
}

RopeInferCase::Param::Param(int64_t pBatch, int64_t pHiddensizeQ, int64_t pHiddensizeK, int64_t pHeadDim, bool pLargeDim, int64_t pRotaryCoefficiency,
    std::string pLayout, ge::DataType pDataType, ge::DataType pDataTypeCos)
    : batch(pBatch), hiddensizeQ(pHiddensizeQ), hiddensizeK(pHiddensizeK),headDim(pHeadDim), largeDim(pLargeDim), rotaryCoefficiency(pRotaryCoefficiency),
    layout(pLayout), dataType(pDataType), dataTypeCos(pDataTypeCos)
{
}

bool RopeInferCase::DoOptiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}