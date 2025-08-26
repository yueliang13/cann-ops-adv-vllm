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
 * \file arope_case.cpp
 * \brief ApplyRotaryPosEmb 测试用例.
 */
#include "arope_case.h"
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
#define AROPE_KERNEL_PARAM                                                                                             \
    (GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR q_out, GM_ADDR k_out, GM_ADDR workspace, GM_ADDR tiling)

typedef void(*ARopeKernelFunc) AROPE_KERNEL_PARAM;

extern "C" __global__ __aicore__ void apply_rotary_pos_emb_fp32 AROPE_KERNEL_PARAM;
extern "C" __global__ __aicore__ void apply_rotary_pos_emb_fp16 AROPE_KERNEL_PARAM;
extern "C" __global__ __aicore__ void apply_rotary_pos_emb_bf16 AROPE_KERNEL_PARAM;

using namespace ops::adv::tests::arope;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunApplyRotaryPosEmb(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                          std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (ARopeKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), outputs[0]->GetDevData(), outputs[1]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingApplyRotaryPosEmbStub(gert::TilingContext *ctx)
{
    auto *aropeCase = static_cast<ARopeCase *>(Case::GetCurrentCase());
    if (aropeCase != nullptr) {
        ARopeCase::DoTilingParam tilingParam;
        tilingParam.ctx = ctx;
        tilingParam.ret = ge::GRAPH_SUCCESS;
        if (!aropeCase->DoOptiling(tilingParam)) {
            return tilingParam.ret;
        }
        return aropeCase->aropeTilingFunc(ctx);
    }
    return ge::GRAPH_FAILED;
}

bool ARopeCase::InitParam()
{
    q = Tensor("q", {mParam.b, mParam.s, mParam.qn, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
    k = Tensor("k", {mParam.b, mParam.s, mParam.kn, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
    cos = Tensor("cos", {mParam.b, mParam.s, 1, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
    sin = Tensor("sin", {mParam.b, mParam.s, 1, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
    q_out = Tensor("q_out", {mParam.b, mParam.s, mParam.qn, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
    k_out = Tensor("k_out", {mParam.b, mParam.s, mParam.kn, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);

    return true;
}

bool ARopeCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("ApplyRotaryPosEmb");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&q, &k, &cos, &sin});
    rst = rst && mCtx.SetOutputs({&q_out, &k_out});
    rst = rst && mCtx.SetAttrs({{"layout", 1}});
    rst = rst && mCtx.SetKernelRunCbf(RunApplyRotaryPosEmb);

    if (mParam.dataType == ge::DataType::DT_FLOAT) {
        rst = rst && mCtx.SetKernelMainFunc((void *)apply_rotary_pos_emb_fp32);
    } else if (mParam.dataType == ge::DataType::DT_FLOAT16) {
        rst = rst && mCtx.SetKernelMainFunc((void *)apply_rotary_pos_emb_fp16);
    } else {
        rst = rst && mCtx.SetKernelMainFunc((void *)apply_rotary_pos_emb_bf16);
    }
    
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    aropeTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("Tiling4ApplyRotaryPosEmb");
    if (aropeTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, rope(%p)", aropeTilingFunc);
        return false;
    }
    IMPL_OP(ApplyRotaryPosEmb).Tiling(TilingApplyRotaryPosEmbStub);

    return rst;
}

bool ARopeCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool ARopeCase::Run()
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

ARopeCase::ARopeCase(const char *name, bool enable, const char *dbgInfo, OpInfo arope, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(arope)), mParam(std::move(param))
{
    this->mOpInfo.mName = "ApplyRotaryPosEmb";
}

ARopeCase::ARopeCase()
{
    this->mName = "ARopeCase";
    this->mOpInfo.mName = "ApplyRotaryPosEmb";
}

ARopeCase::Param::Param()
{
}

ARopeCase::Param::Param(int64_t pb, int64_t ps, int64_t pqn, int64_t pkn, int64_t pd, ge::DataType pDataType)
    : b(pb), s(ps), qn(pqn), kn(pkn), d(pd), dataType(pDataType)
{
}

bool ARopeCase::DoOptiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}