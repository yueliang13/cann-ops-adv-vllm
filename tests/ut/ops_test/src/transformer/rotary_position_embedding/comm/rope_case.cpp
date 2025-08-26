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
 * \brief RotaryPositionEmbedding 测试用例.
 */
#include "rope_case.h"
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
#define ROPE_KERNEL_PARAM (GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)

typedef void(*RopeKernelFunc) ROPE_KERNEL_PARAM;

extern "C" __global__ __aicore__ void rotary_position_embedding ROPE_KERNEL_PARAM;

using namespace ops::adv::tests::rope;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunRotaryPositionEmbedding(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                                 std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (RopeKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                outputs[0]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingRotaryPositionEmbeddingStub(gert::TilingContext *ctx)
{
    auto *ropeCase = static_cast<RopeCase *>(Case::GetCurrentCase());
    if (ropeCase !=nullptr) {
        RopeCase::DoTilingParam tilingParam;
        tilingParam.ctx = ctx;
        tilingParam.ret = ge::GRAPH_SUCCESS;
        if (!ropeCase->DoOptiling(tilingParam)) {
            return tilingParam.ret;
        }
        return ropeCase->ropeTilingFunc(ctx);
    }
    return ge::GRAPH_FAILED;
}

bool RopeCase::InitParam()
{
    if (mParam.layout == "BNSD") {
        x = Tensor("x", {mParam.b, mParam.n, mParam.s, mParam.d}, "BNSD", mParam.dataType, ge::FORMAT_ND);
        cos = Tensor("cos", {mParam.triB, mParam.triN, mParam.s, mParam.d}, "BNSD", mParam.dataType, ge::FORMAT_ND);
        sin = Tensor("sin", {mParam.triB, mParam.triN, mParam.s, mParam.d}, "BNSD", mParam.dataType, ge::FORMAT_ND);
        y = Tensor("y", {mParam.b, mParam.n, mParam.s, mParam.d}, "BNSD", mParam.dataType, ge::FORMAT_ND);
    } else if (mParam.layout == "BSND") {
        x = Tensor("x", {mParam.b, mParam.s, mParam.n, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
        cos = Tensor("cos", {mParam.triB, mParam.s, mParam.triN, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
        sin = Tensor("sin", {mParam.triB, mParam.s, mParam.triN, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
        y = Tensor("y", {mParam.b, mParam.s, mParam.n, mParam.d}, "BSND", mParam.dataType, ge::FORMAT_ND);
    } else if (mParam.layout == "SBND") {
        x = Tensor("x", {mParam.s, mParam.b, mParam.n, mParam.d}, "SBND", mParam.dataType, ge::FORMAT_ND);
        cos = Tensor("cos", {mParam.s, mParam.triB, mParam.triN, mParam.d}, "SBND", mParam.dataType, ge::FORMAT_ND);
        sin = Tensor("sin", {mParam.s, mParam.triB, mParam.triN, mParam.d}, "SBND", mParam.dataType, ge::FORMAT_ND);
        y = Tensor("y", {mParam.s, mParam.b, mParam.n, mParam.d}, "SBND", mParam.dataType, ge::FORMAT_ND);
    }
    return true;
}

bool RopeCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("RotaryPositionEmbedding");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&x, &cos, &sin});
    rst = rst && mCtx.SetOutputs({&y});
    rst = rst && mCtx.SetAttrs({{"mode", mParam.mode}});

    rst = rst && mCtx.SetKernelRunCbf(RunRotaryPositionEmbedding);
    rst = rst && mCtx.SetKernelMainFunc((void *)rotary_position_embedding);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    ropeTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingRotaryPositionEmbedding");
    if (ropeTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, rope(%p)", ropeTilingFunc);
        return false;
    }
    IMPL_OP(RotaryPositionEmbedding).Tiling(TilingRotaryPositionEmbeddingStub);

    return rst;
}

bool RopeCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool RopeCase::Run()
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

RopeCase::RopeCase(const char *name, bool enable, const char *dbgInfo, OpInfo rope, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(rope)), mParam(std::move(param))
{
    this->mOpInfo.mName = "RotaryPositionEmbedding";
}

RopeCase::RopeCase()
{
    this->mName = "RopeCase";
    this->mOpInfo.mName = "RotaryPositionEmbedding";
}

RopeCase::Param::Param()
{
}

RopeCase::Param::Param(int64_t pb, int64_t pn, int64_t ps, int64_t pd, int64_t pTriB, int64_t pTriN, 
    int64_t pMode, std::string pLayout, ge::DataType pDataType)
    : b(pb), n(pn), s(ps), d(pd), triB(pTriB), triN(pTriN), mode(pMode), layout(pLayout), dataType(pDataType)
{
}

bool RopeCase::DoOptiling(DoTilingParam &tilingParam) {
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}