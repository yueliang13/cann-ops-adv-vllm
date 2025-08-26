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
 * \file rope_quant_kvcache_case.cpp
 * \brief RopeQuantKvcache 测试用例.
 */
#include "rope_quant_kvcache_case.h"
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
#define ROPE_QUANT_KVCACHE_KERNEL_PARAM (GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin, GM_ADDR quant_scale, GM_ADDR quant_offset, GM_ADDR k_cache, \
                                         GM_ADDR v_cache, GM_ADDR indice, \
                                         GM_ADDR q_out, GM_ADDR k_out, GM_ADDR v_out, GM_ADDR k_cache_out, GM_ADDR v_cache_out, \
                                         GM_ADDR workspace, GM_ADDR tiling)

typedef void(*RopeQuantKvcacheKernelFunc) ROPE_QUANT_KVCACHE_KERNEL_PARAM;

extern "C" __global__ __aicore__ void rope_quant_kvcache ROPE_QUANT_KVCACHE_KERNEL_PARAM;

using namespace ops::adv::tests::quant_kvcache;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunRopeQuantKvcache(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                         std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (RopeQuantKvcacheKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(), inputs[3]->GetDevData(),
                inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(), inputs[7]->GetDevData(),
                outputs[0]->GetDevData(), outputs[1]->GetDevData(), outputs[2]->GetDevData(), outputs[3]->GetDevData(), 
                outputs[4]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingRopeQuantKvcacheStub(gert::TilingContext *ctx)
{
    auto *ropeQuantKvcacheCase = static_cast<RopeQuantKvcacheCase *>(Case::GetCurrentCase());
    if (ropeQuantKvcacheCase !=nullptr) {
        RopeQuantKvcacheCase::DoTilingParam tilingParam;
        tilingParam.ctx = ctx;
        tilingParam.ret = ge::GRAPH_SUCCESS;
        if (!ropeQuantKvcacheCase->DoOptiling(tilingParam)) {
            return tilingParam.ret;
        }
        return ropeQuantKvcacheCase->ropeQuantKvcacheTilingFunc(ctx);
    }
    return ge::GRAPH_FAILED;
}

bool RopeQuantKvcacheCase::InitParam()
{
    qkv = Tensor("qkv", {4, 1, 1280}, "ND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cos = Tensor("cos", {4, 1, 1, 128}, "ND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    sin = Tensor("sin", {4, 1, 1, 128}, "ND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    quant_scale = Tensor("quant_scale", {128}, "ND", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    quant_offset = Tensor("quant_offset", {128}, "ND", ge::DataType::DT_INT32, ge::FORMAT_ND);
    k_cache = Tensor("k_cache", {4, 2048, 1, 128}, "ND", ge::DataType::DT_INT8, ge::FORMAT_ND);
    v_cache = Tensor("v_cache", {4, 2048, 1, 128}, "ND", ge::DataType::DT_INT8, ge::FORMAT_ND);
    indice = Tensor("indice", {4, 1}, "ND", ge::DataType::DT_INT32, ge::FORMAT_ND);
    q_out = Tensor("q_out", {4, 1, 8, 128}, "ND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    k_out = Tensor("k_out", {4, 1, 1, 128}, "ND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    v_out = Tensor("v_out", {4, 1, 1, 128}, "ND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    k_cache_out = Tensor("k_cache_out", {4, 2048, 1, 128}, "ND", ge::DataType::DT_INT8, ge::FORMAT_ND);
    v_cache_out = Tensor("v_cache_out", {4, 2048, 1, 128}, "ND", ge::DataType::DT_INT8, ge::FORMAT_ND);
    return true;
}

bool RopeQuantKvcacheCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("RopeQuantKvcache");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&qkv, &cos, &sin, &quant_scale, &quant_offset, &k_cache, &v_cache, &indice});
    rst = rst && mCtx.SetOutputs({&q_out, &k_out, &v_out, &k_cache_out, &v_cache_out});
    rst = rst && mCtx.SetAttrs({{"size_splits", mParam.sizeSplits}, {"layout", mParam.layout}, {"kv_output", mParam.kvOutput}});

    rst = rst && mCtx.SetKernelRunCbf(RunRopeQuantKvcache);
    rst = rst && mCtx.SetKernelMainFunc((void *)rope_quant_kvcache);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    ropeQuantKvcacheTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingForRopeQuantKvcache");
    if (ropeQuantKvcacheTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, quant_kvcache(%p)", ropeQuantKvcacheTilingFunc);
        return false;
    }
    IMPL_OP(RopeQuantKvcache).Tiling(TilingRopeQuantKvcacheStub);

    return rst;
}

bool RopeQuantKvcacheCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool RopeQuantKvcacheCase::Run()
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

RopeQuantKvcacheCase::RopeQuantKvcacheCase(const char *name, bool enable, const char *dbgInfo, OpInfo quant_kvcache, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(quant_kvcache)), mParam(std::move(param))
{
    this->mOpInfo.mName = "RopeQuantKvcache";
}

RopeQuantKvcacheCase::RopeQuantKvcacheCase()
{
    this->mName = "RopeQuantKvcacheCase";
    this->mOpInfo.mName = "RopeQuantKvcache";
}

RopeQuantKvcacheCase::Param::Param()
{
}

bool RopeQuantKvcacheCase::DoOptiling(DoTilingParam &tilingParam) {
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}