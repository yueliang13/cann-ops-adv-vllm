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
 * \file dequant_rope_quant_kvcache_case.cpp
 * \brief DequantRopeQuantKvcache 测试用例.
 */
#include "dequant_rope_quant_kvcache_case.h"
#include <tikicpulib.h>
#include "tests/utils/log.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define DEQUANT_ROPE_QUANT_KVCACHE_KERNEL_PARAM                                                                                               \
    (GM_ADDR x, GM_ADDR cosIn, GM_ADDR sinIn, GM_ADDR kcache, GM_ADDR vcache, GM_ADDR indices, GM_ADDR kscale, \
    GM_ADDR vscale, GM_ADDR koffset, GM_ADDR voffset, GM_ADDR weight, GM_ADDR activation, GM_ADDR bias, GM_ADDR qOut, \
    GM_ADDR kOut, GM_ADDR vOut, GM_ADDR k_cache_ref, GM_ADDR v_cache_ref, GM_ADDR workspaceSize, GM_ADDR tiling)

using DequantRopeQuantKvcacheKernelFunc = void(*) DEQUANT_ROPE_QUANT_KVCACHE_KERNEL_PARAM;

extern "C" __global__ __aicore__ void dequant_rope_quant_kvcache DEQUANT_ROPE_QUANT_KVCACHE_KERNEL_PARAM;

using namespace ops::adv::tests::DequantRopeQuantKvcache;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunDequantRopeQuantKvcache(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                               std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (DequantRopeQuantKvcacheKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(),
                inputs[7]->GetDevData(), inputs[8]->GetDevData(), inputs[9]->GetDevData(), inputs[10]->GetDevData(),
                inputs[11]->GetDevData(), inputs[12]->GetDevData(), outputs[0]->GetDevData(), outputs[1]->GetDevData(),
                outputs[2]->GetDevData(), outputs[3]->GetDevData(), outputs[4]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForDequantRopeQuantKvcacheStub(gert::TilingContext *context)
{
    auto *dequantRopeQuantKvcacheCase = static_cast<DequantRopeQuantKvcacheCase *>(Case::GetCurrentCase());
    if (dequantRopeQuantKvcacheCase != nullptr) {
        DequantRopeQuantKvcacheCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!dequantRopeQuantKvcacheCase->DoOpTiling(p)) {
            return p.ret;
        }
        return dequantRopeQuantKvcacheCase->dequantRopeQuantKvcacheTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool DequantRopeQuantKvcacheCase::InitParam()
{
    x = Tensor("x", {mParam.B, mParam.S, mParam.H}, "3", mParam.xDtype, ge::FORMAT_ND);
    cosIn = Tensor("cosIn", {mParam.B, mParam.S, 1, mParam.D}, "4", mParam.sinDtype, ge::FORMAT_ND);
    sinIn = Tensor("sinIn", {mParam.B, mParam.S, 1, mParam.D}, "4", mParam.sinDtype, ge::FORMAT_ND);
    kcache = Tensor("kcache", {mParam.C1, mParam.C2, mParam.Nkv, mParam.D}, "4", ge::DataType::DT_INT8, ge::FORMAT_ND);
    vcache = Tensor("vcache", {mParam.C1, mParam.C2, mParam.Nkv, mParam.D}, "4", ge::DataType::DT_INT8, ge::FORMAT_ND);
    indices = Tensor("indices", {mParam.B}, "1", ge::DataType::DT_INT32, ge::FORMAT_ND);
    kscale = Tensor("kscale", {mParam.Nkv, mParam.D}, "2", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    vscale = Tensor("vscale", {mParam.Nkv, mParam.D}, "2", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    koffset = Tensor("koffset", {mParam.Nkv, mParam.D}, "2", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    voffset = Tensor("voffset", {mParam.Nkv, mParam.D}, "2", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    weight = Tensor("weight", {mParam.H}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    activation = Tensor("activation", {mParam.B*mParam.S}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    bias = Tensor("bias", {mParam.H}, "1", mParam.biasDtype, ge::FORMAT_ND);

    qOut = Tensor("qOut", {mParam.B, mParam.S, mParam.Nq, mParam.D}, "3", mParam.sinDtype, ge::FORMAT_ND);
    kOut = Tensor("kOut", {mParam.B, mParam.S, mParam.Nkv, mParam.D}, "3", mParam.sinDtype, ge::FORMAT_ND);
    vOut = Tensor("vOut", {mParam.B, mParam.S, mParam.Nkv, mParam.D}, "3", mParam.sinDtype, ge::FORMAT_ND);
    return true;
}

bool DequantRopeQuantKvcacheCase::InitOpInfo()
{
    auto *moeKernelFunc = (void *)dequant_rope_quant_kvcache;
    bool rst = mCtx.SetOpName("DequantRopeQuantKvcache");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&x, &cosIn, &sinIn, &kcache, &vcache, &indices, &kscale, &vscale, &koffset, &voffset, &weight, &activation, &bias});
    rst = rst && mCtx.SetOutputs({&qOut, &kOut, &vOut, &kcache, &vcache});
    rst = rst && mCtx.SetAttrs({{"size_splits", mParam.sizeSplits},
                                {"quant_mode", mParam.quantOptional},
                                {"layout", mParam.qlayout},
                                {"kv_output", mParam.outOptional},
                                {"cache_mode", mParam.cacheOptional}});
    rst = rst && mCtx.SetKernelRunCbf(RunDequantRopeQuantKvcache);
    rst = rst && mCtx.SetKernelMainFunc(moeKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    dequantRopeQuantKvcacheTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingDequantRopeQuantKvcache");
    if (dequantRopeQuantKvcacheTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, DequantRopeQuantKvcache(%p)", dequantRopeQuantKvcacheTilingFunc);
        return false;
    }
    IMPL_OP(DequantRopeQuantKvcache).Tiling(TilingForDequantRopeQuantKvcacheStub);
    return rst;
}

bool DequantRopeQuantKvcacheCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool DequantRopeQuantKvcacheCase::Run()
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

DequantRopeQuantKvcacheCase::DequantRopeQuantKvcacheCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                                       Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "DequantRopeQuantKvcache";
}

DequantRopeQuantKvcacheCase::DequantRopeQuantKvcacheCase()
{
}
DequantRopeQuantKvcacheCase::Param::Param()
{
}
DequantRopeQuantKvcacheCase::Param::Param(int64_t pB, int64_t pS, int64_t pNkv, int64_t pNq, int64_t pD, int64_t pC1, int64_t pC2, 
                                            bool pOutOptional, std::string pCacheOptional,
                                            ge::DataType xDtypeIn, ge::DataType sinDtypeIn, ge::DataType biasDtypeIn)
    : B(pB), S(pS), Nkv(pNkv), Nq(pNq), D(pD), H((pNkv+pNkv+pNq)*pD), C1(pC1), C2(pC2), 
      outOptional(pOutOptional), cacheOptional(pCacheOptional), xDtype(xDtypeIn),
      sinDtype(sinDtypeIn), biasDtype(biasDtypeIn)
{
}

bool DequantRopeQuantKvcacheCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}