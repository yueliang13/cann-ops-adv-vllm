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
 * \file mfr_case.cpp
 * \brief MoeFinalizeRoutingV2 测试用例.
 */

#include "mfr_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/mfr/tiling_data.h"
#include "tiling/tiling_base.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define MFR_KERNEL_PARAM                                                                                               \
    (GM_ADDR expandedPermutedRows, \
        GM_ADDR expandedSrcToDstRow,\
        GM_ADDR skip1,\
        GM_ADDR skip2, GM_ADDR bias,\
        GM_ADDR scales,\
        GM_ADDR expertForSourceRow, GM_ADDR out,\
        GM_ADDR workspace, GM_ADDR tiling)

typedef void(*IfaKernelFunc) MFR_KERNEL_PARAM;

extern "C" __global__ __aicore__ void moe_finalize_routing_v2 MFR_KERNEL_PARAM;

// extern "C" __global__ __aicore__ void incre_flash_attention_fp16_int8 MFR_KERNEL_PARAM;

// extern "C" __global__ __aicore__ void incre_flash_attention_bf16_bf16 MFR_KERNEL_PARAM;

// extern "C" __global__ __aicore__ void incre_flash_attention_bf16_int8 MFR_KERNEL_PARAM;

using namespace ops::adv::tests::mfr;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunMoeFinalizeRoutingV2(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                            std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (IfaKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(),
                outputs[0]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingMoeFinalizeRoutingV2(gert::TilingContext *context)
{
    auto *mfrCase = static_cast<MfrCase *>(Case::GetCurrentCase());
    if (mfrCase != nullptr) {
        MfrCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        // p.actualSeqLengthsTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(5)); // 5:act_seq_len idx
        if (!mfrCase->DoOpTiling(p)) {
            return p.ret;
        }
        return mfrCase->mfrTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool MfrCase::InitParam()
{
    if(mParam.dropPadMode == 0 || mParam.dropPadMode == 2) {
        expandedX = Tensor("expandedX", {mParam.numRows * mParam.k, mParam.h}, "2", mParam.dx, ge::FORMAT_ND);
    } else {
        expandedX = Tensor("expandedX", {mParam.e, mParam.c, mParam.h}, "3", mParam.dx, ge::FORMAT_ND);
    }
    // expandedX = Tensor("expandedX", {mParam.n, mParam.h}, "2", mParam.xDtype, ge::FORMAT_ND);
    // expertIdx = Tensor("expertIdx", {mParam.n, mParam.k}, "2", mParam.indexDtype, ge::FORMAT_ND);
    // // expandX的shape由参数决定
    // int64_t firstDim = mParam.n * mParam.k;
    // if (mParam.activeNum > 0 && mParam.activeNum < firstDim) {
    //     firstDim = mParam.activeNum;
    // }
    // expandedX = Tensor("expandedX", {firstDim, mParam.h}, "2", mParam.xDtype, ge::FORMAT_ND);
    // skip1, skip2, bias, scales, expertForSourceRow, out;
    if(mParam.skip1) {
        skip1 = Tensor("skip1", {mParam.numRows, mParam.h}, "2", mParam.dx, ge::FORMAT_ND);
    }
    if(mParam.skip2) {
        skip2 = Tensor("skip1", {mParam.numRows, mParam.h}, "2", mParam.dx, ge::FORMAT_ND);
    }
    if(mParam.bias) {
        bias = Tensor("bias", {mParam.e, mParam.h}, "2", mParam.dx, ge::FORMAT_ND);
    }
    scales = Tensor("scales", {mParam.numRows, mParam.k}, "2", mParam.dx, ge::FORMAT_ND);
    expertForSourceRow = Tensor("expertForSourceRow", {mParam.numRows, mParam.k}, "2", ge::DataType::DT_INT32, ge::FORMAT_ND);
    expandedRowIdx = Tensor("expandedRowIdx", {mParam.numRows * mParam.k}, "1", ge::DataType::DT_INT32, ge::FORMAT_ND);
    out = Tensor("out", {mParam.numRows, mParam.h}, "2", mParam.dx, ge::FORMAT_ND);

    return true;
}

bool MfrCase::InitOpInfo()
{
    auto *mfrKernelFunc = (void *)moe_finalize_routing_v2;

    bool rst = mCtx.SetOpName("MoeFinalizeRoutingV2");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&expandedX, &expandedRowIdx, &skip1, &skip2, &bias, &scales, &expertForSourceRow});
    rst = rst && mCtx.SetOutputs({&out});
    rst = rst && mCtx.SetAttrs({{"dropPadMode", mParam.dropPadMode}});
    rst = rst && mCtx.SetKernelRunCbf(RunMoeFinalizeRoutingV2);
    rst = rst && mCtx.SetKernelMainFunc(mfrKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    mfrTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("Tiling4MoeFinalizeRoutingV2");
    if (mfrTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, mfr(%p)", mfrTilingFunc);
        return false;
    }
    IMPL_OP(MoeFinalizeRoutingV2).Tiling(TilingMoeFinalizeRoutingV2);
    return true;
}

bool MfrCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool MfrCase::Run()
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

MfrCase::MfrCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "MoeFinalizeRoutingV2";
}

MfrCase::MfrCase()
{
}
MfrCase::Param::Param()
{
}
MfrCase::Param::Param(int64_t E, int64_t C, int64_t H, int64_t NUM_ROWS, int64_t K, ge::DataType dx_, int64_t dropPadMode_)
    : e(E), c(C), h(H), numRows(NUM_ROWS), k(K), dropPadMode(dropPadMode_), dx(dx_)
{
}
MfrCase::Param::Param(int64_t E, int64_t C, int64_t H, int64_t NUM_ROWS, int64_t K, ge::DataType dx_, int64_t dropPadMode_, bool skip1_, bool skip2_, bool bias_)
    : e(E), c(C), h(H), numRows(NUM_ROWS), k(K), dropPadMode(dropPadMode_), dx(dx_), skip1(skip1_), skip2(skip2_), bias(bias_)
{
}


bool MfrCase::DoOpTiling(DoTilingParam& tilingParam) {
  if (tilingParam.ctx == nullptr) {
    return false;
  }
  return true;
}