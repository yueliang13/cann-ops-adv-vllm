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
 * \file gmmadd_case.cpp
 * \brief GroupedMatmulAdd 测试用例.
 */

#include "gmm_add_case.h"
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

#define GMM_ADD_KERNEL_PARAM                                                                                               \
    (GM_ADDR x, GM_ADDR weight, GM_ADDR groupList, GM_ADDR y,\
        GM_ADDR yRef, GM_ADDR workspace, GM_ADDR tiling)

typedef void(*GmmAddKernelFunc) GMM_ADD_KERNEL_PARAM;

extern "C" __global__ __aicore__ void grouped_matmul_add GMM_ADD_KERNEL_PARAM;


using namespace ops::adv::tests::gmmadd;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunGroupedMatmulAdd(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                            std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (GmmAddKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), outputs[0]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingGroupedMatmulAdd(gert::TilingContext *context)
{
    auto *gmmaddCase = static_cast<GmmAddCase *>(Case::GetCurrentCase());
    if (gmmaddCase != nullptr) {
        GmmAddCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        // p.actualSeqLengthsTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(5)); // 5:act_seq_len idx
        if (!gmmaddCase->DoOpTiling(p)) {
            return p.ret;
        }
        return gmmaddCase->gmmaddTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool GmmAddCase::InitParam()
{
    x = Tensor("x", {mParam.k, mParam.m}, "2", mParam.dx, ge::FORMAT_ND);
    weight = Tensor("weight", {mParam.k, mParam.n}, "2", mParam.dx, ge::FORMAT_ND);
    groupList = Tensor("groupList", {mParam.groupNum}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND);
    y = Tensor("y", {mParam.groupNum * mParam.m, mParam.n}, "2", ge::DataType::DT_FLOAT, ge::FORMAT_ND);

    return true;
}

bool GmmAddCase::InitOpInfo()
{
    auto *gmmaddKernelFunc = (void *)grouped_matmul_add;

    bool rst = mCtx.SetOpName("GroupedMatmulAdd");
    rst = rst && mCtx.SetTilingDataMaxSize(4096);
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&x, &weight, &groupList, &y});
    rst = rst && mCtx.SetOutputs({&y});
    rst = rst && mCtx.SetKernelRunCbf(RunGroupedMatmulAdd);
    rst = rst && mCtx.SetKernelMainFunc(gmmaddKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    gmmaddTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("Tiling4GroupedMatmulAdd");
    if (gmmaddTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, gmmadd(%p)", gmmaddTilingFunc);
        return false;
    }
    IMPL_OP(GroupedMatmulAdd).Tiling(TilingGroupedMatmulAdd);
    return true;
}

bool GmmAddCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool GmmAddCase::Run()
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

GmmAddCase::GmmAddCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "GroupedMatmulAdd";
}

GmmAddCase::GmmAddCase()
{
}
GmmAddCase::Param::Param()
{
}
GmmAddCase::Param::Param(int64_t M, int64_t N, int64_t K, int64_t GroupedNum, ge::DataType Dx)
    : m(M), n(N), k(K), groupNum(GroupedNum), dx(Dx)
{
}


bool GmmAddCase::DoOpTiling(DoTilingParam& tilingParam) {
  if (tilingParam.ctx == nullptr) {
    return false;
  }
  return true;
}