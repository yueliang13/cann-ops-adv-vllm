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
 * \file moe_init_routing_quant_case.cpp
 * \brief MoeInitRoutingQuant 测试用例.
 */
#include "moe_init_routing_quant_case.h"
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

#define MOE_INIT_ROUTING_QUANT_KERNEL_PARAM          \
    (GM_ADDR x, GM_ADDR rowIdx, GM_ADDR expertIdx,   \
     GM_ADDR expandedX, GM_ADDR expandedRowIdx,      \
     GM_ADDR expandedExpertIdx, GM_ADDR workspace,   \
     GM_ADDR tiling)

using MoeInitRoutingQuantKernelFunc = void(*) MOE_INIT_ROUTING_QUANT_KERNEL_PARAM;

extern "C" __global__ __aicore__ void moe_init_routing_quant MOE_INIT_ROUTING_QUANT_KERNEL_PARAM;

using namespace ops::adv::tests::MoeInitRoutingQuant;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunMoeInitRoutingQuant(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                         std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (MoeInitRoutingQuantKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(),
                inputs[2]->GetDevData(),
                outputs[0]->GetDevData(),
                outputs[1]->GetDevData(), outputs[2]->GetDevData(),
                workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForMoeInitRoutingQuantStub(gert::TilingContext *context)
{
    auto *moeInitRoutingQuantCase = static_cast<MoeInitRoutingQuantCase *>(Case::GetCurrentCase());
    if (moeInitRoutingQuantCase != nullptr) {
        MoeInitRoutingQuantCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!moeInitRoutingQuantCase->DoOpTiling(p)) {
            return p.ret;
        }
        return moeInitRoutingQuantCase->moeInitRoutingQuantTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool MoeInitRoutingQuantCase::InitParam()
{
    x = Tensor("x", {mParam.n, mParam.h}, "2", mParam.optionalOutputDt, ge::FORMAT_ND);
    expertIdx = Tensor("expertIdx", {mParam.n, mParam.k}, "2", ge::DataType::DT_INT32, ge::FORMAT_ND);
    // expandX的shape由参数决定
    rowIdx = Tensor("rowIdx", {mParam.n, mParam.k}, "2", ge::DataType::DT_INT32, ge::FORMAT_ND);
    int64_t firstDim = mParam.n * mParam.k;
    if (mParam.activeNum > 0 && mParam.activeNum* mParam.k < firstDim) {
        firstDim = mParam.activeNum* mParam.k;
    }

    expandedX = Tensor("expandedX", {firstDim, mParam.h}, "2", ge::DataType::DT_INT8, ge::FORMAT_ND);

    expandedRowIdx = Tensor("expandedRowIdx", {mParam.n * mParam.k}, "1", ge::DataType::DT_INT32, ge::FORMAT_ND);
    expanded_expert_idx = Tensor("expanded_expert_idx", {mParam.n * mParam.k}, "1", ge::DataType::DT_INT32, ge::FORMAT_ND);

    return true;
}

bool MoeInitRoutingQuantCase::InitOpInfo()
{
    
    bool rst = mCtx.SetOpName("MoeInitRoutingQuant");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&x, &expertIdx,&rowIdx});
    rst =
        rst && mCtx.SetOutputs({&expandedX, &expandedRowIdx, &expanded_expert_idx});
    rst = rst && mCtx.SetAttrs({{"active_num", mParam.activeNum},
                                {"scale", (float)0.1},
                                {"offset", (float)0.1},
                                });
    rst = rst && mCtx.SetKernelRunCbf(RunMoeInitRoutingQuant);
    rst = rst && mCtx.SetKernelMainFunc((void *)moe_init_routing_quant);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    moeInitRoutingQuantTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingForMoeInitRoutingQuant");
    if (moeInitRoutingQuantTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, moeInitRoutingQuant(%p)", moeInitRoutingQuantTilingFunc);
        return false;
    }
    IMPL_OP(MoeInitRoutingQuant).Tiling(TilingForMoeInitRoutingQuantStub);
    return rst;
}

bool MoeInitRoutingQuantCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool MoeInitRoutingQuantCase::Run()
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

MoeInitRoutingQuantCase::MoeInitRoutingQuantCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                           Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "MoeInitRoutingQuant";
}

MoeInitRoutingQuantCase::MoeInitRoutingQuantCase()
{
}
MoeInitRoutingQuantCase::Param::Param()
{
}
MoeInitRoutingQuantCase::Param::Param(int64_t pN, int64_t pH, int64_t pK, int64_t pActiveNum, int64_t pC, int64_t pE,
                                   int64_t pDropPadMode, int64_t pCountFlag, bool pTokenFlag,
                                   ge::DataType pOptionalOutputDt, int64_t quantModeIn, int64_t smoothtypeIn)

    : n(pN), h(pH), k(pK), activeNum(pActiveNum), c(pC), e(pE), dropPadMode(pDropPadMode), countFlag(pCountFlag),
      tokenFlag(pTokenFlag), optionalOutputDt(pOptionalOutputDt), quantMode(quantModeIn), smoothtype(smoothtypeIn)
{
}


bool MoeInitRoutingQuantCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}