/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_quant_v2_case.cpp
 * \brief MoeInitRoutingQuantV2 测试用例.
 */
#include "moe_init_routing_quant_v2_case.h"
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

#define MOE_INIT_ROUTING_QUANT_V2_KERNEL_PARAM                                                                               \
    (GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX, GM_ADDR expandedRowIdx,           \
     GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity, GM_ADDR dynamicQuantScale, GM_ADDR workspace,           \
     GM_ADDR tiling)

using MoeInitRoutingQuantV2KernelFunc = void(*) MOE_INIT_ROUTING_QUANT_V2_KERNEL_PARAM;

extern "C" __global__ __aicore__ void moe_init_routing_quant_v2 MOE_INIT_ROUTING_QUANT_V2_KERNEL_PARAM;

using namespace ops::adv::tests::MoeInitRoutingQuantV2;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunMoeInitRoutingQuantV2(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                         std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (MoeInitRoutingQuantV2KernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), outputs[0]->GetDevData(),
                outputs[1]->GetDevData(), outputs[2]->GetDevData(), outputs[3]->GetDevData(),
                outputs[4]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForMoeInitRoutingQuantV2Stub(gert::TilingContext *context)
{
    auto *moeInitRoutingQuantV2Case = static_cast<MoeInitRoutingQuantV2Case *>(Case::GetCurrentCase());
    if (moeInitRoutingQuantV2Case != nullptr) {
        MoeInitRoutingQuantV2Case::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!moeInitRoutingQuantV2Case->DoOpTiling(p)) {
            return p.ret;
        }
        return moeInitRoutingQuantV2Case->moeInitRoutingQuantV2TilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool MoeInitRoutingQuantV2Case::InitParam()
{
    x = Tensor("x", {mParam.n, mParam.h}, "2", mParam.optionalOutputDt, ge::FORMAT_ND);
    expertIdx = Tensor("expertIdx", {mParam.n, mParam.k}, "2", ge::DataType::DT_INT32, ge::FORMAT_ND);
    // expandX的shape由参数决定
    if (mParam.quantMode == 0) {
        scale = Tensor("scale", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
        offset = Tensor("offset", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else if (mParam.smoothtype == 0) {
        scale = Tensor("scale", {}, "None", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
        offset = Tensor("offset", {}, "None", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else if (mParam.smoothtype == 1) {
        scale = Tensor("scale", {1, mParam.h}, "2", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
        offset = Tensor("offset", {1, mParam.h}, "2", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        scale = Tensor("scale", {mParam.e, mParam.h}, "2", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
        offset = Tensor("offset", {mParam.e, mParam.h}, "2", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    }
    if (mParam.dropPadMode == 0) {
        int64_t firstDim = mParam.n * mParam.k;
        if (mParam.activeNum > 0 && mParam.activeNum < firstDim) {
            firstDim = mParam.activeNum;
        }
        quantScale = Tensor("dynamicQuantScale", {firstDim}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);

        expandedX = Tensor("expandedX", {firstDim, mParam.h}, "2", ge::DataType::DT_INT8, ge::FORMAT_ND);
    } else {
        quantScale = Tensor("dynamicQuantScale", {mParam.e * mParam.c}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);

        expandedX = Tensor("expandedX", {mParam.e, mParam.c, mParam.h}, "3", ge::DataType::DT_INT8, ge::FORMAT_ND);
    }
    expandedRowIdx = Tensor("expandedRowIdx", {mParam.n * mParam.k}, "1", ge::DataType::DT_INT32, ge::FORMAT_ND);
    expertTokensCountOrCumsum =
        Tensor("expertTokensCountOrCumsum", {mParam.e}, "1", ge::DataType::DT_INT32, ge::FORMAT_ND);
    expertTokensBeforeCapacity =
        Tensor("expertTokensBeforeCapacity", {mParam.e}, "1", ge::DataType::DT_INT32, ge::FORMAT_ND);
    return true;
}

bool MoeInitRoutingQuantV2Case::InitOpInfo()
{
    bool rst = mCtx.SetOpName("MoeInitRoutingQuantV2");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&x, &expertIdx, &scale, &offset});
    rst =
        rst && mCtx.SetOutputs({&expandedX, &expandedRowIdx, &expertTokensCountOrCumsum, &expertTokensBeforeCapacity, &quantScale});
    rst = rst && mCtx.SetAttrs({{"active_num", mParam.activeNum},
                                {"expert_capacity", mParam.c},
                                {"expert_num", mParam.e},
                                {"drop_pad_mode", mParam.dropPadMode},
                                {"expert_tokens_count_or_cumsum_flag", mParam.countFlag},
                                {"expert_tokens_before_capacity_flag", mParam.tokenFlag},
                                {"quant_mode", mParam.quantMode},
                                });
    rst = rst && mCtx.SetKernelRunCbf(RunMoeInitRoutingQuantV2);
    rst = rst && mCtx.SetKernelMainFunc((void *)moe_init_routing_quant_v2);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    moeInitRoutingQuantV2TilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingForMoeInitRoutingQuantV2");
    if (moeInitRoutingQuantV2TilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, moeInitRoutingQuantV2(%p)", moeInitRoutingQuantV2TilingFunc);
        return false;
    }
    IMPL_OP(MoeInitRoutingQuantV2).Tiling(TilingForMoeInitRoutingQuantV2Stub);
    return rst;
}

bool MoeInitRoutingQuantV2Case::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool MoeInitRoutingQuantV2Case::Run()
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

MoeInitRoutingQuantV2Case::MoeInitRoutingQuantV2Case(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                           Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "MoeInitRoutingQuantV2";
}

MoeInitRoutingQuantV2Case::MoeInitRoutingQuantV2Case()
{
}
MoeInitRoutingQuantV2Case::Param::Param()
{
}
MoeInitRoutingQuantV2Case::Param::Param(int64_t pN, int64_t pH, int64_t pK, int64_t pActiveNum, int64_t pC, int64_t pE,
                                   int64_t pDropPadMode, int64_t pCountFlag, bool pTokenFlag,
                                   ge::DataType pOptionalOutputDt, int64_t quantModeIn, int64_t smoothtypeIn)

    : n(pN), h(pH), k(pK), activeNum(pActiveNum), c(pC), e(pE), dropPadMode(pDropPadMode), countFlag(pCountFlag),
      tokenFlag(pTokenFlag), optionalOutputDt(pOptionalOutputDt), quantMode(quantModeIn), smoothtype(smoothtypeIn)
{
}


bool MoeInitRoutingQuantV2Case::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}