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
 * \file apply_rotary_pos_emb_tiling.cpp
 * \brief
 */
#include "apply_rotary_pos_emb_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/ops_log.h"

namespace {
static const int64_t INPUT0 = 0;
static const int64_t INPUT1 = 1;
static const int64_t INPUT2 = 2;
static const int64_t INPUT3 = 3;
static const int64_t DIM_0 = 0;
static const int64_t DIM_1 = 1;
static const int64_t DIM_2 = 2;
static const int64_t DIM_3 = 3;
static const int64_t DIM_4 = 4;
static const int64_t LASTDIM = 128;
static const int64_t BLOCK_SIZE = 32;
static const int64_t REPEAT_FP32 = 64;
static const int64_t REPEAT_FP16 = 128;
static const int64_t ONE_BLOCK_NUM = 8;
static const int64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;

inline int64_t ComputeTimes(const int64_t value, const int64_t factor)
{
    if (factor == 0) {
        return 0;
    }
    int64_t loopTimes = value / factor;
    if (value % factor == 0) {
        loopTimes = loopTimes - 1;
    }
    return loopTimes;
}
} // namespace

namespace optiling {
struct ApplyRotaryPosEmbParams {
    int64_t totalCoreNum = 0;
    int64_t totalUbSize = 0;
    int64_t sysWorkspaceSize = 0;
    int64_t preCoreBatch = 0;
    int64_t lastCoreBatch = 0;
    int64_t oneBlockFp32 = 0;
    int64_t qDims = 0;
    int64_t qDim0 = 0;
    int64_t qDim1 = 0;
    int64_t qDim3 = 0;
    int64_t kDim0 = 0;
    int64_t kDim1 = 0;
    int64_t kDim3 = 0;
    int64_t cosDim0 = 0;
    int64_t cosDim1 = 0;
    int64_t cosDim3 = 0;
    bool isCast = false;
    bool isFp32 = false;
    int64_t castDtypeSize = 0;
    int32_t dtypeSize = 0;
    int64_t oneBlock = 0;
    int64_t useCoreNum = 0;
    int64_t lastDim = 0;
    int64_t halfNum = 0;
    int64_t preCBatchB = 0;
    int64_t preCBatchL = 0;
    int64_t lastCBatchL = 0;
    int64_t comBatchBB = 0;
    int64_t comBatchBBL = 0;
    int64_t comBatchBLL = 0;
    int64_t comBatchLBL = 0;
    int64_t comBatchLLL = 0;
    int64_t qPart1Ub = 0;
    int64_t q2q1Part1Ub = 0;
    int64_t cosPart1Ub = 0;
    int64_t sin1UbSize = 0;
    int64_t preCLTimes = 0;
    int64_t lastCLTimes = 0;
    int64_t preCBBTimes = 0;
    int64_t preCBLTimes = 0;
    int64_t preCLLTimes = 0;
    int64_t qCoreOffset = 0;
    int64_t kCoreOffset = 0;
    int64_t cosCoreOffset = 0;
    int64_t qcNum = 0;
    int64_t kcNum = 0;
    int64_t coscNum = 0;
    int64_t qcdNum = 0;
    int64_t kcdNum = 0;
    int64_t coscdNum = 0;
    int64_t qkcNum = 0;
    int64_t mulNum = 0;
    int64_t qcdHalfNum = 0;
    int64_t dstRepSBr = 0;
    int64_t blockLenQ = 0;
    int64_t srcStrideK = 0;
    int64_t blockLenq2q1 = 0;
    int64_t mask = 0;
    int64_t tilingKey = 0;
};

ge::graphStatus GetInputParams(gert::TilingContext *context, ApplyRotaryPosEmbParams &params)
{
    auto q = context->GetInputTensor(INPUT0);
    OPS_LOG_E_IF_NULL(context, q, return ge::GRAPH_FAILED);
    gert::Shape qShape = q->GetStorageShape();
    params.qDims = qShape.GetDimNum();
    OPS_CHECK(params.qDims != DIM_4, OPS_LOG_E(context, "q shape dims is not four"),
              return ge::GRAPH_FAILED);
    params.qDim0 = qShape.GetDim(DIM_0);
    params.qDim1 = qShape.GetDim(DIM_1);
    params.qcNum = qShape.GetDim(DIM_2);
    params.qDim3 = qShape.GetDim(DIM_3);
    auto k = context->GetInputTensor(INPUT1);
    OPS_LOG_E_IF_NULL(context, k, return ge::GRAPH_FAILED);
    gert::Shape kShape = k->GetStorageShape();
    int64_t kDims = kShape.GetDimNum();
    OPS_CHECK(kDims != params.qDims, OPS_LOG_E(context, "k shape dims is not same as q dims"),
              return ge::GRAPH_FAILED);
    params.kDim0 = kShape.GetDim(DIM_0);
    params.kDim1 = kShape.GetDim(DIM_1);
    params.kcNum = kShape.GetDim(DIM_2);
    params.kDim3 = kShape.GetDim(DIM_3);
    auto cos = context->GetInputTensor(INPUT2);
    OPS_LOG_E_IF_NULL(context, cos, return ge::GRAPH_FAILED);
    gert::Shape cosShape = cos->GetStorageShape();
    int64_t cosDims = cosShape.GetDimNum();
    OPS_CHECK(cosDims != DIM_4, OPS_LOG_E(context, "cos shape dims is not four"),
              return ge::GRAPH_FAILED);
    params.cosDim0 = cosShape.GetDim(DIM_0);
    params.cosDim1 = cosShape.GetDim(DIM_1);
    params.coscNum = cosShape.GetDim(DIM_2);
    params.cosDim3 = cosShape.GetDim(DIM_3);
    auto sin = context->GetInputTensor(INPUT3);
    OPS_LOG_E_IF_NULL(context, sin, return ge::GRAPH_FAILED);
    gert::Shape sinShape = sin->GetStorageShape();
    int64_t sinDims = sinShape.GetDimNum();
    OPS_CHECK((sinDims != DIM_4) || (cosShape != sinShape),
              OPS_LOG_E(context, "cos and sin shape not equal or rank not four"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(qShape.GetShapeSize() == 0 || kShape.GetShapeSize() == 0 || cosShape.GetShapeSize() == 0 ||
                  sinShape.GetShapeSize() == 0,
              OPS_LOG_E(context, "input can not be empty."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckParams(gert::TilingContext *context, ApplyRotaryPosEmbParams &params)
{
    OPS_CHECK(GetInputParams(context, params) != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context, "GetInputParams failed"), return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t *layoutAttr = attrs->GetAttrPointer<int64_t>(0);
    OPS_LOG_E_IF_NULL(context, layoutAttr, return ge::GRAPH_FAILED);
    int64_t layout = *layoutAttr;
    OPS_CHECK(layout != 1, OPS_LOG_E(context, "layout is not one"), return ge::GRAPH_FAILED);
    OPS_CHECK((params.kDim0 != params.qDim0) || (params.cosDim0 != params.kDim0),
              OPS_LOG_E(context, "all input dim0 must equal"), return ge::GRAPH_FAILED);
    OPS_CHECK((params.kDim1 != params.qDim1) || (params.cosDim1 != params.kDim1),
              OPS_LOG_E(context, "all input dim1 must equal"), return ge::GRAPH_FAILED);
    OPS_CHECK((params.kDim3 != params.qDim3) || (params.cosDim3 != params.kDim3),
              OPS_LOG_E(context, "all input dim3 must equal"), return ge::GRAPH_FAILED);
    OPS_CHECK(params.qDim3 != LASTDIM, OPS_LOG_E(context, "last dims is not 128"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(params.coscNum != 1, OPS_LOG_E(context, "cos dim2 is not one"), return ge::GRAPH_FAILED);
    OPS_CHECK(context->GetInputDesc(INPUT0) == nullptr, OPS_LOG_E(context, "input 0 get desc failed"),
              return ge::GRAPH_FAILED);
    ge::DataType qDtype = context->GetInputDesc(INPUT0)->GetDataType();
    for (int32_t i = 1; i < DIM_4; i++) {
        auto desc = context->GetInputDesc(i);
        OPS_CHECK(desc == nullptr, OPS_LOG_E(context, "get input[%d] Desc is null !", i),
                  return ge::GRAPH_FAILED);
        ge::DataType inputDtype = desc->GetDataType();
        OPS_CHECK(inputDtype != qDtype, OPS_LOG_E(context, "input[%d] dtype not right", i),
                  return ge::GRAPH_FAILED);
    }
    params.isCast = qDtype == ge::DT_BF16;
    params.isFp32 = qDtype == ge::DT_FLOAT;
    params.dtypeSize = ge::GetSizeByDataType(qDtype);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ComputeAB(gert::TilingContext *context, ApplyRotaryPosEmbParams &params)
{
    // 一个batch搬运及计算使用的UB size (2: ping-pong)
    int64_t oneLoop = params.qPart1Ub * 2 + params.cosPart1Ub * 4 + params.qPart1Ub + params.q2q1Part1Ub * 2 +
                      static_cast<int64_t>(params.isCast) * (params.sin1UbSize * 2);
    OPS_LOG_D(context, "oneLoop ub size %ld", oneLoop);
    OPS_CHECK(oneLoop > params.totalUbSize || oneLoop <= 0,
              OPS_LOG_E(context, "oneLoop is too large or small than 0"), return ge::GRAPH_FAILED);
    int64_t times = params.totalUbSize / oneLoop; // ub一次可以计算的batch数量
    int64_t shengUb = params.totalUbSize % oneLoop;
    // shengMte：如果有剩余ub空间，再计算几个单batch后一起搬出，减少mte3次数
    int64_t shengMte = shengUb / (params.qPart1Ub * 2 + params.cosPart1Ub * 4 + params.qPart1Ub);
    shengMte = params.isCast ? 0 : shengMte;
    if (times > params.preCoreBatch) { // UB可以一次放下当前核要出的batch数量
        times = params.preCoreBatch;
        shengMte = 0;
    }
    if (shengMte == 0 || params.isCast) { // 搬入搬出一一对应情况
        params.tilingKey = static_cast<int64_t>(ApplyRotaryPosEmbTilingKey::TILINGKEY_AB_CAST);
    } else { // 存在多次mte2搬入，计算，一次mte3搬出的场景
        params.tilingKey = static_cast<int64_t>(ApplyRotaryPosEmbTilingKey::TILINGKEY_AB);
    }
    OPS_LOG_D(context, "times is %ld, shengMte %ld", times, shengMte);

    // ub外循环次数计算
    params.preCBatchB = times + shengMte;                                               // 大核内ub一次计算的batch数
    params.preCLTimes = ComputeTimes(params.preCoreBatch, params.preCBatchB);           // 大核的ub loop times
    params.preCBatchL = params.preCoreBatch - params.preCBatchB * params.preCLTimes;    // 大核内最后一次计算的batch数
    params.lastCLTimes = ComputeTimes(params.lastCoreBatch, params.preCBatchB);         // 尾核的ub loop times
    params.lastCBatchL = params.lastCoreBatch - params.preCBatchB * params.lastCLTimes; // 尾核最后一次计算的batch数

    // ub内循环相关计算
    // ub一次搬运占用空间size
    params.qPart1Ub = params.preCBatchB * params.qPart1Ub;
    params.cosPart1Ub = params.preCBatchB * params.cosPart1Ub;
    // ub一次计算占用空间size
    params.q2q1Part1Ub = times * params.q2q1Part1Ub;
    params.sin1UbSize = times * params.sin1UbSize;
    // BB: before core + before loop; BL: before core + last loop; LL: last core + last loop
    // BBL: before core + before loop + lat batch; BLL: before core + last loop + last batch
    params.comBatchBB = times;
    params.preCBBTimes = ComputeTimes(params.preCBatchB, params.comBatchBB);
    params.comBatchBBL = params.preCBatchB - params.preCBBTimes * params.comBatchBB;
    params.preCBLTimes = ComputeTimes(params.preCBatchL, params.comBatchBB);
    params.comBatchBLL = params.preCBatchL - params.preCBLTimes * params.comBatchBB;
    params.preCLLTimes = ComputeTimes(params.lastCBatchL, params.comBatchBB);
    params.comBatchLLL = params.lastCBatchL - params.preCLLTimes * params.comBatchBB;

    params.blockLenQ = params.qcdNum / params.oneBlock;  // Q_ND block数，搬运
    params.srcStrideK = params.kcdNum / params.oneBlock; // K_ND blcok数，搬入Q时为K预留的拼接空间
    params.dstRepSBr = params.lastDim / params.oneBlockFp32;
    params.blockLenq2q1 = params.halfNum / params.oneBlockFp32;
    params.mulNum = params.mulNum * DIM_2 / params.oneBlockFp32;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Compute(gert::TilingContext *context, ApplyRotaryPosEmbParams &params)
{
    OPS_LOG_D(context, "ApplyRotaryPosEmb compute start");
    params.castDtypeSize = params.isCast ? DIM_4 : params.dtypeSize;
    params.oneBlock = BLOCK_SIZE / params.dtypeSize;                       // 搬运，一个block可以存放的输入数据
    params.oneBlockFp32 = params.isCast ? ONE_BLOCK_NUM : params.oneBlock; // 计算，一个block存放的计算数据量
    params.mask = (params.isCast || params.isFp32) ? REPEAT_FP32 : REPEAT_FP16;
    OPS_LOG_D(context, "isCast is %d, dtypeSize is %d, isFp32 is %d", params.isCast, params.dtypeSize,
              params.isFp32);

    // a, b, c, d: dim0(B), dim1(S), dim2(N), dim3(D)
    int64_t ab = params.kDim0 * params.kDim1;                                   // 总batch = BS
    params.preCoreBatch = (ab + params.totalCoreNum - 1) / params.totalCoreNum; // front核每核处理batch数据量
    params.useCoreNum = (ab + params.preCoreBatch - 1) / params.preCoreBatch;   // 使用的核数
    params.lastCoreBatch = ab - (params.useCoreNum - 1) * params.preCoreBatch;  // 尾核处理batch数据量
    OPS_LOG_D(context, "preCoreBatch %ld, lastCoreBatch is %ld", params.preCoreBatch,
              params.lastCoreBatch);
    params.lastDim = params.kDim3;
    params.halfNum = params.kDim3 / DIM_2;

    params.qcdNum = params.qcNum * params.qDim3;       // Q_n * D
    params.kcdNum = params.kcNum * params.kDim3;       // K_n * D
    params.coscdNum = params.coscNum * params.cosDim3; // coscNum = 1 --> 1 * D
    params.qkcNum = params.qcNum + params.kcNum;       // (Q_n + K_n), Q、K在N轴上进行拼接
    params.mulNum = params.qkcNum * params.halfNum;    // (Q_n + K_n) * D / 2
    params.qcdHalfNum = params.qcNum * params.halfNum; // Q_n * D / 2
    // 单核处理的数据偏移 batch * ND
    params.qCoreOffset = params.preCoreBatch * params.qcdNum;
    params.kCoreOffset = params.preCoreBatch * params.kcdNum;
    params.cosCoreOffset = params.preCoreBatch * params.coscdNum;
    // ub size
    params.qPart1Ub =
        params.qkcNum * params.lastDim * params.castDtypeSize; // 搬运 (Q_n + K_n) * D * 4(fp32)/2(fp16/bf16)
    params.cosPart1Ub = params.coscNum * params.lastDim * params.dtypeSize; // 搬运 1 * D * 4(fp32)/2(fp16/bf16)
    params.q2q1Part1Ub =
        params.qkcNum * params.lastDim * params.castDtypeSize; // 计算 (Q_n + K_n) * D * 4(bf16/fp32)/2(fp16)
    params.sin1UbSize = params.coscNum * params.lastDim * params.castDtypeSize; // 计算 1 * D * 4， 为BF16 cast使用
    int64_t speUb = params.qPart1Ub * 2 + params.cosPart1Ub * 2 + params.q2q1Part1Ub * 2 +
                    static_cast<int64_t>(params.isCast) * (params.sin1UbSize * 2);
    OPS_LOG_D(context, "speUb is %ld, totalUbSize is %ld", speUb, params.totalUbSize);

    // 小shape，每核处理一个batcch
    if (params.preCoreBatch == 1 && speUb <= params.totalUbSize) {
        params.tilingKey = static_cast<int64_t>(ApplyRotaryPosEmbTilingKey::TILINGKEY_SMALL);
        params.mulNum = params.qkcNum * params.lastDim;
        params.blockLenQ = params.halfNum / params.oneBlockFp32; // D/2 占用block数，搬运
        params.dstRepSBr = params.lastDim / params.oneBlockFp32; // D 占用block数，计算
        params.qcdHalfNum = params.lastDim / params.mask;        // D 计算时的repeat次数（按列计算）
        return ge::GRAPH_SUCCESS;
    }
    OPS_CHECK(ComputeAB(context, params) != ge::GRAPH_SUCCESS, OPS_LOG_E(context, "ComputeAB failed"),
              return ge::GRAPH_FAILED);
    OPS_LOG_D(context, "ApplyRotaryPosEmb compute end");
    return ge::GRAPH_SUCCESS;
}

void PrintTilingData(gert::TilingContext *context, ApplyRotaryPosEmbTilingData &tiling, ApplyRotaryPosEmbParams &params)
{
    OPS_LOG_D(context,
              "Print ApplyRotaryPosEmb tilingData: useCoreNum is %ld, lastDim %ld,"
              "halfNum is %ld, preCBatchB %ld, preCBatchL is %ld, lastCBatchL is %ld,"
              "comBatchBB is %ld, comBatchBBL %ld, comBatchBLL is %ld, comBatchLLL is %ld,"
              "qPart1Ub is %ld, q2q1Part1Ub is %ld, cosPart1Ub is %ld, sin1UbSize is %ld,"
              "preCLTimes is %ld, lastCLTimes is %ld, preCBBTimes is %ld, preCBLTimes is %ld,"
              "preCLLTimes is %ld, qCoreOffset is %ld, kCoreOffset is %ld, cosCoreOffset is %ld,"
              "qcNum is %ld, kcNum is %ld, coscNum is %ld, qcdNum is %ld, kcdNum is %ld,"
              "coscdNum is %ld, qkcNum is %ld, mulNum is %ld, qcdHalfNum is %ld, dstRepSBr is %ld,"
              "blockLenQ is %ld, srcStrideK is %ld, blockLenq2q1 is %ld,  mask is %ld, tilingKey is %ld",
              tiling.get_useCoreNum(), tiling.get_lastDim(), tiling.get_halfNum(), tiling.get_preCBatchB(),
              tiling.get_preCBatchL(), tiling.get_lastCBatchL(), tiling.get_comBatchBB(), tiling.get_comBatchBBL(),
              tiling.get_comBatchBLL(), tiling.get_comBatchLLL(), tiling.get_qPart1Ub(), tiling.get_q2q1Part1Ub(),
              tiling.get_cosPart1Ub(), tiling.get_sin1UbSize(), tiling.get_preCLTimes(), tiling.get_lastCLTimes(),
              tiling.get_preCBBTimes(), tiling.get_preCBLTimes(), tiling.get_preCLLTimes(), tiling.get_qCoreOffset(),
              tiling.get_kCoreOffset(), tiling.get_cosCoreOffset(), params.qcNum, params.kcNum, params.coscNum,
              tiling.get_qcdNum(), tiling.get_kcdNum(), tiling.get_coscdNum(), tiling.get_qkcNum(), tiling.get_mulNum(),
              tiling.get_qcdHalfNum(), tiling.get_dstRepSBr(), tiling.get_blockLenQ(), tiling.get_srcStrideK(),
              tiling.get_blockLenq2q1(), tiling.get_mask(), params.tilingKey);
}

void SetTilingData(gert::TilingContext *context, ApplyRotaryPosEmbTilingData &tiling, ApplyRotaryPosEmbParams &params)
{
    tiling.set_useCoreNum(params.useCoreNum);
    tiling.set_lastDim(params.lastDim);
    tiling.set_halfNum(params.halfNum);
    tiling.set_preCBatchB(params.preCBatchB);
    tiling.set_preCBatchL(params.preCBatchL);
    tiling.set_lastCBatchL(params.lastCBatchL);
    tiling.set_comBatchBB(params.comBatchBB);
    tiling.set_comBatchBBL(params.comBatchBBL);
    tiling.set_comBatchBLL(params.comBatchBLL);
    tiling.set_comBatchLLL(params.comBatchLLL);
    tiling.set_qPart1Ub(params.qPart1Ub);
    tiling.set_q2q1Part1Ub(params.q2q1Part1Ub);
    tiling.set_cosPart1Ub(params.cosPart1Ub);
    tiling.set_sin1UbSize(params.sin1UbSize);
    tiling.set_preCLTimes(params.preCLTimes);
    tiling.set_lastCLTimes(params.lastCLTimes);
    tiling.set_preCBBTimes(params.preCBBTimes);
    tiling.set_preCBLTimes(params.preCBLTimes);
    tiling.set_preCLLTimes(params.preCLLTimes);
    tiling.set_qCoreOffset(params.qCoreOffset);
    tiling.set_kCoreOffset(params.kCoreOffset);
    tiling.set_cosCoreOffset(params.cosCoreOffset);
    tiling.set_qcdNum(params.qcdNum);
    tiling.set_kcdNum(params.kcdNum);
    tiling.set_coscdNum(params.coscdNum);
    tiling.set_qkcNum(params.qkcNum);
    tiling.set_mulNum(params.mulNum);
    tiling.set_qcdHalfNum(params.qcdHalfNum);
    tiling.set_dstRepSBr(params.dstRepSBr);
    tiling.set_blockLenQ(params.blockLenQ);
    tiling.set_srcStrideK(params.srcStrideK);
    tiling.set_blockLenq2q1(params.blockLenq2q1);
    tiling.set_mask(params.mask);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(params.useCoreNum);
    context->SetTilingKey(params.tilingKey);

    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = params.sysWorkspaceSize;
}

AROPE_EXTERN_C ge::graphStatus Tiling4ApplyRotaryPosEmb(gert::TilingContext *context)
{
    ApplyRotaryPosEmbTilingData tilingData;
    ApplyRotaryPosEmbParams params;

    // get and check platform information
    OPS_LOG_D(context, "get platform information from ascendc interface.");
    auto platformInfo = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platformInfo, return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    params.totalCoreNum = ascendcPlatform.GetCoreNumAiv();

    uint64_t platformUbSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformUbSize);
    params.totalUbSize = platformUbSize;

    int64_t sysWorkspaceSize = static_cast<int64_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    params.sysWorkspaceSize = sysWorkspaceSize > WORK_SPACE_SIZE ? sysWorkspaceSize : WORK_SPACE_SIZE;
    
    OPS_LOG_D(context, "totalCoreNum is %ld", params.totalCoreNum);
    OPS_CHECK(params.totalCoreNum <= 0, OPS_LOG_E(context, "PrepareTiling fail to get core num."),
              return ge::GRAPH_FAILED);
    OPS_LOG_D(context, "totalUbSize is %ld", params.totalUbSize);
    OPS_CHECK(params.totalUbSize <= 0, OPS_LOG_E(context, "PrepareTiling fail to get ub size."),
              return ge::GRAPH_FAILED);

    if (CheckParams(context, params) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context, "CheckParams return failed.");
        return ge::GRAPH_FAILED;
    }

    if (Compute(context, params) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context, "Compute return failed.");
        return ge::GRAPH_FAILED;
    }

    SetTilingData(context, tilingData, params);
    PrintTilingData(context, tilingData, params);
    return ge::GRAPH_SUCCESS;
}

AROPE_EXTERN_C ge::graphStatus TilingPrepare4ApplyRotaryPosEmb(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyRotaryPosEmb)
    .Tiling(Tiling4ApplyRotaryPosEmb)
    .TilingParse<ApplyRotaryPosEmbCompileInfo>(TilingPrepare4ApplyRotaryPosEmb);
} // namespace optiling
