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
 * \file grouped_matmul_antiquant_a8w4_msd_pre.h
 * \brief
 */

#ifndef ASCENDC_GROUPED_MATMUL_ANTIQUANT_A8W4_MSD_PRE_H
#define ASCENDC_GROUPED_MATMUL_ANTIQUANT_A8W4_MSD_PRE_H

#include "kernel_operator.h"
#ifdef GMM_ANTI_QUANT_A8W4_MSD
namespace GROUPED_MATMUL{
using namespace AscendC;
#define BUFFER_NUM_A8W4_PRE 1
constexpr int TWO = 2;
constexpr int EIGHT = 8;
constexpr size_t LEN_128 = 128;     // 16bit operator
constexpr int DATA_BLOCK_SIZE_32 = 32;
class GMMA8W4PreProcess {
public:
    __aicore__ inline GMMA8W4PreProcess(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR groupList, GM_ADDR workspace, GMMBaseParams& tilingData, TPipe *pipe);
    __aicore__ inline void Process();
private:
    TQue<QuePosition::VECIN, BUFFER_NUM_A8W4_PRE> vecInQueueX, vecInQueueXBak;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueueA1;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueueA2;

    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueueA3;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueueA4;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueueA5;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueue0F;
    TQue<QuePosition::VECIN, BUFFER_NUM_A8W4_PRE> vecInQueueG;
    TQue<QuePosition::VECIN, BUFFER_NUM_A8W4_PRE> vecInQueueGF;
    TQue<QuePosition::VECIN, BUFFER_NUM_A8W4_PRE> vecInWork;

    LocalTensor<int8_t> xTensor;
    LocalTensor<half> xHighHalfTensor;
    LocalTensor<half> xLowHalfTensor;
    LocalTensor<half> xLowHalfTensor2;
    LocalTensor<int4b_t> xHighI4Tensor;
    LocalTensor<int4b_t> xLowI4Tensor;
    LocalTensor<int16_t> xLowI16Tensor;
    LocalTensor<int64_t> groupListTensor;
    LocalTensor<float> groupListFTensor;
    LocalTensor<float> workTensor;
    GlobalTensor<int8_t> xGm;
    GlobalTensor<int8_t> yGm;
    GlobalTensor<int64_t> groupListGm;
    uint32_t vK;
    uint32_t vKAlign;
    uint32_t totalGroup{0};
    uint32_t blockDim;
    uint32_t coreId;
    uint32_t currentNum;
    uint32_t startNum;
    uint32_t groupNum;
};

__aicore__ inline void GMMA8W4PreProcess::Init(GM_ADDR x, GM_ADDR y, GM_ADDR groupList, GM_ADDR workspace, GMMBaseParams& tilingData, TPipe *pipe){
    xGm.SetGlobalBuffer(GetTensorAddr<int8_t>(0, x));
    yGm.SetGlobalBuffer(GetTensorAddr<int8_t>(0, y));
    groupListGm.SetGlobalBuffer((__gm__ int64_t *)groupList);

    vK = tilingData.k;
    pipe->InitBuffer(vecInQueueX, BUFFER_NUM_A8W4_PRE, vK * sizeof(int8_t)); // 2KB
    pipe->InitBuffer(vecOutQueueA1, BUFFER_NUM_A8W4_PRE, vK * sizeof(int4b_t)); // 1KB
    
    pipe->InitBuffer(vecOutQueueA2, BUFFER_NUM_A8W4_PRE, vK * sizeof(int4b_t)); // 1KB
    
    pipe->InitBuffer(vecOutQueueA3, BUFFER_NUM_A8W4_PRE, vK * sizeof(half)); // 4 KB
    pipe->InitBuffer(vecOutQueueA4, BUFFER_NUM_A8W4_PRE, vK * sizeof(half)); // 4 KB
    pipe->InitBuffer(vecOutQueueA5, BUFFER_NUM_A8W4_PRE, vK * sizeof(half)); // 4 KB

    constexpr int BUFFER_SIZE_256B = 128 * sizeof(int16_t);
    pipe->InitBuffer(vecOutQueue0F, BUFFER_NUM_A8W4_PRE, BUFFER_SIZE_256B); // 256 B
    groupNum = static_cast<uint32_t>(tilingData.groupNum);
    pipe->InitBuffer(vecInQueueG, BUFFER_NUM_A8W4_PRE, Ceil(groupNum * sizeof(int64_t), DATA_BLOCK_SIZE_32));
    pipe->InitBuffer(vecInQueueGF, BUFFER_NUM_A8W4_PRE, Ceil(groupNum * sizeof(float), DATA_BLOCK_SIZE_32));
    pipe->InitBuffer(vecInWork, BUFFER_NUM_A8W4_PRE, BUFFER_SIZE_256B * sizeof(float));
    startNum = GetBlockIdx();
    blockDim = GetBlockNum() * GetTaskRation();
}

__aicore__ inline void GMMA8W4PreProcess::Process()
{
    constexpr int32_t MASK = 128;
    xTensor = vecInQueueX.AllocTensor<int8_t>();
    xHighI4Tensor = vecOutQueueA1.AllocTensor<int4b_t>();
    xLowI4Tensor = vecOutQueueA2.AllocTensor<int4b_t>();
    xHighHalfTensor = vecOutQueueA3.AllocTensor<half>();
    xLowHalfTensor = vecOutQueueA4.AllocTensor<half>();
    xLowHalfTensor2 = vecOutQueueA5.AllocTensor<half>();
    xLowI16Tensor = vecOutQueue0F.AllocTensor<int16_t>();
    Duplicate(xLowI16Tensor, static_cast<int16_t>(0x0F0F), MASK);   // get rid of high 4 bits in every int8

    const size_t LEN_VK = (vK / 2) / 128;   // align to 128
    const half one_eight = static_cast<half>(0.0625);
    //先计算要处理的group数
    groupListTensor = vecInQueueG.AllocTensor<int64_t>();
    groupListFTensor = vecInQueueGF.AllocTensor<float>();
    workTensor = vecInWork.AllocTensor<float>();
    DataCopyParams dataCopyParams{1, static_cast<uint16_t>(groupNum * sizeof(int64_t)), 0, 0};
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyPad(groupListTensor, groupListGm, dataCopyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(groupListFTensor, groupListTensor, AscendC::RoundMode::CAST_ROUND, groupNum);
    pipe_barrier(PIPE_V);
    ReduceSum(groupListFTensor, groupListFTensor, workTensor, groupNum);
    pipe_barrier(PIPE_V);
    Cast(groupListTensor, groupListFTensor, AscendC::RoundMode::CAST_ROUND, 1);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    totalGroup = groupListTensor.GetValue(0);

    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    for(uint32_t xloop = startNum; xloop < totalGroup; xloop += blockDim){
        uint64_t startAddr = xloop * vK;
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopy(xTensor, xGm[startAddr], vK);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(xHighHalfTensor, xTensor, AscendC::RoundMode::CAST_NONE, vK);
        pipe_barrier(PIPE_V);
        Muls(xHighHalfTensor, xHighHalfTensor, one_eight, vK);
        pipe_barrier(PIPE_V);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        Cast(xHighI4Tensor, xHighHalfTensor, AscendC::RoundMode::CAST_FLOOR, vK);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(yGm[startAddr], xHighI4Tensor.ReinterpretCast<int8_t>(), vK / 2);  // 2: size int8 -> int4
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        And(xLowHalfTensor.ReinterpretCast<int16_t>(), xTensor.ReinterpretCast<int16_t>(), xLowI16Tensor, LEN_128, LEN_VK, {1,1,1,8,8,0});
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        Cast(xLowHalfTensor2.ReinterpretCast<half>(), xLowHalfTensor.ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE, vK);
        pipe_barrier(PIPE_V);
        const half MINUS_EIGHT = static_cast<half>(-8);     // shrink 0~15 to -8~7
        Adds(xHighHalfTensor, xLowHalfTensor2, MINUS_EIGHT, vK);
        pipe_barrier(PIPE_V);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        Cast(xLowI4Tensor, xHighHalfTensor.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, vK);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        DataCopy(yGm[startAddr + vK / TWO], xLowI4Tensor.ReinterpretCast<int8_t>(), vK / TWO);                          
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    SyncAll<false>();
    vecInQueueX.FreeTensor(xTensor);
    vecOutQueueA1.FreeTensor(xHighI4Tensor);
    vecOutQueueA2.FreeTensor(xLowI4Tensor);

    vecOutQueueA3.FreeTensor(xHighHalfTensor);
    vecOutQueueA4.FreeTensor(xLowHalfTensor);
    vecOutQueueA5.FreeTensor(xLowHalfTensor2);
    vecOutQueue0F.FreeTensor(xLowI16Tensor);
    vecInQueueG.FreeTensor(groupListTensor);
    vecInQueueGF.FreeTensor(groupListFTensor);
    vecInWork.FreeTensor(workTensor);
}

} // namespace
#endif
#endif