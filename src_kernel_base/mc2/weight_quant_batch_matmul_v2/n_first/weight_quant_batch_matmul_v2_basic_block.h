/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_basic_block.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_H

#include "../tool.h"
#include "../weight_quant_batch_matmul_v2_constant.h"
#include "basic_block_config.h"
#include "weight_quant_batch_matmul_v2_vec_compute.h"
#include "weight_quant_batch_matmul_v2_cube_compute.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"

using AscendC::GetBlockIdx;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::GetSubBlockIdx;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

namespace WeightQuantBatchMatmulV2 {

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
    const VecAntiQuantConfig &vecConfig>
class WeightQuantMatmulBasicBlock {
public:
    __aicore__ inline WeightQuantMatmulBasicBlock(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const WeightQuantBatchMatmulV2ASTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void ComputeBasicBlock(const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void Consume(const BasicBlockOffsetParam &offsetParam, uint32_t vecMte2NSize,
                                   uint64_t curVecCoreMte2RealK, uint32_t consumeKSize, uint64_t mte2RealK,
                                   uint64_t kMte2Offset);
    __aicore__ inline void VectorProcess(const BasicBlockOffsetParam &offsetParam, uint64_t antiquantRealK,
                                         uint32_t vecMte2NSize, uint64_t antiquantKOffset, uint64_t mte2RealK);
    __aicore__ inline void CubeProcess(const BasicBlockOffsetParam &offsetParam, uint64_t antiquantRealK,
                                       uint32_t kMte2Offset, uint64_t antiquantKOffset, uint64_t mte2RealK);
    __aicore__ inline void End();
    __aicore__ inline void SetAivToAic();
    __aicore__ inline void WaitAivToAic();
    __aicore__ inline void SetAicToAiv();
    __aicore__ inline void WaitAicToAiv();

    BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig> vectorCompute_;
    WeightQuantBatchMatmulV2CubeCompute<xType, wType, biasType, yType, wqmmConfig> cubeCompute_;

    TPipe *pipe_;
    uint32_t curBlockIdx_;
    bool hasBias_;
    uint64_t cvLoopIdx_ = 0;

    TBuf<TPosition::TSCM> l1Tbuf_;
    LocalTensor<xType> aF16L1_;
    LocalTensor<xType> weightF16L1_;
    LocalTensor<biasType> biasL1_;
    L1DbConfig l1DbConfig_;
};

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig, vecConfig>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, const WeightQuantBatchMatmulV2ASTilingData *tilingData, TPipe *tPipe) {
    pipe_ = tPipe;
    curBlockIdx_ = GetBlockIdx();

    // L1空间分配策略：当前L1上默认B为两块buffer
    //(1) L1上只有A和B，不包含bias和quantScale
    //  ① AL1 使能 2 buffer 时
    //    L1 (0~256KB):   WeightL1_P0 | AL1_P0 |
    //    L1 (256~512KB): AL1_P1 | WeightL1_P1 |
    //  ② AL1 使能 1 buffer 时
    //    L1 (0~512KB):   WeightL1_P0 | AL1_P0 | WeightL1_P1|
    // (2) L1上有bias或quantScale, quantScale预留
    //  ① AL1 使能 2 buffer 时
    //    L1 (0~512KB):   WeightL1_P0 | Bias_P0(4KB) |AL1_P0 | AL1_P1 | Bias_P0(4KB) | WeightL1_P1 | quantScale(8KB)
    //  ② AL1 使能 1 buffer 时
    //    L1 (0~512KB):   WeightL1_P0 | Bias_P0(4KB) |AL1_P0 | Bias_P0(4KB) | WeightL1_P1 | quantScale(8KB)

    int32_t weightL1Space = tilingData->matmulTiling.baseN * tilingData->matmulTiling.stepKb *
                            tilingData->matmulTiling.baseK; // weight单块大小
    hasBias_ = static_cast<bool>(tilingData->matmulTiling.isBias);
    int32_t biasL1Space = hasBias_ ? 4 * HALF_DATA_BENCHMARK : 0; // bias单块分配4K空间
    int32_t aF16L1Offset = weightL1Space + biasL1Space;           // A要跳过WeightL1_P0 + Bias_P0
    if (hasBias_ || IsSameType<yType, int8_t>::value) {
        int32_t aF16L1Space = 504 * HALF_DATA_BENCHMARK - DOUBLE_BUFFER_NUM * aF16L1Offset; // L1上A可占据剩余空间
        pipe_->InitBuffer(l1Tbuf_, 504 * 1024); // 除去quantScale, 共使用504KB
        l1DbConfig_.aF16L1DbOffset = aF16L1Space >> 1;
        l1DbConfig_.weightF16L1DbOffset = 504 * HALF_DATA_BENCHMARK - weightL1Space;
        if (hasBias_) {
            if constexpr (IsSameType<biasType, float>::value) {
                biasL1_ = l1Tbuf_.Get<biasType>()[weightL1Space >> 1];
                l1DbConfig_.biasL1DbOffset = (aF16L1Space + biasL1Space) >> 1;
            } else {
                biasL1_ = l1Tbuf_.Get<biasType>()[weightL1Space];
                l1DbConfig_.biasL1DbOffset = aF16L1Space + biasL1Space;
            }
        }
    } else {
        pipe_->InitBuffer(l1Tbuf_, 512 * 1024);
        l1DbConfig_.aF16L1DbOffset = 256 * HALF_DATA_BENCHMARK - weightL1Space;
        l1DbConfig_.weightF16L1DbOffset = 512 * HALF_DATA_BENCHMARK - weightL1Space;
    }
    aF16L1_ = l1Tbuf_.Get<xType>()[aF16L1Offset];
    weightF16L1_ = l1Tbuf_.Get<xType>();
    if ASCEND_IS_AIC {
        cubeCompute_.Init(x, y, bias, quantScale, quantOffset, tilingData->aPreloadSize, &(tilingData->matmulTiling),
                          tPipe, aF16L1_);
    } else {
        vectorCompute_.Init(weight, antiquantScale, antiquantOffset, tilingData, tPipe);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
    const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig,
    vecConfig>::ComputeBasicBlock(const BasicBlockOffsetParam &offsetParam)
{
    uint32_t vecMte2NSize;
    uint64_t nL1Offset; // 由cube分配到当前vector核计算的数据在N方向的偏移，即当前vector核要计算哪一块cube需要的数据
    uint32_t vecMte2KSize;
    uint32_t consumeKSize; // 单次cube/vector计算的K方向大小
    // NK输入时，N方向到两个vector核计算，KN输入时，K方向到两个vector核计算
    if constexpr (!wqmmConfig.bTrans) {
        vecMte2NSize = offsetParam.nL1Size;
        nL1Offset = 0;
        vecMte2KSize = offsetParam.kbL1Size;
        consumeKSize = offsetParam.kbL1Size >> 1;
    } else {
        vecMte2NSize = offsetParam.nL1Size >> 1;
        nL1Offset = GetSubBlockIdx() * vecMte2NSize;
        vecMte2NSize = GetSubBlockIdx() == 0 ? vecMte2NSize : offsetParam.nL1Size - vecMte2NSize;
        vecMte2KSize = vecConfig.ubMte2InnerSize;
        consumeKSize = offsetParam.kbL1Size;
    }
    uint64_t curVecCoreMte2RealK;
    for (uint64_t kMte2Offset = 0; kMte2Offset < offsetParam.kSize; kMte2Offset += vecMte2KSize) {
        uint64_t mte2RealK =
            (kMte2Offset + vecMte2KSize) > offsetParam.kSize ? offsetParam.kSize - kMte2Offset : vecMte2KSize;
        uint64_t kMte2RealOffset;
        if constexpr (!wqmmConfig.bTrans) {
            curVecCoreMte2RealK = mte2RealK >> 1;
            kMte2RealOffset = kMte2Offset + GetSubBlockIdx() * (mte2RealK >> 1);
            curVecCoreMte2RealK = GetSubBlockIdx() == 0 ? curVecCoreMte2RealK : mte2RealK - curVecCoreMte2RealK;
        } else {
            curVecCoreMte2RealK = mte2RealK;
            kMte2RealOffset = kMte2Offset;
        }

        if ASCEND_IS_AIV {
            vectorCompute_.WaitVToMTE2();
            vectorCompute_.CopyGmToUb(vecMte2NSize, curVecCoreMte2RealK, offsetParam.nOffset + nL1Offset,
                 kMte2RealOffset, offsetParam);
        }

        Consume(offsetParam, vecMte2NSize, curVecCoreMte2RealK, consumeKSize, mte2RealK, kMte2Offset);
        if ASCEND_IS_AIV {
            vectorCompute_.SetVToMTE2();
        }
    }

    if ASCEND_IS_AIC {
        cubeCompute_.GetTensorC(offsetParam);
        cubeCompute_.ClearAFullLoadFlag(); // 清除A全载时之前循环的set同步标记
    }
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
    const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig,
    vecConfig>::Consume(const BasicBlockOffsetParam &offsetParam, uint32_t vecMte2NSize,
                        uint64_t curVecCoreMte2RealK, uint32_t consumeKSize, uint64_t mte2RealK, uint64_t kMte2Offset)
{
    if constexpr (!wqmmConfig.bTrans) {
        if (curVecCoreMte2RealK == 0) {
            // 补充当前核无计算任务时的cv同步
            if ASCEND_IS_AIV {
                if (cvLoopIdx_ > 1) {
                    WaitAicToAiv();
                }
                SetAivToAic();
            }
        }
    }
    // 当前方案下，不会出现N方向计算量小于载入量的情况，所以没有N的循环
    for (uint64_t antiquantKOffset = 0; antiquantKOffset < curVecCoreMte2RealK;
         antiquantKOffset += consumeKSize, cvLoopIdx_++){
        uint64_t antiquantRealK = (antiquantKOffset + consumeKSize) >= curVecCoreMte2RealK ?
            curVecCoreMte2RealK - antiquantKOffset : consumeKSize;
        if ASCEND_IS_AIV {
            if (cvLoopIdx_ > 1) {
                WaitAicToAiv();
            }
            VectorProcess(offsetParam, antiquantRealK, vecMte2NSize, antiquantKOffset, mte2RealK);
            SetAivToAic(); // 对应的等待在CubeProcess中
        } else {
            CubeProcess(offsetParam, antiquantRealK, kMte2Offset, antiquantKOffset, mte2RealK);
            SetAicToAiv();
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
    const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig,
    vecConfig>::CubeProcess(const BasicBlockOffsetParam &offsetParam, uint64_t antiquantRealK,
                            uint32_t kMte2Offset, uint64_t antiquantKOffset, uint64_t mte2RealK)
{
    uint32_t cubeCalculateKSize;
    if constexpr (!wqmmConfig.bTrans) {
        cubeCalculateKSize = mte2RealK;
    } else {
        cubeCalculateKSize = antiquantRealK;
    }
    cubeCompute_.WaitMTE1ToMTE2(cvLoopIdx_);
    cubeCompute_.CopyAAndBiasGmToL1(l1DbConfig_, offsetParam, aF16L1_, biasL1_, kMte2Offset + antiquantKOffset,
        cubeCalculateKSize, offsetParam.nL1Size, cvLoopIdx_);
    WaitAivToAic();

    cubeCompute_.LaunchMatmul(aF16L1_, weightF16L1_, biasL1_, antiquantKOffset + kMte2Offset,
        cubeCalculateKSize, offsetParam, l1DbConfig_, cvLoopIdx_);  // mte1 mmad fixp流水
    cubeCompute_.SetMTE1ToMTE2(cvLoopIdx_);
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
    const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig,
    vecConfig>::VectorProcess(const BasicBlockOffsetParam &offsetParam, uint64_t antiquantRealK,
                              uint32_t vecMte2NSize, uint64_t antiquantKOffset, uint64_t mte2RealK)
{
    UbConsumeConfig ubConsumeConfig;
    L1ConsumeConfig l1ConsumeConfig;
    ubConsumeConfig.l1RequireVfComputeRealK = antiquantRealK;
    ubConsumeConfig.l1RequireVfComputeRealN = vecMte2NSize;
    ubConsumeConfig.kWeightLowBitUbOffset = antiquantKOffset;
    ubConsumeConfig.nWeightLowBitUbOffset = 0;
    if constexpr (!wqmmConfig.bTrans) {
        l1ConsumeConfig.l1SplitTwoVecExternalOffset = GetSubBlockIdx() * (mte2RealK >> 1);
        l1ConsumeConfig.l1RealExternalLen = mte2RealK;
    } else {
        l1ConsumeConfig.l1SplitTwoVecExternalOffset = GetSubBlockIdx() * (offsetParam.nL1Size >> 1);
        l1ConsumeConfig.l1RealExternalLen = offsetParam.nL1Size;
    }
    vectorCompute_.WeightAntiQuantCompute(ubConsumeConfig,
        weightF16L1_[(cvLoopIdx_ & 1) * l1DbConfig_.weightF16L1DbOffset], l1ConsumeConfig);
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
    const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig, vecConfig>::End()
{
    if ASCEND_IS_AIC {
        cubeCompute_.EndSync(cvLoopIdx_);
    } else {
        if (cvLoopIdx_ > 0) {
            WaitAicToAiv();
        }
        if (cvLoopIdx_ > 1) {
            WaitAicToAiv();
        }
        vectorCompute_.End();
    }
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
    const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig, vecConfig>::SetAivToAic()
{
#ifndef __CCE_KT_TEST__
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(WeightQuantBatchMatmulV2::SYNC_AIC_AIV_FLAG);
#endif
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
    const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig, vecConfig>::WaitAivToAic()
{
#ifndef __CCE_KT_TEST__
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
#endif
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
    const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig, vecConfig>::SetAicToAiv()
{
#ifndef __CCE_KT_TEST__
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
#endif
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
    const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig, vecConfig>::WaitAicToAiv()
{
#ifndef __CCE_KT_TEST__
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
#endif
}
}  // namespace WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK

#endif // WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_H
