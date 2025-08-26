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
 * \file weight_quant_batch_matmul_v2_cube_compute.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_CUBE_COMPUTE_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_CUBE_COMPUTE_H

#include "../tool.h"
#include "basic_block_config.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"

using AscendC::BLOCK_CUBE;
using AscendC::HardEvent;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;
using AscendC::IsSameType;
using AscendC::GetBlockIdx;
using AscendC::PipeBarrier;
using matmul::MatmulImpl;
using matmul::MatmulType;

namespace WeightQuantBatchMatmulV2 {
template <typename XType, typename WType, typename BiasType, typename YType, const WqmmConfig &Config>
class WeightQuantBatchMatmulV2CubeCompute {
public:
    __aicore__ inline WeightQuantBatchMatmulV2CubeCompute(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR bias, GM_ADDR quantScale, GM_ADDR quantOffset,
                                uint64_t aPreloadSize, const TCubeTiling *__restrict matmulTiling, TPipe *tPipe,
                                const LocalTensor<XType> &perloadBuffer);
    __aicore__ inline void LaunchMatmul(const LocalTensor<XType> &xl1Local, const LocalTensor<XType> &weightl1Local,
                                        const LocalTensor<BiasType> &biasl1Local, int64_t kbOffset,
                                        int32_t kbL1RealSize, const BasicBlockOffsetParam &param,
                                        const L1DbConfig &l1DbConfig, int32_t l1LoopIdx);
    __aicore__ inline void WaitMTE1ToMTE2(int32_t l1LoopIdx);
    __aicore__ inline void SetMTE1ToMTE2(int32_t l1LoopIdx);
    __aicore__ inline void CopyAAndBiasGmToL1(const L1DbConfig &l1DbConfig, const BasicBlockOffsetParam &param,
                                              const LocalTensor<XType> &al1Local,
                                              const LocalTensor<BiasType> &biasl1LocalBuf, int64_t kaGmOffset,
                                              int64_t kbL1RealSize, int64_t biasRealN, int32_t l1LoopIdx);
    __aicore__ inline void GetTensorC(const BasicBlockOffsetParam &param);
    __aicore__ inline void EndSync(int32_t l1LoopIdx);
    __aicore__ inline void ClearAFullLoadFlag();

private:
    __aicore__ inline void PreloadA(
    uint64_t aPreloadSize,const LocalTensor<XType>& perloadBuffer, const TCubeTiling *__restrict matmulTiling);
    __aicore__ inline void InitGlobalTensor(GM_ADDR x, GM_ADDR y, GM_ADDR bias, GM_ADDR quantScale);
    __aicore__ inline void InitSync();

    int8_t al1DbNum_;
    bool isBias_;
    uint64_t quantScaleValue_;

    using inputXType = MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, XType, Config.aTrans>;
    using inputWType = MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, XType, Config.bTrans>;
    using outputYType = MatmulType<TPosition::GM, CubeFormat::ND, YType>;
    using inputBiasType = MatmulType<TPosition::TSCM, CubeFormat::ND, BiasType>;
    MatmulImpl<inputXType, inputWType, outputYType, inputBiasType, CFG_MDL> mmObj_;

    AscendC::TEventID cubeEventIdsMte1ToMte2_[DOUBLE_BUFFER_NUM];
    AscendC::TEventID cubeEventIdMte2ToMte1_;
    GlobalTensor<XType> xGlobal_;
    GlobalTensor<BiasType> biasGlobal_;
    GlobalTensor<uint64_t> quantScaleGlobal_;
    GlobalTensor<YType> yGlobal_;
};

template <typename XType, typename WType, typename BiasType, typename YType, const WqmmConfig &Config>
__aicore__ inline void WeightQuantBatchMatmulV2CubeCompute<XType, WType, BiasType, YType, Config>::LaunchMatmul(
    const LocalTensor<XType> &xl1Local, const LocalTensor<XType> &weightl1Local,
    const LocalTensor<BiasType> &biasl1Local, int64_t kbOffset, int32_t kbL1RealSize,
    const BasicBlockOffsetParam &param, const L1DbConfig &l1DbConfig, int32_t l1LoopIdx) {
    mmObj_.SetOrgShape(CeilAlign(param.mL1Size, static_cast<uint32_t>(BLOCK_CUBE)),
                       CeilAlign(param.nL1Size, static_cast<uint32_t>(BLOCK_CUBE)),
                       CeilAlign(kbL1RealSize, BLOCK_CUBE),
                       CeilAlign(kbL1RealSize, BLOCK_CUBE), param.nSize);
    if (al1DbNum_ == SINGLE_BUFFER_NUM) {
        mmObj_.SetTensorA(xl1Local[CeilAlign(param.mL1Size, static_cast<uint32_t>(BLOCK_CUBE)) * kbOffset],
                          Config.aTrans);
    } else {
        mmObj_.SetTensorA(xl1Local[(l1LoopIdx & 1) * l1DbConfig.aF16L1DbOffset], Config.aTrans);
    }

    mmObj_.SetTensorB(weightl1Local[(l1LoopIdx & 1) * l1DbConfig.weightF16L1DbOffset], Config.bTrans);

    if (isBias_) {
        mmObj_.SetBias(biasl1Local[(l1LoopIdx & 1) * l1DbConfig.biasL1DbOffset]);
    }

    mmObj_.SetTail(param.mL1Size, param.nL1Size, kbL1RealSize);

    if constexpr (IsSameType<YType, int8_t>::value) {
        if constexpr (Config.quantType == QuantType::PER_TENSOR) {
            mmObj_.SetQuantScalar(quantScaleValue_);
        } else {
            mmObj_.SetQuantVector(quantScaleGlobal_[param.nOffset]);
        }
    }

    mmObj_.Iterate(kbOffset != 0);
}

template <typename XType, typename WType, typename BiasType, typename YType, const WqmmConfig &Config>
__aicore__ inline void WeightQuantBatchMatmulV2CubeCompute<XType, WType, BiasType, YType, Config>::WaitMTE1ToMTE2(
    int32_t l1LoopIdx) {
    // 编译器对成员变量数组访问优化能力较弱，会引入大量scalar，此处抽取局部变量，规避编译器优化问题
    AscendC::TEventID tempEventIdsMte1ToMte2[DOUBLE_BUFFER_NUM] = {cubeEventIdsMte1ToMte2_[0],
                                                                   cubeEventIdsMte1ToMte2_[1]};
    // 单buffer时保证了A一次全载不需要Wait，Double buffer时首次使用不需要Wait
    if (al1DbNum_ > SINGLE_BUFFER_NUM && l1LoopIdx >= DOUBLE_BUFFER_NUM) {
        WaitFlag<HardEvent::MTE1_MTE2>(tempEventIdsMte1ToMte2[l1LoopIdx & 1]);
    }
}

template <typename XType, typename WType, typename BiasType, typename YType, const WqmmConfig &Config>
__aicore__ inline void WeightQuantBatchMatmulV2CubeCompute<XType, WType, BiasType, YType, Config>::SetMTE1ToMTE2(
    int32_t l1LoopIdx) {
        // 编译器对成员变量数组访问优化能力较弱，会引入大量scalar，此处抽取局部变量，规避编译器优化问题
        AscendC::TEventID tempEventIdsMte1ToMte2[DOUBLE_BUFFER_NUM] = {cubeEventIdsMte1ToMte2_[0],
                                                                        cubeEventIdsMte1ToMte2_[1]};
        if (al1DbNum_ > SINGLE_BUFFER_NUM) {
            SetFlag<HardEvent::MTE1_MTE2>(tempEventIdsMte1ToMte2[l1LoopIdx & 1]);
        }
}

template <typename XType, typename WType, typename BiasType, typename YType, const WqmmConfig &Config>
__aicore__ inline void WeightQuantBatchMatmulV2CubeCompute<XType, WType, BiasType, YType, Config>::CopyAAndBiasGmToL1(
    const L1DbConfig& l1DbConfig, const BasicBlockOffsetParam &param, const LocalTensor<XType> &al1Local, const LocalTensor<BiasType> &biasl1LocalBuf,
    int64_t kaGmOffset, int64_t kbL1RealSize, int64_t biasRealN, int32_t l1LoopIdx) {
        if (al1DbNum_ > SINGLE_BUFFER_NUM || (al1DbNum_ == SINGLE_BUFFER_NUM && kaGmOffset == 0)) {
            int64_t aGmOffset;
            if constexpr (!Config.aTrans) {
                aGmOffset = param.mOffset * param.kSize + kaGmOffset;
            } else {
                aGmOffset = kaGmOffset * param.mSize + param.mOffset;
            }
            int64_t al1Offset = 0;
            uint32_t kaL1Len = param.kSize;
            if (al1DbNum_ > SINGLE_BUFFER_NUM) {
                al1Offset = (l1LoopIdx & 1) * l1DbConfig.aF16L1DbOffset;
                kaL1Len = kbL1RealSize;
            }

            AscendC::Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1;
            if constexpr (Config.aTrans) {
                nd2nzParams.nValue = kaL1Len;
                nd2nzParams.dValue = param.mL1Size;
                nd2nzParams.srcDValue = param.mSize;
            } else {
                nd2nzParams.nValue = param.mL1Size;
                nd2nzParams.dValue = kaL1Len;
                nd2nzParams.srcDValue = param.kSize;
            }
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE));
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;

            DataCopy(al1Local[al1Offset], xGlobal_[aGmOffset], nd2nzParams);
        }

        // bias仅与n有关，与k无关，所以只需要拷贝一次
        if (isBias_ && kaGmOffset == 0) {
            DataCopy(biasl1LocalBuf[(l1LoopIdx & 1) * l1DbConfig.biasL1DbOffset], biasGlobal_[param.nOffset],
                     CeilAlign(biasRealN, static_cast<int64_t>(BLOCK_CUBE)));
        }

        SetFlag<HardEvent::MTE2_MTE1>(cubeEventIdMte2ToMte1_);
        WaitFlag<HardEvent::MTE2_MTE1>(cubeEventIdMte2ToMte1_);
}

template <typename XType, typename WType, typename BiasType, typename YType, const WqmmConfig &Config>
__aicore__ inline void WeightQuantBatchMatmulV2CubeCompute<XType, WType, BiasType, YType, Config>::EndSync(
    int32_t l1LoopIdx) {
    AscendC::TEventID tempEventIdsMte1ToMte2[DOUBLE_BUFFER_NUM] = {cubeEventIdsMte1ToMte2_[0],
                                                                   cubeEventIdsMte1ToMte2_[1]};

    // 考虑到只循环一次时， 只需要同步wait第0块缓存。 不止1次时， 2个同步块都需要wait
    if (l1LoopIdx > 1 && al1DbNum_ > SINGLE_BUFFER_NUM) {
        WaitFlag<HardEvent::MTE1_MTE2>(tempEventIdsMte1ToMte2[l1LoopIdx & 1]);
    }

    if (l1LoopIdx > 0 && al1DbNum_ > SINGLE_BUFFER_NUM) {
        WaitFlag<HardEvent::MTE1_MTE2>(tempEventIdsMte1ToMte2[(l1LoopIdx + 1) & 1]);
    }

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(tempEventIdsMte1ToMte2[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(tempEventIdsMte1ToMte2[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(cubeEventIdMte2ToMte1_);
}

template <typename XType, typename WType, typename BiasType, typename YType, const WqmmConfig &Config>
__aicore__ inline void
WeightQuantBatchMatmulV2CubeCompute<XType, WType, BiasType, YType, Config>::ClearAFullLoadFlag() {
    if (al1DbNum_ == SINGLE_BUFFER_NUM) {
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[0]);
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[0]);
    }
}

template <typename XType, typename WType, typename BiasType, typename YType, const WqmmConfig &Config>
__aicore__ inline void WeightQuantBatchMatmulV2CubeCompute<XType, WType, BiasType, YType, Config>::InitSync() {
        cubeEventIdsMte1ToMte2_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        cubeEventIdsMte1ToMte2_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        cubeEventIdMte2ToMte1_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
}

template <typename XType, typename WType, typename BiasType, typename YType, const WqmmConfig &Config>
__aicore__ inline void WeightQuantBatchMatmulV2CubeCompute<XType, WType, BiasType, YType, Config>::InitGlobalTensor(
    GM_ADDR x, GM_ADDR y, GM_ADDR bias, GM_ADDR quantScale) {
    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(x));
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ YType *>(y));
    if (isBias_) {
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType *>(bias));
    }
    if constexpr (IsSameType<YType, int8_t>::value) {
        quantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(quantScale));
    }
}

template <typename XType, typename WType, typename BiasType, typename YType, const WqmmConfig &Config>
__aicore__ inline void WeightQuantBatchMatmulV2CubeCompute<XType, WType, BiasType, YType, Config>::PreloadA(
    uint64_t aPreloadSize,
    const LocalTensor<XType>& perloadBuffer, const TCubeTiling *__restrict matmulTiling)
{
    uint64_t xOffset = GetBlockIdx() * aPreloadSize;
    uint64_t xSizeLimit = matmulTiling->M * matmulTiling->Ka;
    if(aPreloadSize == 0 || xOffset >= xSizeLimit){
        return;
    }
    DataCopy(perloadBuffer, xGlobal_[xOffset],
            xOffset + aPreloadSize > xSizeLimit ? xSizeLimit - xOffset : aPreloadSize);
    PipeBarrier<PIPE_MTE2>();
}

template <typename XType, typename WType, typename BiasType, typename YType, const WqmmConfig &Config>
__aicore__ inline void WeightQuantBatchMatmulV2CubeCompute<XType, WType, BiasType, YType, Config>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR bias, GM_ADDR quantScale, GM_ADDR quantOffset, uint64_t aPreloadSize,
    const TCubeTiling *__restrict matmulTiling, TPipe *tPipe, const LocalTensor<XType>& perloadBuffer) {
    InitGlobalTensor(x, y, bias, quantScale);
    PreloadA(aPreloadSize, perloadBuffer, matmulTiling);
    isBias_ = matmulTiling->isBias;
    int32_t kaL1Size = matmulTiling->stepKa * matmulTiling->baseK;
    if (kaL1Size >= matmulTiling->Ka) {
        al1DbNum_ = SINGLE_BUFFER_NUM;
    } else {
        al1DbNum_ = DOUBLE_BUFFER_NUM;
    }
    mmObj_.SetSubBlockIdx(0);
    mmObj_.Init(matmulTiling, tPipe);
    InitSync();

    if constexpr (IsSameType<YType, int8_t>::value && Config.quantType == QuantType::PER_TENSOR) {
        quantScaleValue_ = this->quantScaleGlobal_.GetValue(0);
    }
}

template <typename XType, typename WType, typename BiasType, typename YType, const WqmmConfig &Config>
__aicore__ inline void WeightQuantBatchMatmulV2CubeCompute<XType, WType, BiasType, YType, Config>::GetTensorC(
    const BasicBlockOffsetParam &param) {
    uint64_t outOffset = param.mOffset * param.nSize + param.nOffset;
#ifndef __CCE_KT_TEST__
    mmObj_.GetTensorC(yGlobal_[outOffset]);
#endif
}
}  // namespace WeightQuantBatchMatmulV2
#endif