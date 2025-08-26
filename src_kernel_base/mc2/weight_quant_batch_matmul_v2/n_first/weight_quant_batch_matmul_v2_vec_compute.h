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
 * \file weight_quant_batch_matmul_v2_vec_compute.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_VEC_COMPUTE_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_VEC_COMPUTE_H

#include "../tool.h"
#include "../weight_quant_batch_matmul_v2_constant.h"
#include "basic_block_config.h"
#include "basic_block_vf.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"

using AscendC::BLOCK_CUBE;
using AscendC::CacheMode;
using AscendC::DataCopyParams;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TEventID;
using AscendC::TPipe;
using AscendC::WaitFlag;
using AscendC::VECTOR_REG_WIDTH;
namespace MicroAPI = AscendC::MicroAPI;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::GetRound;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::TypeGet;

namespace WeightQuantBatchMatmulV2 {

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
class BasicBlockLibVectorAntiQuantCompute {
    using VregType = typename TypeGet<xType>::T;

  public:
    __aicore__ inline BasicBlockLibVectorAntiQuantCompute(){};

    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR antiQuantScale, GM_ADDR antiQuantOffset,
                                const WeightQuantBatchMatmulV2ASTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void WaitVToMTE2();
    __aicore__ inline void SetVToMTE2();
    __aicore__ inline void CopyGmToUb(uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset,
                                      uint64_t ubMte2KOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void CopyWeightGmToUb(uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset,
                                            uint64_t ubMte2KOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void CopyAntiQuantParamsGmToUb(uint64_t ubMte2NSize, uint64_t ubMte2NOffset,
                                                     const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void WeightAntiQuantCompute(const UbConsumeConfig &ubConsumeConfig,
                                                  const LocalTensor<xType> &weightF16L1,
                                                  const L1ConsumeConfig &l1ConsumeConfig);
    __aicore__ inline void AntiQuantProcess(uint64_t vfExternalRealLen, uint64_t nWeightLowBitUbOffset,
                                            uint64_t kWeightLowBitUbOffset);
    __aicore__ inline void AntiQuantB8CommonNdNk(uint64_t ubLoopN, uint64_t nWeightLowBitUbOffset,
                                                 uint64_t kWeightLowBitUbOffset);
    __aicore__ inline void AntiQuantB8CommonNdKn(uint64_t ubLoopK, uint64_t nWeightLowBitUbOffset,
                                                 uint64_t kWeightLowBitUbOffset);
    __aicore__ inline void AntiQuantInt4NdNk(uint64_t ubLoopN, uint64_t nWeightLowBitUbOffset,
                                             uint64_t kWeightLowBitUbOffset);
    __aicore__ inline void AntiQuantInt4NdKn(uint64_t ubLoopK, uint64_t nWeightLowBitUbOffset,
                                             uint64_t kWeightLowBitUbOffset);
    __aicore__ inline void AntiQuantFP8NdNk(uint64_t ubLoopN, uint64_t nWeightLowBitUbOffset,
                                            uint64_t kWeightLowBitUbOffset);
    __aicore__ inline void AntiQuantFP8NdKn(uint64_t ubLoopK, uint64_t nWeightLowBitUbOffset,
                                            uint64_t kWeightLowBitUbOffset);
    __aicore__ inline void CalLocalAddrForVf(uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset,
                                             LocalAddressParam<xType, wType> &localAddressParam);
    __aicore__ inline void WeightF16UbToL1(uint64_t weightF16L1Offset, uint64_t antiQuantRealN, uint64_t antiQuantRealK,
                                           const LocalTensor<xType> &weightF16L1, uint64_t l1RealExternalLen);
    __aicore__ inline void End();

    // mte2搬运计数，用于控制weight输入的buffer和 mte2&&V间同步控制
    uint64_t ubMte2LoopIdx_ = 0;
    // vf中标准计算单元(64,256)的计数，用于控制weight反量化后输出的buffer和V&&mte3间同步控制
    uint64_t ubComputeLoopIdx_ = 0;

    TBuf<> ubBuffer_;

    TEventID vecEventIdVToMte2_[QUADRUPLE_BUFFER_NUM];
    TEventID vecEventIdMte3ToV_[DOUBLE_BUFFER_NUM];

    xType scaleValue_;
    xType offsetValue_;

    GlobalTensor<wType> wGlobal_;
    GlobalTensor<xType> antiQuantOffsetGlobal_;
    GlobalTensor<xType> antiQuantScaleGlobal_;

    LocalTensor<int8_t> ubWeightInputLowBitTotalBuffer_;
    LocalTensor<xType> ubWeightOutputF16TotalBuffer_;
    LocalTensor<xType> ubAntiQuantScaleTotalBuffer_;
    LocalTensor<xType> ubAntiQuantOffsetTotalBuffer_;

    constexpr static uint64_t ubAvailableSize_ = 248 * INT8_DATA_BENCHMARK;
    constexpr static uint64_t weightInputLowBitUbTotalBufferSize_ = 174 * INT8_DATA_BENCHMARK;
    constexpr static uint64_t weightOutputF16UbTotalBufferSize_ = 66 * HALF_DATA_BENCHMARK;
    constexpr static uint64_t antiQuantScaleUbTotalBufferSize_ = 4 * HALF_DATA_BENCHMARK;
    constexpr static uint64_t antiQuantOffsetUbTotalBufferSize_ = 4 * HALF_DATA_BENCHMARK;
    constexpr static uint64_t weightInputLowBitUbSingleBufferSize_ =
        weightInputLowBitUbTotalBufferSize_ / vecConfig.ubMte2BufferNum;
    constexpr static uint64_t antiQuantScaleUbSingleBufferSize_ =
        antiQuantScaleUbTotalBufferSize_ / vecConfig.ubMte2BufferNum;
    constexpr static uint64_t antiQuantOffsetUbSingleBufferSize_ =
        antiQuantOffsetUbTotalBufferSize_ / vecConfig.ubMte2BufferNum;
    constexpr static uint64_t weightOutputF16UbSingleBufferSize_ =
        weightOutputF16UbTotalBufferSize_ / 2; // F16的数据固定为两块buffer 每块为33KB

    constexpr static uint16_t vfNStandardLen = (wqmmConfig.bTrans) ? 64 : VECTOR_REG_WIDTH;
    constexpr static uint16_t vfKStandardLen = (wqmmConfig.bTrans) ? VECTOR_REG_WIDTH : 64;
    constexpr static uint16_t VEC_MAX_ELEM_B16 = VECTOR_REG_WIDTH / sizeof(xType);
};

/*
 * 初始化buffer和同步所需的EventID
 */
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::Init(
    GM_ADDR weight, GM_ADDR antiQuantScale, GM_ADDR antiQuantOffset,
    const WeightQuantBatchMatmulV2ASTilingData *tilingData, TPipe *tPipe) {
    wGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ wType *>(weight));
    antiQuantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiQuantScale));
    antiQuantOffsetGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiQuantOffset));

    if (!tilingData->weightL2Cacheable) {
        wGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }

    tPipe->InitBuffer(ubBuffer_, ubAvailableSize_);
    ubWeightInputLowBitTotalBuffer_ =
        ubBuffer_.template GetWithOffset<int8_t>(weightInputLowBitUbTotalBufferSize_, 0); // 174KB
    ubWeightOutputF16TotalBuffer_ =
        ubBuffer_.template GetWithOffset<xType>(weightOutputF16UbTotalBufferSize_, 174 * INT8_DATA_BENCHMARK); // 33KB*2
    ubAntiQuantScaleTotalBuffer_ =
        ubBuffer_.template GetWithOffset<xType>(antiQuantScaleUbTotalBufferSize_, 240 * INT8_DATA_BENCHMARK); // 4KB
    ubAntiQuantOffsetTotalBuffer_ =
        ubBuffer_.template GetWithOffset<xType>(antiQuantOffsetUbTotalBufferSize_, 244 * INT8_DATA_BENCHMARK); // 4KB

    for (int32_t i = 0; i < DOUBLE_BUFFER_NUM; ++i) {
        vecEventIdMte3ToV_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        vecEventIdVToMte2_[i] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    }
    if (vecConfig.ubMte2BufferNum == QUADRUPLE_BUFFER_NUM) {
        for (int32_t i = 2; i < QUADRUPLE_BUFFER_NUM; ++i) {
            vecEventIdVToMte2_[i] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::WaitVToMTE2() {
    // 用临时变量接一下，优化编译的作用
    TEventID vecEventIdVToMte2[QUADRUPLE_BUFFER_NUM] = {vecEventIdVToMte2_[0], vecEventIdVToMte2_[1],
                                                        vecEventIdVToMte2_[2], vecEventIdVToMte2_[3]};
    if (likely(ubMte2LoopIdx_ > vecConfig.ubMte2BufferNum - 1)) {
        WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2[ubMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1)]);
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::SetVToMTE2() {
    // 用临时变量接一下，优化编译的作用
    TEventID vecEventIdVToMte2[QUADRUPLE_BUFFER_NUM] = {vecEventIdVToMte2_[0], vecEventIdVToMte2_[1],
                                                        vecEventIdVToMte2_[2], vecEventIdVToMte2_[3]};
    SetFlag<HardEvent::V_MTE2>(vecEventIdVToMte2[(ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)]);
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::CopyGmToUb(
    uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset, uint64_t ubMte2KOffset,
    const BasicBlockOffsetParam &offsetParam) {
    // ubMte2NSize和ubMte2KSize为实际MTE2搬运到UB的有效数据，
    // 其按照ubMte2InnerSize进行跳写，垃圾数据无需操作，搬出的时搬运有效数据即可。
    if (ubMte2NSize == 0 || ubMte2KSize == 0) {
        ubMte2LoopIdx_++; // 避免当前核无任务时，SetVToMTE2()对同一个flagID重复SetFlag的问题
        return;
    }
    CopyWeightGmToUb(ubMte2NSize, ubMte2KSize, ubMte2NOffset, ubMte2KOffset, offsetParam);
    CopyAntiQuantParamsGmToUb(ubMte2NSize, ubMte2NOffset, offsetParam);
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    ubMte2LoopIdx_++;
}

/*
 * 该函数weight矩阵的搬运
 * 搬运[ubMte2NSize, ubMte2KSize]或[ubMte2KSize,
 * ubMte2NSize]大小的weight数据从GM进入UB上ubWeightInputLowBitTotalBuffer_中
 * ubWeightInputLowBitTotalBuffer_总共182KB,将其分为vecConfig.ubMte2BufferNum块
 * 每块大小为weightInputLowBitUbSingleBufferSize_.
 */
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::CopyWeightGmToUb(
    uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset, uint64_t ubMte2KOffset,
    const BasicBlockOffsetParam &offsetParam) {
    if constexpr (wqmmConfig.bTrans) {
        DataCopyPad2D(ubWeightInputLowBitTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                      weightInputLowBitUbSingleBufferSize_]
                          .template ReinterpretCast<wType>(),
                      wGlobal_[ubMte2NOffset * offsetParam.kSize + ubMte2KOffset], ubMte2NSize, ubMte2KSize,
                      vecConfig.ubMte2InnerSize, offsetParam.kSize);
    } else {
        DataCopyPad2D(ubWeightInputLowBitTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                      weightInputLowBitUbSingleBufferSize_]
                          .template ReinterpretCast<wType>(),
                      wGlobal_[ubMte2KOffset * offsetParam.nSize + ubMte2NOffset], ubMte2KSize, ubMte2NSize,
                      vecConfig.ubMte2InnerSize, offsetParam.nSize);
    }
}

/*
 * 该函数为量化参数的搬运
 * pertensor场景读取单个数据即可，perchannel场景下搬运[1,
 * ubMte2NSize]大小的antiquantscale&&antiquantoffset数据从GM进入UB上
 * ubAntiQuantScaleBuffer_&&ubAntiQuantOffsetBuffer_中。以antiquantscale为例，
 * antiQuantScaleUbTotalBufferSize_总共为16KB, 将其分为vecConfig.ubMte2BufferNum块
 * 每块大小为antiQuantScaleUbSingleBufferSize_,按照VECTOR_REG_WIDTH(256)对齐写入。
 */
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void
BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::CopyAntiQuantParamsGmToUb(
    uint64_t ubMte2NSize, uint64_t ubMte2NOffset, const BasicBlockOffsetParam &offsetParam) {
    if constexpr (wqmmConfig.antiQuantType == QuantType::PER_CHANNEL) {
        DataCopyPad2D(ubAntiQuantScaleTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                   antiQuantScaleUbSingleBufferSize_],
                      antiQuantScaleGlobal_[ubMte2NOffset], 1, ubMte2NSize,
                      CeilAlign(ubMte2NSize, static_cast<uint64_t>(VECTOR_REG_WIDTH)), offsetParam.nSize);
        if constexpr (wqmmConfig.hasAntiQuantOffset) {
            DataCopyPad2D(ubAntiQuantOffsetTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                        antiQuantOffsetUbSingleBufferSize_],
                          antiQuantOffsetGlobal_[ubMte2NOffset], 1, ubMte2NSize,
                          CeilAlign(ubMte2NSize, static_cast<uint64_t>(VECTOR_REG_WIDTH)), offsetParam.nSize);
        }

    } else if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
        scaleValue_ = antiQuantScaleGlobal_.GetValue(0);
        if constexpr (wqmmConfig.hasAntiQuantOffset) {
            offsetValue_ = antiQuantOffsetGlobal_.GetValue(0);
        }
    } else {
        // PERGROUP
    }
}

/*
 * ubConsumeConfig.l1RequireVfComputeRealK 和 ubConsumeConfig.l1RequireVfComputeRealN  表示L1上需要VEC计算的实际数据量
 * l1ExternalOffset表示L1 上的weight 上由于分两个vec核 带来的外轴的偏移， 用于搬出到L1 的偏移计算使用
 * kWeightS8UbOffset和nWeightS8UbOffset表示 UB上的偏移
 * weightF16L1 表示L1上的地址
 * VEC 上按照 标准VF计算单元(64,256) 进行多次循环计算该数据量
 * 每次一个标准VF计算单元计算完毕后放置在 ubWeightOutputF16TotalBuffer_ 上,
 * ubWeightOutputF16TotalBuffer_为两块33KB的空间,
 * 每块都存放一个VF计算单元的结果，其计算和结果的MTE3搬运使用vecEventIdMte3ToV_控制同步
 */
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::WeightAntiQuantCompute(
    const UbConsumeConfig &ubConsumeConfig, const LocalTensor<xType> &weightF16L1,
    const L1ConsumeConfig &l1ConsumeConfig) {
    uint64_t weightF16L1Offset;
    uint64_t nRealLen;
    uint64_t kRealLen;
    // 用临时变量接一下，优化编译的作用
    TEventID vecEventIdMte3ToV[DOUBLE_BUFFER_NUM] = {vecEventIdMte3ToV_[0], vecEventIdMte3ToV_[1]};
    for (uint64_t antiQuantKOffset = 0; antiQuantKOffset < ubConsumeConfig.l1RequireVfComputeRealK;
         antiQuantKOffset += vfKStandardLen) {
        for (uint64_t antiQuantNOffset = 0; antiQuantNOffset < ubConsumeConfig.l1RequireVfComputeRealN;
             antiQuantNOffset += vfNStandardLen) {
            if (likely(ubComputeLoopIdx_ > 1)) {
                WaitFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV[ubComputeLoopIdx_ & 1]);
            }
            nRealLen = antiQuantNOffset + vfNStandardLen >= ubConsumeConfig.l1RequireVfComputeRealN
                           ? ubConsumeConfig.l1RequireVfComputeRealN - antiQuantNOffset
                           : vfNStandardLen;
            kRealLen = antiQuantKOffset + vfKStandardLen >= ubConsumeConfig.l1RequireVfComputeRealK
                           ? ubConsumeConfig.l1RequireVfComputeRealK - antiQuantKOffset
                           : vfKStandardLen;
            if constexpr (wqmmConfig.bTrans) {
                weightF16L1Offset = CeilAlign(l1ConsumeConfig.l1RealExternalLen, FP16_BLOCK_SIZE) * antiQuantKOffset +
                                    (antiQuantNOffset + l1ConsumeConfig.l1SplitTwoVecExternalOffset) * FP16_BLOCK_SIZE;
                AntiQuantProcess(nRealLen, antiQuantNOffset + ubConsumeConfig.nWeightLowBitUbOffset,
                                 antiQuantKOffset + ubConsumeConfig.kWeightLowBitUbOffset);
            } else {
                weightF16L1Offset = CeilAlign(l1ConsumeConfig.l1RealExternalLen, FP16_BLOCK_SIZE) * antiQuantNOffset +
                                    (antiQuantKOffset + l1ConsumeConfig.l1SplitTwoVecExternalOffset) * FP16_BLOCK_SIZE;
                AntiQuantProcess(kRealLen, antiQuantNOffset + ubConsumeConfig.nWeightLowBitUbOffset,
                                 antiQuantKOffset + ubConsumeConfig.kWeightLowBitUbOffset);
            }

            event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
            SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
            WeightF16UbToL1(weightF16L1Offset, nRealLen, kRealLen, weightF16L1, l1ConsumeConfig.l1RealExternalLen);
            SetFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV[ubComputeLoopIdx_ & 1]);
            ubComputeLoopIdx_++;
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::AntiQuantProcess(
    uint64_t vfExternalRealLen, uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset) {
    if constexpr (IsSameType<wType, int8_t>::value || IsSameType<wType, hifloat8_t>::value) {
        if constexpr (wqmmConfig.bTrans) {
            AntiQuantB8CommonNdNk(vfExternalRealLen, nWeightLowBitUbOffset, kWeightLowBitUbOffset);
        } else {
            AntiQuantB8CommonNdKn(vfExternalRealLen, nWeightLowBitUbOffset, kWeightLowBitUbOffset);
        }
    } else if constexpr(IsSameType<wType, float8_e5m2_t>::value || IsSameType<wType, float8_e4m3_t>::value) {
        LocalAddressParam<xType, wType> fp8AddressParam;
        CalLocalAddrForVf(nWeightLowBitUbOffset, kWeightLowBitUbOffset, fp8AddressParam);
        CalculateParam<xType> calculateParam;
        calculateParam.offsetValue = offsetValue_;
        calculateParam.scaleValue = scaleValue_;
        calculateParam.ubLoop = vfExternalRealLen;
        if constexpr (wqmmConfig.bTrans) {
            AscendC::VF_CALL<AntiQuantFP8NdNkVf<xType, wType, wqmmConfig.hasAntiQuantOffset, vecConfig.ubMte2InnerSize,
                wqmmConfig.antiQuantType>>(fp8AddressParam, calculateParam);
        } else {
            AscendC::VF_CALL<AntiQuantFP8NdKnVf<xType, wType, wqmmConfig.hasAntiQuantOffset, vecConfig.ubMte2InnerSize,
                wqmmConfig.antiQuantType>>(fp8AddressParam, calculateParam);
        }
    } else {
        if constexpr (wqmmConfig.bTrans) {
            AntiQuantInt4NdNk(vfExternalRealLen, nWeightLowBitUbOffset, kWeightLowBitUbOffset);
        } else {
            AntiQuantInt4NdKn(vfExternalRealLen, nWeightLowBitUbOffset, kWeightLowBitUbOffset);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::CalLocalAddrForVf(
    uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset, LocalAddressParam<xType, wType> &localAddressParam)
{
    localAddressParam.antiQuantScaleBasePhyAddr =
        (__local_mem__ xType *)ubAntiQuantScaleTotalBuffer_[
            ((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) *antiQuantScaleUbSingleBufferSize_ +
            nWeightLowBitUbOffset].GetPhyAddr();
    localAddressParam.antiQuantScaleBasePhyAddr1 = localAddressParam.antiQuantScaleBasePhyAddr + VEC_MAX_ELEM_B16;
    localAddressParam.antiQuantOffsetBasePhyAddr =
        (__local_mem__ xType *)ubAntiQuantOffsetTotalBuffer_[
            ((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) * antiQuantScaleUbSingleBufferSize_ +
            nWeightLowBitUbOffset].GetPhyAddr();
    localAddressParam.antiQuantOffsetBasePhyAddr1 = localAddressParam.antiQuantOffsetBasePhyAddr + VEC_MAX_ELEM_B16;

    uint64_t weightLowBitOffset;
    if constexpr (wqmmConfig.bTrans) {
        weightLowBitOffset =
            ((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) * weightInputLowBitUbSingleBufferSize_ +
            nWeightLowBitUbOffset * vecConfig.ubMte2InnerSize + kWeightLowBitUbOffset;
    } else {
        weightLowBitOffset =
            ((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) * weightInputLowBitUbSingleBufferSize_ +
            kWeightLowBitUbOffset * vecConfig.ubMte2InnerSize + nWeightLowBitUbOffset;
    }


    localAddressParam.weightLowBitPhyAddr0 =
        (__local_mem__ wType *)ubWeightInputLowBitTotalBuffer_[weightLowBitOffset].GetPhyAddr();
    localAddressParam.weightLowBitPhyAddr1 = localAddressParam.weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 1);

    localAddressParam.weightF16PhyAddr0 =
        (__local_mem__ xType *)
            ubWeightOutputF16TotalBuffer_[(ubComputeLoopIdx_ & 1) * weightOutputF16UbSingleBufferSize_].GetPhyAddr();
    localAddressParam.weightF16PhyAddr1 = localAddressParam.weightF16PhyAddr0 +
        WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::AntiQuantB8CommonNdNk(
    uint64_t ubLoopN, uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset) {
    __local_mem__ xType *antiQuantScaleBasePhyAddr =
        (__local_mem__ xType *)ubAntiQuantScaleTotalBuffer_[((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) *
                                                                antiQuantScaleUbSingleBufferSize_ +
                                                            nWeightLowBitUbOffset]
            .GetPhyAddr();
    __local_mem__ xType *antiQuantOffsetBasePhyAddr =
        (__local_mem__ xType *)ubAntiQuantOffsetTotalBuffer_[((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) *
                                                                 antiQuantScaleUbSingleBufferSize_ +
                                                             nWeightLowBitUbOffset]
            .GetPhyAddr();

    uint64_t weightLowBitOffset =
        ((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) * weightInputLowBitUbSingleBufferSize_ +
        nWeightLowBitUbOffset * vecConfig.ubMte2InnerSize + kWeightLowBitUbOffset;

    __local_mem__ wType *weightLowBitPhyAddr0 =
        (__local_mem__ wType *)ubWeightInputLowBitTotalBuffer_[weightLowBitOffset].GetPhyAddr();
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 1);

    __local_mem__ xType *weightF16PhyAddr0 =
        (__local_mem__ xType *)
            ubWeightOutputF16TotalBuffer_[(ubComputeLoopIdx_ & 1) * weightOutputF16UbSingleBufferSize_]
                .GetPhyAddr();
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__ {

        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<wType> weightS8Vreg0;
        RegTensor<wType> weightS8Vreg1;
        RegTensor<half> weightFp16Vreg0;
        RegTensor<half> weightFp16Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        static constexpr MicroAPI::CastTrait castS8ToFp16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                  MicroAPI::MaskMergeMode::ZEROING,
                                                                  AscendC::RoundMode::UNKNOWN};

        static constexpr MicroAPI::CastTrait castFp16ToBf16Trait = {
            MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};

        for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < ubLoopN; ubLoopNIdx++) {
            if constexpr (wqmmConfig.hasAntiQuantOffset) {
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(antiQuantOffsetVreg,
                                                                            antiQuantOffsetBasePhyAddr + ubLoopNIdx);
            }
            MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(antiQuantScaleVreg,
                                                                        antiQuantScaleBasePhyAddr + ubLoopNIdx);
            // UNPK_B8 表示按照如下形式载入:
            // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
            // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg0, weightLowBitPhyAddr0 + ubLoopNIdx * vecConfig.ubMte2InnerSize);
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg1, weightLowBitPhyAddr1 + ubLoopNIdx * vecConfig.ubMte2InnerSize);
            // PART_EVEN 表示按照如下形式处理做cast：
            // Vn 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
            // Vd 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 .....
            if constexpr (!IsSameType<VregType, vector_f16>::value) {
                MicroAPI::Cast<half, wType, castS8ToFp16Trait>(weightFp16Vreg0, weightS8Vreg0, maskAll);
                MicroAPI::Cast<half, wType, castS8ToFp16Trait>(weightFp16Vreg1, weightS8Vreg1, maskAll);
                MicroAPI::Cast<xType, half, castFp16ToBf16Trait>(weightF16Vreg0, weightFp16Vreg0, maskAll);
                MicroAPI::Cast<xType, half, castFp16ToBf16Trait>(weightF16Vreg1, weightFp16Vreg1, maskAll);
            } else {
                MicroAPI::Cast<xType, wType, castS8ToFp16Trait>(weightF16Vreg0, weightS8Vreg0, maskAll);
                MicroAPI::Cast<xType, wType, castS8ToFp16Trait>(weightF16Vreg1, weightS8Vreg1, maskAll);
            }
            if constexpr (wqmmConfig.hasAntiQuantOffset) {
                if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
                    MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, offsetValue_, maskAll);
                    MicroAPI::Adds(weightF16Vreg1, weightF16Vreg1, offsetValue_, maskAll);
                } else {
                    MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                    MicroAPI::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg, maskAll);
                }
            }
            if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
                vmuls(weightF16Vreg0, weightF16Vreg0, scaleValue_, maskAll);
                vmuls(weightF16Vreg1, weightF16Vreg1, scaleValue_, maskAll);
            } else {
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
                MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg, maskAll);
            }
            MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
            MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                weightF16PhyAddr1, weightF16Vreg1, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::AntiQuantB8CommonNdKn(
    uint64_t ubLoopK, uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset) {

    __local_mem__ xType *antiQuantScaleBasePhyAddr =
        (__local_mem__ xType *)ubAntiQuantScaleTotalBuffer_[((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) *
                                                                antiQuantScaleUbSingleBufferSize_ +
                                                            nWeightLowBitUbOffset]
            .GetPhyAddr();
    __local_mem__ xType *antiQuantScaleBasePhyAddr1 = antiQuantScaleBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr =
        (__local_mem__ xType *)ubAntiQuantOffsetTotalBuffer_[((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) *
                                                                 antiQuantScaleUbSingleBufferSize_ +
                                                             nWeightLowBitUbOffset]
            .GetPhyAddr();
    __local_mem__ xType *antiQuantOffsetBasePhyAddr1 = antiQuantOffsetBasePhyAddr + VEC_MAX_ELEM_B16;
    uint64_t weightLowBitOffset =
        ((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) * weightInputLowBitUbSingleBufferSize_ +
        kWeightLowBitUbOffset * vecConfig.ubMte2InnerSize + nWeightLowBitUbOffset;

    __local_mem__ wType *weightLowBitPhyAddr0 =
        (__local_mem__ wType *)ubWeightInputLowBitTotalBuffer_[weightLowBitOffset].GetPhyAddr();
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 1);

    __local_mem__ xType *weightF16PhyAddr0 =
        (__local_mem__ xType *)
            ubWeightOutputF16TotalBuffer_[(ubComputeLoopIdx_ & 1) * weightOutputF16UbSingleBufferSize_]
                .GetPhyAddr();
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__ {

        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<xType> antiQuantScaleVreg1;
        RegTensor<xType> antiQuantOffsetVreg1;
        RegTensor<half> weightFp16Vreg0;
        RegTensor<half> weightFp16Vreg1;
        RegTensor<wType> weightS8Vreg0;
        RegTensor<wType> weightS8Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        static constexpr MicroAPI::CastTrait castS8ToFp16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                  MicroAPI::MaskMergeMode::ZEROING,
                                                                  AscendC::RoundMode::UNKNOWN};
        static constexpr MicroAPI::CastTrait castFp16ToBf16Trait = {
            MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        if constexpr (wqmmConfig.hasAntiQuantOffset) {
            MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr);
            MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantOffsetVreg1, antiQuantOffsetBasePhyAddr1);
        }
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg, antiQuantScaleBasePhyAddr);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg1, antiQuantScaleBasePhyAddr1);

        for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < ubLoopK; ubLoopKIdx++) {

            // UNPK_B8 表示按照如下形式载入:
            // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
            // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg0, weightLowBitPhyAddr0 + ubLoopKIdx * vecConfig.ubMte2InnerSize);
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg1, weightLowBitPhyAddr1 + ubLoopKIdx * vecConfig.ubMte2InnerSize);

            // PART_EVEN 表示按照如下形式处理做cast：
            // Vn 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
            // Vd 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 .....
            if constexpr (!IsSameType<VregType, vector_f16>::value) {
                MicroAPI::Cast<half, wType, castS8ToFp16Trait>(weightFp16Vreg0, weightS8Vreg0, maskAll);
                MicroAPI::Cast<half, wType, castS8ToFp16Trait>(weightFp16Vreg1, weightS8Vreg1, maskAll);
                MicroAPI::Cast<xType, half, castFp16ToBf16Trait>(weightF16Vreg0, weightFp16Vreg0, maskAll);
                MicroAPI::Cast<xType, half, castFp16ToBf16Trait>(weightF16Vreg1, weightFp16Vreg1, maskAll);
            } else {
                MicroAPI::Cast<xType, wType, castS8ToFp16Trait>(weightF16Vreg0, weightS8Vreg0, maskAll);
                MicroAPI::Cast<xType, wType, castS8ToFp16Trait>(weightF16Vreg1, weightS8Vreg1, maskAll);
            }

            if constexpr (wqmmConfig.hasAntiQuantOffset) {
                if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
                    MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, offsetValue_, maskAll);
                    MicroAPI::Adds(weightF16Vreg1, weightF16Vreg1, offsetValue_, maskAll);
                } else {
                    MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                    MicroAPI::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg1, maskAll);
                }
            }
            if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
                vmuls(weightF16Vreg0, weightF16Vreg0, scaleValue_, maskAll);
                vmuls(weightF16Vreg1, weightF16Vreg1, scaleValue_, maskAll);
            } else {
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
                MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg1, maskAll);
            }

            MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
            MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                weightF16PhyAddr1, weightF16Vreg1, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::AntiQuantInt4NdNk(
    uint64_t ubLoopN, uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset) {

    __local_mem__ xType *antiQuantScaleBasePhyAddr =
        (__local_mem__ xType *)ubAntiQuantScaleTotalBuffer_[((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) *
                                                                antiQuantScaleUbSingleBufferSize_ +
                                                            nWeightLowBitUbOffset]
            .GetPhyAddr();
    __local_mem__ xType *antiQuantOffsetBasePhyAddr =
        (__local_mem__ xType *)ubAntiQuantOffsetTotalBuffer_[((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) *
                                                                 antiQuantScaleUbSingleBufferSize_ +
                                                             nWeightLowBitUbOffset]
            .GetPhyAddr();

    uint64_t weightLowBitOffset =
        ((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) * weightInputLowBitUbSingleBufferSize_;

    __local_mem__ wType *weightLowBitPhyAddr0 =
        (__local_mem__ wType *)ubWeightInputLowBitTotalBuffer_[weightLowBitOffset].GetPhyAddr() +
        ((nWeightLowBitUbOffset * vecConfig.ubMte2InnerSize + kWeightLowBitUbOffset) >> 1);
    // int4每次处理128个数即为64B, 256>>2=64
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);

    __local_mem__ xType *weightF16PhyAddr0 =
        (__local_mem__ xType *)
            ubWeightOutputF16TotalBuffer_[(ubComputeLoopIdx_ & 1) * weightOutputF16UbSingleBufferSize_]
                .GetPhyAddr();
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__ {

        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<int4x2_t> weightS4Vreg0;
        RegTensor<int4x2_t> weightS4Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        static constexpr MicroAPI::CastTrait castS4ToF16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 AscendC::RoundMode::UNKNOWN};
        for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < ubLoopN; ubLoopNIdx++) {
            if constexpr (wqmmConfig.hasAntiQuantOffset) {
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(antiQuantOffsetVreg,
                                                                            antiQuantOffsetBasePhyAddr + ubLoopNIdx);
            }
            MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(antiQuantScaleVreg,
                                                                        antiQuantScaleBasePhyAddr + ubLoopNIdx);

            // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
            // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
            // Vd 0 1 x x x x x x 2 3 x x x x x x

            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg0,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + ubLoopNIdx * (vecConfig.ubMte2InnerSize >> 1)));

            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg1,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + ubLoopNIdx * (vecConfig.ubMte2InnerSize >> 1)));

            // PART_P0 表示按照如下形式处理做cast：
            // Vn 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
            // Vd 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
            MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
            MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);

            if constexpr (wqmmConfig.hasAntiQuantOffset) {
                if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
                    MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, offsetValue_, maskAll);
                    MicroAPI::Adds(weightF16Vreg1, weightF16Vreg1, offsetValue_, maskAll);
                } else {
                    MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                    MicroAPI::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg, maskAll);
                }
            }

            if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
                vmuls(weightF16Vreg0, weightF16Vreg0, scaleValue_, maskAll);
                vmuls(weightF16Vreg1, weightF16Vreg1, scaleValue_, maskAll);
            } else {
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
                MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg, maskAll);
            }

            MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
            MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                weightF16PhyAddr1, weightF16Vreg1, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::AntiQuantInt4NdKn(
    uint64_t ubLoopK, uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset) {

    __local_mem__ xType *antiQuantScaleBasePhyAddr =
        (__local_mem__ xType *)ubAntiQuantScaleTotalBuffer_[((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) *
                                                                antiQuantScaleUbSingleBufferSize_ +
                                                            nWeightLowBitUbOffset]
            .GetPhyAddr();
    __local_mem__ xType *antiQuantScaleBasePhyAddr1 = antiQuantScaleBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr =
        (__local_mem__ xType *)ubAntiQuantOffsetTotalBuffer_[((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) *
                                                                 antiQuantScaleUbSingleBufferSize_ +
                                                             nWeightLowBitUbOffset]
            .GetPhyAddr();
    __local_mem__ xType *antiQuantOffsetBasePhyAddr1 = antiQuantOffsetBasePhyAddr + VEC_MAX_ELEM_B16;
    uint64_t weightLowBitOffset =
        ((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) * weightInputLowBitUbSingleBufferSize_;

    __local_mem__ wType *weightLowBitPhyAddr0 =
        (__local_mem__ wType *)ubWeightInputLowBitTotalBuffer_[weightLowBitOffset].GetPhyAddr() +
        ((kWeightLowBitUbOffset * vecConfig.ubMte2InnerSize + nWeightLowBitUbOffset) >> 1);
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);

    __local_mem__ xType *weightF16PhyAddr0 =
        (__local_mem__ xType *)
            ubWeightOutputF16TotalBuffer_[(ubComputeLoopIdx_ & 1) * weightOutputF16UbSingleBufferSize_]
                .GetPhyAddr();
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__ {

        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<xType> antiQuantScaleVreg1;
        RegTensor<xType> antiQuantOffsetVreg1;
        RegTensor<int4x2_t> weightS4Vreg0;
        RegTensor<int4x2_t> weightS4Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        static constexpr MicroAPI::CastTrait castS4ToF16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 AscendC::RoundMode::UNKNOWN};

        if constexpr (wqmmConfig.hasAntiQuantOffset) {
            MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr);
            MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantOffsetVreg1, antiQuantOffsetBasePhyAddr1);
        }
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg, antiQuantScaleBasePhyAddr);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg1, antiQuantScaleBasePhyAddr1);

        for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < ubLoopK; ubLoopKIdx++) {

            // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
            // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
            // Vd 0 1 x x x x x x 2 3 x x x x x x
            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg0,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + ubLoopKIdx * (vecConfig.ubMte2InnerSize >> 1)));

            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg1,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + ubLoopKIdx * (vecConfig.ubMte2InnerSize >> 1)));

            // PART_P0 表示按照如下形式处理做cast：
            // Vn 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
            // Vd 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
            MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
            MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);
            if constexpr (wqmmConfig.hasAntiQuantOffset) {
                if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
                    MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, offsetValue_, maskAll);
                    MicroAPI::Adds(weightF16Vreg1, weightF16Vreg1, offsetValue_, maskAll);
                } else {
                    MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                    MicroAPI::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg1, maskAll);
                }
            }
            if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
                vmuls(weightF16Vreg0, weightF16Vreg0, scaleValue_, maskAll);
                vmuls(weightF16Vreg1, weightF16Vreg1, scaleValue_, maskAll);
            } else {
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
                MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg1, maskAll);
            }

            MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
            MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                weightF16PhyAddr1, weightF16Vreg1, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::WeightF16UbToL1(
    uint64_t weightF16L1Offset, uint64_t antiQuantRealN, uint64_t antiQuantRealK, const LocalTensor<xType> &weightF16L1,
    uint64_t l1RealExternalLen) {
    DataCopyParams params;

    if constexpr (wqmmConfig.bTrans) {
        params.blockLen = antiQuantRealN;
        params.blockCount = CeilDiv(antiQuantRealK, (uint64_t)BLOCK_CUBE);
        params.srcStride = WEIGHT_F16_UB_NZ_STRIDE - antiQuantRealN;
        params.dstStride = CeilAlign(l1RealExternalLen, (uint64_t)BLOCK_CUBE) - antiQuantRealN;
    } else {
        params.blockLen = antiQuantRealK;
        params.blockCount = CeilDiv(antiQuantRealN, (uint64_t)BLOCK_CUBE);
        params.srcStride = WEIGHT_F16_UB_NZ_STRIDE - antiQuantRealK;
        params.dstStride = CeilAlign(l1RealExternalLen, (uint64_t)BLOCK_CUBE) - antiQuantRealK;
    }

    DataCopy(weightF16L1[weightF16L1Offset],
             ubWeightOutputF16TotalBuffer_[(ubComputeLoopIdx_ & 1) * weightOutputF16UbSingleBufferSize_], params);
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void BasicBlockLibVectorAntiQuantCompute<xType, wType, wqmmConfig, vecConfig>::End() {

    TEventID vecEventIdVToMte2[QUADRUPLE_BUFFER_NUM] = {vecEventIdVToMte2_[0], vecEventIdVToMte2_[1],
                                                        vecEventIdVToMte2_[2], vecEventIdVToMte2_[3]};
    TEventID vecEventIdMte3ToV[DOUBLE_BUFFER_NUM] = {vecEventIdMte3ToV_[0], vecEventIdMte3ToV_[1]};

    for (int32_t idx = 0; idx < ubComputeLoopIdx_ && idx < DOUBLE_BUFFER_NUM; idx++) {
        WaitFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV[idx]);
    }

    for (int32_t idx = 0; idx < ubMte2LoopIdx_ && idx < vecConfig.ubMte2BufferNum; idx++) {
        WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2[idx]);
    }

    for (int32_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(vecEventIdMte3ToV[i]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vecEventIdVToMte2[i]);
    }

    if (vecConfig.ubMte2BufferNum == QUADRUPLE_BUFFER_NUM) {
        for (int32_t i = 2; i < QUADRUPLE_BUFFER_NUM; i++) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vecEventIdVToMte2[i]);
        }
    }
}

} // namespace WeightQuantBatchMatmulV2

#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_VEC_COMPUTE_H
