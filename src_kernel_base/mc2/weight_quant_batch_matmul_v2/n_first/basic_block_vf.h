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
 * \file basic_block_vf.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_VF_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_VF_H

#include "basic_block_config.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "../tool.h"
#include "../weight_quant_batch_matmul_v2_constant.h"

namespace MicroAPI = AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

namespace WeightQuantBatchMatmulV2 {

template <typename xType, typename wType>
struct LocalAddressParam {
    __local_mem__ xType *antiQuantScaleBasePhyAddr;
    __local_mem__ xType *antiQuantScaleBasePhyAddr1;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr1;
    __local_mem__ wType *weightLowBitPhyAddr0;
    __local_mem__ wType *weightLowBitPhyAddr1;
    __local_mem__ xType *weightF16PhyAddr0;
    __local_mem__ xType *weightF16PhyAddr1;
};

template <typename xType>
struct CalculateParam {
    xType offsetValue;
    xType scaleValue;
    uint64_t ubLoop;
};

static constexpr MicroAPI::CastTrait FP8_TO_FP32_TRAIT_0 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr MicroAPI::CastTrait FP8_TO_FP32_TRAIT_2 = {
    MicroAPI::RegLayout::TWO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

static constexpr MicroAPI::CastTrait FP32_TO_F16_ODD = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};
static constexpr MicroAPI::CastTrait FP32_TO_F16_EVEN = {
    MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize, QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdNkVf(LocalAddressParam<xType, wType> &localAddressParam,
                                          const CalculateParam<xType> &calculateParam)
{
    RegTensor<xType> antiQuantScaleVreg, antiQuantOffsetVreg;
    RegTensor<wType> weightF8Vreg0, weightF8Vreg1;
    RegTensor<xType> weightF16Vreg0, weightF16Vreg1, weightF16Vreg2, weightF16Vreg3;
    RegTensor<float> weightF32Vreg0, weightF32Vreg1, weightF32Vreg2, weightF32Vreg3;
    MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

    for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < calculateParam.ubLoop; ubLoopNIdx++) {
        if constexpr (hasAntiQuantOffset) {
            MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                antiQuantOffsetVreg, localAddressParam.antiQuantOffsetBasePhyAddr + ubLoopNIdx);
        }
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
            antiQuantScaleVreg, localAddressParam.antiQuantScaleBasePhyAddr + ubLoopNIdx);
        // UNPK_B8 表示按照如下形式载入:
        // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
        // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
        MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
            weightF8Vreg0, localAddressParam.weightLowBitPhyAddr0 + ubLoopNIdx * ubMte2InnerSize);
        MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
            weightF8Vreg1, localAddressParam.weightLowBitPhyAddr1 + ubLoopNIdx * ubMte2InnerSize);
        // 奇数、偶数位置分散到2个fp32寄存器存储
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg0, weightF8Vreg0, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg2, weightF8Vreg1, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg1, weightF8Vreg0, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg3, weightF8Vreg1, maskAll);

        MicroAPI::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg0, weightF32Vreg0, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg2, weightF32Vreg2, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg1, weightF32Vreg1, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg3, weightF32Vreg3, maskAll);

        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>(
            (RegTensor<uint16_t>&)weightF16Vreg2, (RegTensor<uint16_t>&)weightF16Vreg2,
            (RegTensor<uint16_t>&)weightF16Vreg3, maskAll);
        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>(
            (RegTensor<uint16_t>&)weightF16Vreg0, (RegTensor<uint16_t>&)weightF16Vreg0,
            (RegTensor<uint16_t>&)weightF16Vreg1, maskAll);

        if constexpr (hasAntiQuantOffset) {
            if constexpr (antiQuantType == QuantType::PER_TENSOR) {
                MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, calculateParam.offsetValue, maskAll);
                MicroAPI::Adds(weightF16Vreg2, weightF16Vreg2, calculateParam.offsetValue, maskAll);
            } else {
                MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                MicroAPI::Add(weightF16Vreg2, weightF16Vreg2, antiQuantOffsetVreg, maskAll);
            }
        }
        if constexpr (antiQuantType == QuantType::PER_TENSOR) {
            vmuls(weightF16Vreg0, weightF16Vreg0, calculateParam.scaleValue, maskAll);
            vmuls(weightF16Vreg2, weightF16Vreg2, calculateParam.scaleValue, maskAll);
        } else {
            MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
            MicroAPI::Mul(weightF16Vreg2, weightF16Vreg2, antiQuantScaleVreg, maskAll);
        }
        MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            localAddressParam.weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
        MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            localAddressParam.weightF16PhyAddr1, weightF16Vreg2, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize, QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdKnVf(LocalAddressParam<xType, wType> &localAddressParam,
                                          const CalculateParam<xType> &calculateParam)
{
    RegTensor<xType> antiQuantScaleVreg0;
    RegTensor<xType> antiQuantScaleVreg1;
    RegTensor<xType> antiQuantOffsetVreg0;
    RegTensor<xType> antiQuantOffsetVreg1;
    RegTensor<wType> weightF8Vreg0;
    RegTensor<wType> weightF8Vreg1;
    RegTensor<xType> weightF16Vreg0;
    RegTensor<xType> weightF16Vreg1;
    RegTensor<xType> weightF16Vreg2;
    RegTensor<xType> weightF16Vreg3;
    RegTensor<float> weightF32Vreg0;
    RegTensor<float> weightF32Vreg1;
    RegTensor<float> weightF32Vreg2;
    RegTensor<float> weightF32Vreg3;
    MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

    if constexpr (hasAntiQuantOffset) {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(
            antiQuantOffsetVreg0, localAddressParam.antiQuantOffsetBasePhyAddr);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(
            antiQuantOffsetVreg1, localAddressParam.antiQuantOffsetBasePhyAddr1);
    }
    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(
        antiQuantScaleVreg0, localAddressParam.antiQuantScaleBasePhyAddr);
    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(
        antiQuantScaleVreg1, localAddressParam.antiQuantScaleBasePhyAddr1);

    for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < calculateParam.ubLoop; ubLoopKIdx++) {
        // UNPK_B8 表示按照如下形式载入:
        // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
        // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
        MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
            weightF8Vreg0, localAddressParam.weightLowBitPhyAddr0 + ubLoopKIdx * ubMte2InnerSize);
        MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
            weightF8Vreg1, localAddressParam.weightLowBitPhyAddr1 + ubLoopKIdx * ubMte2InnerSize);
        // 奇数、偶数位置分散到2个fp32寄存器存储
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg0, weightF8Vreg0, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg1, weightF8Vreg0, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg2, weightF8Vreg1, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg3, weightF8Vreg1, maskAll);

        MicroAPI::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg0, weightF32Vreg0, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg1, weightF32Vreg1, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg2, weightF32Vreg2, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg3, weightF32Vreg3, maskAll);

        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>(
            (RegTensor<uint16_t>&)weightF16Vreg0, (RegTensor<uint16_t>&)weightF16Vreg0,
            (RegTensor<uint16_t>&)weightF16Vreg1, maskAll);
        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>(
            (RegTensor<uint16_t>&)weightF16Vreg2, (RegTensor<uint16_t>&)weightF16Vreg2,
            (RegTensor<uint16_t>&)weightF16Vreg3, maskAll);

        if constexpr (hasAntiQuantOffset) {
            if constexpr (antiQuantType == QuantType::PER_TENSOR) {
                MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, calculateParam.offsetValue, maskAll);
                MicroAPI::Adds(weightF16Vreg2, weightF16Vreg2, calculateParam.offsetValue, maskAll);
            } else {
                MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg0, maskAll);
                MicroAPI::Add(weightF16Vreg2, weightF16Vreg2, antiQuantOffsetVreg1, maskAll);
            }
        }
        if constexpr (antiQuantType == QuantType::PER_TENSOR) {
            vmuls(weightF16Vreg0, weightF16Vreg0, calculateParam.scaleValue, maskAll);
            vmuls(weightF16Vreg2, weightF16Vreg2, calculateParam.scaleValue, maskAll);
        } else {
            MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg0, maskAll);
            MicroAPI::Mul(weightF16Vreg2, weightF16Vreg2, antiQuantScaleVreg1, maskAll);
        }
        MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            localAddressParam.weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
        MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            localAddressParam.weightF16PhyAddr1, weightF16Vreg2, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
    }
}

} // namespace WeightQuantBatchMatmulV2
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_VF_H