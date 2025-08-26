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
 * \file simd.h
 * \brief
 */
#ifndef INCLUDE_SIMD_H
#define INCLUDE_SIMD_H

#include "hardware.h"
#include "kernel_operator.h"


/////////////////////////////////////////////////////
// vconv
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DTypeIn, typename DTypeOut>
__aicore__ inline void conv_v(AscendC::LocalTensor<DTypeOut> dst, AscendC::LocalTensor<DTypeIn> src, uint8_t repeat,
                              uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                              uint16_t srcRepeatStride)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if constexpr (std::is_same<DTypeIn, float>::value && std::is_same<DTypeOut, bfloat16_t>::value) {
        AscendC::Cast<DTypeOut, DTypeIn, false>(
            dst, src, AscendC::RoundMode::CAST_RINT, (uint64_t)0, repeat,
            AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
    } else {
        AscendC::Cast<DTypeOut, DTypeIn, false>(
            dst, src, AscendC::RoundMode::CAST_NONE, (uint64_t)0, repeat,
            AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
    }
#else
    AscendC::Cast<DTypeOut, DTypeIn, false>(
        dst, src, AscendC::RoundMode::CAST_NONE, (uint64_t)0, repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
#endif
}

/////////////////////////////////////////////////////
// vconv_f322bf16r
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DTypeIn, typename DTypeOut>
__aicore__ inline void convr_v(AscendC::LocalTensor<DTypeOut> dst, AscendC::LocalTensor<DTypeIn> src, uint8_t repeat,
                               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                               uint16_t srcRepeatStride)
{
    AscendC::Cast<DTypeOut, DTypeIn, false>(
        dst, src, AscendC::RoundMode::CAST_RINT, (uint64_t)0, repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}

#endif