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
 * \file matmul_reduce_scatter_nd_to_nz.h
 * \brief
 */
#ifndef MATMUL_REDUCE_SCATTER_ND_TO_NZ_H
#define MATMUL_REDUCE_SCATTER_ND_TO_NZ_H

#if defined(__CCE_KT_TEST__)
#define SET_G_CORE_TYPE_IS_AIV thread_local int g_coreType = 2
#define SET_G_CORE_TYPE_IS_AIC thread_local int g_coreType = 1
#define DTYPE_X1 half
#define DTYPE_Y half
#else
#define SET_G_CORE_TYPE_IS_AIV
#define SET_G_CORE_TYPE_IS_AIC
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "../common/mc2_tiling_struct.h"
#include "../common/mat_mul_nd2nz.h"

namespace AscendC {
using namespace matmul;

template <class T>
__aicore__ inline void MatrixBtoNZMc2(GM_ADDR workspace, GM_ADDR src, const MatmulReduceScatterTilingData* tilingData,
                                      bool isTransposeB, TBuf<TPosition::VECCALC> &tmpBuf)
{
    if (g_coreType == AIV) {
        if (block_idx >= tilingData->tileTiling.usedCoreNum) {
            // 未使用的AIV核同步等待
            ffts_cross_core_sync(PIPE_MTE3, 0x21 + (3 << 8));
            return;
        }
        MatrixBtoNZV2<T>(workspace, src, tilingData->tileTiling, isTransposeB, tmpBuf, tilingData->socParam.baseBN, tilingData->socParam.baseBD);
        // 先AIC等待AIV, 再AIC之间一次同步
        ffts_cross_core_sync(PIPE_MTE3, 0x21 + (3 << 8));  // v侧做完才能做c侧
    } else {
#ifndef __CCE_KT_TEST__
        wait_flag_dev(3);
        ffts_cross_core_sync(PIPE_MTE3, 0x01 + (4 << 8));
        wait_flag_dev(4);
#endif
    }
}
#endif // REDUCE_SCATTER_ND_TO_NZ_H
}