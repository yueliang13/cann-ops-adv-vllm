/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mat_mul_asw_kernel.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_ASW_KERNEL_H__
#define __OP_KERNEL_MATMUL_V3_ASW_KERNEL_H__

#include "mat_mul_asw_block.h"

namespace MatmulV3 {

using namespace AscendC;
using namespace matmul;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulAswBlock,
    const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class MatmulAswKernel {
public:
    __aicore__ inline MatmulAswKernel() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM);
    __aicore__ inline void Process(uint8_t enAtomic = 0);
    __aicore__ inline void End() { mm_.End(); }
    __aicore__ inline const BLOCK_TYPE GetBlock() { return block_; }

protected:
    BLOCK_TYPE block_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;

private:
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM);
};


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(GM_ADDR aGM,
    GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM, const void *tilingData,
    TPipe *pipe)
{
    pipe_ = pipe;
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    InitInputs(aGM, bGM, cGM, biasGM);
    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.matmulTilingData_->matmulTiling, pipe_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::InitInputs(GM_ADDR aGM,
    GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM)
{
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
        static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.M) * block_.matmulTilingData_->matmulTiling.Ka);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
        static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.Kb) * block_.matmulTilingData_->matmulTiling.N);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
        static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.M) * block_.matmulTilingData_->matmulTiling.N);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), block_.matmulTilingData_->matmulTiling.N);
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::UpdateGlobalTensor(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    InitInputs(aGM, bGM, cGM, biasGM);
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }

    SetAtomicNone();
    for (uint64_t j = 0; j < block_.params_.round; j++) {
        block_.UpdateBasicIndex(j); // 使能错位分核更新Index
        if (block_.params_.index < block_.params_.totalCnt) {
            block_.UpdateBlockParams(j);
            if (block_.params_.singleCoreM > 0 && block_.params_.singleCoreN > 0) {
                block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();

                mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                    block_.matmulTilingData_->matmulTiling.singleCoreK);
                mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], block_.matmulTilingData_->matmulRunInfo.transA);
                mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], block_.matmulTilingData_->matmulRunInfo.transB);
                if (block_.matmulTilingData_->matmulTiling.isBias) {
                    mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
                }
                if (block_.matmulTilingData_->matmulRunInfo.isHf32) {
                    mm_.SetHF32(true, 1);
                    mm_.Iterate();
                    mm_.SetHF32(false, 0);
                } else {
                    mm_.Iterate();
                }
                mm_.GetTensorC(cGlobal_[block_.offset_.offsetC], enAtomic);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
}

} // namespace MatmulV3

#endif // MMV3_MATMUL_ASW_KERNEL_H
