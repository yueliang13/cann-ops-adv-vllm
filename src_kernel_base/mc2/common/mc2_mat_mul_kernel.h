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
 * \file mc2_mat_mul_kernel.h
 * \brief
 */
#ifndef MC2_MAT_MUL_KERNEL_H
#define MC2_MAT_MUL_KERNEL_H

#include "../mat_mul_v3/mat_mul_base_kernel.h"
#include "mc2_mat_mul_block.h"

using namespace AscendC;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulBaseBlock,
    const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>>
class MC2MatmulBaseKernel : public MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
    MM_CB> {
public:
    __aicore__ inline MC2MatmulBaseKernel() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
        GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe, RCSTiling cfg, bool isTail, bool isGather = false);
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, bool isGather = false);
    __aicore__ inline void SetOrgShape();
    __aicore__ inline void UpdateSlice(uint32_t idx, bool isTail);
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
    class MM_CB>
__aicore__ inline void MC2MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::InitInputs(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, bool isGather)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    uint32_t adjustVal = isGather ? this->block_.cfg_.rankDim : 1;
    this->aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
        static_cast<uint64_t>(this->block_.cfg_.rankM * this->block_.cfg_.rankK * adjustVal));
    this->bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
        static_cast<uint64_t>(this->block_.cfg_.rankK * this->block_.cfg_.rankN));
    this->cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
        static_cast<uint64_t>(this->block_.cfg_.rankM * this->block_.cfg_.rankN * adjustVal));
    this->biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), this->block_.cfg_.rankN);
    SetL2CacheEnable(this->block_.matmulTilingData_->l2cacheUseInfo, this->aGlobal_, this->bGlobal_,
        this->cGlobal_, this->biasGlobal_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
    class MM_CB>
__aicore__ inline void MC2MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::SetOrgShape()
{
    if constexpr (A_TYPE::format == CubeFormat::ND && B_TYPE::format == CubeFormat::NZ) {
        this->mm_.SetOrgShape(this->block_.matmulTilingData_->matmulTiling.M, this->block_.params_.alignedOriN,
            this->block_.matmulTilingData_->matmulTiling.singleCoreK, this->block_.params_.alignedKbSize,
            this->block_.matmulTilingData_->matmulTiling.N);
    } else if constexpr (A_TYPE::format == CubeFormat::ND && B_TYPE::format == CubeFormat::ND) {
        this->mm_.SetOrgShape(this->block_.matmulTilingData_->matmulTiling.M,
            this->block_.matmulTilingData_->matmulTiling.N,
            this->block_.matmulTilingData_->matmulTiling.singleCoreK,
            this->block_.matmulTilingData_->matmulTiling.singleCoreK,
            this->block_.matmulTilingData_->matmulTiling.N);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
    class MM_CB>
__aicore__ inline void MC2MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe, RCSTiling cfg, bool isTail, bool isGather)
{
    this->block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    this->block_.InitForMC2(cfg, isTail, isGather);
    this->pipe_ = pipe;
    InitInputs(aGM, bGM, cGM, biasGM, isGather);
    this->mm_.SetSubBlockIdx(0);
    this->mm_.Init(&this->block_.matmulTilingData_->matmulTiling, this->pipe_);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    this->pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);
    LocalTensor<uint8_t> buf = ubBuf_.template Get<uint8_t>();
    this->mm_.SetLocalWorkspace(buf);
#endif
    SetOrgShape();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
    class MM_CB>
__aicore__ inline void MC2MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::UpdateSlice(
    uint32_t idx, bool isTail)
{
    this->block_.UpdateOffset(idx, isTail);
}
#endif