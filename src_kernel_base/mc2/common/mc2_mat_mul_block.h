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
 * \file mc2_mat_mul_block.h
 * \brief
 */
#ifndef MC2_MAT_MUL_BLOCK_H
#define MC2_MAT_MUL_BLOCK_H

#include "../mat_mul_v3/mat_mul_base_block.h"

#define DEVICE_NUM 32  // group内卡数，目前定义为32，后续根据情况扩展

using namespace AscendC;

class MC2MatmulBaseBlock : public MatmulBaseBlock {
public:
    __aicore__ inline MC2MatmulBaseBlock () {}
    __aicore__ inline void InitForMC2(RCSTiling &cfg, bool isTail, bool isGather = false);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset(uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void UpdateOffset(uint32_t idx, bool isTail);
public:
    RCSTiling cfg_;
private:
    uint64_t offsetsA_[DEVICE_NUM] = {0};
    uint64_t offsetsC_[DEVICE_NUM] = {0};
    uint64_t curSliceM_{0};
    uint64_t headSliceM_{0};
    uint64_t rankM_{0};
    uint64_t rankN_{0};
    uint64_t rankK_{0};
    bool isGather_{false};
};

__aicore__ inline void MC2MatmulBaseBlock::InitForMC2(RCSTiling &cfg, bool isTail, bool isGather)
{
    cfg_ = cfg;
    rankM_ = isGather ? cfg.rankM : cfg.rankM / cfg.rankDim;
    rankN_ = cfg.rankN;
    rankK_ = cfg.rankK;
    headSliceM_ = (rankM_ - cfg.tailM * cfg.tailCnt) / cfg.tileCnt;
    isGather_ = isGather;
    curSliceM_ = isTail ? cfg.tailM : headSliceM_;

    uint64_t calRankNum = isGather ? (cfg.rankDim - 1) : cfg.rankDim;
    params_.mTotalCnt = DivCeil(curSliceM_, params_.blockBaseM) * calRankNum;
    params_.mBaseTail = curSliceM_ % params_.blockBaseM;
    params_.mCnt = DivCeil(params_.mTotalCnt, params_.mTileCntL2);
    params_.totalTileCnt = params_.mCnt * params_.nCnt;
    params_.mCntTail = params_.mTotalCnt - (params_.mTileCntL2 - 1) * params_.mCnt;
    params_.round = DivCeil(params_.totalTileCnt, matmulTilingData_->matmulTiling.usedCoreNum);
    params_.preCoreNum = params_.totalTileCnt % matmulTilingData_->matmulTiling.usedCoreNum;
    params_.mCntUse = params_.mCnt;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MC2MatmulBaseBlock::CalcGMOffset(uint64_t mTileIndex, uint64_t nTileIndex)
{
    uint64_t mCntIndex = params_.index / params_.nCntUse;
    uint64_t nCntIndex = params_.index % params_.nCntUse;
    uint64_t globalBlockMIndex = mTileIndex * params_.mCnt + mCntIndex;
    uint64_t mc2MCnt = DivCeil(curSliceM_, params_.blockBaseM);
    uint64_t mc2MIdx = globalBlockMIndex / mc2MCnt;
    uint64_t mc2MRest = globalBlockMIndex % mc2MCnt;

    if constexpr (A_TYPE::format == CubeFormat::ND) {
        if (!params_.isTransposeA) {
            offset_.offsetA = offsetsA_[mc2MIdx] + mc2MRest * params_.blockBaseM * rankK_;
        }
    }
    params_.singleCoreM = (mc2MRest == mc2MCnt - 1) ? (curSliceM_ - (mc2MCnt - 1) * params_.blockBaseM) :
        params_.blockBaseM;

    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if (params_.isTransposeB) {
            offset_.offsetB = nCntIndex * params_.blockBaseN * rankK_ + params_.nTileAddrOffset * rankK_;
        } else {
            offset_.offsetB = nCntIndex * params_.blockBaseN + params_.nTileAddrOffset;
        }
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        if (params_.isTransposeB) {
            offset_.offsetB = nCntIndex * params_.blockBaseN * params_.c0Size +
                params_.nTileAddrOffset * params_.c0Size;
        } else {
            offset_.offsetB = nCntIndex * params_.blockBaseN * params_.alignedKbSize +
                params_.nTileAddrOffset * params_.alignedKbSize;
        }
    }
    if constexpr (C_TYPE::format == CubeFormat::ND) {
        offset_.offsetC = offsetsC_[mc2MIdx] + mc2MRest * params_.blockBaseM * rankN_ +
            nCntIndex * params_.blockBaseN + params_.nTileAddrOffset;
    }
    if (matmulTilingData_->matmulTiling.isBias) {
        offset_.offsetBias = nCntIndex * params_.blockBaseN + params_.nTileAddrOffset;
    }
}

__aicore__ inline void MC2MatmulBaseBlock::UpdateOffset(uint32_t idx, bool isTail)
{
    uint64_t offsetHeadSliceM = 0;
    uint64_t offsetTailSliceM = 0;
    if (isTail) {
        offsetHeadSliceM = cfg_.tileCnt * headSliceM_;
        offsetTailSliceM = idx * cfg_.tailM;
    } else {
        offsetHeadSliceM = idx * headSliceM_;
    }
    if (cfg_.rankDim > DEVICE_NUM) {
        return;
    }
    for (uint8_t i = 0; i < DEVICE_NUM; i++) {
        offsetsA_[i] = 0;
    }
    uint8_t cnt = 0;
    for (uint32_t i = 0; i < cfg_.rankDim; i++) {
        if (isGather_ && i == cfg_.rankID) {
            continue;
        }
        offsetsA_[cnt] = i * rankM_ * rankK_ + (offsetHeadSliceM + offsetTailSliceM) * rankK_;
        cnt++;
    }
    for (uint8_t i = 0; i < DEVICE_NUM; i++) {
        offsetsC_[i] = 0;
    }
    cnt = 0;
    for (uint32_t i = 0; i < cfg_.rankDim; i++) {
        if (isGather_ && i == cfg_.rankID) {
            continue;
        }
        offsetsC_[cnt] = i * rankM_ * rankN_ + (offsetHeadSliceM + offsetTailSliceM) * rankN_;
        cnt++;
    }
}
#endif