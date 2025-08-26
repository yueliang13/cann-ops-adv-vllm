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
 * \file mat_mul_asw_block.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_ASW_BLOCK_H__
#define __OP_KERNEL_MATMUL_V3_ASW_BLOCK_H__

namespace MatmulV3 {

using namespace AscendC;
using namespace matmul;

struct AswBlockOffset {
    uint64_t offsetA = 0;
    uint64_t offsetB = 0;
    uint64_t offsetC = 0;
    uint64_t offsetBias = 0;
};

struct AswBlockArgs {
    uint64_t index;
    uint64_t mCntIndex;
    uint64_t nCntIndex;
    uint64_t mCnt;
    uint64_t nCnt;
    uint64_t totalCnt;
    uint64_t blockBaseM;
    uint64_t blockBaseN;
    uint64_t nBaseTail;
    uint64_t mBaseTail;
    uint64_t mBaseSplitCnt;
    uint64_t nBaseSplitCnt;
    uint64_t totalSplitCnt;
    uint64_t mSplitAddrOffset;
    uint64_t nSplitAddrOffset;
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t round;
    uint64_t mainRow;
    uint64_t mainWindow;
    uint64_t tailWindow;
};


class MatmulAswBlock {
public:
    __aicore__ inline MatmulAswBlock() {}
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void *tilingData);
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx);
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset();

public:
    AswBlockOffset offset_;
    AswBlockArgs params_;
    const MatmulTilingData *matmulTilingData_;

private:
    const uint64_t WINDOW_LEN = 4;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulAswBlock::Init(const void *tilingData)
{
    matmulTilingData_ = static_cast<const MatmulTilingData *>(tilingData);
    const L2cacheTilePara &tilingL2 = matmulTilingData_->tileL2cacheTiling;

    params_.index = 0;
    params_.singleCoreM = 0;
    params_.singleCoreN = 0;
    params_.mSplitAddrOffset = 0;
    params_.nSplitAddrOffset = 0;
    params_.blockBaseM = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseM);
    params_.blockBaseN = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseN);
    params_.mCnt = (matmulTilingData_->matmulTiling.M + params_.blockBaseM - 1) / params_.blockBaseM; // 总的m方向base块个数
    params_.nCnt = (matmulTilingData_->matmulTiling.N + params_.blockBaseN - 1) / params_.blockBaseN; // 总的n方向base块个数
    params_.totalCnt = params_.mCnt * params_.nCnt;
    params_.nBaseTail = matmulTilingData_->matmulTiling.N - (params_.nCnt - 1) * params_.blockBaseN; // n方向上的base尾块
    params_.mBaseTail = matmulTilingData_->matmulTiling.M - (params_.mCnt - 1) * params_.blockBaseM; // m方向上的base尾块
    params_.round = (params_.totalCnt + matmulTilingData_->matmulTiling.usedCoreNum - 1) /
        matmulTilingData_->matmulTiling.usedCoreNum;
    params_.mainWindow = WINDOW_LEN < params_.mCnt ? WINDOW_LEN : params_.mCnt;
    params_.mainRow = params_.mCnt / params_.mainWindow - 1;
    params_.tailWindow = params_.mCnt - params_.mainRow * params_.mainWindow;

    params_.mBaseSplitCnt = tilingL2.mTileCntL2;
    params_.nBaseSplitCnt = tilingL2.nTileCntL2;
    params_.totalSplitCnt = params_.mBaseSplitCnt * params_.nBaseSplitCnt;
}

__aicore__ inline void MatmulAswBlock::UpdateBasicIndex(uint64_t roundIdx)
{
    uint64_t newBlockIdx = (roundIdx == params_.round - 1) ? (block_idx / params_.totalSplitCnt) : block_idx;
    params_.index = newBlockIdx + roundIdx * matmulTilingData_->matmulTiling.usedCoreNum;
    uint64_t rowIdx = params_.index / params_.nCnt / params_.mainWindow;
    if (rowIdx < params_.mainRow) {
        params_.mCntIndex = rowIdx * params_.mainWindow + params_.index % params_.mainWindow;
        params_.nCntIndex = (params_.index / params_.mainWindow) % params_.nCnt;
    } else {
        rowIdx = params_.mainRow;
        uint64_t tailIndex = params_.index - params_.mainRow * params_.mainWindow * params_.nCnt;
        params_.mCntIndex = params_.mainRow * params_.mainWindow + tailIndex % params_.tailWindow;
        params_.nCntIndex = (tailIndex / params_.tailWindow) % params_.nCnt;
    }
    // mod 2 means even row, need reverse scan
    if (rowIdx % 2 != 0) {
        params_.nCntIndex = params_.nCnt - 1 - params_.nCntIndex;
    }
}

__aicore__ inline void MatmulAswBlock::UpdateBlockParams(uint64_t roundIdx)
{
    params_.singleCoreM = params_.mCntIndex != (params_.mCnt - 1) ? params_.blockBaseM : params_.mBaseTail;
    params_.singleCoreN = params_.nCntIndex != (params_.nCnt - 1) ? params_.blockBaseN : params_.nBaseTail;

    if (roundIdx == params_.round - 1 && (params_.mBaseSplitCnt != 1 || params_.nBaseSplitCnt != 1)) {
        uint64_t singleCoreMSplit = (params_.singleCoreM + params_.mBaseSplitCnt - 1) / params_.mBaseSplitCnt;
        uint64_t singleCoreNSplit = (params_.singleCoreN + params_.nBaseSplitCnt - 1) / params_.nBaseSplitCnt;
        uint64_t mSplitIdx = (block_idx % params_.totalSplitCnt) % params_.mBaseSplitCnt;
        uint64_t nSplitIdx = (block_idx % params_.totalSplitCnt) / params_.mBaseSplitCnt;
        params_.mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        params_.nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (params_.mSplitAddrOffset >= params_.singleCoreM || params_.nSplitAddrOffset >= params_.singleCoreN) {
            params_.singleCoreM = 0;
            params_.singleCoreN = 0;
            return;
        }
        if (mSplitIdx + 1 == params_.mBaseSplitCnt) {
            params_.singleCoreM = params_.singleCoreM - singleCoreMSplit * mSplitIdx;
        } else {
            params_.singleCoreM = singleCoreMSplit;
        }
        if (nSplitIdx + 1 == params_.nBaseSplitCnt) {
            params_.singleCoreN = params_.singleCoreN - singleCoreNSplit * nSplitIdx;
        } else {
            params_.singleCoreN = singleCoreNSplit;
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulAswBlock::CalcGMOffset()
{
    if (matmulTilingData_->matmulRunInfo.transA) {
        offset_.offsetA = params_.mCntIndex * params_.blockBaseM + params_.mSplitAddrOffset;
    } else {
        offset_.offsetA = (params_.mCntIndex * params_.blockBaseM + params_.mSplitAddrOffset) *
            matmulTilingData_->matmulTiling.Ka;
    }
    if (matmulTilingData_->matmulRunInfo.transB) {
        offset_.offsetB = (params_.nCntIndex * params_.blockBaseN + params_.nSplitAddrOffset) *
            matmulTilingData_->matmulTiling.Kb;
    } else {
        offset_.offsetB = params_.nCntIndex * params_.blockBaseN + params_.nSplitAddrOffset;
    }
    offset_.offsetC = (params_.nCntIndex * params_.blockBaseN + params_.nSplitAddrOffset) +
        (params_.mCntIndex * params_.blockBaseM + params_.mSplitAddrOffset) * matmulTilingData_->matmulTiling.N;
    if (matmulTilingData_->matmulTiling.isBias) {
        offset_.offsetBias = params_.nCntIndex * params_.blockBaseN + params_.nSplitAddrOffset;
    }
}

} // namespace MatmulV3

#endif // MMV3_MATMUL_ASW_BLOCK_H
