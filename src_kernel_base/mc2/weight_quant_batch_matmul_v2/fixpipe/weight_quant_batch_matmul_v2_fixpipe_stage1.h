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
 * \file weight_quant_batch_matmul_v2_fixpipe_stage1.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_FIXPIPE_STAGE1_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_FIXPIPE_STAGE1_H
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "../tool.h"

namespace WeightQuantBatchMatmulV2 {

using AscendC::DataCopyParams;
using AscendC::MmadParams;
using AscendC::LoadData2dTransposeParams;

template <bool hasAntiqOffset>
class WeightQuantBatchMatmulV2FixpipeStage1 {
public:
    __aicore__ inline WeightQuantBatchMatmulV2FixpipeStage1() {}

    __aicore__ inline void Init(const LocalTensor<int8_t>& weightS8L0a,
                                const LocalTensor<int8_t>& diagS8L0b,
                                const LocalTensor<int32_t>& antiqOffsetBT,
                                const LocalTensor<uint64_t>& antiqScaleFP,
                                const LocalTensor<int32_t>& weightS32L0c) {
        // L0A DB
        weightS8L0a_ = weightS8L0a;
        diagS8L0b_ = diagS8L0b;
        if constexpr (hasAntiqOffset) {
            antiqOffsetBT_ = antiqOffsetBT;
        }
        antiqScaleFP_ = antiqScaleFP;
        // L0C DB
        weightS32L0c_ = weightS32L0c;

        antiqOffsetParams_.blockCount = 1;
        antiqScaleParams_.blockCount = 1;
    }

    __aicore__ inline void SetOriShape(const uint32_t nOriSize) {
        weightParams_.srcStride = CeilDiv(nOriSize, INT8_ONE_BLK_SIZE);
        SetAntiqOffsetToBTParams(nOriSize);
        SetAntiqScaleToFixpipeParams(nOriSize);
    }

    __aicore__ inline void SetParams(const uint32_t kSize, const uint32_t nSize) {
        SetWeightToL0aParams(kSize, nSize);
        SetMmadParams(kSize, nSize);
    }

    __aicore__ inline void SetWeightToL0aParams(const uint32_t kSize,
                                                const uint32_t nSize) {
        weightParams_.repeatTimes = CeilDiv(kSize, INT8_ONE_BLK_SIZE);
        weightParams_.dstGap = CeilDiv(nSize, INT8_ONE_BLK_SIZE) * INT8_FRAC_NUM - 1;
    }

    // weight8 A1->A2
    __aicore__ inline void WeightToL0A(const LocalTensor<int8_t>& weightS8L1,
                                       const uint64_t pingpongOffset) {
        LoadDataWithTranspose(weightS8L0a_[pingpongOffset],
                              weightS8L1, weightParams_);
    }

    __aicore__ inline void SetAntiqOffsetToBTParams(const uint32_t nSize) {
        // 4是int32的字节数，antiqoffset在L1中是int32，右移2位
        constexpr uint32_t btBlkSizeS32 = BT_BLK_SIZE >> 2;
        antiqOffsetParams_.blockLen = CeilDiv(nSize, btBlkSizeS32);
    }

    // antiqoffset C1->BT
    __aicore__ inline void AntiqOffsetToBT(const LocalTensor<int32_t>& antiqOffsetL1) {
        DataCopy(antiqOffsetBT_, antiqOffsetL1, antiqOffsetParams_);
    }

    __aicore__ inline void SetAntiqScaleToFixpipeParams(const uint32_t nSize) {
        // 8是uint64的字节数，antiqscale在L1中是uint64，右移3位
        constexpr uint32_t fbBlkSizeU64 = FIXP_BLK_SIZE >> 3;
        antiqScaleParams_.blockLen = CeilDiv(nSize, fbBlkSizeU64);
    }

    // antiqscale C1->FB
    __aicore__ inline void AntiqScaleToFixpipe(const LocalTensor<uint64_t>& antiqScaleL1) {
        DataCopy(antiqScaleFP_, antiqScaleL1, antiqScaleParams_);
    }

    __aicore__ inline void SetMmadParams(const uint32_t kSize,
                                         const uint32_t nSize) {
        mmadParams_.m = kSize;
        mmadParams_.n = nSize;
        mmadParams_.k = nSize;
        if constexpr (hasAntiqOffset) {
            mmadParams_.cmatrixInitVal = false;
        } else {
            mmadParams_.cmatrixInitVal = true;
        }
    }

    // weight * diag + antiqoffset
    __aicore__ inline void Compute(const uint64_t pingpongOffset,
                                   const uint64_t biasOffset) {
        if constexpr (hasAntiqOffset) {
            Mmad(weightS32L0c_[pingpongOffset], weightS8L0a_[pingpongOffset],
                 diagS8L0b_, antiqOffsetBT_[biasOffset], mmadParams_);
        } else {
            Mmad(weightS32L0c_[pingpongOffset], weightS8L0a_[pingpongOffset],
                 diagS8L0b_, mmadParams_);
        }
    }

    // weight16 CO1->B1
    __aicore__ inline void WeightFixpDequant(const LocalTensor<half>& weightFP16L1,
                                             uint64_t pingpongOffset,
                                             uint64_t antiqScaleOffset,
                                             const uint32_t kSize,
                                             const uint32_t nSize) {
        SetFixPipeConfig(antiqScaleFP_[antiqScaleOffset]);
        /*
         * 接口原型：void copy_matrix_cc_to_cbuf(__cbuf__ half* dst, __cc__ int32_t* src,
         *                                      uint8_t sid, uint16_t NSize, uint16_t MSize,
         *                                      uint32_t dstStride_dst_D, uint16_t srcStride,
         *                                      uint8_t UnitFlagMode, QuantMode_t QuantPRE,
         *                                      uint8_t ReLUPRE, bool channelSplite, bool NZ2ND_EN);
         * 入参说明：
         * dst              : 目的操作数
         * src              : 源操作数
         * sid              : 预留参数
         * NSize            : src的N方向的size大小
         * MSize            : src的M方向的size大小
         * dstStride_dst_D  : dst相邻连续数据片段间隔，单位32B
         * srcStride        : src相邻连续数据片段间隔，单位16*sizeof(T)
         * UnitFlagMode     : 预留参数
         * QuantPRE         : 量化模式
         * ReLUPRE          : 是否使能relu的开关
         * channelSplite    : 是否是能通道拆分的功能
         * NZ2ND_EN         : 是否使能NZ2ND
         */
        copy_matrix_cc_to_cbuf((__cbuf__ half*)weightFP16L1.GetPhyAddr(),
                               (__cc__ int32_t*)weightS32L0c_[pingpongOffset].GetPhyAddr(),
                               0, nSize, kSize, 256, kSize,
                               0, QuantMode_t::VDEQF16, 0, 0, 0);
    }

    __aicore__ inline void Process(const LocalTensor<half>& weightFP16L1,
                                   const LocalTensor<int8_t>& weightS8L1,
                                   const uint64_t biasOffset,
                                   const uint32_t kSize,
                                   const uint32_t nSize) {
        uint64_t pingpongOffset = (processCount_ & 1) * pingpongSize_;
        WeightToL0A(weightS8L1, pingpongOffset);
        event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
        SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
        WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
        Compute(pingpongOffset, biasOffset);
        event_t eventIdMToFix = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_FIX));
        SetFlag<HardEvent::M_FIX>(eventIdMToFix);
        WaitFlag<HardEvent::M_FIX>(eventIdMToFix);
        WeightFixpDequant(weightFP16L1, pingpongOffset, biasOffset, kSize, nSize);
        processCount_++;
    }

    __aicore__ inline uint64_t Process1(const LocalTensor<int8_t>& weightS8L1) {
        uint64_t pingpongOffset = (processCount_ & 1) * pingpongSize_;
        WeightToL0A(weightS8L1, pingpongOffset);
        processCount_++;
        return pingpongOffset;
    }

    __aicore__ inline void Process2(const LocalTensor<half>& weightFP16L1,
                                    const uint64_t biasOffset,
                                    const uint64_t pingpongOffset,
                                    const uint32_t kSize,
                                    const uint32_t nSize) {
        Compute(pingpongOffset, biasOffset);
        event_t eventIdMToFix = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_FIX));
        SetFlag<HardEvent::M_FIX>(eventIdMToFix);
        WaitFlag<HardEvent::M_FIX>(eventIdMToFix);
        WeightFixpDequant(weightFP16L1, pingpongOffset, biasOffset, kSize, nSize);
    }

private:
    LoadData2dTransposeParams weightParams_;
    DataCopyParams antiqOffsetParams_;
    DataCopyParams antiqScaleParams_;
    MmadParams mmadParams_;
    uint64_t processCount_ = 0;
    // l1的切分固定，可用的db空间为32*256
    static constexpr uint64_t pingpongSize_ = 32 * 256;
    LocalTensor<int8_t> weightS8L0a_;
    LocalTensor<int8_t> diagS8L0b_;
    LocalTensor<int32_t> antiqOffsetBT_;
    LocalTensor<uint64_t> antiqScaleFP_;
    LocalTensor<int32_t> weightS32L0c_;
};
} // namespace WeightQuantBatchMatmulV2
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_FIXPIPE_STAGE1_H