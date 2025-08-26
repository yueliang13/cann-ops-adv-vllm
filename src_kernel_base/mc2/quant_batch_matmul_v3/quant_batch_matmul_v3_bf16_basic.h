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
 * \file quant_batch_matmul_v3_bf16_basic.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_V3_BF16_BASIC_H
#define QUANT_BATCH_MATMUL_V3_BF16_BASIC_H

#include "quant_batch_matmul_v3_block.h"
#include "quant_batch_matmul_v3_update.h"

namespace AscendC {
template <TemplateBasicType>
class BmmBasicDequantBf16 {
public:
    __aicore__ inline BmmBasicDequantBf16() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR bias, GM_ADDR y, GM_ADDR workSpace,
                                const QuantBatchMatmulV3TilingData *__restrict tilingData, TPipe *tPipe)
    {
        blockIdx = GetBlockIdx();
        blockIdx /= GetTaskRation();
        if (GetSubBlockIdx() > 0) {
            return;
        }

        usedCoreNum_ = tilingData->matmulTiling.usedCoreNum;
        if (blockIdx >= usedCoreNum_) {
            return;
        }
        pipe_ = tPipe;
        mm_.Init(&(tilingData->matmulTiling), pipe_);
        InitTilingData(tilingData);
        InitGlobalBuffers(x1, x2, scale, bias, y, workSpace);
        InitLocalBuffers();
        offsetWorkspaceC_ = BUFFER_NUM * blockIdx * baseM_ * baseN_;

        block_.Init(tilingData);
        update_.template Init<x1Format, x2Format, aTrans, bTrans>(&tilingData->matmulTiling, block_.params_);
        loop_ = 0;  // all_gather_quant_batch_mat_mul.h循环调Init和Process，管理CV同步的计数器每次都要清零
    }

    __aicore__ inline void Process()
    {
        if (blockIdx >= usedCoreNum_ || GetSubBlockIdx() > 0) {
            return;
        }

        bool reverse = true;
        bool pongSwitch = false;
        uint64_t pingOffsetC = offsetWorkspaceC_;
        // 首块计算，兼容无L2cache切分场景，减少scalar计算
        block_.InitFirstTileBlockIndex();
        OneTileCompute(0, 0, pingOffsetC, pongSwitch);

        for (uint64_t mTileIndex = 0; mTileIndex < block_.params_.mTileCntL2; mTileIndex++) {
            reverse = !reverse;
            for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < block_.params_.nTileCntL2; nTileIndexTemp++) {
                uint64_t nTileIndex = reverse ? (block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
                if (mTileIndex > 0 || nTileIndex > 0) {  // 跳过首块
                    block_.UpdateBlockCnt(mTileIndex, nTileIndex);
                    block_.InitBlockIndex();
                    OneTileCompute(mTileIndex, nTileIndex, pingOffsetC, pongSwitch);
                }
            }
        }

        End();
    }

    __aicore__ inline UPDATE_TYPE &GetUpdateObj()
    {
        return update_;
    }

private:
    __aicore__ inline void InitTilingData(const QuantBatchMatmulV3TilingData *tilingData)
    {
        isPerTensor_ = tilingData->params.isPerTensor;

        m_ = tilingData->matmulTiling.M;
        n_ = tilingData->matmulTiling.N;
        ka_ = tilingData->matmulTiling.Ka;

        baseM_ = tilingData->matmulTiling.baseM;
        baseN_ = tilingData->matmulTiling.baseN;

        hasBias_ = tilingData->matmulTiling.isBias;
        biasDtype_ = tilingData->params.biasDtype;
        if (biasDtype_ == DT_INT32 || biasDtype_ == DT_FLOAT) {
            biasDtypeSize_ = sizeof(int32_t);
        } else {
            biasDtypeSize_ = sizeof(half);
        }

        ubCalcM_ = tilingData->params.ubCalcM;
        ubCalcN_ = tilingData->params.ubCalcN;
        ubTmpBuffer_ = tilingData->params.needUbBuffer;
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR bias, GM_ADDR y,
                                             GM_ADDR workSpace)
    {
        if (isPerTensor_) {
            scaleScalar_ = *((__gm__ scaleType *)scale);
        }
        xGm_.SetGlobalBuffer((__gm__ x1Type *)x1);
        weightGm_.SetGlobalBuffer((__gm__ x2Type *)x2);
        if (m_ <= baseM_) {
            weightGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if (hasBias_ != 0) {
            if (biasDtype_ == DT_BF16) {
                biasGmBf16_.SetGlobalBuffer((__gm__ bfloat16_t *)bias);
            } else if (biasDtype_ == DT_FLOAT16) {
                biasGmFp16_.SetGlobalBuffer((__gm__ half *)bias);
            } else if (biasDtype_ == DT_FLOAT) {
                biasGmFp32_.SetGlobalBuffer((__gm__ float *)bias);
            } else {
                biasGmInt32_.SetGlobalBuffer((__gm__ int32_t *)bias);
            }
        }
        yGm_.SetGlobalBuffer((__gm__ yType *)y);
        scaleGm_.SetGlobalBuffer((__gm__ scaleType *)scale);
        mmOutGm_.SetGlobalBuffer((__gm__ int32_t *)workSpace, BUFFER_NUM * usedCoreNum_ * baseM_ * baseN_);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        pipe_->InitBuffer(vecQueSrc_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(int32_t));
        pipe_->InitBuffer(vecQueTmp_, ubTmpBuffer_);
        pipe_->InitBuffer(vecQueOut_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(yType));
        if (biasDtype_ != DT_INT32) {
            pipe_->InitBuffer(biasFp32Tmp_, ubCalcN_ * sizeof(float));
            pipe_->InitBuffer(outFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
            pipe_->InitBuffer(vecQueBias_, BUFFER_NUM, ubCalcN_ * biasDtypeSize_);
        }
        if (!isPerTensor_) {
            pipe_->InitBuffer(vecQueScale_, BUFFER_NUM, ubCalcN_ * sizeof(scaleType));
        }
    }

    __aicore__ inline void OneTileCompute(uint64_t mTileIndex, uint64_t nTileIndex, uint64_t pingOffsetC,
                                          bool &pongSwitch)
    {
        for (uint64_t j = 0; j < block_.realRound_; j++) {
            // 更新此次基本块的大小和输入输出地址
            update_.template UpdateBlockParamsAndCalcGmOffset<x1Format, x2Format, aTrans, bTrans>(
                block_.params_, offset_, mTileIndex, nTileIndex);

            offsetWorkspaceC_ = pingOffsetC + pongSwitch * baseM_ * baseN_;
            BasicMMDequantCompute(block_.params_.singleCoreM, block_.params_.singleCoreN,
                                    C2V_PING_FLAG | pongSwitch, V2C_PING_FLAG | pongSwitch);
            pongSwitch = !pongSwitch;
            block_.UpdateBlockIndex();
        }
    }

    __aicore__ inline void BasicMMDequantCompute(uint32_t basicM, uint32_t basicN, uint16_t v2cSyncFlag,
                                                 uint16_t c2vSyncFlag)
    {
        if ASCEND_IS_AIC {
            if (++loop_ > 2) {  // 2表示跳过第一次ping和第一次pong
                WaitEvent(v2cSyncFlag);
            }
            BasicMMCompute(basicM, basicN);
            NotifyEvent<PIPE_FIX>(c2vSyncFlag);
        }

        if ASCEND_IS_AIV {
            WaitEvent(c2vSyncFlag);
            BasicDequantCompute(mmOutGm_, basicM, basicN);
            NotifyEvent<PIPE_MTE2>(v2cSyncFlag);
        }
    }

    __aicore__ inline void BasicMMCompute(uint32_t basicM, uint32_t basicN)
    {
        mm_.SetSingleShape(basicM, basicN, ka_);
        mm_.SetTensorA(xGm_[offset_.offsetA], aTrans);
        mm_.SetTensorB(weightGm_[offset_.offsetB], bTrans);
        if (hasBias_ != 0 && biasDtype_ == DT_INT32) {
            mm_.SetBias(biasGmInt32_[offset_.offsetBias]);
        }
        mm_.Iterate();
        mm_.GetTensorC(mmOutGm_[offsetWorkspaceC_], 0, true);
    }

    __aicore__ inline void BasicDequantCompute(GlobalTensor<int32_t> &curMmOutGm, uint32_t curAicM, uint32_t curAicN)
    {
        LocalTensor<float> dstLocalFp32;
        LocalTensor<float> biasFp32;
        LocalTensor<bfloat16_t> oriBiasBf16;
        LocalTensor<half> oriBiasFp16;
        LocalTensor<float> oriBiasFp32;
        uint32_t curAivM = ubCalcM_;
        // calcN in ub is equal to aicN
        uint32_t curAivN = curAicN;
        uint32_t mUbLoops = DequantBmm::CeilDiv(curAicM, ubCalcM_);
        DataCopyParams gm2UbParams{1, 0, 0, 0};
        DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        DataCopyPadParams padParams;
        DequantParams dequantParams;
        DequantBmm::CalcDequantParams(mUbLoops == 1 ? curAicM : ubCalcM_, curAicN, dequantParams);
        for (uint32_t mUbLoopIdx = 0; mUbLoopIdx < mUbLoops; ++mUbLoopIdx) {
            if (mUbLoopIdx == mUbLoops - 1) {
                curAivM = curAicM - ubCalcM_ * (mUbLoops - 1);
                DequantBmm::CalcDequantParams(curAivM, curAicN, dequantParams, mUbLoops != 1 && curAivM != ubCalcM_);
            }
            LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
            LocalTensor<yType> dstLocal = vecQueOut_.AllocTensor<yType>();
            LocalTensor<uint8_t> tmpLocal = vecQueTmp_.Get<uint8_t>();
            // datacopypad 32B aligned
            gm2UbParams.blockLen = curAivN * sizeof(int32_t);
            gm2UbParams.blockCount = curAivM;
            gm2UbParams.srcStride = 0;
            uint32_t curAicAivOffset = offsetWorkspaceC_ + mUbLoopIdx * ubCalcM_ * curAivN;
            DataCopyPad(srcLocal, curMmOutGm[curAicAivOffset], gm2UbParams, padParams);
            set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EVENT_ID0));
            wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EVENT_ID0));
            if (biasDtype_ != DT_INT32) {
                BiasTensorInit(dstLocalFp32, biasFp32, oriBiasBf16, oriBiasFp16, oriBiasFp32);
                BiasGm2Ub(oriBiasBf16, oriBiasFp16, oriBiasFp32, padParams, curAicN);
            }
            if (isPerTensor_) {
                set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EVENT_ID1));
                wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EVENT_ID1));
                if (biasDtype_ != DT_INT32) {
                    AscendDequant(dstLocalFp32, srcLocal, scaleScalar_, tmpLocal, dequantParams);
                } else {
                    AscendDequant(dstLocal, srcLocal, scaleScalar_, tmpLocal, dequantParams);
                }
            } else {
                LocalTensor<scaleType> scaleLocal = vecQueScale_.AllocTensor<scaleType>();
                Bf16ScaleGm2Ub(scaleLocal, scaleGm_, padParams, curAicN);
                set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EVENT_ID1));
                wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EVENT_ID1));
                if (biasDtype_ != DT_INT32) {
                    AscendDequant(dstLocalFp32, srcLocal, scaleLocal, tmpLocal, dequantParams);
                } else {
                    AscendDequant(dstLocal, srcLocal, scaleLocal, tmpLocal, dequantParams);
                }
                vecQueScale_.FreeTensor(scaleLocal);
            }
            if (biasDtype_ != DT_INT32) {
                CalBiasAdd(dstLocalFp32, biasFp32, oriBiasBf16, oriBiasFp16, oriBiasFp32, dstLocal, curAivN, curAivM);
            }
            set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(EVENT_ID2));  // 2: event_id of ub->gm
            vecQueSrc_.FreeTensor(srcLocal);
            // dst from ub -> gm
            ub2GmParams.blockLen = curAivN * sizeof(yType);
            ub2GmParams.blockCount = curAivM;
            ub2GmParams.dstStride = (n_ - curAivN) * sizeof(yType);
            uint64_t aivOffset = mUbLoopIdx * ubCalcM_ * n_;
            wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(EVENT_ID2));  // 2: event_id of ub->gm
            DataCopyPad(yGm_[offset_.offsetC + aivOffset], dstLocal, ub2GmParams);
            vecQueOut_.FreeTensor(dstLocal);
        }
    }

    __aicore__ inline void BiasTensorInit(LocalTensor<float>& dstLocalFp32, LocalTensor<float>& biasFp32,
                                          LocalTensor<bfloat16_t>& oriBiasBf16, LocalTensor<half>& oriBiasFp16,
                                          LocalTensor<float>& oriBiasFp32)
    {
        dstLocalFp32 = outFp32Tmp_.Get<float>();
        biasFp32 = biasFp32Tmp_.Get<float>();
        if (biasDtype_ == DT_BF16) {
            oriBiasBf16 = vecQueBias_.AllocTensor<bfloat16_t>();  // free in CalBiasAdd
        } else if (biasDtype_ == DT_FLOAT16) {
            oriBiasFp16 = vecQueBias_.AllocTensor<half>();  // free in CalBiasAdd
        } else if (biasDtype_ == DT_FLOAT) {
            oriBiasFp32 = vecQueBias_.AllocTensor<float>();  // free in CalBiasAdd
        }
    }

    __aicore__ inline void BiasGm2Ub(LocalTensor<bfloat16_t> &oriBiasBf16, LocalTensor<half> &oriBiasFp16,
                                     LocalTensor<float> &oriBiasFp32, DataCopyPadParams padParams, uint32_t curAivN)
    {
        DataCopyParams bias2UbParams{1, 0, 0, 0};
        bias2UbParams.blockLen = curAivN * biasDtypeSize_;

        if (biasDtype_ == DT_BF16) {
            DataCopyPad(oriBiasBf16, biasGmBf16_[offset_.offsetBias], bias2UbParams, padParams);
        } else if (biasDtype_ == DT_FLOAT16) {
            DataCopyPad(oriBiasFp16, biasGmFp16_[offset_.offsetBias], bias2UbParams, padParams);
        } else if (biasDtype_ == DT_FLOAT) {
            DataCopyPad(oriBiasFp32, biasGmFp32_[offset_.offsetBias], bias2UbParams, padParams);
        }
    }

    __aicore__ inline void Bf16ScaleGm2Ub(LocalTensor<scaleType> &scaleLocal, GlobalTensor<scaleType> &scaleGm_,
                                          DataCopyPadParams &padParams, uint32_t curAivN)
    {
        DataCopyParams scale2UbParams{1, 0, 0, 0};
        scale2UbParams.blockLen = curAivN * sizeof(scaleType);
        DataCopyPad(scaleLocal, scaleGm_[offset_.offsetScale], scale2UbParams, padParams);
    }

    __aicore__ inline void CalBiasAdd(LocalTensor<float> &dstLocalFp32, LocalTensor<float> &biasFp32,
                                      LocalTensor<bfloat16_t> &oriBiasBf16, LocalTensor<half> &oriBiasFp16,
                                      LocalTensor<float> &oriBiasFp32, LocalTensor<yType> &dstLocal, uint32_t curAivN,
                                      uint32_t curAivM)
    {
        uint32_t computedAivN = DequantBmm::Align(curAivN, 8U);  // 8: 32B aligned for int32_t
        uint32_t ubResAlignedN = DequantBmm::Align(curAivN);     // 16: sizeof(yType) is 2, 32B / 2
        pipe_barrier(PIPE_V);
        if (biasDtype_ == DT_BF16) {
            Cast(biasFp32, oriBiasBf16, RoundMode::CAST_NONE, ubResAlignedN);
            pipe_barrier(PIPE_V);
            vecQueBias_.FreeTensor(oriBiasBf16);
        } else if (biasDtype_ == DT_FLOAT16) {
            Cast(biasFp32, oriBiasFp16, RoundMode::CAST_NONE, ubResAlignedN);
            pipe_barrier(PIPE_V);
            vecQueBias_.FreeTensor(oriBiasFp16);
        } else if (biasDtype_ == DT_FLOAT) {
            biasFp32 = oriBiasFp32;
            pipe_barrier(PIPE_V);
            vecQueBias_.FreeTensor(oriBiasFp32);
        }
        LocalTensor<float> dstLocalFp32Pad;
        if (computedAivN != ubResAlignedN) {
            dstLocalFp32Pad = vecQueTmp_.Get<float>();
            for (int32_t mIdx = 0; mIdx < curAivM; ++mIdx) {
                Add(dstLocalFp32Pad[mIdx * ubResAlignedN], dstLocalFp32[mIdx * computedAivN], biasFp32, computedAivN);
            }
        } else {
            for (int32_t mIdx = 0; mIdx < curAivM; ++mIdx) {
                Add(dstLocalFp32[mIdx * ubResAlignedN], dstLocalFp32[mIdx * ubResAlignedN], biasFp32, ubResAlignedN);
            }
        }
        pipe_barrier(PIPE_V);
        if (computedAivN != ubResAlignedN) {
            Cast(dstLocal, dstLocalFp32Pad, RoundMode::CAST_RINT, curAivM * ubResAlignedN);
        } else {
            Cast(dstLocal, dstLocalFp32, RoundMode::CAST_RINT, curAivM * ubResAlignedN);
        }
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            // AIC跳过前两次Wait，也就是一次ping一次pong，这里补上
            if (loop_ > 2) {  // 大于2表示需要补上开头跳过的ping
                WaitEvent(C2V_PING_FLAG);
            }

            if (loop_ > 3) {  // 大于3表示需要补上开头跳过的pong
                WaitEvent(C2V_PONG_FLAG);
            }
            mm_.End();
        }
    }

private:
    GlobalTensor<x1Type> xGm_;
    GlobalTensor<x2Type> weightGm_;
    GlobalTensor<int32_t> biasGmInt32_;
    GlobalTensor<bfloat16_t> biasGmBf16_;
    GlobalTensor<half> biasGmFp16_;
    GlobalTensor<float> biasGmFp32_;
    GlobalTensor<yType> yGm_;
    GlobalTensor<scaleType> scaleGm_;
    GlobalTensor<int32_t> mmOutGm_;

    TPipe *pipe_;
    // define the que
    TQue<QuePosition::VECIN, BUFFER_NUM> vecQueSrc_;
    TQue<QuePosition::VECIN, BUFFER_NUM> vecQueScale_;
    TQue<QuePosition::VECIN, BUFFER_NUM> vecQueBias_;
    TBuf<TPosition::VECCALC> vecQueTmp_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> vecQueOut_;
    // used when bias type is bf16/fp16/fp32, dequant result should be fp32
    TBuf<TPosition::VECCALC> biasFp32Tmp_;
    TBuf<TPosition::VECCALC> outFp32Tmp_;

    scaleType scaleScalar_;

    // tilling data
    bool isPerTensor_;
    uint32_t usedCoreNum_;
    uint32_t m_;
    uint32_t n_;
    uint32_t ka_;
    uint32_t baseM_;
    uint32_t baseN_;
    uint32_t hasBias_;
    uint32_t biasDtype_ = 0;
    uint32_t biasDtypeSize_ = 0;
    // vector
    uint32_t ubCalcM_;
    uint32_t ubCalcN_;
    uint32_t ubTmpBuffer_;
    // basic block
    uint32_t blockIdx;
    uint64_t offsetWorkspaceC_ = 0;
    uint64_t loop_ = 0;

    QuantBatchMatmulV3BaseBlock block_;
    UPDATE_TYPE update_; // 量化mm或mc2的更新计算大小和地址的接口
    QBmmBlockOffset offset_;

    using AMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, x1Type, aTrans>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, DequantBmm::GetFormat(x2Format), x2Type, bTrans>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    // notice: the TPos of ctype must be ub given by mm api when iterate<false>,
    // but actually we can move data to gm then to ub.
    using CMatmulType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t>;
    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, CFG_MDL> mm_;
};

}  // namespace AscendC

#endif  // QUANT_BATCH_MATMUL_V3_BF16_BASIC_H