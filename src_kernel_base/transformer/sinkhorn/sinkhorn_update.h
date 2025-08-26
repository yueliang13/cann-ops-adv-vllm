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
 * \file sinkhorn_update.h
 * \brief
 */
#ifndef SINKHORN_UPDATE_H_
#define SINKHORN_UPDATE_H_

namespace AscendC {
    // 通用float/half
    template<typename T, typename IT>
    template<typename _IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::CopyInFromP(uint16_t row, const GlobalTensor<_IT> &pG)
    {
        // 搬入cost
        uint16_t blockLen = totalCol * sizeof(T);
        LocalTensor<T> costLocal = costInQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{(uint16_t)row, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
#ifndef __CCE_KT_TEST__
        DataCopyPad(costLocal, pG, copyParams, padParams);
#endif
        costInQueue.EnQue(costLocal);
    }

    // 通用float/half
    template<typename T, typename IT>
    template<typename _IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::SaveP(uint16_t row, const GlobalTensor<_IT> &pG, const LocalTensor<T> &localTensor)
    {
        uint16_t blockLen = totalCol * sizeof(T);
        DataCopyExtParams copyParams{(uint16_t)row, blockLen, 0, 0, 0};
#ifndef __CCE_KT_TEST__
        DataCopyPad(pG, localTensor, copyParams);
#endif
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ComputeD0(uint32_t row, LocalTensor<T> costSrcLocal, LocalTensor<T> d1InLocal)
    {
        // costOutQueue = d1 * cost: 逐行计算 Mul
        {
            LocalTensor<T> costDstLocal = costOutQueue.AllocTensor<T>();
            for (int r = 0; r < row; r++) {
                uint32_t rowIdx = r * rowLengthAligned;
                Mul(costDstLocal[rowIdx], costSrcLocal[rowIdx], d1InLocal, totalCol);
            }
            costOutQueue.EnQue(costDstLocal);
        }

        // d0OutLocal = sum
        {
            LocalTensor<T> d0OutLocal = d0OutQueue.AllocTensor<T>();
            LocalTensor<T> d0OutLocal2 = d0OutQueue2.AllocTensor<T>();
            LocalTensor<T> d0OutLocal3 = d0OutQueue3.AllocTensor<T>();
            LocalTensor<T> costDstLocal = costOutQueue.DeQue<T>();

            event_t eventId_V_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            for (int r = 0; r < row; r++) {
                uint32_t rowIdx = r * rowLengthAligned;
                ReduceSum(d0OutLocal2, costDstLocal[rowIdx], d0OutLocal3, totalCol);
                SetFlag<HardEvent::V_S>(eventId_V_S);
                WaitFlag<HardEvent::V_S>(eventId_V_S);
                d0OutLocal.SetValue(r, d0OutLocal2.GetValue(0));
            }
            d0OutQueue.EnQue(d0OutLocal);
            d0OutQueue2.FreeTensor(d0OutLocal2);
            d0OutQueue3.FreeTensor(d0OutLocal3);
            costOutQueue.FreeTensor(costDstLocal);
        }

        // + eps
        {
            LocalTensor<T> d0OutLocal2 = d0OutQueue2.AllocTensor<T>();
            LocalTensor<T> d0OutLocal3 = d0OutQueue3.AllocTensor<T>();
            LocalTensor<T> d0OutLocal = d0OutQueue.DeQue<T>();
            DUMP_LT_3(d0OutLocal, row, "d0[sum]: ");
            Adds(d0OutLocal2, d0OutLocal, (T)eps, row);
            DUMP_LT_3(d0OutLocal2, row, "d0[eps]: ");
            PipeBarrier<PIPE_V>();
            Duplicate(d0OutLocal3, static_cast<T>((1.0f)/totalRow), row);
            PipeBarrier<PIPE_V>();
            Div(d0OutLocal, d0OutLocal3, d0OutLocal2, row);
            DUMP_LT_3(d0OutLocal, row, "d0[div]: ");
            d0OutQueue.EnQue(d0OutLocal);
            d0OutQueue3.FreeTensor(d0OutLocal3);
            d0OutQueue2.FreeTensor(d0OutLocal2);
        }
    }

    // 计算每个Tile的d1   torch.sum(d0.unsqueeze(1) * cost, 1)
    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ComputeD1(uint32_t row, LocalTensor<T> costSrcLocal, LocalTensor<T> d0OutLocal)
    {
        LocalTensor<T> d1OutLocal = d1OutQueue.AllocTensor<T>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        SetAtomicAdd<T>();
        event_t eventId_MTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
        for (int r = 0; r < row; r++) {
            uint32_t rowIdx = r * rowLengthAligned;
            T d0 = d0OutLocal.GetValue(r);
            WaitFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
            Muls(d1OutLocal, costSrcLocal[rowIdx], d0, totalCol);
            event_t eventId_V_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
#ifndef __CCE_KT_TEST__
            DataCopyPad(d1BlockInWS, d1OutLocal, copyParams);
#endif
            SetFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
        }
        WaitFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
        SetAtomicNone();
        d1OutQueue.FreeTensor(d1OutLocal);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::CopyInD1BlockInWS(DataCopyExtParams copyParams, DataCopyPadExtParams<T> padParams)
    {
        LocalTensor<T> d1OutLocal = d1OutQueue.AllocTensor<T>();
#ifndef __CCE_KT_TEST__
        DataCopyPad(d1OutLocal, d1BlockInWS, copyParams, padParams);
#endif
        d1OutQueue.EnQue(d1OutLocal);
        DUMP_LT_2(d1BlockInWS, totalCol, " d1[sm1]: ");
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::SumD1Block(DataCopyExtParams copyParams)
    {
        LocalTensor<T> d1OutLocal = d1OutQueue.DeQue<T>();
        DUMP_LT_2(d1OutLocal, totalCol, " d1[sm2]: ");
        SetAtomicAdd<T>();
#ifndef __CCE_KT_TEST__
        DataCopyPad(d1GlobalInWSNew, d1OutLocal, copyParams);
#endif
        DataCacheClean(d1GlobalInWSNew);
        SetAtomicNone();
        d1OutQueue.FreeTensor(d1OutLocal);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::UpdateD0()
    {
        // d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)

        DataCacheClean(d1GlobalInWS);
        DataCacheClean(d1GlobalInWSNew);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        // d1InQueue = d1
        {
            LocalTensor<T> d1InLocal = d1InQueue.AllocTensor<T>();
#ifndef __CCE_KT_TEST__
            DataCopyPad(d1InLocal, d1GlobalInWS, copyParams, padParams);
#endif
            d1InQueue.EnQue(d1InLocal);
        }

        LocalTensor<T> d1InLocal = d1InQueue.DeQue<T>();

        // d1BlockInWS = 0
        {
            LocalTensor<T> d1OutLocal = d1OutQueue.AllocTensor<T>();
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            Duplicate(d1OutLocal, static_cast<T>(0), totalCol);
            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);
#ifndef __CCE_KT_TEST__
            DataCopyPad(d1BlockInWS, d1OutLocal, copyParams);
#endif
            d1OutQueue.FreeTensor(d1OutLocal);
            DataCacheClean(d1BlockInWS);
        }

        for (int t = 0; t < tileNum; t++) {
            uint32_t tileIdx = t * this->tileLength;
            uint32_t length = this->tileLength;
            uint32_t row = tileRow;
            if (t == this->tileNum - 1) {
                length = this->lastTileLength;
                row = lastTileRow;
            }
            
            // 搬入cost
            CopyInFromP<IT>(row, pGlobal[tileIdx]);

            // 计算d0
            LocalTensor<T> costSrcLocal = costInQueue.DeQue<T>();
            ComputeD0(row, costSrcLocal, d1InLocal);

            LocalTensor<T> d0OutLocal = d0OutQueue.DeQue<T>();

            ComputeD1(row, costSrcLocal, d0OutLocal);
            DataCacheClean(d1BlockInWS);
            DUMP_LT_2(d1BlockInWS, totalCol, "d1[sum]: ");

            // copy to workspace
            {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(row * sizeof(T)), 0, 0, 0};
#ifndef __CCE_KT_TEST__
                DataCopyPad(d0BlockInWS[tileRow * t], d0OutLocal, copyParams);
#endif
            }
            DUMP_LT_2(d1BlockInWS, totalCol, "d1[s01]: ");

            d0OutQueue.FreeTensor(d0OutLocal);
            costInQueue.FreeTensor(costSrcLocal);
        }

#ifdef MULTI_CORE_SUM_FOR_D1
        // copy to d1 new Global
        CopyInD1BlockInWS(copyParams, padParams);
        SumD1Block(copyParams);
#endif
        d1InQueue.FreeTensor(d1InLocal);
#ifndef __CCE_KT_TEST__
        SyncAll();
#endif
        DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(headerInWS);
        DUMP_LT_0_2(d0BlockInWS, blockRow, "d0 : ");
#ifdef MULTI_CORE_SUM_FOR_D1
        DUMP_LT_0_2(d1GlobalInWSNew, totalCol, "d1[sum] : ");
#endif
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::SumD1(int block)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        // copy from core level
        {
            LocalTensor<T> d1InLocal = d1InQueue.AllocTensor<T>();
#ifndef __CCE_KT_TEST__
            DataCopyPad(d1InLocal, d1BlockInWS[block * totalCol], copyParams, padParams);
#endif
            d1InQueue.EnQue(d1InLocal);
        }
        {
            LocalTensor<T> d1OutLocal = d1OutQueue.DeQue<T>();
            LocalTensor<T> d1InLocal = d1InQueue.DeQue<T>();
            DUMP_LT_2(d1InLocal, totalCol, " d1[   ]: ");
            Add(d1OutLocal, d1InLocal, d1OutLocal, totalCol);
            d1OutQueue.EnQue(d1OutLocal);
            d1InQueue.FreeTensor(d1InLocal);
        }
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::UpdateD1()
    {
        if (blockIdx == 0) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

            // d1OutQueue = new d1
#ifdef MULTI_CORE_SUM_FOR_D1
            {
                LocalTensor<T> d1OutLocal = d1OutQueue.AllocTensor<T>();
#ifndef __CCE_KT_TEST__
                DataCopyPad(d1OutLocal, d1GlobalInWSNew, copyParams, padParams);
#endif
                d1OutQueue.EnQue(d1OutLocal);
            }
#else
            {
                LocalTensor<T> d1OutLocal = d1OutQueue.AllocTensor<T>();
                Duplicate(d1OutLocal, static_cast<T>(0.0f), totalCol);
                d1OutQueue.EnQue(d1OutLocal);
                for (int i = 0; i < blockDim; i++) {
                    SumD1(i);
                }
            }
#endif

            // d1OutQueue2 = new d1 + eps
            {
                LocalTensor<T> d1OutLocal = d1OutQueue.DeQue<T>();
                DUMP_LT_2(d1OutLocal, totalCol, " d1[sum]: ");
                LocalTensor<T> d1OutLocal2 = d1OutQueue2.AllocTensor<T>();
                Adds(d1OutLocal2, d1OutLocal, (T)eps, totalCol);
                d1OutQueue2.EnQue(d1OutLocal2);
                d1OutQueue.FreeTensor(d1OutLocal);
            }

            // d1OutQueue3 = 1.0 / totalCol
            {
                LocalTensor<T> d1OutLocal3 = d1OutQueue3.AllocTensor<T>();
                Duplicate(d1OutLocal3, static_cast<T>(1.0f / totalCol), totalCol);
                d1OutQueue3.EnQue(d1OutLocal3);
            }

            // d1InQueue = d1OutQueue3 / d1OutQueue2 =  (1.0 / totalCol) / (d1 new)
            {
                LocalTensor<T> d1InLocal = d1InQueue.AllocTensor<T>();
                LocalTensor<T> d1OutLocal2 = d1OutQueue2.DeQue<T>();
                LocalTensor<T> d1OutLocal3 = d1OutQueue3.DeQue<T>();

                Div(d1InLocal, d1OutLocal3, d1OutLocal2, totalCol);
                d1InQueue.EnQue(d1InLocal);
                d1OutQueue2.FreeTensor(d1OutLocal2);
                d1OutQueue3.FreeTensor(d1OutLocal3);
            }

            // d1OutQueue2 = old d1
            {
                LocalTensor<T> d1OutLocal2 = d1OutQueue2.AllocTensor<T>();
#ifndef __CCE_KT_TEST__
                DataCopyPad(d1OutLocal2, d1GlobalInWS, copyParams, padParams);
#endif
                d1OutQueue2.EnQue(d1OutLocal2);
            }

            // 取出
            LocalTensor<T> d1OutLocal2 = d1OutQueue2.DeQue<T>();
            LocalTensor<T> d1InLocal = d1InQueue.DeQue<T>();

            // 写新值
            {
#ifndef __CCE_KT_TEST__
                DataCopyPad(d1GlobalInWS, d1InLocal, copyParams);
#endif
                DataCacheClean(d1GlobalInWS);
            }

            // 清空新值
            {
                LocalTensor<T> d1OutLocal3 = d1OutQueue3.AllocTensor<T>();
                Duplicate(d1OutLocal3, static_cast<T>(0.0f), totalCol);
                event_t eventId_V_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
                WaitFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
#ifndef __CCE_KT_TEST__
                DataCopyPad(d1GlobalInWSNew, d1OutLocal3, copyParams);
#endif
                d1OutQueue3.FreeTensor(d1OutLocal3);
            }

            {
                DataCacheClean(d1GlobalInWSNew);
                LocalTensor<T> d1OutLocal3 = d1OutQueue3.AllocTensor<T>();
                LocalTensor<T> d1OutLocal = d1OutQueue.AllocTensor<T>();

                // d1_old - d1
                Sub(d1OutLocal3, d1OutLocal2, d1InLocal, totalCol);
                DUMP_LT_3(d1OutLocal3, totalCol, " d1[sub]: ");
                PipeBarrier<PIPE_V>();
                Abs(d1OutLocal, d1OutLocal3, totalCol);
                DUMP_LT_3(d1OutLocal, totalCol, " d1[abs]: ");
                PipeBarrier<PIPE_V>();
                ReduceSum(d1OutLocal2, d1OutLocal, d1OutLocal3, totalCol);
                event_t eventId_V_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventId_V_S);
                WaitFlag<HardEvent::V_S>(eventId_V_S);
                float tolerance = (float)(d1OutLocal2.GetValue(0)) / totalCol;
                OP_LOGD_0_1("tol[%d]: " FLOAT_FMT, loopCount, tolerance);
#ifdef __CCE_KT_TEST__
                // 孪生调试直接退出循环
                SetLoopFlag(0);
#else
                if (tolerance <= tol) {
                    // 退出循环
                    SetLoopFlag(0);
                }
#endif
                d1OutQueue.FreeTensor(d1OutLocal);
                d1OutQueue3.FreeTensor(d1OutLocal3);
            }

            d1InQueue.FreeTensor(d1InLocal);
            d1OutQueue2.FreeTensor(d1OutLocal2);
        }

#ifndef __CCE_KT_TEST__
        SyncAll();
#endif
        DataCacheClean(d1GlobalInWS);
        DataCacheClean(d1GlobalInWSNew);
        DUMP_LT_0_2(d1GlobalInWS, totalCol, " d1[gl ]: ");
        DUMP_LT_0_2(d1GlobalInWSNew, totalCol, " d1[gln]: ");
    }

    // 特殊处理 T = float, IT = bfloat16_t
    template<>
    template<>
    __aicore__ inline void KernelSinkhorn<float, bfloat16_t>::CopyInFromP<bfloat16_t>(uint16_t row, const GlobalTensor<bfloat16_t> &pG)
    {
        // 从costGlobal复制到tmpLocal
        uint16_t blockLen = totalCol * sizeof(float);
        LocalTensor<bfloat16_t> tmpLocal = costOutQueue.AllocTensor<bfloat16_t>();
        DataCopyExtParams copyParams{row, blockLen, 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
#ifndef __CCE_KT_TEST__
        DataCopyPad(tmpLocal, pGlobal, copyParams, padParams);
#endif
        costOutQueue.EnQue(tmpLocal);

        // bf16 ==> float
        tmpLocal = costOutQueue.DeQue<bfloat16_t>();
        LocalTensor<float> costLocal = costInQueue.AllocTensor<float>();
        Cast(costLocal, tmpLocal, RoundMode::CAST_NONE, row * totalColAligned);
        costInQueue.EnQue(costLocal);
        costOutQueue.FreeTensor(tmpLocal);
    }

    // 特殊处理 T = float, IT = half
    template<>
    template<>
    __aicore__ inline void KernelSinkhorn<float, half>::CopyInFromP<half>(uint16_t row, const GlobalTensor<half> &pG)
    {
        // 从costGlobal复制到tmpLocal
        uint16_t blockLen = totalCol * sizeof(float);
        LocalTensor<half> tmpLocal = costOutQueue.AllocTensor<half>();
        DataCopyExtParams copyParams{row, blockLen, 0, 0, 0};
        DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
#ifndef __CCE_KT_TEST__
        DataCopyPad(tmpLocal, pGlobal, copyParams, padParams);
#endif
        costOutQueue.EnQue(tmpLocal);

        // bf16 ==> float
        tmpLocal = costOutQueue.DeQue<half>();
        LocalTensor<float> costLocal = costInQueue.AllocTensor<float>();
        Cast(costLocal, tmpLocal, RoundMode::CAST_NONE, row * totalColAligned);
        costInQueue.EnQue(costLocal);
        costOutQueue.FreeTensor(tmpLocal);
    }

    // 特殊处理 T = float, IT = bfloat16_t
    template<>
    template<>
    __aicore__ inline void KernelSinkhorn<float, bfloat16_t>::SaveP<bfloat16_t>(uint16_t row, const GlobalTensor<bfloat16_t> &pG, const LocalTensor<float> &localTensor)
    {
        // float ==> bf16
        uint16_t blockLen = totalCol * sizeof(float);
        LocalTensor<bfloat16_t> tmpLocal = costOutQueue.AllocTensor<bfloat16_t>();
        Cast(tmpLocal, localTensor, RoundMode::CAST_TRUNC, row * totalColAligned);
        costOutQueue.EnQue(tmpLocal);

        // bf16 ==> pG
        tmpLocal = costOutQueue.DeQue<bfloat16_t>();
        DataCopyExtParams copyParams{row, static_cast<uint32_t>(blockLen / 2), 0, 0, 0};
#ifndef __CCE_KT_TEST__
        DataCopyPad(pG, tmpLocal, copyParams);
#endif
        costOutQueue.FreeTensor(tmpLocal);
    }

    // 特殊处理 T = float, IT = half
    template<>
    template<>
    __aicore__ inline void KernelSinkhorn<float, half>::SaveP<half>(uint16_t row, const GlobalTensor<half> &pG, const LocalTensor<float> &localTensor)
    {
        // float ==> bf16
        uint16_t blockLen = totalCol * sizeof(float);
        LocalTensor<half> tmpLocal = costOutQueue.AllocTensor<half>();
        Cast(tmpLocal, localTensor, RoundMode::CAST_TRUNC, row * totalColAligned);
        costOutQueue.EnQue(tmpLocal);

        // bf16 ==> pG
        tmpLocal = costOutQueue.DeQue<half>();
        DataCopyExtParams copyParams{row, static_cast<uint32_t>(blockLen / 2), 0, 0, 0};
#ifndef __CCE_KT_TEST__
        DataCopyPad(pG, tmpLocal, copyParams);
#endif
        costOutQueue.FreeTensor(tmpLocal);
    }
} // namespace AscendC

#endif // SINKHORN_UPDATE_H_
