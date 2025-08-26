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
 * \file mc2_nd_to_nz.h
 * \brief
 */
#ifndef MC2_ND_TO_NZ_H
#define MC2_ND_TO_NZ_H

#define ENALBE_ND2NZ 1

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
#include "mc2_tiling_struct.h"

namespace AscendC {
using A_DTYPE = DTYPE_X1;
using B_DTYPE = DTYPE_X1;
using C_DTYPE = DTYPE_Y;
using BIAS_DTYPE = DTYPE_Y;

constexpr uint32_t MC2_ALIGN_SIZE = 512;
constexpr uint32_t EVENT_ID_4 = 4;
constexpr uint32_t EVENT_ID_5 = 5;
constexpr uint32_t EVENT_ID_6 = 6;
constexpr uint32_t MOVE_LEFT_SIZE = 8;
constexpr uint32_t UB_ALIGN_SIZE = 32;

using namespace matmul;
template <class T>
__aicore__ inline void CopyGmToUbufAlignMc2(__ubuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t nBurst,
                                         uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                                         uint32_t srcGap, uint32_t dstGap) {
    if constexpr (sizeof(T) == 1) {
        copy_gm_to_ubuf_align_b8(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(T) == 2) {
        copy_gm_to_ubuf_align_b16(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(T) == 4) {
        copy_gm_to_ubuf_align_b32(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else {
        ASSERT(false);
    }
}

template <typename T>
__aicore__ inline void CopyUbufToGmAlign(__gm__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t nBurst,
                                         uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                                         uint32_t srcGap, uint32_t dstGap) {
    if constexpr (sizeof(T) == 1) {
        copy_ubuf_to_gm_align_b8(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(T) == 2) {
        copy_ubuf_to_gm_align_b16(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(T) == 4) {
        copy_ubuf_to_gm_align_b32(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else {
        ASSERT(false);
    }
}

// outputOrgWidth should be 512 byte aligned
// outputWidth should be 32 byte aligned
template <typename T>
__aicore__ inline void CopyPad(const GlobalTensor<T> &outputGlobal, const LocalTensor<T> &tmpUb,
                               const GlobalTensor<T> &inputGlobal, uint32_t nBurst, uint32_t ubDstGap,
                               uint32_t inputWidth, uint32_t outputWidth, uint32_t inputOrgWidth,
                               uint32_t outputOrgWidth, uint8_t pingpongID) {
    CopyGmToUbufAlignMc2<T>((__ubuf__ void *)tmpUb.GetPhyAddr(), (__gm__ void *)inputGlobal.GetPhyAddr(), 0,
                         static_cast<uint16_t>(nBurst), inputWidth * sizeof(T), 0, 0,
                         (inputOrgWidth - inputWidth) * sizeof(T), ubDstGap);
    set_flag(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(pingpongID));
    wait_flag(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(pingpongID));

    CopyUbufToGmAlign<T>((__gm__ void *)outputGlobal.GetPhyAddr(), (__ubuf__ void *)tmpUb.GetPhyAddr(), 0,
                         static_cast<uint16_t>(nBurst), outputWidth * sizeof(T), 0, 0, ubDstGap,
                         (outputOrgWidth - outputWidth) * sizeof(T));
}

__aicore__ __inline__ GM_ADDR GetTailA(GM_ADDR aGM, TCubeTiling &tiling, uint32_t size) {
    return aGM + (tiling.M * tiling.Ka) * sizeof(A_DTYPE) * size;
}
__aicore__ __inline__ GM_ADDR GetTailC(GM_ADDR cGM, TCubeTiling &tiling, uint32_t size) {
    return cGM + (tiling.M * tiling.N) * sizeof(C_DTYPE) * size;
}

// curBlock 为 GetBlockIdx()
// userBlock 指 V core的数量
template <class T>
__aicore__ inline void Gm2GmTrans(GM_ADDR output, GM_ADDR aGm, uint32_t row, uint32_t col, uint32_t curBlock,
                                  uint32_t userBlock) {
    if (g_coreType != AIV) {
        return;
    }
    if (curBlock >= userBlock) {
        return;
    }
    int allCoreSize = row * col;
    int singleVCoreSize = (allCoreSize + userBlock - 1) / userBlock;
    if (curBlock == userBlock - 1) {
        singleVCoreSize = allCoreSize - singleVCoreSize * (userBlock - 1);
    }  // 尾核可能有尾块
    auto dataSize = TOTAL_UB_SIZE / sizeof(T);

    GlobalTensor<T> gmSrc;
    GlobalTensor<T> gmDst;
    gmSrc.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(aGm), allCoreSize);
    gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output), allCoreSize);

    TBuf<TPosition::VECCALC> totalUbBuf;
    GetTPipePtr()->InitBuffer(totalUbBuf, TOTAL_UB_SIZE);
    LocalTensor<T> fullUbTensor = totalUbBuf.Get<T>();
    LocalTensor<T> tmpUb = fullUbTensor[0];
    int repeat = singleVCoreSize / dataSize;
    int tail = singleVCoreSize % dataSize;

    for (int i = 0; i < repeat; i++) {
        CopyGmToUbufAlignMc2<T>((__ubuf__ void *)tmpUb.GetPhyAddr(),
                             (__gm__ void *)gmSrc[i * dataSize + curBlock * singleVCoreSize].GetPhyAddr(), 0, 1,
                             dataSize * sizeof(T), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_MTE3, 0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, 0);
        CopyUbufToGmAlign<T>((__gm__ void *)gmDst[i * dataSize + curBlock * singleVCoreSize].GetPhyAddr(),
                             (__ubuf__ void *)tmpUb.GetPhyAddr(), 0, 1, dataSize * sizeof(T), 0, 0, 0, 0);
        set_flag(PIPE_MTE3, PIPE_MTE2, 0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, 0);
    }
    if (tail != 0) {
        CopyGmToUbufAlignMc2<T>((__ubuf__ void *)tmpUb.GetPhyAddr(),
                             (__gm__ void *)gmSrc[repeat * dataSize + curBlock * singleVCoreSize].GetPhyAddr(), 0, 1,
                             tail * sizeof(T), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_MTE3, 0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, 0);
        CopyUbufToGmAlign<T>((__gm__ void *)gmDst[repeat * dataSize + curBlock * singleVCoreSize].GetPhyAddr(),
                             (__ubuf__ void *)tmpUb.GetPhyAddr(), 0, 1, tail * sizeof(T), 0, 0, 0, 0);
    }
}

#if ENALBE_ND2NZ

#define DB1
#ifdef DB

template <class T>
__aicore__ inline void PreCopyPadNd2Nz(const LocalTensor<T> &tmpUb, const GlobalTensor<T> &inputGlobal, uint32_t nBurst,
                                       uint32_t gmSrcGap, uint32_t inputWidth, uint32_t outputWidth, uint16_t pad_size,
                                       uint8_t pingpongID) {
    DataCopyParams copyParams{static_cast<uint16_t>(nBurst), static_cast<uint16_t>(inputWidth * sizeof(T)),
                              static_cast<uint16_t>(gmSrcGap), 0};
    DataCopyPadParams padParams{false, 0, 0, 0};
    if (outputWidth != inputWidth) {
        padParams.isPad = true;
        padParams.rightPadding = outputWidth - inputWidth;
    }

    DataCopyPadGm2UBImpl((__ubuf__ T *)tmpUb.GetPhyAddr(), (__gm__ T *)inputGlobal.GetPhyAddr(), copyParams, padParams);
    set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(pingpongID));
    wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(pingpongID));
    if (pad_size != 0 && nBurst > pad_size) {
        auto pad_offset = (nBurst - pad_size) * outputWidth;
        uint64_t mask_count = 32 / sizeof(T) * nBurst;
        set_vector_mask(0, pad_size * outputWidth);
        DuplicateIntrinsicsImpl((__ubuf__ T *)tmpUb[pad_offset].GetPhyAddr(), (T)0, 1, 1, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, mask_count);
    }
}

template <class T>
__aicore__ inline void CopyPadNd2Nz(const GlobalTensor<T> &outputGlobal, const LocalTensor<T> &tmpUb,
                                    const LocalTensor<T> &transUb, const GlobalTensor<T> &inputGlobal, uint32_t nBurst,
                                    uint32_t gmSrcGap, uint32_t inputWidth, uint32_t outputWidth, uint32_t height,
                                    uint16_t pad_size, uint8_t pingpongID) {
    PreCopyPadNd2Nz(tmpUb, inputGlobal, nBurst, inputWidth, outputWidth, pad_size, pingpongID);
    // use vmuls to nd2nz
    uint32_t c0Size;
    if (sizeof(T) == sizeof(float)) {
        c0Size = 8;
    } else if (sizeof(T) == sizeof(int8_t)) {
        c0Size = 32;
    } else {
        c0Size = 16;
    }
    int widthStep = Ceil(inputWidth, c0Size);  // 行方向搬运多少次
    uint64_t mask_count = c0Size * nBurst;
    uint16_t dstBlkStride = 1;
    uint16_t srcBlkStride = widthStep;
    uint16_t dstRepStride = nBurst;
    uint16_t srcRepStride = 1;
    int VCOPY_MAX_REPEAT = 255;
    int split_num = widthStep / VCOPY_MAX_REPEAT;
    int tail_num = widthStep % VCOPY_MAX_REPEAT;
    int dstOffset = 0;
    int srcOffset = 0;
    for (int i = 0; i < split_num; ++i) {
        dstOffset = VCOPY_MAX_REPEAT * mask_count;
        srcOffset = VCOPY_MAX_REPEAT * c0Size;
        vcopy((__ubuf__ uint16_t *)transUb[i * dstOffset].GetPhyAddr(),
              (__ubuf__ uint16_t *)tmpUb[i * srcOffset].GetPhyAddr(), (uint8_t)VCOPY_MAX_REPEAT, dstBlkStride,
              srcBlkStride, dstRepStride, srcRepStride);
    }
    if (tail_num != 0) {
        dstOffset = VCOPY_MAX_REPEAT * mask_count * split_num;
        srcOffset = VCOPY_MAX_REPEAT * c0Size * split_num;
        vcopy((__ubuf__ uint16_t *)transUb[dstOffset].GetPhyAddr(), (__ubuf__ uint16_t *)tmpUb[srcOffset].GetPhyAddr(),
              (uint8_t)tail_num, dstBlkStride, srcBlkStride, dstRepStride, srcRepStride);
    }
    // 要在k方向做repeat切分
    set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(pingpongID));
    wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(pingpongID));
    DataCopyParams copyOutParams;
    copyOutParams.blockLen = nBurst;       // c0 size 32byte
    copyOutParams.blockCount = widthStep;  // 最大4095， 需要切分
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = height * 2 - nBurst;  // 分了两个核，所以需要在m方向上跳转一大块
    DataCopy(outputGlobal, transUb, copyOutParams);
}

// widthAligned < 192 * 1024 / sizeof(T)
template <class T>
__aicore__ inline void PrePaddingImplNd2Nz(const GlobalTensor<T> &mmWorkspace, const GlobalTensor<T> &mmGlobal,
                                           uint32_t height, uint32_t width, uint32_t widthAligned, int height_pad_size,
                                           int ori_width) {
    int pad_size = 0;
    // 当前切出来的height是大于16的，所以只有第二个核需要进行padding补齐
    if (GetSubBlockIdxImpl() == 1) {
        pad_size = height_pad_size;
    }
    int dbFactor = 2;
    auto usedUbSize = TOTAL_UB_SIZE / dbFactor;

    // height is half alignedSingleCoreM
    uint32_t nBurst = (usedUbSize / 2) / (widthAligned * sizeof(T));
    uint32_t nBurstTimes = height / nBurst;
    uint32_t nBurstTail = height - nBurstTimes * nBurst;
    uint32_t c0Size;
    if (sizeof(T) == sizeof(float)) {
        c0Size = 8;
    } else if (sizeof(T) == sizeof(int8_t)) {
        c0Size = 32;
    } else {
        c0Size = 16;
    }

    set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
    set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(1));
    uint8_t pingpongEventId = 0;
    LocalTensor<T> ubPingPong = tmpUbPing;
    LocalTensor<T> transUbPingPong = transUbPing;

    uint32_t gmSrcGap = (ori_width - width) * sizeof(T);
    set_mask_count();
    uint64_t mask_count = c0Size * nBurst;
    set_vector_mask(0, mask_count);
    for (int i = 0; i < nBurstTimes; ++i) {
        if ((i & 1) == 0) {
            pingpongEventId = 0;
            ubPingPong = tmpUbPing;
            transUbPingPong = transUbPing;
        } else {
            pingpongEventId = 1;
            ubPingPong = tmpUbPong;
            transUbPingPong = transUbPong;
        }

        wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(pingpongEventId));
        CopyPadNd2Nz(mmWorkspace[i * nBurst * c0Size], ubPingPong, transUbPingPong, mmGlobal[i * nBurst * width],
                     nBurst, gmSrcGap, width, widthAligned, height, 0, pingpongEventId);
        set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(pingpongEventId));
    }
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    if ((nBurstTimes & 1) == 0) {
        pingpongEventId = 0;
        ubPingPong = tmpUbPing;
        transUbPingPong = transUbPing;
    } else {
        pingpongEventId = 1;
        ubPingPong = tmpUbPong;
        transUbPingPong = transUbPong;
    }

    wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
    wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(1));
    if (nBurstTail > 0) {
        set_mask_count();
        mask_count = c0Size * nBurstTail;
        set_vector_mask(0, mask_count);
        CopyPadNd2Nz(mmWorkspace[nBurstTimes * nBurst * c0Size], ubPingPong, transUbPingPong,
                     mmGlobal[nBurstTimes * nBurst * width], nBurstTail, gmSrcGap, width, widthAligned, height,
                     pad_size, pingpongEventId);
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
}

#else

template <class T>
__aicore__ inline void CopyPadNd2Nz(const GlobalTensor<T> &outputGlobal, const LocalTensor<T> &tmpUb,
                                    const LocalTensor<T> &transUb, const GlobalTensor<T> &inputGlobal, uint32_t nBurst,
                                    uint32_t gmSrcGap, uint32_t inputWidth, uint32_t outputWidth, uint32_t height,
                                    uint16_t pad_size) {
    // use vmuls to nd2nz
    uint32_t c0Size;
    if (sizeof(T) == sizeof(float)) {
        c0Size = 8;
    } else if (sizeof(T) == sizeof(int8_t)) {
        c0Size = 32;
    } else {
        c0Size = 16;
    }
    uint16_t realDataBurst = static_cast<uint16_t>(nBurst);
    // 需要在高方向补pad 并防止反转
    if (pad_size != 0 && nBurst > pad_size) {
        realDataBurst -= pad_size;
    }
    uint8_t rightPadding = 0;
    if (outputWidth != inputWidth) {
        rightPadding = outputWidth - inputWidth;
        set_mov_pad_val(0);
    }
    CopyGmToUbufAlignMc2<T>((__ubuf__ void*)tmpUb.GetPhyAddr(),
                        (__gm__ void*)inputGlobal.GetPhyAddr(), 0, static_cast<uint16_t>(realDataBurst),
                        static_cast<uint16_t>(inputWidth * sizeof(T)), 0, rightPadding, gmSrcGap, 0);
    set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
    wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
    uint64_t mask_count = c0Size * nBurst;
    if (pad_size != 0 && nBurst > pad_size) {
        auto pad_offset = (nBurst - pad_size) * outputWidth;
        if constexpr (sizeof(T) == sizeof(int8_t)) {
            mask_count /= 2;
            set_vector_mask(0, pad_size * outputWidth / 2);
            DuplicateIntrinsicsImpl((__ubuf__ uint16_t*)tmpUb[pad_offset].GetPhyAddr(), (uint16_t)0, 1, 1, 8);
        } else {
            set_vector_mask(0, pad_size * outputWidth);
            DuplicateIntrinsicsImpl((__ubuf__ T*)tmpUb[pad_offset].GetPhyAddr(), (T)0, 1, 1, 8);
        }
        pipe_barrier(PIPE_V);
        set_vector_mask(0, mask_count);
    }

    int widthStep = Ceil(inputWidth, c0Size);  // 行方向搬运多少次
    uint16_t dstBlkStride = 1;
    uint16_t srcBlkStride = widthStep;
    uint16_t dstRepStride = nBurst;
    uint16_t srcRepStride = 1;
    int VCOPY_MAX_REPEAT = 255;
    int split_num = widthStep / VCOPY_MAX_REPEAT;
    int tail_num = widthStep % VCOPY_MAX_REPEAT;
    int dstOffset = 0;
    int srcOffset = 0;

    for (int i = 0; i < split_num; ++i) {
        dstOffset = VCOPY_MAX_REPEAT * mask_count;
        srcOffset = VCOPY_MAX_REPEAT * c0Size;
        vcopy((__ubuf__ uint16_t *)transUb[i * dstOffset].GetPhyAddr(),
              (__ubuf__ uint16_t *)tmpUb[i * srcOffset].GetPhyAddr(), (uint8_t)VCOPY_MAX_REPEAT, dstBlkStride,
              srcBlkStride, dstRepStride, srcRepStride);
    }
    if (tail_num != 0) {
        dstOffset = VCOPY_MAX_REPEAT * mask_count * split_num;
        srcOffset = VCOPY_MAX_REPEAT * c0Size * split_num;
        vcopy((__ubuf__ uint16_t *)transUb[dstOffset].GetPhyAddr(), (__ubuf__ uint16_t *)tmpUb[srcOffset].GetPhyAddr(),
              (uint8_t)tail_num, dstBlkStride, srcBlkStride, dstRepStride, srcRepStride);
    }
    // 要在k方向做repeat切分
    set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    DataCopyParams copyOutParams;
    copyOutParams.blockLen = nBurst;
    copyOutParams.blockCount = widthStep;  // 最大4095， 需要切分
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = height * 2 - nBurst;  // 分了两个核，所以需要在m方向上跳转一大块
    DataCopy(outputGlobal, transUb, copyOutParams);
}

template <class T>
__aicore__ inline void PrePaddingImplNd2Nz(const GlobalTensor<T> &mmWorkspace, const GlobalTensor<T> &mmGlobal,
                                           uint32_t height, uint32_t width, uint32_t widthAligned, int height_pad_size,
                                           int ori_width, TBuf<TPosition::VECCALC> &totalUbBuf) {
    int pad_size = 0;
    // 当前切出来的height是大于16的，所以只有第二个核需要进行padding补齐
    if (GetSubBlockIdxImpl() == 1) {
        pad_size = height_pad_size;
    }
    // height is half alignedSingleCoreM
    LocalTensor<T> fullUbTensor = totalUbBuf.Get<T>();
    LocalTensor<T> srcUb = fullUbTensor[0];
    LocalTensor<T> transUb = fullUbTensor[(TOTAL_UB_SIZE / 2) / sizeof(T)];
    uint32_t nBurst = (TOTAL_UB_SIZE / 2) / (widthAligned * sizeof(T));
    uint32_t nBurstTimes = height / nBurst;
    uint32_t nBurstTail = height - nBurstTimes * nBurst;
    uint32_t c0Size;
    if (sizeof(T) == sizeof(float)) {
        c0Size = 8;
    } else if (sizeof(T) == sizeof(int8_t)) {
        c0Size = 32;
    } else {
        c0Size = 16;
    }
    // N > 491520时，UB的大小每次只够一行数据进行nd2nz，所以nBurstTime的搬运次数需要减去补的pad
    if (GetSubBlockIdxImpl() == 1 && nBurst == 1) {
        nBurstTimes -= pad_size;
        height -= pad_size;
    }

    set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
    uint32_t gmSrcGap = (ori_width - width) * sizeof(T);

    set_mask_count();
    uint64_t mask_count = c0Size * nBurst;
    set_vector_mask(0, mask_count);
    for (int i = 0; i < nBurstTimes; ++i) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
        CopyPadNd2Nz(mmWorkspace[i * nBurst * c0Size], srcUb, transUb, mmGlobal[i * nBurst * ori_width], nBurst,
                     gmSrcGap, width, widthAligned, height, 0);
        set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
    }

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
    if (nBurstTail > 0) {
        set_mask_count();
        mask_count = c0Size * nBurstTail;
        set_vector_mask(0, mask_count);
        CopyPadNd2Nz(mmWorkspace[nBurstTimes * nBurst * c0Size], srcUb, transUb,
                     mmGlobal[nBurstTimes * nBurst * ori_width], nBurstTail, gmSrcGap, width, widthAligned, height,
                     pad_size);
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
}

#endif

template <class T>
__aicore__ inline void MatrixND2NZ(GM_ADDR outGm, GM_ADDR srcGm, uint32_t high, uint32_t width, uint32_t orgWidth,
                                   TBuf<TPosition::VECCALC> &totalUbBuf) {
    const uint32_t alignRow = 16;
    int size = DivCeil(high, alignRow) * DivCeil(width, alignRow) * 256;
    auto alignedNSize = DivCeil(width, alignRow) * alignRow;

    uint32_t c0Size;
    if (sizeof(T) == sizeof(float)) {
        c0Size = 8;
    } else if (sizeof(T) == sizeof(int8_t)) {
        c0Size = 32;
    } else {
        c0Size = 16;
    }

    GlobalTensor<T> tempSrcGlobal;
    GlobalTensor<T> tempDtsGlobal;
    tempDtsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(outGm), size);
    tempSrcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), size);

    auto alignedKSize = DivCeil(high, alignRow) * alignRow;
    int height_pad_size = alignedKSize - high;
    PrePaddingImplNd2Nz<T>(tempDtsGlobal[GetSubBlockIdxImpl() * c0Size * alignedKSize / 2],
                           tempSrcGlobal[GetSubBlockIdxImpl() * orgWidth * alignedKSize / 2], alignedKSize / 2, width,
                           alignedNSize, height_pad_size, orgWidth, totalUbBuf);
}

template <class T>
__aicore__ inline void MatrixBtoNZ(GM_ADDR workspace, GM_ADDR src, const RCSTiling &cfg,
                                   TBuf<TPosition::VECCALC> &totalUbBuf)
{
    if (g_coreType == AIV) {
        auto alignedNSize = Ceil(cfg.rankN, (uint32_t)16) * 16;  // N轴转换成分型
        auto alignedKSize = Ceil(cfg.rankK, (uint32_t)16) * 16;  // K轴转换成分型
        auto spliteWidth = cfg.rankN;                            // 切N轴时用N
        if (cfg.isTransposeB) {
            spliteWidth = cfg.rankK;
        }  // 切K轴时用K
        // rangN取分形块数， 如1920 则生成120个16分形
        auto fractalPerNum = Ceil(spliteWidth, (uint32_t)16);
        auto userCodeNum = cfg.aicCoreNum;  // 使用最大的核数

        uint32_t oneBlockFactalNum = Ceil(fractalPerNum, userCodeNum);  // 每个core需要计算的分型
        // 本核需要计算的分型开始位置
        int32_t curBlockCount = (fractalPerNum - oneBlockFactalNum * block_idx);
        if (curBlockCount <= 0) {
            ffts_cross_core_sync(PIPE_MTE3, 0x21 + (5 << 8));  // v侧做完才能做c侧
            return;
        }
        // 本核计算的分型长度
        int32_t oneBlock;
        if (curBlockCount <= oneBlockFactalNum) {
            oneBlock = (spliteWidth - block_idx * oneBlockFactalNum * 16);  // 当前为尾核
        } else {
            oneBlock = oneBlockFactalNum * 16;
        }

        if (cfg.isTransposeB) {
            uint64_t workspaceLen = alignedNSize * oneBlockFactalNum * 16 * sizeof(T);
            GM_ADDR gmBTrans = workspace + block_idx * workspaceLen;
            auto bTemp = src + block_idx * oneBlockFactalNum * 16 * sizeof(T);
            MatrixND2NZ<T>(gmBTrans, bTemp, cfg.rankN, oneBlock, cfg.rankK, totalUbBuf);
        } else {
            uint64_t workspaceLen = alignedKSize * oneBlockFactalNum * 16 * sizeof(T);
            GM_ADDR gmBTrans = workspace + block_idx * workspaceLen;
            auto bTemp = src + block_idx * oneBlockFactalNum * 16 * sizeof(T);
            MatrixND2NZ<T>(gmBTrans, bTemp, cfg.rankK, oneBlock, cfg.rankN, totalUbBuf);
        }
        // 先AIC 等待AIV, 再AIC之间一次同步
        ffts_cross_core_sync(PIPE_MTE3, 0x21 + (5 << 8));  // v侧做完才能做c侧
    } else {
#ifndef __CCE_KT_TEST__
        wait_flag_dev(5);
        ffts_cross_core_sync(PIPE_MTE3, 0x01 + (4 << 8));
        wait_flag_dev(4);
#endif
    }
}

#endif  // ENALBE_ND2NZ

__aicore__ inline void CastBFtoFloat(GM_ADDR dst, GM_ADDR src, int size, TBuf<TPosition::VECCALC> &totalUbBuf) {
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) { // 先aiv，后aic
    #ifndef __CCE_KT_TEST__
        wait_flag_dev(EVENT_ID_5);
        ffts_cross_core_sync(PIPE_MTE3, 0x01 + (EVENT_ID_4<<MOVE_LEFT_SIZE));
        wait_flag_dev(EVENT_ID_4);
        return;
    #endif
    }
    if (GetBlockIdx() != 0) {
        ffts_cross_core_sync(PIPE_MTE3, 0x21 + (EVENT_ID_5<<MOVE_LEFT_SIZE));
        return;
    }

    // 1. 初始化global tensor
    GlobalTensor<bfloat16_t> gmSrc;
    GlobalTensor<float> gmDst;
    gmSrc.SetGlobalBuffer((__gm__ bfloat16_t*)(src), size);
    gmDst.SetGlobalBuffer((__gm__ float*)(dst), size);

    // 2. 初始化local tensor
    LocalTensor<bfloat16_t> fullBf16 = totalUbBuf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> xLocal = fullBf16[0];
    LocalTensor<float> yLocal = fullBf16[Ceil(size, UB_ALIGN_SIZE) * UB_ALIGN_SIZE].template ReinterpretCast<float>();

    // 3. GM数据拷贝至UB
    uint16_t cpInLen = size * sizeof(bfloat16_t);
    DataCopyParams cpInParams{1, cpInLen, 0, 0};
    DataCopyPadParams padParams{false, 0, 0, 0}; // 不需要填充数据
    DataCopyPad(xLocal, gmSrc, cpInParams, padParams);

    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    // 4. 进行cast转换
    Cast(yLocal, xLocal, RoundMode::CAST_NONE, size);
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    // 5. UB数据拷贝至GM
    uint16_t cpOutLen = size * sizeof(float);
    DataCopyParams cpOutParams{1, cpOutLen, 0, 0};
    DataCopyPad(gmDst, yLocal, cpOutParams);
    ffts_cross_core_sync(PIPE_MTE3, 0x21 + (EVENT_ID_5<<MOVE_LEFT_SIZE));
#endif
}
#endif  // MC2_GATHER_COMM_H
}
