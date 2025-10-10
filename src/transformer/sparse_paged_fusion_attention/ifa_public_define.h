/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ifa_public_define.h
 * \brief
 */
#ifndef FUSION_IFA_PUBLIC_DEFINE_H
#define FUSION_IFA_PUBLIC_DEFINE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::SoftmaxConfig;
using AscendC::WaitFlag;
using matmul::Matmul;
using matmul::MatmulType;

#define FLT_MAX 3.402823466e+38F

constexpr MatmulConfig CFG_NORM_EXCEED = GetNormalConfig(true);
constexpr MatmulConfig CFG_MDL_EXCEED = GetMDLConfig(true);

// CFG_NORM_EXCEED_INIT: doNorm, enable intrinsicsCheck and Init
constexpr MatmulConfig CFG_NORM_EXCEED_INIT{.doNorm = true,
                                            .doBasicBlock = false,
                                            .doMultiDataLoad = false,
                                            .basicM = 0,
                                            .basicN = 0,
                                            .basicK = 0,
                                            .intrinsicsCheck = true,
                                            .isNBatch = false,
                                            .enVecND2NZ = false,
                                            .doSpecialBasicBlock = false,
                                            .doMTE2Preload = false,
                                            .singleCoreM = 0,
                                            .singleCoreN = 0,
                                            .singleCoreK = 0,
                                            .stepM = 0,
                                            .stepN = 0,
                                            .baseMN = 0,
                                            .singleCoreMN = 0,
                                            .enUnitFlag = true,
                                            .isPerTensor = false,
                                            .hasAntiQuantOffset = false,
                                            .doIBShareNorm = false,
                                            .doSpecialMDL = false,
                                            .enableInit = false,
                                            .batchMode = BatchMode::NONE,
                                            .enableEnd = false,
                                            .enableGetTensorC = false,
                                            .enableSetOrgShape = true,
                                            .enableSetBias = false,
                                            .enableSetTail = true,
                                            .enableQuantVector = false,
                                            .enableSetDefineData = true,
                                            .iterateMode = IterateMode::ITERATE_MODE_ALL};

// CFG_MDL_EXCEED_INIT: enable MDL, intrinsicsCheck and Init
constexpr MatmulConfig CFG_MDL_EXCEED_INIT{.doNorm = false,
                                           .doBasicBlock = false,
                                           .doMultiDataLoad = true,
                                           .basicM = 0,
                                           .basicN = 0,
                                           .basicK = 0,
                                           .intrinsicsCheck = true,
                                           .isNBatch = false,
                                           .enVecND2NZ = false,
                                           .doSpecialBasicBlock = false,
                                           .doMTE2Preload = false,
                                           .singleCoreM = 0,
                                           .singleCoreN = 0,
                                           .singleCoreK = 0,
                                           .stepM = 0,
                                           .stepN = 0,
                                           .baseMN = 0,
                                           .singleCoreMN = 0,
                                           .enUnitFlag = false,
                                           .isPerTensor = false,
                                           .hasAntiQuantOffset = false,
                                           .doIBShareNorm = false,
                                           .doSpecialMDL = false,
                                           .enableInit = false,
                                           .batchMode = BatchMode::NONE,
                                           .enableEnd = false,
                                           .enableGetTensorC = false,
                                           .enableSetOrgShape = true,
                                           .enableSetBias = false,
                                           .enableSetTail = true,
                                           .enableQuantVector = false,
                                           .enableSetDefineData = true,
                                           .iterateMode = IterateMode::ITERATE_MODE_ALL};

// CFG_MDL_EXCEED_INIT_CALLBACK: enable MDL, intrinsicsCheck and Init, enable CALLBACK, enable unitflag
constexpr MatmulConfig CFG_MDL_EXCEED_INIT_CALLBACK{.doNorm = false,
                                                    .doBasicBlock = false,
                                                    .doMultiDataLoad = true,
                                                    .basicM = 0,
                                                    .basicN = 0,
                                                    .basicK = 0,
                                                    .intrinsicsCheck = true,
                                                    .isNBatch = false,
                                                    .enVecND2NZ = false,
                                                    .doSpecialBasicBlock = false,
                                                    .doMTE2Preload = false,
                                                    .singleCoreM = 0,
                                                    .singleCoreN = 0,
                                                    .singleCoreK = 0,
                                                    .stepM = 0,
                                                    .stepN = 0,
                                                    .baseMN = 0,
                                                    .singleCoreMN = 0,
                                                    .enUnitFlag = true, // enable unitflag
                                                    .isPerTensor = false,
                                                    .hasAntiQuantOffset = false,
                                                    .doIBShareNorm = false,
                                                    .doSpecialMDL = false,
                                                    .enableInit = false,
                                                    .batchMode = BatchMode::NONE,
                                                    .enableEnd = false,
                                                    .enableGetTensorC = false,
                                                    .enableSetOrgShape = true,
                                                    .enableSetBias = false,
                                                    .enableSetTail = true,
                                                    .enableQuantVector = false,
                                                    .enableSetDefineData = true,
                                                    .iterateMode = IterateMode::ITERATE_MODE_ALL,
                                                    .enableReuse = false};

constexpr SoftmaxConfig IFA_SOFTMAX_FLASHV2_CFG = {false}; // 将isCheckTiling设置为false

constexpr float FLOAT_ZERO = 0;
constexpr float FLOAT_MAX = FLT_MAX;

constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;

constexpr uint32_t MAX_UINT16 = 65535;
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t REPEAT_BLOCK_BYTE = 256;
constexpr uint32_t IFA_MAX_REPEAT_TIMES = 256;

#define VMLA_ONE_REPEATE_ROW_COUNT 4
#define VMLA_ONE_REPEATE_COLUMN_COUNT 16
#define FP16_ONE_BLOCK_SIZE 16
#define FP32_ONE_BLOCK_SIZE 8
#define ALIGN_BLOCK_SIZE 16

enum class LAYOUT {
    BSH = 0,
    SBH,
    BNSD,
    BSND
};

template <typename Q_T, typename KV_T, typename OUT_T, typename ORIGIN_T, const bool PAGE_ATTENTION = false,
          const bool FLASH_DECODE = false, LAYOUT LAYOUT_T = LAYOUT::BSH, const uint8_t ANTIQUANT_MODE = 0,
          const bool SHARED_PREFIX = false, typename... Args>
struct IFAType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    using orginalType = ORIGIN_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr LAYOUT layout = LAYOUT_T;
    static constexpr uint8_t antiquantMode = ANTIQUANT_MODE;
    static constexpr bool sharedPrefix = SHARED_PREFIX;
};

constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

__aicore__ inline void VecMulMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec mul by row
    // dstUb[i, j] = src0Ub[j] * src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, actualColumnCount] dstUb:[dealRowCount, columnCount]
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParams.src0RepStride = 0;
    repeatParams.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t loopCount = actualColumnCount / mask;
    uint32_t remainCount = actualColumnCount % mask;
    uint32_t offset = 0;
    for (int i = 0; i < loopCount; i++) {
    // offset = i * mask
        Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
        offset += mask;
    }
    if (remainCount > 0) {
    // offset = loopCount * mask
        Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
    }
}

__aicore__ inline void VecAddMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec add by row
    // dstUb[i, j] = src0Ub[j] + src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, columnCount] dstUb:[dealRowCount, columnCount]
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParams.src0RepStride = 0;
    repeatParams.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t loopCount = actualColumnCount / mask;
    uint32_t remainCount = actualColumnCount % mask;

    uint64_t offset = 0;
    for (int i = 0; i < loopCount; i++) {
        Add(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
        offset += mask;
    }
    if (remainCount > 0) {
        Add(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
    }
}

__aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

__aicore__ inline void RowMuls(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;

    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            // offset = i * dtypeMask
            Mul(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParams);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask,
                columnRepeatCount, columnRepeatParams);
        }
    }

    if (dRemain > 0) {
        Mul(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParams);
    }
}

__aicore__ inline void RowSum(LocalTensor<float> &dstUb, LocalTensor<float> srcUb, uint32_t dealRowCount,
                              uint32_t columnCount, uint32_t actualColumnCount)
{
    // sum by row, 按行求和
    // dstUb[i] = sum(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockCount = actualColumnCount / dtypeMask;
    uint32_t remain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / (BYTE_BLOCK / sizeof(float));
    repeatParamsMax.src1RepStride = columnCount / (BYTE_BLOCK / sizeof(float));
    repeatParamsMax.dstRepStride = columnCount / (BYTE_BLOCK / sizeof(float));
    if (blockCount > 0 && remain > 0) {
        Add(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, dealRowCount, repeatParamsMax);
        pipe_barrier(PIPE_V);
    }

    for (uint32_t loopCount = blockCount / 2; loopCount > 0; loopCount = blockCount / 2) {
        blockCount = (blockCount + 1) / 2;
        for (uint32_t j = 0; j < loopCount; j++) {
            Add(srcUb[j * dtypeMask], srcUb[j * dtypeMask], srcUb[(j + blockCount) * dtypeMask], dtypeMask,
                dealRowCount, repeatParamsMax);
        }
        pipe_barrier(PIPE_V);
    }

    WholeReduceSum(dstUb, srcUb, (actualColumnCount < dtypeMask) ? actualColumnCount : dtypeMask, dealRowCount, 1, 1,
                   columnCount / (BYTE_BLOCK / sizeof(float)));
}

__aicore__ inline void RowMax(LocalTensor<float> &dstUb, LocalTensor<float> &srcUb, uint32_t dealRowCount,
                              uint32_t columnCount, uint32_t actualColumnCount)
{
    // max by row, 按行求最大值
    // dstUb[i] = max(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockCount = actualColumnCount / dtypeMask;
    uint32_t remain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    if (blockCount > 0 && remain > 0) {
        Max(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, dealRowCount, repeatParamsMax);
        pipe_barrier(PIPE_V);
    }

    for (uint32_t loopCount = blockCount / 2; loopCount > 0; loopCount = blockCount / 2) {
        blockCount = (blockCount + 1) / 2;
        for (uint32_t j = 0; j < loopCount; j++) {
            Max(srcUb[j * dtypeMask], srcUb[j * dtypeMask], srcUb[(j + blockCount) * dtypeMask], dtypeMask,
                dealRowCount, repeatParamsMax);
        }
        pipe_barrier(PIPE_V);
    }

    WholeReduceMax(dstUb, srcUb, (actualColumnCount < dtypeMask) ? actualColumnCount : dtypeMask, dealRowCount, 1, 1,
                   columnCount / FP32_BLOCK_ELEMENT_NUM, ReduceOrder::ORDER_ONLY_VALUE);
}

#endif // FUSION_IFA_PUBLIC_DEFINE_H