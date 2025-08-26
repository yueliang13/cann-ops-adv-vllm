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
 * \file moe_gating_top_k_softmax_v2_perf.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_SOFTMAX_V2_PERF
#define MOE_GATING_TOP_K_SOFTMAX_V2_PERF

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace MoeGatingTopKSoftmaxV2 {
using namespace AscendC;

constexpr int32_t DB_BUFFER_NUM = 2;
constexpr int32_t BLOCK_BYTES = 32;
constexpr int32_t BLOCK_B32_SIZE = 8;
constexpr int32_t REPEAT_B32_SIZE = 64;
constexpr int32_t REDUCE_MAX_SIZE = 64;
constexpr int32_t CONSTANT_TWO = 2;
constexpr int32_t CONSTANT_THREE = 3;
constexpr int32_t CONSTANT_FOUR = 4;
constexpr int32_t CONSTANT_FIVE = 5;
constexpr int32_t CONSTANT_SIX = 6;
constexpr int32_t CONSTANT_SEVEN = 7;
constexpr int32_t CONSTANT_EIGHT = 8;
constexpr int32_t ZERO_MASK = 0b0;
constexpr int32_t SORT_UNIT = 32;
constexpr int32_t MERGE_UNIT = 128;
constexpr int32_t MERGE_LIST_MAX_NUM = 4;
constexpr int32_t MERGE_TWO = 0b0011;
constexpr int32_t MERGE_FOUR = 0b1111;

enum class ColRangeEnum {
  SMALLER_THAN_8 = 0,
  FROM_8_TO_64,
  BIGGER_THAN_64
};

template <typename T, int32_t renorm, ColRangeEnum colRange>
class MoeGatingTopKSoftmaxV2Perf {
 public:
  __aicore__ inline MoeGatingTopKSoftmaxV2Perf(){};
  __aicore__ inline void Init(GM_ADDR gating, GM_ADDR finished, GM_ADDR out, GM_ADDR indicesOut, GM_ADDR softmaxOut,
                              GM_ADDR workspace, const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    // init gm inputs
    int64_t formerblockLength = tilingData->blockFormer * tilingData->col;
    this->softmaxFlag = tilingData->softmaxFlag;
    int64_t blockLength =
        (GetBlockIdx() != tilingData->blockNum - 1) ? formerblockLength : tilingData->blockTail * tilingData->col;
    gatingTensorGM.SetGlobalBuffer((__gm__ T*)gating + formerblockLength * GetBlockIdx(), blockLength);
    if (finished != nullptr) {
      exitFinished = true;
      int64_t blockLengthFinished =
          (GetBlockIdx() != tilingData->blockNum - 1) ? tilingData->blockFormer : tilingData->blockTail;
      finishedTensorGM.SetGlobalBuffer((__gm__ bool*)finished + tilingData->blockFormer * GetBlockIdx(),
                                       blockLengthFinished);
    }
    // init gm outputs
    int64_t outFormerBlockLength = tilingData->blockFormer * tilingData->k;
    int64_t outBlockLength =
        (GetBlockIdx() != tilingData->blockNum - 1) ? outFormerBlockLength : tilingData->blockTail * tilingData->k;
    outTensorGM.SetGlobalBuffer((__gm__ T*)out + outFormerBlockLength * GetBlockIdx(), outBlockLength);
    indicesOutTensorGM.SetGlobalBuffer((__gm__ int32_t*)indicesOut + outFormerBlockLength * GetBlockIdx(),
                                       outBlockLength);
    softmaxResultGM.SetGlobalBuffer((__gm__ float*)softmaxOut + formerblockLength * GetBlockIdx(), blockLength);
    // init queues
    int32_t bufferSize = tilingData->bufferElemSize * sizeof(int32_t);
    pipe.InitBuffer(gatingQueue, DB_BUFFER_NUM, tilingData->bufferElemSize * sizeof(T));
    pipe.InitBuffer(finishedQueue, DB_BUFFER_NUM,
                    (tilingData->ubFormer * sizeof(bool) + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES);
    pipe.InitBuffer(topKOutsQueue, 1, bufferSize * CONSTANT_TWO);
    pipe.InitBuffer(patternTensor, BLOCK_BYTES + BLOCK_BYTES + BLOCK_BYTES);
    pipe.InitBuffer(tmpTensor, bufferSize * CONSTANT_TWO);
    pipe.InitBuffer(softmaxResultOutQueue, 1, bufferSize);
  }

  __aicore__ inline void Process(const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    int32_t ubLoopCount =
        (GetBlockIdx() == tilingData->blockNum - 1) ? tilingData->ubLoopOfTailBlock : tilingData->ubLoopOfFormerBlock;
    int32_t tailRowsNum =
        (GetBlockIdx() == tilingData->blockNum - 1) ? tilingData->ubTailOfTailBlock : tilingData->ubTailOfFormerBlock;
    // preload
    CopyIn(0, (0 == ubLoopCount - 1) ? tailRowsNum : tilingData->ubFormer, tilingData);
    ComputePhase0(tilingData);
    for (int32_t i = 0; i < ubLoopCount - 1; i++) {
      ComputePhase1(i, tilingData->ubFormer, tilingData);
      CopyIn(i + 1, (i == ubLoopCount - 1 - 1) ? tailRowsNum : tilingData->ubFormer, tilingData);
      CopyOutPhase0(i, tilingData->ubFormer, tilingData);
    }
    ComputePhase1(ubLoopCount - 1, tailRowsNum, tilingData);
    CopyOutPhase0(ubLoopCount - 1, tailRowsNum, tilingData);
  }

 private:
  __aicore__ inline void ArithProgressionPerf(const LocalTensor<int32_t>& dst, const int32_t firstValue,
                                              const int32_t diffValue, const int32_t countAlign) {
    // countAlign must be eight aligned
    dst.SetValue(0, firstValue);
    dst.SetValue(1, firstValue + diffValue * 1);
    dst.SetValue(CONSTANT_TWO, firstValue + diffValue * CONSTANT_TWO);
    dst.SetValue(CONSTANT_THREE, firstValue + diffValue * CONSTANT_THREE);
    dst.SetValue(CONSTANT_FOUR, firstValue + diffValue * CONSTANT_FOUR);
    dst.SetValue(CONSTANT_FIVE, firstValue + diffValue * CONSTANT_FIVE);
    dst.SetValue(CONSTANT_SIX, firstValue + diffValue * CONSTANT_SIX);
    dst.SetValue(CONSTANT_SEVEN, firstValue + diffValue * CONSTANT_SEVEN);
    auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID);
    WaitFlag<HardEvent::S_V>(eventID);
    if (countAlign > BLOCK_B32_SIZE) {
      int32_t offset;
      if (countAlign > REPEAT_B32_SIZE) {
        for (int32_t i = 1; i < BLOCK_B32_SIZE; i++) {
          offset = i * BLOCK_B32_SIZE;
          Adds(dst[offset], dst, diffValue * offset, BLOCK_B32_SIZE, 1, {1, 1, 1, 1});
        }
        pipe_barrier(PIPE_V);
        int32_t loopTimes = (countAlign + REPEAT_B32_SIZE - 1) / REPEAT_B32_SIZE;
        for (int32_t i = 1; i < loopTimes - 1; i++) {
          offset = i * REPEAT_B32_SIZE;
          Adds(dst[offset], dst, diffValue * offset, REPEAT_B32_SIZE, 1, {1, 1, 1, 1});
        }
        offset = (loopTimes - 1) * REPEAT_B32_SIZE;
        Adds(dst[offset], dst, diffValue * offset, countAlign - offset, 1, {1, 1, 1, 1});
      } else {
        for (int32_t i = 1; i < countAlign / BLOCK_B32_SIZE; i++) {
          offset = i * BLOCK_B32_SIZE;
          Adds(dst[offset], dst, diffValue * offset, BLOCK_B32_SIZE, 1, {1, 1, 1, 1});
        }
      }
      pipe_barrier(PIPE_V);
    }
  }

  __aicore__ inline void ComputePhase0(const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    // currently, only k <= 8 is supported in this class, so one uint32 number is enough to do GatherMask
    LocalTensor<uint32_t> patternTensorLocal = patternTensor.Get<uint32_t>();
    // select topk values after VMS
    patternTensorLocal.SetValue(0, tilingData->topKValuesMask);
    patternTensorLocal.SetValue(1, ZERO_MASK);
    // select topk indices after VMS
    patternTensorLocal.SetValue(BLOCK_B32_SIZE, tilingData->topKIndicesMask);
    patternTensorLocal.SetValue(BLOCK_B32_SIZE + 1, ZERO_MASK);
    // select k values
    patternTensorLocal.SetValue(BLOCK_B32_SIZE + BLOCK_B32_SIZE, (1 << tilingData->k) - 1);
    auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID);
    WaitFlag<HardEvent::S_V>(eventID);
  }

  __aicore__ inline void CopyIn(const int32_t OuterIdx, const int32_t curRowsNum,
                                const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    LocalTensor<T> gatingLocal = gatingQueue.AllocTensor<T>();
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams intriParams;
    if (tilingData->colBytesAlign - tilingData->col != 0) {
      intriParams.blockCount = curRowsNum;
      intriParams.blockLen = tilingData->col * sizeof(T);
    } else {
      intriParams.blockCount = 1;
      intriParams.blockLen = curRowsNum * tilingData->col * sizeof(T);
    }
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
#ifndef __CCE_KT_TEST__
    DataCopyPad(gatingLocal, gatingTensorGM[tilingData->ubFormer * tilingData->col * OuterIdx], intriParams, padParams);
#endif
    gatingQueue.EnQue(gatingLocal);

    if (exitFinished) {
        LocalTensor<bool> finishedLocal = finishedQueue.AllocTensor<bool>();
        DataCopyParams intriParamsFinished;
        intriParamsFinished.blockCount = 1;
        intriParamsFinished.blockLen = curRowsNum * sizeof(bool);
        intriParamsFinished.srcStride = 0;
        intriParamsFinished.dstStride = 0;
#ifndef __CCE_KT_TEST__
        DataCopyPad(finishedLocal, finishedTensorGM[tilingData->ubFormer * OuterIdx], intriParamsFinished, padParams);
#endif
        finishedQueue.EnQue(finishedLocal);
    }
  }

  __aicore__ inline void ReduceMaxFP32Perf(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                           const LocalTensor<float>& tmpBuffer, const int32_t curRowsNum,
                                           const uint32_t colBytesAlign, const uint32_t col) {
    uint32_t tmp = 0xFF800000;          // -inf
    DuplicatePadValue(src, curRowsNum, colBytesAlign, col, *((float *)&tmp));
    if constexpr (colRange == ColRangeEnum::SMALLER_THAN_8 || renorm == 1) {
      AscendCUtils::SetMaskCount<float>();
      set_vector_mask(0, curRowsNum * BLOCK_B32_SIZE);
      BlockReduceMaxIntrinsicsImpl((__ubuf__ float*)dst.GetPhyAddr(), (__ubuf__ float*)src.GetPhyAddr(), 1, 1, 1,
                                   CONSTANT_EIGHT);
      AscendCUtils::SetMaskNorm<float>();
    } else if constexpr (colRange == ColRangeEnum::FROM_8_TO_64) {
      WholeReduceMax(dst, src, colBytesAlign, curRowsNum, 1, 1, colBytesAlign / CONSTANT_EIGHT,
                     ReduceOrder::ORDER_ONLY_VALUE);
    } else {
      int32_t loopTimes = (colBytesAlign + REPEAT_B32_SIZE - 1) / REPEAT_B32_SIZE;
      DataCopyParams copyIntriParams;
      // copy first REPEAT_B32_SIZE elements of each row
      copyIntriParams.blockCount = curRowsNum;
      copyIntriParams.blockLen = CONSTANT_EIGHT;
      copyIntriParams.srcStride = colBytesAlign / CONSTANT_EIGHT - CONSTANT_EIGHT;
      copyIntriParams.dstStride = 0;
      DataCopy(dst, src, copyIntriParams);
      pipe_barrier(PIPE_V);
      BinaryRepeatParams intriParams;
      intriParams.dstBlkStride = 1;
      intriParams.src0BlkStride = 1;
      intriParams.src1BlkStride = 1;
      intriParams.dstRepStride = CONSTANT_EIGHT;
      intriParams.src0RepStride = CONSTANT_EIGHT;
      intriParams.src1RepStride = colBytesAlign / CONSTANT_EIGHT;
      for (int32_t i = 1; i < loopTimes - 1; i++) {
        Max(dst, dst, src[i * REPEAT_B32_SIZE], REPEAT_B32_SIZE, curRowsNum, intriParams);
        pipe_barrier(PIPE_V);
      }
      Max(dst, dst, src[(loopTimes - 1) * REPEAT_B32_SIZE], col - (loopTimes - 1) * REPEAT_B32_SIZE,
          curRowsNum, intriParams);
      pipe_barrier(PIPE_V);
      AscendCUtils::SetMaskCount<float>();
      set_vector_mask(0, curRowsNum * REPEAT_B32_SIZE);
      BlockReduceMaxIntrinsicsImpl((__ubuf__ float*)tmpBuffer.GetPhyAddr(), (__ubuf__ float*)dst.GetPhyAddr(), 1, 1, 1,
                                   CONSTANT_EIGHT);
      pipe_barrier(PIPE_V);
      set_vector_mask(0, curRowsNum * BLOCK_B32_SIZE);
      BlockReduceMaxIntrinsicsImpl((__ubuf__ float*)dst.GetPhyAddr(), (__ubuf__ float*)tmpBuffer.GetPhyAddr(), 1, 1, 1,
                                   CONSTANT_EIGHT);
      AscendCUtils::SetMaskNorm<float>();
    }
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void ReduceSumFP32Perf(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                           const LocalTensor<float>& tmpBuffer, const int32_t curRowsNum,
                                           const uint32_t colBytesAlign, const uint32_t col) {
    DuplicatePadValue(src, curRowsNum, colBytesAlign, col, 0.0f);
    if constexpr (colRange == ColRangeEnum::SMALLER_THAN_8 || renorm == 1) {
      AscendCUtils::SetMaskCount<float>();
      set_vector_mask(0, curRowsNum * BLOCK_B32_SIZE);
      BlockReduceSumIntrinsicsImpl((__ubuf__ float*)dst.GetPhyAddr(), (__ubuf__ float*)src.GetPhyAddr(), 1, 1, 1,
                                   CONSTANT_EIGHT);
      AscendCUtils::SetMaskNorm<float>();
    } else if constexpr (colRange == ColRangeEnum::FROM_8_TO_64) {
      WholeReduceSum(dst, src, colBytesAlign, curRowsNum, 1, 1, colBytesAlign / CONSTANT_EIGHT);
    } else {
      int32_t loopTimes = (colBytesAlign + REPEAT_B32_SIZE - 1) / REPEAT_B32_SIZE;
      DataCopyParams copyIntriParams;
      // copy first REPEAT_B32_SIZE elements of each row
      copyIntriParams.blockCount = curRowsNum;
      copyIntriParams.blockLen = CONSTANT_EIGHT;
      copyIntriParams.srcStride = colBytesAlign / CONSTANT_EIGHT - CONSTANT_EIGHT;
      copyIntriParams.dstStride = 0;
      DataCopy(dst, src, copyIntriParams);
      pipe_barrier(PIPE_V);
      BinaryRepeatParams intriParams;
      intriParams.dstBlkStride = 1;
      intriParams.src0BlkStride = 1;
      intriParams.src1BlkStride = 1;
      intriParams.dstRepStride = CONSTANT_EIGHT;
      intriParams.src0RepStride = CONSTANT_EIGHT;
      intriParams.src1RepStride = colBytesAlign / CONSTANT_EIGHT;
      for (int32_t i = 1; i < loopTimes - 1; i++) {
        Add(dst, dst, src[i * REPEAT_B32_SIZE], REPEAT_B32_SIZE, curRowsNum, intriParams);
        pipe_barrier(PIPE_V);
      }
      Add(dst, dst, src[(loopTimes - 1) * REPEAT_B32_SIZE], col - (loopTimes - 1) * REPEAT_B32_SIZE,
          curRowsNum, intriParams);
      pipe_barrier(PIPE_V);
      AscendCUtils::SetMaskCount<float>();
      set_vector_mask(0, curRowsNum * REPEAT_B32_SIZE);
      BlockReduceSumIntrinsicsImpl((__ubuf__ float*)tmpBuffer.GetPhyAddr(), (__ubuf__ float*)dst.GetPhyAddr(), 1, 1, 1,
                                   CONSTANT_EIGHT);
      pipe_barrier(PIPE_V);
      set_vector_mask(0, curRowsNum * BLOCK_B32_SIZE);
      BlockReduceSumIntrinsicsImpl((__ubuf__ float*)dst.GetPhyAddr(), (__ubuf__ float*)tmpBuffer.GetPhyAddr(), 1, 1, 1,
                                   CONSTANT_EIGHT);
      AscendCUtils::SetMaskNorm<float>();
    }
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void SubInlineBrcFP32Perf(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                              const LocalTensor<float>& src1, const LocalTensor<float>& tmpBuffer,
                                              const int32_t curRowsNum, const uint32_t colBytesAlign) {
    Brcb(tmpBuffer, src1, (curRowsNum + CONSTANT_EIGHT - 1) / CONSTANT_EIGHT, {1, CONSTANT_EIGHT});
    pipe_barrier(PIPE_V);
    if constexpr (colRange == ColRangeEnum::SMALLER_THAN_8 || renorm == 1) {
      Sub(dst, src0, tmpBuffer, curRowsNum * colBytesAlign);
    } else {
      BinaryRepeatParams intriParams;
      intriParams.dstBlkStride = 1;
      intriParams.src0BlkStride = 1;
      intriParams.src1BlkStride = 0;
      intriParams.dstRepStride = colBytesAlign / CONSTANT_EIGHT;
      intriParams.src0RepStride = colBytesAlign / CONSTANT_EIGHT;
      intriParams.src1RepStride = 1;
      int32_t loopTimes = (colBytesAlign + REPEAT_B32_SIZE - 1) / REPEAT_B32_SIZE;
      int32_t offset;
      if constexpr (colRange == ColRangeEnum::BIGGER_THAN_64) {      
        for (int32_t i = 0; i < loopTimes - 1; i++) {
          offset = i * REPEAT_B32_SIZE;
          Sub(dst[offset], src0[offset], tmpBuffer, REPEAT_B32_SIZE, curRowsNum, intriParams);
        }
      }
      offset = (loopTimes - 1) * REPEAT_B32_SIZE;
      Sub(dst[offset], src0[offset], tmpBuffer, colBytesAlign - offset, curRowsNum, intriParams);
    }
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void DivInlineBrcFP32Perf(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                              const LocalTensor<float>& src1, const LocalTensor<float>& tmpBuffer,
                                              const int32_t curRowsNum, const uint32_t colBytesAlign) {
    Brcb(tmpBuffer, src1, (curRowsNum + CONSTANT_EIGHT - 1) / CONSTANT_EIGHT, {1, CONSTANT_EIGHT});
    pipe_barrier(PIPE_V);
    if constexpr (colRange == ColRangeEnum::SMALLER_THAN_8 || renorm == 1) {
      Div(dst, src0, tmpBuffer, curRowsNum * colBytesAlign);
    } else {
      BinaryRepeatParams intriParams;
      intriParams.dstBlkStride = 1;
      intriParams.src0BlkStride = 1;
      intriParams.src1BlkStride = 0;
      intriParams.dstRepStride = colBytesAlign / CONSTANT_EIGHT;
      intriParams.src0RepStride = colBytesAlign / CONSTANT_EIGHT;
      intriParams.src1RepStride = 1;
      int32_t loopTimes = (colBytesAlign + REPEAT_B32_SIZE - 1) / REPEAT_B32_SIZE;
      int32_t offset;
      if constexpr (colRange == ColRangeEnum::BIGGER_THAN_64) {
        for (int32_t i = 0; i < loopTimes - 1; i++) {
          offset = i * REPEAT_B32_SIZE;
          Div(dst[offset], src0[offset], tmpBuffer, REPEAT_B32_SIZE, curRowsNum, intriParams);
        }
      }
      offset = (loopTimes - 1) * REPEAT_B32_SIZE;
      Div(dst[offset], src0[offset], tmpBuffer, colBytesAlign - offset, curRowsNum, intriParams);
    }
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void SoftmaxFP32Perf(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                         const LocalTensor<float>& tmpBuffer0, const LocalTensor<float>& tmpBuffer1,
                                         const int32_t curRowsNum, const uint32_t colBytesAlign, const uint32_t col) {
    ReduceMaxFP32Perf(tmpBuffer0, src, tmpBuffer1, curRowsNum, colBytesAlign, col);
    SubInlineBrcFP32Perf(dst, src, tmpBuffer0, tmpBuffer1, curRowsNum, colBytesAlign);
    Exp(dst, dst, curRowsNum * colBytesAlign);
    pipe_barrier(PIPE_V);
    ReduceSumFP32Perf(tmpBuffer0, dst, tmpBuffer1, curRowsNum, colBytesAlign, col);
    DivInlineBrcFP32Perf(dst, dst, tmpBuffer0, tmpBuffer1, curRowsNum, colBytesAlign);
  }

  __aicore__ inline void Rearrange(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                   const int32_t curRowsNum,
                                   const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    // softmax is 32 bytes aligned while topk must be 32 element aligned
    DataCopyParams intriParams;
    intriParams.blockCount = curRowsNum;
    intriParams.blockLen = tilingData->colBytesAlign / CONSTANT_EIGHT;
    intriParams.srcStride = 0;
    intriParams.dstStride = (tilingData->colAlign - tilingData->colBytesAlign) / CONSTANT_EIGHT;
    DataCopy(dst, src, intriParams);
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void DuplicatePad(const LocalTensor<float>& dst, const int32_t curRowsNum,
                                      const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    if (tilingData->colAlign - tilingData->col != 0) {
      uint32_t tmp = 0xFF800000;                  // -inf
      uint64_t mask[2] = {(((uint64_t)1 << (tilingData->colAlign - tilingData->col)) - 1)
                              << (SORT_UNIT - (tilingData->colAlign - tilingData->col)),
                          0};
      Duplicate(dst[tilingData->colAlign - SORT_UNIT], *((float *)&tmp), mask, curRowsNum, 1,
                tilingData->colAlign / CONSTANT_EIGHT);
      pipe_barrier(PIPE_V);
    }
  }

  __aicore__ inline void DuplicatePadValue(const LocalTensor<float>& dst, const int32_t curRowsNum,
                                           const uint32_t colBytesAlign, const uint32_t col, const float value) {
    if (colBytesAlign - col != 0) {
      uint64_t mask[2] = {(((uint64_t)1 << (colBytesAlign - col)) - 1)
                              << (CONSTANT_EIGHT - (colBytesAlign - col)),
                          0};
      Duplicate(dst[colBytesAlign - CONSTANT_EIGHT], value, mask, curRowsNum, 1, colBytesAlign / CONSTANT_EIGHT);
      pipe_barrier(PIPE_V);
    }
  }

  __aicore__ inline void InitIndex(const LocalTensor<int32_t>& dst, const int32_t curRowsNum,
                                   const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID);
    WaitFlag<HardEvent::V_S>(eventID);
    ArithProgressionPerf(dst, 0, 1, tilingData->colAlign);
    pipe_barrier(PIPE_V);
    if (curRowsNum - 1 > 0) {
      CopyRepeatParams repeatParams;
      repeatParams.dstStride = 1;
      repeatParams.srcStride = 1;
      repeatParams.dstRepeatSize = tilingData->colAlign / CONSTANT_EIGHT;
      repeatParams.srcRepeatSize = 0;
      int32_t loopTimes = (tilingData->colAlign + REPEAT_B32_SIZE - 1) / REPEAT_B32_SIZE;
      int32_t offset;
      for (int32_t i = 0; i < loopTimes - 1; i++) {
        offset = i * REPEAT_B32_SIZE;
        Copy(dst[tilingData->colAlign + offset], dst[offset], REPEAT_B32_SIZE, curRowsNum - 1, repeatParams);
      }
      offset = (loopTimes - 1) * REPEAT_B32_SIZE;
      Copy(dst[tilingData->colAlign + offset], dst[offset], tilingData->colAlign - offset, curRowsNum - 1,
           repeatParams);
    }
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void SortFP32Perf(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                      const LocalTensor<int32_t>& src1, const int32_t curRowsNum,
                                      const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    Sort32(dst, src0, src1.ReinterpretCast<uint32_t>(), tilingData->colAlign * curRowsNum / SORT_UNIT);
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void MergeSortFP32PerfCopy(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                               const int32_t curRowsNum, const int32_t srcColsNum,
                                               const int32_t dstColsNum) {
    if (srcColsNum != dstColsNum * MERGE_LIST_MAX_NUM) {
      uint32_t tmp = 0xFF800000;                              // -inf
      Duplicate(dst, *((float *)&tmp), dstColsNum * CONSTANT_TWO * curRowsNum);
    }
    pipe_barrier(PIPE_V);
    DataCopyParams intriParams;
    intriParams.blockCount = curRowsNum;
    intriParams.blockLen = BLOCK_B32_SIZE * CONSTANT_TWO / CONSTANT_EIGHT;
    intriParams.srcStride = (srcColsNum - BLOCK_B32_SIZE) * CONSTANT_TWO / CONSTANT_EIGHT;
    intriParams.dstStride = (dstColsNum - BLOCK_B32_SIZE) * CONSTANT_TWO / CONSTANT_EIGHT;
    // 32 -> 8
    for (int32_t i = 0; i < srcColsNum / SORT_UNIT; i++) {
      DataCopy(dst[BLOCK_B32_SIZE * CONSTANT_TWO * i], src[SORT_UNIT * CONSTANT_TWO * i], intriParams);
    }
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void MergeSortFP32PerfBlockMerge(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                     const int32_t repeatTimes) {
    MrgSort4Info params;
    MrgSortSrcList<float> srcList;
    params.ifExhaustedSuspension = false;
    params.elementLengths[0] = BLOCK_B32_SIZE;
    params.elementLengths[1] = BLOCK_B32_SIZE;
    params.elementLengths[CONSTANT_TWO] = BLOCK_B32_SIZE;
    params.elementLengths[CONSTANT_THREE] = BLOCK_B32_SIZE;
    params.validBit = MERGE_FOUR;
    params.repeatTimes = repeatTimes;
    srcList.src1 = src[0];
    srcList.src2 = src[BLOCK_B32_SIZE * CONSTANT_TWO];
    srcList.src3 = src[BLOCK_B32_SIZE * CONSTANT_TWO * CONSTANT_TWO];
    srcList.src4 = src[BLOCK_B32_SIZE * CONSTANT_TWO * CONSTANT_THREE];
    MrgSort(dst, srcList, params);
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void MergeSortFP32Perf2To1(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                               const LocalTensor<float>& tmpBuffer, const int32_t curRowsNum,
                                               const int32_t tailOffset) {
    // former + tail -> 1
    MrgSort4Info params;
    MrgSortSrcList<float> srcList;
    params.ifExhaustedSuspension = false;
    params.elementLengths[0] = BLOCK_B32_SIZE;
    params.elementLengths[1] = BLOCK_B32_SIZE;
    params.validBit = MERGE_TWO;
    params.repeatTimes = 1;
    for (int32_t i = 0; i < curRowsNum; i++) {
      srcList.src1 = src[i * BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM * CONSTANT_TWO];
      srcList.src2 = src[i * BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM * CONSTANT_TWO + tailOffset];
      MrgSort(tmpBuffer[i * BLOCK_B32_SIZE * CONSTANT_TWO * CONSTANT_TWO], srcList, params);
    }
    // so that col == BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM
    DataCopyParams intriParams;
    intriParams.blockCount = curRowsNum;
    intriParams.blockLen = BLOCK_B32_SIZE * CONSTANT_TWO * CONSTANT_TWO / CONSTANT_EIGHT;
    intriParams.srcStride = 0;
    intriParams.dstStride =
        (BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM - BLOCK_B32_SIZE * CONSTANT_TWO) * CONSTANT_TWO / CONSTANT_EIGHT;
    DataCopy(dst, tmpBuffer, intriParams);
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void MergeSortFP32Perf(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                           const LocalTensor<float>& tmpBuffer, const int32_t curRowsNum,
                                           const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    if (tilingData->colAlign <= MERGE_UNIT * MERGE_LIST_MAX_NUM) {
      int32_t curColsNum = tilingData->colAlign;
      if (tilingData->colAlign > MERGE_UNIT) {
        // 16 -> 4
        MergeSortFP32PerfCopy(tmpBuffer, src, curRowsNum, tilingData->colAlign, MERGE_UNIT);
        MergeSortFP32PerfBlockMerge(src, tmpBuffer, curRowsNum * MERGE_LIST_MAX_NUM);
        curColsNum = MERGE_UNIT;
      }
      // 4 -> 1
      MergeSortFP32PerfCopy(tmpBuffer, src, curRowsNum, curColsNum, BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM);
      MergeSortFP32PerfBlockMerge(dst, tmpBuffer, curRowsNum);
    } else {
      // former is 512
      DataCopyParams intriParams;
      intriParams.blockCount = curRowsNum;
      intriParams.blockLen = BLOCK_B32_SIZE * CONSTANT_TWO / CONSTANT_EIGHT;
      intriParams.srcStride = (tilingData->colAlign - BLOCK_B32_SIZE) * CONSTANT_TWO / CONSTANT_EIGHT;
      intriParams.dstStride = (MERGE_UNIT - BLOCK_B32_SIZE) * CONSTANT_TWO / CONSTANT_EIGHT;
      for (int32_t i = 0; i < MERGE_UNIT * MERGE_LIST_MAX_NUM / SORT_UNIT; i++) {
        DataCopy(tmpBuffer[BLOCK_B32_SIZE * CONSTANT_TWO * i], src[SORT_UNIT * CONSTANT_TWO * i], intriParams);
      }
      // tail is colAlign - 512
      int32_t tailColsNum = tilingData->colAlign - MERGE_UNIT * MERGE_LIST_MAX_NUM;
      int32_t dstColsNum = tailColsNum > MERGE_UNIT ? MERGE_UNIT : BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM;
      int32_t tailOffset = MERGE_UNIT * CONSTANT_TWO * curRowsNum;
      if (tailColsNum != dstColsNum * MERGE_LIST_MAX_NUM) {
        uint32_t tmp = 0xFF800000;                      // -inf
        Duplicate(tmpBuffer[tailOffset], *((float *)&tmp), dstColsNum * CONSTANT_TWO * curRowsNum);
      }
      pipe_barrier(PIPE_V);
      intriParams.dstStride = (dstColsNum - BLOCK_B32_SIZE) * CONSTANT_TWO / CONSTANT_EIGHT;
      for (int32_t i = 0; i < tailColsNum / SORT_UNIT; i++) {
        DataCopy(tmpBuffer[BLOCK_B32_SIZE * CONSTANT_TWO * i + tailOffset],
                 src[SORT_UNIT * CONSTANT_TWO * i + MERGE_UNIT * MERGE_LIST_MAX_NUM * CONSTANT_TWO], intriParams);
      }
      pipe_barrier(PIPE_V);
      // former merge
      MergeSortFP32PerfBlockMerge(dst, tmpBuffer, curRowsNum * MERGE_LIST_MAX_NUM);
      MergeSortFP32PerfCopy(tmpBuffer, dst, curRowsNum, MERGE_UNIT, BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM);
      MergeSortFP32PerfBlockMerge(dst, tmpBuffer, curRowsNum);
      // tail merge
      if (tailColsNum > MERGE_UNIT) {
        MergeSortFP32PerfBlockMerge(dst[tailOffset], tmpBuffer[tailOffset], curRowsNum * MERGE_LIST_MAX_NUM);
        MergeSortFP32PerfCopy(tmpBuffer[tailOffset], dst[tailOffset], curRowsNum, MERGE_UNIT,
                              BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM);
      }
      MergeSortFP32PerfBlockMerge(dst[tailOffset], tmpBuffer[tailOffset], curRowsNum);
      MergeSortFP32Perf2To1(dst, dst, tmpBuffer, curRowsNum, tailOffset);
    }
  }

  __aicore__ inline void ExtractKFP32Perf(const LocalTensor<float>& dstValues, const LocalTensor<int32_t>& dstIndices,
                                          const LocalTensor<int32_t>& src, const LocalTensor<uint32_t>& pattern,
                                          const int32_t curRowsNum,
                                          const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    GatherMaskParams params;
    uint64_t rsvdCnt = 0;
    params.src0BlockStride = 1;
    params.repeatTimes = curRowsNum;
    params.src0RepeatStride = BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM * CONSTANT_TWO / CONSTANT_EIGHT;
    params.src1RepeatStride = 0;
    GatherMask(dstValues.ReinterpretCast<int32_t>(), src, pattern, true, REPEAT_B32_SIZE, params, rsvdCnt);
    GatherMask(dstIndices, src, pattern[BLOCK_B32_SIZE], true, REPEAT_B32_SIZE, params, rsvdCnt);
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void UpdateIndices(const LocalTensor<int32_t>& src, const LocalTensor<int32_t>& tmpBuffer,
                                       const LocalTensor<uint32_t>& pattern, const int32_t curRowsNum,
                                       const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    LocalTensor<bool> finishedLocal = finishedQueue.DeQue<bool>();
    Cast(tmpBuffer.ReinterpretCast<half>(), finishedLocal.ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, curRowsNum);
    pipe_barrier(PIPE_V);
    Cast(tmpBuffer[tilingData->bufferElemSize], tmpBuffer.ReinterpretCast<half>(), RoundMode::CAST_FLOOR, curRowsNum);
    pipe_barrier(PIPE_V);
    Muls(tmpBuffer[tilingData->bufferElemSize], tmpBuffer[tilingData->bufferElemSize],
         static_cast<int32_t>(tilingData->col), curRowsNum);
    pipe_barrier(PIPE_V);
    Brcb(tmpBuffer, tmpBuffer[tilingData->bufferElemSize], (curRowsNum + CONSTANT_EIGHT - 1) / CONSTANT_EIGHT,
         {1, CONSTANT_EIGHT});
    pipe_barrier(PIPE_V);
    GatherMaskParams params;
    uint64_t rsvdCnt = 0;
    params.src0BlockStride = 1;
    params.repeatTimes = curRowsNum;
    params.src0RepeatStride = 1;
    params.src1RepeatStride = 0;
    GatherMask(tmpBuffer[tilingData->bufferElemSize], tmpBuffer, pattern[BLOCK_B32_SIZE + BLOCK_B32_SIZE], true,
               BLOCK_B32_SIZE, params, rsvdCnt);
    pipe_barrier(PIPE_V);
    Max(src, src, tmpBuffer[tilingData->bufferElemSize], curRowsNum * tilingData->k);
    pipe_barrier(PIPE_V);
    finishedQueue.FreeTensor(finishedLocal);
  }

  __aicore__ inline void TopKFP32Perf(const LocalTensor<float>& dstValues, const LocalTensor<int32_t>& dstIndices,
                                      const LocalTensor<float>& src, const LocalTensor<uint32_t>& pattern,
                                      const LocalTensor<float>& tmpBuffer, const int32_t curRowsNum,
                                      const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    if (tilingData->k == 1 && tilingData->colBytesAlign <= REDUCE_MAX_SIZE) {
      WholeReduceMax(tmpBuffer, src, tilingData->colBytesAlign, curRowsNum, BLOCK_BYTES, 1,
                      tilingData->colBytesAlign / CONSTANT_EIGHT, ReduceOrder::ORDER_VALUE_INDEX);
      pipe_barrier(PIPE_V);
    } else {
      Rearrange(dstValues, src, curRowsNum, tilingData);
      DuplicatePad(dstValues, curRowsNum, tilingData);
      InitIndex(dstIndices, curRowsNum, tilingData);
      SortFP32Perf(tmpBuffer, dstValues, dstIndices, curRowsNum, tilingData);
      if (tilingData->colAlign > SORT_UNIT) {
        MergeSortFP32Perf(tmpBuffer, tmpBuffer, dstValues, curRowsNum, tilingData);
      }
    }
    ExtractKFP32Perf(dstValues, dstIndices, tmpBuffer.ReinterpretCast<int32_t>(), pattern, curRowsNum, tilingData);
    if (exitFinished) {
      UpdateIndices(dstIndices, tmpBuffer.ReinterpretCast<int32_t>(), pattern, curRowsNum, tilingData);
    }
  }

  __aicore__ inline void CopyOutSoftmax(const int32_t outerIdx, const int32_t curRowsNum,
                                        LocalTensor<float>& softmaxResultOutLocal,
                                        const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    softmaxResultOutQueue.EnQue<float>(softmaxResultOutLocal);
    softmaxResultOutLocal = softmaxResultOutQueue.DeQue<float>();
    DataCopyParams intriParams;
    if (tilingData->colBytesAlign != tilingData->col) {
      intriParams.blockCount = curRowsNum;
      intriParams.blockLen = tilingData->col * sizeof(float);
    } else {
      intriParams.blockCount = 1;
      intriParams.blockLen = curRowsNum * tilingData->col * sizeof(float);
    }
#ifndef __CCE_KT_TEST__
    DataCopyPad(softmaxResultGM[tilingData->ubFormer * tilingData->col * outerIdx], softmaxResultOutLocal, intriParams);
#endif
    auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventID);
    WaitFlag<HardEvent::MTE3_V>(eventID);
  }

  __aicore__ inline void GatherSoftmaxResult(const LocalTensor<int32_t>& softmaxResult,
                                             const LocalTensor<uint32_t>& pattern, const int32_t curRowsNum,
                                             const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    GatherMaskParams params;
    uint64_t rsvdCnt = 0;
    params.src0BlockStride = 1;
    params.repeatTimes = curRowsNum;
    params.src0RepeatStride = 1;
    params.src1RepeatStride = 0;
    GatherMask(softmaxResult, softmaxResult, pattern[BLOCK_B32_SIZE + BLOCK_B32_SIZE], true, BLOCK_B32_SIZE, params, rsvdCnt);
  }

  __aicore__ inline void ComputeRenorm(LocalTensor<T>& gatingLocal, const LocalTensor<int32_t>& topKOutsLocal,
                                       const LocalTensor<float>& tmpTensorLocal,
                                       const LocalTensor<uint32_t>& patternTensorLocal, const int32_t curRowsNum,
                                       const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    if constexpr (IsSameType<T, float>::value) {
      TopKFP32Perf(topKOutsLocal.ReinterpretCast<float>(), topKOutsLocal[tilingData->bufferElemSize], gatingLocal,
                    patternTensorLocal, tmpTensorLocal, curRowsNum, tilingData);
      gatingQueue.FreeTensor(gatingLocal);
      pipe_barrier(PIPE_V);
      SoftmaxFP32Perf(topKOutsLocal.ReinterpretCast<float>(), topKOutsLocal.ReinterpretCast<float>(), tmpTensorLocal,
                      tmpTensorLocal[tilingData->bufferElemSize], curRowsNum, tilingData->kAlign, tilingData->k);
      pipe_barrier(PIPE_V);
      GatherSoftmaxResult(topKOutsLocal, patternTensorLocal, curRowsNum, tilingData);
    } else {
      if (tilingData->col < CONSTANT_EIGHT) {
        Cast(tmpTensorLocal, gatingLocal, RoundMode::CAST_NONE, tilingData->colBytesAlign, curRowsNum, {1, 1, 1, 1});
      } else {
        Cast(tmpTensorLocal, gatingLocal, RoundMode::CAST_NONE, curRowsNum * tilingData->colBytesAlign);
      }
      gatingQueue.FreeTensor(gatingLocal);
      pipe_barrier(PIPE_V);
      TopKFP32Perf(topKOutsLocal.ReinterpretCast<float>(), topKOutsLocal[tilingData->bufferElemSize], tmpTensorLocal,
                    patternTensorLocal, tmpTensorLocal[tilingData->bufferElemSize], curRowsNum, tilingData);
      pipe_barrier(PIPE_V);

      SoftmaxFP32Perf(topKOutsLocal.ReinterpretCast<float>(), topKOutsLocal.ReinterpretCast<float>(), tmpTensorLocal,
                      tmpTensorLocal[tilingData->bufferElemSize], curRowsNum, tilingData->kAlign, tilingData->k);
      pipe_barrier(PIPE_V);
      GatherSoftmaxResult(topKOutsLocal, patternTensorLocal, curRowsNum, tilingData);                      
      pipe_barrier(PIPE_V);
      Cast(topKOutsLocal.ReinterpretCast<T>(), topKOutsLocal.ReinterpretCast<float>(), RoundMode::CAST_RINT,
           tilingData->k * curRowsNum);
    }
  }

  __aicore__ inline void ComputePhase1(const int32_t OuterIdx, const int32_t curRowsNum,
                                       const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    LocalTensor<T> gatingLocal = gatingQueue.DeQue<T>();
    LocalTensor<int32_t> topKOutsLocal = topKOutsQueue.AllocTensor<int32_t>();
    LocalTensor<float> tmpTensorLocal = tmpTensor.Get<float>();
    LocalTensor<uint32_t> patternTensorLocal = patternTensor.Get<uint32_t>();
    LocalTensor<float> softmaxResultOutLocal = softmaxResultOutQueue.AllocTensor<float>();
    if constexpr (renorm == 0) {
      if constexpr (IsSameType<T, float>::value) {
        SoftmaxFP32Perf(softmaxResultOutLocal, gatingLocal, tmpTensorLocal, tmpTensorLocal[tilingData->bufferElemSize],
                        curRowsNum, tilingData->colBytesAlign, tilingData->col);
        gatingQueue.FreeTensor(gatingLocal);
        if (this->softmaxFlag == 1) {
          CopyOutSoftmax(OuterIdx, curRowsNum, softmaxResultOutLocal, tilingData);
        }
        pipe_barrier(PIPE_V);
        TopKFP32Perf(topKOutsLocal.ReinterpretCast<float>(), topKOutsLocal[tilingData->bufferElemSize],
                     softmaxResultOutLocal, patternTensorLocal, tmpTensorLocal, curRowsNum, tilingData);
      } else {
        if (tilingData->col < CONSTANT_EIGHT) {
          Cast(tmpTensorLocal, gatingLocal, RoundMode::CAST_NONE, tilingData->colBytesAlign, curRowsNum, {1, 1, 1, 1});
        } else {
          Cast(tmpTensorLocal, gatingLocal, RoundMode::CAST_NONE, curRowsNum * tilingData->colBytesAlign);
        }
        gatingQueue.FreeTensor(gatingLocal);
        pipe_barrier(PIPE_V);
        SoftmaxFP32Perf(softmaxResultOutLocal, tmpTensorLocal, topKOutsLocal.ReinterpretCast<float>(),
                        tmpTensorLocal[tilingData->bufferElemSize], curRowsNum, tilingData->colBytesAlign, tilingData->col);
        pipe_barrier(PIPE_V);
        if (this->softmaxFlag == 1) {
          CopyOutSoftmax(OuterIdx, curRowsNum, softmaxResultOutLocal, tilingData);
        }
        TopKFP32Perf(topKOutsLocal.ReinterpretCast<float>(), topKOutsLocal[tilingData->bufferElemSize],
                    softmaxResultOutLocal, patternTensorLocal, tmpTensorLocal, curRowsNum, tilingData);
        pipe_barrier(PIPE_V);
        Cast(topKOutsLocal.ReinterpretCast<T>(), topKOutsLocal.ReinterpretCast<float>(), RoundMode::CAST_RINT,
             curRowsNum * tilingData->k);
      }
    } else {
      ComputeRenorm(gatingLocal, topKOutsLocal, tmpTensorLocal, patternTensorLocal, curRowsNum, tilingData);
    }
    topKOutsQueue.EnQue(topKOutsLocal);
    softmaxResultOutQueue.FreeTensor(softmaxResultOutLocal);
  }

  __aicore__ inline void CopyOutPhase0(const int32_t OuterIdx, const int32_t curRowsNum,
                                       const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData) {
    LocalTensor<int32_t> topKOutsLocal = topKOutsQueue.DeQue<int32_t>();
    int64_t gmIndex = tilingData->ubFormer * tilingData->k * OuterIdx;
    DataCopyParams intriParams;
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    intriParams.blockCount = 1;
    intriParams.blockLen = curRowsNum * tilingData->k * sizeof(int32_t);
#ifndef __CCE_KT_TEST__
    DataCopyPad(indicesOutTensorGM[gmIndex], topKOutsLocal[tilingData->bufferElemSize], intriParams);
#endif

    auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
    intriParams.blockCount = 1;
    intriParams.blockLen = curRowsNum * tilingData->k * sizeof(T);
#ifndef __CCE_KT_TEST__
    DataCopyPad(outTensorGM[gmIndex], topKOutsLocal.ReinterpretCast<T>(), intriParams);
#endif
    topKOutsQueue.FreeTensor(topKOutsLocal);
  }

  private:
  TPipe pipe;

  TQue<QuePosition::VECIN, DB_BUFFER_NUM> gatingQueue;
  TQue<QuePosition::VECIN, DB_BUFFER_NUM> finishedQueue;
  TQue<QuePosition::VECOUT, 1> topKOutsQueue;
  TQue<QuePosition::VECOUT, 1> softmaxResultOutQueue;

  TBuf<> tmpTensor;
  TBuf<> patternTensor;

  GlobalTensor<T> gatingTensorGM;
  GlobalTensor<bool> finishedTensorGM;
  GlobalTensor<T> outTensorGM;
  GlobalTensor<int32_t> indicesOutTensorGM;
  GlobalTensor<float> softmaxResultGM;

  bool exitFinished{false};
  int64_t softmaxFlag;
};
}  // namespace MoeGatingTopKSoftmaxV2
#endif
