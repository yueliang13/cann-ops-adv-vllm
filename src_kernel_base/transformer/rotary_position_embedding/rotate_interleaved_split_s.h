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
 * \file rotate_interleaved_split_s.h
 * \brief
 */
#ifndef ROTATE_INTERLEAVED_SPLIT_S_H
#define ROTATE_INTERLEAVED_SPLIT_S_H
#include "rotate_interleaved_common.h"

namespace RotateInterleavedN {
using namespace AscendC;

template <typename T>
class InterleavedSplitS {
 public:
  __aicore__ inline InterleavedSplitS(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                              const RotaryPositionEmbeddingTilingData& tiling, TPipe *pipe);
  __aicore__ inline void Process();

 protected:
  GlobalTensor<T> xGm, cosGm, sinGm, yGm;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueX, inQueCos;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueY;
  TBuf<TPosition::VECCALC> tmpFp32Buf1, tmpFp32Buf2, tmpFp32Buf3;
  TBuf<TPosition::VECCALC> gatherOffsetBuf;

  // tilingdata
  uint64_t batchSize;
  uint64_t seqLen;
  uint64_t numHeads;
  uint64_t headDim;
  uint64_t frontCoreNum;
  uint64_t tailCoreNum;
  uint64_t coreCalcNum;
  uint64_t coreCalcTail;
  uint64_t ubCalcNum;
  uint64_t ubCalcLoop;
  uint64_t ubCalcTail;
  uint64_t ubCalcTailNum;
  uint64_t ubCalcTailLoop;
  uint64_t ubCalcTailTail;

  // init tmp data
  uint32_t blockIdx;
  uint32_t ubCalcSeq;
  uint32_t ubCalcSeqTail;
  uint32_t ubCalcSeqLoop;
  uint64_t ioOffset;
  uint64_t triOffset;
  uint64_t bufferBsndSize;
  uint64_t bufferSdSize;
  uint64_t bufferNdSize;
  uint64_t bufferLenSize;
  uint64_t gatherOffsetLenSize;

  __aicore__ inline void InitData(const RotaryPositionEmbeddingTilingData& tiling);
  __aicore__ inline void CopyInX(LocalTensor<T>& x, uint32_t loopIdx, uint32_t calcLen);
  __aicore__ inline void CopyInCos(LocalTensor<T>& cos, uint32_t loopIdx, uint32_t calcLen);
  __aicore__ inline void CopyInSin(LocalTensor<T>& sin, uint32_t loopIdx, uint32_t calcLen);
  __aicore__ inline void CopyOut(uint32_t loopIdx, uint32_t calcLen);
  __aicore__ inline void Compute(uint32_t loopIdx, LocalTensor<uint32_t>& gatherOffsetCast, uint32_t calcLen);
  __aicore__ inline void ComputeCastFp32(uint32_t loopIdx, LocalTensor<uint32_t>& gatherOffsetCast, uint32_t calcLen);
};

template <typename T>
__aicore__ inline void InterleavedSplitS<T>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                  const RotaryPositionEmbeddingTilingData& tiling, TPipe *pipe) {
  InitData(tiling);

  blockIdx = GetBlockIdx();
  bufferSdSize = seqLen * headDim;
  bufferNdSize = numHeads * headDim;

  if (blockIdx < frontCoreNum) {
    ubCalcSeq = ubCalcNum;
    ubCalcSeqTail = ubCalcTail;
    ubCalcSeqLoop = ubCalcLoop;
    ioOffset = blockIdx * coreCalcNum * bufferNdSize;
    triOffset = blockIdx * coreCalcNum * headDim;
  } else if (coreCalcTail != 0) {
    ubCalcSeq = ubCalcTailNum;
    ubCalcSeqTail = ubCalcTailTail;
    ubCalcSeqLoop = ubCalcTailLoop;
    ioOffset = frontCoreNum * coreCalcNum * bufferNdSize + (blockIdx - frontCoreNum) * coreCalcTail * bufferNdSize;
    triOffset = frontCoreNum * coreCalcNum * headDim + (blockIdx - frontCoreNum) * coreCalcTail * headDim;
  }

  bufferBsndSize = batchSize * seqLen * bufferNdSize;
  xGm.SetGlobalBuffer((__gm__ T*)x + ioOffset, bufferBsndSize);
  yGm.SetGlobalBuffer((__gm__ T*)y + ioOffset, bufferBsndSize);
  cosGm.SetGlobalBuffer((__gm__ T*)cos + triOffset, bufferSdSize);
  sinGm.SetGlobalBuffer((__gm__ T*)sin + triOffset, bufferSdSize);

  bufferLenSize = batchSize * ubCalcSeq * bufferNdSize * sizeof(T);
  pipe->InitBuffer(inQueX,   BUFFER_NUM, bufferLenSize);
  pipe->InitBuffer(inQueCos, BUFFER_NUM, bufferLenSize);
  pipe->InitBuffer(outQueY,  BUFFER_NUM, bufferLenSize);

  if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
    bufferLenSize = batchSize * ubCalcSeq * bufferNdSize * sizeof(float);
    pipe->InitBuffer(tmpFp32Buf1, bufferLenSize);
    pipe->InitBuffer(tmpFp32Buf2, bufferLenSize);
    pipe->InitBuffer(tmpFp32Buf3, bufferLenSize);
  }

  gatherOffsetLenSize = bufferNdSize * sizeof(int32_t);
  pipe->InitBuffer(gatherOffsetBuf, gatherOffsetLenSize);
}

template <typename T>
__aicore__ inline void InterleavedSplitS<T>::InitData(const RotaryPositionEmbeddingTilingData& tiling) {
  const RopeInterleavedParams& rotateInterleavedTiling = tiling.ropeInterleavedParams;
  batchSize = rotateInterleavedTiling.batchSize;
  seqLen = rotateInterleavedTiling.seqLen;
  numHeads = rotateInterleavedTiling.numHeads;
  headDim = rotateInterleavedTiling.headDim;
  frontCoreNum = rotateInterleavedTiling.frontCoreNum;
  tailCoreNum = rotateInterleavedTiling.tailCoreNum;
  coreCalcNum = rotateInterleavedTiling.coreCalcNum;
  coreCalcTail = rotateInterleavedTiling.coreCalcTail;
  ubCalcNum = rotateInterleavedTiling.ubCalcNum;
  ubCalcLoop = rotateInterleavedTiling.ubCalcLoop;
  ubCalcTail = rotateInterleavedTiling.ubCalcTail;
  ubCalcTailNum = rotateInterleavedTiling.ubCalcTailNum;
  ubCalcTailLoop = rotateInterleavedTiling.ubCalcTailLoop;
  ubCalcTailTail = rotateInterleavedTiling.ubCalcTailTail;
}

template <typename T>
__aicore__ inline void InterleavedSplitS<T>::CopyInX(LocalTensor<T>& x, uint32_t loopIdx, uint32_t calcLen) {
  DataCopyExtParams dataCopyParams;
  dataCopyParams.blockCount = batchSize;
  dataCopyParams.blockLen   = calcLen * bufferNdSize * sizeof(T);
  dataCopyParams.srcStride  = (seqLen - calcLen) * bufferNdSize * sizeof(T);
  dataCopyParams.dstStride  = 0;
#ifndef __CCE_KT_TEST__
  DataCopyPad(x, xGm[loopIdx * ubCalcSeq * bufferNdSize], dataCopyParams, {false, 0, 0, 0});
#endif
  event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
  SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
  WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
}

template <typename T>
__aicore__ inline void InterleavedSplitS<T>::CopyInCos(LocalTensor<T>& cos, uint32_t loopIdx, uint32_t calcLen) {
  DataCopyExtParams dataCopyTriParams;
  dataCopyTriParams.blockCount = calcLen;
  dataCopyTriParams.blockLen   = headDim * sizeof(T);
  dataCopyTriParams.srcStride  = 0;
  dataCopyTriParams.dstStride  = static_cast<uint16_t>((numHeads - 1) * headDim * sizeof(T) / BLOCK_SIZE);
#ifndef __CCE_KT_TEST__
  DataCopyPad(cos, cosGm[loopIdx * ubCalcSeq * headDim], dataCopyTriParams, {false, 0, 0, 0});
#endif
  event_t eventId2MTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
  SetFlag<HardEvent::MTE2_V>(eventId2MTE2ToV);
  WaitFlag<HardEvent::MTE2_V>(eventId2MTE2ToV);
  BroadCastTriToBsnd(cos, batchSize, calcLen, numHeads, headDim);
}

template <typename T>
__aicore__ inline void InterleavedSplitS<T>::CopyInSin(LocalTensor<T>& sin, uint32_t loopIdx, uint32_t calcLen) {
  event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
  SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
  WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
  DataCopyExtParams dataCopyTriParams;
  dataCopyTriParams.blockCount = calcLen;
  dataCopyTriParams.blockLen   = headDim * sizeof(T);
  dataCopyTriParams.srcStride  = 0;
  dataCopyTriParams.dstStride  = static_cast<uint16_t>((numHeads - 1) * headDim * sizeof(T) / BLOCK_SIZE);
#ifndef __CCE_KT_TEST__
  DataCopyPad(sin, sinGm[loopIdx * ubCalcSeq * headDim], dataCopyTriParams, {false, 0, 0, 0});
#endif
  event_t eventId3MTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
  SetFlag<HardEvent::MTE2_V>(eventId3MTE2ToV);
  WaitFlag<HardEvent::MTE2_V>(eventId3MTE2ToV);
  BroadCastTriToBsnd(sin, batchSize, calcLen, numHeads, headDim);
}

template <typename T>
__aicore__ inline void InterleavedSplitS<T>::CopyOut(uint32_t loopIdx, uint32_t calcLen) {
  DataCopyExtParams dataCopyParams;
  dataCopyParams.blockCount = batchSize;
  dataCopyParams.blockLen   = calcLen * bufferNdSize * sizeof(T);
  dataCopyParams.srcStride  = 0;
  dataCopyParams.dstStride  = (seqLen - calcLen) * bufferNdSize * sizeof(T);
  LocalTensor<T> y = outQueY.DeQue<T>();
#ifndef __CCE_KT_TEST__
  DataCopyPad(yGm[loopIdx * ubCalcSeq * bufferNdSize], y, dataCopyParams);
#endif
  outQueY.FreeTensor(y);
}

template <typename T>
__aicore__ inline void InterleavedSplitS<T>::Process() {
  LocalTensor<int32_t> gatherOffset = gatherOffsetBuf.Get<int32_t>();
  SetGatherSrcOffset(gatherOffset, headDim * numHeads, static_cast<int32_t>(sizeof(float)));
  LocalTensor<uint32_t> gatherOffsetCast = gatherOffset.ReinterpretCast<uint32_t>();
  if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
    for (uint32_t i = 0; i < (ubCalcSeqTail == 0 ? ubCalcSeqLoop : ubCalcSeqLoop - 1); ++i) {
      ComputeCastFp32(i, gatherOffsetCast, ubCalcSeq);
      CopyOut(i, ubCalcSeq);
    }
    if (ubCalcSeqTail != 0) {
      ComputeCastFp32(ubCalcSeqLoop - 1, gatherOffsetCast, ubCalcSeqTail);
      CopyOut(ubCalcSeqLoop - 1, ubCalcSeqTail);
    }
  } else {
    for (uint32_t i = 0; i < (ubCalcSeqTail == 0 ? ubCalcSeqLoop : ubCalcSeqLoop - 1); ++i) {
      Compute(i, gatherOffsetCast, ubCalcSeq);
      CopyOut(i, ubCalcSeq);
    }
    if (ubCalcSeqTail != 0) {
      Compute(ubCalcSeqLoop - 1, gatherOffsetCast, ubCalcSeqTail);
      CopyOut(ubCalcSeqLoop - 1, ubCalcSeqTail);
    }
  }
}

template <typename T>
__aicore__ inline void InterleavedSplitS<T>::Compute(uint32_t loopIdx, LocalTensor<uint32_t>& gatherOffsetCast,
                                                     uint32_t calcLen) {
  uint64_t calcTotalNum = calcLen * batchSize * bufferNdSize;
  
  LocalTensor<T> x = inQueX.AllocTensor<T>();
  CopyInX(x, loopIdx, calcLen);

  LocalTensor<T> cos = inQueCos.AllocTensor<T>();
  CopyInCos(cos, loopIdx, calcLen);

  LocalTensor<T> y = outQueY.AllocTensor<T>();
  Mul(y, x, cos, calcTotalNum);
  for (uint32_t i = 0; i < batchSize * calcLen; ++i) {
    Gather(x[i * bufferNdSize], x[i * bufferNdSize], gatherOffsetCast, 0, bufferNdSize);
  }

  CopyInSin(cos, loopIdx, calcLen);

  Mul(x, x, cos, calcTotalNum);
  inQueCos.FreeTensor(cos);
  InterleavedInversion(x, calcTotalNum);
  Add(y, y, x, calcTotalNum);
  inQueX.FreeTensor(x);
  outQueY.EnQue(y);
}

template <typename T>
__aicore__ inline void InterleavedSplitS<T>::ComputeCastFp32(uint32_t loopIdx, LocalTensor<uint32_t>& gatherOffsetCast,
                                                             uint32_t calcLen) {
  uint64_t calcTotalNum = calcLen * batchSize * bufferNdSize;

  LocalTensor<T> x = inQueX.AllocTensor<T>();
  CopyInX(x, loopIdx, calcLen);
  LocalTensor<float> tmp32Buf1 = tmpFp32Buf1.Get<float>();
  Cast(tmp32Buf1, x, RoundMode::CAST_NONE, calcTotalNum);
  inQueX.FreeTensor(x);

  LocalTensor<T> cos = inQueCos.AllocTensor<T>();
  CopyInCos(cos, loopIdx, calcLen);
  LocalTensor<float> tmp32Buf2 = tmpFp32Buf2.Get<float>();
  Cast(tmp32Buf2, cos, RoundMode::CAST_NONE, calcTotalNum);

  LocalTensor<float> tmp32Buf3 = tmpFp32Buf3.Get<float>();
  Mul(tmp32Buf3, tmp32Buf1, tmp32Buf2, calcTotalNum);

  for (uint32_t i = 0; i < batchSize * calcLen; ++i) {
    Gather(tmp32Buf1[i * bufferNdSize], tmp32Buf1[i * bufferNdSize], gatherOffsetCast, 0, bufferNdSize);
  }

  CopyInSin(cos, loopIdx, calcLen);
  Cast(tmp32Buf2, cos, RoundMode::CAST_NONE, calcTotalNum);
  inQueCos.FreeTensor(cos);

  Mul(tmp32Buf1, tmp32Buf1, tmp32Buf2, calcTotalNum);
  InterleavedInversion(tmp32Buf1, calcTotalNum);
  Add(tmp32Buf3, tmp32Buf3, tmp32Buf1, calcTotalNum);

  LocalTensor<T> y = outQueY.AllocTensor<T>();
  Cast(y, tmp32Buf3, RoundMode::CAST_RINT, calcTotalNum);
  outQueY.EnQue(y);
}

} // namespace end

#endif   // ROTATE_INTERLEAVED_SPLIT_S_H
