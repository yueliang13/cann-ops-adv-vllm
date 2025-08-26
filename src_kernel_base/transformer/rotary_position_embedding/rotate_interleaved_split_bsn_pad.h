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
 * \file rotate_interleaved_split_bsn_pad.h
 * \brief
 */
#ifndef ROTATE_INTERLEAVED_SPLIT_BSN_PAD_H
#define ROTATE_INTERLEAVED_SPLIT_BSN_PAD_H
#include "rotate_interleaved_common.h"

namespace RotateInterleavedN {
using namespace AscendC;

template <typename T>
class InterleavedSplitBSNPad {
 public:
  __aicore__ inline InterleavedSplitBSNPad(){};
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
  uint64_t ubCalcNNum;
  uint64_t ubCalcNLoop;
  uint64_t ubCalcNTail;

  // init tmp data
  uint32_t alignLen;
  uint32_t headDimAlign;
  uint32_t blockIdx;
  uint32_t ubCalcSeqLoop;
  uint64_t ioOffset;
  uint64_t triOffset;
  uint64_t bufferBsndSize;
  uint64_t bufferSdSize;
  uint64_t bufferNdSize;
  uint64_t bufferLenSize;
  uint64_t gatherOffsetLenSize;

  __aicore__ inline void InitData(const RotaryPositionEmbeddingTilingData& tiling);
  __aicore__ inline void CopyInX(LocalTensor<T>& x, uint32_t batchIdx, uint32_t seqIdx, uint32_t numHeadsIdx,
                                 uint32_t calcLen);
  __aicore__ inline void CopyInCos(LocalTensor<T>& cos, uint32_t seqIdx, uint32_t calcLen);
  __aicore__ inline void CopyInSin(LocalTensor<T>& sin, uint32_t seqIdx, uint32_t calcLen);
  __aicore__ inline void CopyOut(uint32_t batchIdx, uint32_t seqIdx, uint32_t numHeadsIdx, uint32_t calcLen);
  __aicore__ inline void Compute(uint32_t batchIdx, uint32_t seqIdx, uint32_t numHeadsIdx,
                                 LocalTensor<uint32_t>& gatherOffsetCast, uint32_t calcLen);
  __aicore__ inline void ComputeCastFp32(uint32_t batchIdx, uint32_t seqIdx, uint32_t numHeadsIdx,
                                         LocalTensor<uint32_t>& gatherOffsetCast, uint32_t calcLen);
};

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                       const RotaryPositionEmbeddingTilingData& tiling, TPipe *pipe) {
  InitData(tiling);

  blockIdx = GetBlockIdx();
  bufferSdSize = seqLen * headDim;
  bufferNdSize = numHeads * headDim;

  if (blockIdx < frontCoreNum) {
    ubCalcSeqLoop = coreCalcNum;
    ioOffset = blockIdx * coreCalcNum * bufferNdSize;
    triOffset = blockIdx * coreCalcNum * headDim;
  } else if (coreCalcTail != 0) {
    ubCalcSeqLoop = coreCalcTail;
    ioOffset = frontCoreNum * coreCalcNum * bufferNdSize + (blockIdx - frontCoreNum) * coreCalcTail * bufferNdSize;
    triOffset = frontCoreNum * coreCalcNum * headDim + (blockIdx - frontCoreNum) * coreCalcTail * headDim;
  }

  bufferBsndSize = batchSize * seqLen * bufferNdSize;
  xGm.SetGlobalBuffer((__gm__ T*)x + ioOffset, bufferBsndSize);
  yGm.SetGlobalBuffer((__gm__ T*)y + ioOffset, bufferBsndSize);
  cosGm.SetGlobalBuffer((__gm__ T*)cos + triOffset, bufferSdSize);
  sinGm.SetGlobalBuffer((__gm__ T*)sin + triOffset, bufferSdSize);

  bufferLenSize = ubCalcNNum * headDimAlign * sizeof(T);
  pipe->InitBuffer(inQueX,   BUFFER_NUM, bufferLenSize);
  pipe->InitBuffer(inQueCos, BUFFER_NUM, bufferLenSize);
  pipe->InitBuffer(outQueY,  BUFFER_NUM, bufferLenSize);

  if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
    bufferLenSize = ubCalcNNum * headDimAlign * sizeof(float);
    pipe->InitBuffer(tmpFp32Buf1, bufferLenSize);
    pipe->InitBuffer(tmpFp32Buf2, bufferLenSize);
    pipe->InitBuffer(tmpFp32Buf3, bufferLenSize);
  }

  gatherOffsetLenSize = ubCalcNNum * headDimAlign * sizeof(int32_t);
  pipe->InitBuffer(gatherOffsetBuf, gatherOffsetLenSize);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::InitData(const RotaryPositionEmbeddingTilingData& tiling) {
  const RopeInterleavedParams& rotateInterleavedTiling = tiling.ropeInterleavedParams;
  batchSize = rotateInterleavedTiling.batchSize;
  seqLen = rotateInterleavedTiling.seqLen;
  numHeads = rotateInterleavedTiling.numHeads;
  headDim = rotateInterleavedTiling.headDim;
  frontCoreNum = rotateInterleavedTiling.frontCoreNum;
  tailCoreNum = rotateInterleavedTiling.tailCoreNum;
  coreCalcNum = rotateInterleavedTiling.coreCalcNum;
  coreCalcTail = rotateInterleavedTiling.coreCalcTail;
  ubCalcNNum = rotateInterleavedTiling.ubCalcNNum;
  ubCalcNLoop = rotateInterleavedTiling.ubCalcNLoop;
  ubCalcNTail = rotateInterleavedTiling.ubCalcNTail;

  alignLen = (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) ? ALIGN_16 : ALIGN_32;
  headDimAlign = (headDim + alignLen - 1) / alignLen * alignLen;
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::CopyInX(LocalTensor<T>& x, uint32_t batchIdx, uint32_t seqIdx,
                                                          uint32_t numHeadsIdx, uint32_t calcLen) {
  DataCopyExtParams dataCopyParams;
  dataCopyParams.blockCount = calcLen;
  dataCopyParams.blockLen   = headDim * sizeof(T);
  dataCopyParams.srcStride  = 0;
  dataCopyParams.dstStride  = 0;
#ifndef __CCE_KT_TEST__
  DataCopyPad(x, xGm[batchIdx * seqLen * bufferNdSize + seqIdx * bufferNdSize + numHeadsIdx * ubCalcNNum * headDim],
              dataCopyParams, {false, 0, 0, 0});
#endif              
  event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
  SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
  WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::CopyInCos(LocalTensor<T>& cos, uint32_t seqIdx, uint32_t calcLen) {
  DataCopyExtParams dataCopyTriParams;
  dataCopyTriParams.blockCount = 1;
  dataCopyTriParams.blockLen   = headDim * sizeof(T);
  dataCopyTriParams.srcStride  = 0;
  dataCopyTriParams.dstStride  = 0;
#ifndef __CCE_KT_TEST__
  DataCopyPad(cos, cosGm[seqIdx * headDim], dataCopyTriParams, {false, 0, 0, 0});
#endif
  event_t eventId2MTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
  SetFlag<HardEvent::MTE2_V>(eventId2MTE2ToV);
  WaitFlag<HardEvent::MTE2_V>(eventId2MTE2ToV);
  BroadCastTriToB1nd(cos, 1, calcLen, headDimAlign);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::CopyInSin(LocalTensor<T>& sin, uint32_t seqIdx, uint32_t calcLen) {
  event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
  SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
  WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);

  DataCopyExtParams dataCopyTriParams;
  dataCopyTriParams.blockCount = 1;
  dataCopyTriParams.blockLen   = headDim * sizeof(T);
  dataCopyTriParams.srcStride  = 0;
  dataCopyTriParams.dstStride  = 0;
#ifndef __CCE_KT_TEST__
  DataCopyPad(sin, sinGm[seqIdx * headDim], dataCopyTriParams, {false, 0, 0, 0});
#endif
  event_t eventId3MTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
  SetFlag<HardEvent::MTE2_V>(eventId3MTE2ToV);
  WaitFlag<HardEvent::MTE2_V>(eventId3MTE2ToV);
  BroadCastTriToB1nd(sin, 1, calcLen, headDimAlign);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::CopyOut(uint32_t batchIdx, uint32_t seqIdx, uint32_t numHeadsIdx,
                                                          uint32_t calcLen) {
  DataCopyExtParams dataCopyParams;
  dataCopyParams.blockCount = calcLen;
  dataCopyParams.blockLen   = headDim * sizeof(T);
  dataCopyParams.srcStride  = 0;
  dataCopyParams.dstStride  = 0;
  LocalTensor<T> y = outQueY.DeQue<T>();
#ifndef __CCE_KT_TEST__
  DataCopyPad(yGm[batchIdx * seqLen * bufferNdSize + seqIdx * bufferNdSize + numHeadsIdx * ubCalcNNum * headDim], y,
              dataCopyParams);
#endif              
  outQueY.FreeTensor(y);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::Process() {
  LocalTensor<int32_t> gatherOffset = gatherOffsetBuf.Get<int32_t>();
  SetGatherSrcOffset(gatherOffset, ubCalcNNum * headDimAlign, static_cast<int32_t>(sizeof(float)));
  LocalTensor<uint32_t> gatherOffsetCast = gatherOffset.ReinterpretCast<uint32_t>();

  if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
    for (uint32_t i = 0; i < batchSize; ++i) {
      for (uint32_t j = 0; j < ubCalcSeqLoop; ++j) {
        for (uint32_t z = 0; z < (ubCalcNTail == 0 ? ubCalcNLoop : ubCalcNLoop - 1); ++z) {
          ComputeCastFp32(i, j, z, gatherOffsetCast, ubCalcNNum);
          CopyOut(i, j, z, ubCalcNNum);
        }
        if (ubCalcNTail != 0) {
          ComputeCastFp32(i, j, ubCalcNLoop - 1, gatherOffsetCast, ubCalcNTail);
          CopyOut(i, j, ubCalcNLoop - 1, ubCalcNTail);
        }
      }
    }
  } else {
    for (uint32_t i = 0; i < batchSize; ++i) {
      for (uint32_t j = 0; j < ubCalcSeqLoop; ++j) {
        for (uint32_t z = 0; z < (ubCalcNTail == 0 ? ubCalcNLoop : ubCalcNLoop - 1); ++z) {
          Compute(i, j, z, gatherOffsetCast, ubCalcNNum);
          CopyOut(i, j, z, ubCalcNNum);
        }
        if (ubCalcNTail != 0) {
          Compute(i, j, ubCalcNLoop - 1, gatherOffsetCast, ubCalcNTail);
          CopyOut(i, j, ubCalcNLoop - 1, ubCalcNTail);
        }
      }
    }
  }
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::Compute(uint32_t batchIdx, uint32_t seqIdx,
                                                          uint32_t numHeadsIdx, LocalTensor<uint32_t>& gatherOffsetCast,
                                                          uint32_t calcLen) {
  uint64_t calcTotalNum = calcLen * headDimAlign;
  
  LocalTensor<T> x = inQueX.AllocTensor<T>();
  CopyInX(x, batchIdx, seqIdx, numHeadsIdx, calcLen);

  LocalTensor<T> cos = inQueCos.AllocTensor<T>();
  CopyInCos(cos, seqIdx, calcLen);

  LocalTensor<T> y = outQueY.AllocTensor<T>();
  Mul(y, x, cos, calcTotalNum);

  Gather(x, x, gatherOffsetCast, 0, calcTotalNum);

  CopyInSin(cos, seqIdx, calcLen);

  Mul(x, x, cos, calcTotalNum);
  inQueCos.FreeTensor(cos);
  InterleavedInversion(x, calcTotalNum);
  Add(y, y, x, calcTotalNum);
  inQueX.FreeTensor(x);
  outQueY.EnQue(y);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::ComputeCastFp32(uint32_t batchIdx, uint32_t seqIdx,
                                                                  uint32_t numHeadsIdx,
                                                                  LocalTensor<uint32_t>& gatherOffsetCast,
                                                                  uint32_t calcLen) {
  uint64_t calcTotalNum = calcLen * headDimAlign;

  LocalTensor<T> x = inQueX.AllocTensor<T>();
  CopyInX(x, batchIdx, seqIdx, numHeadsIdx, calcLen);
  LocalTensor<float> tmp32Buf1 = tmpFp32Buf1.Get<float>();
  Cast(tmp32Buf1, x, RoundMode::CAST_NONE, calcTotalNum);
  inQueX.FreeTensor(x);

  LocalTensor<T> cos = inQueCos.AllocTensor<T>();
  CopyInCos(cos, seqIdx, calcLen);
  LocalTensor<float> tmp32Buf2 = tmpFp32Buf2.Get<float>();
  Cast(tmp32Buf2, cos, RoundMode::CAST_NONE, calcTotalNum);

  LocalTensor<float> tmp32Buf3 = tmpFp32Buf3.Get<float>();
  Mul(tmp32Buf3, tmp32Buf1, tmp32Buf2, calcTotalNum);

  Gather(tmp32Buf1, tmp32Buf1, gatherOffsetCast, 0, calcTotalNum);

  CopyInSin(cos, seqIdx, calcLen);
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

#endif   // ROTATE_INTERLEAVED_SPLIT_BSN_PAD_H
