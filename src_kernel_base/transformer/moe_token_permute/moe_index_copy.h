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
 * \file moe_index_copy.h
 * \brief
 */
#ifndef MOE_INDEX_COPY_H
#define MOE_INDEX_COPY_H

#include "moe_common.h"

namespace MoeTokenPermute {
using namespace AscendC;

template <typename T, bool ifNumOutTokens>
class MoeindexCopyOp {
 public:
  __aicore__ inline MoeindexCopyOp(){};
  __aicore__ inline void Init(GM_ADDR src, GM_ADDR indices, GM_ADDR dst,
                              const MoeTokenPermuteTilingData* tilingData,
                              TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(int64_t Offset, DataCopyExtParams &dataCopyExtParams , DataCopyPadExtParams<T> &DataCopyPadExtParams);
  __aicore__ inline void CopyOut(int64_t innerLoop, int64_t outTokenNum,
                                 DataCopyExtParams &dataCopyExtParams,
                                 DataCopyPadExtParams<T> &DataCopyPadExtParams);
  __aicore__ inline void CopyInIndices(int64_t progress,
                                       DataCopyExtParams &dataCopyExtParams,
                                       DataCopyPadExtParams<int32_t> &DataCopyPadExtParams);
  __aicore__ inline void SyncAll();

 private:
  TPipe* pipe;
  TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> copyInQueue;
  TQue<QuePosition::VECOUT, 1> indicesBuffer;

  GlobalTensor<T> srcGm;
  GlobalTensor<T> dstGm;

  GlobalTensor<int32_t> indicesGm;
  LocalTensor<int32_t> indicesLocal;

  const IndexCopyComputeTilingData* indexCopyTilingData;

  event_t indicesMte2ToS;
  event_t indicesSToMte2;
  int64_t cols;
  int64_t colsAlign;
  int64_t topK;
  int64_t needCoreNum;
  int64_t frontCoreNum;
  int64_t tailCoreNum;
  int64_t coreCalcNum;
  int64_t oneTokenBtypeSize;
  int64_t onceIndicesTokenMoveTimes;
  int64_t onceUbTokenNums;
  int64_t onceIndicesTokenNums;
  int64_t onceIndices;
  int64_t oneTokenlastMove;
  int64_t oneTokenOnceMove;
  int64_t oneTokenMoveTimes;
  int64_t CoreLoop;
  int64_t CoreLastTokenNums;

  int64_t LastonceIndicesTokenMoveTimes;
  int64_t LastIndicesLastTokenNums;
  int64_t numOutTokens;

  int64_t onceIndicesTokenOffset;
  int64_t onceUbTokenCols;
  int64_t onceUbIndicesCols;
  int64_t coreNum;
  int64_t blockIdx;
  int64_t totalLength;

  int64_t onceIndicesNums;
  DataCopyExtParams indicesCopyParams;
  DataCopyExtParams indicesCopyLastParams;
  DataCopyExtParams tokenCopyParams;
  DataCopyExtParams tokenCopyLastParams;
  DataCopyExtParams tokenCopyOutParams;
  DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
  DataCopyPadExtParams<int32_t> indicesPadParams{false, 0, 0, 0};
};

template <typename T, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, ifNumOutTokens>::CopyIn(int64_t Offset,
                                                                 DataCopyExtParams &dataCopyExtParams ,
                                                                 DataCopyPadExtParams<T> &DataCopyPadExtParams) {
  LocalTensor<T> inLocal = copyInQueue.AllocTensor<T>();
  DataCopyPadCustom(inLocal, srcGm[Offset], dataCopyExtParams, DataCopyPadExtParams);
  copyInQueue.EnQue<T>(inLocal);
}

template <typename T, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, ifNumOutTokens>::CopyInIndices(int64_t progress,
                                                     DataCopyExtParams &dataCopyExtParams,
                                                     DataCopyPadExtParams<int32_t> &DataCopyPadExtParams) {
  indicesLocal = indicesBuffer.AllocTensor<int32_t>();
  DataCopyPadCustom(indicesLocal, indicesGm[progress * onceIndicesNums], dataCopyExtParams, DataCopyPadExtParams);
}

template <typename T, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, ifNumOutTokens>::CopyOut(int64_t innerLoop, int64_t outTokenNum,
                                                                  DataCopyExtParams &dataCopyExtParams,
                                                                  DataCopyPadExtParams<T> &DataCopyPadExtParams) {
  LocalTensor<T> inLocal = copyInQueue.DeQue<T>();
  int64_t offset = innerLoop * onceUbIndicesCols;
  for (int32_t tokensId = 0; tokensId < outTokenNum; tokensId++) {
    for(int32_t topKId = 0; topKId < topK; topKId++) {
#ifndef __CCE_KT_TEST__
      if constexpr (ifNumOutTokens == true) {
        auto indicesValue = indicesLocal.GetValue(offset);
        if (indicesValue < numOutTokens) {
          DataCopyPadCustom(dstGm[indicesValue * cols],
                      inLocal[tokensId * colsAlign],
                      dataCopyExtParams);
        }
      } else {
        DataCopyPadCustom(dstGm[indicesLocal.GetValue(offset) * cols],
                    inLocal[tokensId * colsAlign],
                    dataCopyExtParams);
      }
#endif 
      offset++;
    }
  }
  copyInQueue.FreeTensor(inLocal);
}

template <typename T, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, ifNumOutTokens>::SyncAll() {
  if (coreNum == 1) {
    return;
  }
#ifndef __CCE_KT_TEST__
  AscendC::SyncAll();
#endif
}

template <typename T, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, ifNumOutTokens>::Init(GM_ADDR src, GM_ADDR indices, GM_ADDR dst,
                                            const MoeTokenPermuteTilingData* tilingData, TPipe* tPipe
                                            ) {
  int64_t blockNum = GetBlockNum();
  this->pipe = tPipe;
  this->blockIdx = GetBlockIdx();
  this->cols = tilingData->cols;
  this->colsAlign = tilingData->colsAlign;

  this->topK = tilingData->topK;

  this->indexCopyTilingData = &(tilingData->indexCopyComputeParamsOp);
  this->coreNum = this->indexCopyTilingData->needCoreNum;

  this->totalLength = this->indexCopyTilingData->numOutTokens * tilingData->cols;
  this->indexCopyTilingData = &(tilingData->indexCopyComputeParamsOp);
  this->needCoreNum = this->indexCopyTilingData->needCoreNum;
  this->frontCoreNum = this->indexCopyTilingData->frontCoreNum;
  this->tailCoreNum = this->indexCopyTilingData->tailCoreNum;
  this->coreCalcNum = this->indexCopyTilingData->coreCalcNum;
  this->oneTokenBtypeSize = this->indexCopyTilingData->oneTokenBtypeSize;
  this->onceIndicesTokenMoveTimes = this->indexCopyTilingData->onceIndicesTokenMoveTimes;
  this->onceUbTokenNums = this->indexCopyTilingData->onceUbTokenNums;
  this->onceIndicesTokenNums = this->indexCopyTilingData->onceIndicesTokenNums;
  this->onceIndices = this->indexCopyTilingData->onceIndices;
  this->oneTokenlastMove = this->indexCopyTilingData->oneTokenlastMove;
  this->oneTokenOnceMove = this->indexCopyTilingData->oneTokenOnceMove;
  this->oneTokenMoveTimes = this->indexCopyTilingData->oneTokenMoveTimes;
  this->numOutTokens = this->indexCopyTilingData->numOutTokens;
  int64_t offset;
  if (this->blockIdx > this->frontCoreNum - 1) {
    this->coreCalcNum = this->indexCopyTilingData->coreCalcTail;
    this->LastonceIndicesTokenMoveTimes = this->indexCopyTilingData->tailLastonceIndicesTokenMoveTimes;
    this->LastIndicesLastTokenNums = this->indexCopyTilingData->tailLastIndicesLastTokenNums;
    this->CoreLoop = this->indexCopyTilingData->tailCoreLoop;
    this->CoreLastTokenNums = this->indexCopyTilingData->tailCoreLastTokenNums;
    offset = this->indexCopyTilingData->coreCalcNum * this->frontCoreNum +
             (this->blockIdx - this->frontCoreNum) * this->indexCopyTilingData->coreCalcTail;
  } else {
    this->coreCalcNum = this->indexCopyTilingData->coreCalcNum;
    this->LastonceIndicesTokenMoveTimes = this->indexCopyTilingData->frontLastonceIndicesTokenMoveTimes;
    this->LastIndicesLastTokenNums = this->indexCopyTilingData->frontLastIndicesLastTokenNums;
    this->CoreLoop = this->indexCopyTilingData->frontCoreLoop;
    this->CoreLastTokenNums = this->indexCopyTilingData->frontCoreLastTokenNums;
    offset = this->coreCalcNum * this->blockIdx;
  }
  int64_t srcoffset = offset * cols;
  int64_t indicesoffset = offset * topK;
  onceUbTokenCols = this->onceUbTokenNums * cols;
  onceUbIndicesCols = this->onceUbTokenNums * topK;

  srcGm.SetGlobalBuffer((__gm__ T*)src + srcoffset, this->coreCalcNum * cols);
  indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices + indicesoffset, this->coreCalcNum * topK);
  dstGm.SetGlobalBuffer((__gm__ T*)dst, this->numOutTokens * cols);
  pipe->InitBuffer(copyInQueue, 2, this->indexCopyTilingData->tokenUB);
  pipe->InitBuffer(indicesBuffer,1, this->indexCopyTilingData->indicesUB);
  onceIndicesNums = onceIndicesTokenNums * topK;
  indicesCopyParams = {(uint16_t)1, (uint32_t)(onceIndicesNums * sizeof(int32_t)), 0, 0, 0};
  indicesCopyLastParams = {(uint16_t)1, (uint32_t)(CoreLastTokenNums * topK * sizeof(int32_t)), 0, 0, 0};
  tokenCopyParams = {(uint16_t)onceUbTokenNums, (uint32_t)(oneTokenBtypeSize), 0, 0, 0};
  tokenCopyLastParams = {(uint16_t)LastIndicesLastTokenNums, (uint32_t)(oneTokenBtypeSize), 0, 0, 0};
  tokenCopyOutParams = {(uint16_t)1, (uint32_t)(oneTokenBtypeSize), 0, 0, 0};
  indicesMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
  indicesSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
  if constexpr (ifNumOutTokens == true) {
    if (GetBlockIdx() == 0) {
      InitOutput<T>(dstGm, this->numOutTokens * this->cols, 0);
    }
    SyncAll();
  }
}

template <typename T, bool ifNumOutTokens>
__aicore__ inline void MoeindexCopyOp<T, ifNumOutTokens>::Process() {
  int64_t outoffset = 0;
  onceIndicesTokenOffset = onceIndicesTokenNums * cols;
  if (this->blockIdx < this->indexCopyTilingData->needCoreNum) {
    for (int64_t outLoop = 0; outLoop < CoreLoop - 1; outLoop++) {
      CopyInIndices(outLoop, indicesCopyParams, indicesPadParams);
      set_flag(PIPE_MTE2, PIPE_S, indicesMte2ToS);
      wait_flag(PIPE_MTE2, PIPE_S, indicesMte2ToS);
      for (int64_t innerLoop = 0; innerLoop < onceIndicesTokenMoveTimes; innerLoop++) {
        CopyIn(outoffset, tokenCopyParams, padParams);
        CopyOut(innerLoop, onceUbTokenNums, tokenCopyOutParams, padParams);
        outoffset += onceUbTokenCols;
      }
      indicesBuffer.FreeTensor(indicesLocal);
      set_flag(PIPE_S, PIPE_MTE2, indicesSToMte2);
      wait_flag(PIPE_S, PIPE_MTE2, indicesSToMte2);
    }
    CopyInIndices(CoreLoop - 1, indicesCopyLastParams, indicesPadParams);
    set_flag(PIPE_MTE2, PIPE_S, indicesMte2ToS);
    wait_flag(PIPE_MTE2, PIPE_S, indicesMte2ToS);
    for (int64_t innerLoop = 0; innerLoop < LastonceIndicesTokenMoveTimes - 1; innerLoop++) {
      CopyIn(outoffset, tokenCopyParams, padParams);
      CopyOut(innerLoop, onceUbTokenNums, tokenCopyOutParams, padParams);
      outoffset += onceUbTokenCols;
    }
    CopyIn(outoffset, tokenCopyLastParams, padParams);
    CopyOut(LastonceIndicesTokenMoveTimes - 1, LastIndicesLastTokenNums, tokenCopyOutParams, padParams);
    indicesBuffer.FreeTensor(indicesLocal);
  }
}

}  // namespace MoeTokenPermute
#endif  // MOE_INDEX_COPY_H