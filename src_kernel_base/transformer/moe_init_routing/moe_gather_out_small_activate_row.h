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
 * \file moe_gather_out_small_activate_row.h
 * \brief
 */
#ifndef MOE_GATHER_OUT_SMALL_ACTIVATE_ROW_H
#define MOE_GATHER_OUT_SMALL_ACTIVATE_ROW_H

#include "moe_common.h"
#include "kernel_operator.h"

namespace MoeInitRouting {
using namespace AscendC;

template <typename T>
class MoeGatherOutSmallActiveRow {
 public:
  __aicore__ inline MoeGatherOutSmallActiveRow(){};
  __aicore__ inline void Init(GM_ADDR inputActivations, GM_ADDR workspace, GM_ADDR expandSrcToDstRow,
                              GM_ADDR expandedActivations, const MoeInitRoutingTilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyInIndices(int64_t progress);

 private:
  TPipe* pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> inputActivationsCopyInQueue;
  TQue<QuePosition::VECIN, BUFFER_NUM> expandDstToSrcRowCopyInQueue;

  GlobalTensor<T> inputActivationsGm;
  GlobalTensor<T> expandedActivationsGm;
  GlobalTensor<int32_t> expandDstToSrcRowGm;

  const GatherOutComputeTilingData* gatherOutTilingData;

  int64_t needCoreNum;
  int64_t blockIdx;
  int64_t cols;
  int64_t n;
  int64_t k;
  int64_t activateRows;
  int64_t currentLoopRows;
  int64_t coreRows;
  int64_t perLoopRows;
  int64_t lastLoopRows;
  int64_t maxColsOneLoop;
  int64_t colsTileLength;

  int64_t indicesOffset;
  int64_t inputOffset;
  int64_t outOffset;
};

template <typename T>
__aicore__ inline void MoeGatherOutSmallActiveRow<T>::CopyInIndices(int64_t progress) {
  this->indicesOffset = progress * this->perLoopRows;
  LocalTensor<int32_t> indicesLocal = expandDstToSrcRowCopyInQueue.AllocTensor<int32_t>();
  DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(this->currentLoopRows * sizeof(int32_t)), 0, 0, 0};
  DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
  DataCopyPad(indicesLocal, expandDstToSrcRowGm[indicesOffset], dataCopyParams, dataCopyPadParams);

  expandDstToSrcRowCopyInQueue.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutSmallActiveRow<T>::Init(GM_ADDR inputActivations, GM_ADDR workspace,
                                                           GM_ADDR expandSrcToDstRow, GM_ADDR expandedActivations,
                                                           const MoeInitRoutingTilingData* tilingData, TPipe* tPipe) {
  this->pipe = tPipe;
  this->blockIdx = GetBlockIdx();
  this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);

  this->needCoreNum = this->gatherOutTilingData->needCoreNum;
  this->activateRows = this->gatherOutTilingData->activateRows;
  this->cols = tilingData->cols;
  this->n = tilingData->n;
  this->k = tilingData->k;
  this->maxColsOneLoop = this->gatherOutTilingData->maxColsOneLoop;

  if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1) {
    this->coreRows = this->gatherOutTilingData->lastCoreRows;
    this->perLoopRows = this->gatherOutTilingData->lastCorePerLoopRows;
    this->lastLoopRows = this->gatherOutTilingData->lastCoreLastLoopRows;
  } else {
    this->coreRows = this->gatherOutTilingData->perCoreRows;
    this->perLoopRows = this->gatherOutTilingData->perCorePerLoopRows;
    this->lastLoopRows = this->gatherOutTilingData->perCoreLastLoopRows;
  }

  inputActivationsGm.SetGlobalBuffer((__gm__ T*)inputActivations, this->coreRows * this->cols);

  expandedActivationsGm.SetGlobalBuffer(
      (__gm__ T*)expandedActivations + this->blockIdx * this->gatherOutTilingData->perCoreRows * this->cols,
      tilingData->n * tilingData->k * this->cols);

  expandDstToSrcRowGm.SetGlobalBuffer(
      (__gm__ int32_t*)workspace + this->blockIdx * this->gatherOutTilingData->perCoreRows,
      Align(this->coreRows, sizeof(int32_t)));

  pipe->InitBuffer(inputActivationsCopyInQueue, BUFFER_NUM, AlignBytes(this->maxColsOneLoop, sizeof(T)));
  pipe->InitBuffer(expandDstToSrcRowCopyInQueue, BUFFER_NUM, AlignBytes(this->perLoopRows, sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void MoeGatherOutSmallActiveRow<T>::Process() {
  if (this->blockIdx < this->needCoreNum) {
    int64_t loops = (coreRows + perLoopRows - 1) / perLoopRows;
    int64_t colsLoops = Ceil(cols, maxColsOneLoop);
    int64_t tailCols = cols - (colsLoops - 1) * maxColsOneLoop;
    currentLoopRows = perLoopRows;
    for (int64_t loop = 0; loop < loops; loop++) {
      CopyInIndices(loop);
      LocalTensor<int32_t> indicesLocal = expandDstToSrcRowCopyInQueue.DeQue<int32_t>();
      if (loop == loops - 1) {
        currentLoopRows = lastLoopRows;
      }
      event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
      SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
      WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
      for (int64_t row = 0; row < currentLoopRows; row++) {
        colsTileLength = maxColsOneLoop;
        for (int64_t colsLoop = 0; colsLoop < colsLoops; colsLoop++) {
          LocalTensor<T> inLocal = inputActivationsCopyInQueue.AllocTensor<T>();
          if (colsLoop == colsLoops - 1) {
            colsTileLength = tailCols;
          }
          inputOffset = indicesLocal.GetValue(row) % this->n * cols + colsLoop * maxColsOneLoop;
          event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
          SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
          WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
          DataCopyParams intriParams;
          intriParams.blockCount = 1;
          intriParams.blockLen = this->colsTileLength * sizeof(T);
          DataCopyPadParams dataCopyPadParams;
          DataCopyPad(inLocal, inputActivationsGm[inputOffset], intriParams, dataCopyPadParams);
          pipe_barrier(PIPE_ALL);
          outOffset = (loop * perLoopRows + row) * cols + colsLoop * maxColsOneLoop;
          DataCopyPad(expandedActivationsGm[outOffset], inLocal, intriParams);
          inputActivationsCopyInQueue.FreeTensor(inLocal);
        }
      }
      expandDstToSrcRowCopyInQueue.FreeTensor(indicesLocal);
    }
  }
}
}  // namespace MoeInitRouting
#endif  // MOE_GATHER_OUT_SMALL_ACTIVATE_ROW_H
