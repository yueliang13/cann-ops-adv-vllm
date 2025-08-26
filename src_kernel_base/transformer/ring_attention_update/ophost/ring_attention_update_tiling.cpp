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
 * \file ring_attention_update_tiling.cpp
 * \brief
 */

#include <iostream>
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "log/ops_log.h"
#include "error/ops_error.h"

#include "tiling/tiling_api.h"

#include "ring_attention_update_tiling.h"

namespace optiling {

constexpr uint32_t DTYPE_KEY_FP16 = 0;
constexpr uint32_t DTYPE_KEY_BF16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t TND_KEY = 10;

constexpr uint32_t ATTN_SHAPE_SIZE = 3;
constexpr uint32_t SOFTMAX_SHAPE_SIZE = 4;

constexpr uint32_t REPEAT_NUM_B32 = 64;
constexpr uint32_t SIZE_B32 = 4;
constexpr uint32_t SIZE_B16 = 2;

constexpr size_t CONST_TWO = 2;
constexpr size_t CONST_THREE = 3;
constexpr size_t CONST_FOUR = 4;

constexpr size_t SOFTMAX_TAIL = 8;

constexpr uint64_t HEAD_DIM_ALIGN_TND = 64;
constexpr uint64_t TND_BUFFER_NUM = 2;
constexpr uint64_t MAX_UB_SIZE = 192 * 1024;
constexpr uint64_t FLOAT_DATA_SIZE = 4;
constexpr uint64_t BUFFER_NUM_IN_QUE = 2;
constexpr uint64_t SEQ_NUM_LOOP_EACH_TND = 1;

static void InitTilingData(RingAttentionUpdateTilingData& tiling) {
  // init param
  tiling.set_batchSize(0);
  tiling.set_headNum(0);
  tiling.set_seqNum(0);
  tiling.set_headDim(0);
  tiling.set_softmaxTailSize(0);

  tiling.set_coreNum(0);
  tiling.set_coreNumGroup(0);
  tiling.set_bnNumGroup(0);
  tiling.set_seqNumCoreEach(0);
  tiling.set_seqNumCoreTail(0);
  tiling.set_seqNumLoopEach(0);
  tiling.set_headNumLoopEach(0);
  tiling.set_headDimLoopEach(0);

  tiling.set_batchSizeCoreEach(0);
  tiling.set_batchSizeCoreTail(0);
}

static void RingAttentionUpdatePrintParam(const gert::TilingContext* context, RingAttentionUpdateTilingData& tiling) {
  // output param
  OPS_LOG_D(context->GetNodeName(), "batchSize = %ld", tiling.get_batchSize());
  OPS_LOG_D(context->GetNodeName(), "headNum = %ld", tiling.get_headNum());
  OPS_LOG_D(context->GetNodeName(), "seqNum = %ld", tiling.get_seqNum());
  OPS_LOG_D(context->GetNodeName(), "headDim = %ld", tiling.get_headDim());
  OPS_LOG_D(context->GetNodeName(), "softmaxTailSize = %ld", tiling.get_softmaxTailSize());

  OPS_LOG_D(context->GetNodeName(), "coreNum = %ld", tiling.get_coreNum());
  OPS_LOG_D(context->GetNodeName(), "coreNumGroup = %ld", tiling.get_coreNumGroup());
  OPS_LOG_D(context->GetNodeName(), "bnNumGroup = %ld", tiling.get_bnNumGroup());
  OPS_LOG_D(context->GetNodeName(), "seqNumCoreEach = %ld", tiling.get_seqNumCoreEach());
  OPS_LOG_D(context->GetNodeName(), "seqNumCoreTail = %ld", tiling.get_seqNumCoreTail());
  OPS_LOG_D(context->GetNodeName(), "seqNumLoopEach = %ld", tiling.get_seqNumLoopEach());
  OPS_LOG_D(context->GetNodeName(), "headNumLoopEach = %ld", tiling.get_headNumLoopEach());
  OPS_LOG_D(context->GetNodeName(), "headDimLoopEach = %ld", tiling.get_headDimLoopEach());

  OPS_LOG_D(context->GetNodeName(), "batchSizeCoreEach = %ld", tiling.get_batchSizeCoreEach());
  OPS_LOG_D(context->GetNodeName(), "batchSizeCoreTail = %ld", tiling.get_batchSizeCoreTail());
}

static ge::graphStatus CheckAttnAndSoftmaxShapeSBH(const gert::TilingContext* context, const gert::Shape prevAttnOutShape, const gert::Shape prevSoftmaxMaxShape) {
  if (prevAttnOutShape.GetDimNum() != CONST_THREE) {
    OPS_LOG_E(context->GetNodeName(), "prev_attn_out shape not support.");
    return ge::GRAPH_FAILED;
  }
  if (prevSoftmaxMaxShape.GetDimNum() != CONST_FOUR) {
    OPS_LOG_E(context->GetNodeName(), "prev_softmax_max shape not support.");
    return ge::GRAPH_FAILED;
  }
  if (prevAttnOutShape.GetDim(0) != prevSoftmaxMaxShape.GetDim(CONST_TWO)
      || prevAttnOutShape.GetDim(1) != prevSoftmaxMaxShape.GetDim(0)
      || prevSoftmaxMaxShape.GetDim(CONST_THREE) != SOFTMAX_TAIL) {
    OPS_LOG_E(context->GetNodeName(), "prev_attn_out shape and prev_softmax_max shape do not match.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus IfShapeSupport(const gert::TilingContext* context, const gert::Shape labelShape, const size_t inputIndex, const size_t outputIndex) {
  gert::Shape dataShape;
  const size_t indexNull = -1;
  if (inputIndex != indexNull) {
    auto dataShapePtr = context->GetInputShape(inputIndex);
    OPS_LOG_E_IF_NULL(context, dataShapePtr, return false);
    dataShape = dataShapePtr->GetStorageShape();
  } else {
    auto dataShapePtr = context->GetOutputShape(outputIndex);
    OPS_LOG_E_IF_NULL(context, dataShapePtr, return false);
    dataShape = dataShapePtr->GetStorageShape();
  }

  size_t shapeSize = labelShape.GetDimNum();
  if (dataShape.GetDimNum() != shapeSize) {
    return ge::GRAPH_FAILED;
  }

  for (size_t dimIndex = 0; dimIndex < shapeSize; ++dimIndex) {
    if (labelShape.GetDim(dimIndex) != dataShape.GetDim(dimIndex) || dataShape.GetDim(dimIndex) == 0) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateCheckShape(const gert::TilingContext* context) {
  auto prevAttnOutShapePtr = context->GetInputShape(0);
  OPS_LOG_E_IF_NULL(context, prevAttnOutShapePtr, return false);
  gert::Shape prevAttnOutShape = prevAttnOutShapePtr->GetStorageShape();

  auto prevSoftmaxMaxShapePtr = context->GetInputShape(1);
  OPS_LOG_E_IF_NULL(context, prevSoftmaxMaxShapePtr, return false);
  gert::Shape prevSoftmaxMaxShape = prevSoftmaxMaxShapePtr->GetStorageShape();

  OPS_CHECK(CheckAttnAndSoftmaxShapeSBH(context, prevAttnOutShape, prevSoftmaxMaxShape) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "CheckAttnAndSoftmaxShapeSBH failed"),
                  return ge::GRAPH_FAILED);

  OPS_CHECK(IfShapeSupport(context, prevSoftmaxMaxShape, 2, -1) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "prev_softmax_sum check failed"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(IfShapeSupport(context, prevAttnOutShape, 3, -1) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "cur_attn_out check failed"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(IfShapeSupport(context, prevSoftmaxMaxShape, 4, -1) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "cur_softmax_max check failed"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(IfShapeSupport(context, prevSoftmaxMaxShape, 5, -1) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "cur_softmax_sum check failed"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(IfShapeSupport(context, prevAttnOutShape, -1, 0) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "attn_out check failed"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(IfShapeSupport(context, prevSoftmaxMaxShape, -1, 1) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "softmax_max check failed"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(IfShapeSupport(context, prevSoftmaxMaxShape, -1, 2) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "softmax_sum check failed"),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus IfDtypeSupport(const gert::TilingContext* context, const ge::DataType labelDtype, const size_t inputIndex, const size_t outputIndex) {
  ge::DataType dataDtype;
  const size_t indexNull = -1;
  if (inputIndex != indexNull) {
    auto inputTensor = context->GetInputDesc(inputIndex);
    OPS_LOG_E_IF_NULL(context, inputTensor, return false);
    dataDtype = inputTensor->GetDataType();
  } else {
    auto outputTensor = context->GetInputDesc(outputIndex);
    OPS_LOG_E_IF_NULL(context, outputTensor, return false);
    dataDtype = outputTensor->GetDataType();
  }
  OPS_CHECK(dataDtype != labelDtype,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "dtype not support"),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateCheckDtype(const gert::TilingContext* context) {
  auto attnTensor = context->GetInputDesc(0);
  OPS_LOG_E_IF_NULL(context, attnTensor, return false);
  ge::DataType attnDtype = attnTensor->GetDataType();

  auto softmaxTensor = context->GetInputDesc(1);
  OPS_LOG_E_IF_NULL(context, softmaxTensor, return false);
  ge::DataType softmaxDtype = softmaxTensor->GetDataType();

  OPS_CHECK(attnDtype != ge::DT_FLOAT16 && attnDtype != ge::DT_BF16 && attnDtype != ge::DT_FLOAT,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "prev_attn_out dtype not support"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(softmaxDtype != ge::DT_FLOAT,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "prev_softmax_max dtype not support"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(IfDtypeSupport(context, softmaxDtype, 2, -1) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "prev_softmax_sum dtype not support"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(IfDtypeSupport(context, attnDtype, 3, -1) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "cur_attn_out dtype not support"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(IfDtypeSupport(context, softmaxDtype, 4, -1) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "cur_softmax_max dtype not support"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(IfDtypeSupport(context, softmaxDtype, 5, -1) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "cur_softmax_sum dtype not support"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(IfDtypeSupport(context, attnDtype, -1, 0) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "attn_out dtype not support"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(IfDtypeSupport(context, softmaxDtype, -1, 1) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "softmax_max dtype not support"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(IfDtypeSupport(context, softmaxDtype, -1, 2) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "softmax_sum dtype not support"),
                  return ge::GRAPH_FAILED);

  auto inputTensor = context->GetInputDesc(6);
  if (inputTensor != nullptr){
    auto dataDtype = inputTensor->GetDataType();
    OPS_CHECK(dataDtype != ge::DT_INT64,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "actual_seq_qlen dtype not support"), 
                    return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateInitShapeInfo(const gert::TilingContext* context, RingAttentionUpdateTilingData& tiling) {
  auto prevAttnOutShapePtr = context->GetInputShape(0);
  OPS_LOG_E_IF_NULL(context, prevAttnOutShapePtr, return false);
  gert::Shape prevAttnOutShape = prevAttnOutShapePtr->GetStorageShape();

  auto prevSoftmaxMaxShapePtr = context->GetInputShape(1);
  OPS_LOG_E_IF_NULL(context, prevSoftmaxMaxShapePtr, return false);
  gert::Shape prevSoftmaxMaxShape = prevSoftmaxMaxShapePtr->GetStorageShape();

  int64_t seqNum = prevAttnOutShape.GetDim(0);
  int64_t batchSize = prevAttnOutShape.GetDim(1);
  int64_t headSize = prevAttnOutShape.GetDim(2);
  int64_t headNum = prevSoftmaxMaxShape.GetDim(1);
  int64_t headDim = headSize / headNum;
  int64_t softmaxTailSize = prevSoftmaxMaxShape.GetDim(3);

  tiling.set_batchSize(batchSize);
  tiling.set_headNum(headNum);
  tiling.set_seqNum(seqNum);
  tiling.set_headDim(headDim);
  tiling.set_softmaxTailSize(softmaxTailSize);
  return ge::GRAPH_SUCCESS;
}

static int64_t RingAttentionUpdateGcd(int64_t inputNum0, int64_t inputNum1) {
  if (inputNum1 == 0) {
    return inputNum0;
  } else {
    return RingAttentionUpdateGcd(inputNum1, inputNum0 % inputNum1);
  }
}

static ge::graphStatus SafeDivisionCheck(int64_t inputNum) {
  if (inputNum == 0) {
    return ge::GRAPH_FAILED;
  } else {
    return ge::GRAPH_SUCCESS;
  }
}

static ge::graphStatus RingAttentionUpdateSplitCore(const gert::TilingContext* context, RingAttentionUpdateTilingData& tiling) {
  const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto maxCoreNum = ascendcPlatform.GetCoreNumAiv();

  int64_t seqNum = tiling.get_seqNum();
  int64_t batchSize = tiling.get_batchSize();
  int64_t headNum = tiling.get_headNum();

  int64_t bnNum = batchSize * headNum;
  int64_t groupNum = RingAttentionUpdateGcd(bnNum, maxCoreNum);
  OPS_CHECK(SafeDivisionCheck(groupNum) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Division by zero(groupNum) is not supported"),
                  return ge::GRAPH_FAILED);
  int64_t coreNumGroup = maxCoreNum / groupNum;
  int64_t bnNumGroup = bnNum / groupNum;
  OPS_CHECK(SafeDivisionCheck(coreNumGroup) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Division by zero(coreNumGroup) is not supported"),
                  return ge::GRAPH_FAILED);
  int64_t seqNumCoreEach = (seqNum + coreNumGroup - 1) / coreNumGroup;
  OPS_CHECK(SafeDivisionCheck(seqNumCoreEach) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Division by zero(seqNumCoreEach) is not supported"),
                  return ge::GRAPH_FAILED);
  coreNumGroup = (seqNum + seqNumCoreEach - 1) / seqNumCoreEach;
  int64_t seqNumCoreTail = seqNum - (coreNumGroup - 1) * seqNumCoreEach;
  int64_t coreNum = coreNumGroup * groupNum;

  tiling.set_coreNum(coreNum);
  tiling.set_coreNumGroup(coreNumGroup);
  tiling.set_bnNumGroup(bnNumGroup);
  tiling.set_seqNumCoreEach(seqNumCoreEach);
  tiling.set_seqNumCoreTail(seqNumCoreTail);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateSplitLoop(const gert::TilingContext* context, RingAttentionUpdateTilingData& tiling) {
  uint64_t maxUbSize;
  const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);

  int64_t headDim = tiling.get_headDim();
  int64_t softmaxTailSize = tiling.get_softmaxTailSize();

  int64_t inputSize;
  auto attnTensor = context->GetInputDesc(0);
  OPS_LOG_E_IF_NULL(context, attnTensor, return false);
  auto attnDtype = attnTensor->GetDataType();
  if (attnDtype == ge::DT_FLOAT16 || attnDtype == ge::DT_BF16) {
    inputSize = SIZE_B16;
  } else if (attnDtype == ge::DT_FLOAT) {
    inputSize = SIZE_B32;
  } else {
    OPS_LOG_E(context->GetNodeName(), "Dtype only support fp16, fp32, bf16 currently.");
    return ge::GRAPH_FAILED;
  }

  int64_t doubleBufferNum = 2;
  int64_t softmaxQueueNum = 6;
  int64_t softmaxBufferTempNum = 2;
  int64_t attnQueueNum = 3;
  int64_t attnBufferTempNum = 2;
  int64_t softmaxEleSize = softmaxQueueNum * doubleBufferNum * SIZE_B32 + softmaxBufferTempNum * SIZE_B32;
  int64_t attnEleSize = attnQueueNum * doubleBufferNum * inputSize + attnBufferTempNum * SIZE_B32;

  int64_t seqNumLoopMin = REPEAT_NUM_B32 / softmaxTailSize;
  int64_t seqNumLoopMax = 255;
  int64_t softmaxSizeMin = seqNumLoopMin * softmaxTailSize * softmaxEleSize;
  int64_t ubSizeRemain = maxUbSize - softmaxSizeMin;

  int64_t headDimLoopAlign = REPEAT_NUM_B32;
  OPS_CHECK(SafeDivisionCheck(seqNumLoopMin) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Division by zero(seqNumLoopMin) is not supported"),
                  return ge::GRAPH_FAILED);
  OPS_CHECK(SafeDivisionCheck(attnEleSize) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Division by zero(attnEleSize) is not supported"),
                  return ge::GRAPH_FAILED);
  int64_t headDimLoopMax = ubSizeRemain / (seqNumLoopMin * attnEleSize);
  headDimLoopMax = headDimLoopMax / headDimLoopAlign * headDimLoopAlign;
  int64_t seqNumLoopEach;
  int64_t headDimLoopEach;
  if (headDim < headDimLoopMax) {
    headDimLoopEach = (headDim + headDimLoopAlign - 1) / headDimLoopAlign * headDimLoopAlign;
    seqNumLoopEach = maxUbSize / (softmaxTailSize * softmaxEleSize + headDimLoopEach * attnEleSize);
    if (seqNumLoopEach > seqNumLoopMax) {
      seqNumLoopEach = seqNumLoopMax;
    }
    seqNumLoopEach = seqNumLoopEach / seqNumLoopMin * seqNumLoopMin;
  } else {
    headDimLoopEach = headDimLoopMax;
    seqNumLoopEach = seqNumLoopMin;
  }

  tiling.set_seqNumLoopEach(seqNumLoopEach);
  tiling.set_headDimLoopEach(headDimLoopEach);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateTNDUbSizeCheck(const gert::TilingContext* context,
                                                        uint64_t headNum, uint64_t headDim) {
  OPS_LOG_D(context->GetNodeName(), "headNum in TND = %lu", headNum);
  OPS_LOG_D(context->GetNodeName(), "headDim in TND = %lu", headDim);
  if (headDim % HEAD_DIM_ALIGN_TND != 0) {
    OPS_LOG_D(context->GetNodeName(), "headDim in TND should be aligned to 64!");
    return ge::GRAPH_FAILED;
  }
  auto attnTensor = context->GetInputDesc(0);
  OPS_LOG_E_IF_NULL(context, attnTensor, return false);
  auto attnDtype = attnTensor->GetDataType();
  int64_t inputSize;
  if (attnDtype == ge::DT_FLOAT) {
    inputSize = SIZE_B32;
  } else if (attnDtype == ge::DT_FLOAT16 || attnDtype == ge::DT_BF16) {
    inputSize = SIZE_B16;
  } else {
    OPS_LOG_E(context->GetNodeName(), "Dtype only support fp16, fp32, bf16 currently.");
    return ge::GRAPH_FAILED;
  }
  OPS_LOG_D(context->GetNodeName(), "input data type size = %ld", inputSize);

  auto attnEleNumLoop = (headNum * headDim + REPEAT_NUM_B32 - 1) / REPEAT_NUM_B32 * REPEAT_NUM_B32;
  auto softmaxEleNumLoop = (headNum * SOFTMAX_TAIL + REPEAT_NUM_B32 - 1) / REPEAT_NUM_B32 * REPEAT_NUM_B32;
  OPS_LOG_D(context->GetNodeName(), "attnEleNumLoop = %lu", attnEleNumLoop);
  OPS_LOG_D(context->GetNodeName(), "softmaxEleNumLoop = %lu", softmaxEleNumLoop);

  auto prevCurAttnOutQueueSize = TND_BUFFER_NUM * attnEleNumLoop * BUFFER_NUM_IN_QUE * inputSize;
  auto prevCurSoftmaxEleQueueSize = TND_BUFFER_NUM * softmaxEleNumLoop * BUFFER_NUM_IN_QUE * FLOAT_DATA_SIZE;
  auto attnOutQueueSize = TND_BUFFER_NUM * attnEleNumLoop * inputSize;
  auto softmaxEleQueueSize = TND_BUFFER_NUM * softmaxEleNumLoop * FLOAT_DATA_SIZE;
  auto tempFp32BufSize = (attnEleNumLoop * TND_BUFFER_NUM + softmaxEleNumLoop * TND_BUFFER_NUM) * FLOAT_DATA_SIZE;

  auto totalBufferSizeUsed = prevCurAttnOutQueueSize + prevCurSoftmaxEleQueueSize * 2 +
                            attnOutQueueSize + softmaxEleQueueSize * 2 + tempFp32BufSize;
  OPS_LOG_D(context->GetNodeName(), "Total UB Size Used = %lu", totalBufferSizeUsed);
  OPS_LOG_D(context->GetNodeName(), "Max UB size = %lu", MAX_UB_SIZE);
  if (totalBufferSizeUsed > MAX_UB_SIZE) {
    OPS_LOG_E(context->GetNodeName(), "UB size doesn't support this shape currently, please try to use smaller N or D.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4RingAttentionUpdateTND(const gert::TilingContext* context, RingAttentionUpdateTilingData& tiling) {
  // infer shape
  auto prevAttnOutShapePtr = context->GetInputShape(0);
  OPS_LOG_E_IF_NULL(context, prevAttnOutShapePtr, return false);
  gert::Shape prevAttnOutShape = prevAttnOutShapePtr->GetStorageShape();

  auto prevSoftmaxMaxShapePtr = context->GetInputShape(1);
  OPS_LOG_E_IF_NULL(context, prevSoftmaxMaxShapePtr, return false);
  gert::Shape prevSoftmaxMaxShape = prevSoftmaxMaxShapePtr->GetStorageShape();

  auto prevActualSeqQlenShapePtr = context->GetInputShape(6);
  OPS_LOG_E_IF_NULL(context, prevActualSeqQlenShapePtr, return false);
  gert::Shape prevActualSeqQlenShape = prevActualSeqQlenShapePtr->GetStorageShape();

  int64_t batchSize = prevActualSeqQlenShape.GetDim(0) - 1;
  int64_t headNum = prevAttnOutShape.GetDim(1);
  int64_t headDim = prevAttnOutShape.GetDim(2);
  int64_t softmaxTailSize = prevSoftmaxMaxShape.GetDim(2);

  tiling.set_batchSize(batchSize);
  tiling.set_headNum(headNum);
  tiling.set_headDim(headDim);
  tiling.set_softmaxTailSize(softmaxTailSize);

  const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto maxCoreNum = ascendcPlatform.GetCoreNumAiv();

  OPS_CHECK(SafeDivisionCheck(maxCoreNum) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Division by zero(maxCoreNum) is not supported"),
                  return ge::GRAPH_FAILED);
  int64_t batchSizeCoreEach = (batchSize + maxCoreNum - 1) / maxCoreNum;
  OPS_CHECK(SafeDivisionCheck(batchSizeCoreEach) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Division by zero(batchSizeCoreEach) is not supported"),
                  return ge::GRAPH_FAILED);
  int64_t coreNum = (batchSize + batchSizeCoreEach - 1) / batchSizeCoreEach;
  int64_t batchSizeCoreTail = batchSize - (coreNum - 1) * batchSizeCoreEach;

  tiling.set_coreNum(coreNum);
  tiling.set_batchSizeCoreEach(batchSizeCoreEach);
  tiling.set_batchSizeCoreTail(batchSizeCoreTail);

  tiling.set_seqNumLoopEach(SEQ_NUM_LOOP_EACH_TND);
  tiling.set_headNumLoopEach(headNum);

  OPS_CHECK(RingAttentionUpdateTNDUbSizeCheck(context, headNum, headDim) != ge::GRAPH_SUCCESS,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Input conflicts with TND constraints! Please check your input."),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus Tiling4RingAttentionUpdate(gert::TilingContext* context) {
  OPS_LOG_D(context->GetNodeName(), "RingAttentionUpdateTiling tiling start");
  // init tiling data
  RingAttentionUpdateTilingData tiling;
  InitTilingData(tiling);
  // get attr
  auto attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return false);
  const char* inputLayout = attrs->GetAttrPointer<char>(0);

  OPS_CHECK(inputLayout == nullptr,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Get required attr input_layout failed, tiling failed"),
                  return ge::GRAPH_FAILED);
  std::string inputLayoutStr = inputLayout;
  OPS_LOG_D(context->GetNodeName(), "test input_layout currently.");
  uint32_t tilingKey = 0;
  if (inputLayoutStr == "TND") {
    OPS_LOG_D(context->GetNodeName(), "Attr input_layout is TND.");
    OPS_CHECK(Tiling4RingAttentionUpdateTND(context, tiling) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Tiling4RingAttentionUpdateTND failed, tiling failed"),
                    return ge::GRAPH_FAILED);
    // check dtype
    OPS_LOG_D(context->GetNodeName(), "RingAttentionUpdateTNDCheckDtype start");
    OPS_CHECK(RingAttentionUpdateCheckDtype(context) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "RingAttentionUpdateTNDCheckDtype failed."),
                    return ge::GRAPH_FAILED);
    tilingKey = TND_KEY;
  } else {
    // check shape
    OPS_LOG_D(context->GetNodeName(), "RingAttentionUpdateCheckShape start");
    OPS_CHECK(RingAttentionUpdateCheckShape(context) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "RingAttentionUpdateCheckShape failed."),
                    return ge::GRAPH_FAILED);
    // check dtype
    OPS_LOG_D(context->GetNodeName(), "RingAttentionUpdateCheckDtype start");
    OPS_CHECK(RingAttentionUpdateCheckDtype(context) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "RingAttentionUpdateCheckDtype failed."),
                    return ge::GRAPH_FAILED);
    // init shape info
    OPS_CHECK(RingAttentionUpdateInitShapeInfo(context, tiling) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "RingAttentionUpdateInitShapeInfo failed."),
                    return ge::GRAPH_FAILED);
    OPS_LOG_D(context->GetNodeName(), "RingAttentionUpdateTiling RingAttentionUpdateInitShapeInfo");
    // tiling core
    OPS_CHECK(RingAttentionUpdateSplitCore(context, tiling) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "RingAttentionUpdateSplitCore failed."),
                    return ge::GRAPH_FAILED);
    OPS_LOG_D(context->GetNodeName(), "RingAttentionUpdateTiling RingAttentionUpdateSplitCore");
    // tiling loop
    OPS_CHECK(RingAttentionUpdateSplitLoop(context, tiling) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "RingAttentionUpdateSplitLoop failed."),
                    return ge::GRAPH_FAILED);
    OPS_LOG_D(context->GetNodeName(), "RingAttentionUpdateTiling RingAttentionUpdateSplitLoop");
  }
  // print tiling param
  RingAttentionUpdatePrintParam(context, tiling);
  // init tiling setting
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  context->SetBlockDim(tiling.get_coreNum());

  size_t sysWorkspaceSize = 16 * 1024 * 1024;
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = sysWorkspaceSize;

  auto attnTensor = context->GetInputDesc(0);
  OPS_LOG_E_IF_NULL(context, attnTensor, return false);
  auto attnDtype = attnTensor->GetDataType();
  if (attnDtype == ge::DT_FLOAT16) {
      tilingKey = tilingKey + DTYPE_KEY_FP16;
  } else if (attnDtype == ge::DT_BF16) {
      tilingKey = tilingKey + DTYPE_KEY_BF16;
  } else if (attnDtype == ge::DT_FLOAT) {
      tilingKey = tilingKey + DTYPE_KEY_FP32;
  } else {
    OPS_LOG_E(context->GetNodeName(), "Dtype only support fp16, fp32, bf16 currently.");
    return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(tilingKey);
  OPS_LOG_D(context->GetNodeName(), "RingAttentionUpdateTiling tiling end");
  return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepare4RingAttentionUpdate(gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

struct RingAttentionUpdateCompileInfo {};

IMPL_OP_OPTILING(RingAttentionUpdate)
    .Tiling(Tiling4RingAttentionUpdate)
    .TilingParse<RingAttentionUpdateCompileInfo>(TilingPrepare4RingAttentionUpdate);

}  // namespace optiling