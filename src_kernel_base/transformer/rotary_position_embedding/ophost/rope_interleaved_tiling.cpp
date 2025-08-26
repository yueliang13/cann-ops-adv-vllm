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
 * \file rope_interleaved_tiling.cpp
 * \brief
 */
#include "rope_interleaved_tiling.h"
#include "rotary_position_embedding_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/ops_log.h"

namespace {
constexpr uint64_t INPUT_X_IDX = 0;
constexpr uint64_t INPUT_COS_IDX = 1;
constexpr uint64_t INPUT_SIN_IDX = 2;
constexpr uint64_t INPUT_DIM_NUM = 4;
constexpr uint64_t IO_NUM = 3;  // sin、cos -> tri
constexpr uint64_t BF16_TBUF_NUM = 3;
constexpr uint64_t INPUT_DIM_0 = 0;
constexpr uint64_t INPUT_DIM_1 = 1;
constexpr uint64_t INPUT_DIM_2 = 2;
constexpr uint64_t INPUT_DIM_3 = 3;
constexpr uint64_t TILING_KEY_FLOAT16 = 0;
constexpr uint64_t TILING_KEY_BFLOAT16 = 10;
constexpr uint64_t TILING_KEY_FLOAT32 = 20;
constexpr uint64_t TILING_KEY_UNPAD = 0;
constexpr uint64_t TILING_KEY_PAD = 1;
constexpr uint64_t TILING_KEY_SPLIT_S = 0;
constexpr uint64_t TILING_KEY_SPLIT_BS = 100;
constexpr uint64_t TILING_KEY_SPLIT_BSN = 200;
constexpr uint64_t FP16_BF16_DTYPE_SIZE = 2;
constexpr uint64_t FP32_DTYPE_SIZE = 4;
constexpr uint64_t INT32_DTYPE_SIZE = 4;
constexpr uint64_t TBUF_SIZE = 0;
constexpr uint64_t ALIGN_32 = 8;
constexpr uint64_t ALIGN_16 = 16;
optiling::RotaryPositionEmbeddingTilingData tiling;

int32_t GetCeilInt(int32_t value1, int32_t value2) {
  if (value2 == 0) return value2;
  return (value1 + value2 - 1) / value2;
}

int32_t GetDiv(int32_t value1, int32_t value2) {
  if (value2 == 0) return value2;
  return value1 / value2;
}

int32_t GetDivRem(int32_t value1, int32_t value2) {
  if (value2 == 0) return value2;
  return value1 % value2;
}
}

namespace optiling {
static void PrintInfo(gert::TilingContext* context) {
  OPS_LOG_D(context, " batchSize=%lu.",      tiling.ropeInterleavedParams.get_batchSize());
  OPS_LOG_D(context, " seqLen=%lu.",         tiling.ropeInterleavedParams.get_seqLen());
  OPS_LOG_D(context, " numHeads=%lu.",       tiling.ropeInterleavedParams.get_numHeads());
  OPS_LOG_D(context, " headDim=%lu.",        tiling.ropeInterleavedParams.get_headDim());
  OPS_LOG_D(context, " frontCoreNum=%lu.",   tiling.ropeInterleavedParams.get_frontCoreNum());
  OPS_LOG_D(context, " tailCoreNum=%lu.",    tiling.ropeInterleavedParams.get_tailCoreNum());
  OPS_LOG_D(context, " coreCalcNum=%lu.",    tiling.ropeInterleavedParams.get_coreCalcNum());
  OPS_LOG_D(context, " coreCalcTail=%lu.",   tiling.ropeInterleavedParams.get_coreCalcTail());
  OPS_LOG_D(context, " ubCalcNum=%lu.",      tiling.ropeInterleavedParams.get_ubCalcNum());
  OPS_LOG_D(context, " ubCalcLoop=%lu.",     tiling.ropeInterleavedParams.get_ubCalcLoop());
  OPS_LOG_D(context, " ubCalcTail=%lu.",     tiling.ropeInterleavedParams.get_ubCalcTail());
  OPS_LOG_D(context, " ubCalcTailNum=%lu.",  tiling.ropeInterleavedParams.get_ubCalcTailNum());
  OPS_LOG_D(context, " ubCalcTailLoop=%lu.", tiling.ropeInterleavedParams.get_ubCalcTailLoop());
  OPS_LOG_D(context, " ubCalcTailTail=%lu.", tiling.ropeInterleavedParams.get_ubCalcTailTail());
  OPS_LOG_D(context, " ubCalcBNum=%lu.",     tiling.ropeInterleavedParams.get_ubCalcBNum());
  OPS_LOG_D(context, " ubCalcBLoop=%lu.",    tiling.ropeInterleavedParams.get_ubCalcBLoop());
  OPS_LOG_D(context, " ubCalcBTail=%lu.",    tiling.ropeInterleavedParams.get_ubCalcBTail());
  OPS_LOG_D(context, " ubCalcNNum=%lu.",     tiling.ropeInterleavedParams.get_ubCalcNNum());
  OPS_LOG_D(context, " ubCalcNLoop=%lu.",    tiling.ropeInterleavedParams.get_ubCalcNLoop());
  OPS_LOG_D(context, " ubCalcNTail=%lu.",    tiling.ropeInterleavedParams.get_ubCalcNTail());
}

ge::graphStatus CheckInputShape(gert::TilingContext* context, const gert::StorageShape* xShape,
                                const gert::StorageShape* cosShape, const gert::StorageShape* sinShape){
  size_t xShapeSize = xShape->GetStorageShape().GetDimNum();
  size_t cosShapeSize = cosShape->GetStorageShape().GetDimNum();
  size_t sinShapeSize = sinShape->GetStorageShape().GetDimNum();
  if (xShapeSize != INPUT_DIM_NUM && cosShapeSize != INPUT_DIM_NUM && sinShapeSize != INPUT_DIM_NUM) {
    OPS_LOG_E(context, "Inconsistent dimensions of input shape.");
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0; i < xShapeSize; ++i) {
    if (cosShape->GetStorageShape().GetDim(i) != sinShape->GetStorageShape().GetDim(i)) {
      OPS_LOG_E(context, "The shape of the input cos and sin is inconsistent.");
      return ge::GRAPH_FAILED;
    }
  }
  uint64_t xHeadDim = xShape->GetStorageShape().GetDim(INPUT_DIM_3);
  uint64_t cosHeadDim = cosShape->GetStorageShape().GetDim(INPUT_DIM_3);
  uint64_t sinHeadDim = sinShape->GetStorageShape().GetDim(INPUT_DIM_3);
  if ((xHeadDim != cosHeadDim) && (xHeadDim != sinHeadDim)) {
    OPS_LOG_E(context, "The last dim of inputs x, cos, sin is inconsistent.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSplitN(gert::TilingContext* context, uint32_t numHeads, uint32_t headDimAlign,
                             uint64_t ubSize, ge::DataType dataDtype, uint64_t tilingKey) {
  const uint64_t dtypeSize = (dataDtype == ge::DT_FLOAT) ? FP32_DTYPE_SIZE : FP16_BF16_DTYPE_SIZE;
  const uint64_t bufferSize = ubSize - TBUF_SIZE;
  uint64_t totalHeadNum1Size = headDimAlign * IO_NUM * dtypeSize + headDimAlign * INT32_DTYPE_SIZE;
  if (dataDtype == ge::DT_BF16 || dataDtype == ge::DT_FLOAT16) {
    totalHeadNum1Size += headDimAlign * FP32_DTYPE_SIZE * BF16_TBUF_NUM;
  }
  uint32_t ubCalcNNum{1}, ubCalcNLoop{numHeads}, ubCalcNTail{0};
  if (bufferSize < totalHeadNum1Size) {
    OPS_LOG_E(context, "The D dimension of the input shape is too large.");
    return ge::GRAPH_FAILED;
  }
  ubCalcNNum = GetDiv(bufferSize, totalHeadNum1Size);
  ubCalcNLoop = GetCeilInt(numHeads, ubCalcNNum);
  ubCalcNTail = GetDivRem(numHeads, ubCalcNNum) != 0 ? numHeads - (ubCalcNLoop - 1) * ubCalcNNum : 0;
  tiling.ropeInterleavedParams.set_ubCalcNNum(ubCalcNNum);
  tiling.ropeInterleavedParams.set_ubCalcNLoop(ubCalcNLoop);
  tiling.ropeInterleavedParams.set_ubCalcNTail(ubCalcNTail);
  tilingKey += TILING_KEY_SPLIT_BSN;
  context->SetTilingKey(tilingKey);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSplitB(gert::TilingContext* context, uint32_t batchSize, uint32_t numHeads,
                             uint32_t headDimAlign, uint64_t ubSize, ge::DataType dataDtype, uint64_t tilingKey) {
  const uint64_t dtypeSize = (dataDtype == ge::DT_FLOAT) ? FP32_DTYPE_SIZE : FP16_BF16_DTYPE_SIZE;
  const uint64_t tBufferSize = numHeads * headDimAlign * INT32_DTYPE_SIZE + TBUF_SIZE;
  const uint64_t bufferSize = ubSize - tBufferSize;
  uint64_t totalBatch1Size = numHeads * headDimAlign * IO_NUM * dtypeSize;

  if (dataDtype == ge::DT_BF16 || dataDtype == ge::DT_FLOAT16) {
    totalBatch1Size += numHeads * headDimAlign * FP32_DTYPE_SIZE * BF16_TBUF_NUM;
  }
  uint32_t ubCalcBNum{1}, ubCalcBLoop{batchSize}, ubCalcBTail{0};
  if (ubSize < tBufferSize || bufferSize < totalBatch1Size) {
    if (TilingSplitN(context, numHeads, headDimAlign, ubSize, dataDtype, tilingKey) != ge::GRAPH_SUCCESS) {
      OPS_LOG_E(context, "TilingSplitN fail.");
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }

  ubCalcBNum = GetDiv(bufferSize, totalBatch1Size);
  ubCalcBLoop = GetCeilInt(batchSize, ubCalcBNum);
  ubCalcBTail = GetDivRem(batchSize, ubCalcBNum) != 0 ? batchSize - (ubCalcBLoop - 1) * ubCalcBNum : 0;

  tiling.ropeInterleavedParams.set_ubCalcBNum(ubCalcBNum);
  tiling.ropeInterleavedParams.set_ubCalcBLoop(ubCalcBLoop);
  tiling.ropeInterleavedParams.set_ubCalcBTail(ubCalcBTail);

  tilingKey += TILING_KEY_SPLIT_BS;
  context->SetTilingKey(tilingKey);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSplitS(gert::TilingContext* context, uint64_t coreNum, uint64_t ubSize, uint64_t tilingKey) {
  auto xDesc = context->GetInputDesc(INPUT_X_IDX);
  OPS_LOG_E_IF_NULL(context, xDesc, return ge::GRAPH_FAILED);
  auto dataDtype = xDesc->GetDataType();
  uint64_t dtypeSize = (dataDtype == ge::DT_FLOAT) ? FP32_DTYPE_SIZE : FP16_BF16_DTYPE_SIZE;
  uint64_t batchSize = tiling.ropeInterleavedParams.get_batchSize();
  uint64_t seqLen = tiling.ropeInterleavedParams.get_seqLen();
  uint64_t numHeads = tiling.ropeInterleavedParams.get_numHeads();
  uint64_t headDim = tiling.ropeInterleavedParams.get_headDim();
  // block split
  uint64_t frontCoreNum = GetDivRem(seqLen, coreNum) != 0 ? GetDivRem(seqLen, coreNum) : coreNum;
  uint64_t tailCoreNum = seqLen <= coreNum ? 0 : coreNum - frontCoreNum;
  uint64_t blockDim = frontCoreNum + tailCoreNum;
  uint64_t coreCalcNum = GetCeilInt(seqLen, coreNum);
  uint64_t coreCalcTail = GetDiv(seqLen, coreNum);
  tiling.ropeInterleavedParams.set_frontCoreNum(frontCoreNum);
  tiling.ropeInterleavedParams.set_tailCoreNum(tailCoreNum);
  tiling.ropeInterleavedParams.set_coreCalcNum(coreCalcNum);
  tiling.ropeInterleavedParams.set_coreCalcTail(coreCalcTail);
  context->SetBlockDim(blockDim);
  uint64_t alignFactor = (dataDtype == ge::DT_FLOAT) ? ALIGN_32 : ALIGN_16;
  uint64_t headDimAlign;
  if (GetDivRem(headDim, alignFactor) == 0) {
    headDimAlign = headDim;
  } else {
    headDimAlign = GetCeilInt(headDim, alignFactor) * alignFactor;
    tilingKey += TILING_KEY_PAD;
  }
  // ub split
  uint64_t tBufferSize = numHeads * headDimAlign * INT32_DTYPE_SIZE + TBUF_SIZE;
  uint64_t bufferSize = ubSize - tBufferSize;
  uint64_t ioUbSize = batchSize * coreCalcNum * numHeads * headDimAlign * IO_NUM * dtypeSize;
  uint64_t totalSeq1Size = batchSize * numHeads * headDimAlign * IO_NUM * dtypeSize;
  if (dataDtype == ge::DT_BF16 || dataDtype == ge::DT_FLOAT16) {
    ioUbSize += batchSize * coreCalcNum * numHeads * headDimAlign * FP32_DTYPE_SIZE * BF16_TBUF_NUM;
    totalSeq1Size += batchSize * numHeads * headDimAlign * FP32_DTYPE_SIZE * BF16_TBUF_NUM;
  }
  if (tBufferSize >= ubSize) {
    if (TilingSplitN(context, numHeads, headDimAlign, ubSize, dataDtype, tilingKey) != ge::GRAPH_SUCCESS) {
      OPS_LOG_E(context, "TilingSplitN fail.");
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }
  if (ubSize < tBufferSize || bufferSize < totalSeq1Size) {
    if (TilingSplitB(context, batchSize, numHeads, headDimAlign, ubSize, dataDtype, tilingKey) != ge::GRAPH_SUCCESS) {
      OPS_LOG_E(context, "TilingSplitB fail.");
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }
  context->SetTilingKey(tilingKey);
  uint64_t ubCalcNum, ubCalcLoop, ubCalcTail;
  if (bufferSize < ioUbSize) {
    ubCalcNum = GetDiv(bufferSize, totalSeq1Size);
    ubCalcLoop = GetCeilInt(coreCalcNum, ubCalcNum);
    ubCalcTail = GetDivRem(coreCalcNum, ubCalcNum) != 0 ? coreCalcNum - (ubCalcLoop - 1) * ubCalcNum : 0;
  } else {
    ubCalcNum = coreCalcNum;
    ubCalcLoop = 1;
    ubCalcTail = 0;
  }
  tiling.ropeInterleavedParams.set_ubCalcNum(ubCalcNum);
  tiling.ropeInterleavedParams.set_ubCalcLoop(ubCalcLoop);
  tiling.ropeInterleavedParams.set_ubCalcTail(ubCalcTail);
  // ub split for tail core
  uint64_t ubCalcTailNum{0}, ubCalcTailLoop{0}, ubCalcTailTail{0};
  if (coreCalcTail != 0) {
    ioUbSize = batchSize * coreCalcTail * numHeads * headDimAlign * IO_NUM * dtypeSize;
    totalSeq1Size = batchSize * numHeads * headDimAlign * IO_NUM * dtypeSize;
    if (dataDtype == ge::DT_BF16 || dataDtype == ge::DT_FLOAT16) {
      ioUbSize += batchSize * coreCalcNum * numHeads * headDimAlign * FP32_DTYPE_SIZE * BF16_TBUF_NUM;
      totalSeq1Size += batchSize * numHeads * headDimAlign * FP32_DTYPE_SIZE * BF16_TBUF_NUM;
    }
    if (bufferSize < ioUbSize) {
      ubCalcTailNum = GetDiv(bufferSize, totalSeq1Size);
      ubCalcTailLoop = GetCeilInt(coreCalcTail, ubCalcTailNum);
      ubCalcTailTail = GetDivRem(coreCalcTail, ubCalcTailNum) != 0 ?    \
                       coreCalcTail - (ubCalcTailLoop - 1) * ubCalcTailNum : 0;
    } else {
      ubCalcTailNum = coreCalcTail;
      ubCalcTailLoop = 1;
      ubCalcTailTail = 0;
    }
  }
  tiling.ropeInterleavedParams.set_ubCalcTailNum(ubCalcTailNum);
  tiling.ropeInterleavedParams.set_ubCalcTailLoop(ubCalcTailLoop);
  tiling.ropeInterleavedParams.set_ubCalcTailTail(ubCalcTailTail);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSplit(gert::TilingContext* context, const gert::StorageShape* xShape,
                            const gert::StorageShape* cosShape, uint64_t tilingKey) {
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();
  uint64_t ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  uint64_t xShape0 = xShape->GetStorageShape().GetDim(INPUT_DIM_0);
  uint64_t xShape1 = xShape->GetStorageShape().GetDim(INPUT_DIM_1);
  uint64_t xShape2 = xShape->GetStorageShape().GetDim(INPUT_DIM_2);
  uint64_t xHeadDim = xShape->GetStorageShape().GetDim(INPUT_DIM_3);
  uint64_t cosShape0 = cosShape->GetStorageShape().GetDim(INPUT_DIM_0);
  uint64_t cosShape1 = cosShape->GetStorageShape().GetDim(INPUT_DIM_1);
  uint64_t cosShape2 = cosShape->GetStorageShape().GetDim(INPUT_DIM_2);
  uint64_t batchSizeOut{1}, seqLenOut{1}, numHeadsOut{1};
  if (cosShape1 == 1 && cosShape2 == 1 && xShape0 == cosShape0) {
    seqLenOut = cosShape0;
    numHeadsOut = xShape1 * xShape2;      // SBND -> 1S(BN)D -> 1SND
  } else if (cosShape0 == 1 && cosShape2 == 1 && xShape1 == cosShape1) {
    seqLenOut = cosShape1;
    batchSizeOut = xShape0;               // BSND
    numHeadsOut = xShape2;
  } else if (cosShape0 == 1 && cosShape1 == 1 && xShape2 == cosShape2) {
    seqLenOut = cosShape2;
    batchSizeOut = xShape0 * xShape1;     // BNSD -> (BN)S1D -> BS1D
  } else {
    OPS_LOG_E(context, "The shape of the input x, cos and sin is not supported.");
    return ge::GRAPH_FAILED;
  }
  tiling.ropeInterleavedParams.set_batchSize(batchSizeOut);
  tiling.ropeInterleavedParams.set_seqLen(seqLenOut);
  tiling.ropeInterleavedParams.set_numHeads(numHeadsOut);
  tiling.ropeInterleavedParams.set_headDim(xHeadDim);

  if (TilingSplitS(context, coreNum, ubSize, tilingKey) != ge::GRAPH_SUCCESS) {
    OPS_LOG_E(context, "TilingSplitS fail.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4RopeInterleaved(gert::TilingContext* context) {
  const gert::StorageShape* xShape = context->GetInputShape(INPUT_X_IDX);
  OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED);
  const gert::StorageShape* cosShape = context->GetInputShape(INPUT_COS_IDX);
  OPS_LOG_E_IF_NULL(context, cosShape, return ge::GRAPH_FAILED);
  const gert::StorageShape* sinShape = context->GetInputShape(INPUT_SIN_IDX);
  OPS_LOG_E_IF_NULL(context, sinShape, return ge::GRAPH_FAILED);

  if (CheckInputShape(context, xShape, cosShape, sinShape) != ge::GRAPH_SUCCESS) {
    OPS_LOG_E(context, "CheckInputShape fail.");
    return ge::GRAPH_FAILED;
  }

  auto dataDtype = context->GetInputDesc(INPUT_X_IDX)->GetDataType();
  // tilingKey / 100 % 10 : 0=split_s  1=split_bs  2=split_bsn
  // tilingKey / 10 % 10 : 0=float16  1=bfloat16  2=float32
  // tilingKey % 10 : 0=unpad  1=pad
  uint64_t tilingKey = 2000;
  if (dataDtype == ge::DT_FLOAT16) {
    tilingKey += TILING_KEY_FLOAT16;
  } else if (dataDtype == ge::DT_BF16) {
    tilingKey += TILING_KEY_BFLOAT16;
  } else if (dataDtype == ge::DT_FLOAT) {
    tilingKey += TILING_KEY_FLOAT32;
  }

  if (TilingSplit(context, xShape, cosShape, tilingKey) != ge::GRAPH_SUCCESS) {
    OPS_LOG_E(context, "TilingSplit fail.");
    return ge::GRAPH_FAILED;
  }
  
  OPS_LOG_D(context, "[tilingKey]: %lu", context->GetTilingKey());
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t usrWorkspaceSize = 0;
  size_t sysWorkspaceSize = 16 * 1024 * 1024;
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = usrWorkspaceSize + sysWorkspaceSize;

  PrintInfo(context);

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
