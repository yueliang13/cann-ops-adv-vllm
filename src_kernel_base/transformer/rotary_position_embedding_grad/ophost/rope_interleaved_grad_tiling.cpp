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
 * \file rope_interleaved_grad_tiling.cpp
 * \brief
 */
#include "rope_interleaved_grad_tiling.h"
#include "rotary_position_embedding_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/ops_log.h"

namespace {
  constexpr uint64_t INPUT_GRAD_IDX = 0;
  constexpr uint64_t INPUT_COS_IDX = 1;
  constexpr uint64_t INPUT_SIN_IDX = 2;
  constexpr uint64_t INPUT_X_IDX = 3;
  constexpr uint64_t INPUT_DIM_NUM = 4;
  constexpr uint64_t NEED_BACKWARD_ATTR_IDX = 1;
  constexpr uint64_t LAYOUT_ATTR_IDX = 1;
  constexpr uint64_t INPUT_DIM_0 = 0;
  constexpr uint64_t INPUT_DIM_1 = 1;
  constexpr uint64_t INPUT_DIM_2 = 2;
  constexpr uint64_t INPUT_DIM_3 = 3;
  constexpr uint64_t LAYOUT_BSND = 0;
  constexpr uint64_t LAYOUT_BNSD = 1;
  constexpr uint64_t LAYOUT_SBND = 2;
  constexpr uint64_t TILING_KEY_FLOAT16 = 0;
  constexpr uint64_t TILING_KEY_BFLOAT16 = 10;
  constexpr uint64_t TILING_KEY_FLOAT32 = 20;
  constexpr uint64_t TILING_KEY_SMALL = 0;
  constexpr uint64_t TILING_KEY_LARGE = 100;
  constexpr uint64_t TILING_KEY_NONEEDBACKWARD = 0;
  constexpr uint64_t TILING_KEY_NEEDBACKWARD = 1000;
  constexpr uint64_t SIZE_FLOAT16 = 2;
  constexpr uint64_t SIZE_BFLOAT16 = 2;
  constexpr uint64_t SIZE_FLOAT32 = 4;
  constexpr uint64_t BASE_TILING_KEY = 20000;
  constexpr uint64_t ALIGN_FP32_BLOCK = 8;
  constexpr uint64_t ALIGN_FP16_BLOCK = 16;
  constexpr uint64_t INPUT_OUTPUT_NUM = 7;
  constexpr uint64_t CALC_NUM = 3;
  constexpr uint64_t DOUBLE_BUFFER = 2;
  constexpr uint64_t FP32_DIVIDE_FP16 = 2;
  constexpr uint64_t FP16_EXTRA = 2;
  constexpr uint64_t FP32_EXTRA = 1;

  // other 
  constexpr uint64_t EXTRA_FP16_BF16_BUFFER_NUM = 8;
  constexpr uint64_t BASE_BUFFER_NUM = 14 + 3;
  constexpr uint64_t ONE_BLOCK = 32;
  constexpr uint64_t ONE_KB = 1024;
  constexpr uint64_t RESERVE_NUM = 4;

  uint64_t dtypeSize = 2;
  uint64_t batchSize;
  uint64_t seqLen;
  uint64_t numHeads;
  uint64_t headDim;
  uint64_t alignHeadDim;
  uint64_t padHeadDim;

  uint64_t bufferSize;
  uint64_t wholeBufferNum;
  uint64_t maxElementNum;
  uint64_t seqFrontLen;
  uint64_t seqTailLen;
  uint64_t layout;
  uint64_t wholeBufferBytes;
  uint64_t tilingKey;

  uint64_t GetDiv(uint64_t value1, uint64_t value2) {
    if (value2 == 0) return value2;
    return value1 / value2;
  }
  
  uint64_t GetCeilInt(uint64_t value1, uint64_t value2) {
    if (value2 == 0) return value2;
    return (value1 + value2 - 1) / value2;
  }

  uint64_t GetDivRem(uint64_t value1, uint64_t value2) {
    if (value2 == 0) return value2;
    return value1 % value2;
  }
}

namespace optiling {
  
  RotaryPositionEmbeddingGradTilingData tiling;

  static void PrintInfo(gert::TilingContext* context) {
      OPS_LOG_D(context, " batchSize=%lu.", tiling.ropeInterleavedGradParams.get_batchSize());
      OPS_LOG_D(context, " seqLen=%lu.", tiling.ropeInterleavedGradParams.get_seqLen());
      OPS_LOG_D(context, " numHeads=%lu.", tiling.ropeInterleavedGradParams.get_numHeads());
      OPS_LOG_D(context, " headDim=%lu.", tiling.ropeInterleavedGradParams.get_headDim());
      OPS_LOG_D(context, " alignHeadDim=%lu.", tiling.ropeInterleavedGradParams.get_alignHeadDim());
      OPS_LOG_D(context, " padHeadDim=%lu.", tiling.ropeInterleavedGradParams.get_padHeadDim());
      OPS_LOG_D(context, " frontCoreNum=%lu.", tiling.ropeInterleavedGradParams.get_frontCoreNum());
      OPS_LOG_D(context, " tailCoreNum=%lu.", tiling.ropeInterleavedGradParams.get_tailCoreNum());
      OPS_LOG_D(context, " seqFrontLen=%lu.", tiling.ropeInterleavedGradParams.get_seqFrontLen());
      OPS_LOG_D(context, " seqTailLen=%lu.", tiling.ropeInterleavedGradParams.get_seqTailLen());

      OPS_LOG_D(context, " seqFrontCalcNum=%lu.", tiling.ropeInterleavedGradParams.get_seqFrontCalcNum());
      OPS_LOG_D(context, " seqFrontCalcLoop=%lu.", tiling.ropeInterleavedGradParams.get_seqFrontCalcLoop());
      OPS_LOG_D(context, " seqFrontCalcTail=%lu.", tiling.ropeInterleavedGradParams.get_seqFrontCalcTail());
      OPS_LOG_D(context, " seqTailCalcNum=%lu.", tiling.ropeInterleavedGradParams.get_seqTailCalcNum());
      OPS_LOG_D(context, " seqTailCalcLoop=%lu.", tiling.ropeInterleavedGradParams.get_seqTailCalcLoop());
      OPS_LOG_D(context, " seqTailCalcTail=%lu.", tiling.ropeInterleavedGradParams.get_seqTailCalcTail());
      OPS_LOG_D(context, " numHeadsLength=%lu.", tiling.ropeInterleavedGradParams.get_numHeadsLength());
      OPS_LOG_D(context, " numHeadsLoop=%lu.", tiling.ropeInterleavedGradParams.get_numHeadsLoop());
      OPS_LOG_D(context, " numHeadsTail=%lu.", tiling.ropeInterleavedGradParams.get_numHeadsTail());
      OPS_LOG_D(context, " batchNumHeadsLength=%lu.", tiling.ropeInterleavedGradParams.get_batchNumHeadsLength());
      OPS_LOG_D(context, " batchNumHeadsLoop=%lu.", tiling.ropeInterleavedGradParams.get_batchNumHeadsLoop());
      OPS_LOG_D(context, " batchNumHeadsTail=%lu.", tiling.ropeInterleavedGradParams.get_batchNumHeadsTail());
      OPS_LOG_D(context, " layout=%lu.", tiling.ropeInterleavedGradParams.get_layout());
  }

  static void UbTilingCalc() {
    uint64_t seqFrontCalcNum = 0, seqFrontCalcLoop = 0, seqFrontCalcTail = 0;
    uint64_t seqTailCalcNum = 0, seqTailCalcLoop = 0, seqTailCalcTail = 0;
    uint64_t numHeadsLength = 0, numHeadsLoop = 0, numHeadsTail = 0;
    uint64_t batchNumHeadsLength = 0, batchNumHeadsLoop = 0, batchNumHeadsTail = 0;
    if (maxElementNum >= seqFrontLen * batchSize * numHeads * alignHeadDim) {  // full bsnd
      seqFrontCalcNum = seqFrontLen;
      seqFrontCalcLoop = 1;
      seqFrontCalcTail = 0;
    } else if (maxElementNum >= batchSize * numHeads * alignHeadDim) {     // full bnd and split s
      seqFrontCalcNum = GetDiv(bufferSize, (batchSize * numHeads * alignHeadDim * wholeBufferNum * dtypeSize));
      seqFrontCalcLoop = GetCeilInt(seqFrontLen, seqFrontCalcNum);
      seqFrontCalcTail = GetDivRem(seqFrontLen, seqFrontCalcNum) != 0 ? seqFrontLen - (seqFrontCalcLoop - 1) * seqFrontCalcNum : 0;
    } else if (maxElementNum >= alignHeadDim) {        // full d
      if (layout == 0) {
        numHeadsLength = GetDiv(bufferSize, (alignHeadDim * wholeBufferNum * dtypeSize));
        numHeadsLoop = GetCeilInt(numHeads, numHeadsLength);
        numHeadsTail = GetDivRem(numHeads, numHeadsLength) != 0 ? numHeads - (numHeadsLoop - 1) * numHeadsLength : 0;
      } else {
        batchNumHeadsLength = GetDiv(bufferSize, (alignHeadDim * wholeBufferNum * dtypeSize));
        batchNumHeadsLoop = GetCeilInt(batchSize * numHeads, batchNumHeadsLength);
        batchNumHeadsTail = GetDivRem(batchSize * numHeads, batchNumHeadsLength) != 0 ? (batchSize * numHeads) - (batchNumHeadsLoop - 1) * batchNumHeadsLength : 0;
      }
    }
    
    if (seqTailLen != 0) {
      if (maxElementNum >= seqTailLen * batchSize * numHeads * alignHeadDim) {
        seqTailCalcNum = seqTailLen;
        seqTailCalcLoop = 1;
        seqTailCalcTail = 0;
      } else if (maxElementNum >= batchSize * numHeads * alignHeadDim) {
        seqTailCalcNum = GetDiv(bufferSize, (batchSize * numHeads * alignHeadDim * wholeBufferNum * dtypeSize));
        seqTailCalcLoop = GetCeilInt(seqTailLen, seqTailCalcNum);
        seqTailCalcTail = GetDivRem(seqTailLen, seqTailCalcNum) != 0 ? seqTailLen - (seqTailCalcLoop - 1) * seqTailCalcNum : 0;
      }
    }
    if (numHeadsLoop == 0 && batchNumHeadsLoop == 0) {
      tilingKey += TILING_KEY_SMALL;
    } else {
      tilingKey += TILING_KEY_LARGE;
    }
    tiling.ropeInterleavedGradParams.set_seqFrontCalcNum(seqFrontCalcNum);
    tiling.ropeInterleavedGradParams.set_seqFrontCalcLoop(seqFrontCalcLoop);
    tiling.ropeInterleavedGradParams.set_seqFrontCalcTail(seqFrontCalcTail);
    
    tiling.ropeInterleavedGradParams.set_numHeadsLength(numHeadsLength);
    tiling.ropeInterleavedGradParams.set_numHeadsLoop(numHeadsLoop);
    tiling.ropeInterleavedGradParams.set_numHeadsTail(numHeadsTail);

    tiling.ropeInterleavedGradParams.set_batchNumHeadsLength(batchNumHeadsLength);
    tiling.ropeInterleavedGradParams.set_batchNumHeadsLoop(batchNumHeadsLoop);
    tiling.ropeInterleavedGradParams.set_batchNumHeadsTail(batchNumHeadsTail);
  
    tiling.ropeInterleavedGradParams.set_seqTailCalcNum(seqTailCalcNum);
    tiling.ropeInterleavedGradParams.set_seqTailCalcLoop(seqTailCalcLoop);
    tiling.ropeInterleavedGradParams.set_seqTailCalcTail(seqTailCalcTail);

    tiling.ropeInterleavedGradParams.set_layout(layout);
  }

  ge::graphStatus RopeCheckInputShape(gert::TilingContext *context, const gert::StorageShape *xShape,
                                      const gert::StorageShape *cosShape, const gert::StorageShape *sinShape)
  {
      if (xShape == nullptr) {
          OPS_LOG_E(context, "[RopeCheckInputShape] xShape is null.");
          return ge::GRAPH_FAILED;
      }
      if (cosShape == nullptr) {
          OPS_LOG_E(context, "[RopeCheckInputShape] cosShape is null.");
          return ge::GRAPH_FAILED;
      }
      if (sinShape == nullptr) {
          OPS_LOG_E(context, "[RopeCheckInputShape] sinShape is null.");
          return ge::GRAPH_FAILED;
      }
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
      uint32_t xHeadDim = xShape->GetStorageShape().GetDim(INPUT_DIM_3);
      uint32_t cosHeadDim = cosShape->GetStorageShape().GetDim(INPUT_DIM_3);
      uint32_t sinHeadDim = sinShape->GetStorageShape().GetDim(INPUT_DIM_3);
      if ((xHeadDim != cosHeadDim) && (xHeadDim != sinHeadDim)) {
          OPS_LOG_E(context, "The last dim of inputs x, cos, sin is inconsistent.");
          return ge::GRAPH_FAILED;
      }
      return ge::GRAPH_SUCCESS;
  }

  static ge::graphStatus TilingSplitS(gert::TilingContext* context, uint64_t coreNum, uint64_t ubSize) {
    batchSize = tiling.ropeInterleavedGradParams.get_batchSize();
    seqLen = tiling.ropeInterleavedGradParams.get_seqLen();
    numHeads = tiling.ropeInterleavedGradParams.get_numHeads();
    headDim = tiling.ropeInterleavedGradParams.get_headDim();
    auto inputGradIdx = context->GetInputDesc(INPUT_GRAD_IDX);
    if (inputGradIdx == nullptr) {
        OPS_LOG_E(context, "[TilingSplitS] inputGradIdx is null.");
        return ge::GRAPH_FAILED;
    }
    auto dataDtype = inputGradIdx->GetDataType();
    if (dataDtype == ge::DT_FLOAT16 || dataDtype == ge::DT_BF16) {
      alignHeadDim = GetCeilInt(headDim, ALIGN_FP16_BLOCK) * ALIGN_FP16_BLOCK;
    } else {
      alignHeadDim = GetCeilInt(headDim, ALIGN_FP32_BLOCK) * ALIGN_FP32_BLOCK;
    }
    padHeadDim = alignHeadDim - headDim;
    tiling.ropeInterleavedGradParams.set_alignHeadDim(alignHeadDim);
    tiling.ropeInterleavedGradParams.set_padHeadDim(padHeadDim);

    uint64_t frontCoreNum = seqLen % coreNum != 0 ? seqLen % coreNum : coreNum;
    uint64_t tailCoreNum = seqLen < coreNum ? 0 : coreNum - frontCoreNum;
    uint64_t blockDim = frontCoreNum + tailCoreNum;

    seqFrontLen = GetCeilInt(seqLen, coreNum);
    seqTailLen = GetDiv(seqLen, coreNum);

    tiling.ropeInterleavedGradParams.set_frontCoreNum(frontCoreNum);
    tiling.ropeInterleavedGradParams.set_tailCoreNum(tailCoreNum);
    tiling.ropeInterleavedGradParams.set_seqFrontLen(seqFrontLen);
    tiling.ropeInterleavedGradParams.set_seqTailLen(seqTailLen);

    context->SetBlockDim(blockDim);

    uint64_t reserveBufferSize = ONE_BLOCK;
    if (ubSize < reserveBufferSize) {
      OPS_LOG_E(context, "Because the size of the shape D is too large, it exceeds the ub range!");
      return ge::GRAPH_FAILED;
    }

    bufferSize = ubSize - reserveBufferSize;
    if (dataDtype == ge::DT_FLOAT16 || dataDtype == ge::DT_BF16) {
      wholeBufferNum = (INPUT_OUTPUT_NUM + CALC_NUM) * FP32_DIVIDE_FP16 + INPUT_OUTPUT_NUM * DOUBLE_BUFFER + FP16_EXTRA;
    } else {
      wholeBufferNum = INPUT_OUTPUT_NUM * DOUBLE_BUFFER + CALC_NUM;
    }

    if (dataDtype == ge::DT_FLOAT16 || dataDtype == ge::DT_BF16) {
      if (layout == LAYOUT_BSND || layout == LAYOUT_SBND) {
        wholeBufferNum += FP16_EXTRA;
      }
      wholeBufferBytes = wholeBufferNum * SIZE_FLOAT16;
    } else {
      if (layout == LAYOUT_BSND || layout == LAYOUT_SBND) {
        wholeBufferNum += FP32_EXTRA;
      }
      wholeBufferBytes = wholeBufferNum * SIZE_FLOAT32;
    }

    maxElementNum = GetDiv(bufferSize, wholeBufferBytes);
    UbTilingCalc();
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus TilingLayoutSplit(gert::TilingContext* context, const gert::StorageShape* xShape, 
                                      const gert::StorageShape* cosShape) {
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
    if (cosShape0 == 1 && cosShape2 == 1 && xShape1 == cosShape1) {
      // BSND
      tiling.ropeInterleavedGradParams.set_batchSize(xShape0);
      tiling.ropeInterleavedGradParams.set_seqLen(xShape1);
      tiling.ropeInterleavedGradParams.set_numHeads(xShape2);
      tiling.ropeInterleavedGradParams.set_headDim(xHeadDim);
      layout = LAYOUT_BSND;
    } else if (cosShape0 == 1 && cosShape1 == 1 && xShape2 == cosShape2) {
      // BNSD
      tiling.ropeInterleavedGradParams.set_batchSize(xShape0);
      tiling.ropeInterleavedGradParams.set_seqLen(xShape2);
      tiling.ropeInterleavedGradParams.set_numHeads(xShape1);
      tiling.ropeInterleavedGradParams.set_headDim(xHeadDim);
      layout = LAYOUT_BNSD;
    } else if (cosShape1 == 1 && cosShape2 == 1 && xShape0 == cosShape0) {
      // SBND
      tiling.ropeInterleavedGradParams.set_batchSize(xShape1);
      tiling.ropeInterleavedGradParams.set_seqLen(xShape0);
      tiling.ropeInterleavedGradParams.set_numHeads(xShape2);
      tiling.ropeInterleavedGradParams.set_headDim(xHeadDim);
      layout = LAYOUT_SBND;
    } else {
      OPS_LOG_E(context, "The shape of the input x, cos and sin is not supported.");
      return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(INPUT_X_IDX) != nullptr) {
      tilingKey += TILING_KEY_NEEDBACKWARD;
    } else {
      tilingKey += TILING_KEY_NONEEDBACKWARD;
    }
    return TilingSplitS(context, coreNum, ubSize);
  }

  ge::graphStatus Tiling4RopeInterleavedGrad(gert::TilingContext* context) {
    const gert::StorageShape* xShape = context->GetInputShape(INPUT_GRAD_IDX);
    const gert::StorageShape* cosShape = context->GetInputShape(INPUT_COS_IDX);
    const gert::StorageShape* sinShape = context->GetInputShape(INPUT_SIN_IDX);

    if (RopeCheckInputShape(context, xShape, cosShape, sinShape) != ge::GRAPH_SUCCESS) {
      OPS_LOG_E(context, "RopeCheckInputShape fail.");
      return ge::GRAPH_FAILED;
    }
    tilingKey = BASE_TILING_KEY;
    auto dataDtype = context->GetInputDesc(INPUT_GRAD_IDX)->GetDataType();
    if (dataDtype == ge::DT_FLOAT16) {
      dtypeSize = SIZE_FLOAT16;
      tilingKey += TILING_KEY_FLOAT16;
    } else if (dataDtype == ge::DT_BF16) {
      dtypeSize = SIZE_BFLOAT16;
      tilingKey += TILING_KEY_BFLOAT16;
    } else if (dataDtype == ge::DT_FLOAT) {
      dtypeSize = SIZE_FLOAT32;
      tilingKey += TILING_KEY_FLOAT32;
    } else {
      OPS_LOG_E(context, "Operator only support bf16, fp16, fp32 dtype");
      return ge::GRAPH_FAILED;
    }
    if (TilingLayoutSplit(context, xShape, cosShape) != ge::GRAPH_SUCCESS) {
      OPS_LOG_E(context, "TilingSplitS fail.");
      return ge::GRAPH_FAILED;
    }
    
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t usr_workspace_size = 0;
    size_t sys_work_space_size = 16 * 1024 * 1024;
    size_t* current_workspace = context->GetWorkspaceSizes(1);
    current_workspace[0] = usr_workspace_size + sys_work_space_size;
    PrintInfo(context);
    return ge::GRAPH_SUCCESS;
  }
}
