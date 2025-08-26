/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
// ifa must include before pfa
#define FIA_ENABLE_MLA
#include "../incre_flash_attention/incre_flash_attention.cpp"
#include "../prompt_flash_attention/prompt_flash_attention.cpp"

extern "C" __global__ __aicore__ void fused_infer_attention_score(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                             __gm__ uint8_t* pse_shift, __gm__ uint8_t* attenMask,
                                                             __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                                             __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                             __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                             __gm__ uint8_t* quant_offset2, __gm__ uint8_t* antiquantScale,
                                                             __gm__ uint8_t* antiquantOffset, __gm__ uint8_t* blocktable,
                                                             __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                                             __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset,
                                                             __gm__ uint8_t* valueAntiquantScale, __gm__ uint8_t* valueAntiquantOffset,
                                                             __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                                             __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope, __gm__ uint8_t* keyRopeAntiquantScale,
                                                             __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                                             __gm__ uint8_t* tiling) {
  // judge ifa or pfa by range of tilingKey
  if(TILING_KEY_VAR >= 1000000000000000000) {
    prompt_flash_attention_FIAS(query, key, value, pse_shift, attenMask, actualSeqLengths, 
                            actualSeqLengthsKV, deq_scale1, quant_scale1,
                            deq_scale2, quant_scale2, quant_offset2, antiquantScale, 
                            antiquantOffset, blocktable, queryPaddingSize, kvPaddingSize, 
                            keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, 
                            valueAntiquantOffset, keySharedPrefix, valueSharedPrefix, 
                            actualSharedPrefixLen, attentionOut, softmaxLse, workspace, tiling);

  } else {
    incre_flash_attention_FIAS(query, key, value, pse_shift, attenMask, actualSeqLengths,
                          actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2,
                          quant_scale2, quant_offset2, antiquantScale, antiquantOffset, blocktable, kvPaddingSize,
                          keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
                          keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, nullptr, nullptr, nullptr,
                          attentionOut, softmaxLse, workspace, tiling);
  }
}
