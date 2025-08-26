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
 * \file flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab_def.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

/////////////////////////////////////////////////////////////////////////
// S1S2_BNGS1S2
/////////////////////////////////////////////////////////////////////////
BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradS1S2BNGS1S2SABBaseParams)
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, n2);
TILING_DATA_FIELD_DEF(int64_t, g);
TILING_DATA_FIELD_DEF(int64_t, s1);
TILING_DATA_FIELD_DEF(int64_t, s2);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(int64_t, s1Token); // pre_tokens
TILING_DATA_FIELD_DEF(int64_t, s2Token); // next_tokens
TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
TILING_DATA_FIELD_DEF(uint32_t, isSparse);
TILING_DATA_FIELD_DEF(int64_t, attenMaskS2Size);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskCompressMode);
TILING_DATA_FIELD_DEF(int64_t, qStartIdx);
TILING_DATA_FIELD_DEF(int64_t, kvStartIdx);
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS1);
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS2);
TILING_DATA_FIELD_DEF(uint32_t, pseType);
TILING_DATA_FIELD_DEF(uint32_t, pseOptional);
TILING_DATA_FIELD_DEF(uint32_t, pseShapeType);
TILING_DATA_FIELD_DEF(uint32_t, pseDtype);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskOptional);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskDtype);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskShapeType);
TILING_DATA_FIELD_DEF(uint32_t, pad);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradS1S2BNGS1S2SABBaseParamsOp, FlashAttentionScoreGradS1S2BNGS1S2SABBaseParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradS1S2BNGS1S2SABSplitCoreParams)
TILING_DATA_FIELD_DEF(int64_t, s1Outer);
TILING_DATA_FIELD_DEF(uint32_t, s1Inner);
TILING_DATA_FIELD_DEF(uint32_t, s1CvInner);
TILING_DATA_FIELD_DEF(uint32_t, s1Tail);
TILING_DATA_FIELD_DEF(uint32_t, s1CvTail);
TILING_DATA_FIELD_DEF(int64_t, s2Outer);
TILING_DATA_FIELD_DEF(uint32_t, s2Inner);
TILING_DATA_FIELD_DEF(uint32_t, s2CvInner);
TILING_DATA_FIELD_DEF(uint32_t, s2Tail);
TILING_DATA_FIELD_DEF(uint32_t, baseMN);
TILING_DATA_FIELD_DEF(uint32_t, blockOuter);
TILING_DATA_FIELD_DEF(int64_t, bandIdx);
END_TILING_DATA_DEF;
// 固定写法不能换行，会失败
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradS1S2BNGS1S2SABSplitCoreParamsOp, FlashAttentionScoreGradS1S2BNGS1S2SABSplitCoreParams)


BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb)
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradS1S2BNGS1S2SABBaseParams, s1s2BNGS1S2BaseParams);
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradS1S2BNGS1S2SABSplitCoreParams, s1s2BNGS1S2SplitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm3TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
TILING_DATA_FIELD_DEF_STRUCT(PreParams, preTilingData);
TILING_DATA_FIELD_DEF_STRUCT(PreSfmgParams, preSfmgTilingData);
TILING_DATA_FIELD_DEF_STRUCT(PostParams, postTilingData);
END_TILING_DATA_DEF;
// 固定写法不能换行，会失败
// BSND 1000000xxxxxxxx0x434(5)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb)
} // namespace optiling
