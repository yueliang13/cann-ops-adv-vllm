/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file swin_attention_score_quant_tiling.h
 * \brief SwinAttentionScoreQuant tiling define
 */
#ifndef SWIN_ATTENTION_SCORE_QUANT_TILING_H
#define SWIN_ATTENTION_SCORE_QUANT_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(SwinAttentionScoreQuantTilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreLoops);
TILING_DATA_FIELD_DEF(uint32_t, dimB);
TILING_DATA_FIELD_DEF(uint32_t, dimN);
TILING_DATA_FIELD_DEF(uint32_t, dimS);
TILING_DATA_FIELD_DEF(uint32_t, dimH);
TILING_DATA_FIELD_DEF(uint32_t, qSize);
TILING_DATA_FIELD_DEF(uint32_t, kSize);
TILING_DATA_FIELD_DEF(uint32_t, pSize);
TILING_DATA_FIELD_DEF(uint32_t, vSize);
TILING_DATA_FIELD_DEF(uint32_t, cubeSharedUbSize);
TILING_DATA_FIELD_DEF(uint32_t, vecSharedUbSize);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, qkBmmTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, pvBmmTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SwinAttentionScoreQuant, SwinAttentionScoreQuantTilingData)
}
#endif