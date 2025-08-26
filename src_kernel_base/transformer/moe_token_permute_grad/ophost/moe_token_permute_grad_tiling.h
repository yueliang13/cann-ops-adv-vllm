/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file moe_token_permute_grad_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_PERMUTE_GRAD_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_PERMUTE_GRAD_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "log/ops_log.h"
namespace optiling {
namespace permutegrad {
/*
 * permuteGrad复用unpermute代码
 */
constexpr int64_t FLOAT_DATA_SIZE = 4;
constexpr int64_t MIN_BUFFER_NUM = 2;
constexpr int64_t ALIGN_512 = 512;
constexpr int64_t ALIGN_256 = 256;
constexpr int64_t QUE_NUM = 2;
constexpr int64_t CAST_NUM = 2;
constexpr int64_t TILINGKEY_PROBS = 1;
constexpr int64_t TILINGKEY_FLOAT16 = 2;
constexpr int64_t TILINGKEY_FLOAT = 4;
constexpr int64_t TILINGKEY_MIX_BF16 = 8;
constexpr int64_t TILINGKEY_MIX_FP16 = 16;
constexpr int64_t TILINGKEY_MIX_FP32 = 24;

struct CoreParam {
    int64_t maxCoreMemery = 0;
    int64_t maxCoreNum = 0;
    int64_t usedCoreNum = 0;
    int64_t remainMemerySpace = 0;
    int64_t bufferNum = 0;
    int64_t tilingKey = 0;
};

struct InputParam {
    int64_t tokensNum = 0;
    int64_t topK = 0;
    int64_t hiddenSize = 0;
    int64_t totalLength = 0;
    int64_t numOutTokens = 0;
    int64_t tokensDtypeSize = 0;
    int64_t indicesDtypeSize = 0;
    int64_t probsDtypeSize = 0;
    bool haveProbs = false;
};

struct TilingParam {
    int64_t length = 0;
    int64_t num = 0;
    int64_t remain = 0;
};

struct MoeTokenUnpermuteParam {
    InputParam input;
    TilingParam hiddenTiling;
    TilingParam tokenTiling;
    TilingParam tokenPerCore;
    CoreParam core;
};

ge::graphStatus TilingCompute(gert::TilingContext *context, const int64_t topK);

BEGIN_TILING_DATA_DEF(MoeTokenPermuteGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, hidden_size);
TILING_DATA_FIELD_DEF(int64_t, top_k);
TILING_DATA_FIELD_DEF(int64_t, num_out_tokens);

TILING_DATA_FIELD_DEF(int64_t, hidden_splited_length);
TILING_DATA_FIELD_DEF(int64_t, hidden_splited_num);
TILING_DATA_FIELD_DEF(int64_t, hidden_splited_remain);

TILING_DATA_FIELD_DEF(int64_t, tokens_core_length);
TILING_DATA_FIELD_DEF(int64_t, tokens_core_remain);
TILING_DATA_FIELD_DEF(int64_t, tokens_splited_length);
TILING_DATA_FIELD_DEF(int64_t, tokens_splited_num);
TILING_DATA_FIELD_DEF(int64_t, tokens_splited_remain);

TILING_DATA_FIELD_DEF(int64_t, buffer_num);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeTokenPermuteGrad, MoeTokenPermuteGradTilingData)
} // namespace permutegrad
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_PERMUTE_GRAD_H
