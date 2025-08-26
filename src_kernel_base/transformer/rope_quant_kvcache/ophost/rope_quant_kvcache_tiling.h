/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file rope_quant_kvcache_tiling.h
 * \brief
 */
#ifndef ROPE_QUANT_KVCACHE_TILING_H
#define ROPE_QUANT_KVCACHE_TILING_H
#include <cmath>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(RopeQuantKvcacheTilingData)
TILING_DATA_FIELD_DEF(uint64_t, qHeadNum);
TILING_DATA_FIELD_DEF(uint64_t, kvHeadNum);
TILING_DATA_FIELD_DEF(uint64_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint64_t, cacheSeqlen);
TILING_DATA_FIELD_DEF(uint64_t, qHiddenSize);
TILING_DATA_FIELD_DEF(uint64_t, kHiddenSize);
TILING_DATA_FIELD_DEF(uint64_t, vHiddenSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RopeQuantKvcache, RopeQuantKvcacheTilingData)
struct RopeQuantKvcacheCompileInfo {};

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGE(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
#define OP_TILING_CHECK(cond, log_func, expr)  \
  do {                                         \
    if (cond) {                                \
      std::printf(log_func);                     \
      expr;                                      \
    }                                          \
  } while (0)

}  // namespace optiling
#endif