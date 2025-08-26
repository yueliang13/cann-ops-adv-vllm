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
 * \file rope_quant_kvcache.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "rope_quant_kvcache.h"
using namespace AscendC;
using namespace RopeQuantKvcacheND;

extern "C" __global__ __aicore__ void rope_quant_kvcache(GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin, GM_ADDR quant_scale,
                                                         GM_ADDR quant_offset, GM_ADDR k_cache, GM_ADDR v_cache,
                                                         GM_ADDR indice, GM_ADDR q_out, GM_ADDR k_out, GM_ADDR v_out,
                                                         GM_ADDR k_cache_out, GM_ADDR v_cache_out, GM_ADDR workspace,
                                                         GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);

  RopeQuantKvcache op(&tilingData);

  op.Init(qkv, cos, sin, quant_scale, quant_offset, k_cache, v_cache, indice, q_out, k_cache_out, v_cache_out);
  op.Process();
}