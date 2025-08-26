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
 * \file cache_tiling.cc
 * \brief
 */
#include "cube/include/cache_tiling.h"

#include "cube/algorithm/cache_tiling_impl.h"
#include "op_log.h"

namespace optiling {
namespace cachetiling {


static FactoryInst<std::shared_ptr<CacheTilingImpl>> inst_;

bool GenTiling(const CubeTilingParam &params, CubeTiling &tiling) {
  if (!params.IsValid()) {
    OPS_LOG_E(params.op_type, "Invalid input param");
    return false;
  }

  auto impl = inst_.Get(params.type);
  if (impl == nullptr) {
    impl = CacheTilingFactory().Create(params.type, params);
    if (impl == nullptr) {
      OPS_LOG_E(params.op_type, "Creator TilingGenerator failed");
      return false;
    }
    inst_.Add(params.type, impl);
  }
  if (!impl->Init(params)) {
    OPS_LOG_E(params.op_type, "Failed to init TilingImpl!");
    return false;
  }
  bool res = impl->GenTiling(tiling);
  impl->Clear();
  if (!tiling.IsValid()) {
    OPS_LOG_E(params.op_type, "Invalid output param");
    return false;
  }
  return res;
}

void DestoryTilingFactory() {
}
}  // namespace cachetiling
}  // namespace optiling
