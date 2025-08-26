/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mat_mul_v3_l2_cache.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_L2_CACHE_H__
#define __OP_HOST_MATMUL_V3_L2_CACHE_H__

#include "mat_mul_v3_tiling.h"
#include "mat_mul_v3_common.h"

namespace optiling {
namespace matmul_v3 {
class L2Cache {
public:
    L2Cache(MatmulV3Args &args, MatmulTilingData &tilingData)
        : args_(args), tilingData_(tilingData) {
    }
    void SetL2CacheFlag(TilingEnable tilingEnable, uint64_t l2Size, uint32_t &l2CacheFlag);
private:
    void SetL2CacheFlag(bool aEnableL2Cache, bool bEnableL2Cache, bool cEnableL2Cache,
                        bool biasEnableL2Cache, uint32_t &l2CacheFlag);
    void SetL2CacheFlagMultiCoreSplitK(bool &aEnableL2Cache, bool &bEnableL2Cache) const;
    void SetL2CacheFlagSingleCoreSplitK(bool &aEnableL2Cache, bool &bEnableL2Cache) const;
    void SetL2CacheFlagBase(bool &aEnableL2Cache, bool &bEnableL2Cache) const;
private:
    MatmulV3Args &args_;
    MatmulTilingData &tilingData_;
};
}
}
#endif // __OP_HOST_MATMUL_V3_L2_CACHE_H__