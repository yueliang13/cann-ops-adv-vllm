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
 * \file sparse_paged_attention_tiling_mla.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_MLA_NEW_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_MLA_NEW_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/data_copy_transpose_tiling.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐

BEGIN_TILING_DATA_DEF(SparsePagedAttentionBaseParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup) // G
TILING_DATA_FIELD_DEF(uint32_t, actualLenDims)
TILING_DATA_FIELD_DEF(uint32_t, antiquantMode)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag) // 根据指针判， 待删除
TILING_DATA_FIELD_DEF(uint32_t, attenMaskSize) //  待删除
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparsePagedAttentionBaseParamsMlaOp, SparsePagedAttentionBaseParamsMla)

BEGIN_TILING_DATA_DEF(SparsePagedAttentionCoreParamsMla) // 分核相关 26 保证数组8字节对齐
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIC_CORE_NUM, coreSidxEnd);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparsePagedAttentionCoreParamsMlaOp, SparsePagedAttentionCoreParamsMla)

BEGIN_TILING_DATA_DEF(SparsePagedAttentionSingleCoreParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSize);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, groupSplitSize); // G 切分
TILING_DATA_FIELD_DEF(uint32_t, s1SplitSize);    // S 切分
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparsePagedAttentionSingleCoreParamsMlaOp, SparsePagedAttentionSingleCoreParamsMla)

BEGIN_TILING_DATA_DEF(SparsePagedAttentionSingleCoreTensorSizeMla)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparsePagedAttentionSingleCoreTensorSizeMlaOp, SparsePagedAttentionSingleCoreTensorSizeMla)

BEGIN_TILING_DATA_DEF(SparsePagedAttentionSplitKVParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, s2)             // S2切分份数
TILING_DATA_FIELD_DEF(uint32_t, sInnerLoopSize) 
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize)   // FD workspace
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize)  // FD workspace
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparsePagedAttentionSplitKVParamsMlaOp, SparsePagedAttentionSplitKVParamsMla)

BEGIN_TILING_DATA_DEF(SparsePagedAttentionTilingDataMla)
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedAttentionBaseParamsMla, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedAttentionSplitKVParamsMla, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedAttentionCoreParamsMla, sparsePagedAttentionCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedAttentionSingleCoreParamsMla, sparsePagedAttentionSingleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SparsePagedAttentionSingleCoreTensorSizeMla, sparsePagedAttentionSingleCoreTensorSize);
END_TILING_DATA_DEF

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_MLA_NEW_H_
