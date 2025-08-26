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
 * \file quant_matmul_all_reduce_add_rms_norm_tiling.h
 * \brief
 */
#ifndef _QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_
#define _QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_
#include <memory>
#include "ophost/tiling/quant_matmul_all_reduce_tiling.h"
#include "common_add_rms_norm_tiling.h"
#include "add_rms_norm_tiling.h"
#include "context_transfer.h"
#include "log/ops_log.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(QuantMatmulAllReduceAddRmsNormTilingData)
TILING_DATA_FIELD_DEF_STRUCT(QuantMatmulAllReduceTilingData, qunatMatmulAllReduceTilingData);
TILING_DATA_FIELD_DEF_STRUCT(AddRMSNormTilingData, addRMSNormTileTilingData);
TILING_DATA_FIELD_DEF_STRUCT(AddRMSNormTilingData, addRMSNormTailTilingData);
TILING_DATA_FIELD_DEF_STRUCT(AddRMSNormTilingeKeyData, addRmsNormTilingeKeyData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_0, QuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_1, QuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_0, QuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_1, QuantMatmulAllReduceAddRmsNormTilingData);

class QuantMMNTilingTransferHelper;
class QuantMatmulAllReduceAddRmsNormTiling : public TilingBaseClass {
    friend class QuantMMNTilingTransferHelper;

public:
    explicit QuantMatmulAllReduceAddRmsNormTiling(gert::TilingContext *context);
    ~QuantMatmulAllReduceAddRmsNormTiling() override = default;

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus CheckMRNInput(const MRNCtxInfo &mrnCtxInfo);

private:
    bool HasTail() const;
    MRNCtxInfo mrnCtxInfo_;
    QuantMatmulAllReduceAddRmsNormTilingData tilingData_;
    bool hasTail_;
    TilingOut tilingOutAddRmsNormTile_;
    TilingOut tilingOutAddRmsNormTail_;
    std::unique_ptr<QuantMMNTilingTransferHelper> helper_;
};

class QuantMMNTilingTransferHelper : public QuantMatmulAllReduceTiling {
public:
    QuantMMNTilingTransferHelper(QuantMatmulAllReduceAddRmsNormTiling &quantMatmulAllReduceAddRmsNormTiling,
                                 QuantMatmulAllReduceTilingData &data);
    ge::graphStatus GetShapeAttrsInfo() override;

private:
    QuantMatmulAllReduceAddRmsNormTiling &tilingProcesser_;
};
} // namespace optiling
#endif // _QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_