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
 * \file weight_quant_matmul_all_reduce_add_rms_norm_tiling.h
 * \brief
 */
#ifndef _WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_
#define _WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_
#include <memory>
#include "ophost/tiling/weight_quant_matmul_all_reduce_tiling.h"
#include "common_add_rms_norm_tiling.h"
#include "add_rms_norm_tiling.h"
#include "context_transfer.h"
#include "log/ops_log.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(WeightQuantMatmulAllReduceAddRmsNormTilingData)
TILING_DATA_FIELD_DEF_STRUCT(WeightQuantMatmulAllReduceTilingData, weightQuantMatmulAllReduceTilingData);
TILING_DATA_FIELD_DEF_STRUCT(AddRMSNormTilingData, addRMSNormTileTilingData);
TILING_DATA_FIELD_DEF_STRUCT(AddRMSNormTilingData, addRMSNormTailTilingData);
TILING_DATA_FIELD_DEF_STRUCT(AddRMSNormTilingeKeyData, addRmsNormTilingeKeyData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_310100, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_311100, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_310110, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_311110, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_310200, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_311200, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_310210, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_311210, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_310300, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_310310, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_311300, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_311310, WeightQuantMatmulAllReduceAddRmsNormTilingData);


REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_310100, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_311100, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_310110, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_311110, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_310200, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_311200, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_310210, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_311210, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_310300, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_310310, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_311300, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_311310, WeightQuantMatmulAllReduceAddRmsNormTilingData);

class WeightQuantMMNTilingTransferHelper;
class WeightQuantMatmulAllReduceAddRmsNormTiling : public TilingBaseClass {
    friend class WeightQuantMMNTilingTransferHelper;

public:
    explicit WeightQuantMatmulAllReduceAddRmsNormTiling(gert::TilingContext *context);
    ~WeightQuantMatmulAllReduceAddRmsNormTiling() override = default;

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
    WeightQuantMatmulAllReduceAddRmsNormTilingData tilingData_;
    bool hasTail_;
    TilingOut tilingOutAddRmsNormTile_;
    TilingOut tilingOutAddRmsNormTail_;
    std::unique_ptr<WeightQuantMMNTilingTransferHelper> helper_;
};

class WeightQuantMMNTilingTransferHelper : public WeightQuantMatmulAllReduceTiling {
public:
    WeightQuantMMNTilingTransferHelper(
        WeightQuantMatmulAllReduceAddRmsNormTiling &weightQuantMatmulAllReduceAddRmsNormTiling,
        WeightQuantMatmulAllReduceTilingData &data);
    ge::graphStatus GetShapeAttrsInfo() override;

private:
    WeightQuantMatmulAllReduceAddRmsNormTiling &tilingProcesser_;
};
} // namespace optiling

#endif // _WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_