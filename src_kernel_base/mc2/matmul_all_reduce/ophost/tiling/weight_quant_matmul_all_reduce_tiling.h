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
 * \file weight_quant_matmul_all_reduce_tiling.h
 * \brief
 */
#ifndef WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_H
#define WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_H
#include "matmul_all_reduce_tiling.h"
#include "ophost/weight_quant_batch_matmul_v2_tiling_custom.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(WeightQuantMatmulAllReduceTilingData)
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
TILING_DATA_FIELD_DEF_STRUCT(WeightQuantBatchMatmulV2TilingData, tilematmulTiling);
TILING_DATA_FIELD_DEF_STRUCT(WeightQuantBatchMatmulV2TilingData, tailmatmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_310100, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_311100, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_310110, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_311110, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_310200, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_311200, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_310210, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_311210, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_310300, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_310310, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_311300, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_311310, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_810200, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_811200, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_810210, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_811210, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10000000000000000008, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(WeightQuantMatmulAllReduceTilingDataOp,
                           WeightQuantMatmulAllReduceTilingData);

constexpr int64_t ANTIQUANT_GROUP_SIZE_MIN_VALUE = 32;

class WeightQuantMatmulAllReduceTiling : public MatmulAllReduceTilingBase {
    friend class WeightQuantTilingTransferHelper;
    friend class WeightQuantMatmulAllReduceAddRmsNormTiling;
public:
    explicit WeightQuantMatmulAllReduceTiling(gert::TilingContext *context);
    WeightQuantMatmulAllReduceTiling(gert::TilingContext *context,
                                     MMRCtxInfo *mmrCtxInfo, WeightQuantMatmulAllReduceTilingData *out);
    ~WeightQuantMatmulAllReduceTiling() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

    Mc2Msg &MutableMc2MsgData() override
    {
        return weightQuantMatmulAllReduceTilingData_.msg;
    }

    RCSTiling &MutableRCSTilingData() override
    {
        return weightQuantMatmulAllReduceTilingData_.param;
    }

    TCubeTiling &MutableTCubeTileTilingData() override
    {
        return weightQuantMatmulAllReduceTilingData_.tilematmulTiling.matmulTiling;
    }

    TCubeTiling &MutableTCubeTailTilingData() override
    {
        return weightQuantMatmulAllReduceTilingData_.tailmatmulTiling.matmulTiling;
    }

    ge::graphStatus DoWeightQuantTiling();

    void DoEmptyTensorTiling() override;

    ge::graphStatus CheckInput() override;

private:
    ge::graphStatus CheckAxisSize();
    WeightQuantMatmulAllReduceTilingData weightQuantMatmulAllReduceTilingDataSelf_;
    WeightQuantMatmulAllReduceTilingData &weightQuantMatmulAllReduceTilingData_;
    uint64_t myWorkSpaceSize_{0U};
};

class WeightQuantTilingTransferHelper : public WeightQuantBatchMatmulV2TilingCustom {
public:
    WeightQuantTilingTransferHelper(WeightQuantMatmulAllReduceTiling &weightQuantMatmulAllReduceTiling,
                                    WeightQuantBatchMatmulV2TilingData &data)
        : WeightQuantBatchMatmulV2TilingCustom(weightQuantMatmulAllReduceTiling.context_, &data),
        tilingProcesser_(weightQuantMatmulAllReduceTiling)
    {
    }
    ge::graphStatus GetShapeAttrsInfo() override;
    void PrintTilingInputParam(WeightQuantBatchMatmulInfo &weightQuantBatchMatmulInfo);
    ge::graphStatus PostTiling() override;

private:
    WeightQuantMatmulAllReduceTiling &tilingProcesser_;
};
}
#endif // WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_H
