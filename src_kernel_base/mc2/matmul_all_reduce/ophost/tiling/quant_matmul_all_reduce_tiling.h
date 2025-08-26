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
 * \file quant_matmul_all_reduce_tiling.h
 * \brief
 */
#ifndef QUANT_MATMUL_ALL_REDUCE_TILING_H
#define QUANT_MATMUL_ALL_REDUCE_TILING_H

#include "tiling/matmul_all_reduce_tiling.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(QuantMatmulAllReduceTilingData)
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
TILING_DATA_FIELD_DEF_STRUCT(QuantBatchMatmulV3TilingData, tilematmulTiling);
TILING_DATA_FIELD_DEF_STRUCT(QuantBatchMatmulV3TilingData, tailmatmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_0, QuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10, QuantMatmulAllReduceTilingData); // 低 bit 通信 bf16
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_1, QuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_11, QuantMatmulAllReduceTilingData); // 低 bit 通信 bf16
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10000, QuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10010, QuantMatmulAllReduceTilingData); // 低 bit 通信 fp16
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10001, QuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10011, QuantMatmulAllReduceTilingData); // 低 bit 通信 fp16
REGISTER_TILING_DATA_CLASS(QuantMatmulAllReduceTilingDataOp,
                           QuantMatmulAllReduceTilingData);

class QuantMatmulAllReduceTiling : public MatmulAllReduceTilingBase {
    friend class QuantTilingTransferHelper;
    friend class QuantMatmulAllReduceAddRmsNormTiling;
public:
    explicit QuantMatmulAllReduceTiling(gert::TilingContext *context);
    QuantMatmulAllReduceTiling(gert::TilingContext *context,
                               MMRCtxInfo *mmrCtxInfo,
                               QuantMatmulAllReduceTilingData *out);
    ~QuantMatmulAllReduceTiling() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

    Mc2Msg &MutableMc2MsgData() override;

    RCSTiling &MutableRCSTilingData() override;

    TCubeTiling &MutableTCubeTileTilingData() override;

    TCubeTiling &MutableTCubeTailTilingData() override;

    ge::graphStatus DoQuantTiling();

    ge::graphStatus CheckInput() override;

    ge::graphStatus CheckDequantScaleType();

private:
    ge::graphStatus CheckAxisSize();
    QuantMatmulAllReduceTilingData quantMatmulAllReduceTilingDataSelf_;
    QuantMatmulAllReduceTilingData &quantMatmulAllReduceTilingData_;
    uint64_t myWorkSpaceSize_{0U};
    bool isCommInt8Enable_ = false;
};

class QuantTilingTransferHelper : public QuantBatchMatmulV3Tiling {
public:
    QuantTilingTransferHelper(QuantMatmulAllReduceTiling &quantMatmulAllReduceTiling,
                              QuantBatchMatmulV3TilingData &data);
    const gert::Shape GetX1Shape(const size_t index) override;
    const gert::Shape GetX2Shape(const size_t index) override;
    const gert::Shape &GetScaleShape(const size_t index) override;
    const gert::StorageShape *GetPertokenShape(const size_t index) override;
    const gert::StorageShape *GetBiasShape(const size_t index) override;
    ge::graphStatus GetShapeAttrsInfo() override;
    void PrintTilingInputParam(QuantBatchMatmulInfo quantBatchMatmulInfo);
    ge::graphStatus PostTiling() override;

private:
    QuantMatmulAllReduceTiling &tilingProcesser_;
};
}
#endif //QUANT_MATMUL_ALL_REDUCE_TILING_H