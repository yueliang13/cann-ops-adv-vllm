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
 * \file matmul_all_reduce_tiling_910.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_TILING_910_H
#define MATMUL_ALL_REDUCE_TILING_910_H

#include "matmul_all_reduce_tiling.h"
#include "ophost/mat_mul_v3_base_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulAllReduce910TilingData)
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
TILING_DATA_FIELD_DEF_STRUCT(MatmulTilingData, tilematmulTiling);
TILING_DATA_FIELD_DEF_STRUCT(MatmulTilingData, tailmatmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10000000000000001100, MatmulAllReduce910TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10000000000000000009, MatmulAllReduce910TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10000000000000000000, MatmulAllReduce910TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10000000000000000001, MatmulAllReduce910TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce910TilingDataOp, MatmulAllReduce910TilingData);


class MatmulAllReduceTiling910 : public MatmulAllReduceTilingBase {
    friend class MMNTilingTransferHelper;
    friend class TilingTransferHelper;
    friend class MatmulAllReduceAddRmsNormTiling;

public:
    explicit MatmulAllReduceTiling910(gert::TilingContext *context);
    MatmulAllReduceTiling910(gert::TilingContext *context,
                             MMRCtxInfo *mmrCtxInfo,
                             MatmulAllReduce910TilingData *out);
    ~MatmulAllReduceTiling910() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus GetWorkspaceSize() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus PostTiling() override;

    ge::graphStatus Do910Tiling();

    Mc2Msg &MutableMc2MsgData() override;

    RCSTiling &MutableRCSTilingData() override;

    TCubeTiling &MutableTCubeTileTilingData() override;

    TCubeTiling &MutableTCubeTailTilingData() override;

    void DoEmptyTensorTiling() override;

    ge::graphStatus CheckInput() override;

private:
    ge::graphStatus CheckAxisSize();
    MatmulAllReduce910TilingData matmulAllReduce910TilingDataSelf_;
    MatmulAllReduce910TilingData &matmulAllReduce910TilingData_;
    uint64_t myWorkSpaceSize_{0U};
};

class TilingTransferHelper : public matmul_v3::MatmulV3BaseTiling {
public:
    TilingTransferHelper(MatmulAllReduceTiling910 &matmulAllReduceTiling910,
                         MatmulTilingData &data);

    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus PostTiling() override;

private:
    MatmulAllReduceTiling910 &tilingProcesser_;
};
}
#endif // MATMUL_ALL_REDUCE_TILING_910_H