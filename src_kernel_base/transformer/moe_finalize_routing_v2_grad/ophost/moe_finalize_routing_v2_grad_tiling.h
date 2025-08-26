/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file moe_finalize_routing_v2_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_FINALIZE_ROUTING_V2_GRAD_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_FINALIZE_ROUTING_V2_GRAD_TILING_H

#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_base.h"
#include "register/op_def_registry.h"
#include "log/ops_log.h"
#include "error/ops_error.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeFinalizeRoutingV2GradTilingData)
TILING_DATA_FIELD_DEF(int64_t, initOutNeedCoreNum);      // 初始化输出过程中需要的核数
TILING_DATA_FIELD_DEF(int64_t, initOutEachCoreBatchNum); // 初始化输出过程中每个核需要处理的行数
TILING_DATA_FIELD_DEF(int64_t, initOutModCoreNum);       // 初始化输出过程中需要多处理1行的核数
TILING_DATA_FIELD_DEF(int64_t, computeNeedCoreNum);      // 运算过程中需要的核数
TILING_DATA_FIELD_DEF(int64_t, computeEachCoreBatchNum); // 运算过程中每个核需要处理的行数
TILING_DATA_FIELD_DEF(int64_t, computeModCoreNum);       // 运算过程中需要多处理1行的核数
TILING_DATA_FIELD_DEF(int64_t, dropPadMode);             // drop_pad_mode
TILING_DATA_FIELD_DEF(int64_t, topK);                    // k的大小 单位元素个数
TILING_DATA_FIELD_DEF(int64_t, hidden);                  // H的大小 单位元素个数
TILING_DATA_FIELD_DEF(int64_t, expandedXDim0);           // grad_expanded_x除最后一维外，其余维度合成一维后的大小
TILING_DATA_FIELD_DEF(int64_t, hiddenPrePart);           // hidden切分后前块元素个数 单位元素个数
TILING_DATA_FIELD_DEF(int64_t, hiddenInnerLoops);        // hidden切分后前块循环次数
TILING_DATA_FIELD_DEF(int64_t, hiddenLastPart);          // hidden切分后尾块元素个数 单位元素个数
TILING_DATA_FIELD_DEF(int64_t, tilingKey);               // 模板
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeFinalizeRoutingV2Grad, MoeFinalizeRoutingV2GradTilingData)

struct MoeFinalizeRoutingV2GradCompileInfo {
    int64_t totalCoreNum = 0;
    int64_t totalUbSize = 0;
};

class MoeFinalizeRoutingV2GradTiling {
public:
    explicit MoeFinalizeRoutingV2GradTiling(gert::TilingContext *context)
        : context_(context), nodeName_(context->GetNodeName()){};

    ge::graphStatus CalcTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetRequiredTensorInfo();
    ge::graphStatus GetOptionalTensorInfo();
    ge::graphStatus GetAttrInfo();
    ge::graphStatus Init();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckRequiredInput();
    ge::graphStatus CheckOptionalInputShape();
    ge::graphStatus CheckOptionalInputDtype();
    ge::graphStatus CheckOutput();
    ge::graphStatus CheckParams();
    ge::graphStatus CalcTilingKey();
    ge::graphStatus FillTilingData();
    void CalcBaseInfo();
    void CalcTilingKeyWithScales();

private:
    gert::TilingContext *context_ = nullptr;
    std::string nodeName_ = "MoeFinalizeRoutingV2Grad";
    MoeFinalizeRoutingV2GradTilingData tilingData_;
    int64_t totalCoreNum_ = 0;
    int64_t totalUbSize_ = 0;

    // shape
    gert::Shape gradYShape_;
    gert::Shape expandedRowIdxShape_;
    gert::Shape expandedXShape_;
    gert::Shape scalesShape_;
    gert::Shape expertIdxShape_;
    gert::Shape biasShape_;
    gert::Shape gradExpandedXShape_;
    gert::Shape gradScalesShape_;

    // dtype
    ge::DataType gradYType_;
    ge::DataType expandedRowIdxType_;
    ge::DataType expandedXType_;
    ge::DataType scalesType_;
    ge::DataType expertIdxType_;
    ge::DataType biasType_;
    ge::DataType gradExpandedXType_;
    ge::DataType gradScalesType_;
    int64_t gradYTypeByteSize_ = 0;

    // attr
    int64_t activeNum_ = 0;
    int64_t expertNum_ = 0;
    int64_t expertCapacity_ = 0;

    // expanded_x
    int64_t expandedXDimNum_ = 0;

    // scales
    bool isScalesExist_ = false;

    // bias
    bool isBiasExist_ = false;

    // tiling data
    int64_t initOutNeedCoreNum_ = 0;
    int64_t initOutEachCoreBatchNum_ = 0;
    int64_t initOutModCoreNum_ = 0;
    int64_t computeNeedCoreNum_ = 0;
    int64_t computeEachCoreBatchNum_ = 0;
    int64_t computeModCoreNum_ = 0;
    int64_t dropPadMode_ = 0;
    int64_t topK_ = 0;
    int64_t hidden_ = 0;
    int64_t expandedXDim0_ = 0;
    int64_t hiddenPrePart_ = 0;
    int64_t hiddenInnerLoops_ = 0;
    int64_t hiddenLastPart_ = 0;
    int64_t tilingKey_ = 0;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_FINALIZE_ROUTING_V2_GRAD_TILING_H
