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
 * \file matmul_all_reduce.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class MatmulAllReduce : public OpDef {
 public:
  explicit MatmulAllReduce(const char *name) : OpDef(name) {
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
                   ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
                   ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                   ge::DT_INT4, ge::DT_INT4, ge::DT_INT8, ge::DT_INT8,
                   ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                   ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                             ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
        .IgnoreContiguous();
    this->Input("bias")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT32,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32,
                   ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("x3")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                   ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("antiquant_scale")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                   ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .IgnoreContiguous();
    this->Input("antiquant_offset")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                   ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .IgnoreContiguous();
    this->Input("dequant_scale")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_UINT64,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT64, ge::DT_FLOAT,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_UINT64,
                   ge::DT_BF16, ge::DT_INT64, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("pertoken_scale")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("comm_quant_scale_1")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                   ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("comm_quant_scale_2")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                   ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                   ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                   ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

    this->Attr("group").AttrType(REQUIRED).String();
    this->Attr("reduce_op").AttrType(OPTIONAL).String("sum");
    this->Attr("is_trans_a").AttrType(OPTIONAL).Bool(false);
    this->Attr("is_trans_b").AttrType(OPTIONAL).Bool(false);
    this->Attr("comm_turn").AttrType(OPTIONAL).Int(0);
    this->Attr("antiquant_group_size").AttrType(OPTIONAL).Int(0);

    OpAICoreConfig aicore_config_910b;
    aicore_config_910b.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("jitCompile.flag", "static_false")
        .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
    this->AICore().AddConfig("ascend910b", aicore_config_910b);
    this->MC2().HcclGroup("group");
  }
};

OP_ADD(MatmulAllReduce);
}  // namespace ops
