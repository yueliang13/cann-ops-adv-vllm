/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score.cpp
 * \brief
 */
#include <cstdint>
#include "register/op_def_registry.h"

// namespace optiling {
// ge::graphStatus TilingFusedInferAttentionScore(gert::TilingContext* context);
// }

namespace ops {
class FusedInferAttentionScore : public OpDef {
public:
    FusedInferAttentionScore(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16, // key datatype
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT4,
                       ge::DT_INT4,    ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16, // value datatype
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT4,
                       ge::DT_INT4,    ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pse_shift")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("atten_mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BOOL,  ge::DT_BOOL,    ge::DT_UINT8,   ge::DT_UINT8, ge::DT_BOOL,
                       ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,    ge::DT_FLOAT16, ge::DT_BOOL,  ge::DT_UINT8,
                       ge::DT_UINT8,   ge::DT_BOOL,  ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_FLOAT16,
                       ge::DT_BOOL,    ge::DT_UINT8, ge::DT_UINT8,   ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,
                       ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,
                       ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_UINT8,   ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,
                       ge::DT_BOOL,    ge::DT_INT8,  ge::DT_BOOL,    ge::DT_FLOAT16, ge::DT_BOOL,  ge::DT_BOOL,
                       ge::DT_UINT8,   ge::DT_UINT8, ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,
                       ge::DT_FLOAT16, ge::DT_BOOL,  ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_BOOL,  ge::DT_BOOL,
                       ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_FLOAT16, ge::DT_BOOL,    ge::DT_UINT8, ge::DT_UINT8,
                       ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,
                       ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_UINT8,
                       ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_INT8,  ge::DT_BOOL,
                       ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_INT8,  ge::DT_INT8,
                       ge::DT_BOOL,    ge::DT_UINT8, ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,
                       ge::DT_BOOL,    ge::DT_INT8,  ge::DT_UINT8,   ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,
                       ge::DT_BOOL,    ge::DT_INT8,  ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_kv")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("dequant_scale1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT, // dequant scale1 datatype
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quant_scale1")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("dequant_scale2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT, // dequant scale2 datatype
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                       ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quant_scale2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_BF16,  ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, // quant scale2 datatype
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16,  ge::DT_BF16,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_BF16,  ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quant_offset2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_BF16,  ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, // quant offset2 datatype
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16,  ge::DT_BF16,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_BF16,  ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, // antiquant scale datatype
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, // antiquant offset datatype
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("query_padding_size")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("kv_padding_size")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, // key antiquant scale datatype
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, // key antiquant offset datatype
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value_antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, // value antiquant scale datatype
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value_antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, // value antiquant offset datatype
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_shared_prefix")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16, // key shared prefix datatype
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16,
                       ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT4,
                       ge::DT_INT4,    ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value_shared_prefix")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16, // value shared prefix datatype
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16,
                       ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT4,
                       ge::DT_INT4,    ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_shared_prefix_len")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("query_rope")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_rope")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16, // key datatype
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT4,
                       ge::DT_INT4,    ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_rope_antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, // key rope antiquant scale datatype
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_BF16,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_INT8})
            .FormatList({ge::FORMAT_ND});
        this->Output("softmax_lse").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Attr("num_heads").AttrType(REQUIRED).Int(1);
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(2147483647); // 2147483647: Maximum value of int32_t.
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(2147483647); // 2147483647: Maximum value of int32_t.
        this->Attr("input_layout").AttrType(OPTIONAL).String("BSH");
        this->Attr("num_key_value_heads").AttrType(OPTIONAL).Int(0);
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("inner_precise").AttrType(OPTIONAL).Int(1);
        this->Attr("block_size").AttrType(OPTIONAL).Int(0);
        this->Attr("antiquant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("softmax_lse_flag").AttrType(OPTIONAL).Bool(false);
        this->Attr("key_antiquant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("value_antiquant_mode").AttrType(OPTIONAL).Int(0);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", aicore_config); // use 910B
    }
};
OP_ADD(FusedInferAttentionScore, optiling::FusedInferAttentionScoreCompileInfo);
} // namespace ops