/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file swin_attention_score_quant_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
    class SwinAttentionScoreQuant : public OpDef {
    public:
        explicit SwinAttentionScoreQuant(const char* name) : OpDef(name)
        {
            this->Input("query")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT8})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("key")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT8})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("value")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT8})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("scale_quant")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("scale_dequant1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_UINT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("scale_dequant2")
                .ParamType(REQUIRED)
                .DataType({ge::DT_UINT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("bias_quant")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("bias_dequant1")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("bias_dequant2")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("padding_mask1")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("padding_mask2")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Output("attention_score")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Attr("query_transpose")
                .AttrType(OPTIONAL)
                .Bool(false);
            this->Attr("key_transpose")
                .AttrType(OPTIONAL)
                .Bool(false);
            this->Attr("value_transpose")
                .AttrType(OPTIONAL)
                .Bool(false);
            this->Attr("softmax_axes")
                .AttrType(OPTIONAL)
                .Int(-1);
            this->AICore().AddConfig("ascend310p");
        }
    };
    
OP_ADD(SwinAttentionScoreQuant);
}  // namespace ops