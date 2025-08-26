/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file rope_quant_kvcache.cpp
 * \brief
 */
#include "register/op_def_registry.h"
namespace ops {
class RopeQuantKvcache : public OpDef {
 public:
  explicit RopeQuantKvcache(const char* name) : OpDef(name) {
    this->Input("qkv")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Input("cos")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Input("sin")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Input("quant_scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Input("quant_offset")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Input("k_cache")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Input("v_cache")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Input("indice")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Output("q")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Output("k")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Output("v")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Output("k_cache")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Output("v_cache")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("size_splits").AttrType(OPTIONAL).ListInt();
    this->Attr("layout").AttrType(OPTIONAL).String("BSND");
    this->Attr("kv_output").AttrType(OPTIONAL).Bool(false);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(RopeQuantKvcache);
}  // namespace ops