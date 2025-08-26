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
 * \file apply_rotary_pos_emb_def.cpp
 * \brief
 */
#include <vector>
#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> qDtype = {
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16
};
static const std::vector<ge::Format> qFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};

class ApplyRotaryPosEmb : public OpDef {
public:
    explicit ApplyRotaryPosEmb(const char* name) : OpDef(name) {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType(qDtype)
            .Format(qFormat)
            .UnknownShapeFormat(qFormat)
	    .AutoContiguous();
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType(qDtype)
            .Format(qFormat)
            .UnknownShapeFormat(qFormat)
	    .AutoContiguous();
        this->Input("cos")
            .ParamType(REQUIRED)
            .DataType(qDtype)
            .Format(qFormat)
            .UnknownShapeFormat(qFormat)
	    .AutoContiguous();
        this->Input("sin")
            .ParamType(REQUIRED)
            .DataType(qDtype)
            .Format(qFormat)
            .UnknownShapeFormat(qFormat)
	    .AutoContiguous();
        this->Output("query")
            .ParamType(REQUIRED)
            .DataType(qDtype)
            .Format(qFormat)
            .UnknownShapeFormat(qFormat);
        this->Output("key")
            .ParamType(REQUIRED)
            .DataType(qDtype)
            .Format(qFormat)
            .UnknownShapeFormat(qFormat);
        this->Attr("layout")
            .AttrType(OPTIONAL)
            .Int(1);
        OpAICoreConfig configApply;
        configApply.DynamicCompileStaticFlag(true)
                   .DynamicFormatFlag(true)
                   .DynamicRankSupportFlag(true)
                   .DynamicShapeSupportFlag(true)
                   .NeedCheckSupportFlag(false);
        this->AICore().AddConfig("ascend910b", configApply);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false);
        aicore_config.Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Input("cos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Input("sin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Output("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Output("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().AddConfig("ascend310p", aicore_config);
        this->AICore().AddConfig("ascend910", aicore_config);
    }
};

OP_ADD(ApplyRotaryPosEmb);
}  // namespace ops
