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
 * \file moe_finalize_routing_v2_grad.cpp
 * \brief
 */
#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class MoeFinalizeRoutingV2Grad : public OpDef {
public:
    explicit MoeFinalizeRoutingV2Grad(const char *name) : OpDef(name)
    {
        this->Input("grad_y")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .AutoContiguous();
        this->Input("expanded_row_idx")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32, ge::DT_INT32, ge::DT_INT32 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .AutoContiguous();
        this->Input("expanded_x")
            .ParamType(OPTIONAL)
            .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .AutoContiguous();
        this->Input("scales")
            .ParamType(OPTIONAL)
            .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .AutoContiguous();
        this->Input("expert_idx")
            .ParamType(OPTIONAL)
            .DataType({ ge::DT_INT32, ge::DT_INT32, ge::DT_INT32 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .AutoContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .AutoContiguous();
        this->Output("grad_expanded_x")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Output("grad_scales")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Attr("drop_pad_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("active_num").AttrType(OPTIONAL).Int(0);
        this->Attr("expert_num").AttrType(OPTIONAL).Int(0);
        this->Attr("expert_capacity").AttrType(OPTIONAL).Int(0);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(MoeFinalizeRoutingV2Grad);
}