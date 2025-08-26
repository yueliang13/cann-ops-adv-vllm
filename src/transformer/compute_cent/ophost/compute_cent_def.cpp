/*!
 * \file compute_cent_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class ComputeCent : public OpDef {
    public:
        explicit ComputeCent(const char* name) : OpDef(name)
        {
            this->Input("query")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("l1_cent")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("key_ids")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            // 删除
            this->Output("d_l1_cent")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})  
                .UnknownShapeFormat({ge::FORMAT_ND});
            // 删除
            this->Input("mask_empty")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            // 删除
            this->Output("select_nprobe")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            // 返回
            this->Output("indices")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});  
            // 返回
            this->Output("token_position")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});  
            OpAICoreConfig aicore_config;
            aicore_config.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(true)
                .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
            this->AICore().AddConfig("ascend910b", aicore_config);
        }
    };
OP_ADD(ComputeCent);
} // namespace ops
