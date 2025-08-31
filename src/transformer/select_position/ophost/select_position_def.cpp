/*!
 * \file select_position_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class SelectPosition : public OpDef {
    public:
        explicit SelectPosition(const char* name) : OpDef(name)
        {
            this->Input("block_ids")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("block_table")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("seq_len")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("indices")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND}); 
            this->Output("token_position")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});    
            this->Output("token_position_length")
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
OP_ADD(SelectPosition);
} // namespace ops
