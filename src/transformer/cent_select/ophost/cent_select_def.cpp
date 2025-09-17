/*!
 * \file cent_select_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class CentSelect : public OpDef {
    public:
        explicit CentSelect(const char* name) : OpDef(name)
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
            this->Input("block_ids")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("block_table")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("seq_len")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            // this->Attr("k").AttrType(REQUIRED).Int(64);
            // this->Input("indices")
            //     .ParamType(REQUIRED)
            //     .DataType({ge::DT_INT32})
            //     .Format({ge::FORMAT_ND})
            //     .UnknownShapeFormat({ge::FORMAT_ND})
            //     .AutoContiguous(); 
            this->Output("page_position")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();    
            this->Output("page_position_length")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});  
            this->Output("max_page_position_length")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            // this->Output("indices")
            //     .ParamType(REQUIRED)
            //     .DataType({ge::DT_INT32})
            //     .Format({ge::FORMAT_ND})
            //     .UnknownShapeFormat({ge::FORMAT_ND})
            //     .AutoContiguous(); 
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
OP_ADD(CentSelect);
} // namespace ops
