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
 * \file moe_init_routing_v2_grad_tiling_base.cpp
 * \brief
 */

 #include "moe_init_routing_v2_grad_tiling.h"
 #include "register/op_def_registry.h"
 #include "tiling/tiling_templates_registry.h"
 #include "log/ops_log.h"
 
 namespace optiling {
 const static size_t DIM_ONE = 1;
 const static size_t DIM_TWO = 2;
 const static size_t DIM_THREE = 3;
 const static int32_t SIZE_16 = 16;
 const static int32_t LENGTH_1024 = 1024;
 
 ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::GetPlatformInfo() {
   auto platformInfo = context_->GetPlatformInfo();
   OPS_ERR_IF(platformInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_, "fail to get platform info"),
                   return ge::GRAPH_FAILED);
   auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
   aivNum = ascendcPlatform.GetCoreNumAiv();
   socVersion = ascendcPlatform.GetSocVersion();
   aicoreParams_.blockDim = aivNum;
   uint64_t ubSizePlatForm;
   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
   aicoreParams_.ubSize = ubSizePlatForm;
 
   return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::CheckDtypeValidity() {
   const std::vector<ge::DataType> DATA_TYPE_SUPPORT = {ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16,
                                                        ge::DataType::DT_BF16};
   // 获取输入dtype
   auto gradExpandedXDesc = context_->GetInputDesc(0);
    OPS_LOG_E_IF_NULL(context_, gradExpandedXDesc, return ge::GRAPH_FAILED);
   inDtype = gradExpandedXDesc->GetDataType();
   auto expandedRowIdxDesc = context_->GetInputDesc(1);
    OPS_LOG_E_IF_NULL(context_, expandedRowIdxDesc, return ge::GRAPH_FAILED);
   auto rowIdxDtype = expandedRowIdxDesc->GetDataType();
   auto gradXDesc = context_->GetOutputDesc(0);
    OPS_LOG_E_IF_NULL(context_, gradXDesc, return ge::GRAPH_FAILED);
   auto outDtype = gradXDesc->GetDataType();
   if (inDtype != outDtype) {
     OPS_LOG_E(context_->GetNodeName(), "inputdtype [%d] must be the same with outputDtype [%d].", inDtype, outDtype);
     return ge::GRAPH_FAILED;
   }
   if (rowIdxDtype != ge::DataType::DT_INT32) {
     OPS_LOG_E(context_->GetNodeName(), "gradExpandedX dtype only support [%d].", ge::DataType::DT_INT32);
     return ge::GRAPH_FAILED;
   }
   auto it = std::find(DATA_TYPE_SUPPORT.begin(), DATA_TYPE_SUPPORT.end(), inDtype);
   if (it == DATA_TYPE_SUPPORT.end()) {
     OPS_LOG_E(context_->GetNodeName(), "inputdtype [%d] must be in float32:[%d], float16:[%d], bfloat16:[%d]", inDtype,
             ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16);
     return ge::GRAPH_FAILED;
   }
   return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::CheckShapeAllPositive(const gert::Shape& shape, std::string name) {
   for (size_t i = 0; i < shape.GetDimNum(); i++) {
     OPS_ERR_IF(
         shape.GetDim(i) <= 0,
         OPS_REPORT_VECTOR_INNER_ERR(context_, "Dim %lu of %s expect be positive, but actual %ld.", i,
                                         name.c_str(), shape.GetDim(i)),
         return ge::GRAPH_FAILED);
   }
   return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::CheckShapeValidity(const gert::Shape& xShape,
                                                                         const gert::Shape& rowIdxShape,
                                                                         const gert::Shape& gradXShape) {
   if (CheckShapeAllPositive(rowIdxShape, "rowIdxShape") != ge::GRAPH_SUCCESS ||
       CheckShapeAllPositive(gradXShape, "gradXShape") != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; }
 
   if (dropPadMode == 0 && activeNum == 0) {
     if (CheckShapeAllPositive(xShape, "xShape") != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; }
   } else {
     // 索引含-1场景，xShape可以有0
     for (size_t i = 0; i < xShape.GetDimNum(); i++) {
       OPS_ERR_IF(xShape.GetDim(i) < 0,
                       OPS_REPORT_VECTOR_INNER_ERR(context_,
                                                       "Dim %lu of x expect not be negtive, but actual %ld.", i,
                                                       xShape.GetDim(i)),
                       return ge::GRAPH_FAILED);
     }
   }
 
   return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::CheckParamsValidity(const gert::Shape& xShape,
                                                                          const gert::Shape& rowIdxShape,
                                                                          const gert::Shape& gradXShape) const {
   // attr属性校验
   if (dropPadMode != 0 && dropPadMode != 1) {
     OPS_LOG_E(context_->GetNodeName(), "Attr drop_pad_mode should in range [0, 1].");
     return ge::GRAPH_FAILED;
   }
 
   if (topK <= 0) {
     OPS_LOG_E(context_->GetNodeName(), "Attr top_k must be larger than zero.");
     return ge::GRAPH_FAILED;
   }
 
   if (activeNum < 0) {
     OPS_LOG_E(context_->GetNodeName(), "Attr active_num should larger than or equal zero.");
     return ge::GRAPH_FAILED;
   }
 
   // shape校验
   size_t xDimNnum = xShape.GetDimNum();
   if (xDimNnum != DIM_TWO && xDimNnum != DIM_THREE) {
     OPS_LOG_E(context_->GetNodeName(), "The dim number of grad_expanded_x should be 2 or 3.");
     return ge::GRAPH_FAILED;
   }
 
   if (dropPadMode == 1 && xDimNnum != DIM_THREE) {
     OPS_LOG_E(context_->GetNodeName(), "GradExpandedX input shape must be 3D under Drop/Pad mode.");
     return ge::GRAPH_FAILED;
   }
 
   if (dropPadMode == 0) {
     if (activeNum == 0 && xShape.GetDim(0) != rowIdxShape.GetDim(0)) {
       OPS_LOG_E(context_->GetNodeName(), "All inputs Dim 0 size should be same under dropless mode.");
       return ge::GRAPH_FAILED;
     }
 
     if (activeNum > 0 && xShape.GetDim(0) != activeNum) {
       OPS_LOG_E(context_->GetNodeName(), "Dim 0 size of GradExpandedX should be equal active_num under active scene.");
       return ge::GRAPH_FAILED;
     }
   }
 
   size_t outDimNum = gradXShape.GetDimNum();
   if (outDimNum != DIM_TWO) {
     OPS_LOG_E(context_->GetNodeName(), "The dim number of grad_x should be 2.");
     return ge::GRAPH_FAILED;
   }
 
   int64_t hiddenSizeX = (dropPadMode == 0) ? xShape.GetDim(1) : xShape.GetDim(2);  // 2: drop/pad 场景，尾轴在第三维
   if (gradXShape.GetDim(1) != hiddenSizeX) {
     OPS_LOG_E(context_->GetNodeName(), "Tail dim size of input and output should be same.");
     return ge::GRAPH_FAILED;
   }
 
   if (gradXShape.GetDim(0) != rowIdxShape.GetDim(0) / topK) {
     OPS_LOG_E(context_->GetNodeName(),
      "output shape invalid, dim 0 of output gradX should be equal to the division of expandedRowIdx dim 0 and topK.");
     return ge::GRAPH_FAILED;
   }
 
   return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::GetShapeAttrsInfo() {
   opName = context_->GetNodeName();
 
   // 获取输入shape
   auto xShapePtr = context_->GetInputShape(0);
    OPS_LOG_E_IF_NULL(context_, xShapePtr, return ge::GRAPH_FAILED);
   const gert::Shape xShape = xShapePtr->GetStorageShape();
   auto rowIdxShapePtr = context_->GetInputShape(1);
    OPS_LOG_E_IF_NULL(context_, rowIdxShapePtr, return ge::GRAPH_FAILED);
   const gert::Shape rowIdxShape = rowIdxShapePtr->GetStorageShape();
   auto gradXShapePtr = context_->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context_, gradXShapePtr, return ge::GRAPH_FAILED);
   const gert::Shape gradXShape = gradXShapePtr->GetStorageShape();
 
   // 获取输入属性
   auto attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);
   const int64_t* topKPtr = attrs->GetAttrPointer<int64_t>(0);
    OPS_LOG_E_IF_NULL(context_, topKPtr, return ge::GRAPH_FAILED);
   topK = *topKPtr;
   const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(1);  // 1: drop_pad_mode attr
   dropPadMode = (dropPadModePtr == nullptr) ? 0 : *dropPadModePtr;
   const int64_t* activeNumPtr = attrs->GetAttrPointer<int64_t>(2);  // 2: active_num attr
   activeNum = (activeNumPtr == nullptr) ? 0 : *activeNumPtr;
 
   ge::graphStatus res = CheckDtypeValidity();
   if (res != ge::GRAPH_SUCCESS) { return res; }
 
   // 参数校验
   res = CheckParamsValidity(xShape, rowIdxShape, gradXShape);
   if (res != ge::GRAPH_SUCCESS) { return res; }
 
   // shape校验
   if (CheckShapeValidity(xShape, rowIdxShape, gradXShape) != ge::GRAPH_SUCCESS) {
     return ge::GRAPH_FAILED;
   }
 
   // 设置tilingData基本参数
   int64_t BSK = rowIdxShape.GetDim(0);  // A: B*S*K
   E = (dropPadMode == 0) ? 0 : xShape.GetDim(0);
   C = (dropPadMode == 0) ? 0 : xShape.GetDim(1);
   hiddenSize = (dropPadMode == 0) ? xShape.GetDim(1) : xShape.GetDim(2);  // 2 for H, when shape is {E, C, H}
   if (BSK % topK != 0) {
     OPS_LOG_E(context_->GetNodeName(), "expanded_row_idx shape or top_k val invalid");
     return ge::GRAPH_FAILED;
   }
   N = BSK / topK;
 
   return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::DoLibApiTiling() {
   return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::GetWorkspaceSize() {
   // 计算workspace大小，无需workspace临时空间，不存在多核同步，预留固定大小即可
   workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;
   return ge::GRAPH_SUCCESS;
 }
 
 ASCENDC_EXTERN_C ge::graphStatus TilingForMoeInitRoutingV2Grad(gert::TilingContext* context) {
   return TilingRegistry::GetInstance().DoTilingImpl(context);
 }
 
 ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForMoeInitRoutingV2Grad(gert::TilingParseContext* context) {
   return ge::GRAPH_SUCCESS;
 }
 
 IMPL_OP(MoeInitRoutingV2Grad)
     .Tiling(TilingForMoeInitRoutingV2Grad)
     .TilingParse<MoeInitRoutingV2GradCompileInfo>(TilingPrepareForMoeInitRoutingV2Grad);
 }  // namespace optiling