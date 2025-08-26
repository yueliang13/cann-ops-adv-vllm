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
 * \file moe_finalize_routing_v2_grad_tiling.cc
 * \brief
 */
#include "moe_finalize_routing_v2_grad_tiling.h"

namespace optiling {
constexpr int64_t BYTE_BLOCK = 32;
constexpr size_t WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_THREE = 3;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t NUM_SEVEN = 7;
constexpr int64_t NUM_EIGHT = 8;
constexpr int64_t INPUT_0_IDX = 0;
constexpr int64_t INPUT_1_IDX = 1;
constexpr int64_t INPUT_2_IDX = 2;
constexpr int64_t INPUT_3_IDX = 3;
constexpr int64_t INPUT_4_IDX = 4;
constexpr int64_t INPUT_5_IDX = 5;
constexpr int64_t OUTPUT_0_IDX = 0;
constexpr int64_t OUTPUT_1_IDX = 1;
constexpr int64_t ATTR_0_IDX = 0;
constexpr int64_t ATTR_1_IDX = 1;
constexpr int64_t ATTR_2_IDX = 2;
constexpr int64_t ATTR_3_IDX = 3;
constexpr int64_t TILING_KEY_WITHOUT_SCALE_NOT_CUT_H = 10001;
constexpr int64_t TILING_KEY_WITHOUT_SCALE_CUT_H = 10002;
constexpr int64_t TILING_KEY_WITH_SCALE_NOT_CUT_H = 20001;
constexpr int64_t TILING_KEY_WITH_SCALE_CUT_H = 20002;

const gert::Shape g_vec_1_shape = {1};
inline const gert::Shape &ConfirmNotScalar(const gert::Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context_, platformInfo, return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    totalCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_CHECK((totalCoreNum_ <= 0), OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "get aiv core num failed."),
        return ge::GRAPH_FAILED);

    uint64_t totalUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, totalUbSize);
    totalUbSize_ = totalUbSize;
    OPS_CHECK((totalUbSize_ <= 0), OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "get ub size failed."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::GetRequiredTensorInfo()
{
    auto gradYShapePtr = context_->GetInputShape(INPUT_0_IDX);
    OPS_LOG_E_IF_NULL(context_, gradYShapePtr, return ge::GRAPH_FAILED);
    gradYShape_ = ConfirmNotScalar(gradYShapePtr->GetOriginShape());
    auto gradYDescPtr = context_->GetInputDesc(INPUT_0_IDX);
    OPS_LOG_E_IF_NULL(context_, gradYDescPtr, return ge::GRAPH_FAILED);
    gradYType_ = gradYDescPtr->GetDataType();

    auto expandedRowIdxShapePtr = context_->GetInputShape(INPUT_1_IDX);
    OPS_LOG_E_IF_NULL(context_, expandedRowIdxShapePtr, return ge::GRAPH_FAILED);
    expandedRowIdxShape_ = ConfirmNotScalar(expandedRowIdxShapePtr->GetOriginShape());
    auto expandedRowIdxDescPtr = context_->GetInputDesc(INPUT_1_IDX);
    OPS_LOG_E_IF_NULL(context_, expandedRowIdxDescPtr, return ge::GRAPH_FAILED);
    expandedRowIdxType_ = expandedRowIdxDescPtr->GetDataType();

    auto gradExpandedXShapePtr = context_->GetOutputShape(OUTPUT_0_IDX);
    OPS_LOG_E_IF_NULL(context_, gradExpandedXShapePtr, return ge::GRAPH_FAILED);
    gradExpandedXShape_ = ConfirmNotScalar(gradExpandedXShapePtr->GetOriginShape());
    auto gradExpandedXDescPtr = context_->GetOutputDesc(OUTPUT_0_IDX);
    OPS_LOG_E_IF_NULL(context_, gradExpandedXDescPtr, return ge::GRAPH_FAILED);
    gradExpandedXType_ = gradExpandedXDescPtr->GetDataType();

    auto gradScalesShapePtr = context_->GetOutputShape(OUTPUT_1_IDX);
    OPS_LOG_E_IF_NULL(context_, gradScalesShapePtr, return ge::GRAPH_FAILED);
    gradScalesShape_ = ConfirmNotScalar(gradScalesShapePtr->GetOriginShape());
    auto gradScalesDescPtr = context_->GetOutputDesc(OUTPUT_1_IDX);
    OPS_LOG_E_IF_NULL(context_, gradScalesDescPtr, return ge::GRAPH_FAILED);
    gradScalesType_ = gradScalesDescPtr->GetDataType();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::GetOptionalTensorInfo()
{
    auto scalesInputShape = context_->GetOptionalInputShape(INPUT_3_IDX);
    if (scalesInputShape == nullptr) {
        isScalesExist_ = false;
    } else {
        isScalesExist_ = true;
        scalesShape_ = ConfirmNotScalar(scalesInputShape->GetOriginShape());
        auto scalesDesc = context_->GetOptionalInputDesc(INPUT_3_IDX);
        OPS_LOG_E_IF_NULL(context_, scalesDesc, return ge::GRAPH_FAILED);
        scalesType_ = scalesDesc->GetDataType();

        auto expandedXInputShape = context_->GetOptionalInputShape(INPUT_2_IDX);
        OPS_LOG_E_IF_NULL(context_, expandedXInputShape, return ge::GRAPH_FAILED);
        expandedXShape_ = ConfirmNotScalar(expandedXInputShape->GetOriginShape());
        auto expandedXDesc = context_->GetOptionalInputDesc(INPUT_2_IDX);
        OPS_LOG_E_IF_NULL(context_, expandedXDesc, return ge::GRAPH_FAILED);
        expandedXType_ = expandedXDesc->GetDataType();

        auto biasInputShape = context_->GetOptionalInputShape(INPUT_5_IDX);
        if (biasInputShape == nullptr) {
            isBiasExist_ = false;
        } else {
            isBiasExist_ = true;
            biasShape_ = ConfirmNotScalar(biasInputShape->GetOriginShape());
            auto biasDesc = context_->GetOptionalInputDesc(INPUT_5_IDX);
            OPS_LOG_E_IF_NULL(context_, biasDesc, return ge::GRAPH_FAILED);
            biasType_ = biasDesc->GetDataType();

            auto expertIdxInputShape = context_->GetOptionalInputShape(INPUT_4_IDX);
            OPS_LOG_E_IF_NULL(context_, expertIdxInputShape, return ge::GRAPH_FAILED);
            expertIdxShape_ = ConfirmNotScalar(expertIdxInputShape->GetOriginShape());
            auto expertIdxDesc = context_->GetOptionalInputDesc(INPUT_4_IDX);
            OPS_LOG_E_IF_NULL(context_, expertIdxDesc, return ge::GRAPH_FAILED);
            expertIdxType_ = expertIdxDesc->GetDataType();
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::GetAttrInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);

    if (attrs->GetAttrNum() > ATTR_0_IDX) {
        dropPadMode_ = *(attrs->GetAttrPointer<int64_t>(ATTR_0_IDX));
    }
    if (attrs->GetAttrNum() > ATTR_1_IDX) {
        activeNum_ = *(attrs->GetAttrPointer<int64_t>(ATTR_1_IDX));
    }

    if (dropPadMode_ == 1) {
        OPS_CHECK((attrs->GetAttrNum() <= ATTR_3_IDX),
            OPS_LOG_E(nodeName_, "if drop_pad_mod is 1, expert_num and expert_capacity is required."),
            return ge::GRAPH_FAILED);
        expertNum_ = *(attrs->GetAttrPointer<int64_t>(ATTR_2_IDX));
        expertCapacity_ = *(attrs->GetAttrPointer<int64_t>(ATTR_3_IDX));
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::Init()
{
    OPS_CHECK((GetPlatformInfo() != ge::GRAPH_SUCCESS),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "GetPlatformInfo failed."), return ge::GRAPH_FAILED);

    OPS_CHECK((GetRequiredTensorInfo() != ge::GRAPH_SUCCESS),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "GetRequiredTensorInfo failed."), return ge::GRAPH_FAILED);

    OPS_CHECK((GetOptionalTensorInfo() != ge::GRAPH_SUCCESS),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "GetOptionalTensorInfo failed."), return ge::GRAPH_FAILED);

    OPS_CHECK((GetAttrInfo() != ge::GRAPH_SUCCESS),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "GetAttrInfo failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CheckAttr()
{
    OPS_CHECK(((dropPadMode_ != 0) && (dropPadMode_ != 1)),
        OPS_LOG_E(nodeName_, "drop_pad_mode must be 1 or 2, but got %ld.", dropPadMode_), return ge::GRAPH_FAILED);

    if (dropPadMode_ == 1) {
        OPS_CHECK(((expertNum_ <= 0) || (expertCapacity_ <= 0)),
            OPS_LOG_E(nodeName_, "if drop_pad_mod is 1, expert_num and expert_capacity must be greater than 0."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CheckRequiredInput()
{
    OPS_CHECK((gradYShape_.GetDimNum() != NUM_TWO),
        OPS_LOG_E(nodeName_, "grad_y dimnum must be 2, but got %zu.", gradYShape_.GetDimNum()), return ge::GRAPH_FAILED);
    OPS_CHECK((expandedRowIdxShape_.GetDimNum() != 1),
        OPS_LOG_E(nodeName_, "expanded_row_idx dimnum must be 1, but got %zu.", expandedRowIdxShape_.GetDimNum()),
        return ge::GRAPH_FAILED);

    OPS_CHECK(((gradYType_ != ge::DT_FLOAT) && (gradYType_ != ge::DT_BF16) && (gradYType_ != ge::DT_FLOAT16)),
        OPS_LOG_E(nodeName_, "grad_y dtype must be FLOAT or FLOAT16 or BFLOAT16."), return ge::GRAPH_FAILED);
    OPS_CHECK((expandedRowIdxType_ != ge::DT_INT32),
        OPS_LOG_E(nodeName_, "expanded_row_idx dtype must be DT_INT32."), return ge::GRAPH_FAILED);

    gradYTypeByteSize_ = ge::GetSizeByDataType(gradYType_);
    OPS_CHECK((gradYTypeByteSize_ <= 0),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "grad_y dtype byte size must be greater than 0."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CheckOptionalInputShape()
{
    if (expandedXDimNum_ == NUM_TWO) {
        OPS_CHECK((expandedXShape_.GetDim(0) != expandedXDim0_),
            OPS_LOG_E(nodeName_, "expanded_x dim0 must be equal to %ld, but got %ld.", expandedXDim0_,
            expandedXShape_.GetDim(0)),
            return ge::GRAPH_FAILED);
    } else {
        OPS_CHECK((expandedXShape_.GetDim(0) * expandedXShape_.GetDim(1) != expandedXDim0_),
            OPS_LOG_E(nodeName_, "expanded_x dim0 * dim1 must be equal to %ld, but got %ld.", expandedXDim0_,
            expandedXShape_.GetDim(0) * expandedXShape_.GetDim(1)),
            return ge::GRAPH_FAILED);
    }
    OPS_CHECK((expandedXShape_.GetDim(expandedXDimNum_ - 1) != gradYShape_.GetDim(1)),
        OPS_LOG_E(nodeName_, "expanded_x last dim and grad_y dim1 must be same, but got %ld, %ld.",
        expandedXShape_.GetDim(expandedXDimNum_ - 1), gradYShape_.GetDim(1)),
        return ge::GRAPH_FAILED);

    OPS_CHECK((scalesShape_.GetDim(0) != gradYShape_.GetDim(0)),
        OPS_LOG_E(nodeName_, "scales and grad_y dim0 must be same, but got %ld, %ld.", scalesShape_.GetDim(0),
        gradYShape_.GetDim(0)),
        return ge::GRAPH_FAILED);
    OPS_CHECK((scalesShape_.GetShapeSize() != expandedRowIdxShape_.GetShapeSize()),
        OPS_LOG_E(nodeName_, "scales and expanded_row_idx shape size must be same, but got %ld, %ld.",
        scalesShape_.GetShapeSize(), expandedRowIdxShape_.GetShapeSize()),
        return ge::GRAPH_FAILED);

    if (isBiasExist_) {
        OPS_CHECK((expertIdxShape_.GetDim(0) != gradYShape_.GetDim(0)),
            OPS_LOG_E(nodeName_, "expert_idx and grad_y dim0 must be same, but got %ld, %ld.", expertIdxShape_.GetDim(0),
            gradYShape_.GetDim(0)),
            return ge::GRAPH_FAILED);
        OPS_CHECK((expertIdxShape_.GetShapeSize() != expandedRowIdxShape_.GetShapeSize()),
            OPS_LOG_E(nodeName_, "expert_idx and expanded_row_idx shape size must be same, but got %ld, %ld.",
            expertIdxShape_.GetShapeSize(), expandedRowIdxShape_.GetShapeSize()),
            return ge::GRAPH_FAILED);

        OPS_CHECK((biasShape_.GetDim(0) <= 0),
            OPS_LOG_E(nodeName_, "bias dim0 must be greater than 0, but got %ld.", biasShape_.GetDim(0)),
            return ge::GRAPH_FAILED);
        if (dropPadMode_ == 1) {
            OPS_CHECK((biasShape_.GetDim(0) != expertNum_),
                OPS_LOG_E(nodeName_, "bias dim0 must be equal to %ld, but got %ld.", expertNum_, biasShape_.GetDim(0)),
                return ge::GRAPH_FAILED);
        }
        OPS_CHECK((biasShape_.GetDim(1) != gradYShape_.GetDim(1)),
            OPS_LOG_E(nodeName_, "grad_y and bias dim1 must be same, but got %ld, %ld.", gradYShape_.GetDim(1),
            biasShape_.GetDim(1)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CheckOptionalInputDtype()
{
    OPS_CHECK((expandedXType_ != gradYType_), OPS_LOG_E(nodeName_, "expanded_x and grad_y dtype must be same."),
        return ge::GRAPH_FAILED);
    OPS_CHECK((scalesType_ != gradYType_), OPS_LOG_E(nodeName_, "scales and grad_y dtype must be same."),
        return ge::GRAPH_FAILED);

    if (isBiasExist_) {
        OPS_CHECK((expertIdxType_ != expandedRowIdxType_),
            OPS_LOG_E(nodeName_, "expert_idx and expanded_row_idx dtype must be same."), return ge::GRAPH_FAILED);
        OPS_CHECK((biasType_ != gradYType_), OPS_LOG_E(nodeName_, "bias and grad_y dtype must be same."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CheckOutput()
{
    OPS_CHECK(((gradExpandedXShape_.GetDimNum() != NUM_TWO) && (gradExpandedXShape_.GetDimNum() != NUM_THREE)),
        OPS_LOG_E(nodeName_, "grad_expanded_x dimnum must be 2 or 3, but got %zu.", gradExpandedXShape_.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_CHECK((gradScalesShape_.GetDimNum() != NUM_TWO),
        OPS_LOG_E(nodeName_, "grad_scales dimnum must be 2, but got %zu.", gradScalesShape_.GetDimNum()),
        return ge::GRAPH_FAILED);

    if (expandedXDimNum_ == NUM_TWO) {
        OPS_CHECK((gradExpandedXShape_.GetDim(0) != expandedXDim0_),
            OPS_LOG_E(nodeName_, "grad_expanded_x dim0 must be equal to %ld, but got %ld.", expandedXDim0_,
            gradExpandedXShape_.GetDim(0)),
            return ge::GRAPH_FAILED);
        OPS_CHECK((gradExpandedXShape_.GetDim(1) != gradYShape_.GetDim(1)),
            OPS_LOG_E(nodeName_, "grad_expanded_x and grad_y dim1 must be same, but got %ld, %ld.",
            gradExpandedXShape_.GetDim(1), gradYShape_.GetDim(1)),
            return ge::GRAPH_FAILED);
    } else {
        OPS_CHECK((gradExpandedXShape_.GetDim(0) * gradExpandedXShape_.GetDim(1) != expandedXDim0_),
            OPS_LOG_E(nodeName_, "grad_expanded_x dim0 * dim1 must be equal to %ld, but got %ld.", expandedXDim0_,
            gradExpandedXShape_.GetDim(0) * gradExpandedXShape_.GetDim(1)),
            return ge::GRAPH_FAILED);
        OPS_CHECK((gradExpandedXShape_.GetDim(NUM_TWO) != gradYShape_.GetDim(1)),
            OPS_LOG_E(nodeName_, "grad_expanded_x dim2 and grad_y dim1 must be same, but got %ld, %ld.",
            gradExpandedXShape_.GetDim(NUM_TWO), gradYShape_.GetDim(1)),
            return ge::GRAPH_FAILED);
    }

    OPS_CHECK((gradScalesShape_.GetDim(0) != gradYShape_.GetDim(0)),
        OPS_LOG_E(nodeName_, "grad_scales and grad_y dim0 must be same, but got %ld, %ld.", gradScalesShape_.GetDim(0),
        gradYShape_.GetDim(0)),
        return ge::GRAPH_FAILED);
    OPS_CHECK((gradScalesShape_.GetShapeSize() != expandedRowIdxShape_.GetShapeSize()),
        OPS_LOG_E(nodeName_, "grad_scales and expanded_row_idx shape size must be same, but got %ld, %ld.",
        gradScalesShape_.GetShapeSize(), expandedRowIdxShape_.GetShapeSize()),
        return ge::GRAPH_FAILED);

    OPS_CHECK((gradExpandedXType_ != gradYType_),
        OPS_LOG_E(nodeName_, "grad_expanded_x and grad_y dtype must be same."), return ge::GRAPH_FAILED);
    OPS_CHECK((gradScalesType_ != gradYType_), OPS_LOG_E(nodeName_, "grad_scales and grad_y dtype must be same."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CheckParams()
{
    OPS_CHECK((CheckAttr() != ge::GRAPH_SUCCESS),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "CheckAttr check failed."), return ge::GRAPH_FAILED);

    OPS_CHECK((CheckRequiredInput() != ge::GRAPH_SUCCESS),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "CheckRequiredInput check failed."), return ge::GRAPH_FAILED);

    expandedXDimNum_ = NUM_TWO;
    expandedXDim0_ = expandedRowIdxShape_.GetDim(0);
    if (dropPadMode_ == 0 && activeNum_ > 0 && activeNum_ < expandedXDim0_) {
        expandedXDim0_ = activeNum_;
    } else if (dropPadMode_ == 1) {
        expandedXDimNum_ = NUM_THREE;
        expandedXDim0_ = expertNum_ * expertCapacity_;
    }

    if (isScalesExist_) {
        OPS_CHECK(((expandedXShape_.GetDimNum() != NUM_TWO) && (expandedXShape_.GetDimNum() != NUM_THREE)),
            OPS_LOG_E(nodeName_, "expanded_x dimnum must be 2 or 3, but got %zu.", expandedXShape_.GetDimNum()),
            return ge::GRAPH_FAILED);
        OPS_CHECK((scalesShape_.GetDimNum() != NUM_TWO),
            OPS_LOG_E(nodeName_, "scales dimnum must be 2, but got %zu.", scalesShape_.GetDimNum()),
            return ge::GRAPH_FAILED);
        if (isBiasExist_) {
            OPS_CHECK((expertIdxShape_.GetDimNum() != NUM_TWO),
                OPS_LOG_E(nodeName_, "expert_idx dimnum must be 2, but got %zu.", expertIdxShape_.GetDimNum()),
                return ge::GRAPH_FAILED);
            OPS_CHECK((biasShape_.GetDimNum() != NUM_TWO),
                OPS_LOG_E(nodeName_, "bias dimnum must be 2, but got %zu.", biasShape_.GetDimNum()),
                return ge::GRAPH_FAILED);
        }
        OPS_CHECK((CheckOptionalInputShape() != ge::GRAPH_SUCCESS),
            OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "CheckOptionalInputShape check failed."),
            return ge::GRAPH_FAILED);
        OPS_CHECK((CheckOptionalInputDtype() != ge::GRAPH_SUCCESS),
            OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "CheckOptionalInputDtype check failed."),
            return ge::GRAPH_FAILED);
    } else {
        OPS_CHECK((gradYShape_.GetDim(0) != expandedRowIdxShape_.GetDim(0)),
            OPS_LOG_E(nodeName_, "grad_y and expanded_row_idx dim0 must be same, but got %ld, %ld.",
            gradYShape_.GetDim(0), expandedRowIdxShape_.GetDim(0)),
            return ge::GRAPH_FAILED);
    }

    OPS_CHECK((CheckOutput() != ge::GRAPH_SUCCESS),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "CheckOutput check failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MoeFinalizeRoutingV2GradTiling::CalcBaseInfo()
{
    initOutEachCoreBatchNum_ = expandedXDim0_ / totalCoreNum_;
    initOutModCoreNum_ = expandedXDim0_ % totalCoreNum_;
    if (initOutEachCoreBatchNum_ == 0) {
        initOutNeedCoreNum_ = initOutModCoreNum_;
    } else {
        initOutNeedCoreNum_ = totalCoreNum_;
    }

    int64_t expandedRowIdxDim0 = expandedRowIdxShape_.GetDim(0);
    computeEachCoreBatchNum_ = expandedRowIdxDim0 / totalCoreNum_;
    computeModCoreNum_ = expandedRowIdxDim0 % totalCoreNum_;
    if (computeEachCoreBatchNum_ == 0) {
        computeNeedCoreNum_ = computeModCoreNum_;
    } else {
        computeNeedCoreNum_ = totalCoreNum_;
    }

    topK_ = isScalesExist_ ? scalesShape_.GetDim(1) : 1;
    hidden_ = gradYShape_.GetDim(1);
}

void MoeFinalizeRoutingV2GradTiling::CalcTilingKeyWithScales()
{
    int64_t gradYTypePart = gradYTypeByteSize_;
    int64_t expandedXTypePart = gradYTypeByteSize_;
    int64_t gradExpandedXTypePart = gradYTypeByteSize_;
    int64_t totalPart = 0;

    if (gradYType_ == ge::DT_FLOAT) {
        totalPart = gradYTypePart + expandedXTypePart + gradExpandedXTypePart;
        if (isBiasExist_) {
            uint64_t biasTypePart = gradYTypeByteSize_;
            totalPart = totalPart + biasTypePart;
        }
    } else {
        // for cast to float
        totalPart = gradYTypePart * NUM_TWO + expandedXTypePart * NUM_TWO + gradExpandedXTypePart;
        if (isBiasExist_) {
            uint64_t biasTypePart = gradYTypeByteSize_;
            totalPart = totalPart + biasTypePart * NUM_TWO;
        }
    }

    hiddenPrePart_ =
        (totalUbSize_ - NUM_FOUR * BYTE_BLOCK) / totalPart * gradYTypePart / BYTE_BLOCK * BYTE_BLOCK / gradYTypePart;
    if (hidden_ <= hiddenPrePart_) {
        tilingKey_ = TILING_KEY_WITH_SCALE_NOT_CUT_H;
    } else {
        tilingKey_ = TILING_KEY_WITH_SCALE_CUT_H;
    }
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CalcTilingKey()
{
    if (isScalesExist_) {
        CalcTilingKeyWithScales();
    } else {
        hiddenPrePart_ = (totalUbSize_ - BYTE_BLOCK) / BYTE_BLOCK * BYTE_BLOCK / gradYTypeByteSize_;
        if (hidden_ <= hiddenPrePart_) {
            tilingKey_ = TILING_KEY_WITHOUT_SCALE_NOT_CUT_H;
        } else {
            tilingKey_ = TILING_KEY_WITHOUT_SCALE_CUT_H;
        }
    }

    OPS_CHECK((hiddenPrePart_ <= 0), OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "hiddenPrePart_ is less than 0."),
        return ge::GRAPH_FAILED);

    if (hidden_ <= hiddenPrePart_) {
        hiddenInnerLoops_ = 0;
        hiddenLastPart_ = 0;
    } else {
        hiddenInnerLoops_ = hidden_ / hiddenPrePart_;
        hiddenLastPart_ = hidden_ % hiddenPrePart_;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::FillTilingData()
{
    tilingData_.set_initOutNeedCoreNum(initOutNeedCoreNum_);
    tilingData_.set_initOutEachCoreBatchNum(initOutEachCoreBatchNum_);
    tilingData_.set_initOutModCoreNum(initOutModCoreNum_);
    tilingData_.set_computeNeedCoreNum(computeNeedCoreNum_);
    tilingData_.set_computeEachCoreBatchNum(computeEachCoreBatchNum_);
    tilingData_.set_computeModCoreNum(computeModCoreNum_);
    tilingData_.set_dropPadMode(dropPadMode_);
    tilingData_.set_topK(topK_);
    tilingData_.set_hidden(hidden_);
    tilingData_.set_expandedXDim0(expandedXDim0_);
    tilingData_.set_hiddenPrePart(hiddenPrePart_);
    tilingData_.set_hiddenInnerLoops(hiddenInnerLoops_);
    tilingData_.set_hiddenLastPart(hiddenLastPart_);
    tilingData_.set_tilingKey(tilingKey_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());

    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(initOutNeedCoreNum_);
    if (computeNeedCoreNum_ > initOutNeedCoreNum_) {
        context_->SetBlockDim(computeNeedCoreNum_);
    }
    context_->SetTilingKey(tilingKey_);

    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL(context_, workSpaces, return ge::GRAPH_FAILED);
    workSpaces[0] = WORKSPACE_SIZE;

    OPS_LOG_I(nodeName_,
        "MoeFinalizeRoutingV2Grad tilingData is initOutNeedCoreNum:%ld, initOutEachCoreBatchNum:%ld, "
        "initOutModCoreNum:%ld, computeNeedCoreNum:%ld, computeEachCoreBatchNum:%ld, computeModCoreNum:%ld, "
        "dropPadMode:%ld, topK:%ld, hidden:%ld, expandedXDim0:%ld, hiddenPrePart:%ld, hiddenInnerLoops:%ld, "
        "hiddenLastPart:%ld, tilingKey:%ld",
        tilingData_.get_initOutNeedCoreNum(), tilingData_.get_initOutEachCoreBatchNum(),
        tilingData_.get_initOutModCoreNum(), tilingData_.get_computeNeedCoreNum(),
        tilingData_.get_computeEachCoreBatchNum(), tilingData_.get_computeModCoreNum(), tilingData_.get_dropPadMode(),
        tilingData_.get_topK(), tilingData_.get_hidden(), tilingData_.get_expandedXDim0(),
        tilingData_.get_hiddenPrePart(), tilingData_.get_hiddenInnerLoops(), tilingData_.get_hiddenLastPart(),
        tilingData_.get_tilingKey());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CalcTiling()
{
    OPS_CHECK((Init() != ge::GRAPH_SUCCESS), OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "Init failed."),
        return ge::GRAPH_FAILED);

    OPS_CHECK((CheckParams() != ge::GRAPH_SUCCESS),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "CheckParams failed."), return ge::GRAPH_FAILED);

    CalcBaseInfo();

    OPS_CHECK((CalcTilingKey() != ge::GRAPH_SUCCESS),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "CalcTilingKey failed."), return ge::GRAPH_FAILED);

    OPS_CHECK((FillTilingData() != ge::GRAPH_SUCCESS),
        OPS_REPORT_VECTOR_INNER_ERR(nodeName_, "FillTilingData failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus Tiling4MoeFinalizeRoutingV2Grad(gert::TilingContext *context)
{
    MoeFinalizeRoutingV2GradTiling tilingObject(context);
    OPS_CHECK(tilingObject.CalcTiling() != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "CalcTiling failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepare4MoeFinalizeRoutingV2Grad(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeFinalizeRoutingV2Grad)
    .Tiling(Tiling4MoeFinalizeRoutingV2Grad)
    .TilingParse<MoeFinalizeRoutingV2GradCompileInfo>(TilingPrepare4MoeFinalizeRoutingV2Grad);
} // namespace optiling