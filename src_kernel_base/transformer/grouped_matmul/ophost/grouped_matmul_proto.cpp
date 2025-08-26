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
 * \file grouped_matmul.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/ops_log.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {

static constexpr size_t INDEX_IN_X = 0;
static constexpr size_t INDEX_IN_WEIGHT = 1;
static constexpr size_t INDEX_IN_BIAS = 2;
static constexpr size_t INDEX_IN_SCALE = 3;
static constexpr size_t INDEX_IN_OFFSET = 4;
static constexpr size_t INDEX_IN_ANTIQUANT_SCALE = 5;
static constexpr size_t INDEX_IN_ANTIQUANT_OFFSET = 6;
static constexpr size_t INDEX_IN_GROUP_LIST = 7;
static constexpr size_t INDEX_IN_PERTOKEN_SCALE = 8;

static constexpr size_t INDEX_OUT_Y = 0;
static constexpr size_t INDEX_ATTR_SPLIT_ITEM = 0;
static constexpr size_t INDEX_ATTR_OUTPUT_DTYPE = 1;
static constexpr size_t INDEX_ATTR_TRANSPOSE_W = 2;
static constexpr size_t INDEX_ATTR_TRANSPOSE_X = 3;
static constexpr size_t INDEX_ATTR_GROUP_TYPE = 4;
static constexpr size_t INDEX_ATTR_GROUP_LIST_TYPE = 5;
static constexpr size_t INDEX_ATTR_ACT_TYPE = 6;

static constexpr int64_t X_Y_SEPARATED = 0;  // x,y have been separated
static constexpr int64_t Y_SEPARATED = 1;   // y has been separated
static constexpr int64_t X_SEPARATED = 2;   // x has been separated
static constexpr int64_t NO_SEPARATED = 3;  // x,y have not been separated

static constexpr int64_t NO_SPLIT = -1;
static constexpr int64_t SPLIT_M = 0;
static constexpr int64_t SPLIT_N = 1;
static constexpr int64_t SPLIT_K = 2;

static constexpr int64_t OUT_DTYPE_INT32 = 2;

static constexpr int64_t MAX_GROUP_LIST_SIZE_ARRAY = 128;
static constexpr int64_t MAX_GROUP_LIST_SIZE_TENSOR = 1024;
static constexpr int64_t MAX_INNER_AXIS = 65535;
static constexpr size_t MAX_FM_DIM = 6;
static constexpr size_t MIN_FM_DIM = 2;
static constexpr size_t SEPARATED_WEIGHT_DIM = 2;
static constexpr size_t SPLIT_M_SINGLE_WEIGHT_DIM = 3;
static constexpr size_t SPLIT_K_SINGLE_WEIGHT_DIM = 2;

enum class PlatformID {
    UNKNOWN,
    ASCEND310P,
    ASCEND910B
};

enum class GMMActType : int64_t {
    GMM_ACT_TYPE_NONE,
    GMM_ACT_TYPE_RELU,
    GMM_ACT_TYPE_GELU_TANH,
    GMM_ACT_TYPE_GELU_ERR_FUNC,
    GMM_ACT_TYPE_FAST_GELU,
    GMM_ACT_TYPE_SILU,
    END_ACT_TYPE_ENUM
};

static const std::map<int64_t, DataType> OUTPUT_DTYPE_MAP = {
    {0, DataType::DT_FLOAT16},
    {1, DataType::DT_BF16},
    {2, DataType::DT_INT32},
    {-1, DataType::DT_INT8}
};

struct GMMParamsInfo {
    size_t numX;
    size_t numWeight;
    size_t numY;
    int64_t lenGroupList;
    size_t groupNum;
    size_t numScale;
    size_t numOffset;
    size_t numAntiquantScale;
    size_t numAntiquantOffset;
    PlatformID platform;
};

struct GMMAttrs {
    int64_t splitItem;
    int64_t dtype;
    int64_t groupType;
    bool transposeX;
    bool transposeWeight;
    int64_t activeType;
};

static inline std::string ToString(const std::int64_t value) {
    return std::to_string(value);
}

static ge::graphStatus CheckSplitItem(int64_t splitItem) {
    if (splitItem == X_Y_SEPARATED || splitItem == NO_SEPARATED ||
        splitItem == X_SEPARATED || splitItem == Y_SEPARATED) {
        return GRAPH_SUCCESS;
    } else {
        return GRAPH_FAILED;
    }
}

static bool IsTensorListNullOrEmpty(const gert::InferShapeContext* context, size_t index) {
    auto shape = context->GetDynamicInputShape(index, 0);
    if (shape == nullptr) {
        return true;
    }
    if (shape->GetDimNum() == 0 || (shape->GetDimNum() == 1 && shape->GetDim(0) == 0)) {
        if (context->GetDynamicInputShape(index, 1) == nullptr) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus CheckGroupType(const gert::InferShapeContext* context, int64_t groupType) {
    if (groupType == NO_SPLIT || groupType == SPLIT_M || groupType == SPLIT_K) {
        return GRAPH_SUCCESS;
    } else if (groupType == SPLIT_N) {
        OPS_LOG_E(context->GetNodeName(), "Splitting tensor along the N-axis is not supported yet.");
        return GRAPH_FAILED;
    } else {
        OPS_LOG_E(context->GetNodeName(), "GroupType can only be -1/0/2 now, but actually %ld is given.", groupType);
        return GRAPH_FAILED;
    }
}

static ge::graphStatus UpdateShapeYMultiDim(gert::InferShapeContext* context, size_t idxY, const gert::Shape* xShape,
                                            const gert::Shape* weightShape) {
    gert::Shape* yShape = context->GetOutputShape(idxY);
    OPS_LOG_E_IF_NULL(context, yShape, return ge::GRAPH_FAILED);
    *yShape = *xShape;
    size_t dimY = yShape->GetDimNum();
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const bool* transposeWPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANSPOSE_W);
    const bool* transposeXPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANSPOSE_X);

    OPS_LOG_E_IF_NULL(context, weightShape, return ge::GRAPH_FAILED);
    if (transposeWPtr != nullptr && *transposeWPtr) {
        yShape->SetDim(dimY - 1, weightShape->GetDim(weightShape->GetDimNum() - 2));  // -2: transpose weight
    } else {
        yShape->SetDim(dimY - 1, weightShape->GetDim(weightShape->GetDimNum() - 1));
    }
    if (transposeXPtr != nullptr && *transposeXPtr) {
        yShape->SetDim(dimY - 2, xShape->GetDim(xShape->GetDimNum() - 1));  // -2: last two dim of Y
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus UpdateShapeY(gert::InferShapeContext* context, size_t idxY, std::vector<int64_t> yDims) {
    gert::Shape* yShape = context->GetOutputShape(idxY);
    OPS_LOG_E_IF_NULL(context, yShape, return ge::GRAPH_FAILED);
    yShape->SetDimNum(yDims.size());
    for (size_t dim = 0; dim < yDims.size(); ++dim) {
        yShape->SetDim(dim, yDims[dim]);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus UpdateMultipleShapeY(gert::InferShapeContext* context, const gert::Tensor* groupListTensor,
                                            size_t weightDimN, bool isXTransposed, size_t xDimM) {
    auto groupListData = groupListTensor->GetData<int64_t>();
    OPS_CHECK(groupListData == nullptr,
              OPS_LOG_E(context->GetNodeName(), "Failed to obtain necessary data from groupListTensor."),
              return GRAPH_FAILED);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* groupListTypePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_GROUP_LIST_TYPE);
    OPS_LOG_E_IF_NULL(context, groupListTypePtr, return ge::GRAPH_FAILED);
    const gert::Shape* x0Shape = context->GetDynamicInputShape(INDEX_IN_X, 0);
    OPS_LOG_E_IF_NULL(context, x0Shape, return ge::GRAPH_FAILED);
    const gert::Shape* weight0Shape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, 0);
    OPS_LOG_E_IF_NULL(context, weight0Shape, return ge::GRAPH_FAILED);
    int64_t preOffset = 0;
    for (int idx = 0; idx < groupListTensor->GetShapeSize(); ++idx) {
        const gert::Shape* weightShape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, idx);
        if (weightShape == nullptr) {
            weightShape = weight0Shape;
        }
        if (isXTransposed) {
            const gert::Shape* xShape = context->GetDynamicInputShape(INDEX_IN_X, idx);
            if (xShape == nullptr) {
                xShape = x0Shape;
            }
            std::vector<int64_t> yDims = {xShape->GetDim(xDimM), weightShape->GetDim(weightDimN)};
            OPS_CHECK(UpdateShapeY(context, INDEX_OUT_Y + idx, yDims) != GRAPH_SUCCESS, OPS_LOG_E(context->GetNodeName(),
                      "Failed to update shape of y."), return GRAPH_FAILED);
        } else {
            std::vector<int64_t> yDims;
            if (*groupListTypePtr == 0) {
                yDims = {groupListData[idx] - preOffset, weightShape->GetDim(weightDimN)};
            } else if (*groupListTypePtr == 1) {
                yDims = {groupListData[idx], weightShape->GetDim(weightDimN)};
            } else {
                OPS_LOG_E(context->GetNodeName(), "Invalid groupListType = %ld", *groupListTypePtr);
                return GRAPH_FAILED;
            }
            OPS_CHECK(UpdateShapeY(context, INDEX_OUT_Y + idx, yDims) != GRAPH_SUCCESS, OPS_LOG_E(context->GetNodeName(),
                      "Failed to update shape of y."), return GRAPH_FAILED);
            preOffset = groupListData[idx];
        }
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus MultiInMultiOutWithoutGroupList(gert::InferShapeContext* context) {
    size_t idx = 0;
    size_t idw = 0;
    const gert::Shape* w0Shape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, 0);
    OPS_LOG_E_IF_NULL(context, w0Shape, return ge::GRAPH_FAILED);
    while (true) {
        const gert::Shape* xShape = context->GetDynamicInputShape(INDEX_IN_X, idx);
        if (xShape == nullptr) {
            break;
        }
        ++idx;
        const gert::Shape* wShape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, idw);
        if (wShape) {
            ++idw;
        } else {
            wShape = w0Shape;
        }
        OPS_CHECK(UpdateShapeYMultiDim(context, INDEX_OUT_Y + idx - 1, xShape, wShape) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* groupTypePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_GROUP_TYPE);
    bool success = true;
    if (w0Shape->GetDimNum() == 2) {  // 2 two-dim weight tensor
        if (groupTypePtr != nullptr && *groupTypePtr == 2) {
            success = true;
        } else {
            success = idx == idw;
        }
    } else {
        success = static_cast<int64_t>(idx) == w0Shape->GetDim(0);
    }
    OPS_CHECK(!success,
              OPS_LOG_E(context->GetNodeName(),
                        "x tensorList's length[%zu] != weight tensor's first dim[%ld] and length[%zu]",
                        idx, w0Shape->GetDim(0), idw),
             return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrs(gert::InferShapeContext* context, GMMAttrs& gmmAttrs) {
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);

    const int64_t* splitItemPtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SPLIT_ITEM);
    OPS_LOG_E_IF_NULL(context, splitItemPtr, return ge::GRAPH_FAILED);
    gmmAttrs.splitItem = *splitItemPtr;
    OPS_CHECK(CheckSplitItem(gmmAttrs.splitItem) != GRAPH_SUCCESS, OPS_LOG_E(context->GetNodeName(),
              "Invalid splitItem, which can only be one of 0/1/2/3."), return GRAPH_FAILED);
    OPS_LOG_I(context->GetNodeName(), "splitItem = %ld", gmmAttrs.splitItem);

    const int64_t* dtypePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_OUTPUT_DTYPE);
    OPS_LOG_E_IF_NULL(context, dtypePtr, return ge::GRAPH_FAILED);
    gmmAttrs.dtype = *dtypePtr;
    OPS_LOG_I(context->GetNodeName(), "dtype = %ld", gmmAttrs.dtype);

    const int64_t* groupTypePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_GROUP_TYPE);
    OPS_LOG_E_IF_NULL(context, groupTypePtr, return ge::GRAPH_FAILED);
    gmmAttrs.groupType = *groupTypePtr;
    OPS_CHECK(CheckGroupType(context, gmmAttrs.groupType) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Invalid groupType."), return GRAPH_FAILED);
    OPS_LOG_I(context->GetNodeName(), "groupType = %ld", gmmAttrs.groupType);

    const bool* transposeWPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANSPOSE_W);
    OPS_LOG_E_IF_NULL(context, transposeWPtr, return ge::GRAPH_FAILED);
    gmmAttrs.transposeWeight = *transposeWPtr;
    OPS_LOG_I(context->GetNodeName(), "isWeightTransposed = %d", gmmAttrs.transposeWeight);

    const bool* transposeXPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANSPOSE_X);
    OPS_LOG_E_IF_NULL(context, transposeXPtr, return ge::GRAPH_FAILED);
    gmmAttrs.transposeX = *transposeXPtr;
    OPS_LOG_I(context->GetNodeName(), "isXTransposed = %d", gmmAttrs.transposeX);

    const int64_t* activeType = attrs->GetInt(INDEX_ATTR_ACT_TYPE);
    OPS_LOG_E_IF_NULL(context, activeType, return ge::GRAPH_FAILED);
    OPS_CHECK(*activeType < 0 || *activeType >= static_cast<int64_t>(GMMActType::END_ACT_TYPE_ENUM),
              OPS_LOG_E(context->GetNodeName(), "activeType must be no less than 0 and smaller than 6"),
              return GRAPH_FAILED);
    OPS_CHECK(*activeType == static_cast<int64_t>(GMMActType::GMM_ACT_TYPE_GELU_ERR_FUNC),
              OPS_LOG_E(context->GetNodeName(), "Activation function not support GELU_ERR_FUNC now."),
              return GRAPH_FAILED);
    gmmAttrs.activeType = *activeType;
    OPS_LOG_I(context->GetNodeName(), "activeType = %ld", gmmAttrs.activeType);

    return GRAPH_SUCCESS;
}

static ge::graphStatus GetNumOfInputs(const gert::InferShapeContext* context, size_t& numX,
                                      size_t& numWeight, int64_t& lenGroupList) {
    ge::graphStatus res = GRAPH_SUCCESS;
    const gert::Shape* shape = nullptr;
    while (true) {
        shape = context->GetDynamicInputShape(INDEX_IN_X, numX);
        if (shape == nullptr) {  // last shape
            break;
        }
        for (size_t i = 0; i < shape->GetDimNum(); ++i) {
            if (shape->GetDim(i) < 0) {  // shape dim cannot be smaller than 0
                res = GRAPH_FAILED;
                break;
            }
        }
        ++numX;
    }
    OPS_LOG_I(context->GetNodeName(), "numX = %lu", numX);

    while (true) {
        shape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, numWeight);
        if (shape == nullptr) {  // last shape
            break;
        }
        for (size_t i = 0; i < shape->GetDimNum(); ++i) {
            if (shape->GetDim(i) < 0) {  // shape dim cannot be smaller than 0
                res = GRAPH_FAILED;
                break;
            }
        }
        ++numWeight;
    }
    OPS_LOG_I(context->GetNodeName(), "numWeight = %lu", numWeight);

    const gert::Tensor* groupListTensor = context->GetOptionalInputTensor(INDEX_IN_GROUP_LIST);
    if (groupListTensor != nullptr) {
        lenGroupList = groupListTensor->GetShapeSize();
        if (lenGroupList < 0) {  // lenGroupList cannot be smaller than 0
            res = GRAPH_FAILED;
        }
    }
    OPS_LOG_I(context->GetNodeName(), "lenGroupList = %ld", lenGroupList);

    return res;
}

static int64_t GetDim0(const gert::InferShapeContext* context, bool isXTransposed, size_t numX, size_t xDimM) {
    int64_t dim0 = 0;
    if (isXTransposed) {
        const gert::Shape* x0Shape = context->GetDynamicInputShape(INDEX_IN_X, 0);
        dim0 = (x0Shape == nullptr ? 0 : x0Shape->GetDim(xDimM));
    } else {
        for (size_t idx = 0; idx < numX; ++idx) {
            const gert::Shape* xShape = context->GetDynamicInputShape(INDEX_IN_X, idx);
            dim0 += (xShape == nullptr ? 0 : xShape->GetDim(0));
        }
    }

    return dim0;
}

static bool inline IsNonEmpty(const gert::Shape* shape) {
    return (shape != nullptr && !(shape->GetDimNum() == 1 && shape->GetDim(0) == 0));
}

static ge::graphStatus IsGmmAntiQuantEmpty(gert::InferShapeContext* context) {
    OPS_CHECK(!IsTensorListNullOrEmpty(context, INDEX_IN_ANTIQUANT_SCALE),
              OPS_LOG_E(context->GetNodeName(), "antiquantScale is not null or empty!"),
              return GRAPH_FAILED);
    OPS_CHECK(!IsTensorListNullOrEmpty(context, INDEX_IN_ANTIQUANT_OFFSET),
              OPS_LOG_E(context->GetNodeName(), "antiquantOffset is not null or empty!"),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus IsGmmQuantEmpty(gert::InferShapeContext* context) {
    OPS_CHECK(!IsTensorListNullOrEmpty(context, INDEX_IN_SCALE),
              OPS_LOG_E(context->GetNodeName(), "scale is not null or empty!"),
              return GRAPH_FAILED);
    OPS_CHECK(!IsTensorListNullOrEmpty(context, INDEX_IN_OFFSET),
              OPS_LOG_E(context->GetNodeName(), "offset is not null or empty!"),
              return GRAPH_FAILED);
    const gert::Shape* pertokenQuantScale0Shape = context->GetOptionalInputShape(INDEX_IN_PERTOKEN_SCALE);
    OPS_CHECK(IsNonEmpty(pertokenQuantScale0Shape),
              OPS_LOG_E(context->GetNodeName(), "pertokenQuant scale is not null or empty!"),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckNonQuant(gert::InferShapeContext* context) {
    OPS_CHECK(IsGmmQuantEmpty(context) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Detected nonquant, but quant inputs is not empty!"),
              return GRAPH_FAILED);
    OPS_CHECK(IsGmmAntiQuantEmpty(context) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Detected nonquant, but antiquant inputs is not empty!"),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus GetGroupSize(const gert::InferShapeContext* context, GMMParamsInfo& paramsInfo) {
    size_t groupNum = 1;
    size_t maxGroupNum = MAX_GROUP_LIST_SIZE_ARRAY;  // init max value
    if (paramsInfo.numX > 1) {
        groupNum = paramsInfo.numX;
    } else if (paramsInfo.numWeight > 1) {
        groupNum = paramsInfo.numWeight;
    } else if (paramsInfo.numY > 1) {
        groupNum = paramsInfo.numY;
    } else if (paramsInfo.lenGroupList > 0) {
        groupNum = paramsInfo.lenGroupList;
        maxGroupNum = MAX_GROUP_LIST_SIZE_TENSOR;  // only this case allows MAX_GROUP_LIST_SIZE_TENSOR size
    }
    OPS_CHECK(groupNum > maxGroupNum,
              OPS_LOG_E(context->GetNodeName(), "groupNum[%zu] is larger than %zu.",
                        groupNum, maxGroupNum),
              return GRAPH_FAILED);
    paramsInfo.groupNum = groupNum;
    return GRAPH_SUCCESS;
}

static graphStatus CheckDimNumAndPerGroupNum(const gert::InferShapeContext* context, bool isAntiquantInt4,
    const std::tuple<size_t, size_t, int64_t>& dimData, const gert::Shape* tensorShape, const std::string& tensorType) {
    size_t tensorDimNum = std::get<0>(dimData);
    size_t expectedDimNum = std::get<1>(dimData);  // 1: the sceond element
    int64_t weightKDimValue = std::get<2>(dimData);  // 2: the third element
    if (isAntiquantInt4) {
        if (tensorDimNum == expectedDimNum) {
            int64_t perGroupNum = tensorShape->GetDim(tensorDimNum - 2);  // 2: the last 2-th index
            OPS_CHECK(!(perGroupNum > 0 && weightKDimValue % perGroupNum == 0),
                      OPS_LOG_E(context->GetNodeName(), "perGroupNum must be larger than 0, and can evenly divided "
                                "by K[%ld] in A16W4-pergroup case, but now perGroupNum is %ld.", weightKDimValue, perGroupNum),
                      return GRAPH_FAILED);
        } else {
            OPS_CHECK(tensorDimNum != expectedDimNum - 1,
                      OPS_LOG_E(context->GetNodeName(), "%s Dim must be %zu for in perchannel case or "
                                "%zu for pergroup case in A16W4, but now is %zu.",
                                tensorType.c_str(), expectedDimNum - 1, expectedDimNum, tensorDimNum),
                      return GRAPH_FAILED);
        }
    } else {
        OPS_CHECK(tensorDimNum != expectedDimNum - 1,
                  OPS_LOG_E(context->GetNodeName(), "%s Dim must be %zu, but now is %zu.",
                            tensorType.c_str(), expectedDimNum - 1, tensorDimNum),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckOptionalTensorList(gert::InferShapeContext* context, const std::string tensorType,
                                               const GMMParamsInfo& paramsInfo, const GMMAttrs& gmmAttrs, size_t nodeIdx) {
    // check bias，scale, antiquant scale or antiquant offset's size，tensor dimension and shape.
    const size_t& groupNum = paramsInfo.groupNum;
    size_t tensorSize = 0;
    while (context->GetDynamicInputShape(nodeIdx, tensorSize) != nullptr) {
        ++tensorSize;
    }
    uint64_t weightGroupedSize = static_cast<uint64_t>(paramsInfo.numWeight);
    const int64_t& groupType = gmmAttrs.groupType;
    auto shape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, 0);
    OPS_LOG_E_IF_NULL(context, shape, return ge::GRAPH_FAILED);
    uint64_t weightNDimIdx = shape->GetDimNum() - (gmmAttrs.transposeWeight ? 2 : 1);
    auto tensor0Shape = context->GetDynamicInputShape(nodeIdx, 0);
    // tensorList size should equals with weight's size
    OPS_CHECK(tensorSize != weightGroupedSize, OPS_LOG_E(context->GetNodeName(),
              "%s size[%lu] must be equal with weight size[%lu].", tensorType.c_str(), tensorSize, weightGroupedSize), return GRAPH_FAILED);
    bool isSingleWeight = (weightGroupedSize == 1 && groupType != NO_SPLIT);
    auto w0Desc = context->GetDynamicInputDesc(INDEX_IN_WEIGHT, 0);
    OPS_LOG_E_IF_NULL(context, w0Desc, return ge::GRAPH_FAILED);
    bool isAntiquantInt4 = (w0Desc->GetDataType() == DT_INT4 && tensorType.find("antiquant") != std::string::npos);
    if (isSingleWeight) {  // In this case, nodeIdx should have only single tensor, its dim should be 2.
        OPS_CHECK(IsTensorListNullOrEmpty(context, nodeIdx), OPS_LOG_E(context->GetNodeName(),
                  "%s must not be nullptr or empty, but now is nullptr or empty.", tensorType.c_str()), return GRAPH_FAILED);
        size_t tensorDimNum = tensor0Shape->GetDimNum();
        int64_t k = shape->GetDim(shape->GetDimNum() - (gmmAttrs.transposeWeight ? 1 : 2));  // 2: axis index
        // 3: shape is (E,G,N),G is the perGroupNum
        OPS_CHECK(CheckDimNumAndPerGroupNum(context, isAntiquantInt4, {tensorDimNum, 3, k}, tensor0Shape, tensorType) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "CheckDimNumAndPerGroupNum failed."), return GRAPH_FAILED);
        OPS_CHECK(static_cast<size_t>(tensor0Shape->GetDim(0)) != groupNum, OPS_LOG_E(context->GetNodeName(), "%s batch size[%ld] should be "
                  "euqal with groupList length[%lu].", tensorType.c_str(), tensor0Shape->GetDim(0), groupNum), return GRAPH_FAILED);
        // tensor's N axis size should equal with weight's N axis.
        int64_t weightNDimValue = context->GetDynamicInputShape(INDEX_IN_WEIGHT, 0)->GetDim(weightNDimIdx);
        int64_t tensorNDimValue = tensor0Shape->GetDim(tensorDimNum - 1);
        OPS_CHECK(tensorNDimValue != weightNDimValue, OPS_LOG_E(context->GetNodeName(),
                  "NDim[%ld] of %s should be equal with NDim[%ld] of weight.", tensorNDimValue, tensorType.c_str(), weightNDimValue),
                  return GRAPH_FAILED);
    } else {
        for (uint64_t i = 0; i < groupNum; i++) {
            auto tensorShape = context->GetDynamicInputShape(nodeIdx, i);
            OPS_CHECK(tensorShape == nullptr, OPS_LOG_E(context->GetNodeName(),
                      "%s[%lu] must not be nullptr, but now is nullptr.", tensorType.c_str(), i), return GRAPH_FAILED);
            // check each of tensor's dim to be 1
            size_t tensorDimNum = tensorShape->GetDimNum();
            auto wShape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, i);
            OPS_LOG_E_IF_NULL(context, wShape, return ge::GRAPH_FAILED);
            int64_t k = wShape->GetDim(wShape->GetDimNum() - (gmmAttrs.transposeWeight ? 1 : 2));  // 2: axis index
            // 2: shape is (G,N), G is the perGroupNum
            OPS_CHECK(CheckDimNumAndPerGroupNum(context, isAntiquantInt4, {tensorDimNum, 2, k}, tensorShape, tensorType) != GRAPH_SUCCESS,
                      OPS_LOG_E(context->GetNodeName(), "CheckDimNumAndPerGroupNum failed."), return GRAPH_FAILED);
            int64_t weightNDimValue = wShape->GetDim(weightNDimIdx);
            int64_t tensorNDimValue = tensorShape->GetDim(tensorDimNum - 1);
            OPS_CHECK(tensorNDimValue != weightNDimValue, OPS_LOG_E(context->GetNodeName(), "NDim[%ld] of %s[%lu] should be equal with "
                      "NDim[%ld] of weight[%lu].", tensorNDimValue, tensorType.c_str(), i, weightNDimValue, i), return GRAPH_FAILED);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckPerTokenScale(const gert::InferShapeContext* context, const GMMParamsInfo& paramsInfo) {
    // check pertoken scale's size, tensor dimension and shape
    const size_t& xGroupedSize = paramsInfo.numX;
    const size_t& weightGroupedSize = paramsInfo.numWeight;
    const size_t& yGroupedSize = paramsInfo.numY;
    uint64_t xMDimIdx = 0;
    // check pertoken scale's size to be equal with x's
    if (xGroupedSize == 1 && weightGroupedSize == 1 && yGroupedSize == 1) {
        auto perTokenScale0Shape = context->GetOptionalInputShape(INDEX_IN_PERTOKEN_SCALE);
        OPS_CHECK(perTokenScale0Shape == nullptr,
                  OPS_LOG_E(context->GetNodeName(), "perTokenScaleOptional must not be nullptr, but now is nullptr."),
                  return GRAPH_FAILED);
        // tensor dimension of pertoken_scale should be 1.
        size_t tensorDimNum = perTokenScale0Shape->GetDimNum();
        OPS_CHECK(tensorDimNum != 1,
                  OPS_LOG_E(context->GetNodeName(),
                            "perTokenScaleOptional dim num must be 1 when x is single tensor, but now is %zu.", tensorDimNum),
                  return GRAPH_FAILED);
        // check pertoken_scale's tensor shape size to be equal with M axis size of x.
        auto xShape = context->GetDynamicInputShape(INDEX_IN_X, 0);
        OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED);
        int64_t xMDimValue = xShape->GetDim(xMDimIdx);
        int64_t tensorMDimValue = perTokenScale0Shape->GetDim(tensorDimNum - 1);
        OPS_CHECK(tensorMDimValue != xMDimValue,
                  OPS_LOG_E(context->GetNodeName(),
                            "MDim[%ld] of perTokenScaleOptional should be equal with MDim[%ld] of x.",
                            tensorMDimValue, xMDimValue),
                  return GRAPH_FAILED);
    } else {
        OPS_LOG_E(context->GetNodeName(), "per-token quant case is only supported "
                  "when x, weight and y are all single tensor, but now x size is %zu, weight size is %zu, y size is %zu",
                  xGroupedSize, weightGroupedSize, yGroupedSize);
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupedMatmulQuant(gert::InferShapeContext* context, const GMMAttrs& gmmAttrs,
                                               const GMMParamsInfo& paramsInfo) {
    OPS_CHECK(paramsInfo.platform == PlatformID::ASCEND310P,
              OPS_LOG_E(context->GetNodeName(), "quant cases do not support on Ascend310P."),
              return GRAPH_FAILED);
    OPS_CHECK(gmmAttrs.groupType == SPLIT_K,
              OPS_LOG_E(context->GetNodeName(), "quant cases do not support splited axis is K."),
              return GRAPH_FAILED);
    OPS_CHECK(!IsTensorListNullOrEmpty(context, INDEX_IN_OFFSET),
              OPS_LOG_E(context->GetNodeName(), "offset must be nullptr in quant, but now is not nullptr."),
              return GRAPH_FAILED);
    if (gmmAttrs.dtype != OUT_DTYPE_INT32) {  // output dtype is int32, this scene does not need scale
        OPS_CHECK(IsTensorListNullOrEmpty(context, INDEX_IN_SCALE),
                OPS_LOG_E(context->GetNodeName(), "scale must not be nullptr in quant, but now is nullptr."),
                return GRAPH_FAILED);
        OPS_CHECK(CheckOptionalTensorList(context, "scale", paramsInfo, gmmAttrs, INDEX_IN_SCALE) != GRAPH_SUCCESS,
                OPS_LOG_E(context->GetNodeName(), "Invalid scale."),
                return GRAPH_FAILED);
    }
    bool isPerTokenQuant = context->GetOptionalInputShape(INDEX_IN_PERTOKEN_SCALE) != nullptr;
    if (isPerTokenQuant) {
        OPS_CHECK(CheckPerTokenScale(context, paramsInfo) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Check perTokenScale failed!"),
                  return GRAPH_FAILED);
    }
    OPS_CHECK(IsGmmAntiQuantEmpty(context) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Detected quant, but antiquant inputs is not empty!"),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static int64_t GetPergroupSize(const GMMAttrs& gmmAttrs, bool isSingleWeight,
                               const gert::Shape* wShape, const gert::Shape* shape) {
  int64_t pergroupSize = 0;
  size_t shapeDimNum = shape->GetDimNum();
  if (isSingleWeight) {  // antiquant param shape (E, N), (E, G, N)
    if (shapeDimNum > SEPARATED_WEIGHT_DIM) {
      int64_t k = gmmAttrs.transposeWeight ?  wShape->GetDim(2) : wShape->GetDim(1);  // 2: the k axis index
      pergroupSize = k / shape->GetDim(shapeDimNum - 2);  // 2: the last 2-th index
    }
  } else {  //  antiquant param shape (N), (G, N)
    if (shapeDimNum > 1) {
      int64_t k = gmmAttrs.transposeWeight ? wShape->GetDim(1): wShape->GetDim(0);
      pergroupSize = k / shape->GetDim(shapeDimNum - 2);  // 2: the last 2-th index
    }
  }
  return pergroupSize;
}

static ge::graphStatus CheckGroupedMatmulAntiQuantForShape(gert::InferShapeContext* context, const GMMAttrs& gmmAttrs, const GMMParamsInfo& paramsInfo) {
    OPS_CHECK(paramsInfo.platform == PlatformID::ASCEND310P, OPS_LOG_E(context->GetNodeName(),
              "antiquant cases do not support on Ascend310P."), return GRAPH_FAILED);
    OPS_CHECK(gmmAttrs.groupType == SPLIT_K, OPS_LOG_E(context->GetNodeName(), "antiquant cases do not support splited axis is K."),
              return GRAPH_FAILED);
    OPS_CHECK(IsTensorListNullOrEmpty(context, INDEX_IN_ANTIQUANT_SCALE),
              OPS_LOG_E(context->GetNodeName(), "antiquantScale must not be nullptr in antiquant, but now is nullptr or empty."),
              return GRAPH_FAILED);
    OPS_CHECK(IsTensorListNullOrEmpty(context, INDEX_IN_ANTIQUANT_OFFSET),
              OPS_LOG_E(context->GetNodeName(), "antiquantOffset must not be nullptr in antiquant, but now is nullptr or empty."),
              return GRAPH_FAILED);
    // check antiquantScale and antiquantOffset's tensor shape
    OPS_CHECK(CheckOptionalTensorList(context, "antiquantScale", paramsInfo, gmmAttrs, INDEX_IN_ANTIQUANT_SCALE) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Invalid antiquantScale"),
              return GRAPH_FAILED);
    OPS_CHECK(CheckOptionalTensorList(context, "antiquantOffset", paramsInfo, gmmAttrs, INDEX_IN_ANTIQUANT_OFFSET) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Invalid antiquantOffset"),
              return GRAPH_FAILED);
    // check perGroupSize
    auto w0Desc = context->GetDynamicInputDesc(INDEX_IN_WEIGHT, 0);
    if (w0Desc->GetDataType() == DT_INT4) {
        auto antiquantScale0Shape = context->GetDynamicInputShape(INDEX_IN_ANTIQUANT_SCALE, 0);
        auto dimNum = antiquantScale0Shape->GetDimNum();
        bool isSingleWeight = (paramsInfo.numWeight == 1 && gmmAttrs.groupType != NO_SPLIT);
        int64_t pergroupSize = GetPergroupSize(gmmAttrs, isSingleWeight, context->GetDynamicInputShape(INDEX_IN_WEIGHT, 0), antiquantScale0Shape);
        OPS_CHECK(gmmAttrs.transposeWeight && pergroupSize % 2 != 0,  // 2: a factor
                  OPS_LOG_E(context->GetNodeName(), "pergroupSize should be even when weight is transposed"
                  "in A16W4-pergroup case, but now is %ld", pergroupSize), return GRAPH_FAILED);
        for (size_t i = 0; ; ++i) {
            auto antiquantScaleShape = context->GetDynamicInputShape(INDEX_IN_ANTIQUANT_SCALE, i);
            auto antiquantOffsetShape = context->GetDynamicInputShape(INDEX_IN_ANTIQUANT_OFFSET, i);
            if (antiquantScaleShape == nullptr || antiquantOffsetShape == nullptr) {
                break;
            }
            size_t antiquantScaleDimNum = antiquantScaleShape->GetDimNum();
            size_t antiquantOffsetDimNum = antiquantOffsetShape->GetDimNum();
            OPS_CHECK(antiquantScaleDimNum != dimNum || antiquantOffsetDimNum != dimNum,
                      OPS_LOG_E(context->GetNodeName(), "antiquantScale[%zu] dim num[%zu] or antiquantOffset[%zu] dim num[%zu] is not equal with %zu",
                      i, antiquantScaleDimNum, i, antiquantOffsetDimNum, dimNum), return GRAPH_FAILED);
            auto wShape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, i);
            int64_t pergroupSizeOfScale = GetPergroupSize(gmmAttrs, isSingleWeight, wShape, antiquantScaleShape);
            int64_t pergroupSizeOfOffset = GetPergroupSize(gmmAttrs, isSingleWeight, wShape, antiquantOffsetShape);
            OPS_CHECK(pergroupSizeOfScale != pergroupSize || pergroupSizeOfOffset != pergroupSize,
                      OPS_LOG_E(context->GetNodeName(), "antiquantScale[%zu]'s pergroup size[%ld] or antiquantOffset[%zu]'s pergroup size[%ld]"
                      "is not the required value[%ld]", i, pergroupSizeOfScale, i, pergroupSizeOfOffset, pergroupSize),
                      return GRAPH_FAILED);
        }
    }
    OPS_CHECK(IsGmmQuantEmpty(context) != GRAPH_SUCCESS, OPS_LOG_E(context->GetNodeName(),
              "Detected antiquant, but quant inputs is not empty!"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckFunctionParamsForShape(gert::InferShapeContext* context, const GMMAttrs& gmmAttrs,
                                                   GMMParamsInfo& paramsInfo) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    if (ret != ge::GRAPH_SUCCESS) {
        paramsInfo.platform = PlatformID::UNKNOWN;
        OPS_LOG_W(context->GetNodeName(), "Cannot get platform info!");
        return GRAPH_SUCCESS;
    } else {
        paramsInfo.platform = optionalInfo.soc_version.find("310P") != std::string::npos ?
                              PlatformID::ASCEND310P : PlatformID::ASCEND910B;
    }
    auto x0Desc = context->GetDynamicInputDesc(INDEX_IN_X, 0);
    OPS_LOG_E_IF_NULL(context, x0Desc, return ge::GRAPH_FAILED);
    DataType xDtype = x0Desc->GetDataType();
    auto w0Desc = context->GetDynamicInputDesc(INDEX_IN_WEIGHT, 0);
    OPS_LOG_E_IF_NULL(context, w0Desc, return ge::GRAPH_FAILED);
    DataType weightDtype = w0Desc->GetDataType();
    if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT4) {
        return GRAPH_SUCCESS;
    }
    if ((xDtype == DataType::DT_BF16 || xDtype == DataType::DT_FLOAT16 ||
        xDtype == DataType::DT_FLOAT) && xDtype == weightDtype) {
        // nonquant
        return CheckNonQuant(context);
    }
    if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT8) {
        // quant
        return CheckGroupedMatmulQuant(context, gmmAttrs, paramsInfo);
    }
    if ((xDtype == DataType::DT_BF16 || xDtype == DataType::DT_FLOAT16) &&
        (weightDtype == DataType::DT_INT8 || weightDtype == DataType::DT_INT4)) {
        // antiquant
        return CheckGroupedMatmulAntiQuantForShape(context, gmmAttrs, paramsInfo);
    }
    return GRAPH_FAILED;
}

static ge::graphStatus CheckDimNumAndGroupListNoSplitAndFormat(const gert::InferShapeContext* context,
    uint64_t tensorListLength, const size_t numWeight) {
    // when groupList is not empty, check its size equal with the length of x.
    auto groupTensorOptionalShape = context->GetOptionalInputShape(INDEX_IN_GROUP_LIST);
    if (groupTensorOptionalShape != nullptr) {
        OPS_CHECK(groupTensorOptionalShape->GetDim(0) != static_cast<int64_t>(tensorListLength),
                  OPS_LOG_E(context->GetNodeName(), "Size of groupList(tensor) %ld should be equal to size of x %lu.",
                            groupTensorOptionalShape->GetDim(0), tensorListLength),
                  return GRAPH_FAILED);
    }
    auto wShape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, 0);
    OPS_LOG_E_IF_NULL(context, wShape, return ge::GRAPH_FAILED);
    // check dimension
    for (size_t i = 0; i < tensorListLength; ++i) {
        auto xShape = context->GetDynamicInputShape(INDEX_IN_X, i);
        OPS_CHECK(xShape == nullptr,
                  OPS_LOG_E(context->GetNodeName(), "x[%lu] is null, which is not supported.", i),
                  return GRAPH_FAILED);
        if (numWeight > 1) {
            wShape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, i);
            OPS_LOG_E_IF_NULL(context, wShape, return ge::GRAPH_FAILED);
            size_t weightDimNum = wShape->GetDimNum();
            OPS_CHECK(weightDimNum != SEPARATED_WEIGHT_DIM,
                      OPS_LOG_E(context->GetNodeName(),
                                "weight[%lu] dimNum is %lu , but only support 2 when weight separated.",
                                i, weightDimNum),
                      return GRAPH_FAILED);
        }
        size_t xDimNum = xShape->GetDimNum();
        OPS_CHECK(xDimNum > MAX_FM_DIM || xDimNum < MIN_FM_DIM,
                  OPS_LOG_E(context->GetNodeName(), "x[%lu] dimNum is %lu , but only support 2-6.", i, xDimNum),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus TensorType2NodeId(const std::vector<std::string>& tensorType, std::vector<int64_t>& nodeIdx) {
    if (nodeIdx.size() > tensorType.size()) {
        return GRAPH_FAILED;
    }
    for (size_t i(0); i < nodeIdx.size(); ++i) {
        if (tensorType[i] == "x") {
            nodeIdx[i] = INDEX_IN_X;
        } else if (tensorType[i] == "weight") {
            nodeIdx[i] = INDEX_IN_WEIGHT;
        } else if (tensorType[i] == "y") {
            nodeIdx[i] = INDEX_OUT_Y;
        } else {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckDimNum(gert::InferShapeContext* context, uint64_t tensorListLength,
                                            const size_t expectedDimNum, const std::string tensorType) {
    int64_t nodeIdx = 0;
    if (tensorType == "x") {
        nodeIdx = INDEX_IN_X;
    } else if (tensorType == "weight") {
        nodeIdx = INDEX_IN_WEIGHT;
    } else if (tensorType == "y") {
        nodeIdx = INDEX_OUT_Y;
    } else {
        return GRAPH_FAILED;
    }
    const gert::Shape* shape;
    for (size_t i = 0; i < tensorListLength; ++i) {
        if (tensorType == "y") {
            shape = context->GetOutputShape(nodeIdx + i);
        } else {
            shape = context->GetDynamicInputShape(nodeIdx, i);
        }
        OPS_CHECK(shape == nullptr,
                  OPS_LOG_E(context->GetNodeName(), "%s[%lu] is null, which is not supported.", tensorType.c_str(), i),
                  return GRAPH_FAILED);
        size_t dimNum = shape->GetDimNum();
        OPS_CHECK(dimNum != expectedDimNum,
                  OPS_LOG_E(context->GetNodeName(), "%s[%lu] dim num should be %lu in this case, but now is %lu.",
                         tensorType.c_str(), i, expectedDimNum, dimNum),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightShapeInnerAxisEven(const gert::InferShapeContext* context, const size_t weightSize,
                                                     const int64_t innerAxisDimId) {
    auto w0Desc = context->GetDynamicInputDesc(INDEX_IN_WEIGHT, 0);
    OPS_LOG_E_IF_NULL(context, w0Desc, return ge::GRAPH_FAILED);
    DataType wDtype = w0Desc->GetDataType();
    if (wDtype == DataType::DT_INT4) {
        for (size_t i = 0; i < weightSize; ++i) {
            auto wShape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, i);
            OPS_LOG_E_IF_NULL(context, wShape, return ge::GRAPH_FAILED);
            int64_t n = wShape->GetDim(innerAxisDimId);
            OPS_CHECK(n % 2 != 0,
                      OPS_LOG_E(context->GetNodeName(), "w[%zu] dim %ld value %ld should be even when weight is int4 dtype.",
                                i, innerAxisDimId, n),
                      return GRAPH_FAILED);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus IsxSizeEqualWithWeightKAxis(const gert::InferShapeContext* context,
    const GMMParamsInfo& paramsInfo, const gert::Shape* wShape, size_t& wKDimIdx, size_t& wNDimIdx) {
    if (paramsInfo.numWeight == 1 && wShape->GetDimNum() > 2) {  // 2: separated tensor's dim
        wKDimIdx += 1;
        wNDimIdx += 1;
        OPS_CHECK(paramsInfo.numX != static_cast<size_t>(wShape->GetDim(0)),
                  OPS_LOG_E(context->GetNodeName(), "When x and y are separated, and weight is not separated, size of x "
                            "%zu should equal to the first dim of weight tensor %ld.", paramsInfo.numX, wShape->GetDim(0)),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckCaseNoSplit(gert::InferShapeContext* context, bool transposeWeight,
                                        const GMMParamsInfo& paramsInfo) {
    const size_t& xSize = paramsInfo.numX;
    const size_t& weightSize = paramsInfo.numWeight;
    // check group num
    OPS_CHECK(xSize != paramsInfo.numY, OPS_LOG_E(context->GetNodeName(),
              "When y is separated, size of x %lu should equal to size of y %lu.", xSize, paramsInfo.numY), return GRAPH_FAILED);
    OPS_CHECK(weightSize != 1 && xSize != weightSize, OPS_LOG_E(context->GetNodeName(), "When x and weight are separated, "
              "size of x %lu should equal to size of weight %lu.", xSize, weightSize), return GRAPH_FAILED);
    // check dimension
    OPS_CHECK(CheckDimNumAndGroupListNoSplitAndFormat(context, xSize, weightSize) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Dim num or format of tensor in tensor lists or grouplist is invalid."),
              return GRAPH_FAILED);
    // check shape
    auto wShape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, 0);
    OPS_LOG_E_IF_NULL(context, wShape, return ge::GRAPH_FAILED);
    size_t wKDimIdx = transposeWeight ? 1 : 0;
    size_t wNDimIdx = transposeWeight ? 0 : 1;
    OPS_CHECK(IsxSizeEqualWithWeightKAxis(context, paramsInfo, wShape, wKDimIdx, wNDimIdx) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "IsxSizeEqualWithWeightKAxis failed."), return GRAPH_FAILED);
    int64_t weightKDimValue = wShape->GetDim(wKDimIdx);
    int64_t weightNDimValue = wShape->GetDim(wNDimIdx);
    auto w0Desc = context->GetDynamicInputDesc(INDEX_IN_WEIGHT, 0);
    OPS_LOG_E_IF_NULL(context, w0Desc, return ge::GRAPH_FAILED);
    DataType wDtype = w0Desc->GetDataType();
    // 2: an even factor
    OPS_CHECK(wDtype == DataType::DT_INT4 && weightNDimValue % 2 != 0, OPS_LOG_E(context->GetNodeName(),
              "w[0] dim %lu value %ld should be even when weight is int4 dtype.", wNDimIdx, weightNDimValue),
              return GRAPH_FAILED);
    for (size_t i = 0; i < xSize; i++) {
        auto xShape = context->GetDynamicInputShape(INDEX_IN_X, i);
        size_t xDimNum = xShape->GetDimNum();
        // check inner axis of x, which should not be larger than 65535
        int64_t xKDimValue = xShape->GetDim(xDimNum - 1);  // x always is not transposed
        OPS_CHECK(xKDimValue > MAX_INNER_AXIS,
                  OPS_LOG_E(context->GetNodeName(), "x[%lu] dim %lu value %ld should less or equal to %ld.",
                            i, xDimNum - 1, xKDimValue, MAX_INNER_AXIS),
                  return GRAPH_FAILED);
        if (weightSize > 1) {
            wShape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, i);
            weightKDimValue = wShape->GetDim(wKDimIdx);
            weightNDimValue = wShape->GetDim(wNDimIdx);
            // 2: an even factor
            OPS_CHECK(i > 0 && wDtype == DataType::DT_INT4 && weightNDimValue % 2 != 0, OPS_LOG_E(context->GetNodeName(),
                      "w[%lu] dim %lu value %ld should be even when weight is int4 dtype.", i, wNDimIdx, weightNDimValue),
                      return GRAPH_FAILED);
        }
        OPS_CHECK(xKDimValue != weightKDimValue,
                  OPS_LOG_E(context->GetNodeName(), "x[%lu] dim %lu value %ld should equal to weight[%lu] dim 0 value %ld.",
                            i, xDimNum - 1, xKDimValue, i, weightKDimValue),
                  return GRAPH_FAILED);
        // if weight is not transposed, check N aisx; otherwise, check K axis, which can be skiped
        OPS_CHECK(!transposeWeight && weightNDimValue > MAX_INNER_AXIS,
                  OPS_LOG_E(context->GetNodeName(), "w[%zu] dim %zu value %ld should less or equal to %ld.",
                            i, wNDimIdx, weightNDimValue, MAX_INNER_AXIS),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckInnerAxisOfTensorList(const gert::InferShapeContext* context, size_t nodeId,
                                                  int64_t innerAxisDimId, size_t checkNum) {
    for (size_t i = 0; i < checkNum; i++) {
        auto shape = context->GetDynamicInputShape(nodeId, i);
        OPS_LOG_E_IF_NULL(context, shape, return ge::GRAPH_FAILED);
        int64_t innerAxisValue = shape->GetDim(innerAxisDimId);
        OPS_CHECK(innerAxisValue > MAX_INNER_AXIS,
                  OPS_LOG_E(context->GetNodeName(), "Dim %ld value of %zu-th shape should less or equal to %ld, "
                            "but now is %ld.", innerAxisDimId, i, MAX_INNER_AXIS, innerAxisValue),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckShapeSameLengthTensorList(gert::InferShapeContext* context,
                                                      const std::vector<size_t>& dimIds, const int64_t innerAxisDimId,
                                                      const std::vector<std::string> tensorType, uint64_t groupNum) {
    std::vector<int64_t> nodeIdx = {0, 0};
    OPS_CHECK(TensorType2NodeId(tensorType, nodeIdx) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "TensorType2NodeId failed."),
              return GRAPH_FAILED);
    // check two tensorlist's size to be the same, and tensors to have consistant dimension.
    const gert::Shape* shape;
    for (uint64_t i = 0; i < groupNum; i++) {
        shape = context->GetDynamicInputShape(nodeIdx[0], i);
        OPS_LOG_E_IF_NULL(context, shape, return ge::GRAPH_FAILED);
        int64_t dimValue1 = shape->GetDim(dimIds[0]);
        // tensorType[2] indicates whether check tensorList0's inner axis(innerAxisDimId)
        if (tensorType[2] == "true" && innerAxisDimId > -1) {
            auto shape0 = context->GetDynamicInputShape(nodeIdx[0], i);
            OPS_LOG_E_IF_NULL(context, shape0, return ge::GRAPH_FAILED);
            int64_t innerAxisValue = shape0->GetDim(innerAxisDimId);
            OPS_CHECK(innerAxisValue > MAX_INNER_AXIS,
                      OPS_LOG_E(context->GetNodeName(), "Dim %lu value of %s[%lu] should less or equal to %ld, "
                                "but now is %ld.",
                                dimIds[0], tensorType[0].c_str(), i, MAX_INNER_AXIS, innerAxisValue),
                      return GRAPH_FAILED);
        }
        if (tensorType[1] == "y") {
            shape = context->GetOutputShape(nodeIdx[1] + i);
        } else {
            shape = context->GetDynamicInputShape(nodeIdx[1], i);
        }
        OPS_LOG_E_IF_NULL(context, shape, return ge::GRAPH_FAILED);
        int64_t dimValue2 = shape->GetDim(dimIds[1]);
        OPS_CHECK(dimValue1 != dimValue2,
                  OPS_LOG_E(context->GetNodeName(),
                            "Dim %lu value of %s[%lu] should be equal with dim %lu value of %s[%lu]"
                            ", but now is %ld and %ld respectively.", dimIds[0], tensorType[0].c_str(),
                            i, dimIds[1], tensorType[1].c_str(), i, dimValue1, dimValue2),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckShapeDiffLengthTensorList(gert::InferShapeContext* context,
                                                      const std::vector<size_t>& dimIds,
                                                      const int64_t innerAxisdimId,
                                                      const std::vector<std::string> tensorType,
                                                      uint64_t groupNum) {
    std::vector<int64_t> nodeIdx = {0, 0};
    OPS_CHECK(TensorType2NodeId(tensorType, nodeIdx) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "TensorType2NodeId failed."),
              return GRAPH_FAILED);
    // check each tensor's selected dimension size in a multi-tensor tensorlist's to equal with
    // the tensor selected dimension in single-tensor tensorlist.
    // the selected axis is not the split-axis.
    const gert::Shape* singleTensor0;
    if (tensorType[1] == "y") {
        singleTensor0 = context->GetOutputShape(nodeIdx[1]);
    } else {
        singleTensor0 = context->GetDynamicInputShape(nodeIdx[1], 0);
    }
    OPS_LOG_E_IF_NULL(context, singleTensor0, return ge::GRAPH_FAILED);
    int64_t dimValueSingle = singleTensor0->GetDim(dimIds[1]);
    // tensorType[2] indicates whether check single tensorList's inner axis(innerAxisDimId)
    if (tensorType[2] == "true" && innerAxisdimId > -1) {
        int64_t dimValue = singleTensor0->GetDim(innerAxisdimId);
        OPS_CHECK(dimValue > MAX_INNER_AXIS,
                  OPS_LOG_E(context->GetNodeName(),
                            "Dim %ld value of %s[0] should less or equal to %ld, but now is %ld.",
                            innerAxisdimId, tensorType[1].c_str(), MAX_INNER_AXIS, dimValue),
                  return GRAPH_FAILED);
    }
    const gert::Shape* longTensor;
    for (uint64_t i = 0; i < groupNum; i++) {
        if (tensorType[0] == "y") {
            longTensor = context->GetOutputShape(nodeIdx[0] + i);
        } else {
            longTensor = context->GetDynamicInputShape(nodeIdx[0], i);
        }
        OPS_LOG_E_IF_NULL(context, longTensor, return ge::GRAPH_FAILED);
        int64_t dimValueLong = longTensor->GetDim(dimIds[0]);
        OPS_CHECK(dimValueLong != dimValueSingle,
                  OPS_LOG_E(context->GetNodeName(),
                            "Dim %lu value of %s[%lu] %ld should be equal with dim %lu value of %s[0] %ld.",
                            dimIds[0], tensorType[0].c_str(), i, dimValueLong,
                            dimIds[1], tensorType[1].c_str(), dimValueSingle),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupListCommonTensor(const gert::InferShapeContext* context,
                                                  const bool isRequiredGroupList, const int64_t groupNum) {
    auto groupTensorOptionalShape = context->GetOptionalInputShape(INDEX_IN_GROUP_LIST);
    bool isNull = groupTensorOptionalShape == nullptr;
    OPS_CHECK(isNull && isRequiredGroupList,
              OPS_LOG_E(context->GetNodeName(), "groupListOptional(tensor) is required in this case, but get nullptr."),
              return GRAPH_FAILED);
    if (isNull) {
        return GRAPH_SUCCESS;
    }
    int64_t groupListSize = groupTensorOptionalShape->GetDim(0);
    OPS_CHECK(groupListSize > MAX_GROUP_LIST_SIZE_TENSOR,
              OPS_LOG_E(context->GetNodeName(),
                        "When groupList type is tenosr, size of groupList %ld should be less than or equal to %ld.",
                        groupListSize, MAX_GROUP_LIST_SIZE_TENSOR),
              return GRAPH_FAILED);
    OPS_CHECK(!((groupListSize == groupNum && groupNum > 1) || groupNum == 1),
              OPS_LOG_E(context->GetNodeName(),
                        "When groupList is not null, size of groupList(tensor) %ld should be equal to groupNum %ld.",
                        groupListSize, groupNum),
              return GRAPH_FAILED);
    auto groupListDesc = context->GetOptionalInputDesc(INDEX_IN_GROUP_LIST);
    OPS_LOG_E_IF_NULL(context, groupListDesc, return ge::GRAPH_FAILED);
    OPS_CHECK(groupListDesc->GetDataType() != DataType::DT_INT64,
              OPS_LOG_E(context->GetNodeName(), "Invalid dtype: Only int64 is supported for groupList, but now is %s.",
                        ToString(groupListDesc->GetDataType()).data()),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus SplitMSingleXSingleWeightSingleY(gert::InferShapeContext* context, bool transposeWeight,
                                                        const GMMParamsInfo& paramsInfo) {
    std::vector<std::string> tenorXAndWeight{"x", "weight", "true"};
    // check dimension
    OPS_CHECK(CheckDimNum(context, paramsInfo.numX, MIN_FM_DIM, "x") != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Dim num or format of tensor in tensor list x is invalid."),
              return GRAPH_FAILED);
    OPS_CHECK(CheckDimNum(context, paramsInfo.numWeight, SPLIT_M_SINGLE_WEIGHT_DIM, "weight") != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Dim num or format of tensor in tensor list weight is invalid."),
              return GRAPH_FAILED);
    // check shape, x(m,k), weight(b,k,n), y(m,n)
    int64_t innerAxisDimId = 1;  // x always is not transposed, check K axis
    size_t kAxisOfWeight = transposeWeight ? 2 : 1;  // if weight is transposed, 2 is the k axis idx of the weight, otherwise is 1
    OPS_CHECK(CheckShapeSameLengthTensorList(context, {1, kAxisOfWeight}, innerAxisDimId, tenorXAndWeight, paramsInfo.numX) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "k dim value of x and weight is not matched."),
              return GRAPH_FAILED);
    innerAxisDimId = !transposeWeight ? 2 : -1;  // If w is not transposed, check N(2) asix; otherwise, check k axis, which can be skiped
    OPS_CHECK(CheckInnerAxisOfTensorList(context, INDEX_IN_WEIGHT, innerAxisDimId, paramsInfo.numWeight) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "inner axis size of weight is larger than %ld!", MAX_INNER_AXIS),
              return GRAPH_FAILED);
    OPS_CHECK(CheckWeightShapeInnerAxisEven(context, paramsInfo.numWeight, 2) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "weight's N axis size should be even when it is int4 dtype."),
              return GRAPH_FAILED);
    // check groupList
    OPS_CHECK(CheckGroupListCommonTensor(context, true, context->GetDynamicInputShape(INDEX_IN_WEIGHT, 0)->GetDim(0)) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Invalid groupList."),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus SplitMSingleXSeparatedWeightSingleY(gert::InferShapeContext* context, bool transposeWeight,
                                                           const GMMParamsInfo& paramsInfo) {
    std::vector<std::string> tenorWeightAndX{"weight", "x", "true"};
    // check dimension
    OPS_CHECK(CheckDimNum(context, paramsInfo.numX, MIN_FM_DIM, "x") != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Dim num or format of tensor in tensor list x is invalid."),
              return GRAPH_FAILED);
    OPS_CHECK(CheckDimNum(context, paramsInfo.numWeight, SEPARATED_WEIGHT_DIM, "weight") != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Dim num or format of tensor in tensor list weight is invalid."),
              return GRAPH_FAILED);
    // check shape, x(m,k), weight(k,n), y(m,n)
    int64_t innerAxisDimId = 1;  // x always is not transposed, check K axis
    size_t kAxisOfWeight = transposeWeight ? 1 : 0;
    OPS_CHECK(CheckShapeDiffLengthTensorList(context, {kAxisOfWeight, 1}, innerAxisDimId, tenorWeightAndX, paramsInfo.numWeight) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "k dim value of x and weight is not matched."),
              return GRAPH_FAILED);
    innerAxisDimId = !transposeWeight ? 1 : -1;  // if w is not transposed, check N asix; otherwise, check k axis, which can be skiped
    OPS_CHECK(CheckInnerAxisOfTensorList(context, INDEX_IN_WEIGHT, innerAxisDimId, 1) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "inner axis size of weight is larger than %ld!", MAX_INNER_AXIS),
              return GRAPH_FAILED);
    OPS_CHECK(CheckWeightShapeInnerAxisEven(context, paramsInfo.numWeight, 1) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "weight's N axis size should be even when it is int4 dtype."),
              return GRAPH_FAILED);
    // check groupList
    OPS_CHECK(CheckGroupListCommonTensor(context, true, paramsInfo.numWeight) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Invalid groupList."),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus SplitMSeparatedXSeparatedWeightSingleY(gert::InferShapeContext* context,
                                                              bool transposeWeight, const GMMParamsInfo& paramsInfo) {
    const size_t& xSize = paramsInfo.numX;
    const size_t& weightSize = paramsInfo.numWeight;
    std::vector<std::string> tenorWeightAndX{"weight", "x", "true"};
    OPS_CHECK(xSize != weightSize,
              OPS_LOG_E(context->GetNodeName(),
                        "When x and weight are separated, size of x %lu should equal to size of weight %lu.",
                        xSize, weightSize),
              return GRAPH_FAILED);
    // check dimension
    OPS_CHECK(CheckDimNum(context, xSize, MIN_FM_DIM, "x") != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Dim num or format of tensor in tensor list x is invalid."),
              return GRAPH_FAILED);
    OPS_CHECK(CheckDimNum(context, weightSize, SEPARATED_WEIGHT_DIM, "weight") != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Dim num or format of tensor in tensor list weight is invalid."),
              return GRAPH_FAILED);
    // check shape, x(m,k), weight(k,n), y(m,n)
    int64_t innerAxisDimId = 1;  // originalShape's inner axis of weight
    size_t kAxisOfWeight = transposeWeight ? 1 : 0;
    OPS_CHECK(CheckShapeSameLengthTensorList(context, {kAxisOfWeight, 1}, innerAxisDimId, tenorWeightAndX, weightSize) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "k dim value of x and weight is not matched."),
              return GRAPH_FAILED);
    innerAxisDimId = !transposeWeight ? 1 : -1;  // if w is not transposed, N asix has been checked, need to check x's inner axis(K, when x is always not transposed)
    OPS_CHECK(CheckInnerAxisOfTensorList(context, INDEX_IN_X, innerAxisDimId, 1) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "inner axis size of x is larger than %ld!", MAX_INNER_AXIS),
              return GRAPH_FAILED);
    OPS_CHECK(CheckWeightShapeInnerAxisEven(context, weightSize, 1) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "weight's N axis size should be even when it is int4 dtype."),
              return GRAPH_FAILED);
    // check groupList
    OPS_CHECK(CheckGroupListCommonTensor(context, false, xSize) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Invalid groupList."),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckCaseSplitM(gert::InferShapeContext* context, bool transposeWeight,
                                       const GMMParamsInfo& paramsInfo) {
    const size_t& xSize = paramsInfo.numX;
    const size_t& weightSize = paramsInfo.numWeight;
    const size_t& ySize = paramsInfo.numY;
    if (xSize == 1 && weightSize == 1 && ySize == 1) {
        OPS_CHECK(SplitMSingleXSingleWeightSingleY(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Split m, single x, single weight, single y case failed."),
                  return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    if (xSize == 1 && weightSize > 1 && ySize == 1) {
        OPS_CHECK(weightSize != paramsInfo.groupNum, OPS_LOG_E(context->GetNodeName(),
                  "weight Size [%zu] does not equal with groupNum %zu", weightSize, paramsInfo.groupNum),
                  return GRAPH_FAILED);
        OPS_CHECK(SplitMSingleXSeparatedWeightSingleY(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Split m, single x, separated weight, single y case failed."),
                  return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    if (xSize == 1 && weightSize > 1 && ySize > 1) {
        const gert::Tensor* groupListTensor = context->GetOptionalInputTensor(INDEX_IN_GROUP_LIST);
        OPS_CHECK(groupListTensor == nullptr || groupListTensor->GetData<int64_t>() == nullptr,
                  OPS_LOG_E(context->GetNodeName(), "Failed to obtain necessary data from groupListTensor. "
                            "When grouplist is an invalid tensor, split m, single x, separated weight, separated y cases do not support."),
                  return GRAPH_FAILED);
        return GRAPH_SUCCESS;  // skip the check
    }
    if (xSize > 1 && weightSize > 1 && ySize == 1) {
        OPS_CHECK(weightSize != paramsInfo.groupNum, OPS_LOG_E(context->GetNodeName(),
                  "weight Size [%zu] does not equal with groupNum %zu", weightSize, paramsInfo.groupNum),
                  return GRAPH_FAILED);
        OPS_CHECK(SplitMSeparatedXSeparatedWeightSingleY(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Split m, separated x, separated weight, single y case failed."),
                  return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    OPS_LOG_E(context->GetNodeName(), "When groupType is 0, current case with x %zu, weight %zu, y %zu is not supported.",
              xSize, weightSize, ySize);
    return GRAPH_FAILED;
}

static ge::graphStatus CheckCaseSplitK(gert::InferShapeContext* context, bool transposeX, bool transposeWeight,
                                       const GMMParamsInfo& paramsInfo) {
    std::vector<std::string> tenorXAndWeight{"x", "weight", "true"};
    const size_t& xSize = paramsInfo.numX;
    const size_t& weightSize = paramsInfo.numWeight;
    const size_t& ySize = paramsInfo.numY;
    if (xSize == 1 && ySize == 1 && weightSize == 1) {
        OPS_CHECK(!transposeX,
                  OPS_LOG_E(context->GetNodeName(),
                            "When groupType is 2 and x is not separated, tensor in x should be transposed."),
                  return GRAPH_FAILED);
        // check dimension
        OPS_CHECK(CheckDimNum(context, xSize, MIN_FM_DIM, "x") != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Dim num or format of tensor in tensor list x is invalid."),
                  return GRAPH_FAILED);
        OPS_CHECK(CheckDimNum(context, weightSize, SPLIT_K_SINGLE_WEIGHT_DIM, "weight") != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Dim num or format of tensor in tensor list weight is invalid."),
                  return GRAPH_FAILED);
        // check shape, x(m,k), weight(k,n), y(b,m,n)
        int64_t innerAxisDimId = 1;  // x always is transposed, and the inner axis is always the last axis, M axis.
        size_t kAxisOfWeight = transposeWeight ? 1 : 0;
        OPS_CHECK(CheckShapeSameLengthTensorList(context, {0, kAxisOfWeight}, innerAxisDimId, tenorXAndWeight, xSize) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "k dim value of x and weight is not matched."),
                  return GRAPH_FAILED);
        innerAxisDimId = 1;  // w always is not transposed, and the inner axis is always the last axis, N axis.
        OPS_CHECK(CheckInnerAxisOfTensorList(context, INDEX_IN_WEIGHT, innerAxisDimId, weightSize) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "inner axis size of weight is larger than %ld!", MAX_INNER_AXIS),
                  return GRAPH_FAILED);
        // check groupList
        OPS_CHECK(CheckGroupListCommonTensor(context, true, 1) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Invalid groupList."),
                  return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    OPS_LOG_E(context->GetNodeName(),
              "When groupType is 2, only support case with unseparated x, weight and y, "
              "but now x size is %lu, weight size is %lu, y size is %lu.", xSize, weightSize, ySize);
    return GRAPH_FAILED;
}

static ge::graphStatus CheckParamDifferentGroupType(gert::InferShapeContext* context, const GMMAttrs& gmmAttrs,
                                                    const GMMParamsInfo& paramsInfo) {
    OPS_CHECK(paramsInfo.platform == PlatformID::UNKNOWN, OPS_LOG_W(context->GetNodeName(), "Cannot get platform info!"), return GRAPH_SUCCESS);
    const int64_t& groupType = gmmAttrs.groupType;
    const bool& transposeX = gmmAttrs.transposeX;
    const bool& transposeWeight = gmmAttrs.transposeWeight;
    OPS_CHECK(transposeX && transposeWeight, OPS_LOG_E(context->GetNodeName(),
              "x and weight can not be transposed at the same time."), return GRAPH_FAILED);
    auto groupTensorOptionalShape = context->GetOptionalInputShape(INDEX_IN_GROUP_LIST);
    OPS_CHECK(groupTensorOptionalShape != nullptr && (groupTensorOptionalShape->GetDimNum() > 1 ||
              groupTensorOptionalShape->GetDim(0) < 1),
              OPS_LOG_E(context->GetNodeName(), "When groupList is a tensor,  its dim only supports 1 and size of "
                        "elements should be larger than 0, but now are %zu and %ld, respectively.",
                        groupTensorOptionalShape->GetDimNum(), groupTensorOptionalShape->GetDim(0)),
              return GRAPH_FAILED);
    OPS_CHECK(paramsInfo.platform == PlatformID::ASCEND310P && !(groupType == SPLIT_M && paramsInfo.numX == 1 &&
              paramsInfo.numWeight == 1 && paramsInfo.numY == 1),
              OPS_LOG_E(context->GetNodeName(),
                        "When on ASCEND310P, it only supports split m, single x, single weight, single y."),
              return GRAPH_FAILED);

    if (groupType == NO_SPLIT) {
        OPS_CHECK(transposeX, OPS_LOG_E(context->GetNodeName(),
                  "When x, weight and y are all separated, x can not be transposed."), return GRAPH_FAILED);
        OPS_CHECK(CheckCaseNoSplit(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Invalid inputs!"), return GRAPH_FAILED);
    } else if (groupType == SPLIT_M) {
        OPS_CHECK(transposeX,
                  OPS_LOG_E(context->GetNodeName(), "When groupType is 0, x can not be transposed."),
                  return GRAPH_FAILED);
        OPS_CHECK(CheckCaseSplitM(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Invalid inputs!"), return GRAPH_FAILED);
    } else if (groupType == SPLIT_K) {
        OPS_CHECK(!IsTensorListNullOrEmpty(context, INDEX_IN_BIAS),
                  OPS_LOG_E(context->GetNodeName(), "When groupType is 2, bias must be empty."), return GRAPH_FAILED);
        OPS_CHECK(CheckCaseSplitK(context, transposeX, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Invalid inputs!"), return GRAPH_FAILED);
    }
    if (!IsTensorListNullOrEmpty(context, INDEX_IN_BIAS)) {
        OPS_CHECK(CheckOptionalTensorList(context, "bias", paramsInfo, gmmAttrs, INDEX_IN_BIAS) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Invalid bias!"), return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus XNotSingleYSeparated(gert::InferShapeContext* context,
                                            size_t weightDimN, bool isXTransposed, size_t xDimM) {
    const gert::Tensor* groupListTensor = context->GetOptionalInputTensor(INDEX_IN_GROUP_LIST);
    if (groupListTensor != nullptr) {
        OPS_CHECK(UpdateMultipleShapeY(context, groupListTensor, weightDimN, isXTransposed, xDimM) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    } else {
        OPS_CHECK(MultiInMultiOutWithoutGroupList(context)!= GRAPH_SUCCESS, OPS_LOG_E(context->GetNodeName(),
                  "Failed to process multi-in-multi-out case without GroupList."), return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus XSingleYSeparated(gert::InferShapeContext* context,
                                         size_t weightDimN, bool isXTransposed, size_t xDimM) {
    const gert::Tensor* groupListTensor = context->GetOptionalInputTensor(INDEX_IN_GROUP_LIST);
    OPS_CHECK(groupListTensor == nullptr,
              OPS_LOG_E(context->GetNodeName(), "GroupList is required when x is single tensor while y is not."),
              return GRAPH_FAILED);
    OPS_CHECK(UpdateMultipleShapeY(context, groupListTensor, weightDimN, isXTransposed, xDimM) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Failed to update shape of y."),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4GroupedMatmul(gert::InferShapeContext* context) {
    GMMAttrs gmmAttrs{X_Y_SEPARATED, 0, NO_SPLIT, false, false, 0};
    OPS_CHECK(GetAttrs(context, gmmAttrs) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "Failed to get attrs."), return GRAPH_FAILED);

    size_t numX = 0;  // init numX
    size_t numWeight = 0;  // init numWeight
    int64_t lenGroupList = 0;  // init lenGroupList
    size_t numY = context->GetComputeNodeOutputNum();
    if (GetNumOfInputs(context, numX, numWeight, lenGroupList) == GRAPH_SUCCESS) {  // check input shape value inside
        GMMParamsInfo paramsInfo{numX, numWeight, numY, lenGroupList, 0, 0, 0, 0, 0, PlatformID::UNKNOWN};
        OPS_CHECK(GetGroupSize(context, paramsInfo) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "check groupNum failed"), return GRAPH_FAILED);
        OPS_CHECK(CheckFunctionParamsForShape(context, gmmAttrs, paramsInfo) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "CheckFunctionParamsForShape failed."), return GRAPH_FAILED);
        OPS_CHECK(CheckParamDifferentGroupType(context, gmmAttrs, paramsInfo) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "CheckParamDifferentGroupType failed."), return GRAPH_FAILED);
    } else {
        OPS_CHECK(CheckDimNum(context, numX, MIN_FM_DIM, "x") != GRAPH_SUCCESS,  // check dim number of tensors
                  OPS_LOG_E(context->GetNodeName(), "Dim num of tensor in tensorList x is invalid."),
                  return GRAPH_FAILED);
    }

    const gert::Shape* x0Shape = context->GetDynamicInputShape(INDEX_IN_X, 0);
    OPS_LOG_E_IF_NULL(context, x0Shape, return ge::GRAPH_FAILED);
    size_t xDimNum = x0Shape->GetDimNum();
    const gert::Shape* w0Shape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, 0);
    OPS_LOG_E_IF_NULL(context, w0Shape, return ge::GRAPH_FAILED);
    size_t weightDimNum = w0Shape->GetDimNum();
    bool isSingleX = numX == 1 && gmmAttrs.groupType != NO_SPLIT;
    bool isSingleY = numY == 1 && gmmAttrs.groupType != NO_SPLIT;
    size_t xDimM = gmmAttrs.transposeX ? xDimNum - 1 : xDimNum - 2;
    size_t weightDimN = gmmAttrs.transposeWeight ? weightDimNum - 2 : weightDimNum - 1;
    // set y shape
    if (isSingleX && !isSingleY) {
        OPS_CHECK(XSingleYSeparated(context, weightDimN, gmmAttrs.transposeX, xDimM) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    } else if (isSingleX && isSingleY) {
        OPS_CHECK(gmmAttrs.groupType != SPLIT_M && gmmAttrs.groupType != SPLIT_K, OPS_LOG_E(context->GetNodeName(),
                  "When x is single tensor, input tensors can only be split along M or K axis."), return GRAPH_FAILED);
        std::vector<int64_t> yDims = {x0Shape->GetDim(xDimM), w0Shape->GetDim(weightDimN)};
        if (gmmAttrs.groupType == SPLIT_K) { yDims.insert(yDims.begin(), numWeight == 1 ? lenGroupList : numWeight); }
        OPS_CHECK(UpdateShapeY(context, INDEX_OUT_Y, yDims) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Failed to update y shape."), return GRAPH_FAILED);
    } else if (!isSingleX && !isSingleY) {
        OPS_CHECK(XNotSingleYSeparated(context, weightDimN, gmmAttrs.transposeX, xDimM) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    } else if (!isSingleX && isSingleY) {
        std::vector<int64_t> yDims = {GetDim0(context, gmmAttrs.transposeX, numX, xDimM), w0Shape->GetDim(weightDimN)};
        OPS_CHECK(UpdateShapeY(context, INDEX_OUT_Y, yDims) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

// =========================================================================================
// =========================================================================================
static graphStatus CheckTensorListDataType(const gert::InferDataTypeContext* context, uint32_t index,
                                           const DataType dtype) {
    size_t inIdx = 0;
    while (true) {
        auto iDtype = context->GetDynamicInputDataType(index, inIdx);
        if (iDtype == DT_UNDEFINED) {
            break;
        }
        OPS_CHECK(iDtype != dtype,
                  OPS_LOG_E(context->GetNodeName(), "data type of tensors in a tensorList should all be the same!"),
                  return GRAPH_FAILED);
        ++inIdx;
    }
    return GRAPH_SUCCESS;
}

static graphStatus CheckMatmulDataType(gert::InferDataTypeContext* context, const DataType xDtype,
                                       const DataType weightDtype, const DataType biasDtype) {
    OPS_CHECK(CheckTensorListDataType(context, INDEX_IN_X, xDtype) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "x dtype does not match with required dtype[%s].",
                        ToString(xDtype).data()),
              return GRAPH_FAILED);
    OPS_CHECK(CheckTensorListDataType(context, INDEX_IN_WEIGHT, weightDtype) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "weight dtype does not match with required dtype[%s].",
                        ToString(weightDtype).data()),
              return GRAPH_FAILED);
    OPS_CHECK(CheckTensorListDataType(context, INDEX_IN_BIAS, biasDtype) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "bias dtype does not match with required dtype[%s].",
                        ToString(biasDtype).data()),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus CheckFunctionQuantParams(gert::InferDataTypeContext* context) {
    OPS_CHECK(CheckTensorListDataType(context, INDEX_IN_X, DataType::DT_INT8) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "x dtype does not match with required dtype[INT8]."),
              return GRAPH_FAILED);
    OPS_CHECK(CheckTensorListDataType(context, INDEX_IN_WEIGHT, DataType::DT_INT8) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "weight dtype does not match with required dtype[INT8]."),
              return GRAPH_FAILED);
    OPS_CHECK(CheckTensorListDataType(context, INDEX_IN_BIAS, DataType::DT_INT32) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "bias dtype does not match with required dtype int32."),
              return GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* outputDtype = attrs->GetInt(INDEX_ATTR_OUTPUT_DTYPE);
    if (*outputDtype == OUT_DTYPE_INT32) {  // output dtype is int32, this scene does not need scale
        return GRAPH_SUCCESS;
    }
    auto scale0Dtype = context->GetDynamicInputDataType(INDEX_IN_SCALE, 0);
    // Now we cannot make sure if is pertoken quant case, so scale/offset dtype check is remained to the InferShape stage.
    OPS_CHECK(CheckTensorListDataType(context, INDEX_IN_SCALE, scale0Dtype) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "dtypes of scales in the tensorList should all be the same."),
              return GRAPH_FAILED);
    auto offset0Dtype = context->GetDynamicInputDataType(INDEX_IN_OFFSET, 0);
    OPS_CHECK(CheckTensorListDataType(context, INDEX_IN_OFFSET, offset0Dtype) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "dtypes of offsets in the tensorList should all be the same."),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus CheckGroupedMatmulAntiQuantForDtype(gert::InferDataTypeContext* context) {
    auto xDtype = context->GetDynamicInputDataType(INDEX_IN_X, 0);
    OPS_CHECK(CheckTensorListDataType(context, INDEX_IN_ANTIQUANT_SCALE, xDtype) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "antiquantScale dtype does not match with x dtype[%s].", ToString(xDtype).data()),
              return GRAPH_FAILED);
    OPS_CHECK(CheckTensorListDataType(context, INDEX_IN_ANTIQUANT_OFFSET, xDtype) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "antiquantOffset dtype does not match with x dtype[%s].", ToString(xDtype).data()),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus CheckFunctionParamsForDtype(gert::InferDataTypeContext* context) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    graphStatus ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    PlatformID platform = PlatformID::UNKNOWN;
    if (ret != ge::GRAPH_SUCCESS) {
        OPS_LOG_W(context->GetNodeName(), "Cannot get platform info.");
        return GRAPH_SUCCESS;
    } else {
        platform = optionalInfo.soc_version.find("310P") == std::string::npos ? PlatformID::ASCEND910B : PlatformID::ASCEND310P;
    }
    DataType xDtype = context->GetDynamicInputDataType(INDEX_IN_X, 0);
    DataType weightDtype = context->GetDynamicInputDataType(INDEX_IN_WEIGHT, 0);
    if (platform == PlatformID::ASCEND310P) {
        bool isAllInputFP16 = xDtype == DataType::DT_FLOAT16 && weightDtype == DataType::DT_FLOAT16;
        OPS_CHECK(!isAllInputFP16, OPS_LOG_E(context->GetNodeName(),
                  "Only float16 is supported on Ascend310P platforms."), return GRAPH_FAILED);
        auto biasDtype = context->GetOptionalInputDataType(INDEX_IN_BIAS);
        OPS_CHECK(biasDtype != ge::DT_UNDEFINED && biasDtype != DataType::DT_FLOAT16, OPS_LOG_E(context->GetNodeName(),
                  "only bias float16 is supported on Ascend310P platforms."), return GRAPH_FAILED);
    }
    if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT4) { return GRAPH_SUCCESS; }
    if ((xDtype == DataType::DT_BF16 || xDtype == DataType::DT_FLOAT16 || xDtype == DataType::DT_FLOAT) &&
        xDtype == weightDtype) {  // nonquant
        DataType biasDtype = xDtype == DataType::DT_BF16 ? DataType::DT_FLOAT: xDtype;
        OPS_CHECK(CheckMatmulDataType(context, xDtype, weightDtype, biasDtype) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "case with x dtype %s and weight dtype %s is not supported!",
                            ToString(xDtype).data(), ToString(weightDtype).data()),
                  return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT8) {
        // quant
        OPS_CHECK(CheckFunctionQuantParams(context) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "CheckFunctionQuantParams failed."),
                  return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    if ((xDtype == DataType::DT_BF16 || xDtype == DataType::DT_FLOAT16) &&
        (weightDtype == DataType::DT_INT8 || weightDtype == DataType::DT_INT4)) {
        // antiquant
        DataType biasDtype = xDtype == DataType::DT_BF16 ? DataType::DT_FLOAT: DataType::DT_FLOAT16;
        OPS_CHECK(CheckMatmulDataType(context, xDtype, weightDtype, biasDtype) != GRAPH_SUCCESS,
                  OPS_LOG_E(context->GetNodeName(), "case with x dtype %s and weight dtype %s is not supported!",
                            ToString(xDtype).data(), ToString(weightDtype).data()),
                  return GRAPH_FAILED);
        return CheckGroupedMatmulAntiQuantForDtype(context);
    }
    OPS_LOG_E(context->GetNodeName(), "GMM: there is no matching xDtype and weightDtype pattern. "
              "case with x dtype %s and weight dtype %s is not supported.",
              ToString(xDtype).data(), ToString(weightDtype).data());
    return GRAPH_FAILED;
}

static graphStatus CheckQuantParamsDtype(const gert::InferDataTypeContext* context, const int64_t outputDtype, 
                                         const DataType yDtype) {
    size_t i = 0;
    auto scale0Dtype = context->GetDynamicInputDataType(INDEX_IN_SCALE, 0);
    OPS_CHECK(scale0Dtype == ge::DT_UNDEFINED, OPS_LOG_E(context->GetNodeName(), "scale is undefined!"),
              return GRAPH_FAILED);  
    auto perTokenScale0Dtype = context->GetDynamicInputDataType(INDEX_IN_PERTOKEN_SCALE, 0);
    bool isPerTokenQuant = perTokenScale0Dtype != ge::DT_UNDEFINED;
    if (isPerTokenQuant) {
        bool isOutputBF16 = scale0Dtype == DataType::DT_BF16 && outputDtype == 1;
        bool isOutputFloat16 = scale0Dtype == DataType::DT_FLOAT && outputDtype == 0; 
        OPS_CHECK(!isOutputBF16 && !isOutputFloat16,
                  OPS_LOG_E(context->GetNodeName(), "per-token quant case only supports scale data type bfloat16 with "
                            "output data type bfloat16, or scale with data type float32 when output is float16, but "
                            "now scale[%zu] has data type %s and output has data type %s!",
                            i, ToString(scale0Dtype).data(), ToString(yDtype).data()),
                  return GRAPH_FAILED);
    } else {
        bool isOutputInt8 = scale0Dtype == DataType::DT_UINT64 && outputDtype == -1;
        bool isOutputBF16 = scale0Dtype == DataType::DT_BF16 && outputDtype == 1;
        bool isOutputFP16 = scale0Dtype == DataType::DT_FLOAT && outputDtype == 0; 
        OPS_CHECK(!isOutputInt8 && !isOutputBF16 && !isOutputFP16,
                  OPS_LOG_E(context->GetNodeName(), "per-channel quant case only supports scale with data type uint64 "
                            "when output is int8, or data type bfloat16 when output is bfloat16, or data type float32 "
                            "when output is float16, but scale[%zu] has data type %s and output has data type %s!",
                            i, ToString(scale0Dtype).data(), ToString(yDtype).data()), 
                  return GRAPH_FAILED);
    }
    if (isPerTokenQuant) {   
        OPS_CHECK(perTokenScale0Dtype != DataType::DT_FLOAT,
                  OPS_LOG_E(context->GetNodeName(), "pertoken quant case only support perTokenScale with dtype float32,"
                            "but perTokenScale[%zu] has data type %s!", i, ToString(perTokenScale0Dtype).data()),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4GroupedMatmul(gert::InferDataTypeContext* context) {
    OPS_CHECK(CheckFunctionParamsForDtype(context) != GRAPH_SUCCESS,
              OPS_LOG_E(context->GetNodeName(), "CheckFunctionParamsForDtype failed!"),
              return GRAPH_FAILED);

    auto x0Dtype = context->GetDynamicInputDataType(INDEX_IN_X, 0);
    auto weight0Dtype = context->GetDynamicInputDataType(INDEX_IN_WEIGHT, 0);
    size_t numY = context->GetComputeNodeOutputNum();
    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    bool isQuantCase = x0Dtype == ge::DT_INT8 && weight0Dtype == ge::DT_INT8;
    const int64_t* outputDtype = attrs->GetInt(INDEX_ATTR_OUTPUT_DTYPE);
    DataType yDtype = x0Dtype;
    if (isQuantCase && outputDtype != nullptr) {
        auto it = OUTPUT_DTYPE_MAP.find(*outputDtype);
        OPS_CHECK(it == OUTPUT_DTYPE_MAP.end(),
                  OPS_LOG_E(context->GetNodeName(),
                            "value of attr dtype only supports -1/0/1/2, but now is %ld.", *outputDtype),
                  return GRAPH_FAILED);
        yDtype = it->second;
        if (*outputDtype != OUT_DTYPE_INT32) {  // output dtype is int32, this scene does not need scale
            OPS_CHECK(CheckQuantParamsDtype(context, *outputDtype, yDtype) != GRAPH_SUCCESS,
                    OPS_LOG_E(context->GetNodeName(), "Check quant params data type failed!"),
                    return GRAPH_FAILED);
        }
    } 
    for (size_t k = 0; k < numY; k++) {
        context->SetOutputDataType(INDEX_OUT_Y + k, yDtype);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmul)
    .InferShape(InferShape4GroupedMatmul)
    .InferDataType(InferDataType4GroupedMatmul);
}  // namespace ops
