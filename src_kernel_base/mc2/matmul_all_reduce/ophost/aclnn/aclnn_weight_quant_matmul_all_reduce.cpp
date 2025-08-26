/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_weight_quant_matmul_all_reduce.h"
#include "securec.h"

#include "acl/acl.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "matmul_util.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t FOUR_DIMS = 4;
static constexpr size_t THREE_DIMS = 3;
static constexpr size_t TWO_DIMS = 2;
static constexpr size_t ONE_DIM = 1;
static constexpr int64_t NUM_ACL_STOP_ON_FAILURE = 1;
static constexpr size_t DIM_LEN_ONE = 1;
static constexpr size_t DIM_LEN_TWO = 2;
static constexpr int64_t ANTIQUANT_GROUP_SIZE_MIN_VALUE = 32;
constexpr uint8_t MC2_DEBUG_ONLY_CUBE = 1;
constexpr char DEBUG_MODE_ENV[] = "ASCEND_MC2_DEBUG_MODE";

typedef struct {
    uint32_t id;
    const char *funcName;
    bool hasReg;
} NnopbaseDfxId;

extern aclnnStatus aclnnInnerMatmulAllReduceGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                             const aclTensor *bias, const aclTensor *x3,
                                                             const aclTensor *antiquantScale,
                                                             const aclTensor *antiquantOffset,
                                                             const aclTensor *dequantScale,
                                                             const aclTensor *pertokenScale, const aclTensor *commQuantScale1,
                                                             const aclTensor *commQuantScale2,
                                                             const char *group, const char *reduceOp, bool transposeX1,
                                                             bool transposeX2, int64_t commTurn,
                                                             int64_t antiquantGroupSize, const aclTensor *output,
                                                             uint64_t *workspaceSize, aclOpExecutor **executor);
extern aclnnStatus aclnnInnerMatmulAllReduce(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             const aclrtStream stream);
extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void *executor, const size_t irIndex);

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT16, DataType::DT_BF16
};

// op::DataType不支持INT4，使用DateType
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_QUANT = {
  DataType::DT_INT8, DataType::DT_INT4
};

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale, const aclTensor* output) {
  OP_CHECK_NULL(x1, return false);
  OP_CHECK_NULL(x2, return false);
  OP_CHECK_NULL(scale, return false);
  OP_CHECK_NULL(output, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor *scale,
                            const aclTensor *offset, const aclTensor *x3, const aclTensor* output) {
  const auto dtypeSupportList = DTYPE_SUPPORT_LIST;
  // 检查x1、x2、bias、scale、offset、x3、output的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(x1, dtypeSupportList, return false);
  // 对于量化来说，x2只为INT8/INT4
  OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST_QUANT, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(scale, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(output, dtypeSupportList, return false);
  // 检查x1和scale的数据类型是否相同
  OP_CHECK_DTYPE_NOT_SAME(x1, scale, return false);
  // 检查x1和output的数据类型是否相同
  OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);

  // 检查bias、offset、x3的数据类型是否在算子的支持列表内
  if (bias != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(bias, dtypeSupportList, return false);
    // 检查x1和bias的数据类型是否相同
    OP_CHECK_DTYPE_NOT_SAME(bias, x1, return false);
  }
  if (offset != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(offset, dtypeSupportList, return false);
    // 检查scale和offset的数据类型是否相同
    OP_CHECK_DTYPE_NOT_SAME(scale, offset, return false);
  }
  if (x3 != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(x3, dtypeSupportList, return false);
    // 检查x1和x3的数据类型是否相同
    OP_CHECK_DTYPE_NOT_SAME(x3, x1, return false);
  }

  return true;
}


// 检查传入的reduction数值是否在可选范围内
static bool CheckAttr(const char *reduceOp, int64_t streamMode, int64_t antiquantGroupSize, const aclTensor* x1) {
  if (strcmp(reduceOp, REDUCE_OP_SUM)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected reduceOp to be sum, but got %s.", reduceOp);
    return false;
  }
  if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected streamMode to be 1, but got %ld.", streamMode);
    return false;
  }

  const size_t x1Len = x1->GetViewShape().GetDimNum();
  int64_t kLen = x1->GetViewShape().GetDim(x1Len - 1);
  // if kLen equals to 0, no need to check antiquantGroupSize for per-group
  if (kLen == 0) {
    OP_LOGD("WeightQuantMatmulAllReduce, k value is equal to 0.");
    return true;
  }

  // antiquantGroupSize为默认值0或者antiquantGroupSize%32 == 0并且antiquantGroupSize在[32, min(k-1,INT_MAX)]范围内
  if (antiquantGroupSize == 0) {
      return true;
  }
  if (antiquantGroupSize % ANTIQUANT_GROUP_SIZE_MIN_VALUE != 0 ||
      antiquantGroupSize < ANTIQUANT_GROUP_SIZE_MIN_VALUE ||
      antiquantGroupSize > std::min(static_cast<int32_t>(kLen - 1), INT32_MAX)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "antiquantGroupSize should be in range [%ld, min(%ld, INT_MAX)], Actual is %ld.",
              ANTIQUANT_GROUP_SIZE_MIN_VALUE, (kLen - 1), antiquantGroupSize);
      return false;
  }
  return true;
}

static size_t CeilDiv(size_t x, size_t y) {
  if (y == 0) {
    return 0;
  } else {
    return ((x - 1) / y + 1);
  }
}

static bool IsAntiquantScaleShapeValid(const aclTensor* scale, const aclTensor* x1, const aclTensor* output,
                                       int64_t antiquantGroupSize) {
  const size_t scaleLen = scale->GetViewShape().GetDimNum();
  const size_t x1Len = x1->GetViewShape().GetDimNum();
  op::Shape outShape = output->GetViewShape();
  if (antiquantGroupSize == 0) {
    if ((scaleLen == DIM_LEN_ONE && (scale->GetViewShape().GetDim(0) == NUM_ONE ||
      scale->GetViewShape().GetDim(0) == outShape.GetDim(x1Len - 1))) ||
      (scaleLen == DIM_LEN_TWO && scale->GetViewShape().GetDim(0) == NUM_ONE &&
      scale->GetViewShape().GetDim(1) == outShape.GetDim(x1Len - 1))) {
      return true;
    }
    return false;
  }

  size_t kValue = CeilDiv(x1->GetViewShape().GetDim(x1Len - 1), antiquantGroupSize);
  if (antiquantGroupSize > 0) {
    if ((scaleLen == DIM_LEN_TWO && scale->GetViewShape().GetDim(0) == kValue &&
      scale->GetViewShape().GetDim(1) == outShape.GetDim(x1Len - 1))) {
      return true;
    }
    return false;
  }
  return false;
}

static bool IsWeightNZFormat(const aclTensor* x2) {
  auto format = ge::GetPrimaryFormat(x2->GetStorageFormat());
  if (format == Format::FORMAT_ND) {
    OP_LOGD("MatmulAllReduce, Recieved weight format is ACL_FORMAT_ND");
  }
  if (format == Format::FORMAT_FRACTAL_NZ) {
    OP_LOGD("MatmulAllReduce, Recieved weight format is ACL_FORMAT_FRACTAL_NZ");
    uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
    OP_LOGD("MatmulAllReduce, Shape is %lu", storageDimsNum);
    const uint64_t transdataNzDim = 4U;
    if (storageDimsNum == transdataNzDim) {
      return true;
    }
  }
  return false;
}

// 通过TransMatmulWeight接口预处理成NZ格式场景；
static bool IsAclnnPreTransposed(const aclTensor* x2) {
  auto viewFormat = ge::GetPrimaryFormat(x2->GetViewFormat());
  auto storageFormat = ge::GetPrimaryFormat(x2->GetStorageFormat());
  bool isAclnnPreTransposed = false;
  OP_LOGD("MatmulAllReduce, IsAclnnPreTransposed is %d", isAclnnPreTransposed);
  return isAclnnPreTransposed;
}

static void ProcessTransposedX2(const aclTensor* x2, int64_t& x2Dim0, int64_t& x2Dim1, ge::AscendString& x2ShapeStr) {
  op::Shape x2ViewShape = x2->GetViewShape();
  x2ViewShape.SetDim(0, x2Dim0);
  x2ViewShape.SetDim(1, x2Dim1);
  if (IsAclnnPreTransposed(x2)) {
    x2ShapeStr = op::ToString(x2ViewShape);
  }
  OP_LOGD("MatmulAllReduce, x2 view shape is %s", x2ShapeStr.GetString());
}

static bool CheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor *scale,
                       const aclTensor *offset, const aclTensor *x3, const aclTensor* output,
                       int64_t antiquantGroupSize) {
  bool isWeightNZ = IsWeightNZFormat(x2);
  OP_CHECK_MIN_DIM(x1, TWO_DIMS, return false);
  OP_CHECK_MAX_DIM(x1, THREE_DIMS, return false);
  OP_LOGD("MatmulAllReduce, CheckShape isWeightNZ is %d", isWeightNZ);
  if (isWeightNZ) {
    return true;
  }
  uint64_t weightDim = TWO_DIMS;
  OP_LOGD("MatmulAllReduce, CheckShape weightDim is %lu", weightDim);
  // x2的维度为2维,x1的维度为2D或者3D，output的维数与x1一致,weightNZ场景下，x2可能为4维
  OP_CHECK_WRONG_DIMENSION(x2, weightDim, return false);
  int64_t x2Dim0 = IsAclnnPreTransposed(x2) ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);
  int64_t x2Dim1 = IsAclnnPreTransposed(x2) ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
  auto x2ShapeStr = op::ToString(x2->GetViewShape());
  ProcessTransposedX2(x2, x2Dim0, x2Dim1, x2ShapeStr);
  // 仅支持x2矩阵转置，x1不支持转置, x1.GetDimNum(1) == x2.GetDimNum(0)
  const size_t x1Len = x1->GetViewShape().GetDimNum();
  OP_LOGD("MatmulAllReduce, CheckShape x1Len is %lu", x1Len);
  if (x1->GetViewShape().GetDim(x1Len - 1) != x2Dim0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected last dim of x1 to be equal to first dim of x2, but got x1 shape: %s, x2 shape: %s.",
            op::ToString(x1->GetViewShape()).GetString(), x2ShapeStr.GetString());
    return false;
  }

  // output的最后一维与x2的最后一维相同
  op::Shape outShape = x1->GetViewShape();
  outShape.SetDim(x1Len - 1, x2Dim1);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outShape, return false);

  // x1 shape [s,m,k], x2 shape [k,n], output shape [s,m,n], bias shape [n]
  if (bias != nullptr) {
    OP_CHECK_WRONG_DIMENSION(bias, ONE_DIM, return false);
    op::Shape biasShape;
    biasShape.AppendDim(output->GetViewShape().GetDim(x1Len - 1));
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(bias, biasShape, return false);
  }

  // x3 shape [s,m,n]
  if (x3 != nullptr) {
    OP_CHECK_SHAPE_NOT_EQUAL(x3, output, return false);
  }

  int64_t kValue = x1->GetViewShape().GetDim(x1Len - 1);
  // if kValue equals to 0, no need to check antiquantScale/antiquantOffset for per-group
  if (kValue == 0) {
    return true;
  }
  // scale和offset为per-tensor则shape为[1]，为per-channel则shape为[n]或者[1,n]
  OP_CHECK_MIN_DIM(scale, ONE_DIM, return false);
  OP_CHECK_MAX_DIM(scale, TWO_DIMS, return false);
  if (!IsAntiquantScaleShapeValid(scale, x1, output, antiquantGroupSize)) {
    if (antiquantGroupSize == 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "Expected shape of antiquantScale to be [1] or [n] or [1,n] for per-tensor/per-channel."
              " in this case, n is %ld, last dim of scale should be %ld or 1, "
              "but got scale shape: %s.", output->GetViewShape().GetDim(x1Len - 1),
              output->GetViewShape().GetDim(x1Len - 1), op::ToString(scale->GetViewShape()).GetString());
    }
    if (antiquantGroupSize != 0) {
      size_t kValueGroup = CeilDiv(x1->GetViewShape().GetDim(x1Len - 1), antiquantGroupSize);
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "Expected shape of antiquantScale to be [%zu,%ld] for per-group calculation, but got scale shape: %s.",
              kValueGroup, output->GetViewShape().GetDim(x1Len - 1), op::ToString(scale->GetViewShape()).GetString());
    }
    return false;
  }
  if (offset != nullptr) {
    OP_CHECK_SHAPE_NOT_EQUAL(offset, scale, return false);
  }

  return true;
}

static aclnnStatus CheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                               const aclTensor *antiquantScale, const aclTensor *antiquantOffset, const aclTensor *x3,
                               const char *reduceOp, int64_t streamMode, const aclTensor *output,
                               int64_t antiquantGroupSize) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(x1, x2, antiquantScale, output), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(x1, x2, bias, antiquantScale, antiquantOffset, x3, output), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查attr是否符合规则
  CHECK_RET(CheckAttr(reduceOp, streamMode, antiquantGroupSize, x1), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输出shape
  CHECK_RET(CheckShape(x1, x2, bias, antiquantScale, antiquantOffset, x3, output, antiquantGroupSize),
            ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static const aclTensor *CopyTensor(const aclTensor *x2) {
  uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
  std::vector<int64_t> storageDims(storageDimsNum);
  for (size_t i = 0; i < storageDimsNum; i++) {
    storageDims[i] = x2->GetStorageShape().GetDim(i);
  }
  OP_LOGD("WeightQuantMatmulAllReduce, CopyTensor storageDimsNum is %lu.", storageDimsNum);
  aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
  aclGetDataType(x2, &dataType);
  std::vector<int64_t> stride(storageDimsNum, 1);
  for (int64_t i = static_cast<int64_t>(storageDimsNum - DIM_LEN_TWO); i >= 0; i--) {
    stride[i] = storageDims[i + 1] * stride[i + 1];
  }
  auto offset = x2->GetViewOffset();
  aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
  auto stgFormat = ge::GetPrimaryFormat(x2->GetStorageFormat());
  if (stgFormat == Format::FORMAT_ND) {
    OP_LOGD("WeightQuantMatmulAllReduce, CopyTensor format is ACL_FORMAT_ND");
    format = aclFormat::ACL_FORMAT_ND;
  } else if (stgFormat == Format::FORMAT_FRACTAL_NZ) {
    format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
  }
  return aclCreateTensor(storageDims.data(), storageDimsNum, dataType, stride.data(),
      offset, format, storageDims.data(), storageDimsNum, x2->GetTensor()->GetAddr());
}

aclnnStatus aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                            const aclTensor *bias, const aclTensor *antiquantScale,
                                                            const aclTensor *antiquantOffset, const aclTensor *x3,
                                                            const char* group, const char *reduceOp, int64_t commTurn,
                                                            int64_t streamMode, int64_t antiquantGroupSize,
                                                            const aclTensor *output, uint64_t *workspaceSize,
                                                            aclOpExecutor **executor) {
  uint64_t timeStamp = NnopbaseMsprofSysTime();
  // 固定写法，参数检查
  auto retParam = CheckParams(x1, x2, bias, antiquantScale, antiquantOffset, x3, reduceOp, streamMode, output,
                              antiquantGroupSize);
  CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
  const size_t x1DimNum = x1->GetOriginalShape().GetDimNum();
  if (x1DimNum < 1 || x1DimNum > THREE_DIMS) {
    return ACLNN_ERR_INNER;
  }
  // 处理空tensor,x1,x2不为空，scale为空也报错，offset、bias、x3可选不做判断
  int kValue = x1->GetOriginalShape().GetDim(x1DimNum - 1);
  if ((x1->IsEmpty() || x2->IsEmpty() || antiquantScale->IsEmpty()) && (kValue != 0)) {
    // 根据实际支持情况补充
    OP_LOGD("WeightQuantMatmulAllReduce, dealing with empty tensor.");
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 目前不支持x1进行transpose
  bool transposeX1 = false;
  bool transposeX2 = IsTransposeLastTwoDims(x2) || IsAclnnPreTransposed(x2);
  aclTensor *pertokenScale = nullptr;
  aclTensor *commQuantScale1 = nullptr;
  aclTensor *commQuantScale2 = nullptr;
  aclTensor *dequantScale = nullptr;
  auto copyX2 = CopyTensor(x2);
  auto tempX2 = IsWeightNZFormat(x2) ? copyX2 : x2;
  aclnnStatus ret = aclnnInnerMatmulAllReduceGetWorkspaceSize(x1, tempX2, bias, x3, antiquantScale, antiquantOffset,
                                                              dequantScale, pertokenScale, commQuantScale1,
                                                              commQuantScale2, group, reduceOp, transposeX1,
                                                              transposeX2, commTurn, antiquantGroupSize,
                                                              output, workspaceSize, executor);
  OP_LOGD("WeightQuantMatmulAllReduce, aclnnMatmulAllReduceGetWorkspaceSize ret %d", ret);
#ifdef MC2_UT
  ret = 0;
#endif
  if (ret == 0) {
    if (NnopbaseDisableOptionalInput != nullptr) {
      NnopbaseDisableOptionalInput(*executor, 6U); // 6 is input irIndex
      NnopbaseDisableOptionalInput(*executor, 7U); // 7 is input irIndex
      NnopbaseDisableOptionalInput(*executor, 8U); // 8 is input irIndex
      NnopbaseDisableOptionalInput(*executor, 9U); // 9 is input irIndex
    }
  }
  OP_LOGD("WeightQuantMatmulAllReduce, end ret %d", ret);
  static NnopbaseDfxId dfxId = {0x60000, __func__, false};
  NnopbaseReportApiInfo(timeStamp, dfxId);
  return ret;
}

aclnnStatus aclnnWeightQuantMatmulAllReduce(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            const aclrtStream stream) {
  uint64_t timeStamp = NnopbaseMsprofSysTime();
  if (workspace == nullptr || workspaceSize == 0UL) {
    OP_LOGD("Skip the api for empty tensor, workspace addr %p, size %lu.", workspace, workspaceSize);
    return ACLNN_SUCCESS;
  }

  aclnnStatus ret = aclnnInnerMatmulAllReduce(workspace, workspaceSize, executor, stream);
  OP_LOGD("WeightQuantMatmulAllReduce, aclnnWeightQuantMatmulAllReduce ret %d", ret);
  if (ret != 0) {
    OP_LOGE(ACLNN_ERR_INNER, "WeightQuantMatmulAllReduce, This is an error in launch aicore");
    return ACLNN_ERR_INNER;
  }

  static NnopbaseDfxId dfxId = {0x60000, __func__, false};
  NnopbaseReportApiInfo(timeStamp, dfxId);
  return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif