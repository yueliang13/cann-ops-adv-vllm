/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmul_all_reduce_util.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "matmul_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

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
extern aclnnStatus aclnnInnerMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor *x1,
                                                                       const aclTensor *x2,
                                                                       const aclTensor *bias,
                                                                       const aclTensor *residual,
                                                                       const aclTensor *gamma,
                                                                       const aclTensor *antiquantScale,
                                                                       const aclTensor *antiquantOffset,
                                                                       const aclTensor *dequantScale,
                                                                       const char *group,
                                                                       const char *reduceOp,
                                                                       bool transposeX1,
                                                                       bool transposeX2,
                                                                       int64_t commTurn,
                                                                       int64_t antiquantGroupSize,
                                                                       double epsilon,
                                                                       const aclTensor *y,
                                                                       const aclTensor *normOut,
                                                                       uint64_t *workspaceSize,
                                                                       aclOpExecutor **executor);
extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void *executor, const size_t irIndex);

// 根据API定义，需要列出所能支持的所有dtype
const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_310P = {
  op::DataType::DT_FLOAT16
};

// quant
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_BIAS = {
  op::DataType::DT_INT32
};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_QUANT = {
  op::DataType::DT_INT8
};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DEQUANT = {
  op::DataType::DT_UINT64, op::DataType::DT_INT64, op::DataType::DT_BF16, op::DataType::DT_FLOAT
};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_PERTOKEN = {
  op::DataType::DT_FLOAT
};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DEQUANT_310P = {
  op::DataType::DT_UINT64, op::DataType::DT_INT64
};

aclnnStatus MatmulAllReduceCheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *x3,
                                        const aclTensor *bias, const char *reduceOp, int64_t streamMode,
                                        const aclTensor *output) {
  // 1. 检查参数是否为空指针
  CHECK_RET(MatmulAllReduceCheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(MatmulAllReduceCheckDtypeValid(x1, x2, x3, bias, output), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查attr是否符合规则
  CHECK_RET(MatmulAllReduceCheckAttr(reduceOp, streamMode), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输出shape
  CHECK_RET(MatmulAllReduceCheckShape(x1, x2, x3, bias, output), ACLNN_ERR_PARAM_INVALID);

  if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND310P && x3 != nullptr) {
    return ACLNN_ERR_PARAM_INVALID;
  }

  return ACLNN_SUCCESS;
}

// 检查入参是否为nullptr
bool MatmulAllReduceCheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* output) {
  OP_CHECK_NULL(x1, return false);
  OP_CHECK_NULL(x2, return false);
  OP_CHECK_NULL(output, return false);
  return true;
}

bool MatmulAllReduceCheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3,
                                    const aclTensor* bias, const aclTensor* output) {
  const auto& dtypeSupportList =
      op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND310P ?
      DTYPE_SUPPORT_LIST_310P : DTYPE_SUPPORT_LIST;
  // 检查x1、x2、bias、output的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(x1, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(x2, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(output, dtypeSupportList, return false);
  // 检查bias的数据类型是否在算子的支持列表内
  if (bias != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(bias, dtypeSupportList, return false);
  }

  if (x3 != nullptr) {
    // 检查x3的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x3, dtypeSupportList, return false);
    // 检查x1和x3的数据类型是否相同
    OP_CHECK_DTYPE_NOT_SAME(x1, x3, return false);
  }
  // 检查x1和x2的数据类型是否相同
  OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
  // 检查x1和output的数据类型是否相同
  OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);
  // 检查output和bias的数据类型是否相同
  if (bias != nullptr) {
    OP_CHECK_DTYPE_NOT_SAME(bias, x1, return false);
  }
  return true;
}

// 检查传入的reduction数值是否在可选范围内
bool MatmulAllReduceCheckAttr(const char *reduceOp, int64_t streamMode) {
  if (strcmp(reduceOp, REDUCE_OP_SUM)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected reduceOp to be sum, but got %s.", reduceOp);
    return false;
  }
  if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected streamMode to be 1, but got %ld.", streamMode);
    return false;
  }
  return true;
}

bool MatmulAllReduceCheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3,
                                const aclTensor* bias, const aclTensor* output) {
  bool isWeightNZ = MatmulAllReduceIsWeightNZFormat(x2);
  uint64_t weightDim = isWeightNZ ? FOUR_DIMS : TWO_DIMS;
  // x2的维度为2维,x1的维度为2D或者3D，output的维数与x1一致,weightNZ场景下，x2可能为4维
  OP_CHECK_WRONG_DIMENSION(x2, weightDim, return false);
  OP_CHECK_MIN_DIM(x1, TWO_DIMS, return false);
  OP_CHECK_MAX_DIM(x1, THREE_DIMS, return false);
  if (isWeightNZ) {
    return true;
  }
  // 仅支持x2矩阵转置，x1不支持转置, x1.GetDimNum(1) == x2.GetDimNum(0)
  const size_t x1_len = x1->GetViewShape().GetDimNum();
  if (x1->GetViewShape().GetDim(x1_len - 1) != x2->GetViewShape().GetDim(0)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected last dim of x1 to be equal to first dim of x2, but got x1 shape: %s, x2 shape: %s.",
            op::ToString(x1->GetViewShape()).GetString(), op::ToString(x2->GetViewShape()).GetString());
    return false;
  }

  // output的最后一维与x2的最后一维相同
  op::Shape outShape = x1->GetViewShape();
  outShape.SetDim(x1_len - 1, x2->GetViewShape().GetDim(1));
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outShape, return false);

  if (x3 != nullptr) {
    OP_CHECK_SHAPE_NOT_EQUAL(output, x3, return false);
  }

  // x1 shape [s,m,k], x2 shape [k,n], output shape [s,m,n], bias shape [n]
  if (bias != nullptr) {
    OP_CHECK_WRONG_DIMENSION(bias, ONE_DIM, return false);
    op::Shape biasShape;
    biasShape.AppendDim(output->GetViewShape().GetDim(x1_len - 1));
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(bias, biasShape, return false);
  }
  return true;
}

// 全量化场景
aclnnStatus QuantMatmulAllReduceCheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                            const aclTensor *dequantScale, const aclTensor *pertokenScale,
                                            const aclTensor *x3, const char *reduceOp, int64_t streamMode,
                                            const aclTensor *output) {
  // 1. 检查参数是否为空指针
  CHECK_RET(MatmulAllReduceCheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);
  OP_CHECK_NULL(dequantScale, return ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(QuantMatmulAllReduceCheckDtypeValid(x1, x2, bias, dequantScale, pertokenScale, x3, output),
            ACLNN_ERR_PARAM_INVALID);

  // 3. 检查attr是否符合规则
  CHECK_RET(MatmulAllReduceCheckAttr(reduceOp, streamMode), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输出shape
  CHECK_RET(QuantMatmulAllReduceCheckShape(x1, x2, bias, dequantScale, pertokenScale, x3, output),
            ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

bool QuantMatmulAllReduceCheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                          const aclTensor *dequantScale, const aclTensor *pertokenScale,
                                          const aclTensor *x3, const aclTensor* output) {
  const auto& dequantDtypeSupport =
    op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND310P ?
    DTYPE_SUPPORT_LIST_DEQUANT_310P : DTYPE_SUPPORT_LIST_DEQUANT;

  const auto& outDtypeSupport =
    op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND310P ?
    DTYPE_SUPPORT_LIST_310P : DTYPE_SUPPORT_LIST;

  // 检查x1、x2、bias、scale、offset、x3、output的数据类型是否在算子的支持列表内
  // 对于量化来说，x1,x2只为INT8
  OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST_QUANT, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST_QUANT, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(dequantScale, dequantDtypeSupport, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(output, outDtypeSupport, return false);
  // 检查bias、offset、x3的数据类型是否在算子的支持列表内
  if (bias != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST_BIAS, return false);
  }
  if (x3 != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(x3, outDtypeSupport, return false);
    // 检查x3和output的数据类型是否相同
    OP_CHECK_DTYPE_NOT_SAME(x3, output, return false);
  }
  // 检查x1和x2的数据类型是否相同
  OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
  // BF16场景，dequantScale的dtype与output一致，为BF16
  if (output->GetDataType() == op::DataType::DT_BF16 || dequantScale->GetDataType() == op::DataType::DT_BF16) {
    OP_CHECK_DTYPE_NOT_SAME(dequantScale, output, return false);
  }
  // pertokenScale必须为float32
  if (pertokenScale != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(pertokenScale, DTYPE_SUPPORT_LIST_PERTOKEN, return false);
    // FP16场景，dequantScale的dtype与pertokenScale一致，为float32
    if (output->GetDataType() == op::DataType::DT_FLOAT16) {
      OP_CHECK_DTYPE_NOT_SAME(dequantScale, pertokenScale, return false);
    }
  }
  return true;
}

bool MatmulAllReduceIsWeightNZFormat(const aclTensor* x2) {
  aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
  aclGetFormat(x2, &format);
  if (format == aclFormat::ACL_FORMAT_ND) {
    OP_LOGD("MatmulAllReduce, Recieved weight format is ACL_FORMAT_ND");
  }
  if (format == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
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

bool QuantMatmulAllReduceIsWeightNZFormat(const aclTensor* x2) {
  auto format = ge::GetPrimaryFormat(x2->GetStorageFormat());
  OP_LOGD("MatmulAllReduce, Recieved weight format is %d", format);
  if (format == Format::FORMAT_FRACTAL_NZ) {
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
bool QuantMatmulAllReduceIsAclnnPreTransposed(const aclTensor* x2) {
  auto viewFormat = ge::GetPrimaryFormat(x2->GetViewFormat());
  auto storageFormat = ge::GetPrimaryFormat(x2->GetStorageFormat());
  bool isAclnnPreTransposed = op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND310P &&
    viewFormat == Format::FORMAT_ND && storageFormat == Format::FORMAT_FRACTAL_NZ;
  OP_LOGD("MatmulAllReduce, IsAclnnPreTransposed is %d", isAclnnPreTransposed);
  return isAclnnPreTransposed;
}

void QuantMatmulAllReduceProcessTransposedX2(const aclTensor* x2, uint64_t& x2Dim0, uint64_t& x2Dim1,
                                              ge::AscendString& x2ShapeStr) {
  op::Shape x2ViewShape = x2->GetViewShape();
  x2ViewShape.SetDim(0, x2Dim0);
  x2ViewShape.SetDim(1, x2Dim1);
  if (QuantMatmulAllReduceIsAclnnPreTransposed(x2)) {
    x2ShapeStr = op::ToString(x2ViewShape);
  }
  OP_LOGD("MatmulAllReduce, x2 view shape is %s", x2ShapeStr.GetString());
}

bool QuantMatmulAllReduceCheckPertokenScaleShape(const aclTensor *pertokenScale, const aclTensor* x1,
                                                  const size_t x1Len)
{
  // pertokenScale的shape为[m]
  if (pertokenScale != nullptr) {
    OP_CHECK_MIN_DIM(pertokenScale, ONE_DIM, return false);
    OP_CHECK_MAX_DIM(pertokenScale, ONE_DIM, return false);
    const size_t pertokenScaleLen = pertokenScale->GetViewShape().GetDimNum();
    size_t x1_m = 1;
    for (size_t dim = 0; dim < x1Len - 1; dim++) {
      x1_m *= x1->GetViewShape().GetDim(dim);
    }
    if (pertokenScaleLen != DIM_LEN_ONE || static_cast<size_t>(pertokenScale->GetViewShape().GetDim(0)) != x1_m) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "Expected pertokenScale be [%lu], but got pertokenScale shape: %s.", x1_m,
              op::ToString(pertokenScale->GetViewShape()).GetString());
      return false;
    }
  }
  return true;
}

bool QuantMatmulAllReduceCheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                    const aclTensor *dequantScale, const aclTensor *pertokenScale,
                                    const aclTensor *x3, const aclTensor* output) {
  OP_CHECK_MIN_DIM(x1, TWO_DIMS, return false);
  OP_CHECK_MAX_DIM(x1, THREE_DIMS, return false);
  if (QuantMatmulAllReduceIsWeightNZFormat(x2)) {
    return true;
  }
  // x2的维度为2维,x1的维度为2D或者3D，output的维数与x1一致,weightNZ场景下，x2可能为4维
  OP_CHECK_WRONG_DIMENSION(x2, TWO_DIMS, return false);
  uint64_t x2Dim0 = QuantMatmulAllReduceIsAclnnPreTransposed(x2) ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);
  uint64_t x2Dim1 = QuantMatmulAllReduceIsAclnnPreTransposed(x2) ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
  auto x2ShapeStr = op::ToString(x2->GetViewShape());
  QuantMatmulAllReduceProcessTransposedX2(x2, x2Dim0, x2Dim1, x2ShapeStr);
  // 仅支持x2矩阵转置，x1不支持转置, x1.GetDimNum(1) == x2.GetDimNum(0)
  const size_t x1Len = x1->GetViewShape().GetDimNum();
  if (x1Len < 1 || x1Len > THREE_DIMS) {
    return false;
  }
  if (static_cast<uint64_t>(x1->GetViewShape().GetDim(x1Len - 1)) != x2Dim0) {
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
  // scale和offset为per-tensor则shape为[1]，为per-channel则shape为[n]或者[1,n]
  OP_CHECK_MIN_DIM(dequantScale, ONE_DIM, return false);
  OP_CHECK_MAX_DIM(dequantScale, TWO_DIMS, return false);
  const size_t scaleLen = dequantScale->GetViewShape().GetDimNum();
  if (!(scaleLen == DIM_LEN_ONE && (dequantScale->GetViewShape().GetDim(0) == NUM_ONE ||
      dequantScale->GetViewShape().GetDim(0) == outShape.GetDim(x1Len - 1))) &&
      !(scaleLen == DIM_LEN_TWO && dequantScale->GetViewShape().GetDim(0) == NUM_ONE &&
      dequantScale->GetViewShape().GetDim(1) == outShape.GetDim(x1Len - 1))) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected dequantScale be [1] or [n] or [1,n], last dim of dequantScale should be %ld or 1, \
            but got dequantScale shape: %s.", output->GetViewShape().GetDim(x1Len - 1),
            op::ToString(dequantScale->GetViewShape()).GetString());
    return false;
  }
  if (!QuantMatmulAllReduceCheckPertokenScaleShape(pertokenScale, x1, x1Len)) {
    return false;
  }
  return true;
}

aclTensor *QuantMatmulAllReduceCopyTensor(const aclTensor *x2)
{
  uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
  std::vector<int64_t> storageDims(storageDimsNum);
  for (size_t i = 0; i < storageDimsNum; i++) {
    storageDims[i] = x2->GetStorageShape().GetDim(i);
  }
  OP_LOGD("MatmulAllReduce, CopyTensor storageDimsNum is %lu.", storageDimsNum);
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
    OP_LOGD("MatmulAllReduce, CopyTensor format is ACL_FORMAT_ND");
    format = aclFormat::ACL_FORMAT_ND;
  } else if (stgFormat == Format::FORMAT_FRACTAL_NZ) {
    format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
  }
  return aclCreateTensor(storageDims.data(), storageDimsNum, dataType, stride.data(),
      offset, format, storageDims.data(), storageDimsNum, x2->GetTensor()->GetAddr());
}

bool ArnCheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* residual, const aclTensor* gamma)
{
  OP_CHECK_NULL(x1, return false);
  OP_CHECK_NULL(x2, return false);
  OP_CHECK_NULL(residual, return false);
  OP_CHECK_NULL(gamma, return false);
  return true;
}

bool ArnCheckShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *residual)
{
  const size_t x1_len = x1->GetViewShape().GetDimNum();
  const size_t x2_len = x2->GetViewShape().GetDimNum();
  const size_t residual_len = residual->GetViewShape().GetDimNum();
  OP_LOGI("MatmulAllReduceAddRmsNorm, x1 shape: %s, x2 shape: %s, residual shape: %s.",
          op::ToString(x1->GetViewShape()).GetString(), op::ToString(x2->GetViewShape()).GetString(),
          op::ToString(residual->GetViewShape()).GetString());
  if (x1_len < NUM_ONE || x2_len < NUM_ONE || residual_len < NUM_ONE) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "dim of x1, x2 and residual should greater than 1.");
    return false;
  }
  if (x1->GetViewShape().GetDim(x1_len - 1) != x2->GetViewShape().GetDim(0)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected last dim of x1 to be equal to first dim of x2, but got x1 shape: %s, x2 shape: %s.",
            op::ToString(x1->GetViewShape()).GetString(), op::ToString(x2->GetViewShape()).GetString());
    return false;
  }
  if (residual->GetViewShape().GetDim(residual_len - 1) != x2->GetViewShape().GetDim(x2_len - 1)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected last dim of residual to be equal to last dim of x2, but got residual shape: %s, x2 shape: %s.",
            op::ToString(residual->GetViewShape()).GetString(), op::ToString(x2->GetViewShape()).GetString());
    return false;
  }
  return true;
}

// quant Inner
aclnnStatus InnerQuantMatmulAllReduceGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                      const aclTensor *biasOptional, const aclTensor *x3Optional,
                                                      const aclTensor *dequant,
                                                      const aclTensor *pertokenScaleOptional, const char* group,
                                                      const char *reduceOp, int64_t commTurn,
                                                      const aclTensor *output,
                                                      uint64_t *workspaceSize, aclOpExecutor **executor)
{
  // 目前不支持x1进行transpose
  bool transposeX1 = false;
  bool transposeX2 = IsTransposeLastTwoDims(x2) || QuantMatmulAllReduceIsAclnnPreTransposed(x2);
  aclTensor *scale = nullptr;
  aclTensor *offset = nullptr;
  aclTensor *commQuantScale1Optional = nullptr;
  aclTensor *commQuantScale2Optional = nullptr;
  int64_t antiquantGroupSize = 0;
  auto tempX2 = x2;
  if (op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND310P &&
      QuantMatmulAllReduceIsWeightNZFormat(x2)) {
    tempX2 = QuantMatmulAllReduceCopyTensor(x2);
  }
  aclnnStatus ret = aclnnInnerMatmulAllReduceGetWorkspaceSize(x1, tempX2, biasOptional, x3Optional, scale, offset, dequant,
                                                              pertokenScaleOptional, commQuantScale1Optional,
                                                              commQuantScale2Optional, group, reduceOp, transposeX1,
                                                              transposeX2, commTurn, antiquantGroupSize, output,
                                                              workspaceSize, executor);

  OP_LOGI("Group %s, reduce op %s, trans flag %d %d, ret %d.", group, reduceOp, transposeX1, transposeX2, ret);
#ifdef MC2_UT
  ret = 0;
#endif
  if (ret == 0) {
    if (NnopbaseDisableOptionalInput != nullptr) {
      NnopbaseDisableOptionalInput(*executor, 4U); // 4 is input irIndex
      NnopbaseDisableOptionalInput(*executor, 5U); // 5 is input irIndex
      if (pertokenScaleOptional == nullptr) {
        NnopbaseDisableOptionalInput(*executor, 7U); // 7 is input irIndex
      }
      NnopbaseDisableOptionalInput(*executor, 8U); // 8 is input irIndex
      NnopbaseDisableOptionalInput(*executor, 9U); // 9 is input irIndex
    }
  }
  return ret;
}

// AddRmsNorm Inner
aclnnStatus InnerMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                            const aclTensor *bias, const aclTensor *antiquantScale,
                                                            const aclTensor *antiquantOffset,
                                                            const aclTensor *dequant, const aclTensor *residual,
                                                            const aclTensor *gamma, double epsilon,
                                                            const char *group, const char *reduceOp,
                                                            int64_t commTurn, int64_t antiquantGroupSize,
                                                            const aclTensor *y, const aclTensor *normOut,
                                                            uint64_t *workspaceSize, aclOpExecutor **executor)
{
  // 目前不支持x1进行transpose
  bool transposeX1 = false;
  bool transposeX2 = IsTransposeLastTwoDims(x2);
  aclnnStatus ret = aclnnInnerMatmulAllReduceAddRmsNormGetWorkspaceSize(
      x1, x2, bias, residual, gamma, antiquantScale, antiquantOffset, dequant, group, reduceOp, transposeX1,
      transposeX2, commTurn, antiquantGroupSize, epsilon, y, normOut, workspaceSize, executor);
  OP_LOGI("Group %s, reduce op %s, trans flag %d %d, epsilon %lf, ret %d.", group, reduceOp, transposeX1, transposeX2,
          epsilon, ret);
#ifdef MC2_UT
  ret = 0;
#endif
  if (ret == OK) {
    if (NnopbaseDisableOptionalInput != nullptr) {
      if (antiquantScale == nullptr) {
        NnopbaseDisableOptionalInput(*executor, 5U); // 5 is input irIndex
        NnopbaseDisableOptionalInput(*executor, 6U); // 6 is input irIndex
      }
      NnopbaseDisableOptionalInput(*executor, 7U); // 7 is input irIndex
    }
  }
  return ret;
}

#ifdef __cplusplus
}
#endif