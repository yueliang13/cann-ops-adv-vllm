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
 * \file matmul_util.cpp
 * \brief
 */

#include "matmul_util.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "op_api_def.h"

using namespace op;
static const int64_t SPLIT_K_MULTI = 8;
static const int64_t MKN_MAX = 8000000000;
static const int64_t MN_MULTI = 50;
static const size_t MM_DIM = 2;
static const int32_t INNER_AXIS = 1;
static const int32_t OUTER_AXIS = 2;
static const int64_t DIM_EQUAL_ONE = 1;
static const uint64_t SMALL_SHAPE_LIMIT = 524288UL;
static const uint32_t kDeqScaleMul = 0xFFFFE000;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT = {DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> V100_DTYPE_SUPPORT = {DataType::DT_FLOAT16, DataType::DT_BF16};

op::Shape SwapLastTwoDimValue(const op::Shape tensorShape)
{
  op::Shape swapedShape = tensorShape;
  int64_t dimNum = tensorShape.GetDimNum();
  int64_t lastDim = tensorShape.GetDim(dimNum - 1);
  // dimNum - 1, 这里1指的是取最后一维的dim值。dimNum - 2, 这里2指的是取倒数第二维的dim值
  swapedShape.SetDim(dimNum - 1, tensorShape.GetDim(dimNum - 2));
  // dimNum - 2, 这里2指的是取倒数第二维的dim值
  swapedShape.SetDim(dimNum - 2, lastDim);
  return swapedShape;
}

#ifdef __cplusplus
extern "C" {
#endif

bool IsInputSupportFp32() {
  if (op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910B &&
      op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910_93 &&
      op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910_95) {
    return false;
  }
  return true;
}

bool CheckBatchDimBroadcast(size_t batch1DimNum, size_t batch2DimNum, const op::Shape& batch1, const op::Shape& batch2) {
    size_t batchIndex = MM_DIM;
    while (batch1DimNum > batchIndex && batch2DimNum > batchIndex) {
        if (batch1[batch1DimNum - batchIndex - 1] != 1 && batch2[batch1DimNum - batchIndex - 1] != 1 &&
            batch1[batch1DimNum - batchIndex - 1] != batch2[batch1DimNum - batchIndex - 1]) {
            return false;
        }
        batchIndex++;
    }
    return true;
}

// bmm 相对于 mm 取坐标需偏移
int64_t GetOffSet(int64_t DimNum) {
  int64_t rightMove = 0;
  // bmm DimNum 为 3, mm DimNum 为 2 ，bmm需要相对于mm向后偏移一位取行列值，默认rightMove为 0
  rightMove = DimNum == 3 ? 1 : 0;
  return rightMove;
}

// 检查单Tensor是否为支持带bias的mm的dtype
static inline bool CheckDtypeSupport(const aclTensor *tensor) {
  if (!IsInputSupportFp32()) {
    auto iter = std::find(V100_DTYPE_SUPPORT.begin(), V100_DTYPE_SUPPORT.end(), tensor->GetDataType());
    return iter != V100_DTYPE_SUPPORT.end();
  }
  auto iter = std::find(DTYPE_SUPPORT.begin(), DTYPE_SUPPORT.end(), tensor->GetDataType());
  return iter != DTYPE_SUPPORT.end();
}

// 检查是否为支持带bias的mm的dtype
static inline bool CheckDtypeSupportBias(const aclTensor *self, const aclTensor *mat1, const aclTensor *mat2) {
  bool matMulDtypeCorrect = CheckDtypeSupport(mat1) && CheckDtypeSupport(mat2);
  if (mat1->GetDataType() == DataType::DT_BF16) {
    return matMulDtypeCorrect &&
           (self->GetDataType() == DataType::DT_BF16 || self->GetDataType() == DataType::DT_FLOAT);
  }
  return CheckDtypeSupport(self) && matMulDtypeCorrect;
}

// 如果beta==1 && alpha == 1 && self.shape[0] == mat2.shape[1] && 不属于切k，直接走matmul的bias模式
bool NeedToConvertBias(const aclTensor *self, const aclTensor *mat1, const aclTensor *mat2,
                       const aclScalar *beta, const aclScalar *alpha) {
  int64_t mat1DimNum = static_cast<int64_t>(mat1->GetViewShape().GetDimNum());
  // rightMove to distinguish different shape of mm and bmm
  int64_t rightMove = 0;
  rightMove = GetOffSet(mat1DimNum);

  bool isSplitK = false;
  if (op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910B &&
      op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910_93 &&
      op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910_95) {
    isSplitK = IsSplitk(mat1, mat2, mat1->GetDataType(), mat2->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
  }
  op::Shape selfShape = self->GetViewShape();
  op::Shape mat2Shape = mat2->GetViewShape();
  int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
  bool canBeBiasFlag = false;
  // bmm (DimNum==3) only apply the case of batch == 1
  bool batchIsOne = !(mat1DimNum == 3 && mat1->GetViewShape().GetDim(0) != 1);

  if (selfDimNum == 1) {
    canBeBiasFlag = (mat2->GetViewShape().GetDim(1 + rightMove) == self->GetViewShape().GetDim(0)) &&
                     CheckDtypeSupportBias(self, mat1, mat2) && batchIsOne;
    // When input tensor is a 2 dimentional tensor
  } else if (selfDimNum == 2) {
    canBeBiasFlag = (selfShape.GetDim(0) == 1) && (selfShape.GetDim(1) == mat2Shape.GetDim(1 + rightMove)) &&
                     CheckDtypeSupportBias(self, mat1, mat2) && batchIsOne;
  }
  OP_LOGI("Current Shape's canBeBiasFlag = %ld", static_cast<int64_t>(canBeBiasFlag));
  return (std::abs(alpha->ToFloat() - 1.0f) <= std::numeric_limits<float>::epsilon()) &&
         (std::abs(beta->ToFloat() - 1.0f) <= std::numeric_limits<float>::epsilon()) &&
         !isSplitK && canBeBiasFlag;
}

// Nz fp16 in fp32 out experimental rules
bool GetNzSplitKFlag(const aclTensor *self, const aclTensor *mat2, const Format selfSuppFormat, const Format outSuppFormat) {
  if ((selfSuppFormat == Format::FORMAT_ND) && (outSuppFormat == Format::FORMAT_ND)) {
    return true;
  }
  op::Shape selfShape = self->GetViewShape();
  op::Shape mat2Shape = mat2->GetViewShape();
  int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
  // rightMove to distinguish different shape of mm and bmm
  int64_t rightMove = 0;
  rightMove = GetOffSet(selfDimNum);

  int64_t m = selfShape.GetDim(rightMove);
  int64_t k = selfShape.GetDim(rightMove + 1);
  int64_t n = mat2Shape.GetDim(rightMove + 1);
  bool mn_multi = m > n ? m < (MN_MULTI * n) : n < (MN_MULTI * m);
  return (m * n * k < MKN_MAX) && mn_multi;
}

bool IsSplitk(const aclTensor *self, const aclTensor *mat2, const DataType selfDtype, const DataType mat2Dtype, const Format selfSuppFormat, const Format outSuppFormat) {
  op::Shape selfShape = self->GetViewShape();
  op::Shape mat2Shape = mat2->GetViewShape();
  int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
  // rightMove to distinguish different shape of mm and bmm
  int64_t rightMove = 0;
  rightMove = GetOffSet(selfDimNum);
  bool NzSplitKFlag = true;
  // only apply on mm now
  if (!rightMove) {
    NzSplitKFlag = GetNzSplitKFlag(self, mat2, selfSuppFormat, outSuppFormat);
  }

  int64_t k_dim = selfShape.GetDim(1 + rightMove);
  bool dtype_correct = (selfDtype == DataType::DT_FLOAT16) && (mat2Dtype == DataType::DT_FLOAT16);
  return dtype_correct && k_dim >= SPLIT_K_MULTI * std::max(selfShape.GetDim(rightMove), mat2Shape.GetDim(1 + rightMove)) && NzSplitKFlag;
}

bool IsFormatSupportNd(const aclTensor *self, const aclTensor *mat2) {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
    return true;
  }
  if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) {
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    int64_t dimNum = selfShape.GetDimNum();
    auto isAligin = [selfShape, mat2Shape, dimNum]() {
      return (!(static_cast<uint64_t>(selfShape.GetDim(dimNum - 2)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(selfShape.GetDim(dimNum - 1)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(mat2Shape.GetDim(dimNum - 2)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(mat2Shape.GetDim(dimNum - 1)) & 0x0000000F));
    };
    if (isAligin() && self->GetDataType() == op::DataType::DT_FLOAT16) {
      return true;
    }
    return false;
  }
  if ((self->GetDataType() == DataType::DT_FLOAT16 && mat2->GetDataType() == DataType::DT_FLOAT16) ||
      (self->GetDataType() == DataType::DT_BF16 && mat2->GetDataType() == DataType::DT_BF16)) {
    return IsNdToNzOnTheFly(self, mat2);
  }
  return true;
}

bool IsSupportNzNzNd(const aclTensor* self, const aclTensor* mat2) {
  op::Shape selfShape = self->GetViewShape();
  op::Shape mat2Shape = mat2->GetViewShape();
  int64_t dimNum = selfShape.GetDimNum();
  auto isNAligin = [mat2Shape, dimNum]() { return (!(static_cast<uint64_t>(mat2Shape.GetDim(dimNum - 1)) & 0x0000000F)); };
  if (isNAligin() && self->GetDataType() == op::DataType::DT_FLOAT16) {
    return true;
  }
  return false;
}

bool IsNdToNzOnTheFly(const aclTensor *self, const aclTensor *mat2) {
  uint64_t kInnerAxisMinLimit = 128U;
  uint64_t kInnerAxisMaxLimit = 65535U;
  uint64_t kAxisLengthOne = 1U;
  // 如果self或mat2的维度数量小于2，则不符合判断是否16对齐的条件，返回失败
  if (self->GetViewShape().GetDimNum() < 2 || mat2->GetViewShape().GetDimNum() < 2) {
    return false;
  }
  bool isTransposeSelf = IsTransposeLastTwoDims(self);
  bool isTransposeMat2 = IsTransposeLastTwoDims(mat2);
  uint64_t selfInnerAxis = isTransposeSelf ?
                             static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 2)) :
                             static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 1));
  uint64_t mat2InnerAxis = isTransposeMat2 ?
                             static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 2)) :
                             static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 1));

  uint64_t selfOuterAxis = isTransposeSelf ?
                             static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 1)) :
                             static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 2));
  uint64_t mat2OuterAxis = isTransposeMat2 ?
                             static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 1)) :
                             static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 2));
  uint64_t mAxis = static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 2)); //倒数第2维
  uint64_t kAxis = static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 1));
  uint64_t nAxis = static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 1));
  if (selfInnerAxis * selfOuterAxis <= kInnerAxisMaxLimit &&
      mat2InnerAxis * mat2OuterAxis <= kInnerAxisMaxLimit) {
    // too small tensor size
    return true;
  }
  OP_LOGD("Check IsNdToNzOnTheFly, if k=1 scenerio then remains ND.");
  if (kAxis == kAxisLengthOne) {
    return true;
  }

  if (IsSmallMNMultiSplitK(mAxis, kAxis, nAxis, isTransposeSelf, isTransposeMat2)) {
    OP_LOGD("Hit small mn multi split k.");
    return true;
  }

  return ((selfInnerAxis >= kInnerAxisMinLimit && selfInnerAxis <= kInnerAxisMaxLimit) ||
          (selfInnerAxis < kInnerAxisMinLimit && ((selfInnerAxis & 0xF) == 0))) &&
          ((mat2InnerAxis >= kInnerAxisMinLimit && mat2InnerAxis <= kInnerAxisMaxLimit) ||
          (mat2InnerAxis < kInnerAxisMinLimit && ((mat2InnerAxis & 0xF) == 0)));
}

bool IsSmallMNMultiSplitK(const uint64_t mDim, const uint64_t kDim, const uint64_t nDim,
                          const bool transposeX1, const bool transposeX2) {
  constexpr uint64_t align128 = 128;
  constexpr uint64_t numTwo = 2;
  constexpr uint64_t smallMNsplitKThres = 15000;
  bool kIsEnoughMultiCore = kDim >= smallMNsplitKThres;
  bool mnIsNotEnoughCore = (std::ceil(mDim / align128) * std::ceil(nDim / align128) <
                            static_cast<int64_t>(GetCurrentPlatformInfo().GetCubeCoreNum() / numTwo));
  // M/N轴在内轴的场景切m/n不影响MTE2搬运效率，M/N可以切小保证多核能开启，属于cube_bound场景
  return kIsEnoughMultiCore && mnIsNotEnoughCore && !(!transposeX1 && transposeX2);
}

bool IsTransposeLastTwoDims(const aclTensor *tensor) {
  // 当输入tensor的shape小于2或者大于6的时候，返回错误
  if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
    return false;
  }
  int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
  int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
  // BMM 场景下，Batch维度的stride需要等于 N, D 的乘积
  if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
    int64_t tmpNxD = tensor->GetViewShape().GetDim(dim1) * tensor->GetViewShape().GetDim(dim2);
    // 多batch连续，3是batch索引
    for (int64_t batchDim = tensor->GetViewShape().GetDimNum() - 3; batchDim >= 0; batchDim--) {
    if (tensor->GetViewStrides()[batchDim] != tmpNxD) {
        return false;
      }
      tmpNxD *= tensor->GetViewShape().GetDim(batchDim);
    }
    if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
      return false;
    }
    return true;
  }
  return false;
}

aclnnStatus SetMmSupportDType(const aclTensor *self, const aclTensor *mat2, MmOpInfo &mmOpInfo,
                              int8_t cubeMathType) {
  bool dtypeMismatch = mmOpInfo.ori_info.self_dtype != mmOpInfo.ori_info.mat2_dtype;
  bool tensorFloat = mmOpInfo.ori_info.self_dtype == DataType::DT_FLOAT ||
                     mmOpInfo.ori_info.mat2_dtype == DataType::DT_FLOAT;
  bool tensorBfloat16 = mmOpInfo.ori_info.self_dtype == DataType::DT_BF16 ||
                        mmOpInfo.ori_info.mat2_dtype == DataType::DT_BF16;

  if (!IsInputSupportFp32()) {
    mmOpInfo.support_info.self_dtype = DataType::DT_FLOAT16;
    mmOpInfo.support_info.mat2_dtype = DataType::DT_FLOAT16;
  } else if (IsInputSupportFp32() && cubeMathType == USE_FP16 && (!tensorBfloat16)) {
    mmOpInfo.support_info.self_dtype = DataType::DT_FLOAT16;
    mmOpInfo.support_info.mat2_dtype = DataType::DT_FLOAT16;
    mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT16;
  } else if (IsInputSupportFp32() && dtypeMismatch && (tensorFloat || tensorBfloat16)) {
    mmOpInfo.support_info.self_dtype = DataType::DT_FLOAT;
    mmOpInfo.support_info.mat2_dtype = DataType::DT_FLOAT;
    mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT;
  }
  return ACLNN_SUCCESS;
}

aclnnStatus SetMmSupportFormat(const aclTensor* self, const aclTensor* mat2, MmOpInfo& mmOpInfo) {
  if (IsFormatSupportNd(self, mat2)) {
    OP_LOGD("Matmul support NDNDND");
    mmOpInfo.support_info.output_format = Format::FORMAT_ND;
    mmOpInfo.support_info.self_format = Format::FORMAT_ND;
    mmOpInfo.support_info.mat2_format = Format::FORMAT_ND;
  } else {
    OP_LOGD("Matmul do not support NDNDND");
    // if 310p and n%16==0
    bool is310p = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
    if (IsSupportNzNzNd(self, mat2) && is310p) {
      mmOpInfo.support_info.output_format = Format::FORMAT_ND;
      mmOpInfo.support_info.self_format = Format::FORMAT_FRACTAL_NZ;
      mmOpInfo.support_info.mat2_format = Format::FORMAT_FRACTAL_NZ;
      return ACLNN_SUCCESS;
    }
    mmOpInfo.support_info.output_format = Format::FORMAT_FRACTAL_NZ;
    mmOpInfo.support_info.self_format = Format::FORMAT_FRACTAL_NZ;
    mmOpInfo.support_info.mat2_format = Format::FORMAT_FRACTAL_NZ;
  }
  return ACLNN_SUCCESS;
}

aclnnStatus GetMmInfo(MmOpInfo mmOpInfo) {
  OP_LOGI(
    "Self tensor input's ori dtype = %s and format = %s; Mat2 tensor input's ori dtype = %s and format = %s;"
    "Output tensor's ori dtype = %s and ori format = %s;"
    "Self tensor input's Npu dtype = %s and Npu format = %s; Mat2 tensor input's Npu dtype = %s and Npuformat = %s;"
    "Output tensor's Npu dtype = %s and Npu format = %s.",
    op::ToString(mmOpInfo.ori_info.self_dtype).GetString(),
    op::ToString(mmOpInfo.ori_info.self_format).GetString(),
    op::ToString(mmOpInfo.ori_info.mat2_dtype).GetString(),
    op::ToString(mmOpInfo.ori_info.mat2_format).GetString(),
    op::ToString(mmOpInfo.ori_info.output_dtype).GetString(),
    op::ToString(mmOpInfo.ori_info.output_format).GetString(),
    op::ToString(mmOpInfo.support_info.self_dtype).GetString(),
    op::ToString(mmOpInfo.support_info.self_format).GetString(),
    op::ToString(mmOpInfo.support_info.mat2_dtype).GetString(),
    op::ToString(mmOpInfo.support_info.mat2_format).GetString(),
    op::ToString(mmOpInfo.support_info.output_dtype).GetString(),
    op::ToString(mmOpInfo.support_info.output_format).GetString());
  return ACLNN_SUCCESS;
}

aclIntArray* NeedTransPerm(const aclTensor *x, aclOpExecutor *executor) {
  op::Shape shape = x->GetViewShape();
  int64_t dimSize = x->GetViewShape().GetDimNum();
  std::vector<int64_t> valuePerm(dimSize, 0);
  for (int64_t i = 0; i < dimSize; i++) {
    valuePerm[i] = shape[i];
  }
  std::swap(valuePerm[dimSize - INNER_AXIS], valuePerm[dimSize - OUTER_AXIS]);
  return executor->AllocIntArray(valuePerm.data(), dimSize);
}

bool IfKEqual1(const aclTensor *&selfInput, const MmOpInfo& mmOpInfo, const bool &transX1Flag, const aclTensor *&bias) {
  // 判断dtype, 切mul分支仅支持dtype同进同出，且不支持bf16
  if (mmOpInfo.support_info.self_dtype == DataType::DT_BF16 || mmOpInfo.support_info.mat2_dtype == DataType::DT_BF16) {
    return false;
  }
  // 不支持nz场景
  if (mmOpInfo.support_info.self_format == Format::FORMAT_FRACTAL_NZ) {
    return false;
  }
  OP_LOGD("Check MatMul or BatchMatmul k=1 scenario, and support_info is not NZ");
  if (mmOpInfo.support_info.output_dtype != mmOpInfo.support_info.self_dtype ||
      mmOpInfo.support_info.output_dtype != mmOpInfo.support_info.mat2_dtype) {
    return false;
  }
  // 判断是否带bias
  if (bias != nullptr) {
    return false;
  }
  // 判断k轴是否满足切mul需求(等于1)
  int64_t kDimNumWhenNoTrans = selfInput->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t kDimNumWhenTrans = selfInput->GetViewShape().GetDimNum() - OUTER_AXIS;
  int64_t kDim = transX1Flag ? selfInput->GetViewShape().GetDim(kDimNumWhenTrans) :
                 selfInput->GetViewShape().GetDim(kDimNumWhenNoTrans);
  if (kDim != DIM_EQUAL_ONE) {
    return false;
  }
  return true;
}

aclnnStatus IfKEqual1SelfToMK(const aclTensor *&selfInput, const aclTensor *&selfReshapeOutput, bool &transX1Flag,
                              aclOpExecutor *executor) {
  auto x1Perm = NeedTransPerm(selfInput, executor);
  selfReshapeOutput = transX1Flag ? l0op::Reshape(selfInput, x1Perm, executor) : selfInput;
  CHECK_RET(selfReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  transX1Flag = false;
  return ACLNN_SUCCESS;
}

aclnnStatus IfKEqual1Mat2ToKN(const aclTensor *&mat2Input, const aclTensor *&mat2ReshapeOutput, bool &transX2Flag,
                              aclOpExecutor *executor) {
  auto x2Perm = NeedTransPerm(mat2Input, executor);
  mat2ReshapeOutput = transX2Flag ? l0op::Reshape(mat2Input, x2Perm, executor) : mat2Input;
  CHECK_RET(mat2ReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  transX2Flag = false;
  return ACLNN_SUCCESS;
}

aclnnStatus IfMEqual1SelfToMK(const aclTensor *&selfInput, const aclTensor *&selfReshapeOutput,
                              const Format selfInputFormat, bool &transX1Flag, aclOpExecutor *executor) {
  // 不支持nz场景
  if (selfInputFormat == Format::FORMAT_FRACTAL_NZ) {
    return ACLNN_SUCCESS;
  }
  OP_LOGD("Check MatMul or BatchMatmul m=1 scenario, and support_info is not NZ");
  // 首先判断m轴是否已经为外轴，是外轴则return
  if (!transX1Flag) {
    return ACLNN_SUCCESS;
  }
  // 判断m/n轴是否满足等于1，满足则reshape为外轴再进行mm/bmm计算
  int64_t mDimNumWhenInner = selfInput->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t mDimSize = selfInput->GetViewShape().GetDim(mDimNumWhenInner);
  if (mDimSize != DIM_EQUAL_ONE) {
    return ACLNN_SUCCESS;
  }
  auto x1Perm = NeedTransPerm(selfInput, executor);
  selfReshapeOutput = l0op::Reshape(selfInput, x1Perm, executor);
  transX1Flag = false;
  CHECK_RET(selfReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  OP_LOGI("Hit MatMul or BatchMatmul m=1 and m is inner scenario, trans m axis to outer");
  return ACLNN_SUCCESS;
}

aclnnStatus IfNEqual1Mat2ToNK(const aclTensor *&mat2Input, const aclTensor *&mat2ReshapeOutput,
                              const Format mat2InputFormat, bool &transX2Flag, aclOpExecutor *executor) {
  // 不支持nz场景。
  if (mat2InputFormat == Format::FORMAT_FRACTAL_NZ) {
    return ACLNN_SUCCESS;
  }
  OP_LOGD("Check MatMul or BatchMatmul n=1 scenario, and support_info is not NZ");
  // 首先判断n轴是否已经为外轴，是外轴则return
  if (transX2Flag) {
    return ACLNN_SUCCESS;
  }
  // 判断m/n轴是否满足等于1，满足则reshape为外轴再进行mm/bmm计算
  int64_t nDimNumWhenInner = mat2Input->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t nDimSize = mat2Input->GetViewShape().GetDim(nDimNumWhenInner);
  if (nDimSize != DIM_EQUAL_ONE) {
    return ACLNN_SUCCESS;
  }
  auto x2Perm = NeedTransPerm(mat2Input, executor);
  mat2ReshapeOutput = l0op::Reshape(mat2Input, x2Perm, executor);
  transX2Flag = true;
  CHECK_RET(mat2ReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  OP_LOGI("Hit MatMul or BatchMatmul n=1 and n is inner scenario, trans n axis to outer");
  return ACLNN_SUCCESS;
}

uint64_t TransDequantScaleToM1(const float deqScale) {
  union {
    float scaleFloat;
    uint32_t scaleInt;
  } dequantScale;
  dequantScale.scaleFloat = deqScale;
  uint64_t fixpipeDeqScale = static_cast<uint64_t>(dequantScale.scaleInt) & kDeqScaleMul;
  return fixpipeDeqScale;
}

uint32_t GetL2Size(const std::string& socLongVersion) {
  constexpr int32_t l2Size2 = 192;
  constexpr int32_t l2Size4 = 96;
  return socLongVersion == SOC_B4 ? l2Size4 : l2Size2;
}

uint64_t GetL1Size([[maybe_unused]] const std::string& socLongVersion) {
  constexpr int64_t l1Size = 512;
  //支持芯片固定512K,后续开放别的芯片修改此处
  return l1Size;
}

op::FVector<int64_t> GetShape(const aclTensor *tensor) {
  op::FVector<int64_t> shape;
  if (tensor == nullptr) {
    shape.push_back(1);
    OP_LOGW("The input tensor of Func GetShape is nullptr");
    return shape;
  }
  if (tensor->GetViewShape().GetDimNum() == 0U) {
    shape.push_back(1);
  } else {
    size_t dimNum = tensor->GetViewShape().GetDimNum();
    for (size_t idx = 0U; idx < dimNum; idx++) {
      int64_t tmpVal = tensor->GetViewShape().GetDim(idx);
      shape.push_back(tmpVal);
    }
  }
  return shape;
}

const aclTensor *ContiguousBias(const aclTensor *self, const aclTensor *bias, aclOpExecutor *executor) {
    auto contiguousBias = l0op::Contiguous(bias, executor);
    CHECK_RET(contiguousBias != nullptr, nullptr);
    // bias为bf16时cast为fp32保证精度
    if ((contiguousBias->GetDataType() == DataType::DT_BF16 &&
          GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95)||
        self->GetDataType() == DataType::DT_FLOAT) {
        contiguousBias = l0op::Cast(contiguousBias, op::DataType::DT_FLOAT, executor);
        CHECK_RET(contiguousBias != nullptr, nullptr);
    }
    return contiguousBias;
}

#ifdef __cplusplus
}
#endif