/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_tensor.h
 * \brief 封装 ACLNN Tensor, 简化 Tiling 及 Kernel 阶段对 Tensor 操作.
 */

#pragma once

#include "tests/utils/tensor.h"
#include <aclnn/aclnn_base.h>

namespace ops::adv::tests::utils {

class AclnnTensor : public ops::adv::tests::utils::TensorIntf {
public:
    AclnnTensor() = default;
    AclnnTensor(const char *name, const std::initializer_list<int64_t> &shape, const char *shapeType,
                ge::DataType dType, ge::Format format, TensorType type = TensorType::REQUIRED_INPUT);
    AclnnTensor(const char *name, const std::vector<int64_t> &shape, const char *shapeType, ge::DataType dType,
                ge::Format format, TensorType type = TensorType::REQUIRED_INPUT);
    explicit AclnnTensor(const Tensor &t);
    AclnnTensor(const AclnnTensor &o) = default;
    AclnnTensor &operator=(const AclnnTensor &o) = default;
    ~AclnnTensor() override;

    aclDataType GetAclDataType() const;
    aclTensor *GetAclTensor() const;

    uint8_t *AllocDevData(int32_t initVal, int64_t minSize) override;
    void FreeDevData() override;

protected:
    aclDataType aclDataType_ = ACL_DT_UNDEFINED;
    std::vector<int64_t> aclTensorDataStrides_;
    aclTensor *aclTensor_ = nullptr;

    uint8_t *AllocDevDataImpl(int64_t size) override;
    void FreeDevDataImpl(uint8_t *devPtr) override;
    bool MemSetDevDataImpl(uint8_t *devPtr, int64_t devMax, int32_t val, int64_t cnt) override;
    bool MemCpyHostToDevDataImpl(uint8_t *devPtr, int64_t devMax, const void *hostPtr, int64_t cnt) override;
    bool MemCpyDevDataToHostImpl(void *hostPtr, int64_t hostMax, const uint8_t *devPtr, int64_t cnt) override;

private:
    void Destroy();
};

} // namespace ops::adv::tests::utils
