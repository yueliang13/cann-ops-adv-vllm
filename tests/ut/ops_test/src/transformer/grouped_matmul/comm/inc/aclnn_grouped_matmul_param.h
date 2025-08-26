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
 * \file aclnn_grouped_matmul_param.h
 * \brief GroupedMatmul Aclnn 参数信息.
 */

#ifndef UTEST_ACLNN_GROUPED_MATMUL_PARAM_H
#define UTEST_ACLNN_GROUPED_MATMUL_PARAM_H

#include "grouped_matmul_case.h"
#include "tests/utils/aclnn_tensor.h"
#include "tests/utils/aclnn_tensor_list.h"

namespace ops::adv::tests::grouped_matmul {

class AclnnGroupedMatmulParam : public ops::adv::tests::grouped_matmul::Param {
public:
    using AclnnTensor = ops::adv::tests::utils::AclnnTensor;
    using AclnnTensorList = ops::adv::tests::utils::AclnnTensorList;

public:
    enum class FunctionType {
        NO_QUANT,
        QUANT,
        ANTIQUANT,
        QUANT_PERTOKEN
    };

    enum class AclnnGroupedMatmulVersion {
        V1,
        V2,
        V3,
        V4
    };

public:
    FunctionType mFunctionType = FunctionType::NO_QUANT;
    AclnnGroupedMatmulVersion mAclnnGroupedMatmulVersion = AclnnGroupedMatmulVersion::V1;
    /* 输入输出 */
    AclnnTensorList aclnnX, aclnnWeight, aclnnBias, aclnnScale, aclnnOffset, aclnnAntiquantScale, aclnnAntiquantOffset,
        aclnnPerTokenScale, aclnnY;
    AclnnTensor aclnnGroupListTensor;
    aclIntArray *aclnnGroupListIntAry = nullptr;

public:
    AclnnGroupedMatmulParam() = default;
    AclnnGroupedMatmulParam(std::vector<TensorList> inputs, Tensor groupList, std::vector<int64_t> groupListData,
                            std::int32_t splitItem, int32_t dtype, bool transposeWeight, bool transposeX,
                            int32_t groupType, int32_t groupListType, int32_t actType, FunctionType functionType,
                            AclnnGroupedMatmulVersion aclnnGroupedMatmulVersion);

    ~AclnnGroupedMatmulParam();

    bool Init();

private:
    bool InitGroupList();
};

} // namespace ops::adv::tests::grouped_matmul
#endif // UTEST_ACLNN_GROUPEDMATMUL_PARAM_H
