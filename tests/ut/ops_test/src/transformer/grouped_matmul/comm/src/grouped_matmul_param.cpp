#/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file grouped_matmul_param.cpp
 * \brief GroupedMatmul 参数信息.
 */

#include "grouped_matmul_param.h"

using namespace ops::adv::tests::grouped_matmul;

Param::Param(std::vector<TensorList> inputs, Tensor perTokenScale, Tensor groupList,
             std::vector<int64_t> groupListData, int32_t splitItem, int32_t dType,
             bool transposeWeight, bool transposeX, int32_t groupType, int32_t groupListType, int32_t actType)
    : mPerTokenScale(perTokenScale), mGroupListData(std::move(groupListData)), mSplitItem(splitItem),
      mDtype(dType), mTransposeWeight(transposeWeight), mTransposeX(transposeX),
      mGroupType(groupType), mGroupListType(groupListType), mActType(actType)
{
    for (auto &tensorList : inputs) {
        mTensorLists[tensorList.Name()] = tensorList;
    }
    mGroupList = groupList;
}

Tensor ops::adv::tests::grouped_matmul::GenTensor(const char *name, const std::initializer_list<int64_t> &shape,
                                                  ge::DataType dType, ge::Format format)
{
    return Tensor(name, shape, "", dType, format);
}

TensorList ops::adv::tests::grouped_matmul::GenTensorList(const char *name, const std::vector<std::vector<int64_t>> &shapes,
                                                          ge::DataType dType, ge::Format format)
{
    return TensorList(name, shapes, "", dType, format);
}