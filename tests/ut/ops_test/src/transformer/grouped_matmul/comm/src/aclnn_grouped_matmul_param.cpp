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
 * \file aclnn_grouped_matmul_param.cpp
 * \brief GroupedMatmul Aclnn 参数信息.
 */

#include "aclnn_grouped_matmul_param.h"
#include <utility>
#include "tests/utils/case.h"
#include "tests/utils/io.h"
#include "tests/utils/log.h"

using ops::adv::tests::utils::ReadFile;
using ops::adv::tests::utils::WriteFile;
using ops::adv::tests::utils::TensorIntf;
namespace {
template <class T> bool InitAclIntArray(aclIntArray **intArray, std::vector<T> &hostData)
{
    if (intArray == nullptr) {
        LOG_ERR("intArray nil.");
        return false;
    }
    if (*intArray != nullptr) {
        auto ret = aclDestroyIntArray(*intArray);
        LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclDestroyIntArray failed, ERROR: %d", ret), *intArray = nullptr);
    }
    if (hostData.empty()) {
        return true;
    }
    *intArray = aclCreateIntArray(hostData.data(), hostData.size());
    if (*intArray == nullptr) {
        LOG_ERR("aclCreateIntArray failed.");
        return false;
    }
    return true;
}
} // namespace

using namespace ops::adv::tests::grouped_matmul;

AclnnGroupedMatmulParam::AclnnGroupedMatmulParam(std::vector<TensorList> inputs, Tensor groupList, 
    std::vector<int64_t> groupListData, std::int32_t splitItem, int32_t dtype, bool transposeWeight, 
    bool transposeX, int32_t groupType, int32_t groupListType, int32_t actType, FunctionType functionType, 
    AclnnGroupedMatmulVersion aclnnGMMVersion)
    : Param(std::move(inputs), Tensor(), std::move(groupList), std::move(groupListData), splitItem, dtype, transposeWeight, 
    transposeX, groupType, groupListType, actType), mFunctionType(functionType), mAclnnGroupedMatmulVersion(aclnnGMMVersion)
{
}


AclnnGroupedMatmulParam::~AclnnGroupedMatmulParam()
{
    if (aclnnGroupListIntAry != nullptr) {
        auto ret = aclDestroyIntArray(aclnnGroupListIntAry);
        LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclnnGroupListIntAry failed, ERROR: %d", ret),
                    aclnnGroupListIntAry = nullptr);
    }
}

bool AclnnGroupedMatmulParam::Init()
{
    aclnnX = ops::adv::tests::utils::AclnnTensorList(mTensorLists["x"], mTransposeX);
    aclnnWeight = ops::adv::tests::utils::AclnnTensorList(mTensorLists["weight"], mTransposeWeight);
    auto iter = mTensorLists.find("bias");
    if (iter != mTensorLists.end()) {
        aclnnBias = ops::adv::tests::utils::AclnnTensorList(mTensorLists["bias"]);
    }
    aclnnY = ops::adv::tests::utils::AclnnTensorList(mTensorLists["y"]);
    if (mFunctionType == FunctionType::QUANT) {
        aclnnScale = ops::adv::tests::utils::AclnnTensorList(mTensorLists["scale"]);
        iter = mTensorLists.find("offset");
        if (iter != mTensorLists.end()) {
            aclnnOffset = ops::adv::tests::utils::AclnnTensorList(mTensorLists["offset"]);
        }
    } else if (mFunctionType == FunctionType::ANTIQUANT) {
        aclnnAntiquantScale = ops::adv::tests::utils::AclnnTensorList(mTensorLists["antiquant_scale"]);
        aclnnAntiquantOffset = ops::adv::tests::utils::AclnnTensorList(mTensorLists["antiquant_offset"]);
    } else if (mFunctionType == FunctionType::QUANT_PERTOKEN) {
        aclnnScale = ops::adv::tests::utils::AclnnTensorList(mTensorLists["scale"]);
        aclnnPerTokenScale = ops::adv::tests::utils::AclnnTensorList(mTensorLists["pertoken_scale"]);
        iter = mTensorLists.find("offset");
        if (iter != mTensorLists.end()) {
            aclnnOffset = ops::adv::tests::utils::AclnnTensorList(mTensorLists["offset"]);
        }
    }
    auto ret = InitGroupList();
    LOG_IF_EXPR(ret == false, LOG_ERR("InitGroupList faild"), return false);
    auto *cs = static_cast<ops::adv::tests::utils::Case *>(ops::adv::tests::utils::Case::GetCurrentCase());
    LOG_IF_EXPR(cs == nullptr, LOG_ERR("Can't get current case"), return false);
    for (auto *t : 
        std::vector<TensorIntf *>{&aclnnX, &aclnnWeight, &aclnnBias, &aclnnScale, &aclnnOffset, &aclnnAntiquantScale,
                                  &aclnnAntiquantOffset, &aclnnGroupListTensor, &aclnnPerTokenScale, &aclnnY}) {
        t->FreeDevData();
        auto tFormat = t->GetFormat();
        if (t->GetExpDataSize() <= 0) {
            continue;
        }
        uint8_t *devData = nullptr;
        if (tFormat == ge::Format::FORMAT_FRACTAL_NZ) { devData = t->AllocDevDataNz(0, 0); }
        else { devData = t->AllocDevData(0, 0); }
        if (devData == nullptr) {
            return false;
        }
        std::string filePath = std::string(cs->GetRootPath()) + t->Name() + ".bin";
        if (ops::adv::tests::utils::FileExist(filePath)) {
            if (!t->LoadFileToDevData(filePath)) {
                return false;
            }
        }
    }
    return true;
}

bool AclnnGroupedMatmulParam::InitGroupList()
{
    if (mGroupListData.size() == 0) {
        return true;
    }

    if (mAclnnGroupedMatmulVersion == AclnnGroupedMatmulVersion::V3 || 
        mAclnnGroupedMatmulVersion == AclnnGroupedMatmulVersion::V4) {
        size_t dataSize = mGroupListData.size() * sizeof(int64_t);
        std::string fileName = "groupList.bin";
        if (!WriteFile(fileName, mGroupListData.data(), dataSize)) {
            LOG_ERR("Write groupList data to file[%s] failed", fileName.c_str());
            return false;
        }
        aclnnGroupListTensor = ops::adv::tests::utils::AclnnTensor(mGroupList);
    } else if (!InitAclIntArray(&aclnnGroupListIntAry, mGroupListData)) {
        return false;
    }

    return true;
}