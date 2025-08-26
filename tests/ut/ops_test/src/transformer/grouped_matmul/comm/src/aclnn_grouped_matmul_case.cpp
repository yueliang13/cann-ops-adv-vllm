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
 * \file aclnn_grouped_matmul_case.cpp
 * \brief GroupedMatmul Aclnn 测试用例.
 */

#include <utility>
#include "tests/utils/log.h"
#include "aclnn_grouped_matmul.h"
#include "aclnn_grouped_matmul_v2.h"
#include "aclnn_grouped_matmul_v3.h"
#include "aclnn_grouped_matmul_v4.h"
#include "aclnn_grouped_matmul_case.h"

using namespace ops::adv::tests::grouped_matmul;

bool GroupedMatmulTilingRunCbf(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor)
{
    auto *cs = static_cast<AclnnGroupedMatmulCase *>(curCase);
    auto *aclnnParam = &cs->mAclnnParam;

    aclnnStatus ret = ACL_SUCCESS;
    if (aclnnParam->mAclnnGroupedMatmulVersion == AclnnGroupedMatmulVersion::V1) {
        ret = aclnnGroupedMatmulGetWorkspaceSize(
            aclnnParam->aclnnX.GetAclTensorList(), aclnnParam->aclnnWeight.GetAclTensorList(),
            aclnnParam->aclnnBias.GetAclTensorList(), aclnnParam->aclnnScale.GetAclTensorList(),
            aclnnParam->aclnnOffset.GetAclTensorList(), aclnnParam->aclnnAntiquantScale.GetAclTensorList(),
            aclnnParam->aclnnAntiquantOffset.GetAclTensorList(), aclnnParam->aclnnGroupListIntAry,
            aclnnParam->mSplitItem, aclnnParam->aclnnY.GetAclTensorList(), workSpaceSize, opExecutor);
    } else if (aclnnParam->mAclnnGroupedMatmulVersion == AclnnGroupedMatmulVersion::V2) {
        ret = aclnnGroupedMatmulV2GetWorkspaceSize(
            aclnnParam->aclnnX.GetAclTensorList(), aclnnParam->aclnnWeight.GetAclTensorList(),
            aclnnParam->aclnnBias.GetAclTensorList(), aclnnParam->aclnnScale.GetAclTensorList(),
            aclnnParam->aclnnOffset.GetAclTensorList(), aclnnParam->aclnnAntiquantScale.GetAclTensorList(),
            aclnnParam->aclnnAntiquantOffset.GetAclTensorList(), aclnnParam->aclnnGroupListIntAry,   
            aclnnParam->mSplitItem, aclnnParam->mGroupType, aclnnParam->aclnnY.GetAclTensorList(),
            workSpaceSize, opExecutor);
    } else if (aclnnParam->mAclnnGroupedMatmulVersion == AclnnGroupedMatmulVersion::V3) {
        ret = aclnnGroupedMatmulV3GetWorkspaceSize(
            aclnnParam->aclnnX.GetAclTensorList(), aclnnParam->aclnnWeight.GetAclTensorList(),
            aclnnParam->aclnnBias.GetAclTensorList(), aclnnParam->aclnnScale.GetAclTensorList(),
            aclnnParam->aclnnOffset.GetAclTensorList(), aclnnParam->aclnnAntiquantScale.GetAclTensorList(),
            aclnnParam->aclnnAntiquantOffset.GetAclTensorList(), aclnnParam->aclnnGroupListTensor.GetAclTensor(),        
            aclnnParam->mSplitItem, aclnnParam->mGroupType, aclnnParam->aclnnY.GetAclTensorList(),
            workSpaceSize, opExecutor);
    } else if (aclnnParam->mAclnnGroupedMatmulVersion == AclnnGroupedMatmulVersion::V4) {
        ret = aclnnGroupedMatmulV4GetWorkspaceSize(
            aclnnParam->aclnnX.GetAclTensorList(), aclnnParam->aclnnWeight.GetAclTensorList(),
            aclnnParam->aclnnBias.GetAclTensorList(), aclnnParam->aclnnScale.GetAclTensorList(),
            aclnnParam->aclnnOffset.GetAclTensorList(), aclnnParam->aclnnAntiquantScale.GetAclTensorList(),
            aclnnParam->aclnnAntiquantOffset.GetAclTensorList(), aclnnParam->aclnnPerTokenScale.GetAclTensorList(),
            aclnnParam->aclnnGroupListTensor.GetAclTensor(), nullptr, nullptr, nullptr,
            aclnnParam->mSplitItem, aclnnParam->mGroupType, aclnnParam->mGroupListType, aclnnParam->mActType,
            aclnnParam->aclnnY.GetAclTensorList(), nullptr, nullptr, workSpaceSize, opExecutor);
    }
    return ret == ACL_SUCCESS;
}

bool GroupedMatmulKernelRunCbf(void *curCase)
{
    auto *cs = static_cast<AclnnGroupedMatmulCase *>(curCase);
    auto *aclnnParam = &cs->mAclnnParam;
    auto *aclnnCtx = &cs->mAclnnCtx;

    aclnnStatus ret = ACL_SUCCESS;
    if (aclnnParam->mAclnnGroupedMatmulVersion == AclnnGroupedMatmulVersion::V1) {
        ret = aclnnGroupedMatmul(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(), aclnnCtx->GetAclOpExecutor(),
                                 aclnnCtx->GetAclRtStream());
    } else if (aclnnParam->mAclnnGroupedMatmulVersion == AclnnGroupedMatmulVersion::V2) {
        ret = aclnnGroupedMatmulV2(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(), aclnnCtx->GetAclOpExecutor(),
                                   aclnnCtx->GetAclRtStream());
    } else if (aclnnParam->mAclnnGroupedMatmulVersion == AclnnGroupedMatmulVersion::V3) {
        ret = aclnnGroupedMatmulV3(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(), aclnnCtx->GetAclOpExecutor(),
                                   aclnnCtx->GetAclRtStream());
    } else if (aclnnParam->mAclnnGroupedMatmulVersion == AclnnGroupedMatmulVersion::V4) {
        ret = aclnnGroupedMatmulV4(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(), aclnnCtx->GetAclOpExecutor(),
                                   aclnnCtx->GetAclRtStream());
    }
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnGroupedMatmul failed, ERROR: %d", ret));

    return ret == ACL_SUCCESS;
}

AclnnGroupedMatmulCase::AclnnGroupedMatmulCase()
    : GroupedMatmulCase(), mAclnnCtx(AclnnContext()), mAclnnParam(AclnnGroupedMatmulParam())
{
}

AclnnGroupedMatmulCase::AclnnGroupedMatmulCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo,
    AclnnGroupedMatmulParam aclnnParam, int32_t tilingTemplatePriority)
    : GroupedMatmulCase(name, enable, dbgInfo, std::move(opInfo), Param(), tilingTemplatePriority),
      mAclnnParam(std::move(aclnnParam))
{
}

bool AclnnGroupedMatmulCase::InitParam()
{
    return mAclnnParam.Init();
}

bool AclnnGroupedMatmulCase::InitOpInfo()
{
    if (!GroupedMatmulCase::InitOpInfo()) {
        return false;
    }

    auto rst = mAclnnCtx.SetOpName(this->mOpInfo.mName.c_str());
    rst = rst && mAclnnCtx.SetTilingRunCbf(GroupedMatmulTilingRunCbf);
    rst = rst && mAclnnCtx.SetKernelRunCbf(GroupedMatmulKernelRunCbf);
    rst = rst && mOpInfo.SetContext(&mAclnnCtx);
    return rst;
}

bool AclnnGroupedMatmulCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool AclnnGroupedMatmulCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}
