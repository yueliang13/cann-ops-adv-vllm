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
 * \file aclnn_context.cpp
 * \brief  提供 Aclnn Tiling / Kernel 阶段上下文功能, 辅助 Tiling / Kernel 运行.
 */

#include "tests/utils/aclnn_context.h"
#include <acl/acl.h>
#include "tests/utils/log.h"
#include "tests/utils/case.h"

using namespace ops::adv::tests::utils;

AclnnContext::~AclnnContext()
{
    this->Destroy();
}

bool AclnnContext::SetTilingRunCbf(TilingRunCbf cbf)
{
    if (cbf == nullptr) {
        return false;
    }
    tilingRunCbf_ = cbf;
    return true;
}

bool AclnnContext::SetKernelRunCbf(KernelRunCbf cbf)
{
    if (cbf == nullptr) {
        return false;
    }
    kernelRunCbf_ = cbf;
    return true;
}

aclrtStream AclnnContext::GetAclRtStream() const
{
    return aclrtStream_;
}

aclOpExecutor *AclnnContext::GetAclOpExecutor() const
{
    return aclOpExecutor_;
}

bool AclnnContext::RunTiling(std::string &caseName)
{
    if (tilingRunCbf_ == nullptr) {
        LOG_ERR("[%s:%s] TilingCbf nil", opName_.c_str(), caseName.c_str());
        return false;
    }
    auto *curCase = ops::adv::tests::utils::Case::GetCurrentCase();
    if (curCase == nullptr) {
        LOG_ERR("[%s:%s] Current case nil", opName_.c_str(), caseName.c_str());
        return false;
    }
    if (!tilingRunCbf_(curCase, &workspaceSize_, &aclOpExecutor_)) {
        LOG_DBG("[%s:%s] Run Tiling failed", opName_.c_str(), caseName.c_str());
        aclOpExecutor_ = nullptr;
        return false;
    }
    return true;
}

bool AclnnContext::RunKernelProcess(std::string &caseName)
{
    auto *curCase = ops::adv::tests::utils::Case::GetCurrentCase();
    if (curCase == nullptr) {
        LOG_ERR("[%s:%s] Current case nil", opName_.c_str(), caseName.c_str());
        return false;
    }
    if (kernelRunCbf_ == nullptr) {
        LOG_ERR("[%s:%s] KernelCbf nil", opName_.c_str(), caseName.c_str());
        return false;
    }
    if (!kernelRunCbf_(curCase)) {
        LOG_DBG("[%s:%s] Run Kernel failed", opName_.c_str(), caseName.c_str());
        return false;
    }
    return true;
}

uint8_t *AclnnContext::AllocWorkspaceImpl(uint64_t size)
{
    void *ptr = nullptr;
    auto ret = aclrtMalloc((void **)&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclrtMalloc failed, ERROR: %d", ret));
    return (uint8_t *)ptr;
}

void AclnnContext::FreeWorkspaceImpl(uint8_t *ptr)
{
    auto ret = aclrtFree(ptr);
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclrtFree failed, ERROR: %d", ret));
}

void AclnnContext::Destroy()
{
    if (aclrtStream_ != nullptr) {
        auto ret = aclrtDestroyStream(aclrtStream_);
        LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclrtDestroyStream failed, ERROR: %d", ret));
    }
    if (workspacePtr_ != nullptr) {
        this->FreeWorkspaceImpl(workspacePtr_);
        workspacePtr_ = nullptr;
    }
    workspaceSize_ = 0;
}
