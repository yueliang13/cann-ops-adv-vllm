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
 * \file aclnn_context.h
 * \brief  提供 Aclnn Tiling / Kernel 阶段上下文功能, 辅助 Tiling / Kernel 运行.
 */

#pragma once

#include "tests/utils/context_intf.h"
#include <aclnn/aclnn_base.h>

namespace ops::adv::tests::utils {

class AclnnContext : public ops::adv::tests::utils::ContextIntf {
public:
    /**
     * Tiling 运行回调函数
     */
    typedef bool (*TilingRunCbf)(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor);
    typedef bool (*KernelRunCbf)(void *curCase);

public:
    AclnnContext() = default;
    ~AclnnContext() override;

    /* 属性设置 */
    [[maybe_unused]] [[nodiscard]] bool SetTilingRunCbf(TilingRunCbf cbf);
    [[maybe_unused]] [[nodiscard]] bool SetKernelRunCbf(KernelRunCbf cbf);

    aclrtStream GetAclRtStream() const;
    aclOpExecutor *GetAclOpExecutor() const;

    /* Tiling */
    bool RunTiling(std::string &caseName) override;

protected:
    TilingRunCbf tilingRunCbf_ = nullptr;
    KernelRunCbf kernelRunCbf_ = nullptr;
    aclrtStream aclrtStream_ = nullptr;
    aclOpExecutor *aclOpExecutor_ = nullptr;

protected:
    bool RunKernelProcess(std::string &caseName) override;
    uint8_t *AllocWorkspaceImpl(uint64_t size) override;
    void FreeWorkspaceImpl(uint8_t *ptr) override;

private:
    void Destroy();
};

} // namespace ops::adv::tests::utils
