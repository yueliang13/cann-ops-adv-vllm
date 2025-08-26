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
 * \file aclnn_ffn_case.h
 * \brief FFN Aclnn 测试用例.
 */

#ifndef UTEST_ACLNN_FFN_CASE_H
#define UTEST_ACLNN_FFN_CASE_H

#include "ffn_case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/aclnn_context.h"
#include "aclnn_ffn_param.h"

namespace ops::adv::tests::ffn {
using AclnnFFNVersion = ops::adv::tests::ffn::AclnnFFNParam::AclnnFFNVersion;

class AclnnFFNCase : public ops::adv::tests::ffn::FFNCase {
public:
    using AclnnContext = ops::adv::tests::utils::AclnnContext;

public:
    /* 算子控制信息 */
    AclnnContext mAclnnCtx;

    /* 输入/输出 参数 */
    AclnnFFNParam mAclnnParam;

public:
    AclnnFFNCase();
    AclnnFFNCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo, AclnnFFNParam param,
                 int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);
    bool Run() override;

protected:
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::ffn
#endif // UTEST_ACLNN_FFN_CASE_H
