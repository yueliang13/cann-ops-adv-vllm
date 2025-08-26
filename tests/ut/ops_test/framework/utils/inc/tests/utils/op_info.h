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
 * \file op_info.h
 * \brief 测试用例算子信息.
 */

#pragma once

#include <string>
#include "tests/utils/control_info.h"
#include "tests/utils/expect_info.h"
#include "tests/utils/context_intf.h"

namespace ops::adv::tests::utils {

class OpInfo {
public:
    /**
     * 算子名
     */
    std::string mName;

    /**
     * 控制信息
     */
    ControlInfo mCtr;

    /**
     * 期望结果
     */
    ExpectInfo mExp;

    /**
     * 运行上下文
     */
    ContextIntf *mCtx;

public:
    OpInfo();
    explicit OpInfo(const ControlInfo &ctr);
    OpInfo(const ControlInfo &ctr, const ExpectInfo &exp);
    OpInfo(const char *name, const ControlInfo &ctr, const ExpectInfo &exp);
    virtual ~OpInfo() = default;

    bool SetContext(ContextIntf *ctxParam);

    virtual bool ProcessTiling(std::string &caseName);
    virtual bool ProcessKernel(std::string &caseName);

    virtual bool RunTiling(std::string &caseName);
    virtual bool ChkTiling(std::string &caseName);

    virtual bool RunKernel(std::string &caseName);
    virtual bool ChkKernel(std::string &caseName);
};

} // namespace ops::adv::tests::utils
