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
 * \file case.h
 * \brief 测试用例.
 */

#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace ops::adv::tests::utils {

class Case {
public:
    /**
     * Tiling Template Priority 无效值
     */
    static constexpr int32_t kTilingTemplatePriority_Invalid = 0;

public:
    Case();
    Case(const char *name, bool enable, const char *dbgInfo,
         int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);
    virtual ~Case() = default;

    /**
     * 初始化用例
     */
    virtual bool Init();

    /**
     * 获取当前用例
     *
     * \attention 若子类需要调用该接口, 需确保已重写 InitCurrentCasePtr 实现, 以确保所获取指针的正确性.
     */
    static void *GetCurrentCase();

    /**
     * 执行用例
     */
    virtual bool Run() = 0;

    /**
     * 获取用例根目录
     */
    const char *GetRootPath();

protected:
    /**
     * 用例名
     */
    std::string mName;

    /**
     * 用例根目录
     */
    std::string mRootPath;

    /**
     * 用例是否使能
     *
     * \attention
     * 非使能用例会执行 Init 但 Run 流程会返回执行成功.
     */
    bool mEnable = true;

    /**
     * 诊断信息
     */
    std::string mDbgInfo;

    /**
     * Tiling Template Priority
     */
    int32_t mTilingTemplatePriority = kTilingTemplatePriority_Invalid;

    /**
     * 当前用例
     */
    static void *mCurrentCasePtr;

protected:
    /**
     * 初始化参数
     */
    virtual bool InitParam() = 0;

    /**
     * 初始化算子信息
     */
    virtual bool InitOpInfo() = 0;

    /**
     * 设置当前用例
     *
     * \attention 若子类需要调用 GetCurrentCase 接口, 需确保已重写本接口实现, 以确保所获取指针的正确性.
     */
    virtual bool InitCurrentCasePtr();
};

} // namespace ops::adv::tests::utils
