/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file main.cpp
 * \brief 本文件作为 GTest 用例 main 桩文件, 一般不需修改任何内容.
 */

#include <gtest/gtest.h>
#include "tests/utils/platform.h"

using Platform = ops::adv::tests::utils::Platform;

int main(int argc, char **argv)
{
    /**
     * 加载 OpTiling.so
     * 注意必须加载 OpTiling.so, 因为加载时才会触发框架(register) 执行算子处理函数注册.
     */
    Platform platform;
    Platform::SetGlobalPlatform(&platform);
    if (!platform.InitArgsInfo(argc, argv)) {
        return 1;
    }
    if (!platform.LoadOpTilingSo()) {
        return 1;
    }

    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();

    if (!platform.UnLoadOpTilingSo()) {
        return 1;
    }

    return ret;
}
