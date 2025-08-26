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
 * \file ts_arope.h
 * \brief ApplyRotaryPosEmb UTest 相关基类定义.
 */

#include "tests/utest/ts.h"
#include "arope_case.h"

using ARopeCase = ops::adv::tests::arope::ARopeCase;
class Ts_ARope : public Ts<ARopeCase> {};
class Ts_ARope_Ascend910B1 : public Ts_Ascend910B1<ARopeCase> {};