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
 * \file ts_fia.h
 * \brief FusedInferAttentionScore UTest 相关基类定义.
 */

#include "tests/utest/ts.h"
#include "fia_case.h"

using FiaCase = ops::adv::tests::fia::FiaCase;
class Ts_Fia : public Ts<FiaCase> {};
class Ts_Fia_Ascend910B1 : public Ts_Ascend910B1<FiaCase> {};
class Ts_Fia_WithParam : public Ts_WithParam<FiaCase> {};
class Ts_Fia_WithParam_Ascend910B1 : public Ts_WithParam_Ascend910B1<FiaCase> {};
