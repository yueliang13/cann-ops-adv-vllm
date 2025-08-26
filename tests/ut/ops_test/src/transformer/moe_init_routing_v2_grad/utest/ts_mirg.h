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
 * \file ts_mirg.h
 * \brief MoeInitRoutingV2Grad UTest 相关基类定义.
 */
#pragma once

#include "tests/utest/ts.h"
#include "mirg_case.h"
#define DTYPE_GRAD_EXPANDED_X float

using MoeInitRoutingV2GradCase = ops::adv::tests::mirg::MirgCase;

class Ts_mirg : public Ts<MoeInitRoutingV2GradCase> {};
class Ts_mirg_Ascend910B2 : public Ts_Ascend910B2<MoeInitRoutingV2GradCase> {};
class Ts_mirg_Ascend310P3 : public Ts_Ascend310P3<MoeInitRoutingV2GradCase> {};

class Ts_mirg_WithParam : public Ts_WithParam<MoeInitRoutingV2GradCase> {};
class Ts_mirg_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<MoeInitRoutingV2GradCase> {};
class Ts_mirg_WithParam_Ascend310P3 : public Ts_WithParam_Ascend310P3<MoeInitRoutingV2GradCase> {};