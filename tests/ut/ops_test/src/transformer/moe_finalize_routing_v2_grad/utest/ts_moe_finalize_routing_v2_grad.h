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
 * \file ts_moe_finalize_routing_v2_grad.h
 * \brief MoeFinalizeRoutingV2Grad UTest 相关基类定义.
 */
#pragma once

#include "tests/utest/ts.h"
#include "moe_finalize_routing_v2_grad_case.h"

using MoeFinalizeRoutingV2GradCase = ops::adv::tests::MoeFinalizeRoutingV2Grad::MoeFinalizeRoutingV2GradCase;

class Ts_MoeFinalizeRoutingV2Grad : public Ts<MoeFinalizeRoutingV2GradCase> {};