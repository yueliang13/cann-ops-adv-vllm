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
 * \file ts_gmmadd.h
 * \brief MoeFinalizeRoutingV2 测试用例.
 */

 #include "tests/utest/ts.h"
 #include "gmm_add_case.h"
 
 using GmmAddCase = ops::adv::tests::gmmadd::GmmAddCase;

 class Ts_gmmadd : public Ts<GmmAddCase> {};
 class Ts_gmmadd_Ascend910B2 : public Ts_Ascend910B2<GmmAddCase> {};
 class Ts_gmmadd_Ascend310P3 : public Ts_Ascend310P3<GmmAddCase> {};
 
 class Ts_gmmadd_WithParam : public Ts_WithParam<GmmAddCase> {};
 class Ts_gmmadd_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<GmmAddCase> {};
 class Ts_gmmadd_WithParam_Ascend310P3 : public Ts_WithParam_Ascend310P3<GmmAddCase> {};
 