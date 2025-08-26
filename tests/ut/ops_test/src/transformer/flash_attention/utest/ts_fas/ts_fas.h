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
 * \file ts_fas.h
 * \brief FlashAttentionScore UTest 基类定义.
 */

#include "ts_fa.h"
#include "fas_case.h"

using FasCase = ops::adv::tests::fas::FasCase;

class Ts_Fas : public Ts<FasCase> {};
class Ts_Fas_Ascend910B1 : public Ts_Ascend910B1<FasCase> {};
class Ts_Fas_Ascend910B2 : public Ts_Ascend910B2<FasCase> {};
class Ts_Fas_Ascend910B3 : public Ts_Ascend910B3<FasCase> {};

class Ts_Fas_WithParam : public Ts_WithParam<FasCase> {};
class Ts_Fas_WithParam_Ascend910B1 : public Ts_WithParam_Ascend910B1<FasCase> {};
class Ts_Fas_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<FasCase> {};
class Ts_Fas_WithParam_Ascend910B3 : public Ts_WithParam_Ascend910B3<FasCase> {};
