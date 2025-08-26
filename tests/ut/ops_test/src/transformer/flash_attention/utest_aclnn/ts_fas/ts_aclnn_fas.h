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
 * \file ts_aclnn_fas.h
 * \brief FlashAttentionScore UTest 相关基类定义.
 */

#pragma once

#include "ts_aclnn_fa.h"
#include "aclnn_fas_case.h"

using AclnnFasCase = ops::adv::tests::fas::AclnnFasCase;

class Ts_Aclnn_Fas : public Ts<AclnnFasCase> {};
class Ts_Aclnn_Fas_Ascend910B1 : public Ts_Ascend910B1<AclnnFasCase> {};
class Ts_Aclnn_Fas_Ascend910B2 : public Ts_Ascend910B2<AclnnFasCase> {};
class Ts_Aclnn_Fas_Ascend910B3 : public Ts_Ascend910B3<AclnnFasCase> {};

class Ts_Aclnn_Fas_WithParam : public Ts_WithParam<AclnnFasCase> {};
class Ts_Aclnn_Fas_WithParam_Ascend910B1 : public Ts_WithParam_Ascend910B1<AclnnFasCase> {};
class Ts_Aclnn_Fas_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<AclnnFasCase> {};
class Ts_Aclnn_Fas_WithParam_Ascend910B3 : public Ts_WithParam_Ascend910B3<AclnnFasCase> {};
