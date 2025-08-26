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
 * \file ts_grouped_matmul.h
 * \brief GroupedMatmul UTest 相关基类定义.
 */

#ifndef UTEST_TS_GROUPEDMATMUL_H
#define UTEST_TS_GROUPEDMATMUL_H

#include "tests/utest/ts.h"
#include "grouped_matmul_case.h"
#include "aclnn_grouped_matmul_case.h"

using ops::adv::tests::grouped_matmul::AclnnGroupedMatmulCase;
using ops::adv::tests::grouped_matmul::GenTensor;
using ops::adv::tests::grouped_matmul::GenTensorList;
using AclnnGroupedMatmulParam = ops::adv::tests::grouped_matmul::AclnnGroupedMatmulParam;
using FunctionType = ops::adv::tests::grouped_matmul::AclnnGroupedMatmulParam::FunctionType;
using AclnnGroupedMatmulVersion = ops::adv::tests::grouped_matmul::AclnnGroupedMatmulParam::AclnnGroupedMatmulVersion;

class Ts_Aclnn_GroupedMatmul_WithParam_Ascend910B3 : public Ts_WithParam_Ascend910B3<AclnnGroupedMatmulCase> {};

#endif // UTEST_TS_GROUPEDMATMUL_H