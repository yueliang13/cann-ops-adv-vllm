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
 * \file ts_ffn.h
 * \brief FFN UTest 相关基类定义.
 */

#ifndef UTEST_TS_FFN_H
#define UTEST_TS_FFN_H

#include "tests/utest/ts.h"
#include "ffn_case.h"

using ops::adv::tests::ffn::FFNCase;
using ops::adv::tests::ffn::GenTensor;
using ops::adv::tests::ffn::Param;

class Ts_FFN : public Ts<FFNCase> {};
class Ts_FFN_Ascend910B1 : public Ts_Ascend910B1<FFNCase> {};
class Ts_FFN_Ascend910B2 : public Ts_Ascend910B2<FFNCase> {};
class Ts_FFN_Ascend910B3 : public Ts_Ascend910B3<FFNCase> {};

class Ts_FFN_Ascend310P3 : public Ts_Ascend310P3<FFNCase> {};

class Ts_FFN_WithParam : public Ts_WithParam<FFNCase> {};
class Ts_FFN_WithParam_Ascend910B1 : public Ts_WithParam_Ascend910B1<FFNCase> {};
class Ts_FFN_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<FFNCase> {};
class Ts_FFN_WithParam_Ascend910B3 : public Ts_WithParam_Ascend910B3<FFNCase> {};

class Ts_FFN_WithParam_Ascend310P3 : public Ts_WithParam_Ascend310P3<FFNCase> {};

#endif // UTEST_TS_FFN_H