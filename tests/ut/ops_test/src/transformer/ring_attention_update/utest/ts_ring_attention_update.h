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
 * \file ts_ring_attention_update.h
 * \brief RingAttentionUpdate UTest 相关基类定义.
 */
#pragma once

#include "tests/utest/ts.h"
#include "ring_attention_update_case.h"

using RingAttentionUpdateCase = ops::adv::tests::RingAttentionUpdate::RingAttentionUpdateCase;

class Ts_RingAttentionUpdate : public Ts<RingAttentionUpdateCase> {};