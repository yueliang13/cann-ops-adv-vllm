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
 * \file dequant_rope_quant_kvcache_case.h
 * \brief DequantRopeQuantKvcache 测试用例.
 */

#pragma once
#include <register/op_impl_registry.h>
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"

namespace ops::adv::tests::DequantRopeQuantKvcache {
class DequantRopeQuantKvcacheCase : public ops::adv::tests::utils::Case {
using OpInfo = ops::adv::tests::utils::OpInfo;
using Context = ops::adv::tests::utils::Context;
using Tensor = ops::adv::tests::utils::Tensor;
using TensorList = ops::adv::tests::utils::TensorList;

public:
class Param {
public:
int64_t B = 0;
int64_t S = 0;
int64_t Nkv = 0;
int64_t Nq = 0;
int64_t D = 0;
int64_t H = 0;
int64_t C1 = 0;
int64_t C2 = 0;
std::vector<int64_t> sizeSplits = {128 * 8, 128, 128};
bool outOptional = false;
std::string cacheOptional = "contigunous";
ge::DataType xDtype = ge::DataType::DT_FLOAT;
ge::DataType sinDtype = ge::DataType::DT_FLOAT;
ge::DataType biasDtype = ge::DataType::DT_FLOAT;
std::string quantOptional = "static";
std::string qlayout = "BSND";

    Param();
    Param(int64_t pB, int64_t pS, int64_t pNkv, int64_t pNq, int64_t pD, int64_t pC1, int64_t pC2, 
                                            bool pOutOptional, std::string pCacheOptional,
                                            ge::DataType xDtypeIn, ge::DataType sinDtypeIn, ge::DataType biasDtypeIn);
};
class DoTilingParam {
public:
    gert::TilingContext *ctx = nullptr;
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
};
OpInfo mOpInfo;
Context mCtx;
Param mParam;
Tensor x, cosIn, sinIn, kcache, vcache, indices, kscale, vscale, koffset, voffset, weight, activation, bias, qOut, kOut,
    vOut, workspaceSize, tiling;
gert::OpImplRegisterV2::TilingKernelFunc dequantRopeQuantKvcacheTilingFunc = nullptr;
DequantRopeQuantKvcacheCase();
DequantRopeQuantKvcacheCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
bool Run() override;
bool InitParam() override;
bool InitOpInfo() override;
bool InitCurrentCasePtr() override;
bool DoOpTiling(DoTilingParam &tilingParam);
};

} // namespace ops::adv::tests::DequantRopeQuantKvcache