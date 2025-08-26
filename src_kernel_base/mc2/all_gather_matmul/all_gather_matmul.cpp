/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file all_gather_matmul.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "all_gather_matmul_tiling.h"
#include "all_gather_matmul_full_mesh.h"

using namespace AscendC;

#define INVOKE_ALL_GATHER_MATMUL_OP_IMPL(templateClass, ...)                                   \
    do {                                                                                       \
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_DTYPE, true>;       \
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;             \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                          \
        op.Init(aGM, bGM, biasGM, cGM, gatherOut, workspaceGM, contextGM, &tilingData, mc2InitTiling, mc2CcTiling, &pipe); \
        op.Process();                                                                          \
    } while (0)

template <class T> struct BiasType {
    using type = float;
};
template <> struct BiasType<half> {
    using type = half;
};

extern "C" __global__ __aicore__ void all_gather_matmul(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
    GM_ADDR gatherOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AllGatherMatmulTilingData);
    auto tiling = (__gm__ AllGatherMatmulTilingData*)tilingGM;
    __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));
    __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));
    GET_TILING_DATA(tilingData, tilingGM);

    TPipe pipe;
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();

    if (TILING_KEY_IS(110)) {
        // full mesh + nd2nz + no bias cast
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, B_DTYPE, true>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>;
        INVOKE_ALL_GATHER_MATMUL_OP_IMPL(AllGatherMatmulFullMesh, true, false);
    } else if (TILING_KEY_IS(100)) {
        // full mesh + no nd2nz + no bias cast
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, true>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>;
        INVOKE_ALL_GATHER_MATMUL_OP_IMPL(AllGatherMatmulFullMesh, false, false);
    } else if (TILING_KEY_IS(111)) {
        // full mesh + nd2nz + bias cast
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, B_DTYPE, true>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
        INVOKE_ALL_GATHER_MATMUL_OP_IMPL(AllGatherMatmulFullMesh, true, true);
    } else if (TILING_KEY_IS(101)) {
        // full mesh + no nd2nz + bias cast
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, true>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
        INVOKE_ALL_GATHER_MATMUL_OP_IMPL(AllGatherMatmulFullMesh, false, true);
    }
}