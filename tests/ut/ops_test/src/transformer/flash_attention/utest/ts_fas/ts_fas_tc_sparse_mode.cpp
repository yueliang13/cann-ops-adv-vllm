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
 * \file ts_fas_tc_sparse_mode.cpp
 * \brief
 */

#include "ts_fas.h"
#include "tiling/fa/tiling_data.h"

namespace {
uint8_t GetSparseType(const void *tilingData)
{
    auto *fasTilingData = (const FlashAttentionScoreGeneralTilingData *)tilingData;
    return fasTilingData->inputParams.sparseType;
}

uint8_t GetImplMode(const void *tilingData)
{
    auto *fasTilingData = (const FlashAttentionScoreGeneralTilingData *)tilingData;
    return fasTilingData->inputParams.implMode;
}
} // namespace

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_001)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 2;
    cs.mParam.n2 = 8;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::B_N1_1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 2048;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 3;
    cs.mParam.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
    ASSERT_EQ(GetImplMode(cs.mForwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_002)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 10;
    cs.mParam.g = 1;
    cs.mParam.s1 = 4096;
    cs.mParam.s2 = 4096;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 4096;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
    ASSERT_EQ(GetImplMode(cs.mForwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_003)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_004)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 16;
    cs.mParam.g = 1;
    cs.mParam.s1 = 4096;
    cs.mParam.s2 = 4096;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 2048;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_005)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 4;
    cs.mParam.n2 = 4;
    cs.mParam.g = 1;
    cs.mParam.s1 = 128;
    cs.mParam.s2 = 1024;
    cs.mParam.d = 125;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 500;
    cs.mParam.nxtTokens = 300;
    cs.mParam.layoutType = LayoutType::BSND;
    cs.mParam.innerPrecise = 1;
    cs.mParam.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
    ASSERT_EQ(GetImplMode(cs.mForwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_006)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 4;
    cs.mParam.n2 = 4;
    cs.mParam.g = 1;
    cs.mParam.s1 = 128;
    cs.mParam.s2 = 1024;
    cs.mParam.d = 125;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 500;
    cs.mParam.nxtTokens = 300;
    cs.mParam.layoutType = LayoutType::BSND;
    cs.mParam.innerPrecise = 1;
    cs.mParam.sparseMode = 99;

    /**
     * 期望信息
     */
    cs.mForward.mExp.mSuccess = false;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_007)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 4;
    cs.mParam.n2 = 4;
    cs.mParam.g = 1;
    cs.mParam.s1 = 128;
    cs.mParam.s2 = 1024;
    cs.mParam.d = 125;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 500;
    cs.mParam.nxtTokens = 300;
    cs.mParam.layoutType = LayoutType::BSND;
    cs.mParam.innerPrecise = 1;
    cs.mParam.sparseMode = 100;

    /**
     * 期望信息
     */
    cs.mForward.mExp.mSuccess = false;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_008)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 2;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_009)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 1024;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 1;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 1);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_010)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 2048;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_011)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 2049;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_012)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 2048;
    cs.mParam.nxtTokens = 2049;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_013)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = -900;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_014)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = -900;
    cs.mParam.nxtTokens = 1024;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_015)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 1;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 1);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_016)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 0;
    cs.mParam.nxtTokens = 1024;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 1;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 1);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_017)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 2;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_018)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 3096;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_019)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 2048;
    cs.mParam.nxtTokens = 100;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 2;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_020)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 1024;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_021)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = -900;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_022)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = -900;
    cs.mParam.nxtTokens = 1024;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_023)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 10;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 4096;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 2048;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_024)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 10;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 4096;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 2048;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 2;
    cs.mParam.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
    ASSERT_EQ(GetImplMode(cs.mForwardCtx.GetTilingData()), 2);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_025)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_026)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::PREFIXCOMPRESS;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::B;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::TND;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 6;
    cs.mParam.prefixTensorData = {2500};
    cs.mParam.actualSeqQLenTensorData = {2048};
    cs.mParam.actualSeqKVLenTensorData = {3028};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_027)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 1;
    cs.mParam.g = 1;
    cs.mParam.s1 = 317;
    cs.mParam.s2 = 317;
    cs.mParam.d = 80;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::PREFIXCOMPRESS;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::B;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 6;
    cs.mParam.prefixTensorData = {0};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_028)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 4;
    cs.mParam.n2 = 4;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 21;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::BSND;
    cs.mParam.innerPrecise = 1;
    cs.mParam.sparseMode = 0;

    /**
     * 期望信息
     */
    cs.mForward.mExp.mSuccess = true;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_029)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 4;
    cs.mParam.n2 = 16;
    cs.mParam.g = 1;
    cs.mParam.s1 = 512;
    cs.mParam.s2 = 512;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_030)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 4;
    cs.mParam.n2 = 16;
    cs.mParam.g = 1;
    cs.mParam.s1 = 512;
    cs.mParam.s2 = 512;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 1;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 1);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_031)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 16;
    cs.mParam.g = 1;
    cs.mParam.s1 = 512;
    cs.mParam.s2 = 512;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::B;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 10;
    cs.mParam.nxtTokens = 1000;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 5;
    cs.mParam.prefixTensorData = {100};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_032)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 16;
    cs.mParam.g = 1;
    cs.mParam.s1 = 512;
    cs.mParam.s2 = 512;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::PREFIXCOMPRESS;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::B;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 6;
    cs.mParam.prefixTensorData = {0};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_033)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 4;
    cs.mParam.n2 = 16;
    cs.mParam.g = 1;
    cs.mParam.s1 = 4096;
    cs.mParam.s2 = 4096;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_034)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 4;
    cs.mParam.n2 = 16;
    cs.mParam.g = 1;
    cs.mParam.s1 = 4096;
    cs.mParam.s2 = 4096;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 1;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 1);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_035)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 2;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_036)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 3096;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_037)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 2048;
    cs.mParam.nxtTokens = 100;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 2;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_038)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 1024;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_039)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = -900;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_040)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = -900;
    cs.mParam.nxtTokens = 1024;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_041)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 10;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 4096;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 2048;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_042)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 10;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 4096;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 1024;
    cs.mParam.nxtTokens = 2048;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 2;
    cs.mParam.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
    ASSERT_EQ(GetImplMode(cs.mForwardCtx.GetTilingData()), 2);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_043)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 64;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_044)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 16;
    cs.mParam.g = 1;
    cs.mParam.s1 = 4096;
    cs.mParam.s2 = 4096;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::B;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 5;
    cs.mParam.prefixTensorData = {100};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_045)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 16;
    cs.mParam.g = 1;
    cs.mParam.s1 = 4096;
    cs.mParam.s2 = 4096;
    cs.mParam.d = 128;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::PREFIXCOMPRESS;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::B;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 6;
    cs.mParam.prefixTensorData = {0};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_046)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 9;
    cs.mParam.n2 = 13;
    cs.mParam.g = 1;
    cs.mParam.s1 = 16;
    cs.mParam.s2 = 16;
    cs.mParam.d = 64;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_047)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 9;
    cs.mParam.n2 = 13;
    cs.mParam.g = 1;
    cs.mParam.s1 = 16;
    cs.mParam.s2 = 16;
    cs.mParam.d = 256;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 1;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_048)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 9;
    cs.mParam.n2 = 13;
    cs.mParam.g = 1;
    cs.mParam.s1 = 16;
    cs.mParam.s2 = 16;
    cs.mParam.d = 256;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::B;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 5;
    cs.mParam.prefixTensorData = {2, 3, 4, 5, 6, 7, 8, 9, 10};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_049)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 9;
    cs.mParam.n2 = 13;
    cs.mParam.g = 1;
    cs.mParam.s1 = 16;
    cs.mParam.s2 = 16;
    cs.mParam.d = 256;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::PREFIXCOMPRESS;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::B;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::SBH;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 6;
    cs.mParam.prefixTensorData = {2, 3, 4, 5, 6, 7, 8, 9, 10};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_050)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 1;
    cs.mParam.n2 = 2;
    cs.mParam.g = 1;
    cs.mParam.s1 = 2048;
    cs.mParam.s2 = 2048;
    cs.mParam.d = 64;
    cs.mParam.dtype = ge::DT_FLOAT;
    cs.mParam.pseShapeType = PseShapeType::NONE;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::PREFIXCOMPRESS;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::B;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 65536;
    cs.mParam.nxtTokens = 0;
    cs.mParam.layoutType = LayoutType::TND;
    cs.mParam.innerPrecise = 0;
    cs.mParam.sparseMode = 6;
    cs.mParam.prefixTensorData = {2500};
    cs.mParam.actualSeqQLenTensorData = {2048};
    cs.mParam.actualSeqKVLenTensorData = {3028};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_051)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 40;
    cs.mParam.n2 = 12;
    cs.mParam.g = 1;
    cs.mParam.s1 = 256;
    cs.mParam.s2 = 256;
    cs.mParam.d = 144;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 100;
    cs.mParam.nxtTokens = 200;
    cs.mParam.layoutType = LayoutType::BSND;
    cs.mParam.innerPrecise = 1;
    cs.mParam.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.mForwardCtx.GetTilingData()), 4);
    ASSERT_EQ(GetImplMode(cs.mForwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_052)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.mParam.b = 40;
    cs.mParam.n2 = 12;
    cs.mParam.g = 1;
    cs.mParam.s1 = 256;
    cs.mParam.s2 = 256;
    cs.mParam.d = 144;
    cs.mParam.dtype = ge::DT_FLOAT16;
    cs.mParam.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.mParam.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.mParam.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.mParam.attenMaskDtype = ge::DT_BOOL;
    cs.mParam.prefixShapeType = PrefixShapeType::NONE;
    cs.mParam.scale = 0.5f;
    cs.mParam.keepProb = 0.9f;
    cs.mParam.preTokens = 100;
    cs.mParam.nxtTokens = 200;
    cs.mParam.layoutType = LayoutType::BSND;
    cs.mParam.innerPrecise = 1;
    cs.mParam.sparseMode = 4;

    /**
     * 用例 预制条件修改, 期望结果设置
     */
    cs.mPreTilingRunCbf = FaCase::PreTilingRunCbf_SetPlatformInfoNull;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}