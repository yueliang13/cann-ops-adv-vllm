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
 * \file ts_fas_tc_error.cpp
 * \brief FlashAttentionScore 正向用例.
 */

#include "ts_fas.h"

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Layout_Case)
{
    FasCase cs("Tc_Fas_Invalid_Layout_Case", true,                     /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.mForwardCtx.MdfAttrs({"input_layout", std::string("BND")}));
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_HeadNum_Case_001)
{
    FasCase cs("Tc_Fas_Invalid_HeadNum_Case_001", true,                /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.mForwardCtx.MdfAttrs({"head_num", std::int64_t(0)}));
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_HeadNum_Case_002)
{
    FasCase cs("Tc_Fas_Invalid_HeadNum_Case_002", true,                /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.mForwardCtx.MdfAttrs({"head_num", std::int64_t(5)}));
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_QKVShape_Case_001)
{
    FasCase cs("Tc_Fas_Invalid_QKVShape_Case_001", true,               /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init());
    cs.mParam.query = Tensor("query", {cs.mParam.b, cs.mParam.s2, cs.mParam.n2 * cs.mParam.g * cs.mParam.d},
                             "QueryInvalidShape", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_QKVShape_Case_002)
{
    FasCase cs("Tc_Fas_Invalid_QKVShape_Case_002", true,               /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init());
    cs.mParam.key = Tensor("key", {cs.mParam.b, cs.mParam.s2, cs.mParam.n2 * cs.mParam.d}, "KeyInvalidShape",
                           cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_QKVShape_Case_003)
{
    FasCase cs("Tc_Fas_Invalid_QKVShape_Case_003", true,               /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init());
    cs.mParam.key = Tensor("key", {cs.mParam.b + 1, cs.mParam.n2, cs.mParam.s2, cs.mParam.d}, "KeyInvalidShape",
                           cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_QKVShape_Case_004)
{
    FasCase cs("Tc_Fas_Invalid_QKVShape_Case_004", true,               /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init());
    cs.mParam.key = Tensor("key", {cs.mParam.b, cs.mParam.n2, cs.mParam.s2, cs.mParam.d + 1}, "KeyInvalidShape",
                           cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_QKVShape_Case_005)
{
    FasCase cs("Tc_Fas_Invalid_QKVShape_Case_005", true,               /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.mForwardCtx.MdfAttrs({"head_num", std::int64_t(1000)}));
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_QKVShape_Case_006)
{
    FasCase cs("Tc_Fas_Invalid_QKVShape_Case_006", true,               /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init());
    cs.mParam.key = Tensor("key", {cs.mParam.b, cs.mParam.n2 - 1, cs.mParam.s2, cs.mParam.d}, "KeyInvalidShape",
                           cs.mParam.dtype, ge::FORMAT_ND);
    cs.mParam.value = Tensor("value", {cs.mParam.b, cs.mParam.n2 - 1, cs.mParam.s2, cs.mParam.d}, "ValueInvalidShape",
                             cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_AttenMask_Case_001)
{
    FasCase cs("Tc_Fas_Invalid_AttenMask_Case_001", true,              /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.attenMask = Tensor("atten_mask", {2, 2, cs.mParam.s1, cs.mParam.s2}, "AttenMaskInvalidShape",
                                 cs.mParam.attenMaskDtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_AttenMask_Case_002)
{
    FasCase cs("Tc_Fas_Invalid_AttenMask_Case_002", true,              /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.attenMask =
        Tensor("atten_mask", {1000, 1000}, "AttenMaskInvalidShape", cs.mParam.attenMaskDtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_AttenMask_Case_003)
{
    FasCase cs("Tc_Fas_Invalid_AttenMask_Case_003", true,              /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 2,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.attenMask = Tensor("atten_mask", {cs.mParam.s1, cs.mParam.s2}, "AttenMaskInvalidShape",
                                 cs.mParam.attenMaskDtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_AttenMask_Case_004)
{
    FasCase cs("Tc_Fas_Invalid_AttenMask_Case_004", true,              /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 6,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.attenMask = Tensor("atten_mask", {cs.mParam.s1, cs.mParam.s2}, "AttenMaskInvalidShape",
                                 cs.mParam.attenMaskDtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Pse_Case_001)
{
    FasCase cs("Tc_Fas_Invalid_Pse_Case_001", true,                    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.pse = Tensor("pse", {cs.mParam.b, cs.mParam.n2 * cs.mParam.g, 10, cs.mParam.s2}, "PseInvalidShape",
                           cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Pse_Case_002)
{
    FasCase cs("Tc_Fas_Invalid_Pse_Case_002", true,                    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {2048, 4096, 6144})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.pse = Tensor("pse", {cs.mParam.b, cs.mParam.n2 * cs.mParam.g, 1, cs.mParam.s2}, "PseInvalidShape",
                           cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Pse_Case_003)
{
    FasCase cs("Tc_Fas_Invalid_Pse_Case_003", true,                    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 2048, 2048, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_ALIBI_S1_S2,                 /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Pse_Case_004)
{
    FasCase cs("Tc_Fas_Invalid_Pse_Case_004", true,                    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 65535, 65535,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_ALIBI_S1_S2,                 /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {2048, 4096, 6144})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Pse_Case_005)
{
    FasCase cs("Tc_Fas_Invalid_Pse_Case_005", true,                    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 3072, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                       0.08838f, 0.9f, 65535, 65535,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 2,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_ALIBI_S1_S2,                 /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {1024, 4096, 6144})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Pse_Case_006)
{
    FasCase cs("Tc_Fas_Invalid_Pse_Case_006", true,                    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 512, 512, 128,                        /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BSND,     /* Dtype, Layout */
                       0.08838f, 0.9f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_ALIBI_S1_S2,                 /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {},                                             /* ActualSeqQTensorData */
                       {})                                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Prefix_Case_001)
{
    FasCase cs("Tc_Fas_Invalid_Prefix_Case_001", true,                 /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 10, 1, 1024, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 6,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::PREFIXCOMPRESS,             /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::B,                             /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {},                                             /* ActualSeqQTensorData */
                       {})                                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.prefix = Tensor("prefix", {cs.mParam.b - 1}, "PrefixInvalidShape", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Prefix_Case_002)
{
    FasCase cs("Tc_Fas_Invalid_Prefix_Case_002", true,                 /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 10, 1, 1024, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 6,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::PREFIXCOMPRESS,             /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::B,                             /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {},                                             /* ActualSeqQTensorData */
                       {})                                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.prefix = Tensor("prefix", {33}, "PrefixInvalidShape", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Prefix_Case_003)
{
    FasCase cs("Tc_Fas_Invalid_Prefix_Case_003", true,                 /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 10, 1, 1024, 2048, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 6,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::PREFIXCOMPRESS,             /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::B,                             /* PrefixShapeType */
                       {1536, 128, 1536},                              /* PrefixTensorData */
                       {},                                             /* ActualSeqQTensorData */
                       {})                                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Prefix_Case_004)
{
    FasCase cs("Tc_Fas_Invalid_Prefix_Case_004", true,                 /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 10, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 6,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::PREFIXCOMPRESS,             /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::B,                             /* PrefixShapeType */
                       {4096, 128, 1536},                              /* PrefixTensorData */
                       {},                                             /* ActualSeqQTensorData */
                       {})                                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Prefix_Case_005)
{
    FasCase cs("Tc_Fas_Invalid_Prefix_Case_005", true,                 /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 6,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::PREFIXCOMPRESS,             /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::B,                             /* PrefixShapeType */
                       {4096, 512, 512},                               /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {1024, 2048, 3072})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Prefix_Case_006)
{
    FasCase cs("Tc_Fas_Invalid_Prefix_Case_006", true,                 /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 1024, 2048, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 6,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::PREFIXCOMPRESS,             /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::B,                             /* PrefixShapeType */
                       {512, 512, 512},                                /* PrefixTensorData */
                       {1024, 2048, 3072},                             /* ActualSeqQTensorData */
                       {2048, 4096, 6144})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_Prefix_Case_007)
{
    FasCase cs("Tc_Fas_Invalid_Prefix_Case_007", true,                 /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 10, 1, 1024, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, Layout */
                       0.08838f, 0.9f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 5,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::B,                             /* PrefixShapeType */
                       {128, 512, 512},                                /* PrefixTensorData */
                       {},                                             /* ActualSeqQTensorData */
                       {})                                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_KeepProb_Case)
{
    FasCase cs("Tc_Fas_Invalid_KeepProb_Case", true,                   /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, LayoutType */
                       0.08838f, 1.2f, 1024, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case001)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case001", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, LayoutType */
                       0.08838f, 0.8f, -512, -512,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case002)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case002", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, LayoutType */
                       0.08838f, 0.8f, -1024, 512,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case003)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case003", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(24, 10, 1, 1024, 1024, 128,                     /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, LayoutType */
                       0.08838f, 0.8f, 512, -1024,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 4,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                          /* PrefixShapeType */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case004)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case004", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 65535, -512,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {2048, 4096, 6144})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case005)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case005", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 512, 512,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {1024, 2048, 3072})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case006)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case006", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, -1024, 512,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {1024, 2048, 3072})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case007)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case007", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 512, 512,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 3,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {1024, 2048, 3072})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case008)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case008", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, -512, 512,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 4,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {1024, 2048, 3072})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case009)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case009", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 512, -1024,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 4,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {1024, 2048, 3072})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case010)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case010", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 512, 512,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 4,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {1024, 2048, 3072})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case011)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case011", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 512, 512,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 7,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {2048, 4096, 6144})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case012)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case012", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 2048, -512,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 7,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {1024, 2048, 3072})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case013)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case013", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 2048, -2048,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 7,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {2048, 4096, 6144})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Invalid_PreTokens_NextTokens_Case014)
{
    FasCase cs("Tc_Fas_Invalid_PreTokens_NextTokens_Case014", true,    /* CaseName, Enable */
               "",                                                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, LayoutType */
                       0.08838f, 0.9f, 2048, -2048,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 8,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                             /* PseShapeType */
                       DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                       AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                          /* PrefixShapeType */
                       {},                                             /* PrefixTensorData */
                       {2048, 4096, 6144},                             /* ActualSeqQTensorData */
                       {2048, 4096, 6144})                             /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}
