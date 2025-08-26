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
 * \file ts_fas_tc_redline.cpp
 * \brief FlashAttentionScore 正向用例.
 */

#include "ts_fas.h"

class Ts_Fas_Ascend910B2_Redline_FullCore : public Ts_Fas_WithParam_Ascend910B2 {};
class Ts_Fas_Ascend910B3_Redline_FullCore : public Ts_Fas_WithParam_Ascend910B3 {};

TEST_P(Ts_Fas_Ascend910B2_Redline_FullCore, Tc_Redline_FullCore)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mForward.mExp.mSuccess);
}

TEST_P(Ts_Fas_Ascend910B3_Redline_FullCore, Tc_Redline_FullCore)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Redline_FullCore_SpecCase_001)
{
    FasCase cs("Fas_Redline_FullCore_SpecCase_001", true,           /* CaseName, Enable */
               "",                                                  /* DebugInfo */
               OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                      ExpectInfo(true,                              /* ExpectSuccess */
                                 10000000001022430943UL,            /* ExpectTilingKey */
                                 ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                       0.08838f, 0.9f, 65535, 0,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                        /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                          /* PseShapeType */
                       DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                       /* PrefixShapeType */
                       {},                                          /* PrefixTensorData */
                       {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                       {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.attenMask = Tensor("attenMask", {cs.mParam.s1 * cs.mParam.b, cs.mParam.s2 * cs.mParam.b},
                                 "attenMaskInvalidShape", cs.mParam.attenMaskDtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Redline_Pse_001)
{
    FasCase cs("Fas_Redline_Pse_001", true,                         /* CaseName, Enable */
               "",                                                  /* DebugInfo */
               OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                      ExpectInfo(true,                              /* ExpectSuccess */
                                 10000000011022430943UL,            /* ExpectTilingKey */
                                 ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                       0.08838f, 0.9f, 65535, 0,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                        /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                          /* PseShapeType */
                       DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                       /* PrefixShapeType */
                       {},                                          /* PrefixTensorData */
                       {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                       {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.pse = Tensor("pse", {503316480}, "PseShape", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Redline_Pse_002)
{
    FasCase cs("Fas_Redline_Pse_002", true,                         /* CaseName, Enable */
               "",                                                  /* DebugInfo */
               OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                      ExpectInfo(true,                              /* ExpectSuccess */
                                 10000000011022430943UL,            /* ExpectTilingKey */
                                 ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                       0.08838f, 0.9f, 65535, 0,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                        /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                          /* PseShapeType */
                       DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                       /* PrefixShapeType */
                       {},                                          /* PrefixTensorData */
                       {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                       {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.pse = Tensor("pse", {245760}, "PseShape", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Redline_Pse_003)
{
    FasCase cs("Fas_Redline_Pse_003", true,                         /* CaseName, Enable */
               "",                                                  /* DebugInfo */
               OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                      ExpectInfo(true,                              /* ExpectSuccess */
                                 10000000011022430943UL,            /* ExpectTilingKey */
                                 ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                       0.08838f, 0.9f, 65535, 0,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                        /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                          /* PseShapeType */
                       DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                       /* PrefixShapeType */
                       {},                                          /* PrefixTensorData */
                       {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                       {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.pse =
        Tensor("pse", {1, cs.mParam.n2 * cs.mParam.g, 1024, cs.mParam.s2}, "PseShape", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Redline_Pse_004)
{
    FasCase cs("Fas_Redline_Pse_004", true,                         /* CaseName, Enable */
               "",                                                  /* DebugInfo */
               OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                      ExpectInfo(false,                             /* ExpectSuccess */
                                 10000000001022430943UL,            /* ExpectTilingKey */
                                 ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                       0.08838f, 0.9f, 65535, 0,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                       0, 0,                                        /* InnerPrecise, SparseMode */
                       PseShapeType::NONE,                          /* PseShapeType */
                       DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                       PrefixShapeType::NONE,                       /* PrefixShapeType */
                       {},                                          /* PrefixTensorData */
                       {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                       {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.pse = Tensor("pse", {10}, "PseInvaildShape", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Redline_Pse_005)
{
    FasCase cs("Fas_Redline_Pse_005", true,                         /* CaseName, Enable */
               "",                                                  /* DebugInfo */
               OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                      ExpectInfo(false,                             /* ExpectSuccess */
                                 10000001111221320943UL,            /* ExpectTilingKey */
                                 ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(234, 1, 227, 25283, 2133, 32,                /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_BF16, LayoutType::BNSD,     /* Dtype, Layout */
                       0.08838f, 0.9f, 861021531, 278722863,        /* Scale, KeepProb, PreTokens, NxtTokens */
                       1, 0,                                        /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_ALIBI_S1_S2,              /* PseShapeType */
                       DropMaskShapeType::B_N1_S1_S2,               /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                       AttenMaskShapeType::B_1_S1_S2,               /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                       /* PrefixShapeType */
    );

    ASSERT_TRUE(cs.Init());
    cs.mParam.pse =
        Tensor("pse", {1, cs.mParam.n2 * cs.mParam.g, 1024, cs.mParam.s2}, "PseShape", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_Fas_Redline_Pse_006)
{
    FasCase cs("Fas_Redline_Pse_006", true,                         /* CaseName, Enable */
               "",                                                  /* DebugInfo */
               OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                      ExpectInfo(false,                             /* ExpectSuccess */
                                 10000001111221320943UL,            /* ExpectTilingKey */
                                 ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(2, 5, 1, 2048, 768, 64,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_BF16, LayoutType::BNSD,     /* Dtype, Layout */
                       0.08838f, 0.9f, 861021531, 278722863,        /* Scale, KeepProb, PreTokens, NxtTokens */
                       1, 0,                                        /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_ALIBI_S1_S2,              /* PseShapeType */
                       DropMaskShapeType::B_N1_S1_S2,               /* DropMaskShapeType */
                       PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                       AttenMaskShapeType::B_1_S1_S2,               /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                       PrefixShapeType::NONE)                       /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mForward.mExp.mSuccess);
}

const auto Tc_Fas_Redline_FullCore_Case = ::testing::Values(
    FasCase("Fas_Redline_Redline_FullCore_Case_001", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000001111221320943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(234, 1, 227, 25283, 2133, 32,                /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.9f, 861021531, 278722863,        /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                    /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,               /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,               /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                       /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_002", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000001010220230943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 10, 1, 4096, 4096, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,   /* Dtype, Layout */
                    0.08838f, 0.9f, 4096, 0,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                       /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_003", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000001110220230943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 8, 1, 2048, 2048, 128,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,   /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 0,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 3,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                     /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                       /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_004", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000001111220130943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 10, 1, 2304, 2304, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,  /* Dtype, Layout */
                    0.08838f, 0.9f, 1350490028, 783368691,       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                    /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,               /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,               /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                       /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_005", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000001010220230943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 16, 1, 4096, 4096, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,   /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 0,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 3,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                       /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_006", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000001010220230943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 16, 1, 4096, 4096, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,   /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 0,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 2,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                       /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_007", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000010220230943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 4, 1, 4096, 4096, 256,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,   /* Dtype, Layout */
                    0.08838f, 0.9f, 2147483647, 2147483647,      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                       /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_008", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000001002201130953UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(210, 2, 38, 30063, 85, 65,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,  /* Dtype, Layout */
                    0.08838f, 0.9f, 2137226131, 848857676,       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                    /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::NONE,                    /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                       /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_009", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000010102200230953UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(24, 10, 1, 1024, 1024, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,   /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, 0,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 5,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::B,                          /* PrefixShapeType */
                    {512, 1024, 1536},                           /* PrefixTensorData */
                    {},                                          /* ActualSeqQTensorData */
                    {})                                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_010", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000010102200230953UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(33, 10, 1, 1024, 1024, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,   /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, 0,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 5,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,               /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype*/
                    PrefixShapeType::B,                          /* PrefixShapeType */
                    {
                        512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
                        512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
                    },  /* PrefixTensorData */
                    {}, /* ActualSeqQTensorData */
                    {}) /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_011", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000102200130953UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(128, 5, 1, 256, 256, 128,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,   /* Dtype, Layout */
                    0.08838f, 0.9f, 65, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                       /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_012", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000002210130953UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(128, 5, 1, 1024, 512, 64,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,   /* Dtype, Layout */
                    0.08838f, 0.9f, 65535, 65535,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::NONE,                    /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                       /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_013", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 65535, 0,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_014", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001122430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 65535, 0,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,               /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_015", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 1, 100,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_016", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 100, 1,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_017", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 100, 1,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 1,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_018", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 2048,                  /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 2,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_019", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, 4,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 3,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_020", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 128,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_021", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, -100,                 /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 7,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6000},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_022", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 128,                  /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 8,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_023", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 1600, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, -100, 256,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {1600, 3200, 4800},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_024", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(false,                             /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 1600, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, 256,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 7,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {1600, 3200, 4800},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_025", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(false,                             /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 1600, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 100, -256,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {1600, 3200, 4800},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_026", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 0,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_027", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(false,                             /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 8,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 7,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_028", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000011022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 8,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 3, 3,                                     /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_B_N1,                    /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqKVTensorData */
                    {1},                                         /* qStartIdxTensorData */
                    {2})                                         /* kvStartIdxTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_029", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000001110220130943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,  /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 8,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 3, 3,                                     /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_B_N1,                    /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {},                                          /* ActualSeqQTensorData */
                    {},                                          /* ActualSeqKVTensorData */
                    {1},                                         /* qStartIdxTensorData */
                    {2})                                         /* kvStartIdxTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_030", true,      /* CaseName, Enable */
            "",                                                 /* DebugInfo */
            OpInfo(ControlInfo(true, false),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                             /* ExpectSuccess */
                              10000000011021130099UL,           /* ExpectTilingKey */
                              2)),                              /* ExpectTilingBlockDim */
            FaParam(3, 1, 1, 16, 16, 128,                       /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND, /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 8,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                    /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_B_N1,                   /* PseShapeType */
                    DropMaskShapeType::NONE,                    /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                  /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                      /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                      /* PrefixShapeType */
                    {},                                         /* PrefixTensorData */
                    {},                                         /* ActualSeqQTensorData */
                    {},                                         /* ActualSeqKVTensorData */
                    {1},                                        /* qStartIdxTensorData */
                    {2})                                        /* kvStartIdxTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_031", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000001102200130953UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 256, 256, 128,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,  /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 8,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                     /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_B_N1,                    /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {},                                          /* ActualSeqQTensorData */
                    {},                                          /* ActualSeqKVTensorData */
                    {1},                                         /* qStartIdxTensorData */
                    {2})                                         /* kvStartIdxTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_032", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000001102200130953UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 256, 256, 128,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,  /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 8,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                     /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_N1,                      /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {},                                          /* ActualSeqQTensorData */
                    {},                                          /* ActualSeqKVTensorData */
                    {1},                                         /* qStartIdxTensorData */
                    {2})                                         /* kvStartIdxTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_033", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(false,                             /* ExpectSuccess */
                              10000001102200130953UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 256, 256, 128,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,  /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 8,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                     /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::B_N1_S1_S2,                    /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {},                                          /* ActualSeqQTensorData */
                    {},                                          /* ActualSeqKVTensorData */
                    {1},                                         /* qStartIdxTensorData */
                    {2})                                         /* kvStartIdxTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_034", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000001102200130953UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 256, 257, 128,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,  /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 8,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                     /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_B_N1,                    /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {},                                          /* ActualSeqQTensorData */
                    {},                                          /* ActualSeqKVTensorData */
                    {1},                                         /* qStartIdxTensorData */
                    {2})                                         /* kvStartIdxTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_035", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(false,                             /* ExpectSuccess */
                              10000000011022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 4096, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 8,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 3, 3,                                     /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_B_N1,                    /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqKVTensorData */
                    {1},                                         /* qStartIdxTensorData */
                    {2})                                         /* kvStartIdxTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_036", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000001102200130953UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 256, 256, 128,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,  /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 8,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                     /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_B_N1,                    /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {},                                          /* ActualSeqQTensorData */
                    {},                                          /* ActualSeqKVTensorData */
                    {1, 2},                                      /* qStartIdxTensorData */
                    {2})                                         /* kvStartIdxTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_037", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000001102200130953UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 256, 256, 128,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,  /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 8,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                     /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_B_N1,                    /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {},                                          /* ActualSeqQTensorData */
                    {},                                          /* ActualSeqKVTensorData */
                    {1},                                         /* qStartIdxTensorData */
                    {2, 3})                                      /* kvStartIdxTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_038", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000011022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 8,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                     /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_B_N1,                    /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqQTensorData */
                    {2048, 4096, 6144},                          /* ActualSeqKVTensorData */
                    {1},                                         /* qStartIdxTensorData */
                    {2})                                         /* kvStartIdxTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_039", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(false,                             /* ExpectSuccess */
                              10000001110220130943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,  /* Dtype, Layout */
                    0.08838f, 0.9f, 256, 8,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                     /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_B_N1,                    /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {},                                          /* ActualSeqQTensorData */
                    {},                                          /* ActualSeqKVTensorData */
                    {1},                                         /* qStartIdxTensorData */
                    {2})                                         /* kvStartIdxTensorData */
            ));
INSTANTIATE_TEST_SUITE_P(Fas, Ts_Fas_Ascend910B2_Redline_FullCore, Tc_Fas_Redline_FullCore_Case);
INSTANTIATE_TEST_SUITE_P(Fas, Ts_Fas_Ascend910B3_Redline_FullCore, Tc_Fas_Redline_FullCore_Case);

class Ts_Fas_Ascend910B2_Redline_SpecCore : public Ts_Fas_WithParam_Ascend910B2 {};
class Ts_Fas_Ascend910B3_Redline_SpecCore : public Ts_Fas_WithParam_Ascend910B3 {};

TEST_P(Ts_Fas_Ascend910B2_Redline_SpecCore, Tc_Redline_SpecCore)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mForward.mExp.mSuccess);
}

TEST_P(Ts_Fas_Ascend910B3_Redline_SpecCore, Tc_Redline_SpecCore)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mForward.mExp.mSuccess);
}

const auto Tc_Fas_Redline_SpecCore_Case = ::testing::Values(
    FasCase("Fas_Redline_SpecCore_Case_001", true,              /* CaseName, Enable */
            "",                                                 /* DebugInfo */
            OpInfo(ControlInfo(true, false),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                             /* ExpectSuccess */
                              10000000000021330099UL,           /* ExpectTilingKey */
                              1)),                              /* ExpectTilingBlockDim */
            FaParam(2, 1, 1, 128, 128, 128,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD, /* Dtype, Layout */
                    0.08838f, 0.9f, 2147483647, 2147483647,     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                       /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                         /* PseShapeType */
                    DropMaskShapeType::NONE,                    /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                    AttenMaskShapeType::NONE,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                      /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                      /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_002", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000011121130099UL,          /* ExpectTilingKey */
                              1)),                             /* ExpectTilingBlockDim */
            FaParam(2, 2, 1, 256, 80, 64,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH, /* Dtype, Layout */
                    0.08838f, 0.9f, 2137226131, 848857676,     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                  /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,             /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,            /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_f16_run_kernel)", true,  /* CaseName, Enable */
            "",                                                 /* DebugInfo */
            OpInfo(ControlInfo(true, true),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                             /* ExpectSuccess */
                              10000000011121332099UL,           /* ExpectTilingKey */
                              1)),                              /* ExpectTilingBlockDim */
            FaParam(2, 2, 1, 256, 80, 64,                       /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD, /* Dtype, Layout */
                    0.08838f, 0.9f, 2137226131, 848857676,      /* Scale, KeepProb, PreTokens, NxtTokens */
                    2, 0,                                       /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                   /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,              /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                      /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                      /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_bf16_run_kernel)", true, /* CaseName, Enable */
            "",                                                 /* DebugInfo */
            OpInfo(ControlInfo(true, true),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                             /* ExpectSuccess */
                              10000000011121322099UL,           /* ExpectTilingKey */
                              1)),                              /* ExpectTilingBlockDim */
            FaParam(2, 2, 1, 256, 80, 64,                       /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BNSD,    /* Dtype, Layout */
                    0.08838f, 0.9f, 2137226131, 848857676,      /* Scale, KeepProb, PreTokens, NxtTokens */
                    2, 0,                                       /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                   /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,              /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                      /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                      /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_003", true,              /* CaseName, Enable */
            "",                                                 /* DebugInfo */
            OpInfo(ControlInfo(true, false),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                             /* ExpectSuccess */
                              10000000001021330099UL,           /* ExpectTilingKey */
                              1)),                              /* ExpectTilingBlockDim */
            FaParam(2, 1, 1, 128, 128, 128,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD, /* Dtype, Layout */
                    0.08838f, 0.9f, 128, 0,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                       /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                         /* PseShapeType */
                    DropMaskShapeType::NONE,                    /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                      /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                      /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_004", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000010220230943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, 0,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_005", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000001010220230943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, 0,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 5,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                     /* PrefixShapeType */
                    {512, 1024, 1536},                         /* PrefixTensorData */
                    {},                                        /* ActualSeqQTensorData */
                    {})                                        /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_006", true,              /* CaseName, Enable */
            "",                                                 /* DebugInfo */
            OpInfo(ControlInfo(true, false),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                             /* ExpectSuccess */
                              10000001112200130953UL,           /* ExpectTilingKey */
                              8)),                              /* ExpectTilingBlockDim */
            FaParam(4, 4, 1, 128, 1024, 125,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND, /* Dtype, Layout */
                    0.08838f, 0.9f, 500, 300,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 4,                                       /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                   /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,              /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                      /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                      /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_007", true,              /* CaseName, Enable */
            "",                                                 /* DebugInfo */
            OpInfo(ControlInfo(true, false),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                             /* ExpectSuccess */
                              10000001112200130953UL,           /* ExpectTilingKey */
                              8)),                              /* ExpectTilingBlockDim */
            FaParam(4, 4, 1, 128, 1024, 125,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND, /* Dtype, Layout */
                    0.08838f, 0.9f, 500, 300,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 3,                                       /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                   /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,              /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                      /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                      /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_008", true,              /* CaseName, Enable */
            "",                                                 /* DebugInfo */
            OpInfo(ControlInfo(true, false),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                             /* ExpectSuccess */
                              10000001112200130953UL,           /* ExpectTilingKey */
                              8)),                              /* ExpectTilingBlockDim */
            FaParam(4, 4, 1, 128, 1024, 125,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND, /* Dtype, Layout */
                    0.08838f, 0.9f, 500, 300,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 5,                                       /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                   /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,              /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                      /* AttentionMaskDtype */
                    PrefixShapeType::B,                         /* PrefixShapeType */
                    {512, 1024, 1536},                          /* PrefixTensorData */
                    {},                                         /* ActualSeqQTensorData */
                    {})                                         /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_009", true,              /* CaseName, Enable */
            "",                                                 /* DebugInfo */
            OpInfo(ControlInfo(true, false),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                             /* ExpectSuccess */
                              10000001112200130953UL,           /* ExpectTilingKey */
                              8)),                              /* ExpectTilingBlockDim */
            FaParam(4, 4, 1, 128, 1024, 125,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND, /* Dtype, Layout */
                    0.08838f, 0.9f, 500, 300,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 5,                                       /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                   /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,              /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                      /* AttentionMaskDtype */
                    PrefixShapeType::B,                         /* PrefixShapeType*/
                    {512, 1024, 1536},                          /* PrefixTensorData */
                    {},                                         /* ActualSeqQTensorData */
                    {})                                         /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_010", true,              /* CaseName, Enable */
            "",                                                 /* DebugInfo */
            OpInfo(ControlInfo(true, false),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                             /* ExpectSuccess */
                              10000001112200130953UL,           /* ExpectTilingKey */
                              1)),                              /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 64, 1024, 125,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND, /* Dtype, Layout */
                    0.08838f, 0.9f, 500, 300,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 5,                                       /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                   /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,              /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                      /* AttentionMaskDtype */
                    PrefixShapeType::B,                         /* PrefixShapeType */
                    {512, 1024, 1536},                          /* PrefixTensorData */
                    {},                                         /* ActualSeqQTensorData */
                    {})                                         /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_011", true,              /* CaseName, Enable */
            "",                                                 /* DebugInfo */
            OpInfo(ControlInfo(true, false),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                             /* ExpectSuccess */
                              10000001102201130953UL,           /* ExpectTilingKey */
                              1)),                              /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 64, 1023, 125,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND, /* Dtype, Layout */
                    0.08838f, 0.9f, 500, 300,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 5,                                       /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                   /* PseShapeType */
                    DropMaskShapeType::NONE,                    /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                      /* AttentionMaskDtype */
                    PrefixShapeType::B,                         /* PrefixShapeType */
                    {512, 1024, 1536},                          /* PrefixTensorData */
                    {},                                         /* ActualSeqQTensorData */
                    {})                                         /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_012", true,              /* CaseName, Enable */
            "",                                                 /* DebugInfo */
            OpInfo(ControlInfo(true, false),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                             /* ExpectSuccess */
                              10000001102201130953UL,           /* ExpectTilingKey */
                              1)),                              /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 64, 1023, 125,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND, /* Dtype, Layout */
                    0.08838f, 0.9f, 500, 300,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 5,                                       /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                   /* PseShapeType */
                    DropMaskShapeType::NONE,                    /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                      /* AttentionMaskDtype */
                    PrefixShapeType::B,                         /* PrefixShapeType */
                    {512, 1024, 1536},                          /* PrefixTensorData */
                    {},                                         /* ActualSeqQTensorData */
                    {})                                         /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_013", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000001000220230943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, 0,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 2,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_014", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000102200230953UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 1024, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, 0,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 1,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_015", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000010220230943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 0,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_016", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000010220230943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 2049, 0,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_017", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000001010220230943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 2049,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_018", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000010220232943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, -900,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_019", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000010220232943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, -900, 1024,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_020", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000010220230943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, 0,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 1,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_021", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000010220230943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 0, 1024,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 1,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_022", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000010220230943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, 0,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 2,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_023", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000010220230943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 3096, 0,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 3,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_024", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000010220230943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 100,                 /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 2,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_025", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000010220230943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, 1024,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_026", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000010220232943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, -900,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_027", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000010220232943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, -900, 1024,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_028", true,             /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, false),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000001010220230943UL,          /* ExpectTilingKey */
                              16)),                            /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                  /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 2049,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                   /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_029", true,               /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001022430943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,   /* Dtype, Layout */
                    0.08838f, 0.9f, 65535, 0,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 6,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,          /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::B,                          /* PrefixShapeType */
                    {1000, 0, 2952},                             /* PrefixTensorData */
                    {2048, 3072, 4000},                          /* ActualSeqQTensorData */
                    {1024, 2048, 5000})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_030", true,               /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000102200132953UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 14, 1, 968, 512, 128,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,  /* Dtype, Layout */
                    0.08838f, 0.9f, 65535, 0,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 6,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,          /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::B,                          /* PrefixShapeType */
                    {100, 0, 200}, {}, {})                       /* PrefixTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_031", true,           /* CaseName, Enable */
            "",                                              /* DebugInfo */
            OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                   ExpectInfo(true,                          /* ExpectSuccess */
                              10000000010220212943UL,        /* ExpectTilingKey */
                              16)),                          /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 1024, -900,              /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                    /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                      /* PseShapeType */
                    DropMaskShapeType::NONE,                 /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,              /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,           /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                   /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                   /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_032", true,           /* CaseName, Enable */
            "",                                              /* DebugInfo */
            OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                   ExpectInfo(true,                          /* ExpectSuccess */
                              10000000010220212943UL,        /* ExpectTilingKey */
                              16)),                          /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, -900, 1024,              /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                    /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                      /* PseShapeType */
                    DropMaskShapeType::NONE,                 /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,              /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,           /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                   /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                   /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_033", true,           /* CaseName, Enable */
            "",                                              /* DebugInfo */
            OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                   ExpectInfo(true,                          /* ExpectSuccess */
                              10000000010220210943UL,        /* ExpectTilingKey */
                              16)),                          /* ExpectTilingBlockDim */
            FaParam(1, 2, 1, 2048, 2048, 128,                /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 2049,              /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                    /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                      /* PseShapeType */
                    DropMaskShapeType::NONE,                 /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,              /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,           /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                   /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                   /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_034", true,               /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000001022410943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 40, 1, 2048, 2048, 128,                   /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::TND,     /* Dtype, Layout */
                    0.08838f, 0.9f, 65535, 0,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 6,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,          /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::B,                          /* PrefixShapeType */
                    {1000, 0, 2952},                             /* PrefixTensorData */
                    {2048, 3072, 4000},                          /* ActualSeqQTensorData */
                    {1024, 2048, 5000})                          /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_035", true,               /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000102200112953UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 14, 1, 968, 512, 128,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BSND,    /* Dtype, Layout */
                    0.08838f, 0.9f, 65535, 0,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 6,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,          /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::B,                          /* PrefixShapeType */
                    {100, 0, 200}, {}, {})                       /* PrefixTensorData */
            ),
    FasCase("Fas_Redline_Redline_FullCore_Case_036", true,       /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, false),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000010220130943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 2, 1, 2048, 2048, 64,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,   /* Dtype, Layout */
                    0.08838f, 0.9f, 4096, 0,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::NONE,                     /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                   /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                       /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_037", true,           /* CaseName, Enable */
            "",                                              /* DebugInfo */
            OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                   ExpectInfo(true,                          /* ExpectSuccess */
                              10000000001022410943UL,        /* ExpectTilingKey */
                              5)),                           /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 1, 1026, 16,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::TND, /* Dtype, Layout */
                    0.08838f, 0.9f, 65535, 0,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 6,                                    /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                      /* PseShapeType */
                    DropMaskShapeType::NONE,                 /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,              /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                   /* AttentionMaskDtype */
                    PrefixShapeType::B,                      /* PrefixShapeType */
                    {300},                                   /* PrefixTensorData */
                    {1026},                                  /* ActualSeqQTensorData */
                    {1026})                                  /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_038", true,           /* CaseName, Enable */
            "",                                              /* DebugInfo */
            OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                   ExpectInfo(true,                          /* ExpectSuccess */
                              10000000001021110099UL,        /* ExpectTilingKey */
                              1)),                           /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 16, 16, 16,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BSH, /* Dtype, Layout */
                    0.08838f, 0.9f, 65535, 0,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 6,                                    /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                      /* PseShapeType */
                    DropMaskShapeType::NONE,                 /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,              /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                   /* AttentionMaskDtype */
                    PrefixShapeType::B,                      /* PrefixShapeType */
                    {8}, {}, {})                             /* PrefixTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_039", true,           /* CaseName, Enable */
            "",                                              /* DebugInfo */
            OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                   ExpectInfo(true,                          /* ExpectSuccess */
                              10000000010221110943UL,        /* ExpectTilingKey */
                              1)),                           /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 1, 1025, 16,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BSH, /* Dtype, Layout */
                    0.08838f, 0.9f, 65535, 0,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 6,                                    /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                      /* PseShapeType */
                    DropMaskShapeType::NONE,                 /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,              /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                   /* AttentionMaskDtype */
                    PrefixShapeType::B,                      /* PrefixShapeType */
                    {1025}, {}, {})                          /* PrefixTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_040", true,           /* CaseName, Enable */
            "",                                              /* DebugInfo */
            OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                   ExpectInfo(false,                         /* ExpectSuccess */
                              92UL,                          /* ExpectTilingKey */
                              1)),                           /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 1, 0, 16,                       /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BSH, /* Dtype, Layout */
                    0.08838f, 0.9f, 65535, 0,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 1,                                    /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                      /* PseShapeType */
                    DropMaskShapeType::NONE,                 /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,              /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                   /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                   /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_bf16_run_kernel_unpad)", /* CaseName */
            true, "",                                           /* Enable, DebugInfo */
            OpInfo(ControlInfo(true, true),                     /* RunTiling, RunKernel */
                   ExpectInfo(true,                             /* ExpectSuccess */
                              10000000001022430943UL,           /* ExpectTilingKey */
                              1)),                              /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 2048, 2048, 32,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,  /* Dtype, Layout */
                    0.08838f, 1.0f, 65535, 0,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 2,                                       /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                         /* PseShapeType */
                    DropMaskShapeType::NONE,                    /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                      /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                      /* PrefixShapeType */
                    {},                                         /* PrefixTensorData */
                    {256},                                      /* ActualSeqQTensorData */
                    {65})                                       /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_run_kernel_s1s2", true,   /* CaseName, Enable */
            "",                                                  /* DebugInfo */
            OpInfo(ControlInfo(true, true),                      /* RunTiling, RunKernel */
                   ExpectInfo(true,                              /* ExpectSuccess */
                              10000000011221212943UL,            /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 256, 1025, 32,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::SBH,     /* Dtype, Layout */
                    0.08838f, 0.9f, -200, 1024,                  /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                        /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                          /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,               /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                  /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                       /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                       /* PrefixShapeType */
                    {},                                          /* PrefixTensorData */
                    {256},                                       /* ActualSeqQTensorData */
                    {65})                                        /* ActualSeqKVTensorData */
            ),
    FasCase("Fas_Redline_SpecCore_Case_run_kernel_s1s2", true, /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, true),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000011221212943UL,          /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)),                             /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 256, 1025, 32,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::SBH,   /* Dtype, Layout */
                    0.08838f, 0.9f, -200, 1024,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,             /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_run_kernel_s1", true, /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, true),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000112200230953UL,          /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)),                             /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 256, 129, 32,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,   /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 65536,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 1,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,             /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
           ),
    FasCase("Fas_Redline_SpecCore_Case_run_kernel_s1_unAlign", true, /* CaseName, Enable */
            "",                                                /* DebugInfo */
            OpInfo(ControlInfo(true, true),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000112201230953UL,          /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)),                             /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 256, 257, 15,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 65536,              /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,             /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
           ),
    FasCase("Fas_Redline_SpecCore_Case_run_kernel_B_unAlign",/* CaseName */
            true, "",                                        /* Enable , DebugInfo */
            OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                   ExpectInfo(true,                          /* ExpectSuccess */
                              10000000001121130099UL,        /* ExpectTilingKey */
                              13)),                           /* ExpectTilingBlockDim */
            FaParam(49, 1, 1, 16, 16, 15,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH, /* Dtype, Layout */
                    0.08838f, 0.9f, 65535, 65535,            /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                    /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                      /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,           /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,              /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,               /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                   /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                   /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_run_kernel_s1s2_d_unAlign", /* CaseName*/
            true, "",                                          /* Enable, DebugInfo */
            OpInfo(ControlInfo(true, true),                    /* RunTiling, RunKernel */
                   ExpectInfo(true,                            /* ExpectSuccess */
                              10000000011221212943UL,          /* ExpectTilingKey */
                              ExpectInfo::kFullTilingBlockDim)),                             /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 256, 1025, 72,                    /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::SBH,   /* Dtype, Layout */
                    0.08838f, 0.9f, -200, 1024,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                      /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,             /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                    PrefixShapeType::NONE)                     /* PrefixShapeType */
            ),
    FasCase("Fas_Redline_SpecCore_Case_run_kernel_b_d_unAlign", true,           /* CaseName, Enable */
            "",                                              /* DebugInfo */
            OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                   ExpectInfo(true,                          /* ExpectSuccess */
                              10000000001021110099UL,        /* ExpectTilingKey */
                              1)),                           /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 16, 16, 5,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BSH, /* Dtype, Layout */
                    0.08838f, 0.9f, 65535, 0,                /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 6,                                    /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                      /* PseShapeType */
                    DropMaskShapeType::NONE,                 /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,              /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                   /* AttentionMaskDtype */
                    PrefixShapeType::B,                      /* PrefixShapeType */
                    {8}, {}, {})                             /* PrefixTensorData */
            )
);
INSTANTIATE_TEST_SUITE_P(Fas, Ts_Fas_Ascend910B2_Redline_SpecCore, Tc_Fas_Redline_SpecCore_Case);
INSTANTIATE_TEST_SUITE_P(Fas, Ts_Fas_Ascend910B3_Redline_SpecCore, Tc_Fas_Redline_SpecCore_Case);
