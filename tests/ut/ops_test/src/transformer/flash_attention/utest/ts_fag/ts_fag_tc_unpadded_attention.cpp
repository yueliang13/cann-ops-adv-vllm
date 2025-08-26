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
 * \file ts_fag_tc_unpadded_attention.cpp
 * \brief FlashAttentionScoreGrad 算子 UnpaddedAttention UTest 用例.
 */

#include "ts_fag.h"

class Ts_Fag_Ascend910B2_UnpaddedAttention : public Ts_WithParam_Ascend910B2<FagCase> {};

TEST_P(Ts_Fag_Ascend910B2_UnpaddedAttention, Tc_BatchCase)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_UnpaddedAttention_BatchCase = ::testing::Values(

    FagCase("Fag_UnpaddedAttention_Case_TND_000", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000001101033434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 4096, 130, 64,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {4096, 8192, 12288},                            /* ActualSeqQTensorData */
                    {130, 260, 390}),                               /* ActualSeqKVTensorData */

            FaCase::kTilingTemplatePriority_Invalid /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_TND_001", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000001101033434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 4096, 64, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {4096, 8192, 12288},                            /* ActualSeqQTensorData */
                    {64, 128, 192}),                                /* ActualSeqKVTensorData */

            FaCase::kTilingTemplatePriority_Invalid /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_TND_002", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011101033434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 4096, 64, 72,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {4096, 8192, 12288},                            /* ActualSeqQTensorData */
                    {64, 128, 192}),                                /* ActualSeqKVTensorData */

            FaCase::kTilingTemplatePriority_Invalid /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_000", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000001101033434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 28, 28, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {28, 41, 57},                                   /* ActualSeqQTensorData */
                    {28, 46, 56}),                                  /* ActualSeqKVTensorData */

            FaCase::kTilingTemplatePriority_Invalid /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_001", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000001101033434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2800, 2800, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, -10, 100,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2800, 4100, 5700},                             /* ActualSeqQTensorData */
                    {2800, 4600, 5600}),                            /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_002", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000001101033434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 28, 28, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 2,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {28, 41, 57},                                   /* ActualSeqQTensorData */
                    {28, 46, 56}),                                  /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_003", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000001101033434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 28, 28, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 2,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {28, 41, 57},                                   /* ActualSeqQTensorData */
                    {28, 46, 56}),                                  /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_004", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000001101033434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 28, 28, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 4,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {28, 41, 57},                                   /* ActualSeqQTensorData */
                    {28, 46, 56}),                                  /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_005", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000111033434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 2,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_ALIBI_S1_S2,                 /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 2321, 2515},                             /* ActualSeqQTensorData */
                    {2048, 2321, 2515}),                            /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_006", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000111033434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 3,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::_1_N1_ALIBI_S1_S2,                /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 2321, 2515},                             /* ActualSeqQTensorData */
                    {2048, 2321, 2515}),                            /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_007", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000101032434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::TND,         /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 4,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 2321, 2515},                             /* ActualSeqQTensorData */
                    {2048, 2321, 2515}),                            /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_008", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000001101033434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(5, 2, 3, 100, 50, 64,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {100, 120, 130, 230, 234},                      /* ActualSeqQTensorData */
                    {50, 60, 65, 115, 117}),                        /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_009", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000101032434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::TND,         /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 7,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 2321, 2515},                             /* ActualSeqQTensorData */
                    {2048, 2321, 2515}),                            /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_010", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000101032434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::TND,         /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 8,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 2321, 2515},                             /* ActualSeqQTensorData */
                    {2048, 2321, 2515}),                            /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_011", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000101032434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(4, 1, 1, 128, 128, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::TND,         /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 65535,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 6,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::B,                             /* PrefixShapeType */
                    {32, 64, 32, 64},                               /* PrefixTensorData */
                    {64, 192, 256, 384},                            /* ActualSeqQTensorData */
                    {64, 192, 256, 384}),                           /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_012", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000111033434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 2,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::TND_1S,                           /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 2321, 2515},                             /* ActualSeqQTensorData */
                    {2048, 2321, 2515}),                            /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_013", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000111033434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 2,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::TND_SS,                           /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 2321, 2515},                             /* ActualSeqQTensorData */
                    {2048, 2321, 2515}),                            /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_014", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              10000000000101032434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::TND,         /* Dtype, Layout */
                    0.08838f, 0.8f, -10, -10,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 2321, 2515},                             /* ActualSeqQTensorData */
                    {2048, 2321, 2515}),                            /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_015", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              10000000000101032434UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::TND,         /* Dtype, Layout */
                    0.08838f, 0.8f, -100, -100,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 4,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 2321, 2515},                             /* ActualSeqQTensorData */
                    {2048, 2321, 2515}),                            /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_016", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011010000134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 64, 64, 32,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 65535,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {64},                                           /* ActualSeqQTensorData */
                    {64}),                                          /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ));
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_UnpaddedAttention, Tc_Fag_UnpaddedAttention_BatchCase);

class Ts_Fag_Ascend910B2_UnpaddedAttention_InvalidParam : public Ts_Fag_Ascend910B2_UnpaddedAttention {};

TEST_P(Ts_Fag_Ascend910B2_UnpaddedAttention_InvalidParam, Tc_seqlen_nullptr)
{
    ASSERT_TRUE(case_->Init());
    case_->mParam.actualSeqQLen = Tensor("actualSeqQLen", {}, "None", ge::DataType::DT_INT64, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_UnpaddedAttention_InvalidParam, Tc_seqlen_dimerr)
{
    ASSERT_TRUE(case_->Init());
    case_->mParam.actualSeqQLen =
        Tensor("actualSeqQLen", {case_->mParam.b + 1}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_UnpaddedAttention_InvalidParam_BatchCase = ::testing::Values(

    FagCase("Fag_UnpaddedAttention_Case_000", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 28, 28, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {28, 41, 57},                                   /* ActualSeqQTensorData */
                    {28, 46, 56}),                                  /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_UnpaddedAttention_InvalidParam,
                         Tc_Fag_UnpaddedAttention_InvalidParam_BatchCase);

TEST_F(Ts_Fag_Ascend910B2, Tc_seqlen_dim_long)
{
    auto cur = FagCase("Fag_UnpaddedAttention_Case_000", true,                 /* CaseName, Enable */
                       "",                                                     /* DebugInfo */
                       OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                              ExpectInfo(true,                                 /* ExpectSuccess */
                                         10000000001101033434UL,               /* ExpectTilingKey */
                                         ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                       FaParam(2049, 2, 1, 32, 32, 32,                         /* B, N2, G, S1, S2, D */
                               ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                               0.08838f, 0.8f, 65535, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                               1, 0,                              /* InnerPrecise, SparseMode */
                               PseShapeType::NONE,                /* PseShapeType */
                               DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                               PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                               AttenMaskShapeType::S1_S2,         /* AttentionMaskShapeType */
                               ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                               PrefixShapeType::NONE,             /* PrefixShapeType */
                               {},                                /* PrefixTensorData */
                               {},                                /* ActualSeqQTensorData */
                               {}),                               /* ActualSeqKVTensorData */
                       FaCase::kTilingTemplatePriority_Invalid    /* TilingTemplatePriority */
    );
    int64_t tmpData = 32;
    for (int64_t i = 0; i < 2049; i++) {
        cur.mParam.actualSeqQLenTensorData.push_back(tmpData);
        cur.mParam.actualSeqKVLenTensorData.push_back(tmpData);
    }
    ASSERT_TRUE(cur.Init());
    ASSERT_EQ(cur.Run(), (cur.mReverse.mExp.mSuccess));
}
