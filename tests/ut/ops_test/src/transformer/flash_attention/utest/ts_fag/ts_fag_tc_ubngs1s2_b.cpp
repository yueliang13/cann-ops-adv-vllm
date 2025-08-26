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
 * \file ts_fag_tc_ubngs1s2_b.cpp
 * \brief FlashAttentionScoreGrad 算子 Ubngs1s2Bbn 模板 UTest 用例.
 */

#include "ts_fag.h"

class Ts_Fag_Ascend910B2_Ubngs1s2Bb : public Ts_Fag_WithParam_Ascend910B2 {};

#define TEST_BIG_CASE true

TEST_P(Ts_Fag_Ascend910B2_Ubngs1s2Bb, Tc_BatchCase)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Ubngs1s2Bb_BatchCase = ::testing::Values(

    FagCase("Fag_Ubngs1s2Bb_Case_001_BNSD", true,                   /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000110123099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 16, 16, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_001_SBH", true,                    /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010113099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 16, 16, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_001", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010103099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 16, 16, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_001_AttenMask_1", true,            /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010103099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 16, 16, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_001_AttenMask_2", true,            /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010103099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 16, 16, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_001_Pse_1", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010103099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 16, 16, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_002", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010103099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 2, 1, 63, 17, 129,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_003", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010103099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 16, 16, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_004", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010103099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 16, 16, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_005", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010113099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(96, 1, 1, 64, 16, 129,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_006", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010113099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(48, 2, 2, 64, 16, 129,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_007", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000110123099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(48, 2, 2, 64, 16, 129,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_007", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010133099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(48, 2, 2, 64, 16, 129,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,     /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_008", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010102099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 2, 1, 63, 17, 129,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_009", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010112099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 2, 1, 63, 17, 129,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_010", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010132099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 2, 1, 63, 17, 129,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BSND,        /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_011", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000110122099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 2, 1, 63, 17, 129,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_012", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000011113099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(256, 1, 8, 16, 13, 33,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_013", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000011112099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(256, 1, 8, 16, 13, 33,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_Case_014", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010113099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 8, 16, 13, 33,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_TilingFailed_Case_001", true,           /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              10000000000010133099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(48, 2, 2, 64, 64, 129,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_TilingFailed_Case_002", true,           /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              10000000000110133099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(205, 25, 25, 16, 5979, 96,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_TilingFailed_Case_003", true,           /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              10000000000010133099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(205, 25, 25, 16, 5979, 96,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BSND,        /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_TilingFailed_Case_004", true,           /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              10000000000110133099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(4, 20, 20, 16, 16, 197,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_TilingFailed_Case_005", true,           /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              10000000000110133099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(4, 20, 20, 16, 16, 197,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ubngs1s2Bb_TilingFailed_Case_006", true,           /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              10000000000110133099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(4, 20, 20, 16, 16, 197,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_ALIBI_S1_S2,                 /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            )
#if TEST_BIG_CASE
    // 以下case测试大循环两次逻辑，但蓝区机器跑以下case会超时，黄区机器可以跑，开发有必要自测以下case
    ,
    FagCase("Fag_Ubngs1s2Bb_Case_000_Round2", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010103099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(97, 8, 1, 32, 32, 8,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.125f, 0.8f, 2048, 2048,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            )

    // 以下case测试大循环两次逻辑，但蓝区机器跑以下case会超时，黄区机器可以跑，开发有必要自测以下case
    ,
    FagCase("Fag_Ubngs1s2Bb_Case_000_SBH_Round2", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010113099UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(97, 8, 1, 32, 32, 8,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, Layout */
                    0.125f, 0.8f, 2048, 2048,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            )
#endif

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Ubngs1s2Bb, Tc_Fag_Ubngs1s2Bb_BatchCase);
