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
 * \file ts_fag_tc_ungs1s2_bbn.cpp
 * \brief FlashAttentionScoreGrad 算子 Ungs1s2Bbn 模板 UTest 用例.
 */

#include "ts_fag.h"

class Ts_Fag_Ascend910B2_Ungs1s2Bbn : public Ts_Fag_WithParam_Ascend910B2 {};

TEST_P(Ts_Fag_Ascend910B2_Ungs1s2Bbn, Tc_BatchCase)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Ungs1s2Bbn_BatchCase = ::testing::Values(

    FagCase("Fag_Ungs1s2Bbn_Case_001_BNSD", true,                   /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000000023199UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 2, 2, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ungs1s2Bbn_Case_001_SBH", true,                    /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000000013199UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 2, 2, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ungs1s2Bbn_Case_001", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000000003199UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 2, 2, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ungs1s2Bbn_Case_001_AttenMask_1", true,            /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000000003199UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 2, 2, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ungs1s2Bbn_Case_001_AttenMask_2", true,            /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000000003199UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 2, 2, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ungs1s2Bbn_Case_001_Pse_1", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010003199UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 4, 4, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ungs1s2Bbn_Case_002", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000000023199UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 2, 2, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ungs1s2Bbn_Case_003", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000001013199UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(24, 102, 1, 2, 2, 60,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ungs1s2Bbn_Case_004", true,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011111010134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(24, 102, 1, 2, 2, 60,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 4,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTilingTemplatePriority_Invalid                /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ungs1s2Bbn_TilingFailed_Case_001", true,           /* CaseName, Enable */
            "atten mask shape error",                               /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 2, 2, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 2,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ungs1s2Bbn_TilingFailed_Case_002", true,           /* CaseName, Enable */
            "atten mask shape error",                               /* DebugInfo */
            OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(16, 16, 1, 2, 2, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 3,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ungs1s2Bbn_Case_kernel_001", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010003199UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(49, 9, 1, 32, 32, 8,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ungs1s2Bbn_Case_kernel_001_SBH", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000010013199UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(49, 9, 1, 32, 32, 8,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_Ungs1s2Bbn_Case_kernel_001_SBH", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),             /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000000110023199UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(49, 9, 1, 32, 32, 8,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Ungs1s2Bbn, Tc_Fag_Ungs1s2Bbn_BatchCase);
