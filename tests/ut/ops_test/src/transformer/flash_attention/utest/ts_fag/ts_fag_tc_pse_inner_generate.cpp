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
 * \file ts_fag_tc_pse_inner_generate.cpp
 * \brief FlashAttentionScoreGrad 反向pse内生成用例.
 */

#include "ts_fag.h"

class Ts_Fag_Ascend910B2_PseInnerGenerate : public Ts_Fag_WithParam_Ascend910B2 {};

TEST_P(Ts_Fag_Ascend910B2_PseInnerGenerate, Tc_Fag_PseInnerGenerate_Case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_PseInnerGenerate_Case = ::testing::Values(

    FagCase("Fag_PseInnerGenerate_Case_000", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 16, 1, 256, 256, 32,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_N1,                         /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {},                                             /* ActualSeqQTensorData */
                    {},                                             /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2              /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_001", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(4, 16, 1, 256, 256, 32,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_N1,                         /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {},                                             /* ActualSeqQTensorData */
                    {},                                             /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_002", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 16, 1, 32, 32, 32,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 1,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {},                                             /* ActualSeqQTensorData */
                    {},                                             /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Ungs1s2_Bbn                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_003", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 12, 1, 16, 16, 32,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 1,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {},                                             /* ActualSeqQTensorData */
                    {},                                             /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_004", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 16, 1, 256, 256, 32,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_N1,                         /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {256, 256},                                     /* ActualSeqQTensorData */
                    {256, 256},                                     /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2              /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_005", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(4, 16, 1, 256, 256, 32,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_B_N1,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {},                                             /* ActualSeqQTensorData */
                    {},                                             /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_006", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(4, 16, 1, 256, 256, 32,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {},                                             /* ActualSeqQTensorData */
                    {},                                             /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_007", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false, true),                  /* RunTiling, RunKernel, Deterministic */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 1, 33, 33, 32,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::TND,        /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_N1,                         /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {15, 15},                                       /* ActualSeqQTensorData */
                    {15, 15},                                       /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_008", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false, true),                  /* RunTiling, RunKernel, Deterministic */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 1, 33, 33, 32,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::TND,        /* Dtype, Layout */
                    0.08838f, 1.0f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_N1,                         /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {15, 15},                                       /* ActualSeqQTensorData */
                    {15, 15},                                       /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_009", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false, true),                  /* RunTiling, RunKernel, Deterministic */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 1, 33, 33, 32,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::TND,        /* Dtype, Layout */
                    0.08838f, 1.0f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 1,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {15, 15},                                       /* ActualSeqQTensorData */
                    {15, 15},                                       /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_010", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false, true),                  /* RunTiling, RunKernel, Deterministic */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 1, 131, 131, 32,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::TND,        /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 1,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2, 129},                                       /* ActualSeqQTensorData */
                    {2, 129},                                       /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_011", false,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false, true),                  /* RunTiling, RunKernel, Deterministic */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 1, 131, 131, 32,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::TND,        /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_N1,                         /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2, 129},                                       /* ActualSeqQTensorData */
                    {2, 129},                                       /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_012", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false, true),       /* RunTiling, RunKernel, Deterministic */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 1, 33, 33, 32,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::TND,        /* Dtype, Layout */
                    0.08838f, 1.0f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_N1,                         /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {15, 15},                                       /* ActualSeqQTensorData */
                    {15, 15},                                       /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_013", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 12, 1, 16, 16, 32,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.9f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 3,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::SLOPE_N1,                         /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {},                                             /* ActualSeqQTensorData */
                    {},                                             /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Ubngs1s2_Bb                  /* TilingTemplatePriority */
            ),
    FagCase("Fag_PseInnerGenerate_Case_014", true,                  /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false, false),                 /* RunTiling, RunKernel, Deterministic */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 1, 33, 33, 72,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 1.0f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0, 1,                                        /* InnerPrecise, Sparsemode, pseType */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2,                  /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType*/
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype*/
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {15, 15},                                       /* ActualSeqQTensorData */
                    {15, 15},                                       /* ActualSeqKVTensorData */
                    {1},                                            /* qStartIdxTensorData */
                    {2}),                                           /* kvStartIdxTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2              /* TilingTemplatePriority */
            )

);

INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_PseInnerGenerate, Tc_Fag_PseInnerGenerate_Case);
