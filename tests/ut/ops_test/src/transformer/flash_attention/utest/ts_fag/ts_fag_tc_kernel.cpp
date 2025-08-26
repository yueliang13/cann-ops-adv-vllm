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
 * \file ts_fag_tc_kernel.cpp
 * \brief FlashAttentionScoreGrad 反向用例.
 */

#include "ts_fag.h"

class Ts_Fag_Ascend910B2_Kernel : public Ts_Fag_WithParam_Ascend910B2 {};

TEST_P(Ts_Fag_Ascend910B2_Kernel, Tc_Fag_Kernel_Case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Kernel_Case = ::testing::Values(

    FagCase("Fag_Kernel_Case_000", true,                            /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 128, 128, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 1.0f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            Case::kTilingTemplatePriority_Invalid                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Kernel_Case_001", true,                            /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 128, 128, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype, Layout */
                    0.08838f, 1.0f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            Case::kTilingTemplatePriority_Invalid                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Kernel_Case_003", true,                            /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 1, 1024, 16,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 1.0f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            Case::kTilingTemplatePriority_Invalid                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Kernel_Case_004", true,                            /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, true, true),                   /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 1, 1024, 16,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 65535,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 6,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::B,                             /* PrefixShapeType */
                    {32},                                           /* PrefixTensorData */
                    {1},                                            /* ActualSeqQTensorData */
                    {1024}),                                        /* ActualSeqKVTensorData */
            Case::kTilingTemplatePriority_Invalid                   /* TilingTemplatePriority */
            )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Kernel, Tc_Fag_Kernel_Case);
