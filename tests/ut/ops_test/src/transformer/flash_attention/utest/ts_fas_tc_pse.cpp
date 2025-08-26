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
 * \file ts_fa_tc_pse.cpp
 * \brief FlashAttentionScore 正反向用例.
 */

#include "ts_fa.h"

class Ts_Fa_Ascend910B2_Pse : public Ts_Fa_WithParam_Ascend910B2 {};

TEST_P(Ts_Fa_Ascend910B2_Pse, Tc_Case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mForward.mExp.mSuccess);
}

const auto Tc_Fa_Pse_Case = ::testing::Values(

    FaCase("Fa_Pse_Tc_000", true,                                  /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(3, 1, 1, 16, 16, 16,                            /* B, N2, G, S1, S2, D */
                   ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                   0.08838f, 1.0f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                   0, 8, 2,                                        /* InnerPrecise, SparseMode, PseType */
                   PseShapeType::SLOPE_B_N1,                       /* PseShapeType */
                   DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE,                          /* PrefixShapeType */
                   {},                                             /* PrefixTensorData */
                   {1, 2, 3},                                      /* ActualSeqQLenTensorData */
                   {1, 2, 3},                                      /* actualSeqKVLenTensorData */
                   {},                                             /* QStartIdxTensorData */
                   {})                                             /* KVStartIdxTensorData */
           ),
    FaCase("Fa_Pse_Tc_001", true,                                  /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(3, 1, 1, 16, 16, 16,                            /* B, N2, G, S1, S2, D */
                   ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                   0.08838f, 1.0f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                   0, 8, 3,                                        /* InnerPrecise, SparseMode, PseType */
                   PseShapeType::SLOPE_B_N1,                       /* PseShapeType */
                   DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE,                          /* PrefixShapeType */
                   {},                                             /* PrefixTensorData */
                   {1, 2, 3},                                      /* ActualSeqQLenTensorData */
                   {1, 2, 3},                                      /* actualSeqKVLenTensorData */
                   {},                                             /* QStartIdxTensorData */
                   {})                                             /* KVStartIdxTensorData */
           ),
    FaCase("Fa_Pse_Tc_002", true,                                  /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(3, 1, 1, 16, 16, 16,                            /* B, N2, G, S1, S2, D */
                   ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                   0.08838f, 1.0f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                   0, 7, 2,                                        /* InnerPrecise, SparseMode, PseType */
                   PseShapeType::SLOPE_B_N1,                       /* PseShapeType */
                   DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE,                          /* PrefixShapeType */
                   {},                                             /* PrefixTensorData */
                   {1, 2, 3},                                      /* ActualSeqQLenTensorData */
                   {1, 2, 3},                                      /* actualSeqKVLenTensorData */
                   {},                                             /* QStartIdxTensorData */
                   {})                                             /* KVStartIdxTensorData */
           ),
    FaCase("Fa_Pse_Tc_003", true,                                  /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(3, 1, 1, 16, 16, 16,                            /* B, N2, G, S1, S2, D */
                   ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                   0.08838f, 1.0f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                   0, 7, 3,                                        /* InnerPrecise, SparseMode, PseType */
                   PseShapeType::SLOPE_B_N1,                       /* PseShapeType */
                   DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE,                          /* PrefixShapeType */
                   {},                                             /* PrefixTensorData */
                   {1, 2, 3},                                      /* ActualSeqQLenTensorData */
                   {1, 2, 3},                                      /* actualSeqKVLenTensorData */
                   {},                                             /* QStartIdxTensorData */
                   {})                                             /* KVStartIdxTensorData */
           ),
    FaCase("Fa_Pse_Tc_004", true,                                  /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(3, 1, 1, 16, 16, 16,                            /* B, N2, G, S1, S2, D */
                   ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                   0.08838f, 1.0f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                   0, 8, 2,                                        /* InnerPrecise, SparseMode, PseType */
                   PseShapeType::SLOPE_N1,                         /* PseShapeType */
                   DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE,                          /* PrefixShapeType */
                   {},                                             /* PrefixTensorData */
                   {1, 2, 3},                                      /* ActualSeqQLenTensorData */
                   {1, 2, 3},                                      /* actualSeqKVLenTensorData */
                   {},                                             /* QStartIdxTensorData */
                   {})                                             /* KVStartIdxTensorData */
           ),
    FaCase("Fa_Pse_Tc_005", true,                                  /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(3, 1, 1, 16, 16, 16,                            /* B, N2, G, S1, S2, D */
                   ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                   0.08838f, 1.0f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                   0, 8, 3,                                        /* InnerPrecise, SparseMode, PseType */
                   PseShapeType::SLOPE_N1,                         /* PseShapeType */
                   DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE,                          /* PrefixShapeType */
                   {},                                             /* PrefixTensorData */
                   {1, 2, 3},                                      /* ActualSeqQLenTensorData */
                   {1, 2, 3},                                      /* actualSeqKVLenTensorData */
                   {},                                             /* QStartIdxTensorData */
                   {})                                             /* KVStartIdxTensorData */
           ),
    FaCase("Fa_Pse_Tc_006", true,                                  /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(3, 1, 1, 16, 16, 16,                            /* B, N2, G, S1, S2, D */
                   ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                   0.08838f, 1.0f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                   0, 7, 2,                                        /* InnerPrecise, SparseMode, PseType */
                   PseShapeType::SLOPE_N1,                         /* PseShapeType */
                   DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE,                          /* PrefixShapeType */
                   {},                                             /* PrefixTensorData */
                   {1, 2, 3},                                      /* ActualSeqQLenTensorData */
                   {1, 2, 3},                                      /* actualSeqKVLenTensorData */
                   {},                                             /* QStartIdxTensorData */
                   {})                                             /* KVStartIdxTensorData */
           ),
    FaCase("Fa_Pse_Tc_007", true,                                  /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(3, 1, 1, 16, 16, 16,                            /* B, N2, G, S1, S2, D */
                   ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                   0.08838f, 1.0f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                   0, 7, 3,                                        /* InnerPrecise, SparseMode, PseType */
                   PseShapeType::SLOPE_N1,                         /* PseShapeType */
                   DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE,                          /* PrefixShapeType */
                   {},                                             /* PrefixTensorData */
                   {1, 2, 3},                                      /* ActualSeqQLenTensorData */
                   {1, 2, 3},                                      /* actualSeqKVLenTensorData */
                   {},                                             /* QStartIdxTensorData */
                   {})                                             /* KVStartIdxTensorData */
           ),
    FaCase("Fa_Pse_Tc_008", true,                                  /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(3, 1, 1, 32, 32, 16,                            /* B, N2, G, S1, S2, D */
                   ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                   0.08838f, 1.0f, 65535, 65535,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                   0, 8, 2,                                        /* InnerPrecise, SparseMode, PseType */
                   PseShapeType::SLOPE_B_N1,                       /* PseShapeType */
                   DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE,                          /* PrefixShapeType */
                   {},                                             /* PrefixTensorData */
                   {1, 3, 6, 8},                                   /* ActualSeqQLenTensorData */
                   {1, 2, 4, 8},                                   /* actualSeqKVLenTensorData */
                   {},                                             /* QStartIdxTensorData */
                   {})                                             /* KVStartIdxTensorData */
           ),
    FaCase("Fa_Pse_Tc_009", true,                                  /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(3, 1, 1, 32, 32, 16,                            /* B, N2, G, S1, S2, D */
                   ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                   0.08838f, 1.0f, 65535, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                   0, 8, 2,                                        /* InnerPrecise, SparseMode, PseType */
                   PseShapeType::SLOPE_B_N1,                       /* PseShapeType */
                   DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE,                          /* PrefixShapeType */
                   {},                                             /* PrefixTensorData */
                   {3, 5, 7, 8},                                   /* ActualSeqQLenTensorData */
                   {1, 3, 6, 9},                                   /* actualSeqKVLenTensorData */
                   {},                                             /* QStartIdxTensorData */
                   {})                                             /* KVStartIdxTensorData */
           ),
    FaCase("Fa_Pse_Tc_010", true,                                  /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(3, 1, 1, 32, 32, 16,                            /* B, N2, G, S1, S2, D */
                   ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                   0.08838f, 1.0f, 65535, 65535,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                   0, 8, 3,                                        /* InnerPrecise, SparseMode, PseType */
                   PseShapeType::SLOPE_N1,                         /* PseShapeType */
                   DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE,                          /* PrefixShapeType */
                   {},                                             /* PrefixTensorData */
                   {3, 1, 6, 3},                                   /* ActualSeqQLenTensorData */
                   {4, 2, 1, 2},                                   /* actualSeqKVLenTensorData */
                   {},                                             /* QStartIdxTensorData */
                   {})                                             /* KVStartIdxTensorData */
           ),
    FaCase("Fa_Pse_Tc_011", true,                                  /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(3, 1, 1, 32, 32, 16,                            /* B, N2, G, S1, S2, D */
                   ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                   0.08838f, 1.0f, 65535, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                   0, 8, 3,                                        /* InnerPrecise, SparseMode, PseType */
                   PseShapeType::SLOPE_N1,                         /* PseShapeType */
                   DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE,                          /* PrefixShapeType */
                   {},                                             /* PrefixTensorData */
                   {5, 4, 2, 4},                                   /* ActualSeqQLenTensorData */
                   {4, 6, 5, 3},                                   /* actualSeqKVLenTensorData */
                   {},                                             /* QStartIdxTensorData */
                   {})                                             /* KVStartIdxTensorData */
           )
);
INSTANTIATE_TEST_SUITE_P(Fas, Ts_Fa_Ascend910B2_Pse, Tc_Fa_Pse_Case);
