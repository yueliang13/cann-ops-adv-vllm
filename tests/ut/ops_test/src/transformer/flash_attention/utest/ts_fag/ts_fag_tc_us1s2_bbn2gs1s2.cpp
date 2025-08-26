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
 * \file ts_fag_tc_us1s2_bbn2gs1s2.cpp
 * \brief FlashAttentionScoreGrad 算子 Us1s2Bbn2gs1s2 模板 UTest 用例.
 */

#include "ts_fag.h"
#include "tests/utils/log.h"
#include "tiling/fa/tiling_data.h"

class FagCaseUs1s2Bbn2gs1s2 : public FagCase {
public:
    bool mChkTilingData = true;
    uint32_t mIsSparseValue = 1;

    FagCaseUs1s2Bbn2gs1s2() = default;
    FagCaseUs1s2Bbn2gs1s2(const char *name, bool enable, const char *dbgInfo, const OpInfo &reverse,
                          const FaParam &param, bool chkTilingData = true, uint32_t isSparseValue = 1)
        : FagCase(name, enable, dbgInfo, reverse, param, FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2),
          mChkTilingData(chkTilingData), mIsSparseValue(isSparseValue)
    {
    }

    bool Run() override
    {
        if (!mEnable) {
            return true;
        }
        if (!mReverse.ProcessTiling(mName)) {
            return false;
        }
        if (mChkTilingData) {
            auto *td = (const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *)(mReverseCtx.GetTilingData());
            if (td->s1s2BNGS1S2BaseParams.isSparse != mIsSparseValue) {
                LOG_IF(mReverse.mExp.mSuccess,
                       LOG_ERR("Case[%s:%s] TilingData check failed(isSparse), Exp=%u, Act=%u", mName.c_str(),
                               mReverse.mName.c_str(), mIsSparseValue, td->s1s2BNGS1S2BaseParams.isSparse));
                return false;
            }
        }
        if (!mReverse.ProcessKernel(mName)) {
            return false;
        }
        return true;
    }
};

class Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_TilingFailed : public Ts_Ascend910B2<FagCaseUs1s2Bbn2gs1s2> {};

TEST_F(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_TilingFailed, Tc_Us1s2Bbn2gs1s2_IllegalPseShape_001)
{
    FagCaseUs1s2Bbn2gs1s2 cs("Tc_Us1s2Bbn2gs1s2_IllegalPseShape_001", true,          /* CaseName, Enable */
                             "",                                                     /* DebugInfo */
                             OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                    ExpectInfo(false,                                /* ExpectSuccess */
                                               ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                               ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                             FaParam(2, 2, 2, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                     ge::DataType::DT_FLOAT, LayoutType::BSND,       /* Dtype, Layout */
                                     0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                     1, 3,                              /* InnerPrecise, SparseMode */
                                     PseShapeType::_1_N1_ALIBI_S1_S2,   /* PseShapeType */
                                     DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                     PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                     AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                     ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                     PrefixShapeType::NONE)             /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init());
    cs.mParam.pse = Tensor("pse", {1, 2, 3, 4, 5}, "1_2_3_4_5", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mReverse.mExp.mSuccess);
}

TEST_F(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_TilingFailed, Tc_Us1s2Bbn2gs1s2_IllegalPseShape_002)
{
    FagCaseUs1s2Bbn2gs1s2 cs("Tc_Us1s2Bbn2gs1s2_IllegalPseShape_002", true,          /* CaseName, Enable */
                             "",                                                     /* DebugInfo */
                             OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                    ExpectInfo(false,                                /* ExpectSuccess */
                                               ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                               ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                             FaParam(2, 2, 2, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                     ge::DataType::DT_FLOAT, LayoutType::BSND,       /* Dtype, Layout */
                                     0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                     1, 3,                              /* InnerPrecise, SparseMode */
                                     PseShapeType::_1_N1_ALIBI_S1_S2,   /* PseShapeType */
                                     DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                     PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                     AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                     ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                     PrefixShapeType::NONE)             /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init());
    cs.mParam.pse = Tensor("pse", {16, 16, 16, 16}, "16_16_16_16", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mReverse.mExp.mSuccess);
}

class Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2 : public Ts_WithParam_Ascend910B2<FagCaseUs1s2Bbn2gs1s2> {};

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2, Tc_BatchCase)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Us1s2Bbn2gs1s2_BatchCase = ::testing::Values(
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_000", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000101003434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::S1_S2,         /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_001", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000101003434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_1_S1_S2,     /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_002", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000101003434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 1, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_003", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111012434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 3, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  0, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_1_S2,           /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_1_S1_S2,     /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_004", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111001434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 2, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSND,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::S1_S2,         /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_005", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111001434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 2, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSND,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 2,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::_1_N1_S1_S2,         /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_1_S1_S2,     /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_006", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111001434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 2, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSND,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 3,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_ALIBI_S1_S2,    /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_007", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111021434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 5,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::_1_N1_ALIBI_S1_S2,   /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_1_S1_S2,     /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::B,                /* PrefixShapeType */
                                  {1, 2},                            /* PrefixTensorData */
                                  {},                                /* ActualSeqQTensorData */
                                  {})                                /* ActualSeqKVTensorData */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_008", true,                    /* CaseName, Enable */
                          "compress atten_mask mode not support s1 s2 2048 2048", /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(false,                                /* ExpectSuccess */
                                            10000000000011001434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 2, 1024, 1024, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSND,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 2,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::S1_S2,         /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_009", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000101003434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 2, 2048, 128, 64,                        /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::S1_S2,         /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_010", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000101003434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 2, 2048, 256, 64,                        /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::S1_S2,         /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_011", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000101003434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 1, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                                  0.08838f, 0.8f, 100, 100,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 4,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::SPARSE,        /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_012", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000101003434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                                  0.08838f, 0.8f, 100, 100,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 4,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::SPARSE,        /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_013", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(false,                                /* ExpectSuccess */
                                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                                  0.08838f, 0.8f, -100, 10,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_014", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111021434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 1,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE),            /* PrefixShapeType */
                          true, 0                                    /* CheckTilingData, isSparseValue */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_015", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111021434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 128, 128,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_016", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111021434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 2050, 0,           /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_017", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111021434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 2050, 2050,        /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE),            /* PrefixShapeType */
                          true, 0                                    /* CheckTilingData, isSparseValue */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_018", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111021434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 2050, 2050,        /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 2,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_019", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111021434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 2050, 2050,        /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 3,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_020", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111021434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 128, 128, 128,                         /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 2050, 2050,        /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 4,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::SPARSE,        /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE),            /* PrefixShapeType */
                          true, 0                                    /* CheckTilingData, isSparseValue */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_021", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111021434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 100, 2050,         /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 4,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::SPARSE,        /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_022", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000111021434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 100, 2050,         /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 4,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::SPARSE,        /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_023", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000101001434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(4, 10, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSH,        /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 65504,       /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 6,                               /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                 /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8,  /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,        /* PaddingMaskShapeType */
                                  AttenMaskShapeType::PREFIXCOMPRESS, /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,              /* AttentionMaskDtype */
                                  PrefixShapeType::B,                 /* PrefixShapeType */
                                  {117, 319, 312, 334},               /* PrefixTensorData */
                                  {},                                 /* ActualSeqQTensorData */
                                  {})                                 /* ActualSeqKVTensorData */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_024", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(true,                                 /* ExpectSuccess */
                                            10000000000101001434UL,               /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 1, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSH,        /* Dtype, Layout */
                                  0.08838f, 0.8f, 100, 100,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 4,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::SPARSE,        /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_025", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(false,                                /* ExpectSuccess */
                                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(1, 1, 1, 256, 128, 128,                         /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 8,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::S1_S2,         /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2, Tc_Fag_Us1s2Bbn2gs1s2_BatchCase);

class Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape : public Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2 {};

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_SoftmaxMax_Invalid)
{
    ASSERT_TRUE(case_->Init());
    case_->mParam.softmaxMax = Tensor("softmaxMax", {case_->mParam.b, case_->mParam.n1, case_->mParam.s1, 1},
                                      "B_N1_S1_1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_SoftmaxSum_Invalid)
{
    ASSERT_TRUE(case_->Init());
    case_->mParam.softmaxSum = Tensor("softmaxSum", {case_->mParam.b, case_->mParam.n1, case_->mParam.s1, 1},
                                      "B_N1_S1_1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_SoftmaxMax_DimErr)
{
    ASSERT_TRUE(case_->Init());
    case_->mParam.softmaxMax = Tensor("softmaxMax", {case_->mParam.b, case_->mParam.n1, case_->mParam.s1},
                                      "B_N1_S1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_SoftmaxSum_DimErr)
{
    ASSERT_TRUE(case_->Init());
    case_->mParam.softmaxSum = Tensor("softmaxSum", {case_->mParam.b, case_->mParam.n1, case_->mParam.s1},
                                      "B_N1_S1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_AttenmaskErr)
{
    ASSERT_TRUE(case_->Init());
    case_->mParam.attenMask =
        Tensor("attenMask", {case_->mParam.s1, 1}, "S1_1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_AttenmaskErrDim)
{
    ASSERT_TRUE(case_->Init());
    case_->mParam.attenMask = Tensor("attenMask", {1, case_->mParam.s1, 1, case_->mParam.s2}, "1_S1_1_S2(Invalid)",
                                     case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_AttenmaskErrDimNum)
{
    ASSERT_TRUE(case_->Init());
    case_->mParam.attenMask =
        Tensor("attenMask", {1, case_->mParam.s1, 1}, "1_S1_1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Us1s2Bbn2gs1s2_InvalidShape_BatchCase = ::testing::Values(

    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_000", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(false,                                /* ExpectSuccess */
                                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 1, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape,
                         Tc_Fag_Us1s2Bbn2gs1s2_InvalidShape_BatchCase);

class Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidPrefixCompressShape : public Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2 {};

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidPrefixCompressShape, Tc_AttenmaskErrShapeForPrefixCompress)
{
    ASSERT_TRUE(case_->Init());
    case_->mParam.attenMask =
        Tensor("attenMask", {2048, 2048}, "not_3072_2048(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidPrefixCompressShape, Tc_PrefixErrShapeForPrefixCompress)
{
    ASSERT_TRUE(case_->Init());
    case_->mParam.prefix =
        Tensor("prefix", {110, 110, 110, 110, 110}, "prefixN_gt_B(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Us1s2Bbn2gs1s2_InvalidPrefixCompressShape_BatchCase = ::testing::Values(

    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_001", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfo(false,                                /* ExpectSuccess */
                                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(4, 10, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSH,        /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 65504,       /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 6,                               /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                 /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8,  /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,        /* PaddingMaskShapeType */
                                  AttenMaskShapeType::PREFIXCOMPRESS, /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,              /* AttentionMaskDtype */
                                  PrefixShapeType::B,                 /* PrefixShapeType */
                                  {117, 319, 312, 334},               /* PrefixTensorData */
                                  {},                                 /* ActualSeqQTensorData */
                                  {})                                 /* ActualSeqKVTensorData */
                          )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidPrefixCompressShape,
                         Tc_Fag_Us1s2Bbn2gs1s2_InvalidPrefixCompressShape_BatchCase);
