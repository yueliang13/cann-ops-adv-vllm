/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_grad_tiling_s1s2_bn2.h
 * \brief
 */

#pragma once

#include "flash_attention_score_grad_tiling_common.h"
#include "tiling/tiling_base.h"
#include "tiling/tiling_type.h"
#include "flash_attention_score_grad_tiling_s1s2_bn2_def.h"

namespace optiling {

constexpr uint32_t CORE_LIST_NUM = 50;
struct TempParamsS1s2Bn2 {
    uint32_t dataTypeSize;
    uint32_t mask;
    uint32_t queryType;
    uint32_t branch;
    uint32_t calcMode;
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    uint32_t layout;
    int64_t pseType = PSE_OUTER_ADD_MUL_TYPE;
    int64_t pseAlibiBaseS1 = 0;
    int64_t pseAlibiBaseS2 = 0;
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    int64_t bN2idxStarts[CORE_LIST_NUM];
    int64_t bN2idxEnds[CORE_LIST_NUM];
    std::vector<int64_t> prefixN;
    std::vector<int64_t> actualSeqQlen;
    std::vector<int64_t> actualSeqKvlen;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t sumS1S2Product = 0;
    int64_t dropMaskSize = 0;
    std::vector<int64_t> s1s2Weight;
    PseConfig pse_cfg = PseConfig::NO_PSE;
    AttenMaskConfig atten_mask_cfg = AttenMaskConfig::NO_ATTEN_MASK;
    DropOutConfig drop_out_cfg = DropOutConfig::NO_DROP_OUT;
    bool isMM345NZOut = false;
};

class FlashAttentionScoreGradTilingS1s2Bn2 : public TilingBaseClass {
public:
    explicit FlashAttentionScoreGradTilingS1s2Bn2(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }

    FlashAttentionScoreGradTilingDataS1s2Bn2 td_;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus GetLayoutInfo();
    ge::graphStatus GetBaseShapeInfo();
    ge::graphStatus CheckOutOfTokens(const int64_t s1, const int64_t s2);
    ge::graphStatus CheckTokens();
    ge::graphStatus GetAttenMaskInfo();
    ge::graphStatus GetPseInfo();
    ge::graphStatus ProcessNormalPse();
    ge::graphStatus CheckAttenMaskShape();
    ge::graphStatus DoBlockTiling();
    ge::graphStatus DoCastTiling();
    void DoPreTiling();
    ge::graphStatus SetBmm1TilingData(uint32_t sOut, uint32_t sFla, uint32_t l1SizeRemain);
    ge::graphStatus SetBmm31TilingData(uint32_t l1SizeRemain);
    ge::graphStatus SetBmm4TilingData(uint32_t l1SizeRemain);
    ge::graphStatus SetBaseInfo(const gert::Shape &queryShape, const gert::Shape &keyShape, int64_t dimN1);
    ge::graphStatus ProcessDropInfo();
    ge::graphStatus SetMaskShapeType(const gert::Shape &storageShape, const uint32_t maskShapeDims);
    virtual void SetQKVStartIdx();
    void PrintShapeInfo();
    void DecideBranch();
    void NMDStrategy();
    void DecideBaseMND();
    bool ProcessPrefix();
    ge::graphStatus SetSparseParams();
    void SetBandIdx();
    bool SparseTokenProcess();
    bool IsModuleOneShape();
    void VectorBaseMNSplit();
    void MatmulBaseMNSplit();
    void SFTBaseMDSplit();
    bool CheckForDichotomy(std::vector<int64_t> &nums, uint32_t x, uint32_t m);
    int64_t GetSplitArrayMinMaxSum(std::vector<int64_t> &s1s2WeightNums, uint32_t coreNum);

    TempParamsS1s2Bn2 tmpData_;
    uint32_t tensorSize{0};
    int64_t dimD{0};
    int64_t dimS2{0};
    int64_t dimS1{0};
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t baseMmm{0};
    uint32_t baseNmm{0};
    uint32_t sftBaseM{0};
    uint32_t sftSingleM{0};
    uint32_t singleM{0};
    uint32_t singleN{0};
    uint32_t baseD{0};
    uint32_t s1Ratio{0};
    uint32_t s2Ratio{0};
    uint32_t mmRatio{0};
    bool needSplitD{false};
    bool needSetFixSplit{false};
    bool needAdjustBlockDim{false};
    bool tnd2bsh{false};
    uint32_t attenMaskCompressMode{0};
    int64_t attenMaskS1Size{0};
    int64_t attenMaskS2Size{0};
    SparseMode sparseMode = NO_MASK;
    uint64_t l2CacheSize{0};
};

class FlashAttentionScoreGradTilingDeterministic : public FlashAttentionScoreGradTilingS1s2Bn2 {
public:
    explicit FlashAttentionScoreGradTilingDeterministic(gert::TilingContext *context)
        : FlashAttentionScoreGradTilingS1s2Bn2(context)
    {
    }

protected:
    bool IsCapable() override;
};

} // namespace optiling
