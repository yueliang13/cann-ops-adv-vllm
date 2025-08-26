/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.h
 * \brief
 */

#pragma once

#include "flash_attention_score_grad_tiling_common.h"
#include "tiling/tiling_base.h"
#include "flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab_def.h"

namespace optiling {

struct SameAbFuzzyBaseInfoParams { // 频繁使用的基础参数
    int64_t coreNum;
    int64_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0cSize;

    int64_t b;
    int64_t n1;
    int64_t n2;
    int64_t s1;
    int64_t s2;
    int64_t g;
    int64_t d;
    uint32_t s1Align;
    int64_t s1Outer;
    uint32_t s1Inner;
    uint32_t s1CvInner;
    uint32_t s2CvInner;
    uint32_t s1Tail;
    uint32_t s1CvTail;
    uint32_t s2Align;
    int64_t s2Outer;
    uint32_t s2Inner;
    uint32_t s2Tail;
    uint32_t s2CvTail;
    int64_t sfmgNormalAxisSize = 0;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t sumS1S2Product = 0;

    int64_t pseType = PSE_OUTER_ADD_MUL_TYPE;
    int64_t pseAlibiBaseS1 = 0;
    int64_t pseAlibiBaseS2 = 0;
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    uint32_t pseOptional;
    uint32_t pseShapeType = 0;
    uint32_t pseDtype = 0;

    uint32_t queryType;
    uint32_t attenMaskOptional;
    uint32_t attenMaskShapeType = 0;
    uint32_t attenMaskDtype = 0;
    uint32_t attenMaskCompressMode;
    int64_t attenMaskS1Size = 0;
    int64_t attenMaskS2Size = 0;
    uint32_t dropoutIsDivisibleBy8 = 0;
    uint32_t layoutType;
    float scaleValue;
    float keepProb;
    int64_t bandIdx;

    uint32_t dataTypeSize;
    uint32_t dataBlockNum;
    uint32_t calTypeSize;
    uint32_t calBlockNum;
    uint32_t blockNum;
    int64_t s1Token;
    int64_t s2Token;
    uint32_t blockOuter;
    int64_t blockFactor;

    int64_t qSize;
    int64_t kvSize;
    int64_t qSizeAlign;
    int64_t kvSizeAlign;
    int64_t dropMaskSize;

    uint32_t baseMN;
    uint32_t sparseMode;
    std::vector<int64_t> prefixN;

    std::vector<int64_t> actualSeqQlen;
    std::vector<int64_t> actualSeqKvlen;

    bool isSparse;
    bool isBf16;
    bool mm1IsNZOut;
    bool mm2IsNZOut;
    bool isDeterministic;

    uint32_t tmpBufferSize = 0;

    TilingDataType mode;
};

class FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb : public TilingBaseClass {
public:
    explicit FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }
    FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb tilingData;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus DoSplit();
    void SetMatmulTilingBufferInfo(TCubeTiling &mmTiling);
    ge::graphStatus DoPreTiling();
    ge::graphStatus DoPostTiling();
    void DetermineMode();
    virtual void SetQKVStartIdx();
    ge::graphStatus CheckAttenMaskShape();
    ge::graphStatus ProcessPseInfo(const char *inputLayout);
    ge::graphStatus ProcessDropInfo(const char *inputLayout);
    ge::graphStatus ProcessPseNormal(const char *inputLayout);
    ge::graphStatus ProcessSparseModeInfo();
    ge::graphStatus ProcessTokensInfo();
    ge::graphStatus SaveToTilingData();
    int64_t FindBandIdx();
    bool SetSparseParams();
    void PrintShapeInfo();
    ge::graphStatus GetBaseShapeInfo();
    void DoPreSfmgTiling();
    void AdjustCvInner();

private:
    SameAbFuzzyBaseInfoParams fBaseParams;
};

class FlashAttentionScoreGradTilingSameABDeterministic : public FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb {
public:
    explicit FlashAttentionScoreGradTilingSameABDeterministic(gert::TilingContext *context)
        : FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb(context)
    {
    }

protected:
    bool IsCapable() override;
};

} // namespace optiling
