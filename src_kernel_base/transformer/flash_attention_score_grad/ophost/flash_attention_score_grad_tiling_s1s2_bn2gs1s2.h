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
 * \file flash_attention_score_grad_tiling_s1s2_bn2gs1s2.h
 * \brief
 */

#pragma once

#include "flash_attention_score_grad_tiling_common.h"
#include "tiling/tiling_base.h"
#include "flash_attention_score_grad_tiling_s1s2_bn2gs1s2_def.h"

namespace optiling {

constexpr uint32_t CORE_LIST_NUM = 50;
constexpr uint32_t ARRAY_LENGTH = 3;
struct FuzzyBaseInfoParams { // 频繁使用的基础参数
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
    uint32_t s1Tail;
    uint32_t s1CvTail;
    uint32_t s2Align;
    int64_t s2Outer;
    uint32_t s1CvRatio = 1;
    uint32_t s2CvRatio = 1;
    uint32_t mm1CalRatio = 1;
    uint32_t cvS2Inner;
    uint32_t s2Inner;
    uint32_t s2Tail;
    uint32_t s2CvTail;
    uint32_t sfmgdOuter;
    uint32_t sfmgdInner;
    uint32_t sfmgdTail;
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
    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    uint32_t sparseMode;
    std::vector<int64_t> prefixN;

    std::vector<int64_t> actualSeqQlen;
    std::vector<int64_t> actualSeqKvlen;

    bool isSparse;
    bool isBf16;
    bool mm1IsNZOut;
    bool mm2IsNZOut;

    uint32_t tmpBufferSize = 0;

    TilingDataType mode;
};

class FlashAttentionScoreGradTilingS1s2Bn2gs1s2 : public TilingBaseClass {
public:
    explicit FlashAttentionScoreGradTilingS1s2Bn2gs1s2(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }
    FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 tilingData;

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
    ge::graphStatus DoSparse();
    bool CheckFuzzyArgsLegal(uint32_t s1Inner, uint32_t s2Inner);
    std::tuple<uint32_t, uint32_t, uint32_t> FuzzyForBestSplit();
    void SetMatmulTilingBufferInfo(TCubeTiling &mmTiling);
    ge::graphStatus GetSparseBlockInfo();
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
    ge::graphStatus GetSparsePrefixBlockInfo();
    ge::graphStatus GetSparseUnpadBlockInfo();
    int64_t FindBandIdx();
    void FillBlockInfo(std::vector<std::vector<std::vector<int64_t>>> &calculatedBlockInfo,
                       std::vector<std::vector<int64_t>> &totalBlockInfo);

    bool CheckPrefixNExist(const int64_t bIdx, const int64_t prefixN,
                           std::vector<std::vector<std::pair<int64_t, int64_t>>> &s1ValidIdx);
    void GetCommS1S2OuterInfo(const int64_t s1, const int64_t s2, int64_t (*parseInfo)[ARRAY_LENGTH]);
    void GetParseS1S2OuterInfo(int64_t (*parseInfo)[ARRAY_LENGTH]);
    bool SetSparseParams();
    void GetCommS1S2OuterInfo(const int64_t prefixN, std::vector<std::pair<int64_t, int64_t>> &s1ValidIdx);
    void PrintShapeInfo();
    ge::graphStatus GetBaseShapeInfo();
    ge::graphStatus ProcessTndToBsh();
    bool tnd2bsh = false;

private:
    FuzzyBaseInfoParams fBaseParams;
};

} // namespace optiling
