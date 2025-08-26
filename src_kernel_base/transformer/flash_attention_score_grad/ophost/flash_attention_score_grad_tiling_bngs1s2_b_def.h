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
 * \file flash_attention_score_grad_tiling_bngs1s2_b_def.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradShapeAttrParamsForB)
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, g); // key & value对应的G
TILING_DATA_FIELD_DEF(int64_t, sQ);
TILING_DATA_FIELD_DEF(int64_t, sKV);
TILING_DATA_FIELD_DEF(int64_t, sKVAlign);     // 将sKV对齐到32bytes之后的元素个数
TILING_DATA_FIELD_DEF(int64_t, sKVAlignSize); // sKv轴的size对齐到32Byte之后的结果
TILING_DATA_FIELD_DEF(int64_t, sKVAlignByte);
TILING_DATA_FIELD_DEF(int64_t, hQ);  // n * g * d
TILING_DATA_FIELD_DEF(int64_t, hKV); // n * 1 * d
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, originalDAlign);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(int64_t, preTokens);
TILING_DATA_FIELD_DEF(int64_t, nextTokens);
TILING_DATA_FIELD_DEF(int64_t, headNum);      // query对应n, 也就是通常意义上的N2
TILING_DATA_FIELD_DEF(uint32_t, inputLayout); // layout格式，有BSH，SBH，BSND等
TILING_DATA_FIELD_DEF(int64_t, preTokensBlocks);
TILING_DATA_FIELD_DEF(int64_t, nextTokensBlocks);
TILING_DATA_FIELD_DEF(uint32_t, inputDType);    // query, key, value, dx, attention_in, pse输入的datatype
TILING_DATA_FIELD_DEF(int64_t, inputDTypeSize); // query, key, value, dx, attention_in, pse输入的datatype
TILING_DATA_FIELD_DEF(uint32_t, vecCalcDTypeSize); // 内部vector计算的数据大小，fp32和bf16作为输入的情况下，vector使用fp32计算
TILING_DATA_FIELD_DEF(uint32_t, pseSq);    // 等于sQ或者1
TILING_DATA_FIELD_DEF(uint32_t, existPse); // 0：not exist, 1:exist
TILING_DATA_FIELD_DEF(uint32_t, pseShapeType);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskShapeType); // 0: [b,1,sQ,sKV]或者1: [1,1,sQ,sKV]
TILING_DATA_FIELD_DEF(uint32_t, hasAttenMask);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskCompressMode);
TILING_DATA_FIELD_DEF(int64_t, attenMaskS2Size);
TILING_DATA_FIELD_DEF(uint32_t, precisionMode);
TILING_DATA_FIELD_DEF(uint32_t, syncLen);
TILING_DATA_FIELD_DEF(int64_t, mm1WorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, mm2WorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dqWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dkWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dropGmWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, mulGmWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dropoutWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, placeholder);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradShapeAttrParamsForBOp, FlashAttentionScoreGradShapeAttrParamsForB)

BEGIN_TILING_DATA_DEF(SplitBSplitCoreParams)
TILING_DATA_FIELD_DEF(int64_t, bOut); // 用于分核的外层b轴
TILING_DATA_FIELD_DEF(int64_t, apiClcQueueSize); // softmax、dropout、softmaxgrad这三个高阶api计算可以用到的最大的ubsize
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum); // 实际使用的vector核数
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitBSplitCoreParamsOp, SplitBSplitCoreParams)

BEGIN_TILING_DATA_DEF(SplitBSingleCoreParams)
TILING_DATA_FIELD_DEF(int64_t, bIn); // 用于给单核内凑数据搬运量的内层b轴
// 最后一个核需要处理的batch数，可能会小于singleCoreBatchRange
// 当bOut不能整除核数的时候，最后一个核处理的batch会变少。
TILING_DATA_FIELD_DEF(uint32_t, singleCoreBatchRange);
TILING_DATA_FIELD_DEF(uint32_t, bCvInner);
TILING_DATA_FIELD_DEF(uint32_t, bCvRatio);

// Vector计算用的tmp buf的size，在bf16下计算会取sKVAlign和dAlign的较大值
TILING_DATA_FIELD_DEF(int64_t, innerTmpBufSize);
TILING_DATA_FIELD_DEF(int64_t, clcDInner);
TILING_DATA_FIELD_DEF(int64_t, dSize);
TILING_DATA_FIELD_DEF(int64_t, dInnerTail);
TILING_DATA_FIELD_DEF(int64_t, dInnerTailAlign);

TILING_DATA_FIELD_DEF(int64_t, vecQueIn1Size);
TILING_DATA_FIELD_DEF(int64_t, subRange); // 用sQ / 8 向上取整，最后可能有尾块
TILING_DATA_FIELD_DEF(int64_t, subMask);
TILING_DATA_FIELD_DEF(int64_t, subMaskTail);
TILING_DATA_FIELD_DEF(int64_t, sKVAlignBlockNum);
TILING_DATA_FIELD_DEF(int64_t, rightPadding);
TILING_DATA_FIELD_DEF(int64_t, dstStride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitBSingleCoreParamsOp, SplitBSingleCoreParams)

BEGIN_TILING_DATA_DEF(MulsParamsForB)
TILING_DATA_FIELD_DEF(uint32_t, inputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, outputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, singleUBProcessNum);
TILING_DATA_FIELD_DEF(uint32_t, dqSingleCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, dkvSingleCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, dqTailCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, kvTailCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, dqSingleCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, kvSingleCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, dqTailCoreLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, kvTailCoreLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, dqLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, kvLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MulsParamsForBOp, MulsParamsForB)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradUbngs1s2BbTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradShapeAttrParamsForB, opInfo);
TILING_DATA_FIELD_DEF_STRUCT(SplitBSplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SplitBSingleCoreParams, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(PreParams, preTilingData);
TILING_DATA_FIELD_DEF_STRUCT(PostParams, postTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1AndMm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm31TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm32AndMm4TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000123099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000113099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000103099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000122099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000112099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000102099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000133099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000132099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001112099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001113099, FlashAttentionScoreGradUbngs1s2BbTilingData)
//mm12 nz
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010123099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010113099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010103099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010122099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010112099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010102099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010133099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010132099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011112099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011113099, FlashAttentionScoreGradUbngs1s2BbTilingData)
//mm345 nz
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100123099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100113099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100103099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100122099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100112099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100102099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100133099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100132099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101112099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101113099, FlashAttentionScoreGradUbngs1s2BbTilingData)
//mm all nz
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110123099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110113099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110103099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110122099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110112099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110102099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110133099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110132099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111112099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111113099, FlashAttentionScoreGradUbngs1s2BbTilingData)
//fp32
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000101099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000111099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001111099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000121099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000131099, FlashAttentionScoreGradUbngs1s2BbTilingData)

} // namespace optiling
