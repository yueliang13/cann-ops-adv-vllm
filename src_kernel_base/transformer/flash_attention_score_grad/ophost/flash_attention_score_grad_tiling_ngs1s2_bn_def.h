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
 * \file flash_attention_score_grad_tiling_ngs1s2_bn_def.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradShapeAttrParams1)
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, n); // n是key对应的n, query对应的n_query是 n * g
TILING_DATA_FIELD_DEF(int64_t, g); // key & value对应的G
TILING_DATA_FIELD_DEF(int64_t, sQ);
TILING_DATA_FIELD_DEF(int64_t, sKV);
TILING_DATA_FIELD_DEF(int64_t, sKVAlign);     // 将sKV对齐到32bytes之后的元素个数
TILING_DATA_FIELD_DEF(int64_t, sKVAlignSize); // sKv轴的size对齐到32Byte之后的结果
TILING_DATA_FIELD_DEF(int64_t, sKVAlignVec);  // sKv轴的size对齐到32Byte之后的元素个数
TILING_DATA_FIELD_DEF(int64_t, sKVAlignSizeVec); // sKv轴的size对齐到32Byte之后的结果,使用vector计算的datatype
TILING_DATA_FIELD_DEF(int64_t, sKVAlignByte); // sKv轴的size对齐到32Byte之后的结果,使用byte作为data type
TILING_DATA_FIELD_DEF(int64_t, hQ);           // n * g * d
TILING_DATA_FIELD_DEF(int64_t, hKV);          // n * 1 * d
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
TILING_DATA_FIELD_DEF(uint32_t, inputDType);     // query, key, value, dx, attention_in, pse输入的datatype
TILING_DATA_FIELD_DEF(uint32_t, inputDTypeSize); // query, key, value, dx, attention_in, pse输入的datatype
// 内部vector计算的数据大小，fp32和bf16作为输入的情况下，vector使用fp32计算
TILING_DATA_FIELD_DEF(uint32_t, vecCalcDTypeSize);
TILING_DATA_FIELD_DEF(uint32_t, pseSq); // 等于sQ或者1 等于0表示没有pse
TILING_DATA_FIELD_DEF(uint32_t, pseShapeType);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskShapeType); // 0: [b,1,sQ,sKV]或者1: [1,1,sQ,sKV]
TILING_DATA_FIELD_DEF(uint32_t, hasAttenMask);       // 是否有attenMask可选输入
TILING_DATA_FIELD_DEF(uint32_t, attenMaskCompressMode);
TILING_DATA_FIELD_DEF(int64_t, attenMaskS2Size);
// 针对qkv、attention in、dx输入每个block（32Bytes）对应的元素个数
TILING_DATA_FIELD_DEF(uint32_t, elementPerBlock);
TILING_DATA_FIELD_DEF(uint32_t, precisionMode);
TILING_DATA_FIELD_DEF(uint32_t, resv);
TILING_DATA_FIELD_DEF(int64_t, mm1WorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, mm2WorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dqWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dkWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dropGmWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, mulGmWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dropoutWorkspaceLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradShapeAttrParams1Op, FlashAttentionScoreGradShapeAttrParams1)

BEGIN_TILING_DATA_DEF(SplitNSplitCoreParams)
TILING_DATA_FIELD_DEF(int64_t, totalBatch); // 总的batch数，等于b * nOut
TILING_DATA_FIELD_DEF(int64_t, nOut);       // 用于分核的外层n轴
// softmax、dropout、softmaxgrad这三个高阶api计算可以用到的最大的ubsize
TILING_DATA_FIELD_DEF(int64_t, apiClcQueueSize);
TILING_DATA_FIELD_DEF(int64_t, mm1ResSize);   // matmul结果的size = sQ * Skv * inputDTypeSize
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum); // 实际使用的vector核数
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitNSplitCoreParamsOp, SplitNSplitCoreParams)

BEGIN_TILING_DATA_DEF(SplitNSingleCoreParams)
TILING_DATA_FIELD_DEF(int64_t, nIn);     // 用于给单核内凑数据搬运量的内层n轴
TILING_DATA_FIELD_DEF(int64_t, nInTail); // n对nIn做了拆分之后，不能整除时剩余的尾巴n_inner的大小
TILING_DATA_FIELD_DEF(uint32_t, singleCoreBatchRange); // 单个核需要处理的batch数，这里的batch是指B * Nout
// 最后一个核需要处理的batch数，可能会小于singleCoreBatchRange
// 当totalBatch不能整除核数的时候，最后一个核处理的batch会变少。
TILING_DATA_FIELD_DEF(uint32_t, singleCoreBatchRangeTail);
/* Vector计算用的tmp buf的size，在bf16下计算会取sKVAlign和dAlign的较大值*/
TILING_DATA_FIELD_DEF(int64_t, nCvInner);
TILING_DATA_FIELD_DEF(int64_t, innerTmpBufSize);
TILING_DATA_FIELD_DEF(int64_t, vecCastSize);
TILING_DATA_FIELD_DEF(int64_t, splitedDAlign); // d轴的切分，注意对齐到block
TILING_DATA_FIELD_DEF(int64_t, dRange);
TILING_DATA_FIELD_DEF(int64_t, vecQueIn1Size);

/* Sub计算用到的参数 */
TILING_DATA_FIELD_DEF(int64_t, subRange); // 用sQ / 8 向上取整，最后可能有尾块
// 表示单条vector指令算多少个数，256 / sizeof(vector计算的数据类型size)
TILING_DATA_FIELD_DEF(int64_t, subMask);
// 表示尾块单条指令算多少个数,(sQ % 8) * (32 / sizeof(vector计算的数据类型size))
TILING_DATA_FIELD_DEF(int64_t, subMaskTail);
// skVAlign占据多少个block = sKv * sizeof(vector计算的数据类型size) / 32
TILING_DATA_FIELD_DEF(int64_t, sKVAlignBlockNumVec);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitNSingleCoreParamsOp, SplitNSingleCoreParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradTilingDataUngs1s2Bbn)
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradShapeAttrParams1, opInfo);
TILING_DATA_FIELD_DEF_STRUCT(SplitNSplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SplitNSingleCoreParams, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(PreParams, preTilingData);
TILING_DATA_FIELD_DEF_STRUCT(PostParams, postTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1AndMm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm31TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm32AndMm4TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000003199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000013199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000023199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000002199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000012199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000022199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001013199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001012199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
// mm12 nzout
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010003199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010013199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010023199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010002199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010012199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010022199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011013199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011012199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
// mm345 nzout
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100003199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100013199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100023199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100002199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100012199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100022199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101013199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101012199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
// mm all nzout
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110003199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110013199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110023199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110002199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110012199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110022199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111012199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111013199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
//fp32
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000001199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000011199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000021199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001011199, FlashAttentionScoreGradTilingDataUngs1s2Bbn)

} // namespace optiling

