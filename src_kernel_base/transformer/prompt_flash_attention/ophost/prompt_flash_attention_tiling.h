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
 * \file prompt_flash_attention_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_PROMPTFLASHATTENTION_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_PROMPTFLASHATTENTION_H_
#include <cstdint>
#include <vector>
#include <queue>
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/data_copy_transpose_tiling_def.h"
#include "tiling/data_copy_transpose_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#ifdef ASCENDC_OP_TEST
#define PFA_EXTERN_C extern "C"
#else
#define PFA_EXTERN_C
#endif

#include "prompt_flash_attention_tiling_compile_info.h"
#include "prompt_flash_attention_tiling_const.h"
#include "prompt_flash_attention_tiling_context.h"
#include "prompt_flash_attention_tiling_struct.h"

namespace optiling { 

BEGIN_TILING_DATA_DEF(PromptAttentionBaseParams)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, headNumSize);
  TILING_DATA_FIELD_DEF(uint32_t, seqSize);
  TILING_DATA_FIELD_DEF(uint32_t, headSize);
  TILING_DATA_FIELD_DEF(float, scaleValue);
  TILING_DATA_FIELD_DEF(int32_t, preTokens);
  TILING_DATA_FIELD_DEF(int32_t, nextTokens);
  TILING_DATA_FIELD_DEF(int32_t, blockSize);
  TILING_DATA_FIELD_DEF(int32_t, blockTableDim2);
  TILING_DATA_FIELD_DEF(int32_t, PABlockNumSum);
  TILING_DATA_FIELD_DEF(uint32_t, dimNumOfseq);
  TILING_DATA_FIELD_DEF(uint32_t, typeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, seqInnerSize);
  TILING_DATA_FIELD_DEF(uint32_t, prefixSeqInnerSize);
  TILING_DATA_FIELD_DEF(uint32_t, usePseShift);
  TILING_DATA_FIELD_DEF(uint32_t, useMask);
  TILING_DATA_FIELD_DEF(uint32_t, headNumRatio);
  TILING_DATA_FIELD_DEF(uint32_t, attenMaskElemType);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftTypeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, pseMaskMaxSize);
  TILING_DATA_FIELD_DEF(uint32_t, maskTypeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, outputTypeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxTypeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
  TILING_DATA_FIELD_DEF(uint32_t, alignedHeadSize);
  TILING_DATA_FIELD_DEF(uint32_t, splitS2);
  TILING_DATA_FIELD_DEF(uint32_t, splitD);
  TILING_DATA_FIELD_DEF(uint32_t, layoutType);
  TILING_DATA_FIELD_DEF(uint32_t, PAlayoutType);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftS1Size);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftS2Size);
  TILING_DATA_FIELD_DEF(uint32_t, maskKVsSize);
  TILING_DATA_FIELD_DEF(uint32_t, maskQsSize);
  TILING_DATA_FIELD_DEF(uint32_t, isLayoutSH);
  TILING_DATA_FIELD_DEF(uint32_t, isActualSeqLengthsNull);
  TILING_DATA_FIELD_DEF(uint32_t, isActualSeqLengthsKVNull);
  TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsSize);
  TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsKVSize);
  TILING_DATA_FIELD_DEF(uint32_t, deqScaleFlag);
  TILING_DATA_FIELD_DEF(uint32_t, deqScale2Flag);
  TILING_DATA_FIELD_DEF(uint32_t, isAntiPerchannel);
  TILING_DATA_FIELD_DEF(uint32_t, isRowInvalid);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxOuterSize);
  TILING_DATA_FIELD_DEF(uint32_t, isQuant2Perchannel);
  TILING_DATA_FIELD_DEF(uint32_t, isQuant2BF16);
  TILING_DATA_FIELD_DEF(uint32_t, isKvContinuous);
  TILING_DATA_FIELD_DEF(uint32_t, fromFused);
  TILING_DATA_FIELD_DEF(uint32_t, isBSNDOut);
  TILING_DATA_FIELD_DEF(uint32_t, isSoftMaxLseEnable);
  TILING_DATA_FIELD_DEF(uint32_t, isActualSharedPrefixLenNull);
  TILING_DATA_FIELD_DEF(uint32_t, isQHasLeftPadding);
  TILING_DATA_FIELD_DEF(uint32_t, isKVHasLeftPadding);
  TILING_DATA_FIELD_DEF(int64_t, keyAntiquantMode);
  TILING_DATA_FIELD_DEF(int64_t, valueAntiquantMode);
  TILING_DATA_FIELD_DEF(uint32_t, hasKeyAntiquantOffset);
  TILING_DATA_FIELD_DEF(uint32_t, isMsd);
  TILING_DATA_FIELD_DEF(uint32_t, isQuant2FP16);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionBaseParamsOp, PromptAttentionBaseParams)

BEGIN_TILING_DATA_DEF(PromptAttentionSeqParams)
  // Temporary reuse
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, CoreHeadNumTail);       // coreNStart
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, actualS1);              // coreNEnd
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, actualCoreNums);        // coreSidStart
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, singleCoreHeadNumSize); // coreSidEnd
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, coreSeqPosStart);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, coreSeqPosEnd);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSeqParamsOp, PromptAttentionSeqParams)

BEGIN_TILING_DATA_DEF(PromptAttentionSingleCoreParams)
  TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSize);
  TILING_DATA_FIELD_DEF(uint32_t, singleProcessSOuterSize);
  TILING_DATA_FIELD_DEF(uint32_t, multiSmaxsInnerLoopTimes);
  TILING_DATA_FIELD_DEF(uint32_t, actualCoreNums);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftBatch);
  TILING_DATA_FIELD_DEF(uint32_t, attenMaskBatch);
  TILING_DATA_FIELD_DEF(uint32_t, kvAntiquantSInnerSize);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSingleCoreParamsOp, PromptAttentionSingleCoreParams)

BEGIN_TILING_DATA_DEF(PromptAttentionSingleCoreTensorSize)
  TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, attenMaskUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, maskSize);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxMaxSize);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxSumSize);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxExpSize);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxValueSize);
  TILING_DATA_FIELD_DEF(uint32_t, spmTmpSize);
  TILING_DATA_FIELD_DEF(uint32_t, scmTmpSize);
  TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, tmpMMResBmm2PreUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, tmpSoftmaxBmm2UbSize);
  TILING_DATA_FIELD_DEF(uint32_t, selectSpaceUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, tmpSoftMaxV2Size);
  TILING_DATA_FIELD_DEF(uint32_t, mm1TmpUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, mm2TmpUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, kvAntiquantUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbMsdSize);
  TILING_DATA_FIELD_DEF(uint32_t, tempBmm2QueueMsdSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdInQueueSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdQRowSumBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdAMaxTmpBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdAMaxResBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdSoftmaxResAmaxBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdSoftmaxRowSumScaleBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdScaleBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdOffsetBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdTmpMm1BuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdTmpMm2BuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdOutQueueSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdComputeLines);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSingleCoreTensorSizeOp, PromptAttentionSingleCoreTensorSize)

BEGIN_TILING_DATA_DEF(PromptAttentionInitOutputParams)
  TILING_DATA_FIELD_DEF(uint32_t, singleCoreSize);
  TILING_DATA_FIELD_DEF(int64_t, totalOutputSize);
  TILING_DATA_FIELD_DEF(int64_t, totalSoftMaxLseOutputSize);
  TILING_DATA_FIELD_DEF(uint32_t, needInit);
  TILING_DATA_FIELD_DEF(uint32_t, isOneN);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionInitOutputParamsOp, PromptAttentionInitOutputParams)

BEGIN_TILING_DATA_DEF(PromptFlashAttentionTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingDataRect);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingDataRect);

  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionBaseParams, promptAttentionBaseParams);
  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionSeqParams, promptAttentionSeqParams);
  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionSingleCoreParams, promptAttentionSingleCoreParams);
  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionSingleCoreTensorSize, promptAttentionTensorSizeRect);
  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionInitOutputParams, promptAttentionInitOutputParams);

  TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingDataRect);
  TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingDataRect);
  TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingDataRect);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PromptFlashAttention, PromptFlashAttentionTilingData)

BEGIN_TILING_DATA_DEF(PFAInputParams)
    TILING_DATA_FIELD_DEF(int64_t, bSize);
    TILING_DATA_FIELD_DEF(int64_t, n2Size);
    TILING_DATA_FIELD_DEF(int64_t, gSize);
    TILING_DATA_FIELD_DEF(int64_t, s1Size);
    TILING_DATA_FIELD_DEF(int64_t, s2Size);
    TILING_DATA_FIELD_DEF(int64_t, alignedS2);
    TILING_DATA_FIELD_DEF(int64_t, dSize);
    TILING_DATA_FIELD_DEF(int64_t, valueDSize);
    TILING_DATA_FIELD_DEF(float, keepProb);
    TILING_DATA_FIELD_DEF(float, scaleValue);
    TILING_DATA_FIELD_DEF(int64_t, preTokens);
    TILING_DATA_FIELD_DEF(int64_t, nextTokens);
    // in pse encoding scenes, s1 and s2 might not equal with s1, s2 in Q, K
    TILING_DATA_FIELD_DEF(int64_t, pseS1Size);
    TILING_DATA_FIELD_DEF(int64_t, pseS2Size);
    TILING_DATA_FIELD_DEF(uint32_t, pseBSize);
    TILING_DATA_FIELD_DEF(uint32_t, bandIndex);

    // 1: BSH/BSND, 2: SBH, 3: BNSD
    TILING_DATA_FIELD_DEF(uint8_t, layoutType);
    // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    TILING_DATA_FIELD_DEF(uint8_t, pseShapeType);
    // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
    TILING_DATA_FIELD_DEF(uint8_t, attenMaskShapeType);
    // 0: fp16, 1: bool(uint8)
    TILING_DATA_FIELD_DEF(uint8_t, attenMaskDataType);
    // ALL: 0, NONE: 1, ANY: 2, CAUSAL: 3, BAND: 4 };
    TILING_DATA_FIELD_DEF(uint8_t, attenMaskCompressMode);
    // 0: high precise, 1: high performance, 2: invalid line high precise
    TILING_DATA_FIELD_DEF(uint8_t, implMode);
    TILING_DATA_FIELD_DEF(uint8_t, sparseType);
    TILING_DATA_FIELD_DEF(uint8_t, fromFused);
    TILING_DATA_FIELD_DEF(uint8_t, pseEncodeType);
    TILING_DATA_FIELD_DEF(uint8_t, isSoftMaxLseEnable);
    TILING_DATA_FIELD_DEF(uint16_t, remain);
    TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size);
    TILING_DATA_FIELD_DEF(uint32_t, pseType);
    TILING_DATA_FIELD_DEF(uint32_t, rsv1);
    TILING_DATA_FIELD_DEF(int64_t, qStartIdx);
    TILING_DATA_FIELD_DEF(int64_t, kvStartIdx);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PFAInputParamsOp, PFAInputParams)

BEGIN_TILING_DATA_DEF(PFAMultiCoreParams)
    TILING_DATA_FIELD_DEF(int32_t, coreNum);
    TILING_DATA_FIELD_DEF(int32_t, reserve);
    // BN2GS1.o
    TILING_DATA_FIELD_DEF(int64_t, totalSize);
    // BN2GS1.o / core_num
    TILING_DATA_FIELD_DEF(int64_t, splitFactorSize);
    TILING_DATA_FIELD_DEF(int64_t, splitFactorTailSize);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 48, sparseStartIdx);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PFAMultiCoreParamsOp, PFAMultiCoreParams)

BEGIN_TILING_DATA_DEF(PFACoreParams)
    TILING_DATA_FIELD_DEF(int32_t, s1BaseSize);
    TILING_DATA_FIELD_DEF(int32_t, s1BaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, s1OuterSize);
    TILING_DATA_FIELD_DEF(int32_t, s1Vec2BaseSize);
    TILING_DATA_FIELD_DEF(int32_t, s1Vec2BaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, s1Vec2OuterSize);
    TILING_DATA_FIELD_DEF(int32_t, s2BaseSize);
    TILING_DATA_FIELD_DEF(int32_t, s2BaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, s2OuterSize);
    TILING_DATA_FIELD_DEF(int32_t, dBaseSize);
    TILING_DATA_FIELD_DEF(int32_t, dBaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, dOuterSize);
    TILING_DATA_FIELD_DEF(int32_t, bBaseSize);
    TILING_DATA_FIELD_DEF(int32_t, bBaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, bOuterSize);
    TILING_DATA_FIELD_DEF(int32_t, n2BaseSize);
    TILING_DATA_FIELD_DEF(int32_t, n2BaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, n2OuterSize);
    TILING_DATA_FIELD_DEF(int32_t, gBaseSize);
    TILING_DATA_FIELD_DEF(int32_t, gBaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, gOuterSize);
    TILING_DATA_FIELD_DEF(int32_t, nRatio);
    TILING_DATA_FIELD_DEF(int32_t, rsvd);
    TILING_DATA_FIELD_DEF(int64_t, s1SparseValidSize);
    TILING_DATA_FIELD_DEF(int64_t, s2SparseValidSize);
    TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS1);
    TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS2);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PFACoreParamsOp, PFACoreParams)

BEGIN_TILING_DATA_DEF(PFATensorSizeParams)
    TILING_DATA_FIELD_DEF(int32_t, bmm1ResUbSize);
    TILING_DATA_FIELD_DEF(int32_t, attenMaskUbSize);
    TILING_DATA_FIELD_DEF(int32_t, pseUbSize);
    TILING_DATA_FIELD_DEF(int32_t, dropMaskUbSize);
    TILING_DATA_FIELD_DEF(int32_t, castUbSize);
    TILING_DATA_FIELD_DEF(int32_t, softmaxMaxUbSize);
    TILING_DATA_FIELD_DEF(int32_t, softmaxSumUbSize);
    TILING_DATA_FIELD_DEF(int32_t, softmaxExpUbSize);
    TILING_DATA_FIELD_DEF(int32_t, apiTmpBufferBytes);
    TILING_DATA_FIELD_DEF(int32_t, bmm2ResUbSize);
    TILING_DATA_FIELD_DEF(int32_t, inputQueBytes);
    TILING_DATA_FIELD_DEF(int32_t, outputQueBytes);
    // API buffer use remain space of ub
    TILING_DATA_FIELD_DEF(int32_t, tmpBufBytes);
    TILING_DATA_FIELD_DEF(int32_t, softmaxMaxOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, softmaxSumOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, maxSumApiOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, customSoftmaxApiOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, pseTbufOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, dropoutApiOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, maxSumApiSize);
    TILING_DATA_FIELD_DEF(int32_t, customSoftmaxApiSize);
    TILING_DATA_FIELD_DEF(int32_t, dropoutApiSize);
    TILING_DATA_FIELD_DEF(int32_t, attenMaskApiSize);
    TILING_DATA_FIELD_DEF(int32_t, attenMaskApiOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, bmm1ProcessTInStage2Size);
    TILING_DATA_FIELD_DEF(int32_t, bmm1ProcessTInStage2OffsetBytes);
    // workspace
    TILING_DATA_FIELD_DEF(int32_t, wkspSection1OffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, wkspSection2OffsetBytes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PFATensorSizeParamsOp, PFATensorSizeParams)

BEGIN_TILING_DATA_DEF(MLAGeneralTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(PFAInputParams, PFAinputParams);
    TILING_DATA_FIELD_DEF_STRUCT(PFAMultiCoreParams, PFAmultiCoreParams);
    TILING_DATA_FIELD_DEF_STRUCT(PFACoreParams, PFAcoreParams);
    TILING_DATA_FIELD_DEF_STRUCT(PFATensorSizeParams, PFAtensorSizeParams);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
    TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingData);
    TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingData);
    TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingDataTailCore);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000000000000, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000000000001, MLAGeneralTilingData)

class BufferNum {
public:
    // sum and max always use fp32, shape is (S1, 1), inner axis align 32B.
    size_t bufferS1S2Num; // unit: input dtype
    size_t bufferS1DNum;
    size_t bufferExpNum; // unit: input dtype, shape: [S1, 1], inner axis align 32B.
};

class PromptFlashAttentionTiling {
public:
    PromptFlashAttentionTiling(fe::PlatFormInfos* platFormInfo): ascendcPlatform(platFormInfo) {}
    ge::graphStatus RunBigKernelTilingWithParams(ContextParamsForPFATiling& contextKeyParams,
                                                uint64_t& tilingKey, uint32_t& blockDimToBeSet,
                                                PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus PromptFlashAttentionSetTilingData(gert::TilingContext* context,
                                                    PromptFlashAttentionTilingData& tilingData);
    bool CheckNonEmptyShapeExceptions(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* shape,
                                      const std::string &sName);
    bool fromPFA_ = true;
protected:
    ge::graphStatus ConvertContextToPFAParams(gert::TilingContext* context, ContextParamsForPFATiling& contextKeyParams);
    ge::graphStatus TilingGetTilingKeyAttentionAscendC(uint64_t& tilingKey, ContextParamsForPFATiling& contextKeyParams,
                                                       bool useNewTiling, PromptFlashAttentionTilingData& tilingData);
    void PromptFlashAttentionSplitNS(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData, uint32_t curCoreNum, std::vector<int64_t>& actualSeqLengths);
    void PromptFlashAttentionSplitNSNew(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData, uint32_t curCoreNum, std::vector<int64_t>& actualSeqLengths,
                                                    std::vector<int64_t>& actualSeqLengthsKV, int64_t actualSharedPrefixLen, bool useBalanceTiling);
    void GetPreNextTokensLeftUp(PromptFlashAttentionTilingData& tilingData, uint32_t actualSeqLength, uint32_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp);
    void SetSplitCoreMode(PromptFlashAttentionTilingData& tilingData, uint32_t sOuterFactor);
    void PromptFlashAttentionSplitSeqOneN(PromptFlashAttentionTilingData& tilingData, uint32_t curCoreNum, bool isVectorCore);
    bool EnableMTE2BmmPipe(PromptFlashAttentionTilingData& tilingData, matmul_tiling::MatmulApiTiling& bmm,
                           TCubeTiling& bmmTilingData, uint32_t sOuterFactor, uint32_t sInnerFactor);
    void EnableBmmDoubleBuffer(TCubeTiling& bmmTilingData);
    void PromptFlashAttention310PSetBmm1(matmul_tiling::MatmulApiTiling& bmm1);
    void PromptFlashAttention310PSetBmm2(matmul_tiling::MatmulApiTiling& bmm2);
    bool PromptFlashAttentionCheckBmm1(PromptFlashAttentionTilingData& tilingData, TCubeTiling& bmm1TilingData,
                                       int64_t l1SizeRemain, int64_t l0CSize,
                                       uint32_t sOuterFactor, uint32_t sInnerFactor,
                                       bool allGM = false, bool autoBaseMNK = false);
    bool PromptFlashAttentionCheckBmm2(PromptFlashAttentionTilingData& tilingData, TCubeTiling& bmm1TilingData,
                                       int64_t l1SizeRemain, int64_t l0CSize,
                                       uint32_t sOuterFactor, uint32_t sInnerFactor,
                                       uint32_t dSplitFactor, bool allGM = false, bool autoBaseMNK = false);
    void PromptFlashAttentionSetTensorSize(PromptFlashAttentionTilingData& tilingData,
                        PromptAttentionSingleCoreTensorSize& tensorSize, uint32_t sOuterFactor, uint32_t sInnerFactor);
    bool PromptFlashAttentionCheckArgsLegal(PromptFlashAttentionTilingData& tilingData, int64_t ubSize, int64_t l1Size,
                                            int64_t l0CSize, uint32_t typeByteSize, uint32_t& sOuterFactor,
                                            uint32_t sInnerFactor, bool& updateDiv, uint32_t maskTypeSize, uint32_t dSplitFactor);
    ge::graphStatus AdjustBasicBlock(PromptFlashAttentionTilingData& tilingData, uint32_t& sOuterFactor);
    ge::graphStatus PromptFlashAttentionApiTiling(PromptFlashAttentionTilingData& tilingData, uint32_t typeSize,
                                                  uint32_t sOuterFactor, uint32_t softmaxSInnerFactor, uint32_t softmaxSOuterFactor);
    ge::graphStatus GetRectangleFactor(uint32_t seqSplit, std::queue<uint32_t>& sQueue, int32_t threshold = 16);
    ge::graphStatus SetInputLayout(const char* layout);
    bool GetApiTmpSize(const uint32_t sOuterFactor, const uint32_t sInnerFactor,
                        const uint32_t typeByteSize);
    uint32_t CalculateL1SizeUsed(PromptFlashAttentionTilingData& tilingData, const uint32_t typeByteSize);
    bool CheckInputDimAndHeadNum(ContextParamsForPFATiling& contextKeyParams, uint32_t nQAttr, uint32_t nKVAttr);
    bool SetTilingHeadNumRatio(ContextParamsForPFATiling& contextKeyParams, const int32_t* numQueryHeads,
                               const int32_t* numKeyValueHeads, PromptFlashAttentionTilingData& tilingData);
    void PromptFlashAttentionInitOutputSplit(uint64_t totalSize, PromptFlashAttentionTilingData &tilingData,
                                             uint32_t curCoreNum);
    void PromptFlashAttentionInitSoftmaxLseOutputSplit(uint64_t totalSize, PromptFlashAttentionTilingData &tilingData);
    void Align(uint32_t &num);
    ge::graphStatus GetBasicShape(uint32_t &b, uint32_t &s, uint32_t &h, uint32_t &seqInnerSize,
                                const gert::StorageShape *queryShape, const gert::StorageShape *keyShape, const uint32_t n);
    ge::graphStatus GetBasicShape310P(uint32_t &b, uint32_t &bKV, uint32_t &s, uint32_t &h, uint32_t &seqInnerSize,
                                      const gert::StorageShape *queryShape, const gert::StorageShape *keyShape, const uint32_t n,
                                      size_t actualLenDims, size_t actualLenDimsKV);
    ge::graphStatus GetBasicShape910B(uint32_t &b, uint32_t &s, uint32_t &h, uint32_t &seqInnerSize,
                                      const gert::StorageShape *queryShape, const gert::StorageShape *keyShape, const uint32_t n);

    size_t GetPFAWorkSpaceSize(PromptFlashAttentionTilingData& tilingData);
    void GetMatMulType(matmul_tiling::DataType &mmInputType, matmul_tiling::DataType &mmOutputType);
    ge::graphStatus CheckKeyValueParamsConsistency(const ContextParamsForPFATiling& contextKeyParams);
    bool CheckActualSeqLength(ContextParamsForPFATiling& contextKeyParams, uint32_t b, uint32_t sQ, uint32_t sKV,
                              const gert::Tensor* actualSeqLenQ, const gert::Tensor* actualSeqLenKV, InputLayout inLayout, PromptFlashAttentionTilingData& tilingData);
    bool CheckPseShiftTypeAndShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape *pseShiftShape,
                                   uint32_t b, uint32_t n, uint32_t s1, uint32_t s2);
    bool CheckPATypeAndShape(ContextParamsForPFATiling& contextKeyParams, const gert::Tensor* actualSeqLenKV,
                                   int32_t b, int32_t n, int32_t h, int32_t headNumRatio);
    bool CheckAttenMaskShape(ContextParamsForPFATiling& contextKeyParams, const int32_t* sparseMode, const gert::StorageShape* attenMaskShape,
                             uint32_t sQ, uint32_t sK, uint32_t batchSize);
    bool CheckAntiquantParamsShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* antiquantScaleShape,
                                   const gert::StorageShape* antiquantOffsetShape, const uint32_t n, const uint32_t d, const uint32_t h,
                                   PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus CheckPostQuantParams(const ContextParamsForPFATiling& contextKeyParams, uint32_t h, uint32_t n) const;
    ge::graphStatus PromptFlashAttentionCVDiffSetTensorSize(PromptFlashAttentionTilingData& tilingData,
        PromptAttentionSingleCoreTensorSize& tensorSize, uint32_t sOuterFactor,
        uint32_t sInnerFactor, uint32_t softmaxSOuterFactor);
    bool PromptFlashAttentionComputeCVDiffParams(PromptFlashAttentionTilingData& tilingData,
        int64_t ubSize, int64_t l1Size, int64_t l0CSize, uint32_t typeByteSize,
        uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t maskTypeSize, uint32_t &softmaxSOuterFactor);
    bool FindOptimalTilingBasicBLock(PromptFlashAttentionTilingData& tilingData,
        uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t &softmaxSOuterFactor,
        int64_t ubSize, uint32_t typeByteSize, uint32_t maskTypeSize);
    bool FindOptimalTilingSouter(PromptFlashAttentionTilingData& tilingData,
        uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t &softmaxSOuterFactor,
        int64_t ubSize, uint32_t typeByteSize, uint32_t maskTypeSize);
    void InferTilingMod(const ContextParamsForPFATiling& contextKeyParams, const std::vector<int64_t>& actualSeqLengths, const std::vector<int64_t>& actualSeqLengthsKV,
        uint32_t actualSeqArrayLen, uint32_t hDivN, uint32_t seqInnerSize, int32_t sparseModeVal);
    ge::graphStatus AdjustCVTiling(uint32_t hDivN, uint32_t n, int64_t middle_actualSeqLengths,
        int64_t ubSize, int64_t l1Size, int64_t l0CSize, uint32_t maskElemSize,
        uint32_t& sOuterFactor, uint32_t& sInnerFactor, PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus AdjustCVTilingCVDiff(int64_t ubSize, int64_t l1Size, int64_t l0CSize,
        uint32_t maskElemSize, uint32_t& sOuterFactor, uint32_t& sInnerFactor, uint32_t& softmaxSOuterFactor,
        PromptFlashAttentionTilingData& tilingData);
    bool CheckSparseModeRightDown(ContextParamsForPFATiling& contextKeyParams, const std::vector<int64_t>& actualSeqLengths,
                                  const std::vector<int64_t>& actualSeqLengthsKV, size_t lenDims);
    ge::graphStatus GetAndCheckEmptyQueryShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape *queryShape) const;
    void UpdateTilingKeyFlag(ContextParamsForPFATiling& contextKeyParams, uint64_t& tilingKey);
    int64_t PromptFlashAttentionSetMsdUbSize(PromptFlashAttentionTilingData& tilingData, PromptAttentionSingleCoreTensorSize& tensorSize, int32_t sInnerFactorTmp) const;
  
    ge::graphStatus CheckIOType(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData, int32_t& outputDataTypeSize);
    ge::graphStatus CheckMaskType(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData, uint32_t& maskElemSize);
    void SetMaskSize(const gert::StorageShape* attenMaskShape, PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus CheckShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* queryShape, const gert::StorageShape* keyShape, 
                               const gert::StorageShape* valueShape, const gert::StorageShape* outShape, const gert::StorageShape* pseShiftShape,
                               const gert::StorageShape* attenMaskShape);
    uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum);
    bool CalcUBSize();
    void SetDataCopyTransposeTiling();
    void SetSoftMaxTiling();
    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1);
    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2);
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                        int64_t tmpDBasicBlock, int64_t batch,
                        matmul_tiling::MatmulApiTiling &bmm1,
                        matmul_tiling::MatmulApiTiling &bmm2);
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch = 1);
    void CalcS1S2BasicBlock(const BufferNum &bufferNum);
    int64_t CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum);
    int64_t CalcMaxS2BasicBlockSize(const BufferNum &bufferNum, int64_t tmpS1BasicBlock);
    bool IsBasicBlockInSoftMax(const ge::Shape &shape);
    void GetBufferNum(BufferNum &bufferNum);
    void SetMultiBatchCoreParams();
    void MatchTemplate();
    void SetMultiCoreParams();
    void SetTensorSizeParams();
    void SetSparseParams();
    bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx);
    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, PFAMultiCoreParams &multiCoreParams);
    bool PartitionSparseData(const std::vector<int64_t> &sparseRollingArray,
                            int64_t sparseRollingArraySum, int64_t sparseArraySize,
                            int64_t loadMaxEachCore, std::vector<int64_t> &partitionResult);
    // TND新增
    void GetActualSeqLenData(int64_t inputIdx, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &res, int64_t &actualLen);
    void SetMultiCoreParamsTND();
    void SetSparseParamsTND();
    bool InitSparseValidArrayTND(std::vector<int64_t> &sparseValidArray, int64_t bIdx);
    bool SetSparseStartIdxTND(const std::vector<int64_t> &sparseValidArray, PFAMultiCoreParams &multiCoreParams);
    int64_t GetS2RealSize(uint8_t sparseType, int32_t bOutIdx, int64_t s1OutIdx);
    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray, PFAMultiCoreParams &multiCoreParams,
                    std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx);
    bool InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAivNum, int64_t totalSize,
                      const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue);
    ge::graphStatus CheckInputShapeWhenLayoutIsTND(ContextParamsForPFATiling& contextKeyParams);
    ge::graphStatus CheckActSeqWhenLayoutIsTND(ContextParamsForPFATiling& contextKeyParams);

    protected:
        ContextParamsForPFATiling* contextKeyParamsPtr = nullptr;
        int64_t ubSizeRemain = 1;
        bool isSOuterNoTail = true;
        bool isSInnerNoTail = true;
        bool isDNoTail = true;
        bool enableKvAntiquant = false;
        bool enableMsd = false;
        bool enableQuantBF16 = false;
        bool enableMatmulNorm = false;
        bool enablePA = false;
        bool isKVHasPrefix = false;
        InputLayout inputLayout = InputLayout::BSH;
        ge::DataType inputType{ge::DT_FLOAT16};
        ge::DataType outputType{ge::DT_FLOAT16};
        ge::DataType intputKeyType{ge::DT_FLOAT16};
        ge::DataType intputValueType{ge::DT_FLOAT16};
        ge::DataType pseShiftElemType{ge::DT_FLOAT16};
        uint32_t dataTypeSize = FLOAT32SIZE;
        uint32_t coreNum;
        uint32_t aivNum;
        uint32_t aicNum;
        uint32_t typeByteNum;
        uint32_t outputTypeByteNum;
        uint32_t softmaxTypeByteNum;
        uint32_t pseShiftTypeByteNum = 0;
        uint32_t pseShiftElemSize = 0;
        uint32_t pseMaskMaxSize = 0;
        uint32_t pseShiftBatch = 0;
        uint32_t pseShiftS1 = 0;
        uint32_t pseShiftS2 = 0;
        uint32_t usePseShift = 0;
        uint32_t tmpS2 = 0;  // In the PA scenario, there is no S2 axis. Use the change amount to normalize the S2 length in both PA and non PA scenarios
        int32_t blockTableDim2 = 1;
        int32_t PABlockNumSum = 1;
        uint32_t maskTypeByteNum;
        uint32_t maxQuerySeq = 0;
        int64_t apiTmpSize = 1;
        uint32_t softmaxDataTypeNZ_ = FLOAT32SIZE;
        uint32_t softmaxDataTypeSize = FLOAT32SIZE; // BF16 calculates through FP32
        platform_ascendc::SocVersion curShortSocName;
        uint32_t dataTypeSize_ = 4;
        uint32_t layoutType = 0;
        uint32_t PAlayoutType = 0;
        platform_ascendc::PlatformAscendC ascendcPlatform;
        TilingMod tilingMod = TilingMod::CVSAME;
        SplitCoreMode splitCoreMode = SplitCoreMode::SPLIT_NBS_VECTOR;
        uint32_t splitD = 0;
        uint32_t splitS2 = 1; // It can only be 0 when the D axis is split
        uint64_t innerPrecise = HIGH_PERFORMANCE;
        size_t defaultSysWorkspaceSize;
        matmul_tiling::PlatformInfo ascendPlatformInfo;

        int64_t alignedS1;
        int64_t alignedS2;
        int64_t alignedD;

        int64_t s1BasicBlock;
        int64_t s2BasicBlock;
        int64_t s1BasicBlockBest;
        int64_t s1VecBasicBlock;
        int64_t dBasicBlock;
        int64_t batchBasic = 1LL;
        int64_t nRatio;

        int64_t s1Size;
        int64_t s2Size;
        int64_t dSize;
        int64_t valueDSize;
        int64_t s1SparseValidSize;
        int64_t s2SparseValidSize;

        int64_t apiMaxUBSize = 0;

        bool atbRunFlag_ = false;
        bool mlaRunFlag_ = false;

        // TND新增
        int64_t realT1Size;
        std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenData;
        std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenKvData;
        int64_t accumS1;
        int64_t accumS2;
        int64_t bandIndex;

        int64_t bSize;
        int64_t gSize;
        int64_t n1Size;
        int64_t n2Size;
        int64_t s1StrideSize; // query Shape S inner axes, for bmm1
        int64_t s2StrideSize; // key Shape S inner axes, for bmm1
        int64_t maxS1Val;
        int64_t maxS2Val;

        int64_t h1;
        int64_t h2;

        int64_t s2sizeLimitMax = 1024;
        int64_t accumS1BlockNum;

        bool isSameAB = true;
        MLAGeneralTilingData mlaTilingData;
    };
    // end of class PromptFlashAttention
  PFA_EXTERN_C ge::graphStatus TilingPromptFlashAttention(gert::TilingContext* context);
}

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_PROMPTFLASHATTENTION_H_
