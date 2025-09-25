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
 * \file sparse_paged_fusion_attention.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "sparse_paged_fusion_attention_allvec_new.h"
#if (__CCE_AICORE__ > 200)
#include "sparse_paged_fusion_attention_split_Bbn2s2_Us2.h"
#endif
using namespace AscendC;

#define NEED_CUBE_TILING (true)
#define NOT_NEED_CUBE_TILING (false)

#define INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(templateClass, ...)                                                          \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA_PREFIX(tiling, NEED_CUBE_TILING);                                                             \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling, op.mm1Sp,              \
                          bmm1tilingPrefix, op.mm2Sp, bmm2tilingPrefix);                                               \
        op.InitPrefix(query, keySharedPrefix, valueSharedPrefix, pseShift, attenMask, actualSharedPrefixLen,           \
                      blocktable, kvPaddingSize, blockPosition, attentionOut, softmaxLse, user, &tiling_data->tilingPrefix, tiling,   \
                      &tPipe);                                                                                         \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user);          \
        op.Process();                                                                                                  \
        SyncAll(); /* workspace改为每个核单独使用即可去掉此处同步 */                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, blocktable, kvPaddingSize, blockPosition, attentionOut,  \
                softmaxLse, user, &tiling_data->tilingBase, tiling, nullptr);                                          \
        op.Process();                                                                                                  \
        op.ProcessSysPrefixCombine();                                                                                  \
    } while (0)

#define INVOKE_IFA_GENERAL_OP_IMPL(templateClass, ...)                                                                 \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA(tiling, NEED_CUBE_TILING);                                                                    \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                       \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, blocktable, kvPaddingSize, blockPosition, attentionOut,     \
                softmaxLse, user, tiling_data, tiling, &tPipe);                                                        \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user);          \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_IFA_ALL_VEC_OP_IMPL(templateClass, ...)                                                                 \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA_NO_CUBE(tiling);                                                                              \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, blocktable, kvPaddingSize, blockPosition, attentionOut,     \
                softmaxLse, user, tiling_data, &tPipe);                                                                \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     user);                                                                                            \
        op.Process();                                                                                                  \
    } while (0)

#ifdef __DAV_C220_CUBE__
#define COPY_TILING_DATA(tiling, need_cube)                                                                            \
    if constexpr (!need_cube) {                                                                                        \
        return;                                                                                                        \
    }                                                                                                                  \
    GET_TILING_DATA_MEMBER(SparsePagedFusionAttentionTilingDataV2, tilingBase.bmm1TilingData, bmm1TilingDataVar, tiling);                  \
    GET_TILING_DATA_MEMBER(SparsePagedFusionAttentionTilingDataV2, tilingBase.bmm2TilingData, bmm2TilingDataVar, tiling);                  \
    const SparsePagedFusionAttentionTilingData *__restrict tiling_data = nullptr;                                             \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar;

#define COPY_TILING_DATA_PREFIX(tiling, need_cube)                                                                     \
    if constexpr (!need_cube) {                                                                                        \
        return;                                                                                                        \
    }                                                                                                                  \
    GET_TILING_DATA_MEMBER(SparsePagedFusionAttentionTilingDataV2, tilingBase.bmm1TilingData, bmm1TilingDataVar, tiling);     \
    GET_TILING_DATA_MEMBER(SparsePagedFusionAttentionTilingDataV2, tilingBase.bmm2TilingData, bmm2TilingDataVar, tiling);     \
    GET_TILING_DATA_MEMBER(SparsePagedFusionAttentionTilingDataV2, tilingPrefix.base.bmm1TilingData, bmm1TilingDataVarPrefix, \
                           tiling);                                                                                    \
    GET_TILING_DATA_MEMBER(SparsePagedFusionAttentionTilingDataV2, tilingPrefix.base.bmm2TilingData, bmm2TilingDataVarPrefix, \
                           tiling);                                                                                    \
    const SparsePagedFusionAttentionTilingDataV2 *__restrict tiling_data = nullptr;                                           \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm1tilingPrefix = &bmm1TilingDataVarPrefix;                                         \
    const TCubeTiling *__restrict bmm2tilingPrefix = &bmm2TilingDataVarPrefix;

#define COPY_TILING_DATA_NO_CUBE(tiling) COPY_TILING_DATA(tiling, NOT_NEED_CUBE_TILING);

#define COPY_BMM1_TILING_DATA(tiling, need_cube)                                                                       \
    if constexpr (!need_cube) {                                                                                        \
        return;                                                                                                        \
    }                                                                                                                  \
    GET_TILING_DATA_MEMBER(SparsePagedFusionAttentionTilingDataV2, tilingBase.bmm1TilingData, bmm1TilingDataVar, tiling);                  \
    const SparsePagedFusionAttentionTilingData *__restrict tiling_data = nullptr;                                             \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;

#define COPY_BMM2_TILING_DATA(tiling, need_cube)                                                                       \
    if constexpr (!need_cube) {                                                                                        \
        return;                                                                                                        \
    }                                                                                                                  \
    GET_TILING_DATA_MEMBER(SparsePagedFusionAttentionTilingDataV2, tilingBase.bmm2TilingData, bmm2TilingDataVar, tiling);                  \
    const SparsePagedFusionAttentionTilingData *__restrict tiling_data = nullptr;                                             \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar;
#else
#define COPY_TILING_DATA(tiling, need_cube)                                                                            \
    GET_TILING_DATA_MEMBER(SparsePagedFusionAttentionTilingDataV2, tilingBase, tiling_data_in, tiling);                                \
    const SparsePagedFusionAttentionTilingData *__restrict tiling_data = &tiling_data_in;                                     \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                         \
    const TCubeTiling *__restrict bmm2tiling = nullptr;

#define COPY_TILING_DATA_PREFIX(tiling, need_cube)                                                                     \
    GET_TILING_DATA_WITH_STRUCT(SparsePagedFusionAttentionTilingDataV2, tiling_data_in, tiling);                              \
    const SparsePagedFusionAttentionTilingDataV2 *__restrict tiling_data = &tiling_data_in;                                   \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm2tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm1tilingPrefix = nullptr;                                                          \
    const TCubeTiling *__restrict bmm2tilingPrefix = nullptr;

#define COPY_TILING_DATA_NO_CUBE(tiling)                                                                               \
    GET_TILING_DATA_MEMBER(SparsePagedFusionAttentionTilingDataV2, tilingBase, tiling_data_in, tiling);                                \
    const SparsePagedFusionAttentionTilingData *__restrict tiling_data = &tiling_data_in;

#define COPY_BMM1_TILING_DATA(tiling, need_cube) COPY_TILING_DATA(tiling, need_cube)
#define COPY_BMM2_TILING_DATA(tiling, need_cube) COPY_TILING_DATA(tiling, need_cube)
#endif

extern "C" __global__ __aicore__ void sparse_paged_fusion_attention_FIAS(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1,
    __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
    __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset,
    __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix,
    __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *keyRopeAntiquantScale,
    __gm__ uint8_t *blockPosition,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;

    /*
    获取Op可用WorkSpace空间
    **/
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

#if (__CCE_AICORE__ > 200)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif


#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_FLOAT16)
    if (TILING_KEY_IS(11000000000000000)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, half, half, half, false, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(11000000000100000)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, half, half, half, false, true,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(11000000000000001)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, half, half, half, false, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(11000000000100001)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, half, half, half, false, true, LAYOUT::BSH);
#if (__CCE_AICORE__ > 200)
    } else if (TILING_KEY_IS(10000000000000000)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000000000)) {
        KERNEL_TASK_TYPE(12000000000000000, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000200000)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, true, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000200000)) {
        KERNEL_TASK_TYPE(12000000000200000, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, true, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000100000)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, true,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000300000)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, true, true,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000000001)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000000001)) {
        KERNEL_TASK_TYPE(12000000000000001, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000200001)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, true, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000200001)) {
        KERNEL_TASK_TYPE(12000000000200001, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, true, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000100001)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, true,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000300001)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, true, true,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(20000000000000000)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, false,
                                   LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000100000)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, true,
                                   LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000000001)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, false,
                                   LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000100001)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, true,
                                   LAYOUT::BSH, 0, true);
#else
    } else if (TILING_KEY_IS(11000000000200000)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, half, half, half, true, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(11000000000300000)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, half, half, half, true, true,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(11000000000200001)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, half, half, half, true, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(11000000000300001)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, half, half, half, true, true,
                                   LAYOUT::BSH);
#endif
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_INT8)
#if (__CCE_AICORE__ > 200)
    if (TILING_KEY_IS(10000000000000300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000000300)) {
        KERNEL_TASK_TYPE(12000000000000300, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000200300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000200300)) {
        KERNEL_TASK_TYPE(12000000000200300, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000100300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000300300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, true,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000000301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000000301)) {
        KERNEL_TASK_TYPE(12000000000000301, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000200301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000200301)) {
        KERNEL_TASK_TYPE(12000000000200301, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000100301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000300301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, true,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000400300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(12000000000400300)) {
        KERNEL_TASK_TYPE(12000000000400300, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000600300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(12000000000600300)) {
        KERNEL_TASK_TYPE(12000000000600300, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000500300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000700300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, true,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000400301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(12000000000400301)) {
        KERNEL_TASK_TYPE(12000000000400301, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000600301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(12000000000600301)) {
        KERNEL_TASK_TYPE(12000000000600301, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000500301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000700301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, true,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000002000300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD, 2);
    } else if (TILING_KEY_IS(12000000002000300)) {
        KERNEL_TASK_TYPE(12000000002000300, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD, 2);
    } else if (TILING_KEY_IS(10000000002200300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD, 2);
    } else if (TILING_KEY_IS(12000000002200300)) {
        KERNEL_TASK_TYPE(12000000002200300, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD, 2);
    } else if (TILING_KEY_IS(10000000002100300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                   LAYOUT::BNSD, 2);
    } else if (TILING_KEY_IS(10000000002300300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, true,
                                   LAYOUT::BNSD, 2);
    } else if (TILING_KEY_IS(10000000002000301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH, 2);
    } else if (TILING_KEY_IS(12000000002000301)) {
        KERNEL_TASK_TYPE(12000000002000301, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH, 2);
    } else if (TILING_KEY_IS(10000000002200301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH, 2);
    } else if (TILING_KEY_IS(12000000002200301)) {
        KERNEL_TASK_TYPE(12000000002200301, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH, 2);
    } else if (TILING_KEY_IS(10000000002100301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                   LAYOUT::BSH, 2);
    } else if (TILING_KEY_IS(10000000002300301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, true,
                                   LAYOUT::BSH, 2);
    } else if (TILING_KEY_IS(20000000000000300)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false,
                                          false, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000100300)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                          LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000000301)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false,
                                          false, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000100301)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                          LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000400300)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false,
                                          false, LAYOUT::BNSD, 1, true);
    } else if (TILING_KEY_IS(20000000000500300)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                          LAYOUT::BNSD, 1, true);
    } else if (TILING_KEY_IS(20000000000400301)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false,
                                          false, LAYOUT::BSH, 1, true);
    } else if (TILING_KEY_IS(20000000000500301)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                          LAYOUT::BSH, 1, true);
    } else if (TILING_KEY_IS(20000000002000300)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false,
                                          false, LAYOUT::BNSD, 2, true);
    } else if (TILING_KEY_IS(20000000002100300)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                          LAYOUT::BNSD, 2, true);
    } else if (TILING_KEY_IS(20000000002000301)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false,
                                          false, LAYOUT::BSH, 2, true);
    } else if (TILING_KEY_IS(20000000002100301)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                          LAYOUT::BSH, 2, true);
    }
#else
    if (TILING_KEY_IS(11000000000000300)) { // kvDeq
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(11000000000100300)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, int8_t, half, half, false, true,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(11000000000000301)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(11000000000100301)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, int8_t, half, half, false, true,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(11000000000200300)) { // pageAttention + kvDeq
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(11000000000300300)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, int8_t, half, half, true, true,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(11000000000200301)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(11000000000300301)) {
        INVOKE_IFA_ALL_VEC_OP_IMPL(SparsePagedFusionAttentionAttenAllVecNew, half, int8_t, half, half, true, true,
                                   LAYOUT::BSH);
    }
#endif
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_INT4)
#if (__CCE_AICORE__ > 200)
    if (TILING_KEY_IS(10000000000000400)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000000400)) {
        KERNEL_TASK_TYPE(12000000000000400, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000100400)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, true,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000000401)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000000401)) {
        KERNEL_TASK_TYPE(12000000000000401, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000100401)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, true,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000400400)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(12000000000400400)) {
        KERNEL_TASK_TYPE(12000000000400400, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000500400)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, true,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000400401)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(12000000000400401)) {
        KERNEL_TASK_TYPE(12000000000400401, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000500401)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, true,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000002000400)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD, 2);
    } else if (TILING_KEY_IS(12000000002000400)) {
        KERNEL_TASK_TYPE(12000000002000400, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD, 2);
    } else if (TILING_KEY_IS(10000000002100400)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, true,
                                   LAYOUT::BNSD, 2);
    } else if (TILING_KEY_IS(10000000002000401)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH, 2);
    } else if (TILING_KEY_IS(12000000002000401)) {
        KERNEL_TASK_TYPE(12000000002000401, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH, 2);
    } else if (TILING_KEY_IS(10000000002100401)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, true,
                                   LAYOUT::BSH, 2);
    } else if (TILING_KEY_IS(20000000000000400)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          false, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000100400)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          true, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000000401)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          false, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000100401)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          true, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000400400)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          false, LAYOUT::BNSD, 1, true);
    } else if (TILING_KEY_IS(20000000000500400)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          true, LAYOUT::BNSD, 1, true);
    } else if (TILING_KEY_IS(20000000000400401)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          false, LAYOUT::BSH, 1, true);
    } else if (TILING_KEY_IS(20000000000500401)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          true, LAYOUT::BSH, 1, true);
    } else if (TILING_KEY_IS(20000000002000400)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          false, LAYOUT::BNSD, 2, true);
    } else if (TILING_KEY_IS(20000000002100400)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          true, LAYOUT::BNSD, 2, true);
    } else if (TILING_KEY_IS(20000000002000401)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          false, LAYOUT::BSH, 2, true);
    } else if (TILING_KEY_IS(20000000002100401)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          true, LAYOUT::BSH, 2, true);
    }
#endif
#endif

#if (__CCE_AICORE__ > 200) // new template

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8) && (ORIG_DTYPE_KEY == DT_FLOAT16)
    if (TILING_KEY_IS(10000000000003000)) { // A16W16 fp16, out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000003000)) {
        KERNEL_TASK_TYPE(12000000000003000, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000203000)) { // A16W16 fp16, out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, true, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000203000)) {
        KERNEL_TASK_TYPE(12000000000203000, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, true, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000103000)) { // A16W16 fp16, out int8, FD
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false, true,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000303000)) { // A16W16 fp16, out int8, FD
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, true, true,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000003001)) { // A16W16 fp16, out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000003001)) {
        KERNEL_TASK_TYPE(12000000000003001, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000203001)) { // A16W16 fp16, out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, true, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000203001)) {
        KERNEL_TASK_TYPE(12000000000203001, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, true, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000103001)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false, true,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000303001)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, true, true,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(20000000000003000)) { // A16W16 fp16, out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false,
                                          false, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000103000)) { // A16W16 fp16, out int8, FD
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false,
                                          true, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000003001)) { // A16W16 fp16, out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false,
                                          false, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000103001)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false,
                                          true, LAYOUT::BSH, 0, true);
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8) && (ORIG_DTYPE_KEY == DT_INT8)
    if (TILING_KEY_IS(10000000000003300)) { // A16W16 fp16, out int
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000003300)) {
        KERNEL_TASK_TYPE(12000000000003300, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000203300)) { // A16W8 fp16, out int
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000203300)) {
        KERNEL_TASK_TYPE(12000000000203300, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000103300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, true,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000303300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, true,
                                   LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000003301)) { // A16W8 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000003301)) {
        KERNEL_TASK_TYPE(12000000000003301, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000203301)) { // A16W8 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000203301)) {
        KERNEL_TASK_TYPE(12000000000203301, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000103301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, true,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000303301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, true,
                                   LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000403300)) { // A16W8 fp16, pertoken, out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(12000000000403300)) {
        KERNEL_TASK_TYPE(12000000000403300, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000603300)) { // A16W8 fp16, pertoken, out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(12000000000603300)) {
        KERNEL_TASK_TYPE(12000000000603300, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000503300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, true,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000703300)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, true,
                                   LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000403301)) { // A16W8 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(12000000000403301)) {
        KERNEL_TASK_TYPE(12000000000403301, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000603301)) { // A16W8 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(12000000000603301)) {
        KERNEL_TASK_TYPE(12000000000603301, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000503301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, true,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000703301)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, true,
                                   LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(20000000000003300)) { // A16W8 fp16, out int
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          false, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000103300)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          true, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000003301)) { // A16W8 out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          false, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000103301)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          true, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000403300)) { // A16W8 fp16, pertoken, out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          false, LAYOUT::BNSD, 1, true);
    } else if (TILING_KEY_IS(20000000000503300)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          true, LAYOUT::BNSD, 1, true);
    } else if (TILING_KEY_IS(20000000000403301)) { // A16W8 out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          false, LAYOUT::BSH, 1, true);
    } else if (TILING_KEY_IS(20000000000503301)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          true, LAYOUT::BSH, 1, true);
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_BF16)
    if (TILING_KEY_IS(10000000000022220)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000022220)) {
        KERNEL_TASK_TYPE(12000000000022220, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000222220)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000222220)) {
        KERNEL_TASK_TYPE(12000000000222220, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000122220)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000322220)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, true, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000022221)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000022221)) {
        KERNEL_TASK_TYPE(12000000000022221, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000222221)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000222221)) {
        KERNEL_TASK_TYPE(12000000000222221, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000122221)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000322221)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, true, LAYOUT::BSH);
    } else if (TILING_KEY_IS(20000000000022220)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000122220)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t,
                                          bfloat16_t, bfloat16_t, false, true, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000022221)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000122221)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 0, true);
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_INT8)
    if (TILING_KEY_IS(10000000000022320)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000022320)) {
        KERNEL_TASK_TYPE(12000000000022320, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000222320)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000222320)) {
        KERNEL_TASK_TYPE(12000000000222320, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000122320)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000322320)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, true, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000022321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000022321)) {
        KERNEL_TASK_TYPE(12000000000022321, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000222321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000222321)) {
        KERNEL_TASK_TYPE(12000000000222321, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000122321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000322321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, true, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000422320)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(12000000000422320)) {
        KERNEL_TASK_TYPE(12000000000422320, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000622320)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(12000000000622320)) {
        KERNEL_TASK_TYPE(12000000000622320, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000522320)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000722320)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, true, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000422321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(12000000000422321)) {
        KERNEL_TASK_TYPE(12000000000422321, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000622321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(12000000000622321)) {
        KERNEL_TASK_TYPE(12000000000622321, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000522321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000722321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, true, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(20000000000022320)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000122320)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000022321)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000122321)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000422320)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 1, true);
    } else if (TILING_KEY_IS(20000000000522320)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 1, true);
    } else if (TILING_KEY_IS(20000000000422321)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 1, true);
    } else if (TILING_KEY_IS(20000000000522321)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 1, true);
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_INT4)
    if (TILING_KEY_IS(10000000000022420)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000022420)) {
        KERNEL_TASK_TYPE(12000000000022420, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000122420)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000022421)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000022421)) {
        KERNEL_TASK_TYPE(12000000000022421, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000122421)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000422420)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(12000000000422420)) {
        KERNEL_TASK_TYPE(12000000000422420, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000522420)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000422421)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(12000000000422421)) {
        KERNEL_TASK_TYPE(12000000000422421, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000522421)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000002022420)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 2);
    } else if (TILING_KEY_IS(12000000002022420)) {
        KERNEL_TASK_TYPE(12000000002022420, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 2);
    } else if (TILING_KEY_IS(10000000002122420)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD, 2);
    } else if (TILING_KEY_IS(10000000002022421)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 2);
    } else if (TILING_KEY_IS(12000000002022421)) {
        KERNEL_TASK_TYPE(12000000002022421, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 2);
    } else if (TILING_KEY_IS(10000000002122421)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BSH, 2);
    } else if (TILING_KEY_IS(20000000000022420)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000122420)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000022421)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000122421)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000422420)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 1, true);
    } else if (TILING_KEY_IS(20000000000522420)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 1, true);
    } else if (TILING_KEY_IS(20000000000422421)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 1, true);
    } else if (TILING_KEY_IS(20000000000522421)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 1, true);
    } else if (TILING_KEY_IS(20000000002022420)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 2, true);
    } else if (TILING_KEY_IS(20000000002122420)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 2, true);
    } else if (TILING_KEY_IS(20000000002022421)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 2, true);
    } else if (TILING_KEY_IS(20000000002122421)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 2, true);
    }

#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8) && (ORIG_DTYPE_KEY == DT_BF16)
    if (TILING_KEY_IS(10000000000023220)) { // A16W16 BF16 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000023220)) {
        KERNEL_TASK_TYPE(12000000000023220, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000223220)) { // A16W16 BF16 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000223220)) {
        KERNEL_TASK_TYPE(12000000000223220, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000123220)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000323220)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, true, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000023221)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000023221)) {
        KERNEL_TASK_TYPE(12000000000023221, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000223221)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000223221)) {
        KERNEL_TASK_TYPE(12000000000223221, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000123221)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, true, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000323221)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, true, LAYOUT::BSH);
    } else if (TILING_KEY_IS(20000000000023220)) { // A16W16 BF16 out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000123220)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000023221)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000123221)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 0, true);
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8) && (ORIG_DTYPE_KEY == DT_INT8)
    if (TILING_KEY_IS(10000000000023320)) { // A16W8 BF16 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000023320)) {
        KERNEL_TASK_TYPE(12000000000023320, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000223320)) { // A16W8 BF16 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(12000000000223320)) {
        KERNEL_TASK_TYPE(12000000000223320, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000123320)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000323320)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   true, LAYOUT::BNSD);
    } else if (TILING_KEY_IS(10000000000023321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000023321)) {
        KERNEL_TASK_TYPE(12000000000023321, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000223321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(12000000000223321)) {
        KERNEL_TASK_TYPE(12000000000223321, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000123321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, true, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000323321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   true, LAYOUT::BSH);
    } else if (TILING_KEY_IS(10000000000423320)) { // A16W8 BF16 out int8 per token
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(12000000000423320)) {
        KERNEL_TASK_TYPE(12000000000423320, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000623320)) { // A16W8 BF16 out int8 per token
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(12000000000623320)) {
        KERNEL_TASK_TYPE(12000000000623320, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000523320)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000723320)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   true, LAYOUT::BNSD, 1);
    } else if (TILING_KEY_IS(10000000000423321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(12000000000423321)) {
        KERNEL_TASK_TYPE(12000000000423321, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000623321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(12000000000623321)) {
        KERNEL_TASK_TYPE(12000000000623321, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000523321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, true, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(10000000000723321)) {
        INVOKE_IFA_GENERAL_OP_IMPL(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   true, LAYOUT::BSH, 1);
    } else if (TILING_KEY_IS(20000000000023320)) { // A16W8 BF16 out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000123320)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 0, true);
    } else if (TILING_KEY_IS(20000000000023321)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000123321)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 0, true);
    } else if (TILING_KEY_IS(20000000000423320)) { // A16W8 BF16 out int8 per token
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 1, true);
    } else if (TILING_KEY_IS(20000000000523320)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 1, true);
    } else if (TILING_KEY_IS(20000000000423321)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 1, true);
    } else if (TILING_KEY_IS(20000000000523321)) {
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(SparsePagedFusionAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 1, true);
    }

#endif

#endif // new template
}

extern "C" __global__ __aicore__ void
sparse_paged_fusion_attention(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
                      __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *deqScale1,
                      __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
                      __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                      __gm__ uint8_t *blocktable, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *blockPosition, __gm__ uint8_t *attentionOut,
                      __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    sparse_paged_fusion_attention_FIAS(query, key, value, pseShift, attenMask, nullptr, actualSeqLengths, deqScale1,
                               quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,
                               blocktable, kvPaddingSize, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, blockPosition, attentionOut, nullptr, workspace, tiling);
}
