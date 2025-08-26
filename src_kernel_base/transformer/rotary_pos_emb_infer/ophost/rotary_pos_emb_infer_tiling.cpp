/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotary_pos_emb_infer_tiling.cc
 * \brief
 */

#include <climits>
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "rotary_pos_emb_infer_tiling.h"


#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGE(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }

#define REPORT_CALL_ERROR REPORT_INNER_ERR_MSG

#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                           \
    if ((ptr) == nullptr) {                                                                          \
        const char *name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
        REPORT_CALL_ERROR("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                           \
        return ret;                                                                                  \
    }

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

namespace optiling {
static constexpr uint32_t BLOCK_SIZE = 16;
static constexpr uint32_t BLOCK_BYTE = 32;
static constexpr uint32_t REMAIN_SPACE = 16 * 1024 * 1024;
static constexpr uint32_t NUM_COSIN = 2;
static constexpr uint32_t NUM_EIGHT = 8;
static constexpr uint32_t TOTAL_OFFSET = 6;
static constexpr uint32_t ELE_NUM_FP16 = 16;  // 一个block fp16元素个数
static constexpr uint32_t LARGE_NTOKENS_THRESHOLD = 64;
static constexpr uint32_t SLICE_SIZE_FP16_LARGE_NTOKENS = 4096;
static constexpr uint32_t TILING_BF16_ALIGN_BROARD = 33;
static constexpr uint32_t TILING_BF16_BROARD = 32;
static constexpr uint32_t TILING_HIGH_PREC_BOARD = 31;
static constexpr uint32_t TILING_HIGH_PERF_BROARD = 30;
static constexpr uint32_t TILING_BF16_ALIGN = 24;
static constexpr uint32_t TILING_BF16 = 22;
static constexpr uint32_t TILING_HIGH_PREC = 21;
static constexpr uint32_t TILING_HIGH_PERF = 20;
static constexpr uint32_t TILING_HIGH_PERF_LARGE_NTOKENS = 23;

void PrintTilingData(gert::TilingContext *context, RotaryPosEmbInferTilingData *tilingDataPtr)
{
    OP_LOGD(context->GetNodeName(), "Start RotaryPosEmbInferTilingData priting");
    OP_LOGD(context->GetNodeName(), "------------------------------------------");
    OP_LOGD(context->GetNodeName(), "------------------------------------------");
    OP_LOGD(context->GetNodeName(), "hiddenSizeQ is %u", tilingDataPtr->get_hiddenSizeQ());
    OP_LOGD(context->GetNodeName(), "hiddenSizeK is %u", tilingDataPtr->get_hiddenSizeK());
    OP_LOGD(context->GetNodeName(), "headDim is %u", tilingDataPtr->get_headDim());
    OP_LOGD(context->GetNodeName(), "headNumQ is %u", tilingDataPtr->get_headNumQ());
    OP_LOGD(context->GetNodeName(), "headNumK is %u", tilingDataPtr->get_headNumK());
    OP_LOGD(context->GetNodeName(), "rotaryCoeff is %u", tilingDataPtr->get_rotaryCoeff());
    OP_LOGD(context->GetNodeName(), "ntokens is %u", tilingDataPtr->get_ntokens());
    OP_LOGD(context->GetNodeName(), "realCore is %u", tilingDataPtr->get_realCore());
    OP_LOGD(context->GetNodeName(), "cosFormat is %u", tilingDataPtr->get_cosFormat());
    OP_LOGD(context->GetNodeName(), "batch is %u", tilingDataPtr->get_batch());
    OP_LOGD(context->GetNodeName(), "multiple is %u", tilingDataPtr->get_multiple());
    OP_LOGD(context->GetNodeName(), "maxUbSize is %u", tilingDataPtr->get_maxUbSize());
    OP_LOGD(context->GetNodeName(), "blockDim is %u", context->GetBlockDim());
    OP_LOGD(context->GetNodeName(), "tilingKey is %lu", context->GetTilingKey());
    OP_LOGD(context->GetNodeName(), "------------------------------------------");
    OP_LOGD(context->GetNodeName(), "------------------------------------------");
    OP_LOGD(context->GetNodeName(), "End RotaryPosEmbInferTilingData priting");
}

ge::graphStatus RopeNdProcess(gert::TilingContext *context, RotaryPosEmbInferTilingData *tilingDataPtr)
{
    // 0 is Q index, 1 is dim of hiddenSize
    uint32_t hiddenSizeQ = static_cast<uint32_t>(context->GetInputShape(0)->GetStorageShape().GetDim(1));
    // 1 is K index, 1 is dim of hiddenSize
    uint32_t hiddenSizeK = static_cast<uint32_t>(context->GetInputShape(1)->GetStorageShape().GetDim(1));
    auto cosSize = context->GetInputShape(2)->GetStorageShape().GetDimNum();  // 2 is cos index
    // 2 is cos index, 1 is index
    uint32_t headDim = static_cast<uint32_t>(context->GetInputShape(2)->GetStorageShape().GetDim(cosSize - 1));
    // 0 is dim of ntokens
    uint32_t ntokens = static_cast<uint32_t>(context->GetInputShape(0)->GetStorageShape().GetDim(0));
    // 4 is seqlen index, 0 is dim of batch
    uint32_t batch = static_cast<uint32_t>(context->GetInputShape(4)->GetStorageShape().GetDim(0));

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t maxCore = static_cast<uint32_t>(ascendcPlatform.GetCoreNum());
    OP_TILING_CHECK(maxCore == 0, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "fail to get coreNum"),
                    return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    auto maxUbSize = static_cast<uint32_t>(ubSizePlatForm);

    tilingDataPtr->set_maxUbSize(maxUbSize);

    uint32_t multiple = 1;
    bool condition = tilingDataPtr->get_cosFormat() == 0 && cosSize == NUM_COSIN &&
                     context->GetInputDesc(NUM_COSIN)->GetDataType() == ge::DataType::DT_FLOAT16 &&
                     ntokens >= LARGE_NTOKENS_THRESHOLD &&
                     headDim / tilingDataPtr->get_rotaryCoeff() % ELE_NUM_FP16 == 0;
    if (condition) {  // 不对齐场景, multiple为1
        uint32_t hiddenSize = hiddenSizeK > hiddenSizeQ ? hiddenSizeK : hiddenSizeQ;
        multiple = SLICE_SIZE_FP16_LARGE_NTOKENS / hiddenSize;
        multiple = multiple > 0 ? multiple : 1;
        while (ntokens % multiple != 0 && multiple > 1) {
            --multiple;
        }
        if (ntokens / multiple < maxCore || UINT32_MAX / multiple < hiddenSize) {
            multiple = 1;
        } else {
            ntokens /= multiple;
            batch /= multiple;
            hiddenSizeQ *= multiple;
            hiddenSizeK *= multiple;
            headDim *= multiple;
        }
    }

    uint32_t tempCore = (ntokens + maxCore - 1) / maxCore;
    uint32_t realCore = (ntokens + tempCore - 1) / tempCore;
    tilingDataPtr->set_realCore(realCore);
    tilingDataPtr->set_hiddenSizeQ(hiddenSizeQ);
    tilingDataPtr->set_hiddenSizeK(hiddenSizeK);
    tilingDataPtr->set_headDim(headDim);
    tilingDataPtr->set_ntokens(ntokens);
    tilingDataPtr->set_batch(batch);
    tilingDataPtr->set_multiple(multiple);
    context->SetBlockDim(realCore);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus TilingKeyChose(gert::TilingContext *context, RotaryPosEmbInferTilingData *tilingDataPtr)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "fail to get platform info"),
                    return ge::GRAPH_FAILED);
    // 2 is cos index
    auto cosSize = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    if (cosSize == NUM_COSIN) {
        if (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_BF16) {
            context->SetTilingKey(TILING_BF16);  // first 2 for shape dims of cos, second 2 for BF16
        } else if (context->GetInputDesc(NUM_COSIN)->GetDataType() == ge::DataType::DT_FLOAT) {
            context->SetTilingKey(TILING_HIGH_PREC);  // second 1 for FP32
        } else {
            bool condition = tilingDataPtr->get_ntokens() * tilingDataPtr->get_multiple() >= LARGE_NTOKENS_THRESHOLD &&
                             tilingDataPtr->get_cosFormat() == 0;
            if (condition) {  // ntokens >= 64时，走TILING_HIGH_PERF_LARGE_NTOKENS
                context->SetTilingKey(TILING_HIGH_PERF_LARGE_NTOKENS);
            } else {
                context->SetTilingKey(TILING_HIGH_PERF);  // second 0 for FP16
            }
        }
    } else {
        if (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_BF16) {
            uint32_t alignRotary = (tilingDataPtr->get_headDim() / tilingDataPtr->get_rotaryCoeff()) % ELE_NUM_FP16;
            bool condition = (alignRotary == 0) && (tilingDataPtr->get_ntokens() >= LARGE_NTOKENS_THRESHOLD);
            if (condition) {                                      // ntokens >= 64时，走TILING_BF16_ALIGN_BROARD
                context->SetTilingKey(TILING_BF16_ALIGN_BROARD);  // first 2 for shape dims of cos
            } else {
                context->SetTilingKey(TILING_BF16_BROARD);  // first 3 for shape dims of cos
            }
        } else {
            context->SetTilingKey(TILING_HIGH_PREC_BOARD);  // second 0 for fp16
        }
    }
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus Tiling4RotaryPosEmbInfer(gert::TilingContext *context)
{
    RotaryPosEmbInferTilingData tilingData;

    int32_t rotaryCoeff = *(context->GetAttrs()->GetAttrPointer<int32_t>(0));
    int32_t cosFormat = *(context->GetAttrs()->GetAttrPointer<int32_t>(1));
    OP_TILING_CHECK(rotaryCoeff <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "rotaryCoeff is invalid."),
                    return ge::GRAPH_FAILED);
    tilingData.set_rotaryCoeff(static_cast<uint32_t>(rotaryCoeff));
    OP_TILING_CHECK(
        (cosFormat != 0 && cosFormat != 1),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "wrong cosFormat, cosFormat should be 0 or 1."),
        return ge::GRAPH_FAILED);

    tilingData.set_cosFormat(static_cast<uint32_t>(cosFormat));
    uint32_t headNumQ = 1;
    uint32_t headNumK = 1;
    auto ret = RopeNdProcess(context, &tilingData);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (tilingData.get_headDim() != 0) {
        headNumQ = tilingData.get_hiddenSizeQ() / tilingData.get_headDim();
        headNumK = tilingData.get_hiddenSizeK() / tilingData.get_headDim();
    } else {
        OP_LOGE(context->GetNodeName(), "tilingDataPtr->headDim should not be 0.");
        return ge::GRAPH_FAILED;
    }
    tilingData.set_headNumQ(headNumQ);
    tilingData.set_headNumK(headNumK);
    ret = TilingKeyChose(context, &tilingData);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint64_t sysWorkspaceSize =
        static_cast<uint64_t>(REMAIN_SPACE) +
        static_cast<uint64_t>(TOTAL_OFFSET) * tilingData.get_realCore() * tilingData.get_hiddenSizeQ() *
            sizeof(uint16_t) +
        static_cast<uint64_t>(tilingData.get_ntokens()) * tilingData.get_headDim() * NUM_COSIN * sizeof(uint16_t);
    uint64_t syncWorkspaceSize = tilingData.get_realCore() * BLOCK_BYTE;

    size_t *workspace_size = context->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, workspace_size, false);
    workspace_size[0] = sysWorkspaceSize + syncWorkspaceSize;
    OP_LOGD(context->GetNodeName(), "workspace is %lu", sysWorkspaceSize);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    PrintTilingData(context, &tilingData);
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepare4RotaryPosEmbInfer(gert::TilingParseContext *context)
{
    auto compileInfo = GetCompileInfoPtr<Tiling4RotaryPosEmbInferCompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNum();
    OP_TILING_CHECK((compileInfo->coreNum == 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get core num."),
                    return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_TILING_CHECK(compileInfo->ubSizePlatForm == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get ub size."),
                    return ge::GRAPH_FAILED);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RotaryPosEmbInfer)
    .Tiling(Tiling4RotaryPosEmbInfer)
    .TilingParse<Tiling4RotaryPosEmbInferCompileInfo>(TilingPrepare4RotaryPosEmbInfer);
}  // namespace optiling
