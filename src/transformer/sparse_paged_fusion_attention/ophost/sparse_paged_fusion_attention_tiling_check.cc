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
 * \file sparse_paged_fusion_attention_tiling_check.cc
 * \brief
 */

#include "sparse_paged_fusion_attention_tiling.h"
#include "sparse_paged_fusion_attention_tiling_base.h"
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "error/ops_error.h"
#include "register/op_def_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus SparseFusionIFATiling::CheckPABlockSize()
{
    OPS_ERR_IF(
        blockSize_ == 0,
        OPS_LOG_E(context_->opName, "When Page Attention is enabled, input attribute blocksize can not be 0."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockSize_ > MAX_BLOCK_SIZE,
                OPS_LOG_E(context_->opName,
                            "When Page Attention is enabled, input attribute blocksize %u can not be larger than %u.",
                            blockSize_, MAX_BLOCK_SIZE),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(((inputKvType_ == ge::DT_INT8) && (blockSize_ % 32 != 0)),
                OPS_LOG_E(context_->opName, "When Page Attention is enabled, if kv cache dtype is int8, input attr "
                                            "blocksize[%u] should be 32 aligned.",blockSize_),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(((inputKvType_ == ge::DT_FLOAT16) || (inputKvType_ == ge::DT_BF16)) && (blockSize_ % 16 != 0),
                OPS_LOG_E(context_->opName,
                            "When Page Attention is enabled, "
                            "if kv cache dtype is float16/bfloat16, input attr blocksize should be 16 aligned"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::CheckBaseInputsNull() {
    // Check base input tensors
    OPS_ERR_IF(context_->query.shape == nullptr, OPS_LOG_E(context_->opName, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->query.shape->GetStorageShape().GetShapeSize() == 0,
               OPS_LOG_E(context_->opName, "Tensor q is empty cause shapesize is 0."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->query.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->key.shape == nullptr, OPS_LOG_E(context_->opName, "Shape of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->key.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->value.shape == nullptr, OPS_LOG_E(context_->opName, "Shape of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->value.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->attenOut.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->attenOut.shape == nullptr, OPS_LOG_E(context_->opName, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);

    // Check base input attrs
    OPS_ERR_IF(context_->numHeads == nullptr, OPS_LOG_E(context_->opName, "attr numHeads is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->scaleValue == nullptr, OPS_LOG_E(context_->opName, "attr scaleValue is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->kvHeadNums == nullptr, OPS_LOG_E(context_->opName, "attr kvHeadNums is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->layOut == nullptr, OPS_LOG_E(context_->opName, "attr layOut is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->blockSize == nullptr, OPS_LOG_E(context_->opName, "attr blockSize is nullptr"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::CheckInputParameterFormat()
{
    auto qFormat = context_->query.desc->GetOriginFormat();
    auto kFormat = context_->key.desc->GetOriginFormat();
    auto vFormat = context_->value.desc->GetOriginFormat();

    OPS_ERR_IF((qFormat != ge::FORMAT_ND && qFormat != ge::FORMAT_NCHW && qFormat != ge::FORMAT_NHWC),
               OPS_LOG_E(context_->opName, "Query format %d should be ND/NCHW/NHWC", qFormat), return ge::GRAPH_FAILED);
    OPS_ERR_IF((kFormat != ge::FORMAT_ND && kFormat != ge::FORMAT_NCHW && kFormat != ge::FORMAT_NHWC),
               OPS_LOG_E(context_->opName, "Key format %d should be ND/NCHW/NHWC", kFormat), return ge::GRAPH_FAILED);
    OPS_ERR_IF((vFormat != ge::FORMAT_ND && vFormat != ge::FORMAT_NCHW && vFormat != ge::FORMAT_NHWC),
               OPS_LOG_E(context_->opName, "Value format %d should be ND/NCHW/NHWC", vFormat), return ge::GRAPH_FAILED);
  if(context_->attenMask.desc != nullptr){
    auto mFormat = context_->attenMask.desc->GetOriginFormat();
    OPS_ERR_IF((mFormat != ge::FORMAT_ND && mFormat != ge::FORMAT_NCHW && mFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "atten_mask format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->kvPaddingSize.desc != nullptr){
    auto kvPaddingFormat = context_->kvPaddingSize.desc->GetOriginFormat();
    OPS_ERR_IF((kvPaddingFormat != ge::FORMAT_ND && kvPaddingFormat != ge::FORMAT_NCHW && kvPaddingFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "padding_size format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->keySharedPrefix.desc != nullptr){
    auto kPrefixFormat = context_->keySharedPrefix.desc->GetOriginFormat();
    OPS_ERR_IF((kPrefixFormat != ge::FORMAT_ND && kPrefixFormat != ge::FORMAT_NCHW && kPrefixFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "k_prefix format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->valueSharedPrefix.desc != nullptr){
    auto vPrefixFormat = context_->valueSharedPrefix.desc->GetOriginFormat();
    OPS_ERR_IF((vPrefixFormat != ge::FORMAT_ND && vPrefixFormat != ge::FORMAT_NCHW && vPrefixFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "v_prefix format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::CheckInputAntiquantFormat() {
  if(context_->antiquantScale.desc != nullptr){
    auto aScaleFormat = context_->antiquantScale.desc->GetOriginFormat();
    OPS_ERR_IF((aScaleFormat != ge::FORMAT_ND && aScaleFormat != ge::FORMAT_NCHW && aScaleFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "antiquantScale format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->antiquantOffset.desc != nullptr){
    auto aOffsetFormat = context_->antiquantOffset.desc->GetOriginFormat();
    OPS_ERR_IF((aOffsetFormat != ge::FORMAT_ND && aOffsetFormat != ge::FORMAT_NCHW && aOffsetFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "antiquantOffset format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->keyAntiquantScale.desc != nullptr){
  auto kScaleFormat = context_->keyAntiquantScale.desc->GetOriginFormat();
    OPS_ERR_IF((kScaleFormat != ge::FORMAT_ND && kScaleFormat != ge::FORMAT_NCHW && kScaleFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "keyAntiquantScale format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->keyAntiquantOffset.desc != nullptr){
  auto kOffsetFormat = context_->keyAntiquantOffset.desc->GetOriginFormat();
    OPS_ERR_IF((kOffsetFormat != ge::FORMAT_ND && kOffsetFormat != ge::FORMAT_NCHW && kOffsetFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "keyAntiquantOffset format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->valueAntiquantScale.desc != nullptr){
  auto vScaleFormat = context_->valueAntiquantScale.desc->GetOriginFormat();
    OPS_ERR_IF((vScaleFormat != ge::FORMAT_ND && vScaleFormat != ge::FORMAT_NCHW && vScaleFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "valueAntiquantScale format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->valueAntiquantOffset.desc != nullptr){
  auto vOffsetFormat = context_->valueAntiquantOffset.desc->GetOriginFormat();
    OPS_ERR_IF((vOffsetFormat != ge::FORMAT_ND && vOffsetFormat != ge::FORMAT_NCHW && vOffsetFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "valueAntiquantOffset format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::CheckInputFormatAndLimits() {  
  if(CheckInputParameterFormat() != ge::GRAPH_SUCCESS || CheckInputAntiquantFormat() != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
  }
    OPS_ERR_IF(
        (nNumOfQInOneGroup_ > 64),
        OPS_LOG_E(context_->opName, "numHeads_ / numKvHeads_ = %u, cannot be greater than 64", nNumOfQInOneGroup_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((inputQType_ == ge::DT_INT8 && inputKvType_ == ge::DT_INT8),
               OPS_LOG_E(context_->opName, "IFA not support qkv datatype all int8."), return ge::GRAPH_FAILED);

    OPS_ERR_IF(((inputQType_ == ge::DT_FLOAT16) && (inputKvType_ != ge::DT_FLOAT16 && inputKvType_ != ge::DT_INT8 && inputKvType_ != ge::DT_INT4)),
               OPS_LOG_E(context_->opName, "when input Q type is fp16, KV type %d should be fp16 or int8 or int4", inputKvType_),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(((inputQType_ == ge::DT_BF16) && (inputKvType_ != ge::DT_BF16 && inputKvType_ != ge::DT_INT8 && inputKvType_ != ge::DT_INT4)),
               OPS_LOG_E(context_->opName, "when input Q type is bf16, KV type %d should be bf16 or int8 or int4", inputKvType_),
               return ge::GRAPH_FAILED);

    if (pageAttentionFlag_) {
        OPS_ERR_IF(
            (inputKvType_ == ge::DT_FLOAT16 || inputKvType_ == ge::DT_BF16) && (blockSize_ % 16 != 0),
            OPS_LOG_E(context_->opName, "blockSize=%u, it need align to 16 when kv dtype is fp16/bf16.", blockSize_),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF((inputKvType_ == ge::DT_INT8) && (blockSize_ % 32 != 0),
                   OPS_LOG_E(context_->opName, "blockSize=%u, it need align to 32 when kv dtype is int8.", blockSize_),
                   return ge::GRAPH_FAILED);
    }

    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        OPS_ERR_IF((numHeads_ != numKvHeads_), // unsupport gqa
                   OPS_LOG_E(context_->opName, "numHeads:%u of key must be equal to numHeads:%u of kv when 310P.",
                             numHeads_, numKvHeads_),
                   return ge::GRAPH_FAILED);

        OPS_ERR_IF((batchSize_ > 256),
                   OPS_LOG_E(context_->opName, "batch size:%u cannot be greater than 256 when 310P.", batchSize_),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF((sMax_ > 65536),
                   OPS_LOG_E(context_->opName, "sMax:%u cannot be greater than 65536 when 310P.", sMax_),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF((headDim_ % 16 != 0), OPS_LOG_E(context_->opName, "in 310P, headDim:%u need align to 16.", headDim_),
                   return ge::GRAPH_FAILED);

        OPS_ERR_IF((antiQuantFlag_ && (headDim_ % 32 != 0)),
                   OPS_LOG_E(context_->opName, "in 310P, headDim:%u need align to 32 when kv dtype is int8.", headDim_),
                   return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF((batchSize_ > 65536),
                   OPS_LOG_E(context_->opName, "batch size:%u cannot be greater than 65536.", batchSize_),
                   return ge::GRAPH_FAILED);
    }

    OPS_ERR_IF((headDim_ > 512), OPS_LOG_E(context_->opName, "headDim:%u cannot be greater than 512.", headDim_),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF((numKvHeads_ > 256),
               OPS_LOG_E(context_->opName, "numHead of key and value:%u cannot be greater than 256.", numKvHeads_),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::CheckKVHeadNum(const gert::StorageShape *inputShape)
{
    uint32_t tmpNumHeads = 0;
    std::string layOutStr = context_->layOut;
    if (layOutStr == "BSH") {
        auto H = inputShape->GetStorageShape().GetDim(2);
        tmpNumHeads = H / headDim_;
    } else if (layOutStr == "BNSD") {
        tmpNumHeads = inputShape->GetStorageShape().GetDim(1);
    } else if (layOutStr == "BSND") {
        tmpNumHeads = inputShape->GetStorageShape().GetDim(2);
    }
    OPS_ERR_IF(tmpNumHeads != numKvHeads_,
               OPS_LOG_E(context_->opName, "IFA check input param failed, tensor in list head num(%u) should be %u.",
                         tmpNumHeads, numKvHeads_),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::CheckKVShape(const size_t &size, const gert::StorageShape *keyTensorInList, const gert::StorageShape *valueTensorInList)
{
    /* kv not continuous */
    std::string layOutStr = context_->layOut;
    if (layOutStr == "BSH") {
        OPS_ERR_IF((keyTensorInList->GetStorageShape().GetDimNum() != DIM_BSH) ||
                        (valueTensorInList->GetStorageShape().GetDimNum() != DIM_BSH),
                    OPS_LOG_E(context_->opName,
                                "IFA check input param failed, tensor in list dim num should be 3, k: %lu, v: %lu.",
                                keyTensorInList->GetStorageShape().GetDimNum(),
                                valueTensorInList->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    }
    if ((layOutStr == "BNSD") || (layOutStr == "BSND")) {
        OPS_ERR_IF((keyTensorInList->GetStorageShape().GetDimNum() != DIM_BNSD_OR_BNSD) ||
                        (valueTensorInList->GetStorageShape().GetDimNum() != DIM_BNSD_OR_BNSD),
                    OPS_LOG_E(context_->opName,
                                "IFA check input param failed, tensor in list dim num should be 4, k: %lu, v: %lu.",
                                keyTensorInList->GetStorageShape().GetDimNum(),
                                valueTensorInList->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    }
    OPS_ERR_IF(
        keyTensorInList->GetStorageShape().GetDim(0) != 1,
        OPS_LOG_E(
            context_->opName,
            "IFA check input param failed, b of tensor in tensor list should be 1, now b is: %ld, list index: %lu",
            keyTensorInList->GetStorageShape().GetDim(0), size),
        return ge::GRAPH_FAILED);
    if (CheckKVHeadNum(keyTensorInList) != ge::GRAPH_SUCCESS ||
        CheckKVHeadNum(valueTensorInList) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::CheckQKOutShape()
{
    if (pageAttentionFlag_) { // page_attention don't check this place
        return ge::GRAPH_SUCCESS;
    }
    // queryShape (b, 1, h)
    const gert::StorageShape *queryShape = context_->query.shape;
    const gert::StorageShape *keyShape = context_->kCache[0];
    const std::string inputLayoutStr = context_->layOut;

    auto dimOfQ = queryShape->GetStorageShape().GetDimNum();
    auto dimOfK = keyShape->GetStorageShape().GetDimNum();
    auto dimOfOut = context_->attenOut.shape->GetStorageShape().GetDimNum();
    if (inputLayoutStr == "BSH") {
        OPS_ERR_IF(
            (dimOfQ != DIM_BSH) || (dimOfK != DIM_BSH) || (dimOfOut != DIM_BSH),
            OPS_LOG_E("[IFA]",
                      "When input layout is BSH, the dimension should be 3, dimOfQ: %lu, dimOfK: %lu, dimOfOut: %lu",
                      dimOfQ, dimOfK, dimOfOut),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(queryShape->GetStorageShape().GetDim(1) != 1,
                   OPS_LOG_E("[IFA]", "When input layout is BSH, the 2nd dimOfQ should be 1,the 2nd dimOfQ: %ld",
                             queryShape->GetStorageShape().GetDim(1)),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(queryShape->GetStorageShape().GetDim(2) / numHeads_ !=
                       keyShape->GetStorageShape().GetDim(2) / numKvHeads_,
                   OPS_LOG_E("[IFA]","When input layout is BSH,"
                             "the 3rd dimOfQ/numHeads(%ld) should be equal to the 3rd dimOfK/numKvHeads(%ld)",
                             queryShape->GetStorageShape().GetDim(2) / numHeads_,
                             keyShape->GetStorageShape().GetDim(2) / numKvHeads_),
                   return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF(
            (dimOfQ != DIM_BNSD_OR_BNSD) || (dimOfK != DIM_BNSD_OR_BNSD) || (dimOfOut != DIM_BNSD_OR_BNSD),
            OPS_LOG_E("[IFA]",
                      "When input layout is BNSD/BSND, the dim should be 4, 4th dimOfQ: %lu, 4th dimOfK: %lu, fourth dimOfOut: %lu",
                      dimOfQ, dimOfK, dimOfOut),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            queryShape->GetStorageShape().GetDim(3) != keyShape->GetStorageShape().GetDim(3),
            OPS_LOG_E(
                "[IFA]",
                "When input layout is BNSD/BSND, the 4th dimOfQ not be equal the 4th dimOfK, dimOfQ: %ld, dimOfK: %ld",
                queryShape->GetStorageShape().GetDim(3), keyShape->GetStorageShape().GetDim(3)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::CheckKeyShapeTensor(const gert::Shape &aShape)
{
    auto firstKeyShape = context_->kCache[0];
    std::string layOutStr = context_->layOut;
    for (size_t idx = 0; idx < aShape.GetDimNum(); idx++) {
        if (((layOutStr == "BNSD") && (idx == 2)) || // BNSD s index is 2
            ((layOutStr == "BSND") && (idx == 1)) || // BSND s index is 1
            ((layOutStr == "BSH") && (idx == 1))) {  // BSH s index is 1
            continue;                                // s can be different
        }
        OPS_ERR_IF(firstKeyShape->GetStorageShape().GetDim(idx) != aShape.GetDim(idx),
                   OPS_LOG_E(context_->opName,
                             "IFA check input param failed, tensor in keyShape except S must be same, index:[%lu] is "
                             "not same, k0: %ld, k: %ld",
                             idx, firstKeyShape->GetStorageShape().GetDim(idx), aShape.GetDim(idx)),
                   return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool SparseFusionIFATiling::CheckIfRollBack()
{
    if (sMax_ == 0) {
        return false; // 空tensor由新模板处理
    }

    if (socVersion_ != IfaSocVersion::SOC_ASCEND_310P) {
        // 1 page attention
        if (context_->blockTable.tensor != nullptr) {
            return false;
        }
    }

    // 2 qkv_quant
    if (inputQType_ == ge::DT_INT8) {
        return true;
    }

    // 4 D>=1024
    if (headDim_ >= 1024) {
        return true;
    }

    if (CanChangeToNew()) {
        return false;
    }

    return true;
}

bool SparseFusionIFATiling::ShapeEqual(const gert::Shape &aShape, const gert::Shape &bShape)
{
    if (aShape.GetDimNum() != bShape.GetDimNum()) {
        return false;
    }

    for (size_t idx = 0; idx < aShape.GetDimNum(); idx++) {
        if (aShape.GetDim(idx) != bShape.GetDim(idx)) {
            return false;
        }
    }

    return true;
}

bool SparseFusionIFATiling::CanChangeToNew()
{
    if (inOutMode_ == TilingInOutMode::BF16_BF16) {
        return true;
    }
    if (inOutMode_ == TilingInOutMode::BF16_INT8) {
        return true;
    }

    if (inOutMode_ == TilingInOutMode::FP16_FP16 || inOutMode_ == TilingInOutMode::FP16_INT8) {
        return true;
    }
    return false;
}

ge::graphStatus SparseFusionIFATiling::CheckQuant2Shape(const gert::Shape &inputParaShape)
{
    auto headsize = headDim_; // D
    auto headnum = numHeads_; // Q's N
    gert::Shape expectParamShapeBNSD = gert::Shape({1, headnum, 1, headsize});
    gert::Shape expectParamShapeBNSD_2 = gert::Shape({headnum, 1, headsize});
    gert::Shape expectParamShapeBNSD_3 = gert::Shape({headnum, headsize});
    gert::Shape expectParamShapeBSND = gert::Shape({1, 1, headnum, headsize});
    gert::Shape expectParamShapeBSND_2 = gert::Shape({1, headnum, headsize});
    gert::Shape expectParamShapeBSND_3 = gert::Shape({headnum, headsize});
    gert::Shape expectParamShapeBH = gert::Shape({1, headnum * headsize});
    gert::Shape expectParamShapeBH_2 = gert::Shape({1, 1, headnum * headsize});
    gert::Shape expectParamShapeBH_3 = gert::Shape({headnum * headsize});

    bool validShape = (inputParaShape == expectParamShapeBNSD) || (inputParaShape == expectParamShapeBSND) ||
                      (inputParaShape == expectParamShapeBH) || (inputParaShape == expectParamShapeBNSD_2) ||
                      (inputParaShape == expectParamShapeBSND_2) || (inputParaShape == expectParamShapeBH_2) ||
                      (inputParaShape == expectParamShapeBNSD_3) || (inputParaShape == expectParamShapeBSND_3) ||
                      (inputParaShape == expectParamShapeBH_3);

    if (!validShape && inputParaShape.GetDimNum() == DIM_BNSD) {
        OPS_LOG_E(context_->opName,
                  "The shape of postquant parameter[%ld, %ld, %ld, %ld] is not expected shape."
                  "Expect [1, %u, 1, %u] or [1, 1, %u, %u]",
                  inputParaShape.GetDim(BNSD_B_IDX), inputParaShape.GetDim(BNSD_N_IDX),
                  inputParaShape.GetDim(BNSD_S_IDX), inputParaShape.GetDim(BNSD_D_IDX), headnum, headsize, headnum,
                  headsize);
        return ge::GRAPH_FAILED;
    }

    if (!validShape && inputParaShape.GetDimNum() == 3) { // dim is 3
        OPS_LOG_E(context_->opName,
                  "The shape of postquant parameter[%ld, %ld, %ld] is not expected shape."
                  "Expect [%u, 1, %u], [1, %u, %u] or [1, 1, %u].",
                  inputParaShape.GetDim(BNSD_B_IDX), inputParaShape.GetDim(BNSD_N_IDX),
                  inputParaShape.GetDim(BNSD_S_IDX), headnum, headsize, headnum, headsize, headnum * headsize);
        return ge::GRAPH_FAILED;
    }

    if (!validShape && inputParaShape.GetDimNum() == DIM_BH) {
        OPS_LOG_E(context_->opName, "The shape of postquant parameter[%ld, %ld] is not expected[1, %u] or [%u, %u].",
                  inputParaShape.GetDim(BH_B_IDX), inputParaShape.GetDim(BH_H_IDX), headnum * headsize, headnum,
                  headsize);
        return ge::GRAPH_FAILED;
    }

    if (!validShape && inputParaShape.GetDimNum() == 1) {
        OPS_LOG_E(context_->opName, "The shape of postquant parameter[%ld] is not expected[%u].",
                  inputParaShape.GetDim(BH_B_IDX), headnum * headsize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::CheckKVAntiQuantPerHead(const gert::Shape &inputParaShape)
{
    if (antiquantMode_ == PER_TOKEN_MODE) { // per-token head
        OPS_ERR_IF((inputParaShape.GetDimNum() != 3), // 3: Dim of BGS is 3
                   OPS_LOG_E(context_->opName, "The dim of antiquant should be 3 instead of the current %lu",
                             inputParaShape.GetDimNum()),
                             return ge::GRAPH_FAILED);
        OPS_ERR_IF((inputParaShape.GetDim(0) != batchSize_),
                   OPS_LOG_E(context_->opName, "The 1st dim of antiquant should be %u instead of the current %ld",
                             batchSize_, inputParaShape.GetDim(0)),
                             return ge::GRAPH_FAILED);
        OPS_ERR_IF((inputParaShape.GetDim(1) != numKvHeads_),
                   OPS_LOG_E(context_->opName, "The 2nd dim of antiquant should be %u instead of the current %ld",
                             numKvHeads_, inputParaShape.GetDim(1)),
                             return ge::GRAPH_FAILED);
        OPS_ERR_IF((inputParaShape.GetDim(2) < seqSize_),
                   OPS_LOG_E(context_->opName, "The 3rd dim of antiquant should bigger than or equal to %u instead of the current %ld",
                             seqSize_, inputParaShape.GetDim(2)), // 2 : BGS S index is 2
                   return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    } else { // per-tensor head
        gert::Shape expectParamShape = gert::Shape({numKvHeads_});
        OPS_ERR_IF((inputParaShape != expectParamShape),
                   OPS_LOG_E(context_->opName,
                             "The shape of antiquant parameter[%ld] is not expected. Expect[%u] When per_tensor_head mode.",
                             inputParaShape.GetDim(0), numKvHeads_),
               return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus SparseFusionIFATiling::CheckKVAntiQuantPerChannel(const gert::Shape& inputParaShape) {
  std::string layOutStr = context_->layOut;
  gert::Shape expectParamShapeBNSD = gert::Shape({antiquantNum_, numKvHeads_, 1, headDim_});
  gert::Shape expectParamShapeBSNDType1 = gert::Shape({antiquantNum_, 1, numKvHeads_, headDim_});
  gert::Shape expectParamShapeBSNDType2 = gert::Shape({antiquantNum_, numKvHeads_, headDim_});
  gert::Shape expectParamShapeBH = gert::Shape({antiquantNum_, numKvHeads_ * headDim_});
  bool validOffsetShape = (inputParaShape == expectParamShapeBNSD) || (inputParaShape == expectParamShapeBSNDType1) ||
                          (inputParaShape == expectParamShapeBSNDType2) || (inputParaShape == expectParamShapeBH);

  OPS_ERR_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BNSD),
    OPS_LOG_E(context_->opName, "The shape of antiquant parameter[%ld, %ld, %ld, %ld] is not expected. "
      "Expect [%u, %u, 1, %u] or [%u, %u, %u] or [%u, %u] when per_channel mode.", inputParaShape.GetDim(BNSD_B_IDX),
      inputParaShape.GetDim(BNSD_N_IDX), inputParaShape.GetDim(BNSD_S_IDX), inputParaShape.GetDim(BNSD_D_IDX),
      antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_ * headDim_),
      return ge::GRAPH_FAILED);
  OPS_ERR_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BSND),
    OPS_LOG_E(context_->opName, "The shape of antiquant parameter[%ld, %ld, %ld] is not expected. "
      "Expect [%u, %u, 1, %u] or [%u, %u, %u] or [%u, %u] when per_channel mode.", inputParaShape.GetDim(BND_B_IDX),
      inputParaShape.GetDim(BND_N_IDX), inputParaShape.GetDim(BND_D_IDX), antiquantNum_, numKvHeads_, headDim_,
      antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_ * headDim_),
      return ge::GRAPH_FAILED);
  OPS_ERR_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_BH),
    OPS_LOG_E(context_->opName, "The shape of antiquant parameter[%ld, %ld] is not expected. "
      "Expect [%u, %u, 1, %u] or [%u, %u, %u] or [%u, %u] when per_channel mode.", inputParaShape.GetDim(BH_B_IDX),
      inputParaShape.GetDim(BH_H_IDX), antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_, headDim_,
      antiquantNum_, numKvHeads_ * headDim_),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::CheckAntiQuantParam(const gert::Tensor *antiquantScaleTensor,
                                               const gert::Tensor *antiquantOffsetTensor,
                                               const gert::CompileTimeTensorDesc *antiquantScaleDesc,
                                               const gert::CompileTimeTensorDesc *antiquantOffsetDesc)
{
    OPS_ERR_IF((antiquantMode_ != 0) && (antiquantMode_ != 1) && (antiquantMode_ != 2), // unseparated antiquant need this
               OPS_LOG_E(context_->opName, "antiquantMode value:%u is invalid, it should be 0 or 1 or 2", antiquantMode_),
               return ge::GRAPH_FAILED);
    if (antiquantScaleTensor == nullptr) {
        OPS_LOG_E(context_->opName, "KV antiquant is enabled, but the input antiquant scale is NULL");
        return ge::GRAPH_FAILED;
    }
    if (antiquantOffsetTensor != nullptr &&
        antiquantScaleTensor->GetStorageShape().GetDimNum() != antiquantOffsetTensor->GetStorageShape().GetDimNum()) {
        OPS_LOG_E(context_->opName,
                  "KV antiquant is enabled, but antiquant params have different layouts[scale: %lu, offset: %lu].",
                  antiquantScaleTensor->GetStorageShape().GetDimNum(),
                  antiquantOffsetTensor->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    auto tmpAntiquantScale = antiquantScaleTensor->GetStorageShape();
    if (CheckKVAntiQuantParaShapeLegal(tmpAntiquantScale) == ge::GRAPH_FAILED) {
        OPS_LOG_E(context_->opName, "illegal shape of antiquant scale.");
        return ge::GRAPH_FAILED;
    }
    if (antiquantOffsetTensor != nullptr) {
        auto tmpAntiquantOffset = antiquantOffsetTensor->GetStorageShape();
        if (CheckKVAntiQuantParaShapeLegal(tmpAntiquantOffset) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }

    ge::DataType antiquantScaleType = antiquantScaleDesc->GetDataType();
    if (antiquantMode_ == DEQUANT_PER_CHANNEL_MODE) { // per-tensor and per-channel
        if (antiquantScaleType != inputQType_) {
            OPS_LOG_E(context_->opName, "illegal datatype of antiquant scale, it should be same with input qtype");
            return ge::GRAPH_FAILED;
        }
    }
    if (antiquantMode_ == DEQUANT_PER_TOKEN_MODE) {
        if (antiquantScaleType != ge::DT_FLOAT) {
            OPS_LOG_E(context_->opName, "per-token mode is enabled, datatype of antiquant scale should be float32 ");
            return ge::GRAPH_FAILED;
        }
    }

    if (antiquantOffsetTensor != nullptr && antiquantOffsetDesc != nullptr) {
        ge::DataType antiquantOffsetType = antiquantOffsetDesc->GetDataType();
        if (antiquantScaleType != antiquantOffsetType) {
            OPS_LOG_E(context_->opName, "datatype of antiquant scale and antiquant offset should be the same");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::CheckSupportKVLeftPadding()
{
    if (inputKvType_ == ge::DT_INT4) {
        OPS_LOG_E(context_->opName, "When input Kv Dtypes is INT4 or INT32, KvLeftPadding is not supported currently.");
        return ge::GRAPH_FAILED;
    }
    if (!batchContinuousFlag_ || !actualSeqLenFlag_ || pageAttentionFlag_) {
        OPS_LOG_D(context_->opName, "KVLeftPadding illegal condition:  \
      pagedAttention scene: %d, not isBatchContinues: %d, actualSeqLen not exist: %d.",
                  pageAttentionFlag_, !batchContinuousFlag_, !actualSeqLenFlag_);
        return ge::GRAPH_SUCCESS;
    }
    kvPaddingSizeFlag_ = true;
    OPS_LOG_D(context_->opName, "KVLeftPadding starts to be used.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::SharedPrefixCheckBasic()
{
    OPS_ERR_IF(context_->keySharedPrefix.tensor == nullptr,
               OPS_LOG_E(context_->opName, "tensor  of key_shared_prefix is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->keySharedPrefix.desc == nullptr,
               OPS_LOG_E(context_->opName, "desc  of key_shared_prefix is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->valueSharedPrefix.tensor == nullptr,
               OPS_LOG_E(context_->opName, "tensor of value_shared_prefix is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->valueSharedPrefix.desc == nullptr,
               OPS_LOG_E(context_->opName, "desc  of value_shared_prefix is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->keySharedPrefix.desc->GetDataType() != inputKvType_,
               OPS_LOG_E(context_->opName, "type of key_shared_prefix not equal to type of KV"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->valueSharedPrefix.desc->GetDataType() != inputKvType_,
               OPS_LOG_E(context_->opName, "type of value_shared_prefix not equal to type of KV"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(pageAttentionFlag_, OPS_LOG_E(context_->opName, "shared prefix with page attention is not supported"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(kvPaddingSizeFlag_, OPS_LOG_E(context_->opName, "shared prefix with kv padding is not supported"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(socVersion_ == IfaSocVersion::SOC_ASCEND_310P,
               OPS_LOG_E(context_->opName, "shared prefix is not supported on 310p"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::SharedPrefixCheckShapes(const gert::Shape &keyShape, const gert::Shape &valueShape)
{
    OPS_ERR_IF(!ShapeEqual(keyShape, valueShape),
               OPS_LOG_E(context_->opName, "tensor shape of key_shared_prefix and value_shared_prefix not equal."),
               return ge::GRAPH_FAILED);

    // OPS_ERR_IF(keyShape.GetDimNum() != context_->query.shape->GetStorageShape().GetDimNum(),
    //            OPS_LOG_E(context_->opName, "tensor shape dim of key_shared_prefix[%lu] is not compatable with query",
    //                      keyShape.GetDimNum()),
    //            return ge::GRAPH_FAILED);

    OPS_ERR_IF(keyShape.GetDim(0) != 1,
               OPS_LOG_E(context_->opName, "batch of key_shared_prefix[%ld] must be 1", keyShape.GetDim(0)),
               return ge::GRAPH_FAILED);

    if (inputLayout_ == IfaLayout::BSH_BSND) {
        OPS_ERR_IF(
            keyShape.GetDimNum() == 3 && keyShape.GetDim(2) != numKvHeads_ * headDim_,
            OPS_LOG_E(context_->opName, "H of key_shared_prefix[%lu] is not equal to H of key", keyShape.GetDimNum()),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            keyShape.GetDimNum() == 4 && keyShape.GetDim(2) != numKvHeads_,
            OPS_LOG_E(context_->opName, "N of key_shared_prefix[%lu] is not equal to N of key", keyShape.GetDimNum()),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            keyShape.GetDimNum() == 4 && keyShape.GetDim(3) != headDim_,
            OPS_LOG_E(context_->opName, "D of key_shared_prefix[%lu] is not equal to D of key", keyShape.GetDimNum()),
            return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF(
            keyShape.GetDim(1) != numKvHeads_,
            OPS_LOG_E(context_->opName, "N of key_shared_prefix[%ld] is not equal to N of key", keyShape.GetDim(1)),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            keyShape.GetDim(3) != headDim_,
            OPS_LOG_E(context_->opName, "D of key_shared_prefix[%ld] is not equal to D of key", keyShape.GetDim(3)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::CheckUbSpace()
{
    if (!CalcUbBmm() || !CalcUbSoftMax() || !CalcUbAttenMask()) {
        return false;
    }
    return true;
}

bool SparseFusionIFATiling::IsFlashDecode() const
{
    // if (pageAttentionFlag_ && socVersion_ == IfaSocVersion::SOC_ASCEND_910B) {
    //     return false;
    // }

    float flashDecodeBNRatio = static_cast<float>(0.4); // 0.4, 经验值
    if (perfMode_ == IfaPerfMode::BMM_ALL_BY_VEC) {
        flashDecodeBNRatio = 0.5; // 0.5, 全V模板可以按0.5切分
    }

    if ((batchSize_ * numKvHeads_ < flashDecodeBNRatio * coreNum_) && (nNumOfQInOneGroup_ == 1)) {
        OPS_LOG_D(context_->opName, "Flash Decode Split kv."); // 非gqa时这里只判断bn是否满足
        return true;
    }

    if ((batchSize_ * numKvHeads_ < flashDecodeBNRatio * coreNum_) &&
        (maxActualseq_ >= 2048)) { // 2048, 在flash decode + gqa时的经验值
        OPS_LOG_D(context_->opName, "Flash Decode And GQA Split kv.");
        return true;
    }
    return false;
}

bool SparseFusionIFATiling::EnableAllVec()
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return true;
    }
    // // 暂时不考虑以下因素
    if (pageAttentionFlag_) {// 如果使用page attention，不开启全VEC
        return false;
    }
    if (sysPrefixFlag_) {// 如果使用sys prefix，不开启全VEC
        return false;
    }
    if (nNumOfQInOneGroup_ > 1) {// 如果N/Q不等于1，不开启全VEC
        return false;
    }
    if (headDim_ > 512) { // 全VEC模板仅支持headDim_不大于512
        return false;
    }
    return (inputQType_ == ge::DT_FLOAT16) && (inputKvType_ == ge::DT_FLOAT16) && (outputType_ == ge::DT_FLOAT16);
}

bool SparseFusionIFATiling::EnableC1V1()
{
    if (splitKVFlag_) {
        return false;
    }
    if (sysPrefixFlag_) {
        return false;
    }
    // 2:核数不超过vector总核数一半，可以按1:1启动cube和vector
    return (perfMode_ == IfaPerfMode::NORMAL) && (batchSize_ * numKvHeads_ * 2 <= aivNum_);
}

ge::graphStatus SparseFusionIFATiling::ProcessPseShift()
{
    // get pse shift data
    auto pseShiftInput = context_->pseShift.tensor;
    if (pseShiftInput == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OPS_ERR_IF(context_->pseShift.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of pse shift tensor is null."),
               return ge::GRAPH_FAILED);

    auto pseShiftDataType = context_->pseShift.desc->GetDataType();
    if (pseShiftDataType != ge::DT_FLOAT16 && pseShiftDataType != DT_BF16) {
        OPS_LOG_E(context_->opName, "Data type of pse shift is %s, which is not supported.",
                  SparseFusionDataTypeToSerialString(pseShiftDataType).c_str());
        return ge::GRAPH_FAILED;
    }

    switch (pseShiftDataType) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            OPS_ERR_IF((inputQType_ != ge::DT_INT8) && (inputQType_ != pseShiftDataType),
                       OPS_LOG_E(context_->opName,
                                 "Data type of pse is %s, which does not match data type of query: %s.",
                                 SparseFusionDataTypeToSerialString(pseShiftDataType).c_str(),
                                 SparseFusionDataTypeToSerialString(inputQType_).c_str()),
                       return ge::GRAPH_FAILED);
            break;
        default:
            OPS_LOG_E(context_->opName, "Data type of pse %s is not currently supported.",
                      SparseFusionDataTypeToSerialString(pseShiftDataType).c_str());
            return ge::GRAPH_FAILED;
    }

    // check pse shift shape (B/1, N, 1, Si)
    const gert::Shape pseShiftShape = pseShiftInput->GetStorageShape();
    uint32_t pseShiftDimNum = pseShiftShape.GetDimNum();
    OPS_ERR_IF(pseShiftDimNum != 4,
               OPS_LOG_E(context_->opName, "The input shape of pse shift must have 4 dims, current dim num is %u.",
                         pseShiftDimNum),
               return GRAPH_FAILED);
    pseShiftBatch_ = pseShiftShape.GetDim(PSE_SHIFT_B);
    uint32_t pseShiftN = pseShiftShape.GetDim(PSE_SHIFT_N);
    uint32_t pseShiftS0 = pseShiftShape.GetDim(PSE_SHIFT_S0);
    pseShiftS1_ = pseShiftShape.GetDim(PSE_SHIFT_S1);

    OPS_ERR_IF(
        (pseShiftBatch_ != 1 && pseShiftBatch_ != batchSize_) || (pseShiftN != numHeads_) || (pseShiftS0 != 1),
        OPS_LOG_E(context_->opName,
                  "The shape of pse shift is (%u, %u, %u, %u), which does not match (B, N, 1, S) or (1, N, 1, S).",
                  pseShiftBatch_, pseShiftN, pseShiftS0, pseShiftS1_),
        return ge::GRAPH_FAILED);

    if (pseShiftS1_ < seqSize_) {
        OPS_LOG_E(context_->opName,
                  "The shape of pse shift is (%u, %u, %u, %u), the 3rd dim S[%u] shouldn't be less than sMax[%u]."
                  "When Page Attention is enabled, sMax is maxBlockNumPerBatch * blockSize.",
                  pseShiftBatch_, pseShiftN, pseShiftS0, pseShiftS1_, pseShiftS1_, seqSize_);
        return GRAPH_FAILED;
    }

    // pse shift D is not 16 aligned
    OPS_ERR_IF(headDim_ % 16 != 0, OPS_LOG_E(context_->opName, "When Pse shift is enabled, D should be 16 aligned."),
               return ge::GRAPH_FAILED);

    pseShiftFlag_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFusionIFATiling::ProcessAttenMask()
{
    auto maskShape = context_->attenMask.tensor; // input shape = 4
    if (maskShape == nullptr) {
        attenMaskFlag_ = false;
        return ge::GRAPH_SUCCESS;
    }

    if (maskShape->GetStorageShape().GetShapeSize() == 0) {
        attenMaskFlag_ = false;
        OPS_LOG_W(context_->opName, "atten_mask tensor exist, but atten_mask shape size is 0.");
        return ge::GRAPH_SUCCESS;
    }

    uint32_t batchSizeOfMask = maskShape->GetStorageShape().GetDim(0);
    if (batchSizeOfMask != batchSize_) {
        OPS_LOG_E(context_->opName, "batchSize[%u] of atten_mask must be equal to batchSize[%u] of query.",
                  batchSizeOfMask, batchSize_);
        return ge::GRAPH_FAILED;
    }

    ge::DataType attenMaskType = context_->attenMask.desc->GetDataType();
    if (attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_INT8 && attenMaskType != ge::DT_UINT8) {
        OPS_LOG_E(context_->opName, "not support atten_mask type %d, only support bool, int8 and uint8.",
                  attenMaskType);
        return ge::GRAPH_FAILED;
    }

    auto dimNumOfMask = maskShape->GetStorageShape().GetDimNum();
    attenMaskSize_ = maskShape->GetStorageShape().GetDim(dimNumOfMask - 1);
    uint32_t minAttenMaskSize = pageAttentionFlag_ ? sMax_ : maxActualseq_;
    if (attenMaskSize_ < minAttenMaskSize) {
        OPS_LOG_E(context_->opName, "s Size[%u] of atten_mask must be greater than or equal to sMax[%u].",
                  attenMaskSize_, minAttenMaskSize);
        return ge::GRAPH_FAILED;
    }

    attenMaskFlag_ = true;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling