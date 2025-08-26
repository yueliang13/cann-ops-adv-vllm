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
 * \file fallback_ffn.cpp
 * \brief
 */

#include "fallback_comm.h"
#include "fallback.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;
constexpr size_t kffnInputX = 0;
constexpr size_t kffnInputWeight1 = 1;
constexpr size_t kffnInputWeight2 = 2;
constexpr size_t kffnInputExpertTokens = 3;
constexpr size_t kffnInputbias1 = 4;
constexpr size_t kffnInputbias2 = 5;
constexpr size_t kffnInputscale = 6;
constexpr size_t kffnInputoffset = 7;
constexpr size_t kffnInputdeqScale1 = 8;
constexpr size_t kffnInputdeqScale2 = 9;
constexpr size_t kffnInputantiquantScale1 = 10;
constexpr size_t kffnInputantiquantScale2 = 11;
constexpr size_t kffnInputantiquantOffset1 = 12;
constexpr size_t kffnInputantiquantOffset2 = 13;
constexpr size_t kffnOutput = 0;

inline static aclTensor *GeTensor2AclTensor(const gert::Tensor *geTensor, bool enableTranspose = false,
                                            bool enableNZ = false)
{
    if (geTensor == nullptr) {
        return nullptr;
    }
    auto gert_shape = geTensor->GetStorageShape();
    if (gert_shape.GetDimNum() <= 1) {
        return ConvertType(geTensor);
    }

    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    OPS_ERR_IF((aclCreateTensor == nullptr), OPS_LOG_E("aclnnfallback", "aclCreateTensor nullptr"), return nullptr);

    void *deviceAddr = const_cast<void *>(geTensor->GetAddr());

    // convert data type
    auto dataType_ge = geTensor->GetDataType();
    auto dataType = ToAclDataType(dataType_ge);
    // convert shape
    std::vector<int64_t> shape;
    for (size_t i = 0; i < gert_shape.GetDimNum(); ++i) {
        shape.push_back(gert_shape.GetDim(i));
    }
    // calculate tensor strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    auto origin_shape = geTensor->GetOriginShape();
    std::vector<int64_t> viewShape;
    for (size_t i = 0; i < origin_shape.GetDimNum(); ++i) {
        viewShape.push_back(origin_shape.GetDim(i));
    }

    // when tensor is transposed, last two dims in strides and viewShape should swap
    if (enableTranspose) {
        auto dimM = shape.size() - 2;
        auto dimN = shape.size() - 1;
        auto swap = strides[dimN];
        strides[dimN] = strides[dimM];
        strides[dimM] = swap;
        viewShape[dimN] = shape[dimM];
        viewShape[dimM] = shape[dimN];
    }
    auto aclFormat = aclFormat::ACL_FORMAT_ND;
    if (enableNZ && GetPrimaryFormat(geTensor->GetStorageFormat()) == ge::Format::FORMAT_FRACTAL_NZ) {
        aclFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    }
    aclTensor *out = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, strides.data(), 0, aclFormat,
                                     shape.data(), shape.size(), deviceAddr);
    OPS_ERR_IF((out == nullptr), OPS_LOG_E("aclnnfallback", "out nullptr"), return nullptr);

    return out;
}

static graphStatus FFNExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OPS_ERR_IF(host_api_ctx == nullptr, OPS_LOG_E("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    auto x_ge = host_api_ctx->GetInputTensor(kffnInputX);
    OPS_ERR_IF((x_ge == nullptr), OPS_LOG_E("aclnnfallback", "x_ge is null"), return GRAPH_FAILED);
    auto x_acl = GeTensor2AclTensor(x_ge, false, true);

    auto weight1_ge = host_api_ctx->GetInputTensor(kffnInputWeight1);
    OPS_ERR_IF((weight1_ge == nullptr), OPS_LOG_E("aclnnfallback", "weight1_ge is null"), return GRAPH_FAILED);
    auto weight1_acl = GeTensor2AclTensor(weight1_ge, false, true);

    auto weight2_ge = host_api_ctx->GetInputTensor(kffnInputWeight2);
    OPS_ERR_IF((weight2_ge == nullptr), OPS_LOG_E("aclnnfallback", "weight2_ge is null"), return GRAPH_FAILED);
    auto weight2_acl = GeTensor2AclTensor(weight2_ge, false, true);

    auto expert_tokens_ge = host_api_ctx->GetOptionalInputTensor(kffnInputExpertTokens);

    auto bias1_ge = host_api_ctx->GetOptionalInputTensor(kffnInputbias1);

    auto bias2_ge = host_api_ctx->GetOptionalInputTensor(kffnInputbias2);

    auto scale_ge = host_api_ctx->GetOptionalInputTensor(kffnInputscale);

    auto offset_ge = host_api_ctx->GetOptionalInputTensor(kffnInputoffset);

    auto deq_scale1_ge = host_api_ctx->GetOptionalInputTensor(kffnInputdeqScale1);

    auto deq_scale2_ge = host_api_ctx->GetOptionalInputTensor(kffnInputdeqScale2);

    auto antiquant_scale1_ge = host_api_ctx->GetOptionalInputTensor(kffnInputantiquantScale1);

    auto antiquant_scale2_ge = host_api_ctx->GetOptionalInputTensor(kffnInputantiquantScale2);

    auto antiquant_offset1_ge = host_api_ctx->GetOptionalInputTensor(kffnInputantiquantOffset1);

    auto antiquant_offset2_ge = host_api_ctx->GetOptionalInputTensor(kffnInputantiquantOffset2);

    auto output_ge = host_api_ctx->GetOutputTensor(kffnOutput);
    auto output_acl = GeTensor2AclTensor(output_ge, false, true);

    OPS_ERR_IF((output_ge == nullptr), OPS_LOG_E("aclnnfallback", "output_ge is null"), return GRAPH_FAILED);
    auto attrs = host_api_ctx->GetAttrs();
    OPS_ERR_IF((attrs == nullptr), OPS_LOG_E("aclnnfallback", "attrs is null"), return GRAPH_FAILED);
    const char *activation_type_ge = attrs->GetAttrPointer<char>(0);
    const int64_t *inner_pricise_ge = attrs->GetAttrPointer<int64_t>(1);
    const bool *tokens_index_flag_ge = attrs->GetAttrPointer<bool>(3);
    // execute opapi
    auto api_ret = EXEC_OPAPI_CMD(aclnnFFNV3, x_acl, weight1_acl, weight2_acl, expert_tokens_ge, bias1_ge, bias2_ge,
                                  scale_ge, offset_ge, deq_scale1_ge, deq_scale2_ge, antiquant_scale1_ge,
                                  antiquant_scale2_ge, antiquant_offset1_ge, antiquant_offset2_ge, activation_type_ge,
                                  *inner_pricise_ge, *tokens_index_flag_ge, output_acl);
    OPS_ERR_IF((api_ret != GRAPH_SUCCESS), OPS_LOG_E("aclnnfallback", "api_ret faild:%u", api_ret),
               return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(FFN).OpExecuteFunc(FFNExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
