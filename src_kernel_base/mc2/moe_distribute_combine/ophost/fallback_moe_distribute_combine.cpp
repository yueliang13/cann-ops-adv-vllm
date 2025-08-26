/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback_comm.h"
#include "fallback.h"
#include "log/ops_log.h"

namespace fallback {
using namespace ge;
using namespace gert;

static graphStatus MoeDistributeCombineExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OPS_LOG_D("start to fallback for moeDisbuteCombine");
    OPS_CHECK(host_api_ctx == nullptr, OPS_LOG_E("aclnnFallback", "host_api_ctx is null"), return ge::GRAPH_FAILED);
    const auto expand_x = host_api_ctx->GetInputTensor(static_cast<size_t>(0));
    OPS_CHECK(expand_x == nullptr, OPS_LOG_E("aclnnFallback", "expand_x is null"), return ge::GRAPH_FAILED);

    const auto expert_ids = host_api_ctx->GetInputTensor(static_cast<size_t>(1));
    OPS_CHECK(expert_ids == nullptr, OPS_LOG_E("aclnnFallback", "expert_ids is null"), return ge::GRAPH_FAILED);

    const auto expand_idx = host_api_ctx->GetInputTensor(static_cast<size_t>(2));
    OPS_CHECK(expand_idx == nullptr, OPS_LOG_E("aclnnFallback", "expand_idx is null"), return ge::GRAPH_FAILED);

    const auto ep_send_counts = host_api_ctx->GetInputTensor(static_cast<size_t>(3));
    OPS_CHECK(ep_send_counts == nullptr, OPS_LOG_E("aclnnFallback", "ep_send_counts is null"), return ge::GRAPH_FAILED);

    const auto expert_scales = host_api_ctx->GetInputTensor(static_cast<size_t>(4));
    OPS_CHECK(expert_scales == nullptr, OPS_LOG_E("aclnnFallback", "expert_scales is null"), return ge::GRAPH_FAILED);

    const auto tp_send_counts = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(5));
    OPS_CHECK(tp_send_counts == nullptr, OPS_LOG_E("aclnnFallback", "tp_send_counts is null"), return ge::GRAPH_FAILED);

    const auto x_active_mask = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(6));

    const auto activation_scale = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(7));

    const auto weight_scale = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(8));

    const auto group_list = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(9));

    const auto expand_scales = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(10));

    const auto y = host_api_ctx->GetOutputTensor(static_cast<size_t>(0));
    OPS_CHECK(y == nullptr, OPS_LOG_E("aclnnFallback", "y is null"), return ge::GRAPH_FAILED);

    const auto attrs = host_api_ctx->GetAttrs();
    OPS_CHECK(attrs == nullptr, OPS_LOG_E("aclnnFallback", "attrs is null"), return ge::GRAPH_FAILED);

    const auto *group_ep = attrs->GetStr(static_cast<size_t>(0));
    OPS_CHECK(group_ep == nullptr, OPS_LOG_E("aclnnFallback", "group_ep is null"), return ge::GRAPH_FAILED);

    const auto *ep_word_size = attrs->GetInt(static_cast<size_t>(1));
    OPS_CHECK(ep_word_size == nullptr, OPS_LOG_E("aclnnFallback", "ep_word_size is null"), return ge::GRAPH_FAILED);

    const auto *ep_rank_id = attrs->GetInt(static_cast<size_t>(2));
    OPS_CHECK(ep_rank_id == nullptr, OPS_LOG_E("aclnnFallback", "ep_rank_id is null"), return ge::GRAPH_FAILED);

    const auto *moe_expert_num = attrs->GetInt(static_cast<size_t>(3));
    OPS_CHECK(moe_expert_num == nullptr, OPS_LOG_E("aclnnFallback", "moe_expert_num is null"), return ge::GRAPH_FAILED);

    const auto *group_tp = attrs->GetStr(static_cast<size_t>(4));
    OPS_CHECK(group_tp == nullptr, OPS_LOG_E("aclnnFallback", "group_tp is null"), return ge::GRAPH_FAILED);

    const auto *tp_word_size = attrs->GetInt(static_cast<size_t>(5));
    OPS_CHECK(tp_word_size == nullptr, OPS_LOG_E("aclnnFallback", "tp_word_size is null"), return ge::GRAPH_FAILED);

    const auto *tp_rank_id = attrs->GetInt(static_cast<size_t>(6));
    OPS_CHECK(tp_rank_id == nullptr, OPS_LOG_E("aclnnFallback", "tp_rank_id is null"), return ge::GRAPH_FAILED);

    const auto *expert_shard_type = attrs->GetInt(static_cast<size_t>(7));
    OPS_CHECK(expert_shard_type == nullptr, OPS_LOG_E("aclnnFallback", "expert_shard_type is null"),
             return ge::GRAPH_FAILED);

    const auto *shared_expert_num = attrs->GetInt(static_cast<size_t>(8));
    OPS_CHECK(shared_expert_num == nullptr, OPS_LOG_E("aclnnFallback", "shared_expert_num is null"),
             return ge::GRAPH_FAILED);

    const auto *shared_expert_rank_num = attrs->GetInt(static_cast<size_t>(9));
    OPS_CHECK(shared_expert_rank_num == nullptr, OPS_LOG_E("aclnnFallback", "shared_expert_rank_num is null"),
             return ge::GRAPH_FAILED);

    const int64_t *global_bs_ptr = attrs->GetInt(static_cast<size_t>(10));
    OPS_CHECK(global_bs_ptr == nullptr, OPS_LOG_E("aclnnFallback", "global_bs_ptr is null"), return ge::GRAPH_FAILED);

    const int64_t *out_dtype_ptr = attrs->GetInt(static_cast<size_t>(11));
    OPS_CHECK(out_dtype_ptr == nullptr, OPS_LOG_E("aclnnFallback", "out_dtype is null"), return ge::GRAPH_FAILED);

    const int64_t *comm_quant_mode_ptr = attrs->GetInt(static_cast<size_t>(12));
    OPS_CHECK(comm_quant_mode_ptr == nullptr, OPS_LOG_E("aclnnFallback", "comm_quant_mode is null"),
             return ge::GRAPH_FAILED);

    const int64_t *group_list_type_ptr = attrs->GetInt(static_cast<size_t>(13));
    OPS_CHECK(group_list_type_ptr == nullptr, OPS_LOG_E("aclnnFallback", "group_list_type is null"),
             return ge::GRAPH_FAILED);

    const auto api_ret = EXEC_OPAPI_CMD(
        aclnnMoeDistributeCombine, expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, tp_send_counts,
        x_active_mask, activation_scale, weight_scale, group_list, expand_scales, group_ep, *ep_word_size, *ep_rank_id,
        *moe_expert_num, group_tp, *tp_word_size, *tp_rank_id, *expert_shard_type, *shared_expert_num,
        *shared_expert_rank_num, *global_bs_ptr, *out_dtype_ptr, *comm_quant_mode_ptr, *group_list_type_ptr, y);
    OPS_CHECK(api_ret != ge::GRAPH_SUCCESS, OPS_LOG_E("aclnnFallback", "aclnn api error code %u", api_ret),
             return api_ret);
    return GRAPH_SUCCESS;
}

IMPL_OP(MoeDistributeCombine).OpExecuteFunc(MoeDistributeCombineExecuteFunc);

} // namespace fallback
