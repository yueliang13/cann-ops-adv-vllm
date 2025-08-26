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

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;

static graphStatus MoeDistributeDispatchExecuteFunc(OpExecuteContext* host_api_ctx)
{
  OPS_LOG_D("start to fallback for moeDisbuteDispatch");
  OPS_CHECK(host_api_ctx == nullptr, OPS_LOG_E("aclnnFallback","host_api_ctx is null"), return ge::GRAPH_FAILED);
  const auto x = host_api_ctx->GetInputTensor(static_cast<size_t>(0));
  OPS_CHECK(x == nullptr, OPS_LOG_E("aclnnFallback","x is null"), return ge::GRAPH_FAILED);

  const auto expand_ids = host_api_ctx->GetInputTensor(static_cast<size_t>(1));
  OPS_CHECK(expand_ids == nullptr, OPS_LOG_E("aclnnFallback","expand_ids is null"), return ge::GRAPH_FAILED);

  const auto scales = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(2));
  const auto x_active_mask = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(3));
  const auto expert_scales = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(4));

  const auto expand_x = host_api_ctx->GetOutputTensor(static_cast<size_t>(0));
  OPS_CHECK(expand_x == nullptr, OPS_LOG_E("aclnnFallback","expand_x is null"), return ge::GRAPH_FAILED);

  const auto dynamic_scales = host_api_ctx->GetOutputTensor(static_cast<size_t>(1));
  OPS_CHECK(dynamic_scales == nullptr, OPS_LOG_E("aclnnFallback","dynamic_scales is null"), return ge::GRAPH_FAILED);

  const auto expand_idx = host_api_ctx->GetOutputTensor(static_cast<size_t>(2));
  OPS_CHECK(expand_idx == nullptr, OPS_LOG_E("aclnnFallback","expand_idx is null"), return ge::GRAPH_FAILED);

  const auto expert_token_nums = host_api_ctx->GetOutputTensor(static_cast<size_t>(3));
  OPS_CHECK(expert_token_nums == nullptr, OPS_LOG_E("aclnnFallback","expert_token_nums is null"), return ge::GRAPH_FAILED);

  const auto ep_recv_count = host_api_ctx->GetOutputTensor(static_cast<size_t>(4));
  OPS_CHECK(ep_recv_count == nullptr, OPS_LOG_E("aclnnFallback","ep_recv_count is null"), return ge::GRAPH_FAILED);

  const auto tp_recv_count = host_api_ctx->GetOutputTensor(static_cast<size_t>(5));
  OPS_CHECK(tp_recv_count == nullptr, OPS_LOG_E("aclnnFallback","tp_recv_count is null"), return ge::GRAPH_FAILED);

  const auto expand_scales = host_api_ctx->GetOutputTensor(static_cast<size_t>(6));
  OPS_CHECK(expand_scales == nullptr, OPS_LOG_E("aclnnFallback", "expand_scales is null"), return ge::GRAPH_FAILED);

  const auto attrs = host_api_ctx->GetAttrs();
  OPS_CHECK(attrs == nullptr, OPS_LOG_E("aclnnFallback","attrs is null"), return ge::GRAPH_FAILED);

  const auto *group_ep = attrs->GetStr(static_cast<size_t>(0));
  OPS_CHECK(group_ep == nullptr, OPS_LOG_E("aclnnFallback","group_ep is null"), return ge::GRAPH_FAILED);

  const auto *ep_world_size = attrs->GetInt(static_cast<size_t>(1));
  OPS_CHECK(ep_world_size == nullptr, OPS_LOG_E("aclnnFallback","ep_world_size is null"), return ge::GRAPH_FAILED);

  const auto *ep_rank_id = attrs->GetInt(static_cast<size_t>(2));
  OPS_CHECK(ep_rank_id == nullptr, OPS_LOG_E("aclnnFallback","ep_rank_id is null"), return ge::GRAPH_FAILED);

  const auto *moe_expert_num = attrs->GetInt(static_cast<size_t>(3));
  OPS_CHECK(moe_expert_num == nullptr, OPS_LOG_E("aclnnFallback","moe_expert_num is null"), return ge::GRAPH_FAILED);

  const auto *group_tp = attrs->GetStr(static_cast<size_t>(4));
  OPS_CHECK(group_tp == nullptr, OPS_LOG_E("aclnnFallback","group_tp is null"), return ge::GRAPH_FAILED);

  const auto *tp_world_size = attrs->GetInt(static_cast<size_t>(5));
  OPS_CHECK(tp_world_size == nullptr, OPS_LOG_E("aclnnFallback","tp_world_size is null"), return ge::GRAPH_FAILED);

  const auto *tp_rank_id = attrs->GetInt(static_cast<size_t>(6));
  OPS_CHECK(tp_rank_id == nullptr, OPS_LOG_E("aclnnFallback","tp_rank_id is null"), return ge::GRAPH_FAILED);

  const auto *expert_shard_type = attrs->GetInt(static_cast<size_t>(7));
  OPS_CHECK(expert_shard_type == nullptr, OPS_LOG_E("aclnnFallback","expert_shard_type is null"), return ge::GRAPH_FAILED);

  const auto *shared_expert_num = attrs->GetInt(static_cast<size_t>(8));
  OPS_CHECK(shared_expert_num == nullptr, OPS_LOG_E("aclnnFallback","shared_expert_num is null"), return ge::GRAPH_FAILED);

  const auto *shared_expert_rank_num = attrs->GetInt(static_cast<size_t>(9));
  OPS_CHECK(shared_expert_rank_num == nullptr, OPS_LOG_E("aclnnFallback","shared_expert_rank_num is null"), return ge::GRAPH_FAILED);

  const auto *quant_mode_ptr = attrs->GetInt(static_cast<size_t>(10));
  OPS_CHECK(quant_mode_ptr == nullptr, OPS_LOG_E("aclnnFallback","quant_mode_ptr is null"), return ge::GRAPH_FAILED);

  const int64_t *global_bs_ptr = attrs->GetInt(static_cast<size_t>(11));
  OPS_CHECK(global_bs_ptr == nullptr, OPS_LOG_E("aclnnFallback","global_bs_ptr is null"), return ge::GRAPH_FAILED);

  const int64_t *expert_token_nums_type_ptr = attrs->GetInt(static_cast<size_t>(12));
  OPS_CHECK(expert_token_nums_type_ptr == nullptr, OPS_LOG_E("aclnnFallback","expert_token_nums_type_ptr is null"), return ge::GRAPH_FAILED);

  const auto api_ret = EXEC_OPAPI_CMD(aclnnMoeDistributeDispatch, x, expand_ids, scales, x_active_mask, expert_scales, group_ep, *ep_world_size, *ep_rank_id,
                                      *moe_expert_num, group_tp, *tp_world_size, *tp_rank_id, *expert_shard_type, *shared_expert_num, *shared_expert_rank_num,
                                      *quant_mode_ptr, *global_bs_ptr, *expert_token_nums_type_ptr, expand_x, dynamic_scales, expand_idx, expert_token_nums, 
                                      ep_recv_count, tp_recv_count, expand_scales);
  OPS_CHECK(api_ret != ge::GRAPH_SUCCESS, OPS_LOG_E("aclnnFallback","aclnn api error code %u", api_ret),
           return ge::GRAPH_FAILED);
  return GRAPH_SUCCESS;
}

IMPL_OP(MoeDistributeDispatch).OpExecuteFunc(MoeDistributeDispatchExecuteFunc);

}  // namespace fallback

#ifdef __cplusplus
}
#endif
