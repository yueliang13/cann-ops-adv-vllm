/**
 * @file function.h
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef FUNCTION_H_
#define FUNCTION_H_


#include <ATen/ATen.h>
#include <string>
#include <vector>

// 声明与 add_custom.cpp 中完全一致的函数
at::Tensor incre_flash_attention_v4_impl_npu(const at::Tensor &query,
                                             const std::vector<at::Tensor> &key_list,
                                             const std::vector<at::Tensor> &value_list,
                                             const at::Tensor &pse_shift,
                                             const at::Tensor &attention_mask,
                                             const at::Tensor &actual_seq_lengths,
                                             const at::Tensor &dequant_scale1,
                                             const at::Tensor &quant_scale1,
                                             const at::Tensor &dequant_scale2,
                                             const at::Tensor &quant_scale2,
                                             const at::Tensor &quant_offset2,
                                             const at::Tensor &antiquant_scale,
                                             const at::Tensor &antiquant_offset,
                                             const at::Tensor &blocktable,
                                             const at::Tensor &kv_padding_size,
                                             int64_t num_heads,
                                             double scale_value,
                                             const std::string &input_layout,
                                             int64_t num_key_value_heads,
                                             int64_t block_size,
                                             int64_t inner_precise);
at::Tensor incre_flash_attention_v5_impl_npu(const at::Tensor &query, const std::vector<at::Tensor> &key_list,
                                             const std::vector<at::Tensor> &value_list, const at::Tensor &pse_shift,
                                             const at::Tensor &attention_mask, const at::Tensor &actual_seq_lengths,
                                             const at::Tensor &dequant_scale1, const at::Tensor &quant_scale1,
                                             const at::Tensor &dequant_scale2, const at::Tensor &quant_scale2,
                                             const at::Tensor &quant_offset2, const at::Tensor &antiquant_scale,
                                             const at::Tensor &antiquant_offset, const at::Tensor &blocktable,
                                             const at::Tensor &kv_padding_size, const at::Tensor &blockposition,
                                             int64_t num_heads, double scale_value, const std::string &input_layout,
                                             int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise);

#endif // FUNCTION_H_
