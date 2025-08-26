/**
 * @file registration.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <torch/library.h>
#include <torch/extension.h>
#include "function.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// // 在myops命名空间里注册add_custom和add_custom_backward两个schema，新增自定义aten ir需要在此注册
// TORCH_LIBRARY(myops, m) {
//     m.def("add_custom(Tensor self, Tensor other) -> Tensor");
//     m.def("add_custom_backward(Tensor self) -> (Tensor, Tensor)");
// }

// // 通过pybind将c++接口和python接口绑定
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("add_custom", &add_custom_autograd, "x + y");
// }

/**
 * @file registration.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/library.h>
#include <torch/extension.h>
#include "function.h"

// 使用与 C++ 函数签名完全匹配的 schema
TORCH_LIBRARY(myops, m)
{
    m.def("incre_flash_attention_v4(Tensor query, Tensor[] key_list, Tensor[] value_list, Tensor pse_shift, Tensor attention_mask, Tensor actual_seq_lengths, Tensor dequant_scale1, Tensor quant_scale1, Tensor dequant_scale2, Tensor quant_scale2, Tensor quant_offset2, Tensor antiquant_scale, Tensor antiquant_offset, Tensor blocktable, Tensor kv_padding_size, int num_heads, float scale_value, str input_layout, int num_key_value_heads, int block_size, int inner_precise) -> Tensor");
    m.def("incre_flash_attention_v5(Tensor query, Tensor[] key_list, Tensor[] value_list, Tensor pse_shift, Tensor attention_mask,Tensor actual_seq_lengths, Tensor dequant_scale1, Tensor quant_scale1, Tensor dequant_scale2,Tensor quant_scale2, Tensor quant_offset2, Tensor antiquant_scale, Tensor antiquant_offset, Tensor blocktable,Tensor kv_padding_size, Tensor blockposition, int num_heads, float scale_value, str input_layout,int num_key_value_heads, int block_size, int inner_precise) ->Tensor");
    m.def("compute_cent(Tensor query, Tensor l1_cent) -> Tensor");
    m.def("select_position(Tensor key_ids, Tensor indices) -> (Tensor, Tensor)");
}

// 算子绑定二选一
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m)
{
    m.impl("incre_flash_attention_v4", &incre_flash_attention_v4_impl_npu);
    m.impl("incre_flash_attention_v5", &incre_flash_attention_v5_impl_npu);
    m.impl("compute_cent", &compute_cent_impl_npu);
    m.impl("select_position", &select_position_impl_npu);
}

// 通过 pybind 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("incre_flash_attention_v4", &incre_flash_attention_v4_impl_npu, "FlashAttention V4 implementation");
    m.def("incre_flash_attention_v5", &incre_flash_attention_v5_impl_npu, "FlashAttention V5 implementation");
    m.def("compute_cent", &compute_cent_impl_npu, "Compute Cent implementation");
    m.def("select_position", &select_position_impl_npu, "Select Position implementation");
}
