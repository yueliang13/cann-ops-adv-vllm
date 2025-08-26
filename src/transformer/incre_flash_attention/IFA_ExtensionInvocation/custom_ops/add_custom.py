import torch
import custom_ops_lib
from typing import List

import torch
import custom_ops_lib
from typing import List, Optional


def incre_flash_attention_v4(query: torch.Tensor,
                           key_list: List[torch.Tensor],
                           value_list: List[torch.Tensor],
                           pse_shift: torch.Tensor,
                           attention_mask: torch.Tensor,
                           actual_seq_lengths: torch.Tensor,
                           dequant_scale1: torch.Tensor,
                           quant_scale1: torch.Tensor,
                           dequant_scale2: torch.Tensor,
                           quant_scale2: torch.Tensor,
                           quant_offset2: torch.Tensor,
                           antiquant_scale: torch.Tensor,
                           antiquant_offset: torch.Tensor,
                           blocktable: torch.Tensor,
                           kv_padding_size: torch.Tensor,
                           num_heads: int,
                           scale_value: float,
                           input_layout: str,
                           num_key_value_heads: int,
                           block_size: int,
                           inner_precise: int) -> torch.Tensor:
    """
    封装 incre_flash_attention_v4 算子的 Python 接口
    """
    
    return custom_ops_lib.incre_flash_attention_v4(
        query, key_list, value_list, pse_shift, attention_mask, actual_seq_lengths,
        dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
        antiquant_scale, antiquant_offset, blocktable, kv_padding_size,
        num_heads, scale_value, input_layout, num_key_value_heads, block_size, inner_precise
    )
def incre_flash_attention_v5(query: torch.Tensor,
                           key: torch.Tensor,
                           value: torch.Tensor,
                           pse_shift: torch.Tensor,
                           attention_mask: torch.Tensor,
                           actual_seq_lengths: torch.Tensor,
                           dequant_scale1: torch.Tensor,
                           quant_scale1: torch.Tensor,
                           dequant_scale2: torch.Tensor,
                           quant_scale2: torch.Tensor,
                           quant_offset2: torch.Tensor,
                           antiquant_scale: torch.Tensor,
                           antiquant_offset: torch.Tensor,
                           blocktable: torch.Tensor,
                           kv_padding_size: torch.Tensor,
                           blockposition: torch.Tensor,
                           num_heads: int,
                           scale_value: float,
                           input_layout: str,
                           num_key_value_heads: int,
                           block_size: int,
                           inner_precise: int) -> torch.Tensor:
    """
    封装 incre_flash_attention_v5 算子的 Python 接口
    """

    key_list = [key]
    value_list = [value]

    return custom_ops_lib.incre_flash_attention_v5(
        query, key_list, value_list, pse_shift, attention_mask, actual_seq_lengths,
        dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
        antiquant_scale, antiquant_offset, blocktable, kv_padding_size,
        blockposition,num_heads, scale_value, input_layout, num_key_value_heads, block_size, inner_precise
    )

def compute_cent(query: torch.Tensor, l1_cent: torch.Tensor, d_l1_cent: torch.Tensor, mask_empty: torch.Tensor, select_nprobe: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    封装 compute_cent 算子的 Python 接口
    """
    return custom_ops_lib.compute_cent(query, l1_cent, d_l1_cent, mask_empty, select_nprobe, indices)