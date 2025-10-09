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
def sparse_paged_attention(query: torch.Tensor,
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
    封装 sparse_paged_attention 算子的 Python 接口
    """

    # key_list = [key]
    # value_list = [value]

    return custom_ops_lib.sparse_paged_attention(
        query, [key], [value], pse_shift, attention_mask, actual_seq_lengths,
        dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
        antiquant_scale, antiquant_offset, blocktable, kv_padding_size,
        blockposition,num_heads, scale_value, input_layout, num_key_value_heads, block_size, inner_precise
    )

def compute_cent(query: torch.Tensor, l1_cent: torch.Tensor) -> torch.Tensor:
    """
    封装 compute_cent 算子的 Python 接口
    """
    return custom_ops_lib.compute_cent(query, l1_cent)

def select_position(block_ids: torch.Tensor, block_table: torch.Tensor, seq_len: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    封装 select_position 算子的 Python 接口
    """
    return custom_ops_lib.select_position(block_ids, block_table, seq_len, indices)

def cent_select(query: torch.Tensor, l1_cent: torch.Tensor, block_ids: torch.Tensor, block_table: torch.Tensor, seq_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    封装 cent_select 算子的 Python 接口
    """
    return custom_ops_lib.cent_select(query, l1_cent, block_ids, block_table, seq_len)


def sparse_paged_fusion_attention(query: torch.Tensor,
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
                                  l1_cent: torch.Tensor,
                                  block_ids: torch.Tensor,
                                  total_seq_len: torch.Tensor,
                                  block_position: torch.Tensor,
                                  page_position_length: torch.Tensor,
                                  max_page_position_length: torch.Tensor,
                                  num_heads: int,
                                  scale_value: float,
                                  input_layout: str,
                                  num_key_value_heads: int,
                                  block_size: int,
                                  inner_precise: int) -> torch.Tensor:
    """
    融合算子 Python 封装：
    - 内部先进行 CentSelect（写回 block_position/page_position_length/max_page_position_length），
    - 再进行稀疏分页注意力，返回 attention_out。
    """
    return custom_ops_lib.sparse_paged_fusion_attention(
        query, [key], [value], pse_shift, attention_mask, actual_seq_lengths,
        dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
        antiquant_scale, antiquant_offset, blocktable, kv_padding_size,
        l1_cent, block_ids, total_seq_len,
        block_position, page_position_length, max_page_position_length,
        num_heads, scale_value, input_layout, num_key_value_heads, block_size, inner_precise
    )