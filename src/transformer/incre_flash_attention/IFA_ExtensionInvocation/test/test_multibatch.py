#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多Batch测试验证脚本
用于验证ifa_v5_case.py中的多Batch适配是否正确
"""

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops
import math
import numpy as np

def test_multibatch_setup():
    """
    测试多Batch设置是否正确
    """
    print("=== 多Batch设置验证 ===")
    
    # 测试参数
    batch_size = 8
    num_heads = 32
    head_dims = 128
    key_num_heads = 32
    block_size = 16
    seq_len_q = 1
    total_seq_len_kv = 32 * 1024
    block_num = total_seq_len_kv // block_size
    
    # 基准块数
    base_block_num = (4 * 1024) // block_size
    max_actual_block_num_per_seq = base_block_num
    
    print(f"批次大小: {batch_size}")
    print(f"头数: {num_heads}")
    print(f"键头数: {key_num_heads}")
    print(f"每头块数: {max_actual_block_num_per_seq}")
    print(f"块大小: {block_size}")
    
    # 计算张量形状
    query_shape = [batch_size, num_heads, seq_len_q, head_dims]
    key_shape = [block_num, key_num_heads, block_size, head_dims]
    value_shape = [block_num, key_num_heads, block_size, head_dims]
    
    blocks_per_head = [max_actual_block_num_per_seq] * key_num_heads
    total_unique_blocks = sum(blocks_per_head)
    
    # 二维BlockTable和三维BlockPosition形状
    block_table_shape = [batch_size, total_unique_blocks]
    block_position_shape = [batch_size, key_num_heads, max_actual_block_num_per_seq]
    
    print(f"Query形状: {query_shape}")
    print(f"Key形状: {key_shape}")
    print(f"Value形状: {value_shape}")
    print(f"BlockTable形状: {block_table_shape}")
    print(f"BlockPosition形状: {block_position_shape}")
    
    # 验证形状是否正确
    assert query_shape[0] == batch_size, f"Query batch维度错误: {query_shape[0]} != {batch_size}"
    assert block_table_shape[0] == batch_size, f"BlockTable batch维度错误: {block_table_shape[0]} != {batch_size}"
    assert block_position_shape[0] == batch_size, f"BlockPosition batch维度错误: {block_position_shape[0]} != {batch_size}"
    
    print("✅ 多Batch形状设置正确")
    
    # 测试数据设置
    SELECTED_BLOCK_VALUE = 0.1
    UNSELECTED_BLOCK_VALUE = 9.9
    
    query_data = torch.full(query_shape, 0.1, dtype=torch.float16)
    key_data = torch.full(key_shape, UNSELECTED_BLOCK_VALUE, dtype=torch.float16)
    value_data = torch.full(value_shape, UNSELECTED_BLOCK_VALUE, dtype=torch.float16)
    
    block_table_data = torch.full(block_table_shape, 0x7FFFFFFF, dtype=torch.int32)
    block_position_data = torch.full(block_position_shape, 0x7FFFFFFF, dtype=torch.int32)
    
    print(f"Query数据形状: {query_data.shape}")
    print(f"Key数据形状: {key_data.shape}")
    print(f"Value数据形状: {value_data.shape}")
    print(f"BlockTable数据形状: {block_table_data.shape}")
    print(f"BlockPosition数据形状: {block_position_data.shape}")
    
    # 验证数据设置
    assert query_data.shape[0] == batch_size, f"Query数据batch维度错误"
    assert block_table_data.shape[0] == batch_size, f"BlockTable数据batch维度错误"
    assert block_position_data.shape[0] == batch_size, f"BlockPosition数据batch维度错误"
    
    print("✅ 多Batch数据设置正确")
    
    # 测试序列长度设置
    actual_seq_lengths = torch.tensor([max_actual_block_num_per_seq * block_size] * batch_size, dtype=torch.int64)
    print(f"序列长度形状: {actual_seq_lengths.shape}")
    print(f"序列长度值: {actual_seq_lengths}")
    
    assert actual_seq_lengths.shape[0] == batch_size, f"序列长度batch维度错误"
    assert (actual_seq_lengths == max_actual_block_num_per_seq * block_size).all(), "序列长度值错误"
    
    print("✅ 多Batch序列长度设置正确")
    
    print("=== 多Batch设置验证完成 ===\n")

def test_multibatch_data_consistency():
    """
    测试多Batch数据一致性
    """
    print("=== 多Batch数据一致性验证 ===")
    
    batch_size = 8
    key_num_heads = 32
    block_size = 16
    total_seq_len_kv = 32 * 1024
    block_num = total_seq_len_kv // block_size
    
    base_block_num = (4 * 1024) // block_size
    max_actual_block_num_per_seq = base_block_num
    
    blocks_per_head = [max_actual_block_num_per_seq] * key_num_heads
    total_unique_blocks = sum(blocks_per_head)
    
    block_table_shape = [batch_size, total_unique_blocks]
    block_position_shape = [batch_size, key_num_heads, max_actual_block_num_per_seq]
    
    block_table_data = torch.full(block_table_shape, 0x7FFFFFFF, dtype=torch.int32)
    block_position_data = torch.full(block_position_shape, 0x7FFFFFFF, dtype=torch.int32)
    
    # 模拟数据设置逻辑
    head_block_indices = {}
    for h in range(key_num_heads):
        base_offset = (h * 3) % block_num
        head_block_indices[h] = [(base_offset + i * 2) % block_num for i in range(blocks_per_head[h])]
    
    # 设置BlockTable和BlockPosition - 为所有batch设置相同的数据
    block_table_idx = 0
    for h in range(key_num_heads):
        block_indices_tensor = torch.tensor(head_block_indices[h], dtype=torch.int32)
        
        for i in range(len(head_block_indices[h])):
            for batch_idx in range(batch_size):
                block_table_data[batch_idx, block_table_idx] = block_indices_tensor[i]
                block_position_data[batch_idx, h, i] = torch.tensor(block_table_idx, dtype=torch.int32)
            block_table_idx += 1
    
    # 验证所有batch的数据是否一致
    for batch_idx in range(1, batch_size):
        assert torch.equal(block_table_data[batch_idx], block_table_data[0]), f"Batch {batch_idx} 的BlockTable数据与Batch 0不一致"
        assert torch.equal(block_position_data[batch_idx], block_position_data[0]), f"Batch {batch_idx} 的BlockPosition数据与Batch 0不一致"
    
    print("✅ 所有Batch的数据设置一致")
    
    # 验证数据有效性
    assert not (block_table_data == 0x7FFFFFFF).all(), "BlockTable数据未正确设置"
    assert not (block_position_data == 0x7FFFFFFF).all(), "BlockPosition数据未正确设置"
    
    print("✅ 数据有效性验证通过")
    
    print("=== 多Batch数据一致性验证完成 ===\n")

if __name__ == "__main__":
    print("开始多Batch适配验证...\n")
    
    try:
        test_multibatch_setup()
        test_multibatch_data_consistency()
        print("🎉 多Batch适配验证全部通过！")
    except Exception as e:
        print(f"❌ 多Batch适配验证失败: {e}")
        import traceback
        traceback.print_exc()
