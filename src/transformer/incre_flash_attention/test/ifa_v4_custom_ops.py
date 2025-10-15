import torch
import torch_npu
import custom_ops # pybind导入麻烦

def test_incre_flash_attention_v4():
    """
    测试 incre_flash_attention_v4 算子的简单调用
    基于 ifa_v5_case.py 的实现结构
    """
    print("开始测试 incre_flash_attention_v4...")
    
    # 设置设备
    torch.npu.set_device(0)
    
    # 基础参数配置
    batch_size = 1
    num_heads = 32
    key_num_heads = 8
    head_dims = 128
    block_size = 128
    total_seq_len_kv = 16 * 1024
    block_num = total_seq_len_kv // block_size
    
    # 实际使用的块数（稀疏场景）
    max_actual_block_num_per_seq = (16 * 1024) // block_size
    
    print(f"测试参数: batch_size={batch_size}, num_heads={num_heads}, key_num_heads={key_num_heads}")
    print(f"head_dims={head_dims}, block_size={block_size}, total_seq_len_kv={total_seq_len_kv}")
    print(f"实际使用块数: {max_actual_block_num_per_seq}")
    
    # 张量形状定义
    query_shape = [batch_size, num_heads,1, head_dims]
    key_shape = [block_num, block_size, key_num_heads * head_dims]  # v4格式: (blocknum, blocksize, H*D)
    value_shape = [block_num, block_size, key_num_heads * head_dims]
    
    # BlockTable和blockPosition形状
    total_unique_blocks = key_num_heads * max_actual_block_num_per_seq
    block_table_shape = [batch_size, total_unique_blocks]
    block_position_shape = [batch_size, key_num_heads, max_actual_block_num_per_seq]
    
    print(f"张量形状: query={query_shape}, key={key_shape}, value={value_shape}")
    print(f"block_table={block_table_shape}, block_position={block_position_shape}")
    
    # 创建测试数据 - 全部使用1.0
    data_type = torch.float16
    query_data = torch.ones(query_shape, dtype=data_type)
    key_data = torch.ones(key_shape, dtype=data_type)
    value_data = torch.ones(value_shape, dtype=data_type)
    
    # BlockTable和BlockPosition数据 - 简化为全0
    block_table_data = torch.zeros(block_table_shape, dtype=torch.int32)
    block_position_data = torch.zeros(block_position_shape, dtype=torch.int32)
    
    # 实际序列长度
    actual_seq_lengths = torch.tensor([max_actual_block_num_per_seq * block_size], dtype=torch.int64)
    
    # 移至NPU
    query_npu = query_data.npu()
    key_npu = key_data.npu()
    value_npu = value_data.npu()
    block_table_npu = block_table_data.npu()
    block_position_npu = block_position_data.npu()
    seq_len_npu = actual_seq_lengths.npu()
    
    # 创建空张量（v4接口需要）
    empty_tensor = torch.empty(0, dtype=torch.float16).npu()
    
    # 算子参数
    scale_value = 1.0
    layout = "BNSD"
    inner_precise = 1
    
    print("调用 incre_flash_attention_v4 算子...")
    
    try:
        # 导入自定义算子
        # from cann_ops_adv_vllm.extension.custom_ops.add_custom import incre_flash_attention_v4
        
        # 预热
        print("预热阶段...")
        for _ in range(5):
            output = custom_ops.incre_flash_attention_v4(
                query_npu, [key_npu], [value_npu], empty_tensor, empty_tensor,
                seq_len_npu, empty_tensor, empty_tensor, empty_tensor, empty_tensor,
                empty_tensor, empty_tensor, empty_tensor, block_table_npu, empty_tensor,
                num_heads, scale_value, layout, key_num_heads, block_size, inner_precise
            )
            
        # import ipdb; ipdb.set_trace()
        # torch_npu.npu.synchronize()
        
        # 性能测试
        print("性能测试阶段...")
        import time
        start = time.time()
        for _ in range(10):
            output = custom_ops.incre_flash_attention_v4(
                query_npu, [key_npu], [value_npu], empty_tensor, empty_tensor,
                seq_len_npu, empty_tensor, empty_tensor, empty_tensor, empty_tensor,
                empty_tensor, empty_tensor, empty_tensor, block_table_npu, empty_tensor,
                num_heads, scale_value, layout, key_num_heads, block_size, inner_precise
            )
        
        # torch_npu.npu.synchronize()
        end = time.time()
        
        # print(f"✅ 算子执行成功!")
        # print(f"执行时间: {(end - start) * 1000000:.2f} us")
        # print(f"输出形状: {output.shape}")
        
        # 结果分析
        # output_cpu = output.cpu().float()
        # print(f"输出统计: mean={output_cpu.mean().item():.6f}, std={output_cpu.std().item():.6f}")
        
        # 检查是否有未选中块的影响
        # has_unselected_value = ((output_cpu - 9.9).abs() < 1.0).any().item()
        # print(f"块选择验证: {'✅ 通过' if not has_unselected_value else '❌ 失败'}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
        
    except Exception as e:
        print(f"❌ 执行错误: {e}")
        return False


if __name__ == "__main__":
    success = test_incre_flash_attention_v4()
    if success:
        print("\n🎉 测试完成!")
    else:
        print("\n💥 测试失败!")
        exit(1)
