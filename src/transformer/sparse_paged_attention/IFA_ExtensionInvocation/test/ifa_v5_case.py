# test_add_custom.py
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
# import custom_ops # pybind导入麻烦
import copy
import math
import numpy as np # 导入 numpy 来进行精确的类型转换

class Test_aclnnSparsePagedAttention(TestCase):
   
    def test_multi_head_block_table_and_position(self):
        """
        测试多头BlockTable模式，对应C++中的TestMultiHeadBlockTableMode_BSBD函数
        KV缓存格式为BNSD (blocknum, keyNumHeads, blockSize, headDims)
        """
        # 保持参数设置
        batch_size = 1
        num_heads = 32
        head_dims = 128
        key_num_heads = 32
        block_size = 16
        seq_len_q = 1
        total_seq_len_kv = 32 * 1024  # 已简化为4K
        block_num = total_seq_len_kv // block_size

        # 基准块数
        base_block_num = (4 * 1024) // block_size
        max_actual_block_num_per_seq = base_block_num

        print(f"批次大小: {batch_size}, 头数: {num_heads}, 键头数: {key_num_heads}")
        print(f"每头块数: {max_actual_block_num_per_seq}, 块大小: {block_size}")

        # 计算张量形状
        query_shape = [batch_size, num_heads, seq_len_q, head_dims]
        key_shape = [block_num, key_num_heads, block_size, head_dims]  
        value_shape = [block_num, key_num_heads, block_size, head_dims]

        blocks_per_head = [max_actual_block_num_per_seq] * key_num_heads

        # 计算总块数
        total_unique_blocks = sum(blocks_per_head)
        print(f"所有头总共需要 {total_unique_blocks} 个块")

        # 二维BlockTable和三维BlockPosition形状
        block_table_shape = [batch_size, total_unique_blocks]
        block_position_shape = [batch_size, key_num_heads, max_actual_block_num_per_seq]
        
        print(f"BlockTable形状: {block_table_shape}")
        print(f"BlockPosition形状: {block_position_shape}")

        # 选中和未选中块值
        SELECTED_BLOCK_VALUE = 0.1
        UNSELECTED_BLOCK_VALUE = 9.9

        # 创建数据
        query_data = torch.full(query_shape, 0.1, dtype=torch.float16)
        key_data = torch.full(key_shape, UNSELECTED_BLOCK_VALUE, dtype=torch.float16)
        value_data = torch.full(value_shape, UNSELECTED_BLOCK_VALUE, dtype=torch.float16)

        # 必须是int32!!!
        block_table_data = torch.full(block_table_shape, 0x7FFFFFFF, dtype=torch.int32)
        block_position_data = torch.full(block_position_shape, 0x7FFFFFFF, dtype=torch.int32)

        # 优化：预先计算每个头的块索引和值
        head_block_indices = {}
        head_values = {}
        
        for h in range(key_num_heads):
            base_offset = (h * 3) % block_num
            # 计算该头的所有块索引
            head_block_indices[h] = [(base_offset + i * 2) % block_num for i in range(blocks_per_head[h])]
            # 计算该头的特定值
            head_values[h] = SELECTED_BLOCK_VALUE + h * 0.001
        
        # 设置BlockTable和BlockPosition
        block_table_idx = 0
        for h in range(key_num_heads):
            print(f"头 {h} 实际块数: {blocks_per_head[h]}/{max_actual_block_num_per_seq}")
            
            selected_blocks = head_block_indices[h][:10]
            if selected_blocks:
                print(f"头 {h} 选择的块: {selected_blocks}" + ("..." if len(head_block_indices[h]) > 10 else ""))
            
            # 为当前头创建int32类型的块索引张量
            block_indices_tensor = torch.tensor(head_block_indices[h], dtype=torch.int32)
            
            # 批量设置BlockPosition和BlockTable
            for i in range(len(head_block_indices[h])):
                # 确保使用int32类型
                block_table_data[0, block_table_idx] = block_indices_tensor[i]
                # 确保使用int32类型
                block_position_data[0, h, i] = torch.tensor(block_table_idx, dtype=torch.int32)
                block_table_idx += 1
        
        # 优化：使用向量化操作填充KV数据
        for h in range(key_num_heads):
            head_value = head_values[h]
            for block_idx in head_block_indices[h]:
                # 计算该块在KV缓存中的起始索引
                block_start = block_idx * key_num_heads * block_size * head_dims + h * block_size * head_dims
                
                # 使用向量化操作一次性填充整个块的数据
                for s in range(block_size):
                    start_idx = block_start + s * head_dims
                    end_idx = start_idx + head_dims
                    if end_idx <= key_data.numel():
                        key_data.view(-1)[start_idx:end_idx] = head_value
                        value_data.view(-1)[start_idx:end_idx] = head_value

        # 设置实际序列长度
        actual_seq_lengths = torch.tensor([max_actual_block_num_per_seq * block_size], dtype=torch.int64)

        # 将数据移至NPU
        query_npu = query_data.npu()
        key_npu = key_data.npu()
        value_npu = value_data.npu()
        block_table_npu = block_table_data.npu()
        block_position_npu = block_position_data.npu()
        seq_len_npu = actual_seq_lengths.npu()

        # 创建空张量
        empty_tensor = torch.empty(0).npu()

        # 设置算子参数
        scale_value = 1.0 / math.sqrt(head_dims)
        layout = "BNSD"  # KV缓存布局为BNSD
        inner_precise = 1

        print("调用算子...")
        # 调用算子
        output = custom_ops.incre_flash_attention_v5(
            query_npu, key_npu, value_npu,
            empty_tensor,  # pse_shift
            empty_tensor,  # mask
            seq_len_npu,
            empty_tensor,  # dequant_scale1
            empty_tensor,  # quant_scale1
            empty_tensor,  # dequant_scale2
            empty_tensor,  # quant_scale2
            empty_tensor,  # quant_offset2
            empty_tensor,  # antiquant_scale
            empty_tensor,  # antiquant_offset
            block_table_npu,
            empty_tensor,  # kv_padding_size
            block_position_npu,
            num_heads,
            np.float32(scale_value),
            layout,
            key_num_heads,
            block_size,
            inner_precise
        )

        # 结果验证
        output_cpu = output.cpu()

        # 分头统计结果
        print("\n=== 计算结果分析 ===")
        for h in range(min(num_heads, 5)):
            head_data = output_cpu[0, h].flatten()
            head_mean = head_data.mean().item()
            head_min = head_data.min().item()
            head_max = head_data.max().item()
            print(f"头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")
            
        if num_heads > 5:
            h = num_heads - 1
            head_data = output_cpu[0, h].flatten()
            head_mean = head_data.mean().item()
            head_min = head_data.min().item()
            head_max = head_data.max().item()
            print(f"头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")

        # 结果验证
        all_data = output_cpu.flatten()
        has_unselected_value = ((all_data - UNSELECTED_BLOCK_VALUE).abs() < 1.0).any().item()

        # 检查不同头的结果差异
        first_head_value = output_cpu[0, 0, 0, 0].item()
        heads_have_different_results = False
        for h in range(1, num_heads):
            if abs(output_cpu[0, h, 0, 0].item() - first_head_value) > 1e-5:
                heads_have_different_results = True
                break
                
        print(f"\n=== 多头BlockTable映射分析 ===")
        print(f"块选择测试: {'✅ 通过，结果不包含未选中块的影响' if not has_unselected_value else '❌ 失败，结果中检测到未选中块的影响'}")
        print(f"多头差异: {'✅ 通过，不同头的结果有差异，说明每个头独立选择块成功' if heads_have_different_results else '❌ 失败，所有头结果相似，多头块选择可能未生效'}")
        print(f"算子功能: {'✅ 二维BlockTable + 三维Position映射工作正常' if not has_unselected_value and heads_have_different_results else '❌ 映射机制可能存在问题，请检查实现'}")
        print("=== 结果分析完成 ===\n")

    def test_block_size_head_layout(self):
        """
        测试Block-Size-Head布局模式，对应C++中的TestBlockSizeHeadLayout_BSH函数
        KV缓存格式为(blocknum, blocksize, H*D)
        """
        # 测试参数
        batch_size = 1
        num_heads = 32
        head_dims = 128
        key_num_heads = 32
        block_size = 16
        seq_len_q = 1
        total_seq_len_kv = 32 * 1024
        block_num = total_seq_len_kv // block_size

        # 设置基准块数和最大差异
        base_block_num = (10 * 1024) // block_size
        max_actual_block_num_per_seq = base_block_num

        print(f"批次大小: {batch_size}, 头数: {num_heads}, 键头数: {key_num_heads}")
        print(f"每头块数: {max_actual_block_num_per_seq}, 块大小: {block_size}")
        print(f"所有头使用相同的块数: {base_block_num}")

        # 张量形状定义 - 注意KV形状为(blocknum, blocksize, H*D)
        query_shape = [batch_size, num_heads, seq_len_q, head_dims]
        key_shape = [block_num, block_size, key_num_heads * head_dims]
        value_shape = [block_num, block_size, key_num_heads * head_dims]
        out_shape = [batch_size, num_heads, seq_len_q, head_dims]

        # 为每个头确定块数
        blocks_per_head = [max_actual_block_num_per_seq] * key_num_heads

        # 计算全局需要的最大块数
        total_unique_blocks = sum(blocks_per_head)
        print(f"所有头总共需要 {total_unique_blocks} 个块")

        # 二维BlockTable和三维blockPosition形状
        block_table_shape = [batch_size, total_unique_blocks]
        block_position_shape = [batch_size, key_num_heads, max_actual_block_num_per_seq]
        
        print(f"BlockTable形状: {block_table_shape}")
        print(f"BlockPosition形状: {block_position_shape}")

        print("张量形状信息:")
        print(f"  Query Shape (BNSD): {query_shape}")
        print(f"  Key Shape (blocknum, blocksize, H*D): {key_shape}")
        print(f"  Value Shape (blocknum, blocksize, H*D): {value_shape}")
        print(f"  Output Shape: {out_shape}")
        print(f"  最大序列长度: {max_actual_block_num_per_seq * block_size}")

        # 选择性块值定义
        SELECTED_BLOCK_VALUE = 0.1
        UNSELECTED_BLOCK_VALUE = 9.9

        # 创建数据
        query_data = torch.full(query_shape, 0.1, dtype=torch.float16)
        key_data = torch.full(key_shape, UNSELECTED_BLOCK_VALUE, dtype=torch.float16)
        value_data = torch.full(value_shape, UNSELECTED_BLOCK_VALUE, dtype=torch.float16)

        # 必须是int32!!!
        block_table_data = torch.full(block_table_shape, 1, dtype=torch.int32)
        block_position_data = torch.full(block_position_shape, 1, dtype=torch.int32)

        # 优化：预先计算每个头的块索引和值
        head_block_indices = {}
        head_values = {}
        
        for h in range(key_num_heads):
            base_offset = (h * 3) % block_num
            # 计算该头的所有块索引
            head_block_indices[h] = [(base_offset + i * 2) % block_num for i in range(blocks_per_head[h])]
            # 计算该头的特定值
            head_values[h] = SELECTED_BLOCK_VALUE + h * 0.001
        
        # 设置BlockTable和BlockPosition
        block_table_idx = 0
        for h in range(key_num_heads):
            print(f"头 {h} 实际块数: {blocks_per_head[h]}/{max_actual_block_num_per_seq}")
            
            selected_blocks = head_block_indices[h][:10]
            if selected_blocks:
                print(f"头 {h} 选择的块: {selected_blocks}" + ("..." if len(head_block_indices[h]) > 10 else ""))
            
            # 为当前头创建int32类型的块索引张量
            block_indices_tensor = torch.tensor(head_block_indices[h], dtype=torch.int32)
            
            # 批量设置BlockPosition和BlockTable
            for i in range(len(head_block_indices[h])):
                # 确保使用int32类型
                block_table_data[0, block_table_idx] = block_indices_tensor[i]
                # 确保使用int32类型
                block_position_data[0, h, i] = torch.tensor(block_table_idx, dtype=torch.int32)
                block_table_idx += 1
        
        # 优化：使用向量化操作填充KV数据 - BSH格式
        for h in range(key_num_heads):
            head_value = head_values[h]
            for block_idx in head_block_indices[h]:
                # 计算该块在KV缓存中的起始位置 - BSH格式
                for s in range(block_size):
                    # BSH格式的起始索引
                    start_idx = block_idx * block_size * key_num_heads * head_dims + s * key_num_heads * head_dims + h * head_dims
                    end_idx = start_idx + head_dims
                    if end_idx <= key_data.numel():
                        key_data.view(-1)[start_idx:end_idx] = head_value
                        value_data.view(-1)[start_idx:end_idx] = head_value
    
        # 设置实际序列长度
        actual_seq_lengths = torch.tensor([max_actual_block_num_per_seq * block_size], dtype=torch.int64)

        # 将数据移至NPU
        query_npu = query_data.npu()
        key_npu = key_data.npu()
        value_npu = value_data.npu()
        block_table_npu = block_table_data.npu()
        block_position_npu = block_position_data.npu()
        seq_len_npu = actual_seq_lengths.npu()

        # 创建空张量
        empty_tensor = torch.empty(0).npu()

        # 设置算子参数
        scale_value = 1.0 / math.sqrt(head_dims)
        layout = "BNSD"  # 注意：虽然KV是BSH布局，但layout参数仍设为BNSD
        inner_precise = 1

        print("调用算子...")
        # 调用算子
        output = custom_ops.incre_flash_attention_v4(
            query_npu, key_npu, value_npu,
            empty_tensor,  # pse_shift
            empty_tensor,  # mask
            seq_len_npu,
            empty_tensor,  # dequant_scale1
            empty_tensor,  # quant_scale1
            empty_tensor,  # dequant_scale2
            empty_tensor,  # quant_scale2
            empty_tensor,  # quant_offset2
            empty_tensor,  # antiquant_scale
            empty_tensor,  # antiquant_offset
            block_table_npu,
            empty_tensor,  # kv_padding_size
            # block_position_npu,
            num_heads,
            np.float32(scale_value),
            layout,
            key_num_heads,
            block_size,
            inner_precise
        )

        # 结果验证
        output_cpu = output.cpu()

        # 分头统计结果
        print("\n=== 计算结果分析 ===")
        for h in range(num_heads):
            head_data = output_cpu[0, h].flatten()
            head_mean = head_data.mean().item()
            head_min = head_data.min().item()
            head_max = head_data.max().item()
            print(f"头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")
            
        # if num_heads > 5:
        #     h = num_heads - 1
        #     head_data = output_cpu[0, h].flatten()
        #     head_mean = head_data.mean().item()
        #     head_min = head_data.min().item()
        #     head_max = head_data.max().item()
        #     print(f"头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")

        # 结果验证
        all_data = output_cpu.flatten()
        has_unselected_value = ((all_data - UNSELECTED_BLOCK_VALUE).abs() < 1.0).any().item()

        # 检查不同头的结果差异
        first_head_value = output_cpu[0, 0, 0, 0].item()
        heads_have_different_results = False
        for h in range(1, num_heads):
            if abs(output_cpu[0, h, 0, 0].item() - first_head_value) > 1e-5:
                heads_have_different_results = True
                break
                
        print(f"\n=== Block-Size-Head布局映射分析 ===")
        print(f"块选择测试: {'✅ 通过，结果不包含未选中块的影响' if not has_unselected_value else '❌ 失败，结果中检测到未选中块的影响'}")
        print(f"多头差异: {'✅ 通过，不同头的结果有差异，说明每个头独立选择块成功' if heads_have_different_results else '❌ 失败，所有头结果相似，多头块选择可能未生效'}")
        print(f"算子功能: {'✅ Block-Size-Head布局 (blocknum, blocksize, H) 工作正常' if not has_unselected_value and heads_have_different_results else '❌ 布局机制可能存在问题，请检查实现'}")
        print("=== 结果分析完成 ===\n")

    
    def test_block_size_head_layout_vllm(self):
        """
        测试Block-Size-Head布局模式，对应C++中的TestBlockSizeHeadLayout_BSH函数
        KV缓存格式为(blocknum, blocksize, H*D)
        """
        # 测试参数
        batch_size = 1 # set to 8
        num_heads = 32 
        head_dims = 128
        key_num_heads = 32
        block_size = 128 # set to 128
        seq_len_q = 1
        total_seq_len_kv = 32 * 1024
        block_num = total_seq_len_kv // block_size

        # 设置基准块数和最大差异
        base_block_num = (10 * 1024) // block_size
        max_actual_block_num_per_seq = base_block_num

        print(f"批次大小: {batch_size}, 头数: {num_heads}, 键头数: {key_num_heads}")
        print(f"每头块数: {max_actual_block_num_per_seq}, 块大小: {block_size}")
        print(f"所有头使用相同的块数: {base_block_num}")

        # 张量形状定义 - 注意KV形状为(blocknum, blocksize, H*D)
        query_shape = [batch_size, num_heads, head_dims]
        key_shape = [block_num, block_size, key_num_heads, head_dims]
        value_shape = [block_num, block_size, key_num_heads, head_dims]
        out_shape = [batch_size, num_heads, head_dims]

        # 为每个头确定块数
        blocks_per_head = [max_actual_block_num_per_seq] * key_num_heads

        # 计算全局需要的最大块数
        total_unique_blocks = sum(blocks_per_head)
        print(f"所有头总共需要 {total_unique_blocks} 个块")

        # 二维BlockTable和三维blockPosition形状
        block_table_shape = [batch_size, total_unique_blocks]
        block_position_shape = [batch_size, key_num_heads, max_actual_block_num_per_seq]
        
        print(f"BlockTable形状: {block_table_shape}")
        print(f"BlockPosition形状: {block_position_shape}")

        print("张量形状信息:")
        print(f"  Query Shape (BNSD): {query_shape}")
        print(f"  Key Shape (blocknum, blocksize, H*D): {key_shape}")
        print(f"  Value Shape (blocknum, blocksize, H*D): {value_shape}")
        print(f"  Output Shape: {out_shape}")
        print(f"  最大序列长度: {max_actual_block_num_per_seq * block_size}")

        # 选择性块值定义
        SELECTED_BLOCK_VALUE = 0.1
        UNSELECTED_BLOCK_VALUE = 9.9

        data_type = torch.float16
        # 创建数据
        query_data = torch.full(query_shape, 0.1, dtype=data_type)
        key_data = torch.full(key_shape, UNSELECTED_BLOCK_VALUE, dtype=data_type)
        value_data = torch.full(value_shape, UNSELECTED_BLOCK_VALUE, dtype=data_type)

        # 必须是int32!!!
        block_table_data = torch.full(block_table_shape, 0x7FFFFFFF, dtype=torch.int32)
        block_position_data = torch.full(block_position_shape, 0x7FFFFFFF, dtype=torch.int32)

        # 优化：预先计算每个头的块索引和值
        head_block_indices = {}
        head_values = {}
        
        for h in range(key_num_heads):
            base_offset = (h * 3) % block_num
            # 计算该头的所有块索引
            head_block_indices[h] = [(base_offset + i * 2) % block_num for i in range(blocks_per_head[h])]
            # 计算该头的特定值
            head_values[h] = SELECTED_BLOCK_VALUE # + h * 0.001
        
        # 设置BlockTable和BlockPosition - 为所有batch设置相同的数据
        block_table_idx = 0
        for h in range(key_num_heads):
            print(f"头 {h} 实际块数: {blocks_per_head[h]}/{max_actual_block_num_per_seq}")
            
            selected_blocks = head_block_indices[h][:10]
            if selected_blocks:
                print(f"头 {h} 选择的块: {selected_blocks}" + ("..." if len(head_block_indices[h]) > 10 else ""))
            
            # 为当前头创建int32类型的块索引张量
            block_indices_tensor = torch.tensor(head_block_indices[h], dtype=torch.int32)
            
            # 批量设置BlockPosition和BlockTable - 为所有batch设置相同的数据
            for i in range(len(head_block_indices[h])):
                # 确保使用int32类型
                for batch_idx in range(batch_size):
                    block_table_data[batch_idx, block_table_idx] = block_indices_tensor[i]
                    # 确保使用int32类型
                    block_position_data[batch_idx, h, i] = torch.tensor(block_table_idx, dtype=torch.int32)
                block_table_idx += 1
        
        # 优化：使用向量化操作填充KV数据 - BSH格式
        for h in range(key_num_heads):
            head_value = head_values[h]
            for block_idx in head_block_indices[h]:
                # 计算该块在KV缓存中的起始位置 - BSH格式
                for s in range(block_size):
                    # # BSH格式的起始索引
                    start_idx = block_idx * block_size * key_num_heads * head_dims + s * key_num_heads * head_dims + h * head_dims
                    end_idx = start_idx + head_dims
                    if end_idx <= key_data.numel():
                        key_data.view(-1)[start_idx:end_idx] = head_value
                        value_data.view(-1)[start_idx:end_idx] = head_value
                    
                    # # 形状: [block_num, block_size, key_num_heads, head_dims]
                    # key_data[block_idx, :, h, :] = head_value
                    # value_data[block_idx, :, h, :] = head_value
    
        # # 验证填充是否正确
        # print("\n=== 填充数据验证 ===")
        # for h in [0, 15, 31]:  # 检查第一个、中间和最后一个头
        #     block_idx = head_block_indices[h][0]  # 检查第一个选择的块
        #     sample_value = key_data[block_idx, 0, h, 0].item()
        #     expected_value = SELECTED_BLOCK_VALUE + h * 0.001
            
        #     print(f"头 {h} 的第一个块 {block_idx} 的样本值: {sample_value:.6f} (期望值: {expected_value:.6f})")
        #     if abs(sample_value - expected_value) < 1e-5:
        #         print("  ✅ 填充正确")
        #     else:
        #         print("  ❌ 填充错误")
            
    
        # 设置实际序列长度
        actual_seq_lengths = torch.tensor([max_actual_block_num_per_seq * block_size], dtype=torch.int64)

        # 将数据移至NPU
        query_npu = query_data.npu()
        key_npu = key_data.npu()
        value_npu = value_data.npu()
        block_table_npu = block_table_data.npu()
        block_position_npu = block_position_data.npu()
        seq_len_npu = actual_seq_lengths.npu()

        # 创建空张量
        empty_tensor = torch.empty(0).npu()

        # 设置算子参数
        scale_value = 1.0 / math.sqrt(head_dims)
        layout = "BNSD"  # 注意：虽然KV是BSH布局，但layout参数仍设为BNSD
        inner_precise = 1

        print("调用算子...")
        # 调用算子
        output = custom_ops.incre_flash_attention_v5(
            query_npu, key_npu, value_npu,
            empty_tensor,  # pse_shift
            empty_tensor,  # mask
            seq_len_npu,
            empty_tensor,  # dequant_scale1
            empty_tensor,  # quant_scale1
            empty_tensor,  # dequant_scale2
            empty_tensor,  # quant_scale2
            empty_tensor,  # quant_offset2
            empty_tensor,  # antiquant_scale
            empty_tensor,  # antiquant_offset
            block_table_npu,
            empty_tensor,  # kv_padding_size
            block_position_npu,
            num_heads,
            np.float32(scale_value),
            layout,
            key_num_heads,
            block_size,
            inner_precise
        )

        # 结果验证
        output_cpu = output.unsqueeze(2).cpu()

        # 分头统计结果
        print("\n=== 计算结果分析 ===")
        for h in range(num_heads):
            head_data = output_cpu[0, h].flatten()
            head_mean = head_data.mean().item()
            head_min = head_data.min().item()
            head_max = head_data.max().item()
            print(f"头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")
       
        # 结果验证
        all_data = output_cpu.flatten()
        has_unselected_value = ((all_data - UNSELECTED_BLOCK_VALUE).abs() < 1.0).any().item()

        # 检查不同头的结果差异
        first_head_value = output_cpu[0, 0, 0, 0].item()
        heads_have_different_results = False
        for h in range(1, num_heads):
            if abs(output_cpu[0, h, 0, 0].item() - first_head_value) > 1e-5:
                heads_have_different_results = True
                break
                
        print(f"\n=== Block-Size-Head布局映射分析 ===")
        print(f"块选择测试: {'✅ 通过，结果不包含未选中块的影响' if not has_unselected_value else '❌ 失败，结果中检测到未选中块的影响'}")
        print(f"多头差异: {'✅ 通过，不同头的结果有差异，说明每个头独立选择块成功' if heads_have_different_results else '❌ 失败，所有头结果相似，多头块选择可能未生效'}")
        print(f"算子功能: {'✅ Block-Size-Head布局 (blocknum, blocksize, H) 工作正常' if not has_unselected_value and heads_have_different_results else '❌ 布局机制可能存在问题，请检查实现'}")
        print("=== 结果分析完成 ===\n")

    def test_block_size_head_layout_vllm_result(self):
        """
        测试Block-Size-Head布局模式，对应C++中的TestBlockSizeHeadLayout_BSH函数
        KV缓存格式为(blocknum, blocksize, H*D)
        """
        # 测试参数
        batch_size = 1 # set to 8
        num_heads = 32 
        head_dims = 128
        key_num_heads = 32
        block_size = 128 # set to 128
        seq_len_q = 1
        total_seq_len_kv = 32 * 1024
        block_num = total_seq_len_kv // block_size

        # 设置基准块数和最大差异
        base_block_num = (10 * 1024) // block_size
        max_actual_block_num_per_seq = base_block_num

        print(f"批次大小: {batch_size}, 头数: {num_heads}, 键头数: {key_num_heads}")
        print(f"每头块数: {max_actual_block_num_per_seq}, 块大小: {block_size}")
        print(f"所有头使用相同的块数: {base_block_num}")

        # 张量形状定义 - 注意KV形状为(blocknum, blocksize, H*D)
        query_shape = [batch_size, num_heads, head_dims]
        key_shape = [block_num, block_size, key_num_heads, head_dims]
        value_shape = [block_num, block_size, key_num_heads, head_dims]
        out_shape = [batch_size, num_heads, head_dims]

        # 为每个头确定块数
        blocks_per_head = [max_actual_block_num_per_seq] * key_num_heads

        # 计算全局需要的最大块数
        total_unique_blocks = sum(blocks_per_head)
        print(f"所有头总共需要 {total_unique_blocks} 个块")

        # 二维BlockTable和三维blockPosition形状
        block_table_shape = [batch_size, total_unique_blocks]
        block_position_shape = [batch_size, key_num_heads, max_actual_block_num_per_seq]
        
        print(f"BlockTable形状: {block_table_shape}")
        print(f"BlockPosition形状: {block_position_shape}")

        print("张量形状信息:")
        print(f"  Query Shape (BNSD): {query_shape}")
        print(f"  Key Shape (blocknum, blocksize, H*D): {key_shape}")
        print(f"  Value Shape (blocknum, blocksize, H*D): {value_shape}")
        print(f"  Output Shape: {out_shape}")
        print(f"  最大序列长度: {max_actual_block_num_per_seq * block_size}")

        # 选择性块值定义
        SELECTED_BLOCK_VALUE = 0.1
        UNSELECTED_BLOCK_VALUE = 9.9

        data_type = torch.float16
        # 创建数据
        query_data = torch.full(query_shape, 0.1, dtype=data_type)
        key_data = torch.full(key_shape, UNSELECTED_BLOCK_VALUE, dtype=data_type)
        value_data = torch.full(value_shape, UNSELECTED_BLOCK_VALUE, dtype=data_type)

        # 必须是int32!!!
        block_table_data = torch.full(block_table_shape, 0x7FFFFFFF, dtype=torch.int32)
        block_position_data = torch.full(block_position_shape, 0x7FFFFFFF, dtype=torch.int32)

        # 优化：预先计算每个头的块索引和值
        head_block_indices = {}
        head_values = {}
        
        for h in range(key_num_heads):
            base_offset = (h * 3) % block_num
            # 计算该头的所有块索引
            head_block_indices[h] = [(base_offset + i * 2) % block_num for i in range(blocks_per_head[h])]
            # 计算该头的特定值
            head_values[h] = SELECTED_BLOCK_VALUE # + h * 0.001
        
        # 设置BlockTable和BlockPosition - 为所有batch设置相同的数据
        block_table_idx = 0
        for h in range(key_num_heads):
            print(f"头 {h} 实际块数: {blocks_per_head[h]}/{max_actual_block_num_per_seq}")
            
            selected_blocks = head_block_indices[h][:10]
            if selected_blocks:
                print(f"头 {h} 选择的块: {selected_blocks}" + ("..." if len(head_block_indices[h]) > 10 else ""))
            
            # 为当前头创建int32类型的块索引张量
            block_indices_tensor = torch.tensor(head_block_indices[h], dtype=torch.int32)
            
            # 批量设置BlockPosition和BlockTable - 为所有batch设置相同的数据
            for i in range(len(head_block_indices[h])):
                # 确保使用int32类型
                for batch_idx in range(batch_size):
                    block_table_data[batch_idx, block_table_idx] = block_indices_tensor[i]
                    # 确保使用int32类型
                    block_position_data[batch_idx, h, i] = torch.tensor(block_table_idx, dtype=torch.int32)
                block_table_idx += 1
        
        # 优化：使用向量化操作填充KV数据 - BSH格式
        for h in range(key_num_heads):
            head_value = head_values[h]
            for block_idx in head_block_indices[h]:
                # 计算该块在KV缓存中的起始位置 - BSH格式
                for s in range(block_size):
                    # # BSH格式的起始索引
                    start_idx = block_idx * block_size * key_num_heads * head_dims + s * key_num_heads * head_dims + h * head_dims
                    end_idx = start_idx + head_dims
                    if end_idx <= key_data.numel():
                        key_data.view(-1)[start_idx:end_idx] = head_value
                        value_data.view(-1)[start_idx:end_idx] = head_value
                    
        # 设置实际序列长度
        actual_seq_lengths = torch.tensor([max_actual_block_num_per_seq * block_size], dtype=torch.int64)

        # 将数据移至NPU
        query_npu = query_data.npu()
        key_npu = key_data.npu()
        value_npu = value_data.npu()
        block_table_npu = block_table_data.npu()
        block_position_npu = block_position_data.npu()
        seq_len_npu = actual_seq_lengths.npu()

        # 创建空张量
        empty_tensor = torch.empty(0).npu()

        # 设置算子参数
        scale_value = 1.0 #/ math.sqrt(head_dims)
        layout = "BNSD"  # 注意：虽然KV是BSH布局，但layout参数仍设为BNSD
        inner_precise = 1

        print("调用算子...")
        
        # # warmup
        for i in range(10):
            # 调用算子
            output = torch_npu.npu_sparse_paged_attention(
                # positional arguments
                query_npu,
                key_npu,
                value_npu,
                # keyword-only arguments
                actual_seq_lengths=seq_len_npu,
                block_table=block_table_npu,
                block_position=block_position_npu,
                num_heads=num_heads,
                scale_value=np.float32(scale_value),
                input_layout=layout,
                num_key_value_heads=key_num_heads,
                block_size=block_size,
                inner_precise=inner_precise
            )
            
        torch_npu.npu.synchronize()   
        import time;
        start = time.time()
        # 调用算子
        # output = custom_ops.sparse_paged_attention(
        #         query_npu, key_npu, value_npu,
        #         empty_tensor,  # pse_shift
        #         empty_tensor,  # mask
        #         seq_len_npu,
        #         empty_tensor,  # dequant_scale1
        #         empty_tensor,  # quant_scale1
        #         empty_tensor,  # dequant_scale2
        #         empty_tensor,  # quant_scale2
        #         empty_tensor,  # quant_offset2
        #         empty_tensor,  # antiquant_scale
        #         empty_tensor,  # antiquant_offset
        #         block_table_npu,
        #         empty_tensor,  # kv_padding_size
        #         block_position_npu,
        #         num_heads,
        #         np.float32(scale_value),
        #         layout,
        #         key_num_heads,
        #         block_size,
        #         inner_precise
        #     )
        output = torch_npu.npu_sparse_paged_attention(
            # positional arguments
            query_npu,
            key_npu,
            value_npu,
            # keyword-only arguments
            actual_seq_lengths=seq_len_npu,
            block_table=block_table_npu,
            block_position=block_position_npu,
            num_heads=num_heads,
            scale_value=np.float32(scale_value),
            input_layout=layout,
            num_key_value_heads=key_num_heads,
            block_size=block_size,
            inner_precise=inner_precise
        )
        
        torch_npu.npu.synchronize()
        end = time.time()
        elapsed_us = (end - start) * 1000000
        print(f"执行时间: {elapsed_us:.2f} us")
        
        # 结果验证
        custom_output = output.unsqueeze(2).cpu()
        
        # 分头统计结果
        print("\n=== 计算结果分析 ===")
        for h in range(num_heads):
            head_data = custom_output[0, h].flatten()
            head_mean = head_data.mean().item()
            head_min = head_data.min().item()
            head_max = head_data.max().item()
            print(f"头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")
       
        # 结果验证
        all_data = custom_output.flatten()
        has_unselected_value = ((all_data - UNSELECTED_BLOCK_VALUE).abs() < 1.0).any().item()

        # 检查不同头的结果差异
        first_head_value = custom_output[0, 0, 0, 0].item()
        heads_have_different_results = False
        for h in range(1, num_heads):
            if abs(custom_output[0, h, 0, 0].item() - first_head_value) > 1e-5:
                heads_have_different_results = True
                break
                
        print(f"\n=== Block-Size-Head布局映射分析 ===")
        print(f"块选择测试: {'✅ 通过，结果不包含未选中块的影响' if not has_unselected_value else '❌ 失败，结果中检测到未选中块的影响'}")
        print(f"多头差异: {'✅ 通过，不同头的结果有差异，说明每个头独立选择块成功' if heads_have_different_results else '❌ 失败，所有头结果相似，多头块选择可能未生效'}")
        print(f"算子功能: {'✅ Block-Size-Head布局 (blocknum, blocksize, H) 工作正常' if not has_unselected_value and heads_have_different_results else '❌ 布局机制可能存在问题，请检查实现'}")
        print("=== 结果分析完成 ===\n")
        
       
        
        # # 调用IFA官方算子 
        # kv_target_shape = [block_num, block_size, key_num_heads * head_dims]
        # # 准备关键字参数
        # kwargs = {
        #     'pse_shift': empty_tensor,
        #     'atten_mask': empty_tensor,
        #     'actual_seq_lengths': seq_len_npu,
        #     'dequant_scale1': empty_tensor,
        #     'quant_scale1': empty_tensor,
        #     'dequant_scale2': empty_tensor,
        #     'quant_scale2': empty_tensor,
        #     'quant_offset2': empty_tensor,
        #     'antiquant_scale': empty_tensor,
        #     'antiquant_offset': empty_tensor,
        #     'block_table': block_table_npu,
        #     'kv_padding_size': empty_tensor,
        #     'num_heads': num_heads,
        #     'scale_value': float(scale_value),  # 确保是float类型
        #     'input_layout': layout,
        #     'num_key_value_heads': key_num_heads,
        #     'block_size': block_size,
        #     'inner_precise': inner_precise
        # }
        # import ipdb;ipdb.set_trace()
        # torch_output = torch_npu.npu_incre_flash_attention(
        #     query_npu.unsqueeze(2), 
        #     key_npu.reshape(kv_target_shape), 
        #     value_npu.reshape(kv_target_shape),
        #     # **kwargs  # 使用关键字参数传递
        # )

        # # 确保形状一致
        # self.assertEqual(custom_output_cpu.shape, torch_output_cpu.shape)
        
        # # 计算误差
        # abs_diff = torch.abs(custom_output_cpu - torch_output_cpu)
        # max_abs_diff = torch.max(abs_diff).item()
        # mean_abs_diff = torch.mean(abs_diff).item()
        
        # print(f"最大绝对误差: {max_abs_diff:.6f}")
        # print(f"平均绝对误差: {mean_abs_diff:.6f}")
        
        # # 设置一个合理的误差容忍度
        # tolerance = 1e-2
        
        # if max_abs_diff < tolerance:
        #     print(f"✅ 精度对比通过 (误差 < {tolerance})")
        # else:
        #     print(f"❌ 精度对比失败 (误差 > {tolerance})")
            
        # self.assertTrue(max_abs_diff < tolerance)




if __name__ == "__main__":
    def run_single_test():
        test = Test_aclnnSparsePagedAttention("test_block_size_head_layout_vllm_result")
        test.setUp()
        test.test_block_size_head_layout_vllm_result()
        test.tearDown()
    def run_single_test_v4():
        test = Test_aclnnSparsePagedAttention("test_block_size_head_layout")
        test.setUp()
        test.test_block_size_head_layout()
        test.tearDown()
        
    # run_tests()  # 注释掉原来的运行所有测试的代码
    run_single_test()  # 只运行指定的测试
    # run_single_test_v4()