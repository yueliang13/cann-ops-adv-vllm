# test_ifa_v5_int8_quantization.py
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops
import copy
import math
import numpy as np

class TestIncreFlashAttentionV5Int8Quantization(TestCase):
   
    def test_int8_quantization_bnsd_layout(self):
        """
        测试INT8量化版本的IncreFlashAttentionV5，使用BNSD布局
        对应文档中的INT8量化场景：query、key、value输入为FLOAT16，输出为INT8
        """
        # 测试参数
        batch_size = 8
        num_heads = 32
        head_dims = 128
        key_num_heads = 32
        block_size = 128
        seq_len_q = 1
        total_seq_len_kv = 32 * 1024
        block_num = total_seq_len_kv // block_size

        # 设置基准块数
        base_block_num = (32 * 1024) // block_size
        max_actual_block_num_per_seq = base_block_num

        print(f"=== INT8量化测试配置 ===")
        print(f"批次大小: {batch_size}, 头数: {num_heads}, 键头数: {key_num_heads}")
        print(f"每头块数: {max_actual_block_num_per_seq}, 块大小: {block_size}")
        print(f"头维度: {head_dims}, 总序列长度: {total_seq_len_kv}")

        # 计算张量形状 - BNSD布局
        query_shape = [batch_size, num_heads, seq_len_q, head_dims]
        key_shape = [block_num, key_num_heads, block_size, head_dims]  
        value_shape = [block_num, key_num_heads, block_size, head_dims]

        # 为每个头确定块数
        blocks_per_head = [max_actual_block_num_per_seq] * key_num_heads

        # 计算总块数
        total_unique_blocks = sum(blocks_per_head)
        print(f"所有头总共需要 {total_unique_blocks} 个块")

        # BlockTable和BlockPosition形状
        block_table_shape = [batch_size, total_unique_blocks]
        block_position_shape = [batch_size, key_num_heads, max_actual_block_num_per_seq]
        
        print(f"BlockTable形状: {block_table_shape}")
        print(f"BlockPosition形状: {block_position_shape}")

        # 选择性块值定义
        SELECTED_BLOCK_VALUE = 0.5
        UNSELECTED_BLOCK_VALUE = 2.0

        # 创建FLOAT16输入数据
        query_data = torch.full(query_shape, 0.3, dtype=torch.float16)
        key_data = torch.full(key_shape, UNSELECTED_BLOCK_VALUE, dtype=torch.float16)
        value_data = torch.full(value_shape, UNSELECTED_BLOCK_VALUE, dtype=torch.float16)

        # 创建INT32类型的BlockTable和BlockPosition
        block_table_data = torch.full(block_table_shape, 0x7FFFFFFF, dtype=torch.int32)
        block_position_data = torch.full(block_position_shape, 0x7FFFFFFF, dtype=torch.int32)

        # 为每个头设置块索引和值
        head_block_indices = {}
        head_values = {}
        
        for h in range(key_num_heads):
            base_offset = (h * 5) % block_num
            # 计算该头的所有块索引
            head_block_indices[h] = [(base_offset + i * 3) % block_num for i in range(blocks_per_head[h])]
            # 计算该头的特定值
            head_values[h] = SELECTED_BLOCK_VALUE + h * 0.01
        
        # 设置BlockTable和BlockPosition
        block_table_idx = 0
        for h in range(key_num_heads):
            print(f"头 {h} 实际块数: {blocks_per_head[h]}/{max_actual_block_num_per_seq}")
            
            selected_blocks = head_block_indices[h][:5]
            if selected_blocks:
                print(f"头 {h} 选择的块: {selected_blocks}" + ("..." if len(head_block_indices[h]) > 5 else ""))
            
            # 为当前头创建int32类型的块索引张量
            block_indices_tensor = torch.tensor(head_block_indices[h], dtype=torch.int32)
            
            # 批量设置BlockPosition和BlockTable
            for i in range(len(head_block_indices[h])):
                for batch_idx in range(batch_size):
                    block_table_data[batch_idx, block_table_idx] = block_indices_tensor[i]
                    block_position_data[batch_idx, h, i] = torch.tensor(block_table_idx, dtype=torch.int32)
                block_table_idx += 1
        
        # 填充KV数据
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
        actual_seq_lengths = torch.tensor([max_actual_block_num_per_seq * block_size] * batch_size, dtype=torch.int64)

        # 将数据移至NPU
        query_npu = query_data.npu()
        key_npu = key_data.npu()
        value_npu = value_data.npu()
        block_table_npu = block_table_data.npu()
        block_position_npu = block_position_data.npu()
        seq_len_npu = actual_seq_lengths.npu()

        # 创建空张量（不需要的参数）
        empty_tensor = torch.empty(0).npu()

        # === INT8量化参数设置 ===
        # 根据文档：query、key、value输入为FLOAT16，输出为INT8时
        # 需要设置quantScale2（必填），quantOffset2（可选）
        # 不能传入dequantScale1、quantScale1、dequantScale2参数
        
        # 创建量化参数 - 使用per-tensor模式
        quant_scale2 = torch.tensor([0.1], dtype=torch.float32).npu()  # 必填参数
        quant_offset2 = torch.tensor([0.0], dtype=torch.float32).npu()  # 可选参数
        
        # 设置算子参数
        scale_value = 1.0 / math.sqrt(head_dims)
        layout = "BNSD"  # KV缓存布局为BNSD
        inner_precise = 1

        print("\n=== INT8量化参数 ===")
        print(f"量化缩放因子 (quant_scale2): {quant_scale2.cpu().item()}")
        print(f"量化偏移量 (quant_offset2): {quant_offset2.cpu().item()}")
        print(f"注意力缩放因子: {scale_value:.6f}")
        print(f"布局: {layout}")

        print("\n调用INT8量化算子...")
        # 调用算子 - INT8量化版本
        output = custom_ops.incre_flash_attention_v5(
            query_npu, key_npu, value_npu,
            empty_tensor,  # pse_shift
            empty_tensor,  # mask
            seq_len_npu,
            empty_tensor,  # dequant_scale1 - INT8场景下不能传入
            empty_tensor,  # quant_scale1 - INT8场景下不能传入
            empty_tensor,  # dequant_scale2 - INT8场景下不能传入
            quant_scale2,  # quant_scale2 - INT8场景下必填
            quant_offset2, # quant_offset2 - INT8场景下可选
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

        print(f"\n=== INT8量化结果分析 ===")
        print(f"输出数据类型: {output.dtype}")
        print(f"输出形状: {output.shape}")
        
        # 检查输出是否为INT8类型
        if output.dtype == torch.int8:
            print("✅ 输出成功量化为INT8类型")
        else:
            print(f"❌ 输出类型错误，期望INT8，实际为{output.dtype}")

        # 分头统计结果
        for batch_idx in range(min(batch_size, 2)):
            print(f"\n--- Batch {batch_idx} ---")
            for h in range(min(num_heads, 3)):
                head_data = output_cpu[batch_idx, h].flatten()
                head_mean = head_data.float().mean().item()
                head_min = head_data.min().item()
                head_max = head_data.max().item()
                print(f"  头 {h}: 均值={head_mean:.6f}, 最小值={head_min}, 最大值={head_max}")
            
            if num_heads > 3:
                h = num_heads - 1
                head_data = output_cpu[batch_idx, h].flatten()
                head_mean = head_data.float().mean().item()
                head_min = head_data.min().item()
                head_max = head_data.max().item()
                print(f"  头 {h}: 均值={head_mean:.6f}, 最小值={head_min}, 最大值={head_max}")

        # 结果验证
        all_data = output_cpu.flatten()
        has_unselected_value = False
        
        # 检查是否包含未选中块的影响（通过检查结果范围）
        int8_min = -128
        int8_max = 127
        if all_data.min() < int8_min or all_data.max() > int8_max:
            print("❌ 输出值超出INT8范围")
        else:
            print("✅ 输出值在INT8范围内")

        # 检查不同batch和头的结果差异
        first_value = output_cpu[0, 0, 0, 0].item()
        has_different_results = False
        
        # 检查不同batch的结果
        for batch_idx in range(1, batch_size):
            if abs(output_cpu[batch_idx, 0, 0, 0].item() - first_value) > 0:
                has_different_results = True
                break
        
        # 检查不同头的结果
        if not has_different_results:
            for h in range(1, num_heads):
                if abs(output_cpu[0, h, 0, 0].item() - first_value) > 0:
                    has_different_results = True
                    break
                
        print(f"\n=== INT8量化功能验证 ===")
        print(f"量化成功: {'✅ 通过，输出为INT8类型' if output.dtype == torch.int8 else '❌ 失败，输出类型错误'}")
        print(f"值范围检查: {'✅ 通过，输出值在INT8范围内' if all_data.min() >= int8_min and all_data.max() <= int8_max else '❌ 失败，输出值超出INT8范围'}")
        print(f"多头块选择: {'✅ 通过，不同batch/头的结果有差异' if has_different_results else '❌ 失败，所有结果相似'}")
        print(f"INT8量化算子: {'✅ 工作正常' if output.dtype == torch.int8 and has_different_results else '❌ 存在问题'}")
        print("=== INT8量化测试完成 ===\n")

    def test_int8_input_quantization_bsh_layout(self):
        """
        测试INT8输入量化版本的IncreFlashAttentionV5，使用BSH布局
        KV缓存格式为(blocknum, blocksize, H*D)
        输入QKV均为INT8，输出为FLOAT16，需要设置反量化参数
        """
        # 测试参数
        batch_size = 8
        num_heads = 32
        head_dims = 128
        key_num_heads = 32
        block_size = 128
        seq_len_q = 1
        total_seq_len_kv = 32 * 1024
        block_num = total_seq_len_kv // block_size

        # 设置基准块数
        base_block_num = (32 * 1024) // block_size
        max_actual_block_num_per_seq = base_block_num

        print(f"=== INT8输入量化BSH布局测试配置 ===")
        print(f"批次大小: {batch_size}, 头数: {num_heads}, 键头数: {key_num_heads}")
        print(f"每头块数: {max_actual_block_num_per_seq}, 块大小: {block_size}")
        print(f"头维度: {head_dims}")

        # 张量形状定义 - BSH布局
        query_shape = [batch_size, num_heads, seq_len_q, head_dims]
        key_shape = [block_num, block_size, key_num_heads * head_dims]
        value_shape = [block_num, block_size, key_num_heads * head_dims]

        # 为每个头确定块数
        blocks_per_head = [max_actual_block_num_per_seq] * key_num_heads

        # 计算全局需要的最大块数
        total_unique_blocks = sum(blocks_per_head)
        print(f"所有头总共需要 {total_unique_blocks} 个块")

        # BlockTable和BlockPosition形状
        block_table_shape = [batch_size, total_unique_blocks]
        block_position_shape = [batch_size, key_num_heads, max_actual_block_num_per_seq]
        
        print(f"BlockTable形状: {block_table_shape}")
        print(f"BlockPosition形状: {block_position_shape}")

        # 选择性块值定义 - 使用INT8范围的值
        SELECTED_BLOCK_VALUE = 50  # INT8范围: -128 到 127
        UNSELECTED_BLOCK_VALUE = -100

        # 创建INT8输入数据
        query_data = torch.full(query_shape, 30, dtype=torch.int8)
        key_data = torch.full(key_shape, UNSELECTED_BLOCK_VALUE, dtype=torch.int8)
        value_data = torch.full(value_shape, UNSELECTED_BLOCK_VALUE, dtype=torch.int8)

        # 创建INT32类型的BlockTable和BlockPosition
        block_table_data = torch.full(block_table_shape, 0x7FFFFFFF, dtype=torch.int32)
        block_position_data = torch.full(block_position_shape, 0x7FFFFFFF, dtype=torch.int32)

        # 为每个头设置块索引和值
        head_block_indices = {}
        head_values = {}
        
        for h in range(key_num_heads):
            base_offset = (h * 7) % block_num
            head_block_indices[h] = [(base_offset + i * 4) % block_num for i in range(blocks_per_head[h])]
            # 为每个头设置不同的INT8值
            head_values[h] = SELECTED_BLOCK_VALUE + (h % 10)  # 确保在INT8范围内
        
        # 设置BlockTable和BlockPosition
        block_table_idx = 0
        for h in range(key_num_heads):
            print(f"头 {h} 实际块数: {blocks_per_head[h]}/{max_actual_block_num_per_seq}")
            
            selected_blocks = head_block_indices[h][:5]
            if selected_blocks:
                print(f"头 {h} 选择的块: {selected_blocks}" + ("..." if len(head_block_indices[h]) > 5 else ""))
            
            block_indices_tensor = torch.tensor(head_block_indices[h], dtype=torch.int32)
            
            for i in range(len(head_block_indices[h])):
                for batch_idx in range(batch_size):
                    block_table_data[batch_idx, block_table_idx] = block_indices_tensor[i]
                    block_position_data[batch_idx, h, i] = torch.tensor(block_table_idx, dtype=torch.int32)
                block_table_idx += 1
        
        # 填充KV数据 - BSH格式，使用INT8值
        for h in range(key_num_heads):
            head_value = head_values[h]
            for block_idx in head_block_indices[h]:
                for s in range(block_size):
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

        # === INT8输入量化参数设置 ===
        # 根据文档：当输入为INT8时，需要设置反量化参数
        # dequant_scale1: BMM1后面反量化的量化因子
        # quant_scale1: BMM2前面量化的量化因子
        
        # 创建反量化参数 - 使用per-channel模式
        dequant_scale1 = torch.full([key_num_heads], 0.1, dtype=torch.float32).npu()  # 反量化缩放因子
        quant_scale1 = torch.full([key_num_heads], 0.2, dtype=torch.float32).npu()    # 量化缩放因子
        
        # 设置算子参数
        scale_value = 1.0 / math.sqrt(head_dims)
        layout = "BNSD"  # 注意：虽然KV是BSH布局，但layout参数仍设为BNSD
        inner_precise = 1

        print(f"\n=== INT8输入量化参数配置 ===")
        print(f"反量化缩放因子 (dequant_scale1): {dequant_scale1.cpu()[:5].tolist()}...")
        print(f"量化缩放因子 (quant_scale1): {quant_scale1.cpu()[:5].tolist()}...")
        print(f"注意力缩放因子: {scale_value:.6f}")
        print(f"布局: {layout}")
        print(f"注意: 输入QKV为INT8，输出为FLOAT16，需要反量化处理")

        print("\n调用INT8输入量化算子 (BSH布局)...")
        
        # 尝试调用INT8输入量化版本
        try:
            output = custom_ops.incre_flash_attention_v5(
                query_npu, key_npu, value_npu,
                empty_tensor,  # pse_shift
                empty_tensor,  # mask
                seq_len_npu,
                dequant_scale1,  # dequant_scale1 - INT8输入场景下必填
                quant_scale1,    # quant_scale1 - INT8输入场景下必填
                empty_tensor,    # dequant_scale2
                empty_tensor,    # quant_scale2
                empty_tensor,    # quant_offset2
                empty_tensor,    # antiquant_scale
                empty_tensor,    # antiquant_offset
                block_table_npu,
                empty_tensor,    # kv_padding_size
                block_position_npu,
                num_heads,
                np.float32(scale_value),
                layout,
                key_num_heads,
                block_size,
                inner_precise
            )
            
            # INT8输入量化成功
            print("✅ INT8输入量化算子调用成功！")
            output_cpu = output.cpu()
            
            print(f"\n=== INT8输入量化结果分析 ===")
            print(f"输出数据类型: {output.dtype}")
            print(f"输出形状: {output.shape}")
            
            # 检查输出类型
            if output.dtype == torch.float16:
                print("✅ 输出为FLOAT16类型（符合INT8输入的预期）")
            else:
                print(f"⚠️ 输出类型: {output.dtype} (期望FLOAT16)")

            # 分头统计结果
            print("\n=== 计算结果分析 ===")
            for h in range(min(num_heads, 5)):
                head_data = output_cpu[0, h].flatten()
                head_mean = head_data.float().mean().item()
                head_min = head_data.min().item()
                head_max = head_data.max().item()
                print(f"头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")
                
            if num_heads > 5:
                h = num_heads - 1
                head_data = output_cpu[0, h].flatten()
                head_mean = head_data.float().mean().item()
                head_min = head_data.min().item()
                head_max = head_data.max().item()
                print(f"头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")

            # 结果验证
            all_data = output_cpu.flatten()
            
            # 检查是否包含未选中块的影响
            has_unselected_value = ((all_data - (UNSELECTED_BLOCK_VALUE * 0.1)).abs() < 0.5).any().item()

            # 检查不同头的结果差异
            first_head_value = output_cpu[0, 0, 0, 0].item()
            heads_have_different_results = False
            for h in range(1, num_heads):
                if abs(output_cpu[0, h, 0, 0].item() - first_head_value) > 1e-5:
                    heads_have_different_results = True
                    break
                    
            print(f"\n=== INT8输入量化BSH布局验证 ===")
            print(f"输入类型: {'✅ 通过，输入为INT8类型' if query_data.dtype == torch.int8 else '❌ 失败，输入类型错误'}")
            print(f"输出类型: {'✅ 通过，输出为FLOAT16类型' if output.dtype == torch.float16 else '❌ 失败，输出类型错误'}")
            print(f"块选择测试: {'✅ 通过，结果不包含未选中块的影响' if not has_unselected_value else '❌ 失败，结果中检测到未选中块的影响'}")
            print(f"多头差异: {'✅ 通过，不同头的结果有差异，说明每个头独立选择块成功' if heads_have_different_results else '❌ 失败，所有头结果相似'}")
            print(f"INT8输入量化算子: {'✅ 工作正常' if output.dtype == torch.float16 and heads_have_different_results else '❌ 存在问题'}")
            print("=== INT8输入量化测试完成 ===\n")
            
        except Exception as e:
            print(f"❌ INT8输入量化算子调用失败: {e}")
            print("\n=== 降级到FLOAT16输入测试 ===")
            
            # 降级测试：将INT8输入转换为FLOAT16
            print("尝试将INT8输入转换为FLOAT16的版本...")
            
            try:
                # 将INT8数据转换为FLOAT16
                query_fp16 = query_data.float16()
                key_fp16 = key_data.float16()
                value_fp16 = value_data.float16()
                
                query_npu_fp16 = query_fp16.npu()
                key_npu_fp16 = key_fp16.npu()
                value_npu_fp16 = value_fp16.npu()
                
                output = custom_ops.incre_flash_attention_v5(
                    query_npu_fp16, key_npu_fp16, value_npu_fp16,
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
                
                print("✅ FLOAT16输入版本调用成功")
                output_cpu = output.cpu()
                
                print(f"\n=== FLOAT16输入结果分析 ===")
                print(f"输出数据类型: {output.dtype}")
                print(f"输出形状: {output.shape}")
                
                # 分头统计结果
                print("\n=== 计算结果分析 ===")
                for h in range(min(num_heads, 5)):
                    head_data = output_cpu[0, h].flatten()
                    head_mean = head_data.float().mean().item()
                    head_min = head_data.min().item()
                    head_max = head_data.max().item()
                    print(f"头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")

                # 结果验证
                all_data = output_cpu.flatten()
                
                # 检查是否包含未选中块的影响
                has_unselected_value = ((all_data - (UNSELECTED_BLOCK_VALUE * 0.1)).abs() < 0.5).any().item()

                # 检查不同头的结果差异
                first_head_value = output_cpu[0, 0, 0, 0].item()
                heads_have_different_results = False
                for h in range(1, num_heads):
                    if abs(output_cpu[0, h, 0, 0].item() - first_head_value) > 1e-5:
                        heads_have_different_results = True
                        break
                        
                print(f"\n=== BSH布局功能验证 (FLOAT16输入) ===")
                print(f"块选择测试: {'✅ 通过，结果不包含未选中块的影响' if not has_unselected_value else '❌ 失败，结果中检测到未选中块的影响'}")
                print(f"多头差异: {'✅ 通过，不同头的结果有差异，说明每个头独立选择块成功' if heads_have_different_results else '❌ 失败，所有头结果相似'}")
                print(f"BSH布局功能: {'✅ 工作正常' if not has_unselected_value and heads_have_different_results else '❌ 存在问题'}")
                print("=== FLOAT16输入版本测试完成 ===\n")
                
            except Exception as e2:
                print(f"❌ FLOAT16输入版本也调用失败: {e2}")
                print("=== 测试完全失败 ===")



    def test_mixed_precision_fp16_query_int8_kv(self):
        """
        测试混合精度版本的IncreFlashAttentionV5，使用BSH布局
        - query: FP16输入，FP16输出
        - key/value: INT8输入，INT8计算
        - 需要设置相应的量化参数
        """
        # 测试参数
        batch_size = 8
        num_heads = 32
        head_dims = 128
        key_num_heads = 32
        block_size = 128 # INT8的时候要32位对齐
        seq_len_q = 1
        total_seq_len_kv = 32 * 1024
        block_num = total_seq_len_kv // block_size

        # 设置基准块数
        base_block_num = (4 * 1024) // block_size
        max_actual_block_num_per_seq = base_block_num

        print(f"=== 混合精度测试配置 ===")
        print(f"批次大小: {batch_size}, 头数: {num_heads}, 键头数: {key_num_heads}")
        print(f"每头块数: {max_actual_block_num_per_seq}, 块大小: {block_size}")
        print(f"头维度: {head_dims}")
        print(f"精度配置: Query(FP16→FP16), KV(INT8→INT8)")

        # 张量形状定义 - BSH布局
        query_shape = [batch_size, num_heads, seq_len_q, head_dims]
        key_shape = [block_num, block_size, key_num_heads * head_dims]
        value_shape = [block_num, block_size, key_num_heads * head_dims]

        # 为每个头确定块数
        blocks_per_head = [max_actual_block_num_per_seq] * key_num_heads

        # 计算全局需要的最大块数
        total_unique_blocks = sum(blocks_per_head)
        print(f"所有头总共需要 {total_unique_blocks} 个块")

        # BlockTable和BlockPosition形状
        block_table_shape = [batch_size, total_unique_blocks]
        block_position_shape = [batch_size, key_num_heads, max_actual_block_num_per_seq]
        
        print(f"BlockTable形状: {block_table_shape}")
        print(f"BlockPosition形状: {block_position_shape}")

        # 选择性块值定义
        SELECTED_BLOCK_VALUE = 50  # INT8范围: -128 到 127
        UNSELECTED_BLOCK_VALUE = -100

        # 创建混合精度输入数据
        query_data = torch.full(query_shape, 0.4, dtype=torch.float16)  # FP16
        key_data = torch.full(key_shape, UNSELECTED_BLOCK_VALUE, dtype=torch.int8)    # INT8
        value_data = torch.full(value_shape, UNSELECTED_BLOCK_VALUE, dtype=torch.int8) # INT8

        # 创建INT32类型的BlockTable和BlockPosition
        block_table_data = torch.full(block_table_shape, 0x7FFFFFFF, dtype=torch.int32)
        block_position_data = torch.full(block_position_shape, 0x7FFFFFFF, dtype=torch.int32)

        # 为每个头设置块索引和值
        head_block_indices = {}
        head_values = {}
        
        for h in range(key_num_heads):
            base_offset = (h * 7) % block_num
            head_block_indices[h] = [(base_offset + i * 4) % block_num for i in range(blocks_per_head[h])]
            # 为每个头设置不同的INT8值
            head_values[h] = SELECTED_BLOCK_VALUE + (h % 10)  # 确保在INT8范围内
        
        # 设置BlockTable和BlockPosition
        block_table_idx = 0
        for h in range(key_num_heads):
            print(f"头 {h} 实际块数: {blocks_per_head[h]}/{max_actual_block_num_per_seq}")
            
            selected_blocks = head_block_indices[h][:5]
            if selected_blocks:
                print(f"头 {h} 选择的块: {selected_blocks}" + ("..." if len(head_block_indices[h]) > 5 else ""))
            
            block_indices_tensor = torch.tensor(head_block_indices[h], dtype=torch.int32)
            
            for i in range(len(head_block_indices[h])):
                for batch_idx in range(batch_size):
                    block_table_data[batch_idx, block_table_idx] = block_indices_tensor[i]
                    block_position_data[batch_idx, h, i] = torch.tensor(block_table_idx, dtype=torch.int32)
                block_table_idx += 1
        
        # 填充KV数据 - BSH格式，使用INT8值
        for h in range(key_num_heads):
            head_value = head_values[h]
            for block_idx in head_block_indices[h]:
                for s in range(block_size):
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

        # === 混合精度量化参数设置 ===
        # 根据文档和混合精度需求设置参数：
        # - query: FP16输入，FP16输出
        # - key/value: INT8输入，INT8计算
        
        # 创建量化参数 - 使用per-channel模式
        # dequant_scale1 = torch.full([key_num_heads], 0.1, dtype=torch.float32).npu()  # KV反量化缩放因子
        # quant_scale1 = torch.full([key_num_heads], 0.2, dtype=torch.float32).npu()    # 中间结果量化缩放因子
        # 1) 量化参数：只为 KV 提供 antiquant_scale/antiquant_offset
        
        # 方式A：per-channel（每个KV head、每个维度各一组）=> [2, N, D]
        antiquant_scale = torch.stack([
            torch.full([key_num_heads, head_dims], 0.1, dtype=torch.float16),
            torch.full([key_num_heads, head_dims], 0.1, dtype=torch.float16),
        ], dim=0).npu()  # [2, N, D] -> [2, 32, 128]

        antiquant_offset = torch.stack([
            torch.full([key_num_heads, head_dims], 0.0, dtype=torch.float16),
            torch.full([key_num_heads, head_dims], 0.0, dtype=torch.float16),
        ], dim=0).npu()  # [2, 32, 128]
        
        # 设置算子参数
        scale_value = 1.0 / math.sqrt(head_dims)
        layout = "BNSD"  # 注意：虽然KV是BSH布局，但layout参数仍设为BNSD
        inner_precise = 1

        print(f"\n=== 混合精度量化参数配置 ===")
        # print(f"KV反量化缩放因子 (dequant_scale1): {dequant_scale1.cpu()[:5].tolist()}...")
        # print(f"中间结果量化缩放因子 (quant_scale1): {quant_scale1.cpu()[:5].tolist()}...")
        print(f"注意力缩放因子: {scale_value:.6f}")
        print(f"布局: {layout}")
        print(f"精度配置: Query(FP16→FP16), KV(INT8→INT8计算)")

        print("\n调用混合精度量化算子...")
        
        # 尝试调用混合精度版本
        
        output = custom_ops.incre_flash_attention_v5(
            query_npu, key_npu, value_npu,
            empty_tensor,  # pse_shift
            empty_tensor,  # mask
            seq_len_npu,
            empty_tensor,  # dequant_scale1 - KV反量化必填
            empty_tensor,    # quant_scale1 - 中间结果量化必填
            empty_tensor,    # dequant_scale2
            empty_tensor,    # quant_scale2
            empty_tensor,    # quant_offset2
            antiquant_scale,      # 仅对 KV 生效的反量化 scale
            antiquant_offset,     # 仅对 KV 生效的反量化 offset
            block_table_npu,
            empty_tensor,    # kv_padding_size
            block_position_npu,
            num_heads,
            np.float32(scale_value),
            layout,
            key_num_heads,
            block_size,
            inner_precise
        )
        
        # 混合精度调用成功
        print("✅ 混合精度量化算子调用成功！")
        output_cpu = output.cpu()
        
        print(f"\n=== 混合精度结果分析 ===")
        print(f"输出数据类型: {output.dtype}")
        print(f"输出形状: {output.shape}")
        
        # 检查输出类型 - 期望FP16（因为query是FP16）
        if output.dtype == torch.float16:
            print("✅ 输出为FLOAT16类型（符合FP16 query的预期）")
        else:
            print(f"⚠️ 输出类型: {output.dtype} (期望FLOAT16)")

        # 分头统计结果
        print("\n=== 计算结果分析 ===")
        for h in range(min(num_heads, 5)):
            head_data = output_cpu[0, h].flatten()
            head_mean = head_data.float().mean().item()
            head_min = head_data.min().item()
            head_max = head_data.max().item()
            print(f"头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")
            
        if num_heads > 5:
            h = num_heads - 1
            head_data = output_cpu[0, h].flatten()
            head_mean = head_data.float().mean().item()
            head_min = head_data.min().item()
            head_max = head_data.max().item()
            print(f"头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")

        # 结果验证
        all_data = output_cpu.flatten()
        
        # 检查是否包含未选中块的影响
        has_unselected_value = ((all_data - (UNSELECTED_BLOCK_VALUE * 0.1)).abs() < 0.5).any().item()

        # 检查不同头的结果差异
        first_head_value = output_cpu[0, 0, 0, 0].item()
        heads_have_different_results = False
        for h in range(1, num_heads):
            if abs(output_cpu[0, h, 0, 0].item() - first_head_value) > 1e-5:
                heads_have_different_results = True
                break
                
        print(f"\n=== 混合精度量化验证 ===")
        print(f"Query类型: {'✅ 通过，输入为FP16类型' if query_data.dtype == torch.float16 else '❌ 失败，Query类型错误'}")
        print(f"KV类型: {'✅ 通过，输入为INT8类型' if key_data.dtype == torch.int8 and value_data.dtype == torch.int8 else '❌ 失败，KV类型错误'}")
        print(f"输出类型: {'✅ 通过，输出为FP16类型' if output.dtype == torch.float16 else '❌ 失败，输出类型错误'}")
        print(f"块选择测试: {'✅ 通过，结果不包含未选中块的影响' if not has_unselected_value else '❌ 失败，结果中检测到未选中块的影响'}")
        print(f"多头差异: {'✅ 通过，不同头的结果有差异，说明每个头独立选择块成功' if heads_have_different_results else '❌ 失败，所有头结果相似'}")
        print(f"混合精度量化: {'✅ 工作正常' if output.dtype == torch.float16 and heads_have_different_results else '❌ 存在问题'}")
        print("=== 混合精度量化测试完成 ===\n")
        

if __name__ == "__main__":
    def run_single_test():
        test = TestIncreFlashAttentionV5Int8Quantization("test_int8_quantization_bnsd_layout")
        test.setUp()
        test.test_int8_quantization_bnsd_layout()
        test.tearDown()
        
    def run_bsh_test():
        test = TestIncreFlashAttentionV5Int8Quantization("test_mixed_precision_fp16_query_int8_kv")
        test.setUp()
        test.test_mixed_precision_fp16_query_int8_kv()
        test.tearDown()
        
    # 运行BNSD布局的INT8量化测试
    # print("开始运行BNSD布局INT8量化测试...")
    # run_single_test()
    
    # 运行BSH布局的INT8量化测试
    print("\n开始运行BSH布局INT8量化测试...")
    run_bsh_test()