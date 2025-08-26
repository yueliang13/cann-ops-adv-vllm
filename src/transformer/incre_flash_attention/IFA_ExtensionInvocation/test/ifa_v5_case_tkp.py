# test_add_custom.py
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops # pybind导入麻烦
import copy
import math
import numpy as np # 导入 numpy 来进行精确的类型转换

class TestIncreFlashAttentionV5(TestCase):
   
    def test_block_size_head_layout(self):
        """
        测试Token Position布局模式 - Token级别选择
        从原来的Block级别选择改为精确的Token级别选择
        KV缓存格式仍为(blocknum, blocksize, H*D)，但现在可以精确选择每个Token
        """
        # 测试参数
        batch_size = 8
        b = 256
        num_heads = 32
        head_dims = 128
        key_num_heads = 8
        block_size = 128
        seq_len_q = 1
        total_seq_len_kv = 32 * 1024
        block_num = total_seq_len_kv // block_size

        # 设置基准Token数量 - 现在以Token为单位而非Block
        base_token_num = 4 * 1024  # 每个头选择的Token数量
        max_tokens_per_head = base_token_num  # 每个头最大Token数

        print(f"=== Token Position 模式测试 ===")
        print(f"批次大小: {batch_size}, 头数: {num_heads}, 键头数: {key_num_heads}")
        print(f"每头Token数: {max_tokens_per_head}, 块大小: {block_size}")
        print(f"总序列长度: {total_seq_len_kv}, 总块数: {block_num}")
        print(f"所有头使用相同的Token数: {base_token_num}")

        # 张量形状定义 - 注意KV形状为(blocknum, blocksize, H*D)
        query_shape = [batch_size, num_heads, seq_len_q, head_dims]
        key_shape = [block_num, block_size, key_num_heads * head_dims]
        value_shape = [block_num, block_size, key_num_heads * head_dims]
        out_shape = [batch_size, num_heads, seq_len_q, head_dims]

        # Token级别的形状定义
        max_seq_len = max_tokens_per_head  # 最大序列长度等于每头最大Token数
        
        # 计算需要的BlockTable大小 - 基于可能用到的所有页
        estimated_blocks_needed = (max_tokens_per_head * key_num_heads) // block_size + key_num_heads  # 估算+冗余
        actual_blocks_needed = min(estimated_blocks_needed, block_num)  # 不能超过总块数
        
        # BlockTable和TokenPosition形状
        block_table_shape = [b, actual_blocks_needed]
        token_position_shape = [batch_size, key_num_heads, max_seq_len]  # Token级别的位置
        
        print(f"BlockTable形状: {block_table_shape}")
        print(f"TokenPosition形状 (原BlockPosition): {token_position_shape}")

        print("\n张量形状信息:")
        print(f"  Query Shape (BNSD): {query_shape}")
        print(f"  Key Shape (blocknum, blocksize, H*D): {key_shape}")
        print(f"  Value Shape (blocknum, blocksize, H*D): {value_shape}")
        print(f"  Output Shape: {out_shape}")
        print(f"  每头最大Token数: {max_tokens_per_head}")

        # 选择性Token值定义
        SELECTED_TOKEN_VALUE = 0.1
        UNSELECTED_TOKEN_VALUE = 9.9

        # 创建数据
        query_data = torch.full(query_shape, 0.1, dtype=torch.float16)
        key_data = torch.full(key_shape, UNSELECTED_TOKEN_VALUE, dtype=torch.float16)
        value_data = torch.full(value_shape, UNSELECTED_TOKEN_VALUE, dtype=torch.float16)

        # BlockTable和TokenPosition数据 - 必须是int32!!!
        block_table_data = torch.full(block_table_shape, 0x7FFFFFFF, dtype=torch.int32)
        token_position_data = torch.full(token_position_shape, 0x7FFFFFFF, dtype=torch.int32)

        print("\n=== Token级别选择策略 ===")
        
        # 预先计算每个头的Token位置和值
        head_token_positions = {}  # 每个头选择的Token位置列表
        head_values = {}
        head_selected_token_count = {}  # 每个头选择的Token数量

        for h in range(key_num_heads):
            # 每个头选择的Token数量（可以不同，这里为了简单都设为相同）
            tokens_for_head = max_tokens_per_head
            head_selected_token_count[h] = min(tokens_for_head, max_seq_len)
            
            # 生成该头的Token位置 - 稀疏选择模式
            base_offset = (h * 17) % total_seq_len_kv  # 每个头不同的起始位置
            token_positions = []
            
            # 创建稀疏Token选择模式：
            # - 前部分：连续选择（模拟窗口注意力）
            # - 中部分：间隔选择（模拟稀疏注意力）
            # - 后部分：随机选择（模拟全局重要Token）
            
            front_tokens = min(512, head_selected_token_count[h] // 3)  # 前1/3连续
            middle_tokens = min(1024, head_selected_token_count[h] // 3)  # 中1/3间隔
            back_tokens = head_selected_token_count[h] - front_tokens - middle_tokens  # 后1/3随机
            
            current_pos = base_offset
            
            # 前部分：连续Token
            for i in range(front_tokens):
                token_pos = (current_pos + i) % total_seq_len_kv
                token_positions.append(token_pos)
            current_pos += front_tokens
            
            # 中部分：间隔Token（每隔3个选1个）
            for i in range(middle_tokens):
                token_pos = (current_pos + i * 4) % total_seq_len_kv  # 每隔4个选1个
                token_positions.append(token_pos)
            current_pos += middle_tokens * 4
            
            # 后部分：随机间隔Token
            for i in range(back_tokens):
                # 使用质数步长创建伪随机模式
                token_pos = (current_pos + i * 7 + h * 13) % total_seq_len_kv
                token_positions.append(token_pos)
            
            # 去重并排序
            token_positions = sorted(list(set(token_positions)))
            
            # 如果去重后数量不足，补充更多Token
            while len(token_positions) < head_selected_token_count[h]:
                # 简单的补充策略
                new_pos = (base_offset + len(token_positions) * 5) % total_seq_len_kv
                if new_pos not in token_positions:
                    token_positions.append(new_pos)
                else:
                    break
            
            # 限制到指定数量
            token_positions = token_positions[:head_selected_token_count[h]]
            
            head_token_positions[h] = token_positions
            head_values[h] = SELECTED_TOKEN_VALUE + h * 0.001  # 每个头略有不同的值

        # 显示Token选择信息
        for h in range(min(key_num_heads, 5)):  # 只显示前5个头的信息
            selected_positions = head_token_positions[h]
            print(f"头 {h}: 选择 {len(selected_positions)} 个Token")
            
            # 显示前10个选择的Token位置
            preview_positions = selected_positions[:10]
            if preview_positions:
                print(f"  Token位置样例: {preview_positions}" + ("..." if len(selected_positions) > 10 else ""))

        print("\n=== 设置TokenPosition数据 ===")
        
        # 设置TokenPosition数据 - 为所有batch设置相同的数据
        for batch_idx in range(batch_size):
            for h in range(key_num_heads):
                selected_positions = head_token_positions[h]
                
                # 设置Token位置数据
                for seq_idx, token_pos in enumerate(selected_positions):
                    if seq_idx < max_seq_len:  # 确保不超出序列长度限制
                        # 为所有batch设置相同的Token位置
                        token_position_data[batch_idx, h, seq_idx] = torch.tensor(token_pos, dtype=torch.int32)

        print("\n=== 设置BlockTable（页表映射）===")
        
        # 收集所有需要的块
        used_blocks = set()
        for h in range(key_num_heads):
            for token_pos in head_token_positions[h]:
                block_idx = token_pos // block_size  # Token所在的块
                used_blocks.add(block_idx)

        # 为使用的块建立映射 - 为所有batch设置相同的映射
        block_table_idx = 0
        block_mapping = {}  # 逻辑块 -> 物理块的映射

        for block_idx in sorted(used_blocks):
            if block_table_idx < block_table_data.shape[1]:
                # 为所有batch设置相同的块映射
                for batch_idx in range(batch_size):
                    block_table_data[batch_idx, block_table_idx] = torch.tensor(block_idx, dtype=torch.int32)
                block_mapping[block_idx] = block_table_idx
                block_table_idx += 1

        print(f"使用的总块数: {len(used_blocks)} / {block_num}")
        if len(used_blocks) <= 10:
            print(f"使用的块: {sorted(used_blocks)}")
        else:
            sample_blocks = sorted(list(used_blocks))[:10]
            print(f"使用的块样例: {sample_blocks}...")

        print("\n=== 填充KV Cache数据（Token级别精确填充）===")

        total_tokens_filled = 0
        for h in range(key_num_heads):
            head_value = head_values[h]
            selected_count = 0
            
            for token_pos in head_token_positions[h]:
                # 将Token位置转换为BSH格式的具体索引
                block_idx = token_pos // block_size  # Token所在的块
                token_in_block = token_pos % block_size  # Token在块内的位置
                
                # BSH格式线性索引计算
                # 格式: [block_idx, token_in_block, head * head_dims : (head+1) * head_dims]
                start_idx = block_idx * block_size * key_num_heads * head_dims + \
                           token_in_block * key_num_heads * head_dims + \
                           h * head_dims
                end_idx = start_idx + head_dims
                
                if end_idx <= key_data.numel():
                    # 为选中的Token设置特殊值
                    key_data.view(-1)[start_idx:end_idx] = head_value
                    value_data.view(-1)[start_idx:end_idx] = head_value
                    selected_count += 1
                    total_tokens_filled += 1
            
            if h < 5 or h == key_num_heads - 1:  # 显示前5个头和最后1个头的信息
                print(f"头 {h}: 成功填充 {selected_count} 个Token的数据")
        
        print(f"总计填充Token数: {total_tokens_filled}")

        # 设置实际序列长度 - 基于Token级别，为所有batch设置相同的长度
        actual_token_counts = [len(head_token_positions[h]) for h in range(key_num_heads)]
        max_actual_tokens = max(actual_token_counts) if actual_token_counts else max_seq_len
        actual_seq_lengths = torch.tensor([max_actual_tokens] * batch_size, dtype=torch.int64)
        print(f"actual_seq_lengths:{actual_seq_lengths}")

        print(f"\n=== 序列长度信息 ===")
        print(f"各头Token数量范围: {min(actual_token_counts)} - {max(actual_token_counts)}")
        print(f"实际最大序列长度: {max_actual_tokens}")


        # 将数据移至NPU
        query_npu = query_data.npu()
        key_npu = key_data.npu()
        value_npu = value_data.npu()
        block_table_npu = block_table_data.npu()
        token_position_npu = token_position_data.npu()
        seq_len_npu = actual_seq_lengths.npu()

        # 创建空张量
        empty_tensor = torch.empty(0).npu()

        # 设置算子参数
        scale_value = 1.0 / math.sqrt(head_dims)
        layout = "BNSD"  # 注意：虽然KV是BSH布局，但layout参数仍设为BNSD
        inner_precise = 0

        print("调用算子...")
        # 调用算子
        print(f"query_npu shape:{query_npu.shape}")
        print(f"key_npu shape:{key_npu.shape}")
        print(f"value_npu shape:{value_npu.shape}")
        print(f"seq_len_npu shape:{seq_len_npu.shape},seq_len_npu:{seq_len_npu}")
        print(f"block_table_npu shape:{block_table_npu.shape}")
        print(f"token_position_npu shape:{token_position_npu.shape}")
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
            token_position_npu,
            num_heads,
            np.float32(scale_value),
            layout,
            key_num_heads,
            block_size,
            inner_precise
        )

        # 结果验证
        output_cpu = output.cpu()

        # 分头统计结果 - 检查所有batch的结果
        print("\n=== 计算结果分析 ===")
        for batch_idx in range(min(batch_size, 3)):  # 只显示前3个batch的结果
            print(f"--- Batch {batch_idx} ---")
            for h in range(min(num_heads, 3)):  # 只显示前3个头
                head_data = output_cpu[batch_idx, h].flatten()
                head_mean = head_data.mean().item()
                head_min = head_data.min().item()
                head_max = head_data.max().item()
                print(f"  头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")
            
            if num_heads > 3:
                h = num_heads - 1
                head_data = output_cpu[batch_idx, h].flatten()
                head_mean = head_data.mean().item()
                head_min = head_data.min().item()
                head_max = head_data.max().item()
                print(f"  头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")

        # 结果验证 - 检查所有batch
        all_data = output_cpu.flatten()
        has_unselected_value = ((all_data - UNSELECTED_TOKEN_VALUE).abs() < 1.0).any().item()

        # 检查不同batch和头的结果差异
        first_value = output_cpu[0, 0, 0, 0].item()
        has_different_results = False
        
        # 检查不同batch的结果
        for batch_idx in range(1, batch_size):
            if abs(output_cpu[batch_idx, 0, 0, 0].item() - first_value) > 1e-5:
                has_different_results = True
                break
        
        # 检查不同头的结果
        if not has_different_results:
            for h in range(1, num_heads):
                if abs(output_cpu[0, h, 0, 0].item() - first_value) > 1e-5:
                    has_different_results = True
                    break
                
        print(f"\n=== Block-Size-Head布局映射分析 ===")
        print(f"块选择测试: {'✅ 通过，结果不包含未选中块的影响' if not has_unselected_value else '❌ 失败，结果中检测到未选中块的影响'}")
        print(f"多batch/多头差异: {'✅ 通过，不同batch/头的结果有差异，说明选择机制成功' if has_different_results else '❌ 失败，所有结果相似，选择机制可能未生效'}")
        print(f"算子功能: {'✅ Block-Size-Head布局 (blocknum, blocksize, H) 工作正常' if not has_unselected_value and has_different_results else '❌ 布局机制可能存在问题，请检查实现'}")
        print("=== 结果分析完成 ===\n")

    def test_block_size_head_layout_sequential(self):
        batch_size = 8  # 改为支持多batch
        num_heads = 32
        head_dims = 128
        key_num_heads = 32
        block_size = 16
        seq_len_q = 1
        total_seq_len_kv = 32 * 1024
        block_num = total_seq_len_kv // block_size

        # 固定连续 Token 数 4096
        max_tokens_per_head = 4096
        assert max_tokens_per_head % block_size == 0
        needed_blocks_per_head = max_tokens_per_head // block_size  # 256
        # 简单场景: 各头使用同一批前 256 个块 (0..255)

        print("=== 连续 Token Position 测试 ===")
        print(f"批次大小: {batch_size}, 每头连续 token 数: {max_tokens_per_head}, 块数: {needed_blocks_per_head}")

        query_shape = [batch_size, num_heads, seq_len_q, head_dims]
        key_shape = [block_num, block_size, key_num_heads * head_dims]
        value_shape = [block_num, block_size, key_num_heads * head_dims]
        out_shape = [batch_size, num_heads, seq_len_q, head_dims]

        # 数据
        SELECTED_TOKEN_VALUE = 0.1
        BACKGROUND_VALUE = 9.9

        query_data = torch.full(query_shape, 0.1, dtype=torch.float16)
        key_data = torch.full(key_shape, BACKGROUND_VALUE, dtype=torch.float16)
        value_data = torch.full(value_shape, BACKGROUND_VALUE, dtype=torch.float16)

        # BlockTable 大小: 覆盖 256 块即可 (可按原逻辑估算, 这里直接设定)
        block_table_shape = [batch_size, needed_blocks_per_head]
        token_position_shape = [batch_size, key_num_heads, max_tokens_per_head]

        block_table_data = torch.full(block_table_shape, 0x7FFFFFFF, dtype=torch.int32)
        token_position_data = torch.full(token_position_shape, 0x7FFFFFFF, dtype=torch.int32)

        # 1. 填写连续 Token Position - 为所有batch设置相同的数据
        # token_position: 直接 0..4095
        base_token_range = torch.arange(max_tokens_per_head, dtype=torch.int32)
        for batch_idx in range(batch_size):
            for h in range(key_num_heads):
                token_position_data[batch_idx, h, :max_tokens_per_head] = base_token_range

        # 2. 填写 BlockTable (块索引 0..255) - 为所有batch设置相同的映射
        for batch_idx in range(batch_size):
            block_table_data[batch_idx, :needed_blocks_per_head] = torch.arange(needed_blocks_per_head, dtype=torch.int32)

        # 3. 填充 KV 中对应 token 的值 (BSH: [block_idx, s, head*D:(head+1)*D])
        for h in range(key_num_heads):
            head_value = SELECTED_TOKEN_VALUE + h * 0.001
            for token_id in range(max_tokens_per_head):
                block_idx = token_id // block_size          # 0..255
                token_in_block = token_id % block_size      # 0..15
                start = (block_idx * block_size * key_num_heads * head_dims
                        + token_in_block * key_num_heads * head_dims
                        + h * head_dims)
                end = start + head_dims
                key_data.view(-1)[start:end] = head_value
                value_data.view(-1)[start:end] = head_value

        # 为所有batch设置相同的序列长度
        actual_seq_lengths = torch.tensor([max_tokens_per_head] * batch_size, dtype=torch.int64)

        # 上 NPU
        q_npu = query_data.npu()
        k_npu = key_data.npu()
        v_npu = value_data.npu()
        block_table_npu = block_table_data.npu()
        token_position_npu = token_position_data.npu()
        seq_len_npu = actual_seq_lengths.npu()
        empty = torch.empty(0, device='npu')

        scale_value = float(1.0 / math.sqrt(head_dims))
        layout = "BNSD"
        inner_precise = 0

        print("调用算子 (连续 Token Position)...")
        # 注意: 如果当前 C++ schema 尚未加入 token_position 参数，请暂用占位空 tensor
        # 并在 C++ 中扩展；下面假设已经扩展，有 token_position 位置。
        output = custom_ops.incre_flash_attention_v5(
            q_npu,
            k_npu,
            v_npu,
            empty,       # pse_shift
            empty,       # mask
            seq_len_npu,
            empty, empty, empty, empty, empty, empty, empty,  # 量化相关占位
            block_table_npu,
            empty,       # kv_padding_size
            token_position_npu,  # 已扩展 schema 的位置
            num_heads,
            scale_value,
            layout,
            key_num_heads,
            block_size,
            inner_precise
        )

        torch.npu.synchronize()
        out_cpu = output.cpu()
        print("输出形状:", out_cpu.shape)
        
        # 结果验证和分析
        print("\n=== 连续Token选择结果分析 ===")
        
        # 检查所有batch的结果
        for batch_idx in range(min(batch_size, 3)):  # 只显示前3个batch
            print(f"--- Batch {batch_idx} ---")
            for h in range(min(num_heads, 3)):  # 只显示前3个头
                head_data = out_cpu[batch_idx, h].flatten()
                head_mean = head_data.mean().item()
                head_min = head_data.min().item()
                head_max = head_data.max().item()
                print(f"  头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")
            
            if num_heads > 3:
                h = num_heads - 1
                head_data = out_cpu[batch_idx, h].flatten()
                head_mean = head_data.mean().item()
                head_min = head_data.min().item()
                head_max = head_data.max().item()
                print(f"  头 {h}: 均值={head_mean:.6f}, 最小值={head_min:.6f}, 最大值={head_max:.6f}")
        
        # 验证结果
        all_data = out_cpu.flatten()
        has_background_value = ((all_data - BACKGROUND_VALUE).abs() < 1.0).any().item()
        
        # 检查不同头的结果差异
        first_value = out_cpu[0, 0, 0, 0].item()
        has_different_results = False
        for h in range(1, num_heads):
            if abs(out_cpu[0, h, 0, 0].item() - first_value) > 1e-5:
                has_different_results = True
                break
        
        print(f"\n=== 连续Token选择验证 ===")
        print(f"背景值过滤: {'✅ 通过，结果不包含背景值' if not has_background_value else '❌ 失败，结果中检测到背景值'}")
        print(f"多头差异: {'✅ 通过，不同头的结果有差异' if has_different_results else '❌ 失败，所有头结果相似'}")
        print(f"连续Token选择: {'✅ 连续Token选择机制工作正常' if not has_background_value and has_different_results else '❌ 连续Token选择机制可能存在问题'}")
        
        print("完成连续 Token Position 测试")

if __name__ == "__main__":
    def run_single_test():
        test = TestIncreFlashAttentionV5("test_block_size_head_layout")
        test.setUp()
        test.test_block_size_head_layout()
        test.tearDown()
        
    def run_sequential_test():
        test = TestIncreFlashAttentionV5("test_block_size_head_layout_sequential")
        test.setUp()
        test.test_block_size_head_layout_sequential()
        test.tearDown()
        
    def run_all_tests():
        print("=== 运行第一个测试：稀疏Token选择 ===")
        run_single_test()
        print("\n" + "="*80 + "\n")
        print("=== 运行第二个测试：连续Token选择 ===")
        run_sequential_test()
        
    # run_tests()  # 注释掉原来的运行所有测试的代码
    run_all_tests()  # 运行两个测试