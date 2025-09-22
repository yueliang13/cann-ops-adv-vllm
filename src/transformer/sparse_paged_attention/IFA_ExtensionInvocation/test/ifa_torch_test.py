import torch
import torch_npu
import math

# --- 1. 参数定义 (解码场景 Q=1, KV=32K, 启用 Page Attention) ---

batch_size = 1
num_heads = 32
head_dims = 128
num_key_value_heads = 32      # MHA 场景, 与 num_heads 相同
block_size = 128              # Page Attention 的块大小
seq_len_q = 1                 # 解码场景，Query 长度为 1
actual_seq_len_kv = 10 * 1024 # KV 缓存中的实际序列长度

# 为 block_table 计算最大所需块数，这里基于一个理论最大长度
max_supported_len = 10 * 1024 # 假设模型最大支持34K
max_blocks_per_seq = (max_supported_len + block_size - 1) // block_size

# 定义 KV Cache 内存池的总大小（总块数）
total_num_blocks = actual_seq_len_kv // block_size

print("--- Page Attention 场景参数 ---")
print(f"Batch Size: {batch_size}")
print(f"Query/KV Heads: {num_heads}/{num_key_value_heads}")
print(f"Head Dims: {head_dims}")
print(f"Query Seq Len: {seq_len_q}")
print(f"KV History Len: {actual_seq_len_kv}")
print(f"Block Size: {block_size}")
print("-------------------------------")

# --- 2. 构造输入张量 (严格遵循手册要求) ---

# Query (q): 必须使用 "BNSD" 布局
# B = Batch, N = Num Heads, S = Seq Len, D = Head Dims
# Shape: (batch_size, num_heads, seq_len_q, head_dims) -> (8, 32, 1, 128)
q = torch.randn(batch_size, num_heads, seq_len_q, head_dims, dtype=torch.float16).npu()

# Key Cache (key) & Value Cache (value): 使用高性能的 Page Attention 布局
# Shape: (total_num_blocks, num_key_value_heads, block_size, head_dims) -> (8192, 32, 128, 128)
key_cache = torch.randn(total_num_blocks, num_key_value_heads, block_size, head_dims, dtype=torch.float16).npu()
value_cache = torch.randn(total_num_blocks, num_key_value_heads, block_size, head_dims, dtype=torch.float16).npu()

# Block Table: 必须是 2D Tensor
# Shape: (batch_size, max_blocks_per_seq) -> (8, 272)
block_table = torch.randint(0, total_num_blocks, (batch_size, max_blocks_per_seq), dtype=torch.int32).npu()

# Actual Sequence Lengths: 必须是 1D Tensor, int64
# Shape: (batch_size,) -> (8,)
actual_seq_lengths = torch.full((batch_size,), actual_seq_len_kv, dtype=torch.int64)

# 缩放因子
scale = 1.0 / math.sqrt(head_dims)


# --- 3. 调用 npu_incre_flash_attention 并启用 Page Attention ---
for i in range(10):
    out = torch_npu.npu_incre_flash_attention(
        q,
        key_cache,
        value_cache,
        # --- 启用 Page Attention 的关键参数 ---
        block_table=block_table,
        actual_seq_lengths=actual_seq_lengths,
        block_size=block_size,
        # --- 其他必要参数 ---
        num_heads=num_heads,
        scale_value=scale,
        input_layout="BNSD", # 必须设为 BNSD
        num_key_value_heads=num_key_value_heads
    )


torch_npu.npu.synchronize()     
import time;
start = time.time()

out = torch_npu.npu_incre_flash_attention(
        q,
        key_cache,
        value_cache,
        # --- 启用 Page Attention 的关键参数 ---
        block_table=block_table,
        actual_seq_lengths=actual_seq_lengths,
        block_size=block_size,
        # --- 其他必要参数 ---
        num_heads=num_heads,
        scale_value=scale,
        input_layout="BNSD", # 必须设为 BNSD
        num_key_value_heads=num_key_value_heads
    )

torch_npu.npu.synchronize()
end = time.time()
elapsed_us = (end - start) * 1000000
print(f"官方32K 执行时间: {elapsed_us:.2f} us")



# --- 4. 打印结果形状进行验证 ---
print("\n--- 张量形状验证 ---")
print(f"Input q shape (BNSD): {q.shape}")
print(f"Input key_cache shape: {key_cache.shape}")
print(f"Input value_cache shape: {value_cache.shape}")
print(f"Input block_table shape: {block_table.shape}")
print(f"Input actual_seq_lengths: {actual_seq_lengths.shape}, value: {actual_seq_lengths[0].item()}")
print(f"Output out shape: {out.shape}")
print("--------------------")

# 输出的 shape 应该和 query 的 shape 一致
assert out.shape == q.shape
print("\n代码修改完成，已正确启用 Page Attention 功能！")
