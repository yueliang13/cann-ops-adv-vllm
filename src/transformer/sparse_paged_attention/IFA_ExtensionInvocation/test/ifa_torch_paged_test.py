import torch
import torch_npu
import math

# 最小可运行的 _npu_paged_attention 接口测试
torch.manual_seed(0)

# 配置小规模参数
num_tokens = 1
num_heads = 32
num_kv_heads = 32
head_size = 128
head_size_v = 128
block_size = 128

max_seq_len = 32 * 1024
actual_seq_len = 4 * 1024

num_blocks = max_seq_len // block_size

max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size

scale_value = 1.0 / math.sqrt(head_size)

# 构造输入并放至 NPU
query = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16).npu()
key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.float16).npu()
value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size_v, dtype=torch.float16).npu()
block_table = torch.randint(0, num_blocks, (num_tokens, max_blocks_per_seq), dtype=torch.int32).npu()
context_lens = torch.full((num_tokens,), actual_seq_len, dtype=torch.int32)
output = torch.zeros(num_tokens, num_heads, head_size_v, dtype=torch.float16).npu()

# import ipdb;ipdb.set_trace()

# # 执行算子
for i in range(10):
    torch_npu._npu_paged_attention(
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                num_heads,
                scale_value,
                block_table,
                context_lens,
                output)

torch_npu.npu.synchronize()     
import time;
start = time.time()
torch_npu._npu_paged_attention(
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                num_heads,
                scale_value,
                block_table,
                context_lens,
                output)

torch_npu.npu.synchronize()
end = time.time()
elapsed_us = (end - start) * 1000000
print(f"执行时间: {elapsed_us:.2f} us")

# print(output)
# print("_npu_paged_attention output shape:", tuple(output.shape))

