import torch

batchSize = 8
qHead = 32
blockSize = 30
coreNum = 48

# 生成随机数据
pageLength = torch.randint(30, 256, (batchSize, qHead))

# 计算块数
blockNumPerHead = (pageLength + blockSize - 1) // blockSize
blockNumPerBatch = torch.sum(blockNumPerHead, dim=1)
totalBlockNum = torch.sum(blockNumPerBatch)

print(f"数据形状:")
print(f"blockNumPerHead: {blockNumPerHead.shape}")
print(f"blockNumPerBatch: {blockNumPerBatch.shape}")
print(f"totalBlockNum: {totalBlockNum}")

# 方法1：预计算累积偏移量（推荐）
def build_lookup_tables(blockNumPerBatch, blockNumPerHead):
    """构建查找表"""
    # 批次偏移量：每个批次的起始blockId
    batch_offsets = torch.cat([torch.tensor([0]), torch.cumsum(blockNumPerBatch[:-1], dim=0)])
    
    # 头偏移量：每个批次内每个头的起始blockId
    head_offsets = torch.zeros_like(blockNumPerHead)
    for batch_idx in range(batchSize):
        head_offsets[batch_idx] = torch.cat([
            torch.tensor([0]), 
            torch.cumsum(blockNumPerHead[batch_idx][:-1], dim=0)
        ])
    
    return batch_offsets, head_offsets

def get_batch_head_idx(blockId, batch_offsets, blockNumPerBatch, blockNumPerHead, head_offsets):
    """根据blockId获取batchIdx和headIdx"""
    # 找到batchIdx - 使用right=False确保正确找到批次
    batchIdx = torch.searchsorted(batch_offsets, blockId, right=False) - 1
    batchIdx = torch.clamp(batchIdx, 0, batchSize - 1)
    
    # 计算该批次内的相对blockId
    relative_blockId = blockId - batch_offsets[batchIdx]
    
    # 找到headIdx - 使用right=False确保正确找到头
    headIdx = torch.searchsorted(head_offsets[batchIdx], relative_blockId, right=False) - 1
    headIdx = torch.clamp(headIdx, 0, qHead - 1)
    
    return batchIdx, headIdx

# 构建查找表
batch_offsets, head_offsets = build_lookup_tables(blockNumPerBatch, blockNumPerHead)

print(f"\n查找表:")
print(f"batch_offsets: {batch_offsets}")
print(f"head_offsets[0]: {head_offsets[0]}")  # 第一个批次的头偏移量

# 测试查找功能
print(f"\n测试查找功能:")
test_blockIds = [0, 10, 50, 100, 200, 500, 1000, totalBlockNum-1]
for blockId in test_blockIds:
    if blockId < totalBlockNum:
        batchIdx, headIdx = get_batch_head_idx(blockId, batch_offsets, blockNumPerBatch, blockNumPerHead, head_offsets)
        print(f"blockId {blockId:4d} -> batchIdx {batchIdx}, headIdx {headIdx}")

# 方法2：向量化查找（批量处理）
def get_batch_head_idx_vectorized(blockIds, batch_offsets, blockNumPerBatch, blockNumPerHead, head_offsets):
    """向量化查找多个blockId"""
    # 找到每个blockId对应的batchIdx
    batchIndices = torch.searchsorted(batch_offsets, blockIds, right=False) - 1
    batchIndices = torch.clamp(batchIndices, 0, batchSize - 1)
    
    # 计算相对blockId
    relative_blockIds = blockIds - batch_offsets[batchIndices]
    
    # 找到headIdx
    headIndices = torch.zeros_like(blockIds)
    for i, (batchIdx, rel_blockId) in enumerate(zip(batchIndices, relative_blockIds)):
        headIdx = torch.searchsorted(head_offsets[batchIdx], rel_blockId, right=False) - 1
        headIndices[i] = torch.clamp(headIdx, 0, qHead - 1)
    
    return batchIndices, headIndices

# 测试向量化查找
print(f"\n测试向量化查找:")
test_blockIds_tensor = torch.tensor([0, 10, 50, 100, 200, 500, 1000, totalBlockNum-1])
batchIndices, headIndices = get_batch_head_idx_vectorized(test_blockIds_tensor, batch_offsets, blockNumPerBatch, blockNumPerHead, head_offsets)
print(f"blockIds: {test_blockIds_tensor}")
print(f"batchIndices: {batchIndices}")
print(f"headIndices: {headIndices}")

# 验证正确性：与完整索引对比
print(f"\n验证正确性:")
# 生成完整的索引
full_batchIdx = torch.repeat_interleave(torch.arange(batchSize), blockNumPerBatch)
full_headIdx = torch.cat([
    torch.repeat_interleave(torch.arange(qHead), blockNumPerHead[batch_idx]) 
    for batch_idx in range(batchSize)
])

# 随机选择一些blockId进行验证
import random
random.seed(42)
test_indices = random.sample(range(totalBlockNum), min(10, totalBlockNum))
correct_count = 0
for idx in test_indices:
    batchIdx, headIdx = get_batch_head_idx(idx, batch_offsets, blockNumPerBatch, blockNumPerHead, head_offsets)
    expected_batch = full_batchIdx[idx]
    expected_head = full_headIdx[idx]
    is_correct = batchIdx == expected_batch and headIdx == expected_head
    if is_correct:
        correct_count += 1
    print(f"blockId {idx:4d}: 查找结果 ({batchIdx}, {headIdx}) vs 期望结果 ({expected_batch}, {expected_head}) - {'✓' if is_correct else '✗'}")

print(f"\n正确率: {correct_count}/{len(test_indices)} = {correct_count/len(test_indices):.2%}")

print(f"\n存储空间对比:")
print(f"完整索引存储: {len(full_batchIdx) + len(full_headIdx)} 个元素")
print(f"查找表存储: {len(batch_offsets) + head_offsets.numel()} 个元素")
print(f"节省空间: {1 - (len(batch_offsets) + head_offsets.numel()) / (len(full_batchIdx) + len(full_headIdx)):.2%}")

# 性能测试
print(f"\n性能测试:")
import time

# 单次查找性能
start_time = time.time()
for _ in range(1000):
    get_batch_head_idx(500, batch_offsets, blockNumPerBatch, blockNumPerHead, head_offsets)
single_lookup_time = time.time() - start_time
print(f"1000次单次查找耗时: {single_lookup_time:.4f}秒")

# 批量查找性能
test_blockIds_batch = torch.randint(0, totalBlockNum, (1000,))
start_time = time.time()
batchIndices, headIndices = get_batch_head_idx_vectorized(test_blockIds_batch, batch_offsets, blockNumPerBatch, blockNumPerHead, head_offsets)
batch_lookup_time = time.time() - start_time
print(f"1000次批量查找耗时: {batch_lookup_time:.4f}秒")
