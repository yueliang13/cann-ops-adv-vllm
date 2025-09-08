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

print(f"原始数据形状:")
print(f"pageLength: {pageLength.shape}")
print(f"blockNumPerHead: {blockNumPerHead.shape}")
print(f"blockNumPerBatch: {blockNumPerBatch.shape}")
print(f"totalBlockNum: {totalBlockNum}")

# 方法1：使用torch.repeat_interleave - 最简洁高效
batchIdx = torch.repeat_interleave(torch.arange(batchSize), blockNumPerBatch)

# 对于headIdx，需要按批次和头来生成
headIdx = torch.cat([
    torch.repeat_interleave(torch.arange(qHead), blockNumPerHead[batch_idx]) 
    for batch_idx in range(batchSize)
])

print(f"\n精简后的索引:")
print(f"batchIdx: {batchIdx.shape} - 每个批次索引重复对应块数次数")
print(f"headIdx: {headIdx.shape} - 每个头索引重复对应块数次数")

# 验证数据一致性
print(f"\n验证:")
print(f"batchIdx长度: {len(batchIdx)}")
print(f"headIdx长度: {len(headIdx)}")
print(f"长度是否一致: {len(batchIdx) == len(headIdx)}")

# 如果需要更紧凑的表示，可以使用稀疏索引
# 只存储每个批次/头的起始位置和长度
batch_starts = torch.cat([torch.tensor([0]), torch.cumsum(blockNumPerBatch[:-1], dim=0)])
head_starts = torch.cat([torch.tensor([0]), torch.cumsum(blockNumPerHead.flatten()[:-1], dim=0)])

print(f"\n稀疏表示:")
print(f"batch_starts: {batch_starts}")
print(f"head_starts: {head_starts}")

# 更精简的方法：只存储必要的信息
print(f"\n最精简的表示:")
print(f"只需要存储 blockNumPerBatch: {blockNumPerBatch}")
print(f"只需要存储 blockNumPerHead: {blockNumPerHead}")
print(f"总存储空间: {blockNumPerBatch.numel() + blockNumPerHead.numel()} 个元素")
print(f"相比完整索引节省: {1 - (blockNumPerBatch.numel() + blockNumPerHead.numel()) / (len(batchIdx) + len(headIdx)):.2%}")

# 展示如何从精简数据重建索引
print(f"\n重建示例:")
print(f"从 blockNumPerBatch 重建 batchIdx:")
print(f"blockNumPerBatch = {blockNumPerBatch}")
print(f"重建的 batchIdx = {torch.repeat_interleave(torch.arange(batchSize), blockNumPerBatch)}")

print(f"\n从 blockNumPerHead 重建 headIdx (前几个元素):")
print(f"blockNumPerHead[0] = {blockNumPerHead[0]}")
print(f"重建的 headIdx[0] = {torch.repeat_interleave(torch.arange(qHead), blockNumPerHead[0])}")
