import torch

batchSize = 8
qHead = 32

blockSize = 30
coreNum = 48

pageLength = torch.randint(30, 256, (batchSize, qHead))
print(f"pageLength: {pageLength}")

totalBatchPageLength = torch.sum(pageLength, dim=1)
print(f"totalBatchPageLength: {totalBatchPageLength}")

blockNumPerBatch = (totalBatchPageLength + blockSize - 1) // blockSize
print(f"blockNumPerBatch: {blockNumPerBatch}")

blockNumPerHead = (pageLength + blockSize - 1) // blockSize
print(f"blockNumPerHead: {blockNumPerHead}")

totalBlockNum = torch.sum(blockNumPerBatch)
print(f"totalBlockNum: {totalBlockNum}")

blockNumPerCore = (totalBlockNum + coreNum - 1) // coreNum
print(f"blockNumPerCore: {blockNumPerCore}")

# 根据blockNumPerBatch生成batchIdx
# 方法1：使用torch.repeat_interleave
batchIdx = torch.repeat_interleave(torch.arange(len(blockNumPerBatch)), blockNumPerBatch)
print(f"batchIdx (method 1): {batchIdx}, size: {batchIdx.size()}")

# # 方法2：使用列表推导式
# batchIdx2 = torch.tensor([i for i, count in enumerate(blockNumPerBatch) for _ in range(count)])
# print(f"batchIdx (method 2): {batchIdx2}")

# # 方法3：使用torch.cat
# batchIdx3 = torch.cat([torch.full((count,), i) for i, count in enumerate(blockNumPerBatch)])
# print(f"batchIdx (method 3): {batchIdx3}")

# # 根据blockNumPerHead生成headIdx
# # 方法1：使用torch.repeat_interleave
# headIdx = torch.repeat_interleave(torch.arange(qHead), blockNumPerHead.flatten())
# print(f"headIdx (method 1): {headIdx}")

# 方法2：使用列表推导式
headIdx2 = torch.tensor([head for batch_idx in range(batchSize) for head in range(qHead) for _ in range(blockNumPerHead[batch_idx, head])])
print(f"headIdx (method 2): {headIdx2}, size: {headIdx2.size()}")

# 方法3：使用torch.cat
headIdx3 = torch.cat([torch.full((blockNumPerHead[batch_idx, head],), head) for batch_idx in range(batchSize) for head in range(qHead)])
print(f"headIdx (method 3): {headIdx3}, size: {headIdx3.size()}")