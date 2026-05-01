# 稀疏注意力融合算子

## 背景

本项目基于昇腾硬件，将原有的 `IncreFlashAttention`（IFA）增量式 Flash Attention 算子扩展为**支持稀疏注意力**的融合算子库。IFA 使用 Flash Attention 算法实现 self-attention 计算，支持 Paged Attention、KV Cache 量化、位置编码等特性，已在 LLM 推理场景中广泛使用。

## 稀疏注意力方案

为减少 attention 计算量，引入**聚类码本（Centroid）**机制：对历史 KV cache 做聚类，根据 query 与码本的相似度选出最重要的 tokens，仅在这些 tokens 上进行 attention 计算。新增的稀疏注意力算子链如下：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SparsePagedFusionAttention                        │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────────┐   │
│  │ ComputeCent  │ -> │ CentSelect   │ -> │  SparsePagedAttention      │   │
│  │  query·centᵀ │    │ topk+select  │    │  paged sparse attention    │   │
│  └─────────────┘    └──────────────┘    └────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 算子说明

| 算子 | 功能 | 关键输入 | 关键输出 |
|------|------|----------|----------|
| **ComputeCent** | 计算 query 与 L1 聚类码本的相似度（Matmul），输出 topk 聚类索引 | `query [B,N,D]`, `l1_cent [H,C,D]` | `indices [B,N,k]` |
| **SelectPosition** | 根据聚类索引和 block_table，解析出选中的 page position | `block_ids`, `block_table`, `seq_len`, `indices` | `page_position`, `page_position_length`, `block_table_gather` |
| **CentSelect** | 融合 ComputeCent + SelectPosition，一步完成中心选择与位置解析 | `query`, `l1_cent`, `block_ids`, `block_table`, `seq_len` | `page_position`, `page_position_length`, `max_page_position_length` |
| **SparsePagedAttention** | 在选出的稀疏 tokens 上执行 Paged Attention 计算，仅关注前 k 个最相关聚类对应的 tokens | `query`, `key/value`, `blocktable`, `blockposition` | `attention_out` |
| **SparsePagedFusionAttention** | **融合算子**：将 CentSelect + SparsePagedAttention 融合为单算子，中间结果在 device 侧产生和消费，避免 CPU 同步 | 上述两算子的全部输入 | `attention_out`, `block_position`, `page_position_length`, `max_page_position_length` |

### 设计要点

1. **码本聚类（L1 Centroid）**：将 KV 序列按语义聚类为 C 个中心，存储在 `l1_cent [H, C, D]` 中。
2. **相似度选择**：query 与码本做矩阵乘法（`Q * Centᵀ`），TopK 选出相似度最高的 k 个聚类。
3. **位置解析**：通过选中的聚类索引，结合 block_table 和 block_ids 反查对应的 page position。
4. **稀疏注意力**：仅对选出的 page position 进行 attention，大幅减少 KV cache 读取和计算量。
5. **融合优化**：`SparsePagedFusionAttention` 将 CentSelect 与 SparsePagedAttention 融合为一个 kernel 调用，中间变量在 device 端产生和消费，消除 CPU 同步开销。

## 目录结构

```
src/transformer/
├── incre_flash_attention/            # 原 IFA 算子（基础 attention 实现）
│   ├── incre_flash_attention.cpp
│   ├── incre_flash_attention_allvec_new.h
│   ├── incre_flash_attention_split_Bbn2s2_Us2.h
│   ├── ophost/                       # aclnn 接口、tiling、算子定义
│   └── test/
│
├── compute_cent/                     # 聚类码本相似度计算
│   ├── compute_cent.cpp              # Kernel: query·centᵀ Matmul + TopK
│   ├── ophost/
│   └── test/
│
├── cent_select/                      # 重要性聚类选择（ComputeCent + SelectPosition 融合）
│   ├── cent_select.cpp               # Kernel: 融合查询中心 + 位置解析
│   └── ophost/
│
├── select_position/                  # 重要 token 位置选择
│   ├── select_position.cpp           # Kernel: block_table 查表 + 位置归并
│   └── ophost/
│
├── sparse_paged_attention/           # 稀疏分页注意力计算
│   ├── sparse_paged_attention.cpp    # Kernel: 基于选定位置的 Paged Attention
│   ├── sparse_paged_attention_allvec_new.h
│   ├── sparse_paged_attention_split_Bbn2s2_Us2.h
│   ├── ophost/
│   └── IFA_ExtensionInvocation/      # Python 侧调用测试
│
└── sparse_paged_fusion_attention/    # 融合算子（CentSelect + SparsePagedAttention）
    ├── sparse_paged_fusion_attention.cpp  # Kernel: 一次调用完成选择+注意力
    ├── sparse_paged_fusion_attention_allvec_new.h
    ├── sparse_paged_fusion_attention_split_Bbn2s2_Us2.h
    ├── fusion_api_analyze_v2.md       # 融合方案设计文档
    ├── ophost/
    ├── test/
    └── op_eval/                       # 性能评估
```

## Python 调用接口

通过 `custom_ops` 扩展包暴露给 PyTorch：

```python
# 1. 聚类码本计算：query 与聚类中心做矩阵乘 + TopK
indices = custom_ops.compute_cent(query, l1_cent)
# indices: [B, N, k] int32

# 2. 位置选择：根据聚类索引查 block_table，解析 page position
page_position, page_position_length, block_table_gather = \
    custom_ops.select_position(block_ids, block_table, seq_len, indices)

# 3. 重要性聚类选择：一步完成 compute_cent + select_position
page_position, page_position_length, max_page_position_length = \
    custom_ops.cent_select(query, l1_cent, block_ids, block_table, seq_len)

# 4. 稀疏分页注意力：在选出的 page position 上做 attention
attention_out = custom_ops.sparse_paged_attention(
    query, key, value, pse_shift, attention_mask, actual_seq_lengths,
    dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
    antiquant_scale, antiquant_offset, blocktable, kv_padding_size,
    blockposition, num_heads, scale_value, input_layout, num_key_value_heads,
    block_size, inner_precise
)

# 5. 融合算子：cent_select + sparse_paged_attention 融合为一次调用
attention_out, block_position, page_position_length, max_page_position_length = \
    custom_ops.sparse_paged_fusion_attention(
        query, key, value, pse_shift, attention_mask, actual_seq_lengths,
        dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
        antiquant_scale, antiquant_offset, blocktable, kv_padding_size,
        l1_cent, block_ids, total_seq_len,
        block_position, page_position_length, max_page_position_length,
        num_heads, scale_value, input_layout, num_key_value_heads,
        block_size, inner_precise
    )
```

## 融合算子优势

- **减少 CPU 同步**：CentSelect 的中间结果（`block_position`, `page_position_length`）在 device 侧直接消费，无需 CPU 介入。
- **降低 kernel launch 开销**：单次 `aclOpExecutor` 调用替代两次独立算子调用。
- **工作空间复用**：CentSelect 和 Attention 共享 workspace，减少显存占用。

## C++ aclnn 接口

```cpp
// ComputeCent
aclnnStatus aclnnComputeCentGetWorkspaceSize(
    const aclTensor *query, const aclTensor *l1Cent,
    const aclTensor *indices, uint64_t *workspaceSize, aclOpExecutor **executor);

// CentSelect
aclnnStatus aclnnCentSelectGetWorkspaceSize(
    const aclTensor *query, const aclTensor *l1Cent,
    const aclTensor *blockIds, const aclTensor *blockTable,
    const aclTensor *seqLen, const aclTensor *pagePosition,
    const aclTensor *pagePositionLength,
    const aclTensor *workspace, const aclTensor *tiling,
    uint64_t *workspaceSize, aclOpExecutor **executor);

// SelectPosition
aclnnStatus aclnnSelectPositionGetWorkspaceSize(
    const aclTensor *blockIds, const aclTensor *blockTable,
    const aclTensor *seqLen, const aclTensor *indices,
    const aclTensor *pagePosition, const aclTensor *pagePositionLength,
    const aclTensor *blockTableGather, const aclTensor *workspace,
    uint64_t *workspaceSize, aclOpExecutor **executor);

// SparsePagedAttention
aclnnStatus aclnnSparsePagedAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
    const aclTensor *pseShift, const aclTensor *attenMask,
    const aclIntArray *actualSeqLengths, const aclTensor *dequantScale1,
    const aclTensor *quantScale1, const aclTensor *dequantScale2,
    const aclTensor *quantScale2, const aclTensor *quantOffset2,
    const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blocktable, const aclTensor *kvPaddingSize,
    const aclTensor *blockposition, int64_t numHeads, double scaleValue,
    char *inputLayout, int64_t numKeyValueHeads, int64_t blockSize,
    int64_t innerPrecise, const aclTensor *attentionOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

// SparsePagedFusionAttention
aclnnStatus aclnnSparsePagedFusionAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
    const aclTensor *pseShift, const aclTensor *attenMask,
    const aclIntArray *actualSeqLengths, const aclTensor *dequantScale1,
    const aclTensor *quantScale1, const aclTensor *dequantScale2,
    const aclTensor *quantScale2, const aclTensor *quantOffset2,
    const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blocktable, const aclTensor *kvPaddingSize,
    const aclTensor *l1Cent, const aclTensor *blockIds,
    const aclTensor *totalSeqLen,
    const aclTensor *blockposition, const aclTensor *pagePositionLength,
    const aclTensor *maxPagePositionLength,
    int64_t numHeads, double scaleValue, char *inputLayout,
    int64_t numKeyValueHeads, int64_t blockSize, int64_t innerPrecise,
    const aclTensor *attentionOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

## 编译与安装

```bash
mkdir build && cd build
cmake ..
make package -j $(nproc)
```

编译产出为 `output/CANN-custom_ops-<version>-linux.<arch>.run`，安装方式：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./CANN-custom_ops-<version>-linux.<arch>.run --quiet
```

## 测试

```bash
# 单元测试
cmake .. -DTESTS_UT_OPS_TEST=ALL
make ops_test_utest -j

# Extension 侧 Python 测试
cd src/transformer/sparse_paged_fusion_attention/test
bash run.sh
```

## 许可证

[CANN Open Software License Agreement Version 1.0](LICENSE)
