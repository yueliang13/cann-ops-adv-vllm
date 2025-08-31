# IFA_V5 Python接口使用指南

## 1. 环境配置

在使用前，请先配置以下环境变量：

```bash
# 添加CANN运行时库路径
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH

# 添加自定义算子库路径（注意：移除了多余空格）
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib:${LD_LIBRARY_PATH}

# 设置CANN环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 2. 安装方式

### 方式一：安装预编译的二进制包

```bash
BASE_DIR=$(pwd)
cd ${BASE_DIR}/export
pip3 install *.whl --force-reinstall
```

### 方式二：自行编译代码

```bash
# 如果需要自行编译,记得提前先把cann-ops-adv的包编译安装一下
bash build_and_run.sh
```

## 3. 接口测试

执行提供的测试用例：

```bash
cd ${BASE_DIR}/test
python3 ifa_v5_case.py
```

## 4. 接口使用说明

### 基本用法

```python
import custom_ops
import torch
import numpy as np

# 使用IFA_V5算子
output = custom_ops.incre_flash_attention_v5(
    query_tensor,          # 查询张量 [batch_size, num_heads, seq_len_q, head_dims]
    key_list,              # 键张量列表 [blocknum, block_size, key_num_heads * D] or [block_num, key_num_heads, block_size, head_dims]
    value_list,            # 值张量列表 [blocknum, block_size, key_num_heads * D] or [block_num, key_num_heads, block_size, head_dims]
    pse_shift,             # 位置编码
    attention_mask,        # 注意力掩码
    actual_seq_lengths,    # 实际序列长度
    dequant_scale1,        # 反量化缩放参数1
    quant_scale1,          # 量化缩放参数1
    dequant_scale2,        # 反量化缩放参数2
    quant_scale2,          # 量化缩放参数2
    quant_offset2,         # 量化偏移参数2
    antiquant_scale,       # 反量化缩放参数
    antiquant_offset,      # 反量化偏移参数
    block_table,           # 块表 [batch_size, max_block_num_per_batch]
    kv_padding_size,       # KV填充大小
    block_position,        # 块位置映射表 [batch_size, kv_head_nums, max_block_num_per_seq]
    num_heads,             # 头数量
    scale_value,           # 缩放值
    layout,                # 布局方式
    key_num_heads,         # 键头数量
    block_size,            # 块大小
    inner_precise          # 精度控制
)
```

### 参数说明

* **block_position**: 新增参数，形状为[batch_size, kv_head_nums, max_block_num_per_seq]，用于实现多头块选择;(对应不选择的Page块 赋值0x7FFFFFFF处理)
* **block_table**: 块表，形状为[batch_size, max_block_num_per_batch]，存储实际块ID
* 其他参数与[官方文档](https://gitee.com/ascend/cann-ops-adv/blob/master/docs/IncreFlashAttentionV4.md)一致
* 开启PageAttention的情况下,KVCache 支持Shape [blocknum, blocksize, H]和 [blocknum, KV_N, blocksize, D],数据排布需要和Shape保持一致
* page attention场景下，当query的inputLayout为BNSD时，kv cache排布支持（blocknum, blocksize, H）和（blocknum, KV_N, blocksize, D）两种格式，当query的inputLayout为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）一种格式。blocknum不能小于根据actualSeqLengthsKv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。

### 块位置映射使用说明

`block_position`参数实现了两级映射机制：
1. 通过`block_position`中存储的索引值查找`block_table`
2. 通过`block_table`中的块ID找到对应的KV数据

这种机制允许不同头选择不同的块，提高了模型的灵活性。

## 5. 注意事项

* 确保NPU环境正确配置
* 确认传递给算子的张量都已经正确移至NPU设备
* 对于可选参数，可以传递空张量(`torch.empty(0).npu()`)
