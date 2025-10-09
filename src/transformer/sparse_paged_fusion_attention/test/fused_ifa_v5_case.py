import torch
import torch_npu
import numpy as np

# 说明：
# 本测试脚本用于验证融合算子（CentSelect + SparsePagedAttention）与分离算子的一致性。
# 步骤：
# 1) 使用融合算子一次性完成 CentSelect + 注意力，得到 fused_out。
# 2) 使用分离算子：先调用 cent_select 得到 blockPosition/maxPagePositionLength，再调用官方稀疏注意力，得到 sep_out。
# 3) 对比 fused_out 与 sep_out 的差异。

import custom_ops  # 通过 pybind 导出的自定义封装


def run_fused_case():
    torch.npu.set_device(0)

    # 配置（与分离用例保持一致的规模，便于对比）
    B = 1
    Nq = 32
    Nk = 8
    D = 128
    C = 512             # centroids
    block_size = 128
    total_seq_len_kv = 32 * 1024
    kvPageLen = 1280
    maxBatch = 256
    maxPage = 1024
    gSize = Nq // Nk

    # 头与布局
    num_heads = Nq
    num_key_value_heads = Nk
    layout = "BNSD"
    inner_precise = 1
    scale_value = np.float32(1.0)

    # 张量形状
    query_shape = [B, Nq, D]
    l1_shape = [Nk, C, D]
    block_ids_shape = [Nk, kvPageLen]
    block_table_shape = [maxBatch, maxPage]
    total_seq_len_shape = [B]
    block_num = total_seq_len_kv // block_size
    key_shape = [block_num, block_size, Nk, D]
    value_shape = [block_num, block_size, Nk, D]

    # 构造输入（QKV 全 1，不影响 cent_select 的 l1_cent 随机）
    query = torch.ones(query_shape, dtype=torch.float16)
    l1_cent = torch.randn(l1_shape, dtype=torch.float16)
    block_ids = torch.randint(0, C, block_ids_shape, dtype=torch.int32)
    block_table = torch.randint(0, kvPageLen, block_table_shape, dtype=torch.int32)
    total_seq_len = torch.full(total_seq_len_shape, total_seq_len_kv, dtype=torch.int32)
    key = torch.ones(key_shape, dtype=torch.float16)
    value = torch.ones(value_shape, dtype=torch.float16)

    # 可选参数留空
    empty_f16 = torch.empty(0, dtype=torch.float16)
    empty_i32 = torch.empty(0, dtype=torch.int32)
    empty_i64 = torch.empty(0, dtype=torch.int64)

    # 上 NPU
    query = query.npu()
    l1_cent = l1_cent.npu()
    block_ids = block_ids.npu()
    block_table = block_table.npu()
    total_seq_len = total_seq_len.npu()

    # 预分配三项中间结果（由融合算子内部写回）
    block_position = torch.empty(B, Nq, 256, dtype=torch.int32, device='npu')
    page_position_length = torch.empty(B, Nq, 8, dtype=torch.int32, device='npu')
    max_page_position_length = torch.empty(B, 8, dtype=torch.int64, device='npu')

    # kv/可选量化参数
    key = key.npu()
    value = value.npu()
    pse_shift = empty_f16.npu()
    attention_mask = empty_f16.npu()
    # 复用 IFA 的数据：实际长度 = base_block_num * block_size
    base_block_num = (16 * 1024) // block_size
    actual_seq_lengths = torch.tensor([base_block_num * block_size], dtype=torch.int64, device='npu')
    dequant_scale1 = empty_f16.npu()
    quant_scale1 = empty_f16.npu()
    dequant_scale2 = empty_f16.npu()
    quant_scale2 = empty_f16.npu()
    quant_offset2 = empty_f16.npu()
    antiquant_scale = empty_f16.npu()
    antiquant_offset = empty_f16.npu()
    kv_padding_size = empty_i64.npu()

    # 1) 融合算子
    fused_out = custom_ops.sparse_paged_fusion_attention(
        query,
        key, value,
        pse_shift, attention_mask,
        actual_seq_lengths,
        dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
        antiquant_scale, antiquant_offset, block_table, kv_padding_size,
        l1_cent, block_ids, total_seq_len,
        block_position, page_position_length, max_page_position_length,
        num_heads, scale_value, layout, num_key_value_heads, block_size, inner_precise
    )

    # 2) 分离算子：cent_select + 官方稀疏注意力
    sep_block_position, sep_page_position_length, sep_max_page_position_length = custom_ops.cent_select(
        query, l1_cent, block_ids, block_table, total_seq_len
    )

    # 使用 torch_npu 官方稀疏注意力（npu_sparse_paged_attention），与分离结果对齐
    # 注：该 API 期望的 actual_seq_lengths 为 int64 token 长度；我们用 batch 最大值输出
    sep_actual_seq_lengths = (sep_max_page_position_length[:, 0]).to(torch.int64)

    # 这里直接调用 torch_npu 的官方接口（若你需要，也可以改为调用自定义 sparse_paged_attention 包装）
    sep_out = torch_npu.npu_sparse_paged_attention(
        query,
        key,  # 空
        value,  # 空
        actual_seq_lengths=sep_actual_seq_lengths,
        block_table=block_table,
        block_position=sep_block_position,
        num_heads=num_heads,
        scale_value=scale_value,
        input_layout=layout,
        num_key_value_heads=num_key_value_heads,
        block_size=block_size,
        inner_precise=inner_precise
    )

    # 对比
    fused_out_cpu = fused_out.cpu().to(torch.float32)
    sep_out_cpu = sep_out.cpu().to(torch.float32)

    diff = (fused_out_cpu - sep_out_cpu).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    print("[Fused Test] max_abs_diff=", max_abs, "mean_abs_diff=", mean_abs)

    # 简单阈值判断
    tol = 1e-1 if fused_out_cpu.dtype == torch.float32 else 1e-2
    if max_abs < 1e-1:
        print("✅ 融合与分离算子输出一致（在容忍度内）")
    else:
        print("❌ 差异较大，请检查实现")


if __name__ == "__main__":
    run_fused_case()


