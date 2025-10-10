import torch
import torch_npu
import numpy as np
import argparse
import time

# 说明：
# 本测试脚本用于验证融合算子（CentSelect + SparsePagedAttention）与分离算子的一致性。
# 步骤：
# 1) 使用融合算子一次性完成 CentSelect + 注意力，得到 fused_out。
# 2) 使用分离算子：先调用 cent_select 得到 blockPosition/maxPagePositionLength，再调用官方稀疏注意力，得到 sep_out。
# 3) 对比 fused_out 与 sep_out 的差异。

import custom_ops  # 通过 pybind 导出的自定义封装


def build_case_tensors():
    torch.npu.set_device(0)

    # 配置（与分离用例保持一致的规模，便于对比）
    B = 1
    Nq = 32
    Nk = 8
    D = 128
    C = 512             # centroids
    block_size = 128
    total_seq_len_kv = 128 * 1024
    kvPageLen = 1280
    maxBatch = 256
    maxPage = 1024
    gSize = Nq // Nk
    
    sparsity_ratio = 0.125 # 稀疏比例 1/8
    base_block_num = (total_seq_len_kv * sparsity_ratio) // block_size

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
    key_shape = [kvPageLen, block_size, Nk, D]
    value_shape = [kvPageLen, block_size, Nk, D]

    # 构造输入（QKV 常数 0.1）
    eps = 1e-3
    query = (torch.rand(query_shape, dtype=torch.float16) * (1.0 - eps) + eps)
    l1_cent = torch.randn(l1_shape, dtype=torch.float16)
    block_ids = torch.randint(0, C, block_ids_shape, dtype=torch.int32)

    block_table = torch.randint(0, kvPageLen, block_table_shape, dtype=torch.int32)
    total_seq_len = torch.full(total_seq_len_shape, total_seq_len_kv, dtype=torch.int32)
    key = (torch.rand(key_shape, dtype=torch.float16) * (1.0 - eps) + eps)
    value = (torch.rand(value_shape, dtype=torch.float16) * (1.0 - eps) + eps)

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
    
    actual_seq_lengths = torch.full((B,), base_block_num * block_size, dtype=torch.int64, device='npu').contiguous()    
    
    dequant_scale1 = empty_f16.npu()
    quant_scale1 = empty_f16.npu()
    dequant_scale2 = empty_f16.npu()
    quant_scale2 = empty_f16.npu()
    quant_offset2 = empty_f16.npu()
    antiquant_scale = empty_f16.npu()
    antiquant_offset = empty_f16.npu()
    kv_padding_size = empty_i64.npu()

    case = {
        "B": B,
        "Nq": Nq,
        "Nk": Nk,
        "D": D,
        "block_size": block_size,
        "total_seq_len_kv": total_seq_len_kv,
        "kvPageLen": kvPageLen,
        "maxBatch": maxBatch,
        "maxPage": maxPage,
        "num_heads": num_heads,
        "num_key_value_heads": num_key_value_heads,
        "layout": layout,
        "inner_precise": inner_precise,
        "scale_value": scale_value,
        # tensors
        "query": query,
        "l1_cent": l1_cent,
        "block_ids": block_ids,
        "block_table": block_table,
        "total_seq_len": total_seq_len,
        "key": key,
        "value": value,
        "pse_shift": pse_shift,
        "attention_mask": attention_mask,
        "actual_seq_lengths": actual_seq_lengths,
        "dequant_scale1": dequant_scale1,
        "quant_scale1": quant_scale1,
        "dequant_scale2": dequant_scale2,
        "quant_scale2": quant_scale2,
        "quant_offset2": quant_offset2,
        "antiquant_scale": antiquant_scale,
        "antiquant_offset": antiquant_offset,
        "kv_padding_size": kv_padding_size,
        "block_position": block_position,
        "page_position_length": page_position_length,
        "max_page_position_length": max_page_position_length,
    }
    return case


def run_perf(iters: int, warmup: int):
    case = build_case_tensors()
    # warmup
    for _ in range(warmup):
        _ = custom_ops.sparse_paged_fusion_attention(
            case["query"], case["key"], case["value"],
            case["pse_shift"], case["attention_mask"],
            case["actual_seq_lengths"],
            case["dequant_scale1"], case["quant_scale1"], case["dequant_scale2"], case["quant_scale2"], case["quant_offset2"],
            case["antiquant_scale"], case["antiquant_offset"], case["block_table"], case["kv_padding_size"],
            case["l1_cent"], case["block_ids"], case["total_seq_len"],
            case["block_position"], case["page_position_length"], case["max_page_position_length"],
            case["num_heads"], case["scale_value"], case["layout"], case["num_key_value_heads"], case["block_size"], case["inner_precise"]
        )
    torch.npu.synchronize()

    # measure
    t0 = time.time()
    for _ in range(iters):
        _ = custom_ops.sparse_paged_fusion_attention(
            case["query"], case["key"], case["value"],
            case["pse_shift"], case["attention_mask"],
            case["actual_seq_lengths"],
            case["dequant_scale1"], case["quant_scale1"], case["dequant_scale2"], case["quant_scale2"], case["quant_offset2"],
            case["antiquant_scale"], case["antiquant_offset"], case["block_table"], case["kv_padding_size"],
            case["l1_cent"], case["block_ids"], case["total_seq_len"],
            case["block_position"], case["page_position_length"], case["max_page_position_length"],
            case["num_heads"], case["scale_value"], case["layout"], case["num_key_value_heads"], case["block_size"], case["inner_precise"]
        )
    torch.npu.synchronize()
    t1 = time.time()

    avg_ms = (t1 - t0) * 1000.0 / max(iters, 1)
    print(f"[Perf] iters={iters}, warmup={warmup}, avg_latency_ms={avg_ms:.3f}")


def run_acc():
    case = build_case_tensors()

    print("Call fused case...")
    attention_out, fused_block_position_out, fused_max_page_position_length_out = custom_ops.sparse_paged_fusion_attention(
        case["query"],
        case["key"], case["value"],
        case["pse_shift"], case["attention_mask"],
        case["actual_seq_lengths"],
        case["dequant_scale1"], case["quant_scale1"], case["dequant_scale2"], case["quant_scale2"], case["quant_offset2"],
        case["antiquant_scale"], case["antiquant_offset"], case["block_table"], case["kv_padding_size"],
        case["l1_cent"], case["block_ids"], case["total_seq_len"],
        case["block_position"], case["page_position_length"], case["max_page_position_length"],
        case["num_heads"], case["scale_value"], case["layout"], case["num_key_value_heads"], case["block_size"], case["inner_precise"]
    )
# attention_out [1, 32, 128] 
    # 采样每个头的数据（按batch维B、头维Nq）
    with torch.no_grad():
        B = case["B"] if "B" in case else attention_out.shape[0]
        Nq = case["Nq"] if "Nq" in case else attention_out.shape[1]
        D = attention_out.shape[2]
        sample_tokens = min(8, D)
        b_idx = 0 if B > 0 else 0
        print("[Sample Per-Head] batch=", b_idx)
        for h in range(Nq):
            # attention_out: [B, Nq, D]，取前 sample_tokens 个通道做示例
            head_vec = attention_out[b_idx, h, :sample_tokens].cpu().float()
            head_mean = attention_out[b_idx, h, :].mean().item()
            # block_position: [B, Nq, 256]，取前8个位置
            pos_sample = fused_block_position_out[b_idx, h, :8].cpu()
            # page length: [B, 8]（或 [B, tplPadding]），取第0列代表该batch的最大长度通道
            max_page_len_b = fused_max_page_position_length_out[b_idx, 0].item()
            print(f"  head={h:02d} attn_mean={head_mean:.6f} attn_sample={head_vec.tolist()} pos_sample={pos_sample.tolist()} max_page_len={int(max_page_len_b)}")
    
    
    print("Call sep case cent_select...")
    sep_block_position, sep_page_position_length, sep_max_page_position_length = custom_ops.cent_select(
        case["query"], case["l1_cent"], case["block_ids"], case["block_table"], case["total_seq_len"]
    )

    # 校验 fused 与 sep 的 block_position 是否完全一致
    try:
        if fused_block_position_out.shape != sep_block_position.shape:
            print("[Check] block_position shape mismatch:", fused_block_position_out.shape, sep_block_position.shape)
        else:
            same = torch.equal(fused_block_position_out, sep_block_position)
            if same:
                print("[Check] fused_block_position_out == sep_block_position: True")
            else:
                print("[Check] fused_block_position_out == sep_block_position: False")
                # 打印首个不一致位置样例
                diff = (fused_block_position_out != sep_block_position)
                idx = torch.nonzero(diff)
                num_print = min(10, idx.shape[0])
                for i in range(num_print):
                    b, h, p = idx[i].tolist()
                    fv = int(fused_block_position_out[b, h, p].item())
                    sv = int(sep_block_position[b, h, p].item())
                    print(f"  mismatch at (b={b}, h={h}, pos={p}): fused={fv}, sep={sv}")
    except Exception as e:
        print("[Check] block_position compare error:", e)
    
    # import ipdb; ipdb.set_trace()
    sep_actual_seq_lengths = (sep_max_page_position_length[:, 0]).to(torch.int64)
    print("Call sep case sparse_paged_attention...")
    sep_out = torch_npu.npu_sparse_paged_attention(
        case["query"],
        case["key"],  # 这里传入与 fused 相同的 K/V 以对齐对比
        case["value"],
        actual_seq_lengths=sep_actual_seq_lengths,
        block_table=case["block_table"],
        block_position=sep_block_position,
        num_heads=case["num_heads"],
        scale_value=case["scale_value"],
        input_layout=case["layout"],
        num_key_value_heads=case["num_key_value_heads"],
        block_size=case["block_size"],
        inner_precise=case["inner_precise"]
    )

    fused_out_cpu = attention_out.cpu().to(torch.float32)
    sep_out_cpu = sep_out.cpu().to(torch.float32)

    diff = (fused_out_cpu - sep_out_cpu).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    eps = 1e-6
    denom = sep_out_cpu.abs().clamp_min(eps)
    rel = (diff / denom)
    max_rel = rel.max().item()
    mean_rel = rel.mean().item()

    print("[Acc] max_abs_diff=", max_abs, "mean_abs_diff=", mean_abs,
          "max_rel_diff=", max_rel, "mean_rel_diff=", mean_rel)

    abs_tol = 1e-1 if fused_out_cpu.dtype == torch.float32 else 1e-2
    rel_tol = 3e-2
    if mean_rel < rel_tol:
        print("✅ 融合与分离算子输出一致（在容忍度内）")
    else:
        print("❌ 差异较大，请检查实现")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["perf", "acc"], default="acc")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "perf":
        run_perf(args.iters, args.warmup)
    else:
        run_acc()


