import torch
import torch_npu
import numpy as np
import argparse
import time
import torch.profiler
import torch_npu.profiler
import os
import json
from datetime import datetime
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

    # # 预分配三项中间结果（由融合算子内部写回）
    # block_position = torch.zeros(B, Nq, 256, dtype=torch.int32, device='npu')
    # page_position_length = torch.zeros(B, Nq, 8, dtype=torch.int32, device='npu')
    # max_page_position_length = torch.zeros(B, 8, dtype=torch.int64, device='npu')

    # kv/可选量化参数
    key = key.npu()
    value = value.npu()
    pse_shift = empty_f16.npu()
    attention_mask = empty_f16.npu()
    
    actual_seq_lengths = [int(base_block_num * block_size)] #torch.full((1,), base_block_num * block_size, dtype=torch.int64, device='npu').contiguous()    
    # actual_seq_lengths = base_block_num * block_size   
    
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
        # "block_position": block_position,
        # "page_position_length": page_position_length,
        # "max_page_position_length": max_page_position_length,
    }
    return case


def run_perf(iters: int, warmup: int):
    case = build_case_tensors()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"npu_profiler_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[NPU Profiler] Starting profiling with {iters} iterations, {warmup} warmup")
    print(f"[NPU Profiler] Output directory: {output_dir}")
    
    # 配置NPU profiler
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        msprof_tx=True,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None
    )
    
    # 创建NPU profiler
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, 
            warmup=warmup, 
            active=iters, 
            repeat=1, 
            skip_first=0
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        with_flops=True,
        experimental_config=experimental_config
    )
    
    # 记录执行时间
    execution_times = []
    
    # 开始profiling
    prof.start()
    
    # Warmup阶段
    print(f"[NPU Profiler] Starting warmup ({warmup} steps)...")
    for step in range(warmup):
        start_time = time.time()
        attention_out, fused_block_position_out, fused_max_page_position_length_out = torch_npu.npu_sparse_paged_fusion_attention(
            case["query"], case["key"], case["value"],
            case["block_table"], case["l1_cent"], case["block_ids"], case["total_seq_len"],
            actual_seq_lengths=case["actual_seq_lengths"],
            num_heads=case["num_heads"], scale_value=case["scale_value"], 
            input_layout=case["layout"], num_key_value_heads=case["num_key_value_heads"],
            block_size=case["block_size"], inner_precise=case["inner_precise"]
        )
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        execution_times.append(duration)
        print(f"[NPU Profiler] Warmup step {step + 1}/{warmup}: {duration:.3f}ms")
        prof.step()
    
    # Active profiling阶段
    print(f"[NPU Profiler] Starting active profiling ({iters} steps)...")
    
    for step in range(iters):
        start_time = time.time()
        attention_out, fused_block_position_out, fused_max_page_position_length_out = torch_npu.npu_sparse_paged_fusion_attention(
            case["query"], case["key"], case["value"],
            case["block_table"], case["l1_cent"], case["block_ids"], case["total_seq_len"],
            actual_seq_lengths=case["actual_seq_lengths"],
            num_heads=case["num_heads"], scale_value=case["scale_value"], 
            input_layout=case["layout"], num_key_value_heads=case["num_key_value_heads"],
            block_size=case["block_size"], inner_precise=case["inner_precise"]
        )
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        execution_times.append(duration)
        print(f"[NPU Profiler] Active step {step + 1}/{iters}: {duration:.3f}ms")
        prof.step()
    
    # 停止profiling
    prof.stop()
    
    # 计算统计信息
    warmup_times = execution_times[:warmup]
    active_times = execution_times[warmup:]
    
    warmup_avg = np.mean(warmup_times) if warmup_times else 0
    active_avg = np.mean(active_times) if active_times else 0
    warmup_std = np.std(warmup_times) if warmup_times else 0
    active_std = np.std(active_times) if active_times else 0
    


def run_detailed_profiler(iters: int, warmup: int):
    """详细的NPU profiler分析，包含更多调试信息"""
    case = build_case_tensors()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"detailed_npu_profiler_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Detailed NPU Profiler] Starting detailed profiling")
    print(f"[Detailed NPU Profiler] Output directory: {output_dir}")
    
    # 记录每次调用的详细信息
    call_details = []
    
    # 配置NPU profiler
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None
    )
    
    # 创建NPU profiler
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=warmup,
            active=iters,
            repeat=1,
            skip_first=0
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        with_flops=True,
        experimental_config=experimental_config
    )
    
    # 开始profiling
    prof.start()
    
    for step in range(warmup + iters):
        step_start = time.time()
        
        attention_out, fused_block_position_out, fused_max_page_position_length_out = torch_npu.npu_sparse_paged_fusion_attention(
            case["query"], case["key"], case["value"],
            case["block_table"], case["l1_cent"], case["block_ids"], case["total_seq_len"],
            actual_seq_lengths=case["actual_seq_lengths"],
            num_heads=case["num_heads"], scale_value=case["scale_value"], 
            input_layout=case["layout"], num_key_value_heads=case["num_key_value_heads"],
            block_size=case["block_size"], inner_precise=case["inner_precise"]
        )
        
        step_end = time.time()
        step_duration = (step_end - step_start) * 1000  # 转换为毫秒
        
        # 记录每次调用的详细信息
        call_info = {
            "step": step,
            "duration_ms": step_duration,
            "is_warmup": step < warmup,
            "tensor_shapes": {
                "attention_out": list(attention_out.shape),
                "block_position": list(fused_block_position_out.shape),
                "max_page_length": list(fused_max_page_position_length_out.shape)
            },
            "tensor_devices": {
                "attention_out": str(attention_out.device),
                "block_position": str(fused_block_position_out.device),
                "max_page_length": str(fused_max_page_position_length_out.device)
            }
        }
        call_details.append(call_info)
        
        # 每步打印详细信息
        if step < warmup:
            print(f"[Detailed NPU Profiler] Warmup Step {step}: {step_duration:.3f}ms")
        else:
            active_step = step - warmup
            print(f"[Detailed NPU Profiler] Active Step {active_step}: {step_duration:.3f}ms")
        
        # 调用prof.step()
        prof.step()
    
    # 停止profiling
    prof.stop()
    
    # 保存详细的调用信息
    detailed_results = {
        "timestamp": timestamp,
        "config": {
            "iters": iters,
            "warmup": warmup,
            "output_dir": output_dir
        },
        "call_details": call_details,
        "statistics": {
            "warmup_avg_ms": np.mean([c["duration_ms"] for c in call_details if c["is_warmup"]]),
            "active_avg_ms": np.mean([c["duration_ms"] for c in call_details if not c["is_warmup"]]),
            "warmup_std_ms": np.std([c["duration_ms"] for c in call_details if c["is_warmup"]]),
            "active_std_ms": np.std([c["duration_ms"] for c in call_details if not c["is_warmup"]])
        }
    }
    
    # 保存详细结果
    with open(os.path.join(output_dir, "detailed_npu_profiler_results.json"), "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n[Detailed NPU Profiler] Profiling completed!")
    print(f"[Detailed NPU Profiler] Results saved to: {output_dir}")
    print(f"[Detailed NPU Profiler] Warmup avg: {detailed_results['statistics']['warmup_avg_ms']:.3f}ms")
    print(f"[Detailed NPU Profiler] Active avg: {detailed_results['statistics']['active_avg_ms']:.3f}ms")
    print(f"[Detailed NPU Profiler] To view results: tensorboard --logdir={output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["profiler", "detailed"], default="profiler")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "profiler":
        run_perf(args.iters, args.warmup)  # 使用profiler版本的run_perf
    else: # args.mode == "detailed":
        run_detailed_profiler(args.iters, args.warmup)  # 使用详细profiler
    


