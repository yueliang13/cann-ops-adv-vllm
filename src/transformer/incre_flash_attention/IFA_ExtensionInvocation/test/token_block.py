import torch
import time
import numpy as np

SENTINEL_T = torch.tensor(0x7FFFFFFF, dtype=torch.int32)

def tokens_to_blocks_torch(token_position, block_size, max_block_num_per_batch, max_block_num_per_seq, ignore_value=-1):
    """
    将 Token-Position 映射为 Block-Table 与 Block-Position。
    - 输入 token_position: int32 tensor [B, Hkv, Ttok]
    - 输出：
      block_table: int32 tensor [B, max_block_num_per_batch]
      block_position: int32 tensor [B, Hkv, max_block_num_per_seq]
    说明：
    - 物理块ID = page_id = token // block_size
    - block_position 存储的是 block_table 的位置索引，未用填 0x7FFFFFFF
    - 为提升性能，在 CPU 上用 NumPy 做有序去重，然后将输出搬回原设备
    """
    B, Hkv, Ttok = token_position.shape
    out_device = token_position.device

    # 转 CPU + NumPy
    tokens_cpu = token_position.detach().to("cpu").to(torch.int64).numpy()  # [B, Hkv, Ttok]

    block_table = np.zeros((B, max_block_num_per_batch), dtype=np.int32)
    block_position = np.full((B, Hkv, max_block_num_per_seq), np.int32(SENTINEL_T.item()), dtype=np.int32)

    for b in range(B):
        pages_b = (tokens_cpu[b] // block_size).astype(np.int64)  # [Hkv, Ttok]

        # 1) union_pages: 按时间优先（t-major）的首次出现顺序
        # 重排为 [Ttok, Hkv] 再展平成 1D，保持按 t->h 的顺序
        seq = pages_b.transpose(1, 0).reshape(-1)
        # 过滤无效项（<0），ignore_value 保证 token<0；pages<0 即无效
        seq = seq[seq >= 0]
        if seq.size > 0:
            uniq_vals, first_idx = np.unique(seq, return_index=True)
            order = np.argsort(first_idx)
            union_pages = uniq_vals[order].astype(np.int32)
        else:
            union_pages = np.empty((0,), dtype=np.int32)

        if union_pages.size > max_block_num_per_batch:
            raise ValueError(f"Batch {b}: 需要的页数 {union_pages.size} > 上限 {max_block_num_per_batch}")

        block_table[b, :union_pages.size] = union_pages

        # page_id -> 在 block_table 的位置索引
        page_to_btpos = {int(p): i for i, p in enumerate(union_pages.tolist())}

        # 2) 每头的首次出现顺序，写入 block_position
        for h in range(Hkv):
            head_seq = pages_b[h]
            head_seq = head_seq[head_seq >= 0]
            if head_seq.size == 0:
                    continue
            uniq_h, first_idx_h = np.unique(head_seq, return_index=True)
            order_h = np.argsort(first_idx_h)
            head_pages = uniq_h[order_h]

            # 写入，按上限截断
            write_len = min(head_pages.size, max_block_num_per_seq)
            if write_len > 0:
                # 将 page 转换为 block_table 位置索引
                idxs = [page_to_btpos[int(p)] for p in head_pages[:write_len]]
                block_position[b, h, :write_len] = np.array(idxs, dtype=np.int32)
            # 其余位置保持哨兵值

    # 搬回原设备
    block_table_t = torch.from_numpy(block_table).to(out_device)
    block_position_t = torch.from_numpy(block_position).to(out_device)
    return block_table_t, block_position_t

def generate_token_position(B:int, Hkv:int, Ttok:int, total_seq_len_kv:int, pattern:str = "random", device:str = "cpu", ignore_value:int = -1) -> torch.Tensor:
    """
    生成 token_position: int32 tensor [B, Hkv, Ttok]
    - pattern="contiguous": 每个头取从0开始的连续 4K token（可为每个头设置不同偏移）
    - pattern="random": 每个头随机选取不重复的 4K token（范围 [0, total_seq_len_kv)）
    未用位置不填充（本场景满 4K），如需可扩展为 ignore_value
    """
    assert Ttok > 0 and Ttok <= total_seq_len_kv, "Ttok 必须在 (0, total_seq_len_kv]"

    # 统一在 CPU 构造，再按需搬到 NPU
    device_obj = torch.device("cpu")

    if pattern == "contiguous":
        base = torch.arange(Ttok, dtype=torch.int64, device=device_obj)
        tokens_h = base.unsqueeze(0).repeat(Hkv, 1)  # [Hkv, Ttok]
    elif pattern == "random":
        tokens_h = []
        for _ in range(Hkv):
            idx = torch.randperm(total_seq_len_kv, device=device_obj)[:Ttok]
            idx, _ = torch.sort(idx)
            tokens_h.append(idx)
        tokens_h = torch.stack(tokens_h, dim=0)  # [Hkv, Ttok]
    else:
        raise ValueError(f"未知 pattern: {pattern}")

    token_position = tokens_h.unsqueeze(0).to(torch.int32)  # [1, Hkv, Ttok]

    if device == "npu":
        if hasattr(torch, "npu"):
            try:
                token_position = token_position.npu()
            except Exception:
                pass
    return token_position

def count_page_stats(token_position: torch.Tensor, block_size: int, ignore_value: int = -1):
    """
    基于 token_position 统计：
    - 每个头的唯一页数量列表
    - 全 batch（此处 B=1）并集页数量
    返回 (per_head_pages: List[int], union_pages: int)
    """
    B, Hkv, Ttok = token_position.shape
    assert B == 1, "当前统计函数假设 B=1"
    per_head_pages = []
    union_pages_set = set()
    for h in range(Hkv):
        toks = token_position[0, h].to("cpu")
        valid = toks[toks >= 0]
        pages = torch.div(valid.to(torch.int64), block_size, rounding_mode='floor')
        unique_pages = torch.unique(pages).tolist()
        per_head_pages.append(len(unique_pages))
        union_pages_set.update(unique_pages)
    return per_head_pages, len(union_pages_set)

# 新增：验证函数
def validate_mapping(token_position: torch.Tensor, block_table: torch.Tensor, block_position: torch.Tensor, block_size: int, ignore_value: int = -1):
    """
    验证以下性质：
    1) union_pages（按 t->h 首次出现顺序）与 block_table 前缀一致
    2) 每个头的首次出现页序（可能被截断）等于 block_position 映射到 block_table 后的页序
    3) block_position 的非哨兵索引落在 block_table 已用范围内
    """
    B, Hkv, Ttok = token_position.shape
    assert B == 1, "仅验证 B=1 场景"
    device = token_position.device

    # 转 CPU + NumPy
    tok = token_position.detach().to("cpu").to(torch.int64).numpy()[0]       # [Hkv, Ttok]
    bt = block_table.detach().to("cpu").to(torch.int64).numpy()[0]            # [max_block_num_per_batch]
    bp = block_position.detach().to("cpu").to(torch.int64).numpy()[0]         # [Hkv, max_block_num_per_seq]

    pages = (tok // block_size).astype(np.int64)
    # union 按 t->h 顺序
    seq = pages.transpose(1, 0).reshape(-1)
    seq = seq[seq >= 0]
    if seq.size > 0:
        uniq_vals, first_idx = np.unique(seq, return_index=True)
        order = np.argsort(first_idx)
        union_pages = uniq_vals[order]
    else:
        union_pages = np.empty((0,), dtype=np.int64)

    # block_table 前缀应该等于 union_pages
    used_len = union_pages.size
    assert np.array_equal(bt[:used_len], union_pages), "block_table 的前缀不等于批级 union 页序"

    # 建立 page->idx
    page_to_idx = {int(p): i for i, p in enumerate(bt[:used_len].tolist())}

    # 校验每头
    for h in range(Hkv):
        head = pages[h]
        head = head[head >= 0]
        if head.size == 0:
            # 全哨兵
            assert np.all(bp[h] == np.int64(SENTINEL_T.item())), f"头{h} 应全为哨兵"
                    continue
        uniq_h, first_idx_h = np.unique(head, return_index=True)
        order_h = np.argsort(first_idx_h)
        head_pages = uniq_h[order_h]

        # 从 block_position 恢复页序
        valid_mask = bp[h] != np.int64(SENTINEL_T.item())
        idxs = bp[h][valid_mask]
        # 所有索引应落在 used_len 之内
        assert np.all((idxs >= 0) & (idxs < used_len)), f"头{h} 出现越界索引"
        rec_pages = np.array([bt[i] for i in idxs], dtype=np.int64)

        # 期望序列按截断比较
        expect = head_pages[:rec_pages.size]
        assert np.array_equal(rec_pages, expect), f"头{h} 页序与期望不一致"

    print("[验证] token->page 映射、block_table/position 一致性检查通过。")

def benchmark_conversion():
    # 配置：每头 4K token，page_size=16，总 KV 长度 32K，对应 2048 个页
    batch_size = 1
    Hkv = 32
    Ttok = 4096
    block_size = 16
    total_seq_len_kv = 32 * 1024
    max_block_num_per_batch = total_seq_len_kv // block_size  # 2048

    # 设备选择（默认 CPU；如需 NPU 可手动改为 "npu" 且环境可用）
    device = "npu"

    # 检查并初始化 NPU 环境
    use_npu = False
    if device == "npu":
        try:
            import torch_npu  # noqa: F401
            if hasattr(torch, "npu"):
                torch.npu.set_device(0)
                use_npu = True
        except Exception as e:
            print("[警告] 未检测到可用的 NPU 或 torch_npu 未正确安装。请先执行: 'conda activate ascend'，并确保 PyTorch NPU 环境就绪。将回退到 CPU。")
            use_npu = False

    # 生成两种场景的 token_position：连续型 与 随机型
    for pattern in ["contiguous", "random"]:
        gen_device = "npu" if use_npu else "cpu"
        token_position = generate_token_position(batch_size, Hkv, Ttok, total_seq_len_kv, pattern=pattern, device=gen_device)
        per_head_pages, union_pages = count_page_stats(token_position, block_size)

        # 设定 block_position 的最后一维：按每头唯一页数最大值
        max_block_num_per_seq = min(max(per_head_pages), max_block_num_per_batch)

        # 预热
        for _ in range(2):
            _ = tokens_to_blocks_torch(token_position, block_size, max_block_num_per_batch, max_block_num_per_seq)

        # 基准
        t0 = time.perf_counter()
        block_table, block_position = tokens_to_blocks_torch(token_position, block_size, max_block_num_per_batch, max_block_num_per_seq)
        t1 = time.perf_counter()

        # 验证正确性
        validate_mapping(token_position, block_table, block_position, block_size)

        elapsed_ms = (t1 - t0) * 1000.0
        avg_head_pages = sum(per_head_pages) / len(per_head_pages)

        out_dev_str = str(block_table.device)
        print(f"\n[模式={pattern}] 目标设备={'npu' if use_npu else 'cpu'} 实际输出设备={out_dev_str}")
        print(f"token_position.shape={tuple(token_position.shape)}, dtype={token_position.dtype}")
        print(f"block_table.shape={tuple(block_table.shape)}, block_position.shape={tuple(block_position.shape)}")
        print(f"每头唯一页数(均值/最小/最大)={avg_head_pages:.1f}/{min(per_head_pages)}/{max(per_head_pages)}，并集页数={union_pages}")
        print(f"转换耗时: {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    benchmark_conversion()