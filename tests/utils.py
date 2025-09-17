import torch


def torch_select_position(
    block_ids: torch.Tensor,          # [H_kv, kvPageLen]
    block_table: torch.Tensor,        # [maxBatch, maxpageNum]
    seq_len: torch.Tensor,            # [B]
    indices: torch.Tensor,            # [B, H_q, K]
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    依据描述：
    1) 取真实 batch (B = seq_len.shape[0])，从 block_table 中取前 B 行；
    2) 对每个样本 b，用 ceil(seq_len[b]/block_size) 得到 page_len_b；
    3) 取 block_table[b, :page_len_b] 作为物理页索引，
       用其从 block_ids 收集得到 c[b, h_q, page_len_b]（h_q 映射到对应 kv 头）；
    4) 与 indices[b, h_q, k] 做 membership，得到 mask[b, h_q, page_len_b]；
    5) 输出 positions[b, h_q, L_b]，把 True 的位置下标依次写入，
       剩余位置（到 page_len_b）与未使用的尾部（到 max_page_len）填充 int32 最大值；
    6) 同时返回每个 (b,h_q) 的真实匹配数量 length[b,h_q]。
    返回：
      - token_position: [B, H_q, max_page_len] (int32, pad=INT32_MAX)
      - token_position_length: [B, H_q] (int32)
    """
    assert block_ids.dim() == 2, f"block_ids shape invalid: {block_ids.shape}"
    assert block_table.dim() == 2, f"block_table shape invalid: {block_table.shape}"
    assert seq_len.dim() == 1, f"seq_len shape invalid: {seq_len.shape}"
    assert indices.dim() == 3, f"indices shape invalid: {indices.shape}"

    device = block_ids.device
    dtype_i32 = torch.int32
    pad_val = torch.iinfo(dtype_i32).max

    B = seq_len.shape[0]
    H_kv, kvPageLen = block_ids.shape
    H_q, K = indices.shape[1], indices.shape[2]
    assert H_q % H_kv == 0, "num_heads 必须能被 num_kv_heads 整除"
    group_size = H_q // H_kv

    # 真实 batch 的 block_table 视图
    block_table_b = block_table[:B]

    # 各样本的页长度与全局最大页长度
    page_lens = torch.div(seq_len.to(torch.int64) + (block_size - 1), block_size, rounding_mode='floor')
    max_page_len = int(page_lens.max().item())

    # 结果张量
    token_position = torch.full((B, H_q, max_page_len), pad_val, dtype=dtype_i32, device=device)
    token_position_length = torch.zeros((B, H_q), dtype=dtype_i32, device=device)

    # 预先构造 Q->KV 头映射，并按该映射重排 block_ids，便于一次 gather
    map_q2kv = (torch.arange(H_q, device=device) // group_size).to(torch.long)   # [H_q]
    block_ids_qview = block_ids.index_select(0, map_q2kv)  
    # [H_q, kvPageLen]

    c_full = torch.zeros((B, H_q, max_page_len), dtype=block_ids.dtype, device=device)
    for b in range(B):
        pl = int(page_lens[b].item())
        if pl <= 0:
            continue
        # 取该样本有效页索引（裁剪到 kvPageLen，避免越界）
        page_idx = block_table_b[b, :pl].clamp_min(0).clamp_max(kvPageLen - 1)   # [pl]
        # c_b: [H_q, pl]
        c_b = block_ids_qview.index_select(1, page_idx)
        c_full[b, :, :pl] = c_b
        
        # indices_b: [H_q, K]
        indices_b = indices[b].to(c_b.dtype)
        # mask: [H_q, pl]
        mask = (c_b.unsqueeze(-1) == indices_b.unsqueeze(-2)).any(dim=-1)
        # print("mask[0]:", mask[0])

        # 为每个 head 写回匹配位置
        for h in range(H_q):
            pos = mask[h].nonzero(as_tuple=False).squeeze(-1)  # [L]
            if pos.numel() == 0:
                continue
            L = min(pos.numel(), pl)
            token_position[b, h, :L] = pos[:L].to(dtype_i32)
            token_position_length[b, h] = int(L)

        # 对未使用的尾部（pl: max_page_len）保持 pad_val，无需额外处理

    return token_position, token_position_length, c_full


def torch_compute_cent(q, l1_cent):
    """
    PyTorch实现的compute_cent函数
    
    Args:
        q: query tensor, shape [batch, n1, dim]
        l1_cent: l1 centroid tensor, shape [n2, c, dim] 
        
    Returns:
        indices: computed indices, shape [batch, n1, k]
    """
    batch, n1, dim = q.shape
    n2, c, _ = l1_cent.shape
    g = n1 // n2
    k = 64  # 固定k值
    
    # 初始化输出tensor
    dl1Shape = [batch, n1, c]
    out = torch.zeros(dl1Shape, dtype=torch.float, device=q.device)
    
    # 为每个n1Idx找到对应的n2Idx的l1_cent数据
    for b in range(batch):
        for i in range(n1):
            n2_idx = i // g  # n2Idx = n1Idx / nNumOfQInOneGroup
            # 获取对应的l1_cent数据 [clusterNum, dimNum]
            l1_slice = l1_cent[n2_idx]  # shape: [c, dim]
            # 转置为 [dimNum, clusterNum] 以便进行矩阵乘法
            l1_slice_t = l1_slice.transpose(0, 1)  # shape: [dim, c]
            # 执行矩阵乘法 [1, dim] @ [dim, c] => [1, c]
            q_slice = q[b, i]  # shape: [dim]
            out_slice = torch.matmul(q_slice.to(torch.float), l1_slice_t.to(torch.float))  # shape: [c]
            out[b, i] = out_slice

    # 执行topk操作
    values, indices = torch.topk(out, k=k, dim=-1)
    
    return indices


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor,
                   context: str = "",
                   tolerance: float = 1e-6, 
                   check_shape: bool = True,
                   check_dtype: bool = True,
                   check_device: bool = True,
                   verbose: bool = True) -> bool:
    """
    比较两个tensor的数据是否一致
    
    Args:
        tensor1: 第一个tensor
        tensor2: 第二个tensor
        tolerance: 数值容差，用于浮点数比较
        check_shape: 是否检查形状一致性
        check_dtype: 是否检查数据类型一致性
        check_device: 是否检查设备一致性
        verbose: 是否打印详细信息
        
    Returns:
        bool: 是否一致
    """
    if verbose:
        print(f"{context} : 比较kernel和torch的结果")
        print(f"           kernel out shape: {tensor1.shape}, dtype: {tensor1.dtype}, device: {tensor1.device}")
        print(f"           torch out shape: {tensor2.shape}, dtype: {tensor2.dtype}, device: {tensor2.device}")
    
    # 检查形状
    if check_shape and tensor1.shape != tensor2.shape:
        if verbose:
            print(f"{context}  形状不一致: {tensor1.shape} vs {tensor2.shape}")
        return False
    
    # 检查数据类型
    if check_dtype and tensor1.dtype != tensor2.dtype:
        if verbose:
            print(f"{context}  数据类型不一致: {tensor1.dtype} vs {tensor2.dtype}")
        return False
    
    # 检查设备
    if check_device and tensor1.device != tensor2.device:
        if verbose:
            print(f"{context}  设备不一致: {tensor1.device} vs {tensor2.device}")
        return False
    
    # 确保tensor在同一设备上进行比较
    if tensor1.device != tensor2.device:
        tensor1_cpu = tensor1.cpu()
        tensor2_cpu = tensor2.cpu()
    else:
        tensor1_cpu = tensor1
        tensor2_cpu = tensor2
    
    # 检查数值一致性
    if tensor1_cpu.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        # 浮点数比较，使用容差
        diff = torch.abs(tensor1_cpu - tensor2_cpu)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        if verbose:
            print(f"           最大差值: {max_diff}")
            print(f"           平均差值: {mean_diff}")
            print(f"           容差: {tolerance}")
        
        is_equal = torch.all(diff <= tolerance)
    else:
        # 整数比较，直接相等
        is_equal = torch.equal(tensor1_cpu, tensor2_cpu)
        if verbose:
            print(f"           整数比较结果: {is_equal}")
    
    if verbose:
        print(f"           比较结果: {'一致' if is_equal else '不一致'}")
    
    return is_equal


def compare_tensors_detailed(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                            tolerance: float = 1e-6,
                            max_diff_count: int = 10) -> dict:
    """
    详细比较两个tensor，返回详细的比较信息
    
    Args:
        tensor1: 第一个tensor
        tensor2: 第二个tensor
        tolerance: 数值容差
        max_diff_count: 最多显示多少个不同的元素
        
    Returns:
        dict: 包含详细比较信息的字典
    """
    result = {
        'is_equal': False,
        'shape_match': False,
        'dtype_match': False,
        'device_match': False,
        'value_match': False,
        'max_diff': 0.0,
        'mean_diff': 0.0,
        'diff_count': 0,
        'diff_indices': [],
        'diff_values': []
    }
    
    # 基本信息比较
    result['shape_match'] = tensor1.shape == tensor2.shape
    result['dtype_match'] = tensor1.dtype == tensor2.dtype
    result['device_match'] = tensor1.device == tensor2.device
    
    if not result['shape_match']:
        return result
    
    # 确保在同一设备上比较
    if tensor1.device != tensor2.device:
        tensor1_cpu = tensor1.cpu()
        tensor2_cpu = tensor2.cpu()
    else:
        tensor1_cpu = tensor1
        tensor2_cpu = tensor2
    
    # 数值比较
    if tensor1_cpu.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        diff = torch.abs(tensor1_cpu - tensor2_cpu)
        result['max_diff'] = torch.max(diff).item()
        result['mean_diff'] = torch.mean(diff).item()
        result['value_match'] = torch.all(diff <= tolerance)
        
        # 找到不同的位置
        diff_mask = diff > tolerance
        result['diff_count'] = torch.sum(diff_mask).item()
        
        if result['diff_count'] > 0:
            diff_indices = torch.nonzero(diff_mask, as_tuple=True)
            diff_values = diff[diff_mask]
            
            # 取前max_diff_count个
            count = min(result['diff_count'], max_diff_count)
            result['diff_indices'] = [tuple(idx) for idx in zip(*[ind[:count] for ind in diff_indices])]
            result['diff_values'] = diff_values[:count].tolist()
    else:
        # 整数比较
        result['value_match'] = torch.equal(tensor1_cpu, tensor2_cpu)
        if not result['value_match']:
            diff_mask = tensor1_cpu != tensor2_cpu
            result['diff_count'] = torch.sum(diff_mask).item()
            
            if result['diff_count'] > 0:
                diff_indices = torch.nonzero(diff_mask, as_tuple=True)
                count = min(result['diff_count'], max_diff_count)
                result['diff_indices'] = [tuple(idx) for idx in zip(*[ind[:count] for ind in diff_indices])]
                result['diff_values'] = [tensor1_cpu[diff_indices[0][i], diff_indices[1][i]].item() 
                                       for i in range(count)]
    
    result['is_equal'] = result['shape_match'] and result['value_match']
    
    return result


