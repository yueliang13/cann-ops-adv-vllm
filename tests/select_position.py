import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys, os
import custom_ops

torch.npu.config.allow_internal_format = False

def compute_token_position_torch(key_ids: torch.Tensor, indices: torch.Tensor, max_selected_len: int = 8192) -> torch.Tensor:
    """
    参考select逻辑，计算每个(b, h)在序列维上的匹配位置索引。
    
    入参：
    - key_ids: [B, H, S]，序列上的key id
    - indices: [B, 1, H, K]，每个head的候选centroid id集合（nprobe/K）
    - max_selected_len: 输出token_position的最大长度
    
    返回：
    - token_position: [B, H, max_selected_len]，未填充部分为-1
    """
    assert key_ids.dim() == 3, f"key_ids shape invalid: {key_ids.shape}"
    assert indices.dim() == 4 and indices.shape[1] == 1, f"indices shape invalid: {indices.shape}"
    B, H, S = key_ids.shape
    _, _, H2, K = indices.shape
    assert H2 == H, "indices的head数与key_ids不一致"

    device = key_ids.device
    dtype = torch.int32
    pad_value = torch.tensor(2147483647, dtype=dtype, device=device)

    token_position = torch.full((B, H, max_selected_len), pad_value, dtype=dtype, device=device)
    token_position_length = torch.zeros((B, H), dtype=torch.int32, device=device)

    # 遍历batch/head（S和K采用张量算子）
    for b in range(B):
        key_b = key_ids[b]            # [H, S]
        idx_b = indices[b, 0]         # [H, K]
        for h in range(H):
            key_vec = key_b[h]        # [S]
            cent_vals = idx_b[h]      # [K]
            # 逐位置匹配：key_vec中的值是否在cent_vals集合中
            mask = (key_vec.unsqueeze(-1) == cent_vals.unsqueeze(0)).any(dim=-1)  # [S]
            matched_idx = mask.nonzero(as_tuple=False).squeeze(-1)   
                         # [num_matched]
            if matched_idx.numel() > 0:
                length = min(matched_idx.numel(), max_selected_len)
                token_position[b, h, :length] = matched_idx[:length].to(dtype)
                token_position_length[b, h] = length

    return token_position, token_position_length

class TestCustomAdd(TestCase):
    def test_add_custom_ops(self):
        torch.npu.set_device(0)
        
        batch = 8
        n1 = 32
        n2 = 8
        g = n1 // n2
        c = 512
        k = 64
        dim = 128
        seqLen = 32*1024
        maxpageNum = 10*1024
        
        blockIdsShape = [batch, n1, seqLen]
        pagePositionShape = [batch, n1, maxpageNum]
        indicesShape = [batch, n1, k]
        dl1Shape = [batch, n1, c]
        pagePositionLengthShape = [batch, n1, 8]

        print("blockIdsShape:", blockIdsShape)
        print("pagePositionShape:", pagePositionShape)
        print("indicesShape:", indicesShape)
        print("dl1Shape:", dl1Shape)
        print("pagePositionLengthShape:", pagePositionLengthShape)
        d_l1_cent = torch.rand(dl1Shape, device='cpu', dtype=torch.float)
        _, torch_indices = torch.topk(d_l1_cent, k, dim=-1)
        indices = torch_indices.to(torch.int32)
        print("indices shape:", indices.shape)
        # indices = torch.randint(0, c, indicesShape, device='cpu', dtype=torch.int32)

        key_ids = torch.randint(0, c, blockIdsShape, device='cpu', dtype=torch.int32)
        page_position = torch.zeros(pagePositionShape, device='cpu', dtype=torch.int32)
        page_position_length = torch.zeros(pagePositionLengthShape, device='cpu', dtype=torch.int32)
            
        indices_npu = indices.npu()
        key_ids_npu = key_ids.npu()

        page_position_npu, page_position_length_npu = custom_ops.select_position(key_ids_npu, indices_npu)

        # 计算PyTorch参考实现的page_position并比较
        # page_position_ref, page_position_length_ref = compute_page_position_torch(key_ids, indices, max_selected_len=maxpageNum)

        print("page_position_length_npu shape:", page_position_length_npu.shape)
        print("page_position_length_npu:", page_position_length_npu)
        # print("page_position_length_ref:", page_position_length_ref)
      
        # 算子写回的结果在page_position_npu中
        page_position_out = page_position_npu.cpu()
        print("page_position_out shape:", page_position_out.shape)
        print("page_position_out:", page_position_out)

        # 对齐dtype与形状（参考实现为-1填充，算子可能填0，做统一处理可选：此处直接比较整数索引序列部分）
        # 仅比较前若干个非-1位置，或直接逐元素比较（若算子与参考一致应完全相等）
        # is_equal = torch.equal(page_position_ref.to(torch.int32), page_position_out.to(torch.int32))
        # print(f"page position equality (PyTorch vs Operator): {is_equal}")
        # if not is_equal:
        #     # 打印首个不一致位置帮助定位
        #     diff = (page_position_ref.to(torch.int32) != page_position_out.to(torch.int32))
        #     where = diff.nonzero(as_tuple=False)
        #     if where.numel() > 0:
        #         b, h, t = where[0].tolist()
        #         print(f"First mismatch at [b={b}, h={h}, t={t}] ref={page_position_ref[b,h,t]}, op={page_position_out[b,h,t]}")

if __name__ == "__main__":
    run_tests()