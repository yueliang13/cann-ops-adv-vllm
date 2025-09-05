import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys, os
import custom_ops
from utils import torch_select_position, torch_compute_cent, compare_tensors

torch.npu.config.allow_internal_format = False

class TestCustomAdd(TestCase):
    def test_add_custom_ops(self):
        torch.npu.set_device(0)
        
        batch = 4
        n1 = 32
        n2 = 8
        g = n1 // n2
        c = 512
        k = 64
        dim = 128
        blockSize = 128
        seqLen = 32*1024
        kvPageLen = 1280
        maxBatch = 256
        maxpageNum = 1024
        pageLen = (seqLen + (blockSize - 1)) // blockSize
        
        qShape = [batch, n1, dim]
        l1Shape = [n2, c, dim]
        dl1Shape = [batch, n1, 1, c]
        blockIdsShape = [n2, kvPageLen]
        blockTableShape = [maxBatch, maxpageNum]
        indicesShape = [batch, n1, k]
        seqLenShape = [batch]
        pagePositionShape = [batch, n1, 256]
        pagePositionLengthShape = [batch, n1, 8]

        print("qShape:", qShape)
        print("l1Shape:", l1Shape)
        print("blockIdsShape:", blockIdsShape)
        print("blockTableShape:", blockTableShape)
        print("seqLenShape:", seqLenShape)
        print("indicesShape:", indicesShape)

        q = torch.rand(qShape, device='cpu', dtype=torch.float16)
        l1 = torch.rand(l1Shape, device='cpu', dtype=torch.float16)
        block_ids = torch.randint(0, c, blockIdsShape, device='cpu', dtype=torch.int32)
        block_table = torch.randint(0, kvPageLen, blockTableShape, device='cpu', dtype=torch.int32)
        seq_len = torch.full(seqLenShape, seqLen, device='cpu', dtype=torch.int32)

        q_npu = q.npu()
        l1_npu = l1.npu()
        block_ids_npu = block_ids.npu()
        block_table_npu = block_table.npu()
        seq_len_npu = seq_len.npu()

        page_position, page_position_length = custom_ops.cent_select(q_npu, l1_npu, block_ids_npu, block_table_npu, seq_len_npu)

        torch_indices = torch_compute_cent(q_npu, l1_npu)
        torch_page_position, torch_page_position_length, torch_indices_full = torch_select_position(block_ids_npu, block_table_npu, seq_len_npu, torch_indices, block_size=blockSize)

        # print("page_position:", page_position)
        # print("torch_page_position:", torch_page_position)
        # print("page_position_length:", page_position_length[:,:,0])
        # print("torch_page_position_length:", torch_page_position_length)

        compare_tensors(page_position, torch_page_position, context="page_position", check_shape=False, check_dtype=False)
        print("page_position_length:", page_position_length[:,:,0])
        print("torch_page_position_length:", torch_page_position_length)
        compare_tensors(page_position_length[:,:,0], torch_page_position_length, context="page_position_length", check_shape=False, check_dtype=False)
        # compare_tensors(indices_npu, torch_indices_full, context="indices", check_shape=False, check_dtype=False)

if __name__ == "__main__":
    run_tests()