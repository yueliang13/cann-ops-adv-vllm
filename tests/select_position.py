from cgi import print_exception
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys, os
import custom_ops
from utils import torch_select_position, compare_tensors

torch.npu.config.allow_internal_format = False


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
        blockSize = 128
        seqLen = 32*1024
        kvPageLen = 1280
        maxBatch = 256
        maxpageNum = 1024
        pageLen = (seqLen + (blockSize - 1)) // blockSize
        
        blockIdsShape = [n2, kvPageLen]
        blockTableShape = [maxBatch, maxpageNum]
        indicesShape = [batch, n1, k]
        seqLenShape = [batch]
        pagePositionShape = [batch, n1, 256]
        pagePositionLengthShape = [batch, n1, 8]

        print("blockIdsShape:", blockIdsShape)
        print("blockTableShape:", blockTableShape)
        print("seqLenShape:", seqLenShape)
        print("indicesShape:", indicesShape)

        block_ids = torch.randint(0, c, blockIdsShape, device='cpu', dtype=torch.int32)
        block_table = torch.randint(0, kvPageLen, blockTableShape, device='cpu', dtype=torch.int32)
        seq_len = torch.full(seqLenShape, seqLen, device='cpu', dtype=torch.int32)
        indices = torch.randint(0, c, indicesShape, device='cpu', dtype=torch.int32)
        # print("block_ids:", block_ids)
        # print("block_table:", block_table)
        # print("seq_len:", seq_len)
        print("indices[0][0]:", indices[0][0])

        block_ids_npu = block_ids.npu()
        block_table_npu = block_table.npu()
        seq_len_npu = seq_len.npu()
        indices_npu = indices.npu()

        # 自定义算子执行
        page_position_npu, page_position_length_npu, block_table_gather_npu = custom_ops.select_position(block_ids_npu, block_table_npu, seq_len_npu, indices_npu)

        # 参考实现执行
        ref_pos, ref_len, c_full = torch_select_position(block_ids_npu, block_table_npu, seq_len_npu, indices_npu, block_size=blockSize)

        compare_tensors(page_position_npu, ref_pos, context="page_position", check_shape=False, check_dtype=False)
        compare_tensors(page_position_length_npu[:,:,0], ref_len, context="page_position_length", check_shape=False, check_dtype=False)
        compare_tensors(block_table_gather_npu[:,:,:pageLen], c_full, context="block_table_gather", check_shape=False, check_dtype=False)


if __name__ == "__main__":
    run_tests()