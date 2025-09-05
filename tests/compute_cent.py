import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys, os
import custom_ops
from utils import torch_compute_cent, compare_tensors

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
        # BN1SD
        qShape = [batch, n1, dim]
        # N2CD
        l1Shape = [n2, c, dim]
        # BN1SC
        dl1Shape = [batch, n1, c]

        maxPage = 8192
        ivfStartShape = [batch, n1, c]
        ivfLenShape = [batch, n1, c]
        blockClusterTableShape = [batch, n1, maxPage]

        selectNprobeShape = [batch, n1, k]
        indicesShape = [batch, n1, k]

        # # BSN2GD
        # qShape = [batch,1, n2, g, dim]
        # # N2DC
        # l1Shape = [n2, dim, c]
        # BSN2GC
        # dl1Shape = [batch, 1, n2, g, c]

        print("qShape:", qShape)
        print("l1Shape:", l1Shape)
        print("d_l1_centShape:", dl1Shape)

        q = torch.rand(qShape, device='cpu', dtype=torch.float16)
        l1_cent = torch.rand(l1Shape, device='cpu', dtype=torch.float16)
        indices = torch.zeros(indicesShape, device='cpu', dtype=torch.int32)

        q_npu = q.npu()
        l1_cent_npu = l1_cent.npu()
        output = custom_ops.compute_cent(q_npu, l1_cent_npu)

        print("output shape:", output.shape)
        print("output:", output)

        # 使用封装后的torch函数
        torch_output = torch_compute_cent(q, l1_cent)
        torch_output_npu = torch_output.npu()

        print("output shape:", output.shape)
        print("output:", output)
        print("torch_output shape:", torch_output.shape)
        print("torch_output:", torch_output)

        compare_tensors(output, torch_output_npu, check_shape=False, check_dtype=False)

        # # 比较结果
        # diff_indices = torch.abs(output - torch_output_npu)
        # top_values, top_indices = torch.topk(diff_indices.flatten(), 64)
        # print("indices Top 100 differences:", top_values)


if __name__ == "__main__":
    run_tests()