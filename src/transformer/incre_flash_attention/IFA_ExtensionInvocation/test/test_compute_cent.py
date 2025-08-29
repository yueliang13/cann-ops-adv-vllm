import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys, os
import custom_ops
torch.npu.config.allow_internal_format = False


class TestCustomAdd(TestCase):

    def test_add_custom_ops(self):
        torch.npu.set_device(0)
        
        batch = 1
        n1 = 32
        n2 = 8
        g = n1 // n2
        c = 256
        k = 64
        dim = 128
        # BN1SD
        qShape = [batch, n1, 1, dim]
        # N2CD
        l1Shape = [n2, c, dim]
        # BN1SC
        dl1Shape = [batch, n1, 1, c]

        maxPage = 8192
        ivfStartShape = [batch, n1, c]
        ivfLenShape = [batch, n1, c]
        blockClusterTableShape = [batch, n1, maxPage]

        selectNprobeShape = [batch, 1, n1, k]
        indicesShape = [batch, 1, n1, k]

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

        # indices_npu = output
        # torch
        # 为了实现乘法，需要为每个n1Idx找到对应的n2Idx的l1_cent数据
        # n2Idx = n1Idx / nNumOfQInOneGroup
        # 每个n1Idx对应一个n2Idx，所以需要分别处理每个n1Idx
        out = torch.zeros(dl1Shape, dtype=torch.float)
        
        for i in range(n1):
            n2_idx = i // g  # n2Idx = n1Idx / nNumOfQInOneGroup
            # 获取对应的l1_cent数据 [clusterNum, dimNum]
            l1_slice = l1_cent[n2_idx]  # shape: [512, 128]
            # 转置为 [dimNum, clusterNum] 以便进行矩阵乘法
            l1_slice_t = l1_slice.transpose(0, 1)  # shape: [128, 512]
            # 执行矩阵乘法 [1, 128] @ [128, 512] => [1, 512]
            q_slice = q[0, i, 0]  # shape: [128]
            out_slice = torch.matmul(q_slice.to(torch.float), l1_slice_t.to(torch.float))  # shape: [512]
            out[0, i, 0] = out_slice

        values, indices = torch.topk(out, k=k, dim=-1)
        values = values.transpose(1, 2)
        indices = indices.transpose(1, 2)


        diff_indices = torch.abs(output.npu() - indices.npu())
        top_values, top_indices = torch.topk(diff_indices.flatten(), 64)
        print("indices Top 100 differences:", top_values)


if __name__ == "__main__":
    run_tests()
