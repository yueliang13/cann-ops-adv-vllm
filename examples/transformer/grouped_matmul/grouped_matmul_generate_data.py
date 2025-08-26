#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import sys
import numpy as np

def gen_tensor_list(low, high, shape_list, name, dtype=np.float16):
    testCase = sys.argv[1]
    for index, shape in enumerate(shape_list) :
        tensor = np.random.uniform(low, high, shape).astype(dtype)
        tensor.tofile(f"{testCase}_{name}_{index}.bin")     

if __name__ == '__main__':
    testCase = sys.argv[1]
    if testCase == 'grouped_matmul_v2':
        gen_tensor_list(-0.1, 0.1, [(1, 16), (4, 32)], 'x')
        gen_tensor_list(-0.1, 0.1, [(16, 24), (32, 16)], 'weight')
        gen_tensor_list(-0.1, 0.1, [(24), (16)], 'bias')
    if testCase == 'grouped_matmul_v3':
        gen_tensor_list(-1, 1, [(16, 128)], 'x', np.float16)
        gen_tensor_list(-128, 128, [(4, 128, 1024)], 'weight', np.int8)
        gen_tensor_list(-0.5, 0.5, [(4, 1024)], 'bias', np.float16)
        gen_tensor_list(-0.05, 0.05, [(4, 1024)], 'antiquant_scale', np.float16)
        gen_tensor_list(-2, 2, [(4, 1024)], 'antiquant_offset', np.float16)
    if testCase == 'grouped_matmul_v4':
        gen_tensor_list(-128, 128, [(32, 5)], 'x', np.int8)
        gen_tensor_list(-128, 128, [(2, 5, 10)], 'weight', np.int8)
        gen_tensor_list(-256, 256, [(2, 10)], 'bias', np.int32)
        gen_tensor_list(-0.05, 0.05, [(2, 10)], 'scale', np.float32)
        gen_tensor_list(-0.1, 0.1, [(32)], 'pertoken_scale', np.float32)