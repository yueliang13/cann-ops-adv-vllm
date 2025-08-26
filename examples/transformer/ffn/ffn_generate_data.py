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

if __name__ == '__main__':
    testCase = sys.argv[1]
    if testCase == 'ffn_v3_float16_high_precision':
        x = np.random.uniform(-0.1, 0.1, (64, 1024)).astype(np.float16)
        weight1 = np.random.uniform(-0.1, 0.1, (1024, 2048)).astype(np.float16)
        weight2 = np.random.uniform(-0.1, 0.1, (2048, 1024)).astype(np.float16)
        bias1 = np.random.uniform(-0.1, 0.1, (2048)).astype(np.float16)
    elif testCase == 'ffn_v2_antiquant':
        x = np.random.uniform(-1, 1, (64, 1024)).astype(np.float16)
        weight1 = np.random.randint(-128, 128, (4, 1024, 2048)).astype(np.int8)
        weight2 = np.random.randint(-128, 128, (4, 2048, 1024)).astype(np.int8)
        bias1 = np.random.uniform(-0.2, 0.2, (4, 2048)).astype(np.float16)
        antiquantScale1 = np.random.uniform(-0.0007, 0.0007, (4, 2048)).astype(np.float16)
        antiquantScale2 = np.random.uniform(-0.0007, 0.0007, (4, 1024)).astype(np.float16)
        antiquantOffset1 = np.random.uniform(-30, 30, (4, 2048)).astype(np.float16)
        antiquantOffset2 = np.random.uniform(-30, 30, (4, 1024)).astype(np.float16)
        antiquantScale1.tofile(f'{testCase}_antiquant_scale1.bin')
        antiquantScale2.tofile(f'{testCase}_antiquant_scale2.bin')
        antiquantOffset1.tofile(f'{testCase}_antiquant_offset1.bin')
        antiquantOffset2.tofile(f'{testCase}_antiquant_offset2.bin')
    elif testCase == 'ffn_v3_quant_token_index_flag':
        x = np.random.randint(-128, 128, (64, 1024)).astype(np.int8)
        weight1 = np.random.randint(-128, 128, (4, 1024, 2048)).astype(np.int8)
        weight2 = np.random.randint(-128, 128, (4, 2048, 1024)).astype(np.int8)
        bias1 = np.random.randint(-512, 512, (4, 2048)).astype(np.int32)
        scale = np.random.uniform(-0.1, 0.1, (4)).astype(np.float32)
        offset = np.random.uniform(-128, 128, (4)).astype(np.float32)
        deqScale1 = np.random.uniform(-0.2, 0.2, (4, 2048)).astype(np.float32)
        deqScale2 = np.random.uniform(-0.2, 0.2, (4, 1024)).astype(np.float32)
        scale.tofile(f'{testCase}_scale.bin')
        offset.tofile(f'{testCase}_offset.bin')
        deqScale1.tofile(f'{testCase}_deq_scale1.bin')
        deqScale2.tofile(f'{testCase}_deq_scale2.bin')
    else:
        raise ValueError(f'ERROR test case: {testCase}')
    x.tofile(f'{testCase}_x.bin')
    weight1.tofile(f'{testCase}_weight1.bin')
    weight2.tofile(f'{testCase}_weight2.bin')
    bias1.tofile(f'{testCase}_bias1.bin')
