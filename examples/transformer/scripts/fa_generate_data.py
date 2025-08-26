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

import numpy as np
import sys

case_name = sys.argv[1]
if case_name == 'test_flash_attention_score':
    query = np.random.uniform(-0.1, 0.1, (2048, 1, 128)).astype(np.float16)
    key = np.random.uniform(-0.1, 0.1, (2048, 1, 128)).astype(np.float16)
    value = np.random.uniform(-0.1, 0.1, (2048, 1, 128)).astype(np.float16)
    query.tofile('query.bin')
    key.tofile('key.bin')
    value.tofile('value.bin')

elif case_name == 'test_flash_attention_varLen_score':
    query = np.random.uniform(-0.1, 0.1, (256, 4, 128)).astype(np.float16)
    key = np.random.uniform(-0.1, 0.1, (256, 4, 128)).astype(np.float16)
    value = np.random.uniform(-0.1, 0.1, (256, 4, 128)).astype(np.float16)
    query.tofile('query.bin')
    key.tofile('key.bin')
    value.tofile('value.bin')

elif case_name == 'test_flash_attention_score_grad':
    query = np.random.uniform(-0.1, 0.1, (256, 1, 512)).astype(np.float16)
    key = np.random.uniform(-0.1, 0.1, (256, 1, 512)).astype(np.float16)
    value = np.random.uniform(-0.1, 0.1, (256, 1, 512)).astype(np.float16)
    dx = np.random.uniform(-0.1, 0.1, (256, 1, 512)).astype(np.float16)
    attentionIn = np.random.uniform(-0.1, 0.1, (256, 1, 512)).astype(np.float16)
    softmaxMax = np.random.uniform(-0.1, 0.1, (1, 4, 256, 1)).astype(np.float32)
    softmaxMax = np.repeat(softmaxMax, 8, axis=-1)
    softmaxSum = np.random.uniform(0, 1, (1, 4, 256, 1)).astype(np.float32)
    softmaxSum = np.repeat(softmaxSum, 8, axis=-1)
    query.tofile('query.bin')
    key.tofile('key.bin')
    value.tofile('value.bin')
    dx.tofile('dx.bin')
    attentionIn.tofile('attentionIn.bin')
    softmaxMax.tofile('softmaxMax.bin')
    softmaxSum.tofile('softmaxSum.bin')

elif case_name == 'test_flash_attention_unpadding_score_grad':
    query = np.random.uniform(-0.1, 0.1, (256, 4, 128)).astype(np.float16)
    key = np.random.uniform(-0.1, 0.1, (256, 4, 128)).astype(np.float16)
    value = np.random.uniform(-0.1, 0.1, (256, 4, 128)).astype(np.float16)
    dx = np.random.uniform(-0.1, 0.1, (256, 4, 128)).astype(np.float16)
    attentionIn = np.random.uniform(-0.1, 0.1, (256, 4, 128)).astype(np.float16)
    softmaxMax = np.random.uniform(-0.1, 0.1, (256, 4, 1)).astype(np.float32)
    softmaxMax = np.repeat(softmaxMax, 8, axis=-1)
    softmaxSum = np.random.uniform(0, 1, (256, 4, 1)).astype(np.float32)
    softmaxSum = np.repeat(softmaxSum, 8, axis=-1)
    query.tofile('query.bin')
    key.tofile('key.bin')
    value.tofile('value.bin')
    dx.tofile('dx.bin')
    attentionIn.tofile('attentionIn.bin')
    softmaxMax.tofile('softmaxMax.bin')
    softmaxSum.tofile('softmaxSum.bin')
else:
    raise RuntimeError(f"Invalid case name:", case_name)
