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
import os

if __name__ == '__main__':
    test_case = sys.argv[1]
    print_head = "====================== Show sample " + test_case + " result start ================="
    file = test_case + '_y_0.bin'
    if not os.path.isfile(file):
        raise RuntimeError(f"Invalid case name:", test_case)
    y = np.fromfile(file, dtype=np.float16)
    print(f"{test_case} output[0]: ", y)
    file = test_case + '_y_1.bin'
    if  os.path.isfile(file):
        y = np.fromfile(file, dtype=np.float16)
        print(f"{test_case} output[1]: ", y)
