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
import multiprocessing


def show():
    case_name = sys.argv[1]
    print_head = "=========================================== Show sample " + case_name + \
                 " result start ================="
    print(print_head)
    if case_name == 'test_flash_attention_score' or case_name == 'test_flash_attention_varLen_score':
        softmax_max = np.fromfile('softmaxMax.bin', dtype=np.float32)
        print('softmaxMax: ', softmax_max)

        softmax_sum = np.fromfile('softmaxSum.bin', dtype=np.float32)
        print('softmaxSum: ', softmax_sum)

        attention_out = np.fromfile('attentionOut.bin', dtype=np.float16)
        print('attentionOut: ', attention_out)

    else:
        dq = np.fromfile('dq.bin', dtype=np.float16)
        print('dq: ', dq)

        dk = np.fromfile('dk.bin', dtype=np.float16)
        print('dk: ', dk)

        dv = np.fromfile('dv.bin', dtype=np.float16)
        print('dv: ', dv)
    print_end = "=========================================== Show sample " + case_name + \
                " result end ==================="
    print(print_end)


if __name__ == '__main__':
    # 创建进程对象
    p = multiprocessing.Process(target=show)
    # 启动进程
    p.start()
    # 等待进程结束
    p.join()
