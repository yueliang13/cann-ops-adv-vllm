#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set -e

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

case_name=$1
test_program=$2
test_case=$3

echo "=========================================== Run $case_name ===================================="
python3 ${CURRENT_DIR}/grouped_matmul_generate_data.py $test_case
echo "=========================================== Execute $case_name sample start ===================="
# Execute test program
${test_program} $test_case
if [ $? -ne 0 ];then
    echo "Error: Execute ${test_program} failed."
    exit 1
fi
echo "=========================================== Execute $case_name sample end ======================"
python3 ${CURRENT_DIR}/grouped_matmul_print_result.py $test_case
rm -rf *.bin
echo "=========================================== Run $case_name success =============================="
exit 0