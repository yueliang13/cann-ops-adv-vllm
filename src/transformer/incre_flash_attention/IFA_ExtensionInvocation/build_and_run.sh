#!/bin/bash
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
rm -rf build dist *.egg-info

BASE_DIR=$(pwd)

# 在 build_and_run.sh 中添加

export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 编译wheel包
python3 setup.py build bdist_wheel

# # 安装wheel包
cd ${BASE_DIR}/dist
pip3 install *.whl --force-reinstall

# 设置设备相关环境变量
# export ASCEND_DEVICE_ID=3
# export ASCEND_RT_VISIBLE_DEVICES=3

# # 运行测试用例
#cd ${BASE_DIR}/test
#msprof op python3 ifa_v5_case_tkp.py
# python3 run_ifa.py 
# python test_add_custom_v4.py 
# msprof op python3 run_ifa.py 
# if [ $? -ne 0 ]; then
#     echo "ERROR: run add_custom test failed!"
# fi
# echo "INFO: run add_custom test success!"

# 性能测试
# export ASCEND_RT_VISIBLE_DEVICES=0
# msprof op python ifa_v5_case_tkp.py
# msprof op simulator python test/run_ifa.py
# msprof op simulator --soc-version=Ascend910B2 --output=../output_data python test/run_ifa.py
