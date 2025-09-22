# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.14)

# 设置工程名
project(ACLNN_EXAMPLE)

# Compile options
add_compile_options(-std=c++11)

# 添加日志相关的预定义宏
add_definitions(-DOPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING")
add_definitions(-DOPS_UTILS_LOG_PACKAGE_TYPE="OP")
add_definitions(-DDEBUG)

# 设置编译选项
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")    
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -DDEBUG")

# 强制使用DEBUG模式编译，确保日志输出
set(CMAKE_BUILD_TYPE Debug)

# 设置可执行文件名，并指定V5测试文件
add_executable(opapi_test
               test_incre_flash_attention_v5.cpp) 

# 设置ASCEND_PATH（CANN软件包目录，请根据实际路径修改）和INCLUDE_BASE_DIR（头文件目录）
if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
elseif(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()
message(STATUS "ASCEND_PATH=${ASCEND_PATH}")
set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
include_directories(
    ${INCLUDE_BASE_DIR}
    ${INCLUDE_BASE_DIR}/aclnn
    ${ASCEND_PATH}/opp/vendors/customize/op_api/include
    ${ASCEND_PATH}/toolkit/inc
    ${ASCEND_PATH}/toolkit/inc/toolchain
     /root/xzh_workspace/AscendIvfAttention/cann-ops-adv-release/src/utils/inc
     /root/xzh_workspace/AscendIvfAttention/cann-ops-adv-release/src/utils/inc/log
)

# 设置链接的库文件路径
target_link_libraries(opapi_test PRIVATE
                      ${ASCEND_PATH}/lib64/libascendcl.so
                      ${ASCEND_PATH}/lib64/libnnopbase.so
                      ${ASCEND_PATH}/lib64/libascendalog.so
                    #   ${ASCEND_PATH}/lib64/libopapi.so
                      # /root/workspace/operators/AscendIvfAttention/cann-ops-adv/build/libcust_opapi.so
                     ${ASCEND_PATH}/opp/vendors/customize/op_api/lib/libcust_opapi.so # 装包后的自定义算子路径
                    #  /root/xzh_workspace/AscendIvfAttention/cann-ops-adv/build/libcust_opapi.so
)

# 可执行文件在CMakeLists文件所在目录的bin目录下
install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})