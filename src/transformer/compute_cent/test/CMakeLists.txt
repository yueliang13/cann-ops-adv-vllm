# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# CMake最低版本要求
cmake_minimum_required(VERSION 3.14)

# 设置工程名
project(COMPUTE_CENT_TEST)

# 编译选项
add_compile_options(-std=c++11)

# 添加日志相关的预定义宏
add_definitions(-DOPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING")
add_definitions(-DOPS_UTILS_LOG_PACKAGE_TYPE="OP")
add_definitions(-DDEBUG)

# 设置编译选项
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -DDEBUG")

# 强制使用DEBUG模式编译，确保日志输出
set(CMAKE_BUILD_TYPE Debug)

# 设置可执行文件名
add_executable(compute_cent_test
               test_compute_cent.cpp)

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

# 包含目录
include_directories(
    ${INCLUDE_BASE_DIR}
    ${INCLUDE_BASE_DIR}/aclnn
    ${ASCEND_PATH}/opp/vendors/customize/op_api/include
    ${ASCEND_PATH}/toolkit/inc
    ${ASCEND_PATH}/toolkit/inc/toolchain
    ${CMAKE_CURRENT_SOURCE_DIR}/../ophost
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../utils/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../utils/inc/log
)

# 设置链接的库文件路径
target_link_libraries(compute_cent_test PRIVATE
                      ${ASCEND_PATH}/lib64/libascendcl.so
                      ${ASCEND_PATH}/lib64/libnnopbase.so
                      ${ASCEND_PATH}/lib64/libascendalog.so
                      ${ASCEND_PATH}/opp/vendors/customize/op_api/lib/libcust_opapi.so
)

# 可执行文件在CMakeLists文件所在目录的bin目录下
install(TARGETS compute_cent_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}) 