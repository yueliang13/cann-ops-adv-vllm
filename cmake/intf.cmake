# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (BUILD_OPEN_PROJECT)
    include(${OPS_ADV_CMAKE_DIR}/intf_pub.cmake)

    if (TESTS_UT_OPS_TEST)
        include(${OPS_ADV_CMAKE_DIR}/intf_pub_utest.cmake)
    endif ()

    if (TESTS_EXAMPLE_OPS_TEST)
        include(${OPS_ADV_CMAKE_DIR}/intf_pub_examples.cmake)
    endif ()
endif ()
