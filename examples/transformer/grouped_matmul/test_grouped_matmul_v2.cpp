/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_matmul_v2.cpp
 * \brief
 */

#include "grouped_matmul_utils.h"
#include "aclnnop/aclnn_grouped_matmul_v2.h"

namespace grouped_matmul_example {
std::vector<std::vector<int64_t>> yShape;

int PrepareFloat16TensorList(const std::string &testCase, const std::string &currentPath, GroupedMatmulParams &params,
                             GroupedMatmulDevAddr &addrs)
{
    // The shape value must be the same with value in file ./grouped_matmul_generate_data.py.
    std::vector<std::vector<int64_t>> xShape = {{1, 16}, {4, 32}};
    std::vector<std::vector<int64_t>> weightShape = {{16, 24}, {32, 16}};
    std::vector<std::vector<int64_t>> biasShape = {{24}, {16}};
    yShape = {{1, 24}, {4, 16}};
    std::vector<std::vector<int16_t>> yData;
    for (auto &i : yShape) {
        yData.push_back(std::vector<int16_t>(GetShapeSize(i), 0));
    }

    std::string xPath = currentPath + testCase + "_x";
    auto ret = CreateAclTensorList(xPath, xShape, addrs.x, aclDataType::ACL_FLOAT16, &params.x);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor x failed\n"); return ret);
    std::string weightPath = currentPath + testCase + "_weight";
    ret = CreateAclTensorList(weightPath, weightShape, addrs.weight, aclDataType::ACL_FLOAT16, &params.weight);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor weight failed\n"); return ret);
    std::string biasPath = currentPath + testCase + "_bias";
    ret = CreateAclTensorList(biasPath, biasShape, addrs.bias, aclDataType::ACL_FLOAT16, &params.bias);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor bias failed\n"); return ret);
    ret = CreateAclTensorList(yData, yShape, addrs.y, aclDataType::ACL_FLOAT16, &params.y);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor y failed\n"); return ret);

    return ACL_SUCCESS;
}
} // namespace grouped_matmul_example
using namespace grouped_matmul_example;

int main(int argc, char **argv)
{
    if (argv == nullptr || argc < 2) { // 2: Input num, exeFile and testCase.
        LOG_PRINT("Number of input parameter error, except >= 2 but got %d inputs.\n", argc);
        return 0;
    }
    std::string exeFile(argv[0]);
    std::string currentPath = std::string(exeFile.substr(0, exeFile.rfind('/')) + "/");
    std::string testCase(argv[1]);
    // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Use CHECK as required.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output based on the API.
    int64_t splitItem = 0;
    int64_t groupType = -1;
    GroupedMatmulParams params;
    GroupedMatmulDevAddr addrs;
    ret = PrepareFloat16TensorList(testCase, currentPath, params, addrs);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare failed. ERROR: %d\n", ret);
              FreeResource(params, addrs, deviceId, &stream); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnGroupedMatmulV2.
    ret = aclnnGroupedMatmulV2GetWorkspaceSize(params.x, params.weight, params.bias, params.scale, params.offset, 
                                               params.antiquantScale, params.antiquantOffset, params.groupList, 
                                               splitItem, groupType, params.y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); 
              FreeResource(params, addrs, deviceId, &stream); return ret);
              
    // Malloc device memory for workspace based on the workspaceSize calculated from the first interface
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&addrs.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  FreeResource(params, addrs, deviceId, &stream); return ret);
    }

    // Call the second-phase API of aclnnGroupedMatmulV2.
    ret = aclnnGroupedMatmulV2(addrs.workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmul failed. ERROR: %d\n", ret);
              FreeResource(params, addrs, deviceId, &stream); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(params, addrs, deviceId, &stream); return ret);

    // 5. Copy output from device to host and then save to file
    for (int i = 0; i < yShape.size(); ++i) {
        std::string outFile = testCase + "_y_" + std::to_string(i) + ".bin";
        SaveOutResult<float>(outFile, yShape[i], &addrs.y[i], aclDataType::ACL_FLOAT16);    
    }

    // 6. Release aclTensor and device resource.
    FreeResource(params, addrs, deviceId, &stream);
    return 0;
}