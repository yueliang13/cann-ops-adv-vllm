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
 * \file test_ffn_v2.cpp
 * \brief
 */

#include <string>
#include <vector>
#include <sys/stat.h>
#include "ffn_utils.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn.h"
#include "aclnnop/aclnn_ffn_v2.h"


namespace ffn_example {
std::vector<int64_t> yShape;

int PrepareAntiQuantTensor(const std::string &testCase, const std::string &currentPath, FFNParams &params,
                           FFNDevAddr &addrs)
{
    // The shape value must be the same with value in file ../scripts/ffn_generate_data.py.
    int64_t bs = 64;
    int64_t h = 1024;
    int64_t n = 2048;
    int64_t expertNum = 4;

    std::vector<int64_t> xShape = {bs, h};
    std::string xFilePath = currentPath + testCase + "_x.bin";
    auto ret = CreateAclTensor(xFilePath, xShape, &addrs.x, aclDataType::ACL_FLOAT16, &params.x);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor x failed\n"); return ret);

    std::vector<int64_t> weight1Shape = {expertNum, h, n};
    std::string wegiht1FilePath = currentPath + testCase + "_weight1.bin";
    ret = CreateAclTensor(wegiht1FilePath, weight1Shape, &addrs.weight1, aclDataType::ACL_INT8, &params.weight1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor weight1 failed\n"); return ret);

    std::vector<int64_t> weight2Shape = {expertNum, n, h};
    std::string weight2FilePath = currentPath + testCase + "_weight2.bin";
    ret = CreateAclTensor(weight2FilePath, weight2Shape, &addrs.weight2, aclDataType::ACL_INT8, &params.weight2);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor weight2 failed\n"); return ret);

    std::vector<int64_t> expertToken{16, 32, 48, 64};
    params.expertTokens = aclCreateIntArray(expertToken.data(), expertToken.size());

    std::vector<int64_t> bias1Shape = {expertNum, n};
    std::string bias1FilePath = currentPath + testCase + "_bias1.bin";
    ret = CreateAclTensor(bias1FilePath, bias1Shape, &addrs.bias1, aclDataType::ACL_FLOAT16, &params.bias1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor bias1 failed\n"); return ret);

    std::vector<int64_t> antiquant1Shape = {expertNum, n};
    std::string antiquantScale1FilePath = currentPath + testCase + "_antiquant_scale1.bin";
    ret = CreateAclTensor(antiquantScale1FilePath, antiquant1Shape, &addrs.antiquantScale1, aclDataType::ACL_FLOAT16,
                          &params.antiquantScale1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor antiquantScale1 failed\n"); return ret);

    std::vector<int64_t> antiquant2Shape = {expertNum, h};
    std::string antiquantScale2FilePath = currentPath + testCase + "_antiquant_scale2.bin";
    ret = CreateAclTensor(antiquantScale2FilePath, antiquant2Shape, &addrs.antiquantScale2, aclDataType::ACL_FLOAT16,
                          &params.antiquantScale2);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor antiquantScale2 failed\n"); return ret);

    std::string antiquantOffset1FilePath = currentPath + testCase + "_antiquant_offset1.bin";
    ret = CreateAclTensor(antiquantOffset1FilePath, antiquant1Shape, &addrs.antiquantOffset1, aclDataType::ACL_FLOAT16,
                          &params.antiquantOffset1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor antiquantOffset1 failed\n"); return ret);

    std::string antiquantOffset2FilePath = currentPath + testCase + "_antiquant_offset2.bin";
    ret = CreateAclTensor(antiquantOffset2FilePath, antiquant2Shape, &addrs.antiquantOffset2, aclDataType::ACL_FLOAT16,
                          &params.antiquantOffset2);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor antiquantOffset2 failed\n"); return ret);

    std::vector<int64_t> yShape = {bs, h};
    std::vector<int16_t> yData(bs * h, 0);
    ret = CreateAclTensor(yData, yShape, &addrs.y, aclDataType::ACL_FLOAT16, &params.y);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor y failed\n"); return ret);

    return ACL_SUCCESS;
}

} // namespace ffn_example
using namespace ffn_example;

int main(int argc, char **argv)
{
    if (argv == nullptr || argc < 2) { // 2: Input num, exeFile and testCase.
        LOG_PRINT("Number of input parameter error, except >= 2 but got %d inputs.\n", argc);
        return 0;
    }
    std::string exeFile(argv[0]);
    std::string currentPath = std::string(exeFile.substr(0, exeFile.rfind('/')) + "/");
    std::string testCase(argv[1]);
    // 1. Init device/context/stream，refer to AscendCL interfaces docs for details.
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Create input and output for ffn aclnn api.
    const char *activation = "relu";
    int64_t innerPrecise = 1;
    bool tokensIndexFlag = true;
    FFNParams params;
    FFNDevAddr addrs;
    ret = PrepareAntiQuantTensor(testCase, currentPath, params, addrs);
    CHECK_RET(ret == ACL_SUCCESS, FreeResource(params, addrs, deviceId, &context, &stream); return ret);

    uint64_t workspaceSize = 0;

    // 3. Call CANN operator library API.
    aclOpExecutor *executor;
    // Call the first interface.
    ret = aclnnFFNV2GetWorkspaceSize(params.x, params.weight1, params.weight2, params.expertTokens, params.bias1,
                                     params.bias2, params.scale, params.offset, params.deqScale1, params.deqScale2,
                                     params.antiquantScale1, params.antiquantScale2, params.antiquantOffset1,
                                     params.antiquantOffset2, activation, innerPrecise, tokensIndexFlag, params.y,
                                     &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(params, addrs, deviceId, &context, &stream); return ret);

    // Malloc device memory for workspace based on the workspaceSize calculated from the first interface
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&addrs.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  FreeResource(params, addrs, deviceId, &context, &stream); return ret);
    }

    // Call the second interface.
    ret = aclnnFFNV2(addrs.workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNV2 failed. ERROR: %d\n", ret);
              FreeResource(params, addrs, deviceId, &context, &stream); return ret);

    // 4. Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(params, addrs, deviceId, &context, &stream); return ret);

    // 5. Copy output from device to host and then save to file
    std::string outFile = testCase + "_y.bin";
    SaveOutResult<float>(outFile, yShape, &addrs.y);

    // 6. Release aclTensor and device resource.
    FreeResource(params, addrs, deviceId, &context, &stream);

    return 0;
}
