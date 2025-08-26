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
 * \file ffn_utils.h
 * \brief
 */

#ifndef EXAMPLE_FNN_UTILS_H
#define EXAMPLE_FNN_UTILS_H

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

namespace ffn_example {
#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T> void SaveOutResult(std::string &fileName, std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<T> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    std::ofstream file(fileName, std::ios::binary);
    // Save data to file
    file.write(static_cast<char *>((void *)resultData.data()), size * sizeof(T));
    file.close();
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // Init AscendCL
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret);
              aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtDestroyContext(context);
              aclrtResetDevice(deviceId); aclFinalize(); return ret);
    return 0;
}

int ReadBinFileNNop(std::string &filePath, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    CHECK_RET(fileStatus == ACL_SUCCESS, LOG_PRINT("Failed to get file %s\n", filePath.c_str()); return -1);

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    CHECK_RET(file.is_open(), LOG_PRINT("Open file failed.\n"); return -1);

    file.seekg(0, file.end);
    uint64_t binFileBufferLen = file.tellg();
    CHECK_RET(binFileBufferLen == bufferSize, LOG_PRINT("Check file size failed.\n"); file.close(); return -1);

    file.seekg(0, file.beg);
    file.read(static_cast<char *>(buffer), binFileBufferLen);
    file.close();
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Malloc device memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Copy host data to device memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Create strides for contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Create aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int CreateAclTensor(std::string &filePath, const std::vector<int64_t> &shape, void **deviceAddr, aclDataType dataType,
                    aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // Malloc device memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Malloc host memory
    void *binBufferHost = nullptr;
    ret = aclrtMallocHost(&binBufferHost, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);

    // Read input data file
    ret = ReadBinFileNNop(filePath, binBufferHost, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ReadBinFileNNop failed. ERROR: %d\n", ret);
              (void)aclrtFreeHost(binBufferHost); return ret);

    // Copy host data to device memory
    ret = aclrtMemcpy(*deviceAddr, size, binBufferHost, size, ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtFreeHost(binBufferHost);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Create strides for contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Create aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct FFNParams {
    aclTensor *x = nullptr;
    aclTensor *weight1 = nullptr;
    aclTensor *weight2 = nullptr;
    aclIntArray *expertTokens = nullptr;
    aclTensor *expertTokensTensor = nullptr;
    aclTensor *bias1 = nullptr;
    aclTensor *bias2 = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *offset = nullptr;
    aclTensor *deqScale1 = nullptr;
    aclTensor *deqScale2 = nullptr;
    aclTensor *antiquantScale1 = nullptr;
    aclTensor *antiquantScale2 = nullptr;
    aclTensor *antiquantOffset1 = nullptr;
    aclTensor *antiquantOffset2 = nullptr;
    aclTensor *y = nullptr;
};

struct FFNDevAddr {
    void *x = nullptr;
    void *weight1 = nullptr;
    void *weight2 = nullptr;
    void *expertTokens = nullptr;
    void *bias1 = nullptr;
    void *bias2 = nullptr;
    void *scale = nullptr;
    void *offset = nullptr;
    void *deqScale1 = nullptr;
    void *deqScale2 = nullptr;
    void *antiquantScale1 = nullptr;
    void *antiquantScale2 = nullptr;
    void *antiquantOffset1 = nullptr;
    void *antiquantOffset2 = nullptr;
    void *y = nullptr;
    void *workspaceAddr = nullptr;
};

void FreeResource(FFNParams &tensors, FFNDevAddr &addrs, int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // Release aclTensor
    if (tensors.x != nullptr) {
        aclDestroyTensor(tensors.x);
    }
    if (tensors.weight1 != nullptr) {
        aclDestroyTensor(tensors.weight1);
    }
    if (tensors.weight2 != nullptr) {
        aclDestroyTensor(tensors.weight2);
    }
    if (tensors.bias1 != nullptr) {
        aclDestroyTensor(tensors.bias1);
    }
    if (tensors.bias2 != nullptr) {
        aclDestroyTensor(tensors.bias2);
    }
    if (tensors.scale != nullptr) {
        aclDestroyTensor(tensors.scale);
    }
    if (tensors.offset != nullptr) {
        aclDestroyTensor(tensors.offset);
    }
    if (tensors.deqScale1 != nullptr) {
        aclDestroyTensor(tensors.deqScale1);
    }
    if (tensors.deqScale2 != nullptr) {
        aclDestroyTensor(tensors.deqScale2);
    }
    if (tensors.antiquantScale1 != nullptr) {
        aclDestroyTensor(tensors.antiquantScale1);
    }
    if (tensors.antiquantScale2 != nullptr) {
        aclDestroyTensor(tensors.antiquantScale2);
    }
    if (tensors.antiquantOffset1 != nullptr) {
        aclDestroyTensor(tensors.antiquantOffset1);
    }
    if (tensors.antiquantOffset2 != nullptr) {
        aclDestroyTensor(tensors.antiquantOffset2);
    }
    if (tensors.y != nullptr) {
        aclDestroyTensor(tensors.y);
    }
    // Release device resource
    if (addrs.x != nullptr) {
        aclrtFree(addrs.x);
    }
    if (addrs.weight1 != nullptr) {
        aclrtFree(addrs.weight1);
    }
    if (addrs.weight2 != nullptr) {
        aclrtFree(addrs.weight2);
    }
    if (addrs.bias1 != nullptr) {
        aclrtFree(addrs.bias1);
    }
    if (addrs.bias2 != nullptr) {
        aclrtFree(addrs.bias2);
    }
    if (addrs.scale != nullptr) {
        aclrtFree(addrs.scale);
    }
    if (addrs.offset != nullptr) {
        aclrtFree(addrs.offset);
    }
    if (addrs.deqScale1 != nullptr) {
        aclrtFree(addrs.deqScale1);
    }
    if (addrs.deqScale2 != nullptr) {
        aclrtFree(addrs.deqScale2);
    }
    if (addrs.antiquantScale1 != nullptr) {
        aclrtFree(addrs.antiquantScale1);
    }
    if (addrs.antiquantScale2 != nullptr) {
        aclrtFree(addrs.antiquantScale2);
    }
    if (addrs.antiquantOffset1 != nullptr) {
        aclrtFree(addrs.antiquantOffset1);
    }
    if (addrs.antiquantOffset2 != nullptr) {
        aclrtFree(addrs.antiquantOffset2);
    }
    if (addrs.y != nullptr) {
        aclrtFree(addrs.y);
    }
    if (addrs.workspaceAddr != nullptr) {
        aclrtFree(addrs.workspaceAddr);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (context != nullptr) {
        aclrtDestroyContext(context);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}
} // namespace ffn_example

#endif // EXAMPLE_FNN_UTILS_H