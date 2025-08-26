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
 * \file grouped_matmul_utils.cpp
 * \brief
 */

#include "grouped_matmul_utils.h"

int64_t grouped_matmul_example::GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int grouped_matmul_example::Init(int32_t deviceId, aclrtStream *stream)
{
    // (Fixed writing) Initialize AscendCL.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

int grouped_matmul_example::ReadBinFileNNop(const std::string &filePath, void *buffer, size_t bufferSize)
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

int grouped_matmul_example::CreateAclTensor(const std::string& filePath, const std::vector<int64_t> &shape, 
                                            void **deviceAddr, aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // Call aclrtMalloc to allocate memory on the device.
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

    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, binBufferHost, size, ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtFreeHost(binBufferHost);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int grouped_matmul_example::CreateAclTensorList(const std::string& filePath, 
                                                const std::vector<std::vector<int64_t>> &shapes, 
                                                void **deviceAddr, aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++) {
        std::string tensorPath = filePath + "_" + std::to_string(i) + ".bin";
        int ret = CreateAclTensor(tensorPath, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

void grouped_matmul_example::FreeParam(GroupedMatmulParams &params)
{
    if (params.x != nullptr) {
        aclDestroyTensorList(params.x);
    }
    if (params.weight != nullptr) {
        aclDestroyTensorList(params.weight);
    }
    if (params.bias != nullptr) {
        aclDestroyTensorList(params.bias);
    }
    if (params.scale != nullptr) {
        aclDestroyTensorList(params.scale);
    }
    if (params.offset != nullptr) {
        aclDestroyTensorList(params.offset);
    }
    if (params.antiquantScale != nullptr) {
        aclDestroyTensorList(params.antiquantScale);
    }
    if (params.antiquantOffset != nullptr) {
        aclDestroyTensorList(params.antiquantOffset);
    }
    if (params.perTokenScale != nullptr) {
        aclDestroyTensorList(params.perTokenScale);
    }
    if (params.groupList != nullptr) {
        aclDestroyIntArray(params.groupList);
    }
    if (params.groupListTensor != nullptr) {
        aclDestroyTensor(params.groupListTensor);
    }
    if (params.y != nullptr) {
        aclDestroyTensorList(params.y);
    }
}

void grouped_matmul_example::FreeAddr(GroupedMatmulDevAddr &addrs)
{
    int size = TENSOR_SIZE;
    for (int i = 0; i < size; i++) {
        if (addrs.x[i] != nullptr) {
            aclrtFree(addrs.x[i]);
        }
        if (addrs.weight[i] != nullptr) {
            aclrtFree(addrs.weight[i]);
        }
        if (addrs.bias[i] != nullptr) {
            aclrtFree(addrs.bias[i]);
        }
        if (addrs.scale[i] != nullptr) {
            aclrtFree(addrs.scale[i]);
        }
        if (addrs.offset[i] != nullptr) {
            aclrtFree(addrs.offset[i]);
        }
        if (addrs.antiquantScale[i] != nullptr) {
            aclrtFree(addrs.antiquantScale[i]);
        }
        if (addrs.antiquantOffset[i] != nullptr) {
            aclrtFree(addrs.antiquantOffset[i]);
        }
        if (addrs.perTokenScale[i] != nullptr) {
            aclrtFree(addrs.perTokenScale[i]);
        }
        if (addrs.groupList[i] != nullptr) {
            aclrtFree(addrs.groupList[i]);
        }
        if (addrs.groupListTensor[i] != nullptr) {
            aclrtFree(addrs.groupListTensor[i]);
        }
        if (addrs.y[i] != nullptr) {
            aclrtFree(addrs.y[i]);
        }
        if (addrs.workspaceAddr != nullptr) {
            aclrtFree(addrs.workspaceAddr);
        }
    }
}

void grouped_matmul_example::FreeResource(GroupedMatmulParams &params, GroupedMatmulDevAddr &addrs, 
                                          int32_t deviceId, aclrtStream *stream)
{
    FreeParam(params);
    FreeAddr(addrs);
    if (stream != nullptr) {
        aclrtDestroyStream(*stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize(); 
}