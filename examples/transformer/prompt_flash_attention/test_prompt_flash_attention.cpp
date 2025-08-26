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
 * \file test_prompt_flash_attention.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_prompt_flash_attention.h"
#include "securec.h"
 
using namespace std;
 
#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)
 
#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)
 
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}
 
int Init(int32_t deviceId, aclrtStream* stream) {
  // Fixed format, AscendCL initialization
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}
 
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to request device side memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host side data to device side memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
 
  // Calculate the strides of continuous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
 
  // Call the aclCreateTensor interface to create aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}
 
int main() {
  // 1. (Fixed format) Device/stream initialization, refer to AscendCL external interface list
  // Fill in the deviceId based on your actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
 
  // 2. To construct input and output, it is necessary to customize the construction according to the API interface
  std::vector<int64_t> queryShape = {1, 2, 1, 16}; // BNSD
  std::vector<int64_t> keyShape = {1, 2, 2, 16}; // BNSD
  std::vector<int64_t> valueShape = {1, 2, 2, 16}; // BNSD
  std::vector<int64_t> attenShape = {1, 1, 1, 2}; // B 1 S1 S2
  std::vector<int64_t> outShape = {1, 2, 1, 16}; // BNSD
  void *queryDeviceAddr = nullptr;
  void *keyDeviceAddr = nullptr;
  void *valueDeviceAddr = nullptr;
  void *attenDeviceAddr = nullptr;
  void *outDeviceAddr = nullptr;
  aclTensor *queryTensor = nullptr;
  aclTensor *keyTensor = nullptr;
  aclTensor *valueTensor = nullptr;
  aclTensor *attenTensor = nullptr;
  aclTensor *outTensor = nullptr;
  int64_t queryShapeSize = GetShapeSize(queryShape); // BNSD
  int64_t keyShapeSize = GetShapeSize(keyShape); // BNSD
  int64_t valueShapeSize = GetShapeSize(valueShape); // BNSD
  int64_t attenShapeSize = GetShapeSize(attenShape); // B 1 S1 S2
  int64_t outShapeSize = GetShapeSize(outShape); // BNSD
  std::vector<float> queryHostData(queryShapeSize, 1);
  std::vector<float> keyHostData(keyShapeSize, 1);
  std::vector<float> valueHostData(valueShapeSize, 1);
  std::vector<float> attenHostData(attenShapeSize, 1);
  std::vector<float> outHostData(outShapeSize, 1);
 
  // Create query aclTensor
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create key aclTensor
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create value aclTensor
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create atten aclTensor
  ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  std::vector<int64_t> actualSeqlenVector = {2};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());
  int64_t numHeads=2; // N
  int64_t numKeyValueHeads = numHeads;
  double scaleValue= 1 / sqrt(2); // 1/sqrt(d)
  int64_t preTokens = 65535;
  int64_t nextTokens = 65535;
  string sLayerOut = "BNSD";
  char layerOut[sLayerOut.length()];
  ret = strcpy_s(layerOut, strlen(sLayerOut.c_str()) + 1, sLayerOut.c_str());
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first interface
  ret = aclnnPromptFlashAttentionGetWorkspaceSize(queryTensor, keyTensor, valueTensor, nullptr, nullptr, nullptr, numHeads, scaleValue, 
    preTokens, nextTokens, layerOut, numKeyValueHeads, outTensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Apply for device memory based on the workspaceSize calculated from the first interface
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second interface
  ret = aclnnPromptFlashAttention(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttention failed. ERROR: %d\n", ret); return ret);
 
  // 4. (Fixed format) Synchronize and wait for the completion of task execution
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
 
  // 5. Obtain the output value, copy the result from the device side memory to the host side, and modify it according to the specific API interface definition
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
 
  // 6. Release resources
  aclDestroyTensor(queryTensor);
  aclDestroyTensor(keyTensor);
  aclDestroyTensor(valueTensor);
  aclDestroyTensor(attenTensor);
  aclDestroyTensor(outTensor);
  aclDestroyIntArray(actualSeqLengths);
  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(attenDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
