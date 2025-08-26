/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_fused_infer_attention_score_v2.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v2.h"
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
  // Fixed writing method, AscendCL initialization.
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
  // Call aclrtMalloc to request device side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host side data to device side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
 
  // Calculate the strides of continuous tensors.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
 
  // Call the aclCreateTensor interface to create aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}
 
int main() {
  // 1. (Fixed writing method)  device/stream initialization. Refer to AscendCL's list of external interfaces.
  // Fill in the deviceId based on your actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
 
  // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
  std::vector<int64_t> queryShape = {2, 4, 1023, 128}; // BNSD
  std::vector<int64_t> keyShape = {2, 1, 2047, 128}; // BNSD
  std::vector<int64_t> valueShape = {2, 1, 2047, 128}; // BNSD
  std::vector<int64_t> attenShape = {2048, 2048}; // S1 S2
  std::vector<int64_t> outShape = {2, 1023, 4, 128}; // BSND
  std::vector<int64_t> deqScale1Shape = {1};
  std::vector<int64_t> quantScale1Shape = {1};
  std::vector<int64_t> deqScale2Shape = {1};
  std::vector<int64_t> quantScale2Shape = {1};
  std::vector<int64_t> quantOffset2Shape = {1};
  void *queryDeviceAddr = nullptr;
  void *keyDeviceAddr = nullptr;
  void *valueDeviceAddr = nullptr;
  void *attenDeviceAddr = nullptr;
  void *outDeviceAddr = nullptr;
  void *deqScale1DeviceAddr = nullptr;
  void *quantScale1DeviceAddr = nullptr;
  void *deqScale2DeviceAddr = nullptr;
  void *quantScale2DeviceAddr = nullptr;
  void *quantOffset2DeviceAddr = nullptr;
  aclTensor *queryTensor = nullptr;
  aclTensor *keyTensor = nullptr;
  aclTensor *valueTensor = nullptr;
  aclTensor *attenTensor = nullptr;
  aclTensor *outTensor = nullptr;
  aclTensor *deqScale1Tensor = nullptr;
  aclTensor *quantScale1Tensor = nullptr;
  aclTensor *deqScale2Tensor = nullptr;
  aclTensor *quantScale2Tensor = nullptr;
  aclTensor *quantOffset2Tensor = nullptr;
  int64_t queryShapeSize = GetShapeSize(queryShape); // BNSD
  int64_t keyShapeSize = GetShapeSize(keyShape); // BNSD
  int64_t valueShapeSize = GetShapeSize(valueShape); // BNSD
  int64_t attenShapeSize = GetShapeSize(attenShape); // S1 S2
  int64_t outShapeSize = GetShapeSize(outShape); // BNSD
  int64_t deqScale1ShapeSize = GetShapeSize(deqScale1Shape);
  int64_t quantScale1ShapeSize = GetShapeSize(quantScale1Shape);
  int64_t deqScale2ShapeSize = GetShapeSize(deqScale2Shape);
  int64_t quantScale2ShapeSize = GetShapeSize(quantScale2Shape);
  int64_t quantOffset2ShapeSize = GetShapeSize(quantOffset2Shape);
  std::vector<float> queryHostData(queryShapeSize, 1);
  std::vector<float> keyHostData(keyShapeSize, 1);
  std::vector<float> valueHostData(valueShapeSize, 1);
  std::vector<float> attenHostData(attenShapeSize, 1);
  std::vector<float> outHostData(outShapeSize, 1);
  std::vector<float> deqScale1HostData(deqScale1ShapeSize, 0.015625); // 0.015625 : The number is an example and can be replaced as required.
  std::vector<float> quantScale1HostData(quantScale1ShapeSize, 5); // 5 : The number is an example and can be replaced as required.
  std::vector<float> deqScale2HostData(deqScale2ShapeSize, 0.015625); // 0.015625 : The number is an example and can be replaced as required.
  std::vector<float> quantScale2HostData(quantScale2ShapeSize, 1);
  std::vector<float> quantOffset2HostData(quantOffset2ShapeSize, 1);
 
  // Create query aclTensor.
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_INT8, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create key aclTensor.
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_INT8, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int kvTensorNum = 1;
  aclTensor *tensorsOfKey[kvTensorNum];
  tensorsOfKey[0] = keyTensor;
  auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
  // Create value aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_INT8, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor *tensorsOfValue[kvTensorNum];
  tensorsOfValue[0] = valueTensor;
  auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
  // Create atten aclTensor.
  ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create deqScale1 aclTensor.
  ret = CreateAclTensor(deqScale1HostData, deqScale1Shape, &deqScale1DeviceAddr, aclDataType::ACL_UINT64, &deqScale1Tensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create quantScale1 aclTensor.
  ret = CreateAclTensor(quantScale1HostData, quantScale1Shape, &quantScale1DeviceAddr, aclDataType::ACL_FLOAT, &quantScale1Tensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create deqScale2 aclTensor.
  ret = CreateAclTensor(deqScale2HostData, deqScale2Shape, &deqScale2DeviceAddr, aclDataType::ACL_UINT64, &deqScale2Tensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create quantScale2 aclTensor.
  ret = CreateAclTensor(quantScale2HostData, quantScale2Shape, &quantScale2DeviceAddr, aclDataType::ACL_FLOAT, &quantScale2Tensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create quantOffset2 aclTensor.
  ret = CreateAclTensor(quantOffset2HostData, quantOffset2Shape, &quantOffset2DeviceAddr, aclDataType::ACL_FLOAT, &quantOffset2Tensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  std::vector<int64_t> actualSeqlenVector_q = {1000, 1000};
  auto actualSeqLengths_q = aclCreateIntArray(actualSeqlenVector_q.data(), actualSeqlenVector_q.size());
  std::vector<int64_t> actualSeqlenVector_kv = {500, 500};
  auto actualSeqLengths_kv = aclCreateIntArray(actualSeqlenVector_kv.data(), actualSeqlenVector_kv.size());
  int64_t numHeads=4; // N
  int64_t numKeyValueHeads = 1;
  double scaleValue= 0.088388; // 1/sqrt(d)
  int64_t preTokens = 2147483647;
  int64_t nextTokens = 0;
  string sLayerOut = "BNSD_BSND";
  char layerOut[sLayerOut.length()];
  ret = strcpy_s(layerOut, strlen(sLayerOut.c_str()) + 1, sLayerOut.c_str());
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int64_t sparseMode = 2;
  int64_t innerPrecise = 0;
  int blockSize = 0;
  int antiquantMode = 0;
  bool softmaxLseFlag = false;
  int keyAntiquantMode = 0;
  int valueAntiquantMode = 0;
  // 3. Call CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first interface.
  ret = aclnnFusedInferAttentionScoreV2GetWorkspaceSize(queryTensor, tensorKeyList, tensorValueList,  nullptr, attenTensor, actualSeqLengths_q, actualSeqLengths_kv, deqScale1Tensor, quantScale1Tensor, deqScale2Tensor, quantScale2Tensor, quantOffset2Tensor, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
  nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, outTensor, nullptr, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Apply for device memory based on the workspaceSize calculated from the first interface paragraph.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second interface.
  ret = aclnnFusedInferAttentionScoreV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV2 failed. ERROR: %d\n", ret); return ret);
 
  // 4. (Fixed writing method) Synchronize and wait for task execution to end.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
 
  // 5. Retrieve the output value, copy the result from the device side memory to the host side, and modify it according to the specific API interface definition.
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
 
  // 6. Release resources.
  aclDestroyTensor(queryTensor);
  aclDestroyTensor(keyTensor);
  aclDestroyTensor(valueTensor);
  aclDestroyTensor(attenTensor);
  aclDestroyTensor(outTensor);
  aclDestroyTensor(deqScale1Tensor);
  aclDestroyTensor(quantScale1Tensor);
  aclDestroyTensor(deqScale2Tensor);
  aclDestroyTensor(quantScale2Tensor);
  aclDestroyTensor(quantOffset2Tensor);
  aclDestroyIntArray(actualSeqLengths_q);
  aclDestroyIntArray(actualSeqLengths_kv);
  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(attenDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(deqScale1DeviceAddr);
  aclrtFree(quantScale1DeviceAddr);
  aclrtFree(deqScale2DeviceAddr);
  aclrtFree(quantScale2DeviceAddr);
  aclrtFree(quantOffset2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}