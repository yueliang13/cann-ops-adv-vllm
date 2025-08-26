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
 * \file test_fused_infer_attention_score_v2_ifa_antiquant.cpp
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
 
static int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}
 
static int Init(int32_t deviceId, aclrtStream* stream) {
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
  std::vector<int64_t> queryShape = {1, 1, 6, 128}; // BNSD
  std::vector<int64_t> keyShape = {1, 1, 16, 128}; // BNSD
  std::vector<int64_t> valueShape = {1, 1, 16, 128}; // BNSD
  std::vector<int64_t> attenShape = {1, 1, 6, 16}; // B 1 S1 S2
  std::vector<int64_t> outShape = {1, 1, 6, 128}; // BNSD
  //antiquant params
  std::vector<int64_t> keyAntiquantScaleShape = {1, 16};// B S: pertoken
  std::vector<int64_t> keyAntiquantOffsetShape = {1, 16};// B S: pertoken
  std::vector<int64_t> valueAntiquantScaleShape = {1, 16}; // B S: pertoken
  std::vector<int64_t> valueAntiquantOffsetShape = {1, 16}; // B S: pertoken

  void *queryDeviceAddr = nullptr;
  void *keyDeviceAddr = nullptr;
  void *valueDeviceAddr = nullptr;
  void *attenDeviceAddr = nullptr;
  void *outDeviceAddr = nullptr;
  //antiquant params
  void *keyAntiquantScaleDeviceAddr = nullptr;
  void *keyAntiquantOffsetDeviceAddr = nullptr;
  void *valueAntiquantScaleDeviceAddr = nullptr;
  void *valueAntiquantOffsetDeviceAddr = nullptr;

  aclTensor *queryTensor = nullptr;
  aclTensor *keyTensor = nullptr;
  aclTensor *valueTensor = nullptr;
  aclTensor *attenTensor = nullptr;
  aclTensor *outTensor = nullptr;
  //antiquant params
  aclTensor *keyAntiquantScaleTensor = nullptr;
  aclTensor *keyAntiquantOffsetTensor = nullptr;
  aclTensor *valueAntiquantScaleTensor = nullptr;
  aclTensor *valueAntiquantOffsetTensor = nullptr;

  int64_t queryShapeSize = GetShapeSize(queryShape); // BNSD
  int64_t keyShapeSize = GetShapeSize(keyShape); // BNSD
  int64_t valueShapeSize = GetShapeSize(valueShape); // BNSD
  int64_t attenShapeSize = GetShapeSize(attenShape); // B 1 S1 S2
  int64_t outShapeSize = GetShapeSize(outShape); // BNSD
  //antiquant params
  int64_t keyAntiquantScaleShapeSize = GetShapeSize(keyAntiquantScaleShape); 
  int64_t keyAntiquantOffsetShapeSize = GetShapeSize(keyAntiquantOffsetShape); 
  int64_t valueAntiquantScaleShapeSize = GetShapeSize(valueAntiquantScaleShape); 
  int64_t valueAntiquantOffsetShapeSize = GetShapeSize(valueAntiquantOffsetShape); 

  std::vector<uint16_t> queryHostData(queryShapeSize, 0x3C00);
  std::vector<uint8_t> keyHostData(keyShapeSize, 1);
  std::vector<uint8_t> valueHostData(valueShapeSize, 1);
  std::vector<float> attenHostData(attenShapeSize, 1);
  std::vector<float> outHostData(outShapeSize, 1);
  //antiquant params
  std::vector<uint16_t> keyAntiquantScaleHostData(keyAntiquantScaleShapeSize, 0x3C00);
  std::vector<uint16_t> keyAntiquantOffsetHostData(keyAntiquantOffsetShapeSize, 0x3C00);
  std::vector<uint16_t> valueAntiquantScaleHostData(valueAntiquantScaleShapeSize, 0x3C00);
  std::vector<uint16_t> valueAntiquantOffsetHostData(valueAntiquantOffsetShapeSize, 0x3C00);

  // Create query aclTensor.
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_BF16, &queryTensor);
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
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  //antiquant params
  ret = CreateAclTensor(keyAntiquantScaleHostData, keyAntiquantScaleShape, &keyAntiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &keyAntiquantScaleTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(keyAntiquantOffsetHostData, keyAntiquantOffsetShape, &keyAntiquantOffsetDeviceAddr, aclDataType::ACL_FLOAT, &keyAntiquantOffsetTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(valueAntiquantScaleHostData, valueAntiquantScaleShape, &valueAntiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &valueAntiquantScaleTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(valueAntiquantOffsetHostData, valueAntiquantOffsetShape, &valueAntiquantOffsetDeviceAddr, aclDataType::ACL_FLOAT, &valueAntiquantOffsetTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> actualSeqlenVector = {2};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());
  int64_t numHeads=1; // N
  int64_t numKeyValueHeads = numHeads;
  double scaleValue= 1 / sqrt(128); // 1/sqrt(d)
  int64_t preTokens = 65535;
  int64_t nextTokens = 65535;
  std::string sLayerOut = "BNSD";
  int64_t sparseMode = 0;
  int64_t innerPrecise = 1;
  int blockSize = 0;
  int antiquantMode = 0;
  bool softmaxLseFlag = false;
  int keyAntiquantMode = 1;
  int valueAntiquantMode = 1;
  // 3. Call CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first interface.
  ret = aclnnFusedInferAttentionScoreV2GetWorkspaceSize(queryTensor, tensorKeyList, tensorValueList,  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, keyAntiquantScaleTensor, keyAntiquantOffsetTensor, valueAntiquantScaleTensor, valueAntiquantOffsetTensor,
  nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, &sLayerOut[0], numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, outTensor, nullptr, &workspaceSize, &executor);
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
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  // 6. Release resources.
  aclDestroyTensor(queryTensor);
  aclDestroyTensor(keyTensor);
  aclDestroyTensor(valueTensor);
  aclDestroyTensor(attenTensor);
  aclDestroyTensor(outTensor);
  aclDestroyTensor(keyAntiquantScaleTensor);
  aclDestroyTensor(keyAntiquantOffsetTensor);
  aclDestroyTensor(valueAntiquantScaleTensor);
  aclDestroyTensor(valueAntiquantOffsetTensor);
  aclDestroyIntArray(actualSeqLengths);
  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(attenDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(keyAntiquantScaleDeviceAddr);
  aclrtFree(keyAntiquantOffsetDeviceAddr);
  aclrtFree(valueAntiquantScaleDeviceAddr);
  aclrtFree(valueAntiquantOffsetDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}