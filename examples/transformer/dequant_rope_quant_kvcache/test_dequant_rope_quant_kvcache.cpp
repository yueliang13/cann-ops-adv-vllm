/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dequant_rope_quant_kvcache.cpp
 * \brief
 */

 #include <iostream>
 #include <vector>
 #include "acl/acl.h"
 #include "aclnnop/aclnn_dequant_rope_quant_kvcache.h"
 
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
 
 void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
   auto size = GetShapeSize(shape);
   std::vector<int8_t> resultData(size, 0);
   auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                          *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
 }
 
 int Init(int32_t deviceId, aclrtStream* stream) {
   // 固定写法，AscendCL初始化
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
   // 调用aclrtMalloc申请device侧内存
   auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
   // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
   ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
 
   // 计算连续tensor的strides
   std::vector<int64_t> strides(shape.size(), 1);
   for (int64_t i = shape.size() - 2; i >= 0; i--) {
     strides[i] = shape[i + 1] * strides[i + 1];
   }
 
   // 调用aclCreateTensor接口创建aclTensor
   *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                             shape.data(), shape.size(), *deviceAddr);
   return 0;
 }
 
 int main() {
   // 1. 固定写法，device/stream初始化, 参考AscendCL对外接口列表
   // 根据自己的实际device填写deviceId
   int32_t deviceId = 0;
   aclrtStream stream;
   auto ret = Init(deviceId, &stream);
   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
 
   // 2. 构造输入与输出，需要根据API的接口定义构造
   std::vector<int64_t> inputShape = {320, 1, 1280};
   std::vector<int64_t> cosShape = {320, 1, 1, 128};
   std::vector<int64_t> sinShape = {320, 1, 1, 128};
   std::vector<int64_t> kcacheShape = {320, 1280, 1, 128};
   std::vector<int64_t> vcacheShape = {320, 1280, 1, 128};
   std::vector<int64_t> indicesShape = {320};
   std::vector<int64_t> kscaleShape = {128};
   std::vector<int64_t> vscaleShape = {128};
   std::vector<int64_t> koffsetShape = {128};
   std::vector<int64_t> voffsetShape = {128};
 
   std::vector<int64_t> weightShape = {1280};
   std::vector<int64_t> activationShape = {1280};
   std::vector<int64_t> biasShape = {8192};
 
   std::vector<int16_t> inputHostData(320*1280, 1);
   std::vector<int16_t> cosHostData(320*128, 1);
   std::vector<int16_t> sinHostData(320*128, 1);
   std::vector<int8_t> kcacheHostData(320*1280*128, 6);
   std::vector<int8_t> vcacheHostData(320*1280*128, 6);
   std::vector<int32_t> indicesHostData(320, 0);
   std::vector<int32_t> kscaleHostData(128, 2);
   std::vector<int32_t> vscaleHostData(128, 2);
   std::vector<int32_t> koffsetHostData(128, 2);
   std::vector<int32_t> voffsetHostData(128, 2);
 
   std::vector<int32_t> weightHostData(1280, 2);
   std::vector<int32_t> activationHostData(1280, 2);
   std::vector<int32_t> biasHostData(8192, 2);
 
   void* inputDeviceAddr = nullptr;
   void* cosDeviceAddr = nullptr;
   void* sinDeviceAddr = nullptr;
   void* kcacheDeviceAddr = nullptr;
   void* vcacheDeviceAddr = nullptr;
   void* indicesDeviceAddr = nullptr;
   void* kscaleDeviceAddr = nullptr;
   void* vscaleDeviceAddr = nullptr;
   void* koffsetDeviceAddr = nullptr;
   void* voffsetDeviceAddr = nullptr;
 
   void* weightDeviceAddr = nullptr;
   void* activationDeviceAddr = nullptr;
   void* biasDeviceAddr = nullptr;
 
   aclTensor* input = nullptr;
   aclTensor* cos = nullptr;
   aclTensor* sin = nullptr;
   aclTensor* kcache = nullptr;
   aclTensor* vcache = nullptr;
   aclTensor* indices = nullptr;
   aclTensor* kscale = nullptr;
   aclTensor* vscale = nullptr;
   aclTensor* koffset = nullptr;
   aclTensor* voffset = nullptr;
   aclTensor* weight = nullptr;
   aclTensor* activation = nullptr;
   aclTensor* bias = nullptr;
 
   ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_INT32, &input);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
   ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
   ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
   ret = CreateAclTensor(kcacheHostData, kcacheShape, &kcacheDeviceAddr, aclDataType::ACL_INT8, &kcache);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
   ret = CreateAclTensor(vcacheHostData, vcacheShape, &vcacheDeviceAddr, aclDataType::ACL_INT8, &vcache);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
   ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
   ret = CreateAclTensor(kscaleHostData, kscaleShape, &kscaleDeviceAddr, aclDataType::ACL_FLOAT, &kscale);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
   ret = CreateAclTensor(vscaleHostData, vscaleShape, &vscaleDeviceAddr, aclDataType::ACL_FLOAT, &vscale);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
   ret = CreateAclTensor(koffsetHostData, koffsetShape, &koffsetDeviceAddr, aclDataType::ACL_FLOAT, &koffset);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
   ret = CreateAclTensor(voffsetHostData, voffsetShape, &voffsetDeviceAddr, aclDataType::ACL_FLOAT, &voffset);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
 
   ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
   ret = CreateAclTensor(activationHostData, activationShape, &activationDeviceAddr, aclDataType::ACL_FLOAT, &activation);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
   ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
 
 
   std::vector<int64_t> qShape = {320,1,8,128};
   std::vector<int16_t> qHostData(320*8*128, 9);
   aclTensor* q = nullptr;
   void* qDeviceAddr = nullptr;
   std::vector<int64_t> kShape = {320,1,1,128};
   std::vector<int16_t> kHostData(320*128, 10);
   aclTensor* k = nullptr;
   void* kDeviceAddr = nullptr;
   std::vector<int64_t> vShape = {320,1,1, 128};
   std::vector<int16_t> vHostData(320*128, 10);
   aclTensor* v = nullptr;
   void* vDeviceAddr = nullptr;
 
   ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
   ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
   ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
   CHECK_RET(ret == ACL_SUCCESS, return ret);
 
   std::vector<int64_t> splitData = {1024, 128, 128};
   aclIntArray *sizeSplits = aclCreateIntArray(splitData.data(), splitData.size());
 
   // 3. 调用CANN算子库API，需要修改为具体的API
   uint64_t workspaceSize = 0;
   aclOpExecutor* executor;
 
   // 调用aclnnDequantRopeQuantKvcache第一段接口
   ret = aclnnDequantRopeQuantKvcacheGetWorkspaceSize(input, cos, sin, kcache, vcache, indices, kscale, vscale, koffset,
                                                      voffset, weight, activation, bias,sizeSplits, "static", "BSND", true,
                                                      "contiguous", q, k, v, &workspaceSize, &executor);
   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantRopeQuantKvcacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
 
   // 根据第一段接口计算出的workspaceSize申请device内存
   void* workspaceAddr = nullptr;
   if (workspaceSize > 0) {
     ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
   }
 
   // 调用aclnnDequantRopeQuantKvcache第二段接口
   ret = aclnnDequantRopeQuantKvcache(workspaceAddr, workspaceSize, executor, stream);
   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantRopeQuantKvcache failed. ERROR: %d\n", ret); return ret);
 
   // 4. 固定写法，同步等待任务执行结束
   ret = aclrtSynchronizeStream(stream);
   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
 
   // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
   PrintOutResult(kcacheShape, &kcacheDeviceAddr);
   PrintOutResult(vcacheShape, &vcacheDeviceAddr);
 
   // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
   aclDestroyTensor(input);
   aclDestroyTensor(q);
 
   // 7. 释放device 资源
   aclrtFree(inputDeviceAddr);
   aclrtFree(qDeviceAddr);
   if (workspaceSize > 0) {
     aclrtFree(workspaceAddr);
   }
   aclrtDestroyStream(stream);
   aclrtResetDevice(deviceId);
   aclFinalize();
 
   return 0;
 }