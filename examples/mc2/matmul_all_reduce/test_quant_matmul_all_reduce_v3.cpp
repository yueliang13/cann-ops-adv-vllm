/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <thread>
#include "aclnnop/aclnn_quant_matmul_all_reduce_v3.h"

int ndev = 8;

#define ACL_CHECK(ret)                                                                                     \
    do {                                                                                                   \
        auto retcode = ret;                                                                                \
        if (retcode != ACL_SUCCESS) {                                                                      \
            printf("[ERROR] acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, retcode); \
            return retcode;                                                                                \
        }                                                                                                  \
    } while (0)

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

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i: shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
    std::string format;
  };

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
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

int launchOneThreadQuantMatmulAllReduce(Args &args) {
    int ret;
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcom_name[128];
    ret = HcclGetCommName(args.hcclComm, hcom_name);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
              args.context);

    std::vector<int64_t> x1Shape = {32, 64};
    std::vector<int64_t> x2Shape = {64, 128};
    std::vector<int64_t> biasShape = {128};
    std::vector<int64_t> dequantScaleShape = {128};
    std::vector<int64_t> pertokenScaleShape = {32};
    std::vector<int64_t> commQuantScale1Shape = {128};
    std::vector<int64_t> commQuantScale2Shape = {128};
    std::vector<int64_t> x3Shape = {32, 128};
    std::vector<int64_t> outShape = {32, 128};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *dequantScaleDeviceAddr = nullptr;
    void *pertokenScaleDeviceAddr = nullptr;
    void *commQuantScale1DeviceAddr = nullptr;
    void *commQuantScale2DeviceAddr = nullptr;
    void *x3DeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *dequantScale = nullptr;
    aclTensor *pertokenScale = nullptr;
    aclTensor *commQuantScale1 = nullptr;
    aclTensor *commQuantScale2 = nullptr;
    aclTensor *x3 = nullptr;
    aclTensor *out = nullptr;

    int64_t commTurn = 0;
    int64_t streamMode = 1;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long dequantScaleShapeSize = GetShapeSize(dequantScaleShape);
    long long pertokenScaleShapeSize = GetShapeSize(pertokenScaleShape);
    long long commQuantScale1ShapeSize = GetShapeSize(commQuantScale1Shape);
    long long commQuantScale2ShapeSize = GetShapeSize(commQuantScale2Shape);
    long long x3ShapeSize = GetShapeSize(x3Shape);
    long long outShapeSize = GetShapeSize(outShape);

    std::vector<int8_t> x1HostData(x1ShapeSize, 1);
    std::vector<int8_t> x2HostData(x2ShapeSize, 1);
    std::vector<int32_t> biasHostData(biasShapeSize, 1);
    std::vector<int32_t> dequantScaleHostData(dequantScaleShapeSize, 1);
    std::vector<int32_t> pertokenScaleHostData(pertokenScaleShapeSize, 1);
    std::vector<int16_t> commQuantScale1HostData(commQuantScale1ShapeSize, 1);
    std::vector<int16_t> commQuantScale2HostData(commQuantScale2ShapeSize, 1);
    std::vector<int16_t> x3HostData(x3ShapeSize, 1);
    std::vector<int16_t> outHostData(outShapeSize, 0);
    // 创建 tensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dequantScaleHostData, dequantScaleShape, &dequantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &dequantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &pertokenScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(commQuantScale1HostData, commQuantScale1Shape, &commQuantScale1DeviceAddr,
                          aclDataType::ACL_FLOAT16, &commQuantScale1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(commQuantScale2HostData, commQuantScale2Shape, &commQuantScale2DeviceAddr,
                          aclDataType::ACL_FLOAT16, &commQuantScale2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 调用第一段接口
    ret = aclnnQuantMatmulAllReduceV3GetWorkspaceSize(x1, x2, bias, x3, dequantScale, pertokenScale,
                                                      commQuantScale1, commQuantScale2, hcom_name,
                                                      "sum", commTurn, streamMode, out,
                                                      &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnQuantMatmulAllReduceV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用第二段接口
    ret = aclnnQuantMatmulAllReduceV3(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulAllReduceV3 failed. ERROR: %d\n", ret); return ret);
    //（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("device%d aclnnQuantMatmulAllReduceV3 execute success \n", args.rankId);
    // 释放device资源，需要根据具体API的接口定义修改
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (dequantScale != nullptr) {
        aclDestroyTensor(dequantScale);
    }
    if (pertokenScale != nullptr) {
        aclDestroyTensor(pertokenScale);
    }
    if (commQuantScale1 != nullptr) {
        aclDestroyTensor(commQuantScale1);
    }
    if (commQuantScale2 != nullptr) {
        aclDestroyTensor(commQuantScale2);
    }
    if (x3 != nullptr) {
        aclDestroyTensor(x3);
    }
    if (out != nullptr) {
        aclDestroyTensor(out);
    }
    if (x1DeviceAddr != nullptr) {
        aclrtFree(x1DeviceAddr);
    }
    if (x2DeviceAddr != nullptr) {
        aclrtFree(x2DeviceAddr);
    }
    if (biasDeviceAddr != nullptr) {
        aclrtFree(biasDeviceAddr);
    }
    if (dequantScaleDeviceAddr != nullptr) {
        aclrtFree(dequantScaleDeviceAddr);
    }
    if (pertokenScaleDeviceAddr != nullptr) {
        aclrtFree(pertokenScaleDeviceAddr);
    }
    if (commQuantScale1DeviceAddr != nullptr) {
        aclrtFree(commQuantScale1DeviceAddr);
    }
    if (commQuantScale2DeviceAddr != nullptr) {
        aclrtFree(commQuantScale2DeviceAddr);
    }
    if (x3DeviceAddr != nullptr) {
        aclrtFree(x3DeviceAddr);
    }
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(args.stream);
    HcclCommDestroy(args.hcclComm);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    return 0;
}

int main(int argc, char *argv[]) {
    int ret;
    int32_t devices[ndev];
    for (int i = 0; i < ndev; i++) {
        devices[i] = i;
    }
    HcclComm comms[128];
    ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    // 初始化集合通信域
    for (int i = 0; i < ndev; i++) {
        ret = aclrtSetDevice(devices[i]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    }
    ret = HcclCommInitAll(ndev, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
    Args args[ndev];
    aclrtStream stream[ndev];
    aclrtContext context[ndev];
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    }
    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(
                new(std::nothrow) std::thread(&launchOneThreadQuantMatmulAllReduce, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}