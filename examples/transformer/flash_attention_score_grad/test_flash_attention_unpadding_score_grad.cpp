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
 * \file test_flash_attention_unpadding_score_grad.cpp
 * \brief
 */

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"


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
    // 保存文件
    file.write(static_cast<char *>((void *)resultData.data()), size * sizeof(T));
    file.close();
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
                                  aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
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
    CHECK_RET(binFileBufferLen > 0, std::cout << "File size is 0.\n"; file.close(); return -1);

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
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int CreateAclTensor(std::string &filePath, const std::vector<int64_t> &shape, int typeSize, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * typeSize;
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMallocHost申请host侧内存
    void *binBufferHost = nullptr;
    ret = aclrtMallocHost(&binBufferHost, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);

    // 读取文件
    ret = ReadBinFileNNop(filePath, binBufferHost, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ReadBinFileNNop failed. ERROR: %d\n", ret);
                                  (void)aclrtFreeHost(binBufferHost); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, binBufferHost, size, ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtFreeHost(binBufferHost);
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

void FreeResource(aclTensor *q, aclTensor *k, aclTensor *v, aclTensor *attentionIn, aclTensor *softmaxMax,
    aclTensor *softmaxSum, aclTensor *dx, aclTensor *dq, aclTensor *dk, aclTensor *dv, void *qDeviceAddr,
    void *kDeviceAddr, void *vDeviceAddr, void *dxDeviceAddr, void *softmaxMaxDeviceAddr, void *softmaxSumDeviceAddr,
    uint64_t workspaceSize, void *workspaceAddr, void *attentionInDeviceAddr, void *dqDeviceAddr, void *dkDeviceAddr,
    void *dvDeviceAddr, int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    if (q != nullptr) {
        aclDestroyTensor(q);
    }
    if (k != nullptr) {
        aclDestroyTensor(k);
    }
    if (v != nullptr) {
        aclDestroyTensor(v);
    }
    if (attentionIn != nullptr) {
        aclDestroyTensor(attentionIn);
    }
    if (softmaxMax != nullptr) {
        aclDestroyTensor(softmaxMax);
    }
    if (softmaxSum != nullptr) {
        aclDestroyTensor(softmaxSum);
    }
    if (dx != nullptr) {
        aclDestroyTensor(dx);
    }
    if (dq != nullptr) {
        aclDestroyTensor(dq);
    }
    if (dk != nullptr) {
        aclDestroyTensor(dk);
    }
    if (dv != nullptr) {
        aclDestroyTensor(dv);
    }

    // 释放device资源
    if (qDeviceAddr != nullptr) {
        aclrtFree(qDeviceAddr);
    }
    if (kDeviceAddr != nullptr) {
        aclrtFree(kDeviceAddr);
    }
    if (vDeviceAddr != nullptr) {
        aclrtFree(vDeviceAddr);
    }
    if (dxDeviceAddr != nullptr) {
        aclrtFree(dxDeviceAddr);
    }
    if (softmaxMaxDeviceAddr != nullptr) {
        aclrtFree(softmaxMaxDeviceAddr);
    }
    if (softmaxSumDeviceAddr != nullptr) {
        aclrtFree(softmaxSumDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (attentionInDeviceAddr != nullptr) {
        aclrtFree(attentionInDeviceAddr);
    }
    if (dqDeviceAddr != nullptr) {
        aclrtFree(dqDeviceAddr);
    }
    if (dkDeviceAddr != nullptr) {
        aclrtFree(dkDeviceAddr);
    }
    if (dvDeviceAddr != nullptr) {
        aclrtFree(dvDeviceAddr);
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

int main(int argc, char **argv)
{
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 如果需要修改shape值，需要同步修改../scripts/fa_generate_data.py中 test_flash_attention_unpadding_score_grad 分支下生成
    // query、key、value、dx、attentionIn、softmaxMax、softmaxSum对应的shape值，并重新gen data，再执行
    int64_t batch = 2;
    int64_t sq = 128;
    int64_t skv = 128;
    int64_t headDim = 128;
    int64_t headNum = 4;
    int64_t h = headNum * headDim;
    int64_t tq = batch * sq;
    int64_t tkv = batch * sq;
    std::vector<int64_t> qShape = {tq, headNum, headDim};
    std::vector<int64_t> kShape = {tkv, headNum, headDim};
    std::vector<int64_t> vShape = {tkv, headNum, headDim};
    std::vector<int64_t> dxShape = {tq, headNum, headDim};
    std::vector<int64_t> attentionInShape = {tq, headNum, headDim};
    std::vector<int64_t> softmaxMaxShape = {tq, headNum, 8};
    std::vector<int64_t> softmaxSumShape = {tq, headNum, 8};
    std::vector<int64_t> dqShape = {tq, headNum, headDim};
    std::vector<int64_t> dkShape = {tkv, headNum, headDim};
    std::vector<int64_t> dvShape = {tkv, headNum, headDim};
    std::vector<int64_t> cuSeqLenQVec = {128, 256};
    std::vector<int64_t> cuSeqLenKvVec = {128, 256};
    aclIntArray *cuSeqLenQ = aclCreateIntArray(cuSeqLenQVec.data(), cuSeqLenQVec.size());
    aclIntArray *cuSeqLenKv = aclCreateIntArray(cuSeqLenKvVec.data(), cuSeqLenKvVec.size());
    double scaleValue = 0.088388;
    double keepProb = 1;
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;
    int64_t innerPrecise = 0;
    int64_t sparseMod = 0;
    char layOut[] = "TND";

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *vDeviceAddr = nullptr;
    void *dxDeviceAddr = nullptr;
    void *softmaxMaxDeviceAddr = nullptr;
    void *softmaxSumDeviceAddr = nullptr;
    void *attentionInDeviceAddr = nullptr;
    void *dqDeviceAddr = nullptr;
    void *dkDeviceAddr = nullptr;
    void *dvDeviceAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *v = nullptr;
    aclTensor *dx = nullptr;
    aclTensor *pse = nullptr;
    aclTensor *dropMask = nullptr;
    aclTensor *padding = nullptr;
    aclTensor *attenMask = nullptr;
    aclIntArray *prefix = nullptr;
    aclTensor *softmaxMax = nullptr;
    aclTensor *softmaxSum = nullptr;
    aclTensor *softmaxIn = nullptr;
    aclTensor *attentionIn = nullptr;
    aclTensor *dq = nullptr;
    aclTensor *dk = nullptr;
    aclTensor *dv = nullptr;
    aclTensor *dpse = nullptr;

    std::vector<short> dqHostData(sq * batch * h, 0);
    std::vector<short> dkHostData(skv * batch * h, 0);
    std::vector<short> dvHostData(skv * batch * h, 0);
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    if (argv == nullptr || argv[0] == nullptr) {
        LOG_PRINT("Environment error, Argv=%p, Argv[0]=%p", argv, argv == nullptr ? nullptr : argv[0]);
        return 0;
    }
    std::string exeFile(argv[0]);
    std::string currentPath = std::string(exeFile.substr(0, exeFile.rfind('/')) + "/");
    std::string qFilePath = currentPath + "query.bin";
    ret = CreateAclTensor(qFilePath, qShape, 2, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                  vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);
              return ret);

    std::string kFilePath = currentPath + "key.bin";
    ret = CreateAclTensor(kFilePath, kShape, 2, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                  vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);
              return ret);

    std::string vFilePath = currentPath + "value.bin";
    ret = CreateAclTensor(vFilePath, vShape, 2, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                  vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);
              return ret);

    std::string dxFilePath = currentPath + "dx.bin";
    ret = CreateAclTensor(dxFilePath, dxShape, 2, &dxDeviceAddr, aclDataType::ACL_FLOAT16, &dx);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                  vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);
              return ret);
    std::string attentionInPath = currentPath + "attentionIn.bin";
    ret = CreateAclTensor(attentionInPath, attentionInShape, 2, &attentionInDeviceAddr, aclDataType::ACL_FLOAT16,
                          &attentionIn);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                  vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);
              return ret);

    std::string softmaxMaxPath = currentPath + "softmaxMax.bin";
    ret =
        CreateAclTensor(softmaxMaxPath, softmaxMaxShape, 4, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                  vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);
              return ret);
    std::string softmaxSumPath = currentPath + "softmaxSum.bin";
    ret =
        CreateAclTensor(softmaxSumPath, softmaxSumShape, 4, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                  vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);
              return ret);

    ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                  vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);
              return ret);

    ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                  vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_FLOAT16, &dv);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                  vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);
              return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    aclOpExecutor *executor;

    // 调用aaclnnFlashAttentionUnpaddingScoreGrad第一段接口
    ret = aclnnFlashAttentionUnpaddingScoreGradGetWorkspaceSize(
        q, k, v, dx, pse, dropMask, padding, attenMask, softmaxMax, softmaxSum, softmaxIn, attentionIn, prefix,
        cuSeqLenQ, cuSeqLenKv, scaleValue, keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise, sparseMod,
        dq, dk, dv, dpse, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnFlashAttentionUnpaddingScoreGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                  vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                      vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize,
                      workspaceAddr, attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId,
                      &context, &stream);
                  return ret);
    }

    // 调用aclnnFlashAttentionUnpaddingScoreGrad第二段接口
    ret = aclnnFlashAttentionUnpaddingScoreGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionUnpaddingScoreGrad failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                  vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr,
                  vDeviceAddr, dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  attentionInDeviceAddr, dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);
              return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    std::string dqFileName = "dq.bin";
    SaveOutResult<short>(dqFileName, dqShape, &dqDeviceAddr);

    std::string dkFileName = "dk.bin";
    SaveOutResult<short>(dkFileName, dkShape, &dkDeviceAddr);

    std::string dvFileName = "dv.bin";
    SaveOutResult<short>(dvFileName, dvShape, &dvDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改; 释放device资源
    FreeResource(q, k, v, attentionIn, softmaxMax, softmaxSum, dx, dq, dk, dv, qDeviceAddr, kDeviceAddr, vDeviceAddr,
        dxDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr, attentionInDeviceAddr,
        dqDeviceAddr, dkDeviceAddr, dvDeviceAddr, deviceId, &context, &stream);

    return 0;
}
