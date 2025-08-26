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
 * \file test_moe_distribute_dispatch.cpp
 * \brief
 */
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn_moe_distribute_dispatch.h"
#include "aclnn/opdev/fp16_t.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)
#define ACLCHECK(ret) do { \
    if(ret != ACL_SUCCESS)\
    {\
        printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);\
    }\
} while(0)
constexpr int EP_WORLD_SIZE = 16;
constexpr int TP_WORLD_SIZE = 0;
int FIRST_RANK_ID = 0;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i +1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct Args {
    int rankId;
    int epRankId;
    char* groupEpName;
    HcclComm hcclEpComm;
    aclrtStream stream;
};

int launchOneProcess_MoeDistributeCombine(Args &args)
{
    int64_t BS = 8;
    int64_t H = 7168;
    int64_t K = 8;
    int64_t shardType = 0; // dispatch need
    int64_t quantMode = 0; // dispatch need
    bool isQuant = false;  // dispatch need
    int64_t expertTokenNumsType = 0; // dispatch need
    int64_t expertShardType = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t sharedExpertNum = 0;
    int64_t moeExpertNum = 16;
    int64_t globalBS = BS * EP_WORLD_SIZE;      // tiling里处理成BS*world_size
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupList_type = 0;
    const char* groupTpName = "";
    int64_t tpWorldSize = 0;
    int64_t tpRankId = 0;

    int64_t localMoeExpertNum = moeExpertNum / EP_WORLD_SIZE;
    int64_t A = BS * EP_WORLD_SIZE * localMoeExpertNum; // A 表示本卡可能接收的最大token数量
    int64_t epWorldSize = EP_WORLD_SIZE;
    auto outDataType = aclDataType::ACL_BF16;
    if (isQuant) {
        outDataType = aclDataType::ACL_INT8;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    std::vector<int64_t> scalesShape{moeExpertNum, H};              // dispatch need
    std::vector<int64_t> dynamicScalesShape{A};                     // dispatch need
    std::vector<int64_t> expertTokenNumsShape{localMoeExpertNum};   // dispatch need
    std::vector<int64_t> expandScalesShape{A}; // dispatch & combine
    std::vector<int64_t> expandXShape{A, H};
    std::vector<int64_t> expertIdsShape{BS, K};
    std::vector<int64_t> expandIdxShape{BS * K};
    std::vector<int64_t> epSendCountsShape{localMoeExpertNum * EP_WORLD_SIZE};
    std::vector<int64_t> expertScalesShape{BS, K};
    std::vector<int64_t> tpSendCountsShape{1};
    std::vector<int64_t> xActiveMaskShape{BS};
    std::vector<int64_t> activationScaleShape{A};
    std::vector<int64_t> weightScaleShape{1, H};
    std::vector<int64_t> groupListShape{1};
    std::vector<int64_t> xShape{BS, H};

    void *scalesDeviceAddr = nullptr;           // dispatch need
    void *dynamicScalesDeviceAddr = nullptr;    // dispatch need
    void *expertTokenNumsDeviceAddr = nullptr;  // dispatch need
    void *expandScalesDeviceAddr = nullptr;     // dispatch & combine need
    void *expandXDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *expandIdxDeviceAddr = nullptr;
    void *epSendCountsDeviceAddr = nullptr;
    void *expertScalesDeviceAddr = nullptr;
    void *tpSendCountsDeviceAddr = nullptr;
    void *xActiveMaskDeviceAddr = nullptr; 
    void *activationScaleDeviceAddr = nullptr;
    void *weightScaleDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    void *xDeviceAddr = nullptr;
    
    aclTensor *scales = nullptr;            // dispatch need
    aclTensor *dynamicScales = nullptr;     // dispatch need
    aclTensor *expertTokenNums = nullptr;   // dispatch need
    aclTensor *expandScales = nullptr;      // dispatch & combine need
    aclTensor *expandX = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *expandIdx = nullptr;
    aclTensor *epSendCounts = nullptr;
    aclTensor *expertScales = nullptr;
    aclTensor *tpSendCounts = nullptr;
    aclTensor *xActiveMask = nullptr; 
    aclTensor *activationScale = nullptr;
    aclTensor *weightScale = nullptr;
    aclTensor *groupList = nullptr;
    aclTensor *x = nullptr;

    long long scalesShapeSize = GetShapeSize(scalesShape);                      // dispatch need
    long long dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);        // dispatch need
    long long expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);    // dispatch need
    long long expandScalesShapeSize = GetShapeSize(expandScalesShape);          // dispatch & combine need
    long long expandXShapeSize = GetShapeSize(expandXShape);
    long long expertIdsShapeSize = GetShapeSize(expertIdsShape);
    long long expandIdxShapeSize = GetShapeSize(expandIdxShape);
    long long epSendCountsShapeSize = GetShapeSize(epSendCountsShape);
    long long expertScalesShapeSize = GetShapeSize(expertScalesShape);
    long long tpSendCountsShapeSize = GetShapeSize(tpSendCountsShape);
    long long xActiveMaskShapeSize = GetShapeSize(xActiveMaskShape);
    long long activationScaleShapeSize = GetShapeSize(activationScaleShape);
    long long weightScaleShapeSize = GetShapeSize(weightScaleShape);
    long long groupListShapeSize = GetShapeSize(groupListShape);
    long long xShapeSize = GetShapeSize(xShape);

    std::vector<float> scalesHostData(scalesShapeSize, 0);                      // dispatch need
    std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);        // dispatch need
    std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);  // dispatch need
    std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);          // dispatch & combine need
    std::vector<op::fp16_t> expandXHostData(expandXShapeSize, 0);
    std::vector<int32_t> expertIdsHostData(expertIdsShapeSize, 0);
    std::random_device rd; // 随机数设备
    std::mt19937 gen(rd()); // 以随机数设备作为种子的Mersenne Twister生成器
    std::uniform_int_distribution<> dis(sharedExpertRankNum, EP_WORLD_SIZE - 1);
    for (auto& val : expertIdsHostData) {
        val = dis(gen); // 为每个元素生成一个2到15之间的随机数
    }
    std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
    std::vector<int32_t> epSendCountsHostData(epSendCountsShapeSize, 0);
    std::vector<int32_t> tpSendCountsHostData(tpSendCountsShapeSize, 0);
    std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
    std::vector<int8_t> xActiveMaskHostData(xActiveMaskShapeSize, 0);
    std::vector<float> activationScaleHostData(activationScaleShapeSize,0);
    std::vector<float> weightScaleHostData(weightScaleShapeSize,0);
    std::vector<int32_t> groupListHostData(groupListShapeSize,0);
    std::vector<op::fp16_t> xHostData(xShapeSize, 0);

    auto ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);                                // dispatch need
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);         // dispatch need
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums); // dispatch need
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);             // dispatch & combine need
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, aclDataType::ACL_BF16, &expandX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(epSendCountsHostData, epSendCountsShape, &epSendCountsDeviceAddr, aclDataType::ACL_INT32, &epSendCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tpSendCountsHostData, tpSendCountsShape, &tpSendCountsDeviceAddr, aclDataType::ACL_INT32, &tpSendCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xActiveMaskHostData, xActiveMaskShape, &xActiveMaskDeviceAddr, aclDataType::ACL_BOOL, &xActiveMask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(activationScaleHostData, activationScaleShape, &activationScaleDeviceAddr, aclDataType::ACL_FLOAT, &activationScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT, &weightScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT32, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    /******************************先调用dispatch,因为combine需要使用dispatch的数据********************************************/
    ret = aclnnMoeDistributeDispatchGetWorkspaceSize(x, expertIds, 
            (isQuant? scales : nullptr), xActiveMask, 
            expertScales, args.groupEpName, epWorldSize, args.epRankId, moeExpertNum, groupTpName, tpWorldSize, tpRankId, expertShardType, 
            sharedExpertNum,sharedExpertRankNum, quantMode, globalBS, expertTokenNumsType, expandX, dynamicScales, 
            expandIdx, expertTokenNums, epSendCounts, tpSendCounts, expandScales, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchGetWorkspaceSize failed. ret = %d \n", ret);
        return ret;
    }
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchGetWorkspaceSize failed. ret = %d \n", ret); return ret);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // 调用第二阶段接口
    ret = aclnnMoeDistributeDispatch(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatch failed. ret = %d \n", ret);
        return ret);

    // 释放device资源，需要根据具体API的接口定义修改
    if (scales != nullptr) {                // dispatch need
        aclDestroyTensor(scales);
    }
    if (dynamicScales != nullptr) {         // dispatch need
        aclDestroyTensor(dynamicScales);
    }
    if (expertTokenNums != nullptr) {       // dispatch need
        aclDestroyTensor(expertTokenNums);
    }
    if (expandScales != nullptr) {          // dispatch & combine need
        aclDestroyTensor(expandScales);
    }
    if (expandX != nullptr) {
        aclDestroyTensor(expandX);
    }
    if (expertIds != nullptr) {
        aclDestroyTensor(expertIds);
    }
    if (expandIdx != nullptr) {
        aclDestroyTensor(expandIdx);
    }
    if (epSendCounts != nullptr) {
        aclDestroyTensor(epSendCounts);
    }
    if (tpSendCounts != nullptr) {
        aclDestroyTensor(tpSendCounts);
    }
    if (expertScales != nullptr) {
        aclDestroyTensor(expertScales);
    }
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (expandXDeviceAddr != nullptr) {
        aclrtFree(expandXDeviceAddr);
    }
    if (expertIdsDeviceAddr != nullptr) {
        aclrtFree(expertIdsDeviceAddr);
    }
    if (expandIdxDeviceAddr != nullptr) {
        aclrtFree(expandIdxDeviceAddr);
    }
    if (epSendCountsDeviceAddr != nullptr) {
        aclrtFree(epSendCountsDeviceAddr);
    }
    if (tpSendCountsDeviceAddr != nullptr) {
        aclrtFree(tpSendCountsDeviceAddr);
    }
    if (expertScalesDeviceAddr != nullptr) {
        aclrtFree(expertScalesDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(args.stream);
    HcclCommDestroy(args.hcclEpComm);
    aclrtResetDevice(args.rankId);
    return 0;
}

void RunInProcess(int rank, int rankSize)
{
    // 1. acl init
    Args args;
    aclrtStream stream;
    ACLCHECK(aclInit(nullptr));
    ACLCHECK(aclrtSetDevice(rank));
    ACLCHECK(aclrtCreateStream(&stream));

    // 2. create HcclComm by rankFile
    char commName[128] = "";
    HcclComm hcclComm = nullptr;
    char *rankTableFile = getenv("RANK_TABLE_FILE");

    std::string rankTableFileStr(rankTableFile);
    std::cout << "rankTableFilePath is :" << rankTableFileStr << std::endl;
    int rank_id = rank + FIRST_RANK_ID;
    auto ret = HcclCommInitClusterInfo(rankTableFile, rank_id, &hcclComm);
    if (ret != HCCL_SUCCESS || hcclComm == nullptr) {
        std::cout << "HCCL CommInitClusterInfo ERROR" << ret << " should check rankTableFile config" << std::endl;
        return;
    }
    std::cout << "HcclCommInitClusterInfo success, rank_id:" << rank_id << ", rankSize:" << rankSize
                  << ", hcclComm:" << hcclComm;
    HcclGetCommName(hcclComm, commName);
    if (commName == "") { 
        std::cout << "rankTableFile CommName should not be null" << std::endl;
    }

    // 3. launch one process for MoeDistributeCombine
    args.rankId = rank;
    args.groupEpName = commName;
    args.hcclEpComm = hcclComm;
    args.epRankId = rank_id;
    args.stream = stream;
    LOG_PRINT("[INFO] rank = %d, groupEpName = %s, stream = %p\n", args.rankId, commName, args.stream);

    int res = launchOneProcess_MoeDistributeCombine(args);
    if (res != ACL_SUCCESS) {
        std::cout << "run launchOneProcess_MoeDistributeCombine failed, ret = " << res << std::endl;
        return;
    }
}

int main(int argc, char *argv[])
{
    char* env_rankID = getenv("FIRST_RANK_ID");
    if (!env_rankID) {
        std::cerr << "FIRST_RANK_ID环境变量未设置！\n";
        return 1;
    }
    FIRST_RANK_ID = std::stoi(std::string(env_rankID));
    std::cout << "FIRST_RANK_ID is: " << FIRST_RANK_ID << std::endl;

    // 所需的进程数量
    const int processCount = 8;
    pid_t pids[processCount];

    for (int i = 0; i < processCount; ++i) {
        pids[i] = fork();
        if (pids[i] < 0) {
            std::cout << "fork failed ! " << pids[i] << std::endl;
        } else if (pids[i] == 0) {
            // 子进程，完成任务后退出
            RunInProcess(i, processCount);
            exit(0);
        }
    }

    // 父进程等待所有子进程完成
    for (int i = 0; i < processCount; ++i) {
        waitpid(pids[i], nullptr, 0);
    }

    return 0;
}