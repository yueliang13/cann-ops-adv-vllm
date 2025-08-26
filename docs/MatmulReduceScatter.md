# aclnnMatmulReduceScatter

## 支持的产品型号

- Atlas A2 训练系列产品

**说明：** 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnMatmulReduceScatterGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMatmulReduceScatter”接口执行计算。

* `aclnnStatus aclnnMatmulReduceScatterGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode, const aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnMatmulReduceScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能说明

-   **算子功能**：完成mm + reduce\_scatter\_base计算。
-   **计算公式**：

    $$
    output=reducescatter(x1@x2+bias)
    $$

## aclnnMatmulReduceScatterGetWorkspaceSize

-   **参数说明：**
    -   x1（aclTensor\*，计算输入）：Device侧的aclTensor，即计算公式中的x1。数据类型支持FLOAT16、BFLOAT16，且与x2的数据类型保持一致。[数据格式](common/数据格式.md)支持ND。**当前版本仅支持两维输入，且仅支持不转置场景**。
    -   x2（aclTensor\*，计算输入）：Device侧的aclTensor，即计算公式中的x2。数据类型支持FLOAT16、BFLOAT16，且与x1的数据类型保持一致。[数据格式](common/数据格式.md)支持ND。支持通过转置构造的[非连续的Tensor](common/非连续的Tensor.md)。**当前版本仅支持两维输入**。
    -   bias（aclTensor\*，计算输入）：Device侧的aclTensor，即计算公式中的bias。数据类型支持FLOAT16、BFLOAT16。[数据格式](common/数据格式.md)支持ND。支持传入空指针的场景。**当前版本仅支持一维输入，且暂不支持bias输入为非0的场景**。
    -   group（char\*，计算输入）：Host侧的char，标识通信域的字符串，通信域名称。数据类型支持String。通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。
    -   reduceOp（char\*，计算输入）：Host侧的char，reduce操作类型。数据类型支持String。**当前版本仅支持“sum”。**
    -   commTurn（int64\_t，计算输入）：Host侧的整型，通信数据切分数，即总数据量/单次通信量。数据类型支持INT64。**当前版本仅支持输入0。**
    -   streamMode（int64\_t，计算输入）：Host侧的整型，AscendCL流模式的枚举，当前只支持枚举值1，数据类型支持INT64。
    -   output（aclTensor\*，计算输出）：Device侧的aclTensor，mm计算+reducescatter通信的结果，即计算公式中的output。数据类型支持FLOAT16、BFLOAT16，且与x1的数据类型保持一致。[数据格式](common/数据格式.md)支持ND。
    -   workspaceSize（uint64\_t\*，出参）：Device侧的整型，返回需要在Device侧申请的workspace大小。
    -   executor（aclOpExecutor\*\*，出参）：Device侧的aclOpExecutor，返回op执行器，包含了算子计算流程。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

    ```
    第一段接口完成入参校验，出现以下场景时报错：
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的x1、x2或output是空指针。
    161002 (ACLNN_ERR_PARAM_INVALID): 1. x1、x2、bias或output的数据类型不符合约束要求。
                                      2. streamMode不在合法范围内。
                                      3. x1、x2或output的shape不符合约束要求。
    ```

## aclnnMatmulReduceScatter

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMatmulReduceScatterGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- 输入x1为2维，其shape为\(m, k\)，m须为卡数rank\_size的整数倍。
- 输入x2必须是2维，其shape为\(k, n\)，轴满足mm算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。
- x1/x2支持的空tensor场景，m和n可以为空，k不可为空，且需满足以下条件：
    - m为空，k不为空，n不为空；
    - m不为空，k不为空，n为空；
    - m为空，k不为空，n为空。
- x2矩阵支持转置/不转置场景，x1矩阵只支持不转置场景。
- x1、x2计算输入的数据类型要和output计算输出的数据类型一致。
- bias暂不支持输入为非0的场景。
- 输出为2维，其shape为\(m/rank\_size, n\), rank\_size为卡数。
- Atlas A2 训练系列产品支持2、4、8卡

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include <thread>
#include <iostream>
#include <vector>
#include "aclnnop/aclnn_matmul_reduce_scatter.h"

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

constexpr int DEV_NUM = 8;

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
    HcclComm hcclComm;
    aclrtStream stream;
  };

int launchOneThread_MmReduceScatter(Args &args)
{
    int ret = aclrtSetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ERROR: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
    std::vector<int64_t> x1Shape = {1024, 256};
    std::vector<int64_t> x2Shape = {256, 512};
    std::vector<int64_t> biasShape = {512};
    std::vector<int64_t> outShape = {1024 / DEV_NUM, 512};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *out = nullptr;

    int64_t commTurn = 0;
    int64_t streamMode = 1;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long outShapeSize = GetShapeSize(outShape);

    std::vector<int16_t> x1HostData(x1ShapeSize, 0);
    std::vector<int16_t> x2HostData(x2ShapeSize, 0);
    std::vector<int16_t> biasHostData(biasShapeSize, 0);
    std::vector<int16_t> outHostData(outShapeSize, 0);
    // 创建tensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用第一阶段接口
    ret = aclnnMatmulReduceScatterGetWorkspaceSize(
        x1, x2, bias, hcomName, "sum", commTurn, streamMode, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnMatmulReduceScatterGetWorkspaceSize failed. ret = %d \n", ret); return ret);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // 调用第二阶段接口
    ret = aclnnMatmulReduceScatter(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMatmulReduceScatter failed. ret = %d \n", ret); return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
        return ret);
    LOG_PRINT("[INFO] device_%d aclnnMatmulReduceScatter execute successfully.\n", args.rankId);
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
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }
    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }
    // 初始化集合通信域
    HcclComm comms[DEV_NUM];
    ret = HcclCommInitAll(DEV_NUM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[DEV_NUM];
    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThread_MmReduceScatter, std::ref(args[rankId])));    
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }
    for (int i = 0; i < DEV_NUM; i++) {
        auto hcclRet = HcclCommDestroy(comms[i]);
        CHECK_RET(hcclRet == HCCL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestory failed. ret = %d \n", ret); return -1);
    }
    aclFinalize();
    return 0;
}
```
