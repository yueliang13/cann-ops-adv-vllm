声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# MoeInitRouting

## 支持的产品型号
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
- Atlas A3 训练系列产品/Atlas 800I A3 推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 功能说明

- 算子功能：MoE的routing计算，根据[aclnnMoeGatingTopKSoftmax](aclnnMoeGatingTopKSoftmax.md)的计算结果做routing处理。
## 实现原理

- 计算公式：<br>
    $$
    expandedExpertIdx,sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

    $$
    expandedRowIdx[sortedRowIdx[i]]=i
    $$

    $$
    expandedX[i]=x[sortedRowIdx[i]\%numRows]
    $$
## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用 “aclnnMoeInitRoutingGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeInitRouting”接口执行计算。

* `aclnnStatus aclnnMoeInitRoutingGetWorkspaceSize(const aclTensor *x, const aclTensor *rowIdx, const aclTensor *expertIdx, int64_t activeNum, const aclTensor *expandedXOut, const aclTensor *expandedRowIdxOut, const aclTensor *expandedExpertIdxOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeInitRouting(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`


**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

## aclnnMoeInitRoutingGetWorkspaceSize

-   **参数说明**：
    -   x（aclTensor\*，计算输入）：MOE的输入即token特征输入，要求为一个2D的Tensor，shape为 \(NUM\_ROWS, H\)，数据类型支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND，支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   rowIdx（aclTensor\*，计算输入）：指示每个位置对应的原始行位置，shape要求与expertIdx 一致, 数值从0开始，沿着1维递增。数据类型支持int32，[数据格式](common/数据格式.md)要求为ND，支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   expertIdx （aclTensor\*，计算输入）：[aclnnMoeGatingTopKSoftmax](aclnnMoeGatingTopKSoftmax.md)的输出每一行特征对应的K个处理专家，要求是一个2D的shape \(NUM\_ROWS, K\)。数据类型支持int32，[数据格式](common/数据格式.md)要求为ND，支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   activeNum（int64\_t，计算输入）：表示总的最大处理row数且大于等于0，expandedXOut只有这么多行是有效的。
    -   expandedXOut（aclTensor\*，计算输出）：根据expertIdx进行扩展过的特征，要求是一个2D的Tensor，shape \(min\(NUM\_ROWS, activeNum\) \* k, H\)。数据类型同x，支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND，不支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   expandedRowIdxOut（aclTensor\*，计算输出）：expandedX和x的映射关系， 要求是一个1D的Tensor，Shape为\(NUM\_ROWS\*K, \)，数据类型支持int32，[数据格式](common/数据格式.md)要求为ND，不支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   expandedExpertIdxOut（aclTensor\*，计算输出）：输出expertIdx排序后的结果，数据类型支持int32，[数据格式](common/数据格式.md)要求为ND，不支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
    -   executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

-   **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
    ```
    第一段接口完成入参校验，出现以下场景时报错:
    161001(ACLNN_ERR_PARAM_NULLPTR): 1. 输入和输出的Tensor是空指针。
    161002(ACLNN_ERR_PARAM_INVALID): 1. 输入和输出的数据类型不在支持的范围内。
    561002(ACLNN_ERR_INNER_TILING_ERROR): 1. x的shape维度不为2。
                                          2. rowIdx的shape不为2或者rowIdx和expertIdx的shape不相等。
                                          3. activateNum的值小于0。
                                          4. expandedXOut的shape不等于(min(num_rows, activateNum) * k, H)。
                                          5. expandedRowIdxOut和expandedExpertIdxOut的shape不相等，且不等于(num_rows * k, )。
    ```

## aclnnMoeInitRouting

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeInitRoutingGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

无。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing.h"
#include <iostream>
#include <vector>
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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    // 1. 固定写法，device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> expandedXOutShape = {6, 4};
    std::vector<int64_t> idxOutShape = {6};
    void* xDeviceAddr = nullptr;
    void* rowIdxDeviceAddr = nullptr;
    void* expertIdxDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* expandedRowIdxOutDeviceAddr = nullptr;
    void* expandedExpertIdxOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* rowIdx = nullptr;
    aclTensor* expertIdx = nullptr;
    int64_t activeNum = 3;
    aclTensor* expandedXOut = nullptr;
    aclTensor* expandedRowIdxOut = nullptr;
    aclTensor* expandedExpertIdxOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 1, 0, 2};
    std::vector<int> rowIdxHostData = {0, 3, 1, 4, 2, 5};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0};
    std::vector<int> expandedExpertIdxOutHostData = {0, 0, 0, 0, 0, 0};
    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rowIdxHostData, idxShape, &rowIdxDeviceAddr, aclDataType::ACL_INT32, &rowIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, idxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_FLOAT, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData, idxOutShape, &expandedRowIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedExpertIdxOutHostData, idxOutShape, &expandedExpertIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedExpertIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeInitRouting第一段接口
    ret = aclnnMoeInitRoutingGetWorkspaceSize(x, rowIdx, expertIdx, activeNum, expandedXOut, expandedRowIdxOut, expandedExpertIdxOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMoeInitRouting第二段接口
    ret = aclnnMoeInitRouting(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRouting failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<float> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr, expandedXSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %f\n", i, expandedXData[i]);
    }
    auto expandedRowIdxSize = GetShapeSize(idxOutShape);
    std::vector<int> expandedRowIdxData(expandedRowIdxSize, 0);
    ret = aclrtMemcpy(expandedRowIdxData.data(), expandedRowIdxData.size() * sizeof(expandedRowIdxData[0]), expandedRowIdxOutDeviceAddr, expandedRowIdxSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedRowIdxSize; i++) {
        LOG_PRINT("expandedRowIdxData[%ld] is: %d\n", i, expandedRowIdxData[i]);
    }
    auto expandedExpertIdxSize = GetShapeSize(idxOutShape);
    std::vector<int> expandedExpertIdxData(expandedExpertIdxSize, 0);
    ret = aclrtMemcpy(expandedExpertIdxData.data(), expandedExpertIdxData.size() * sizeof(expandedExpertIdxData[0]), expandedExpertIdxOutDeviceAddr, expandedExpertIdxSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedExpertIdxSize; i++) {
        LOG_PRINT("expandedExpertIdxData[%ld] is: %d\n", i, expandedExpertIdxData[i]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(rowIdx);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(expandedRowIdxOut);
    aclDestroyTensor(expandedExpertIdxOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(rowIdxDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(expandedRowIdxOutDeviceAddr);
    aclrtFree(expandedExpertIdxOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

