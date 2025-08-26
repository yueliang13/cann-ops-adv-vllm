# MoeComputeExpertTokens

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品
- Atlas A3 训练系列产品/Atlas 800I A3 推理产品

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnMoeComputeExpertTokensGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeComputeExpertTokens”接口执行计算。

* `aclnnStatus aclnnMoeComputeExpertTokensGetWorkspaceSize(const aclTensor* sortedExperts, int64_t numExperts, const aclTensor* out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeComputeExpertTokens(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能说明

-   **算子功能**：MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。
-   **计算公式**：

    $$
    out_{i}=BinarySearch(sortedExperts,numExpert)
    $$

## aclnnMoeComputeExpertTokensGetWorkspaceSize

-   **参数说明：**
    -   sortedExperts（aclTensor\*，计算输入）：Deivice侧的aclTensor，公式中的sortedExpertForSourceRow，排序后的专家数组，要求是一个1D的Tensor，Tensor中的值取值范围是[0, numExperts-1]，数据类型支持INT32，[数据格式](common/数据格式.md)要求为ND。
    -   numExperts（int64\_t，计算输入）：Host侧的int，总专家数。限制范围以约束说明中说明为准。
    -   out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出，要求的是一个1D的Tensor，shape大小等于专家数，数据类型与sortedExpertForSourceRow保持一致。
    -   workspaceSize（unit64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
    -   executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

    ```
    第一段接口完成入参校验，出现以下场景时报错:
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的sortedExperts是空指针时。
    161002 (ACLNN_ERR_PARAM_INVALID): 1. sortedExperts的数据类型不在支持的范围之内。
                                      2. sortedExperts的format格式不在支持的范围之内。
    561002(ACLNN_ERR_INNER_TILING_ERROR): 1. sortedExperts和out的shape不等于1D的tensor。
    ```

## aclnnMoeComputeExpertTokens

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeComputeExpertTokensGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。


## 约束说明

* sortedExperts的shape大小需要小于2\*\*24。
* numExperts的输入常值需要大于0，但不能超过2048。
* 输入shape大小不要超过device可分配的内存上限，否则会导致异常终止。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_compute_expert_tokens.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void** deviceAddr,
    aclDataType dataType, aclTensor** tensor)
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
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> sortedExpertForSourceRowShape = {6};
    std::vector<int64_t> outShape = {3};

    void* sortedExpertForSourceRowAddr = nullptr;
    void* outAddr = nullptr;

    aclTensor* sortedExperts = nullptr;
    aclTensor* out = nullptr;

    std::vector<int32_t> sortedExpertForSourceRowData = {0, 0, 1, 1, 2, 2};
    std::vector<int32_t> outData = {3, 4, 5};
    std::int32_t numExperts = 3;

    // 创建input aclTensor
    ret = CreateAclTensor(sortedExpertForSourceRowData,
        sortedExpertForSourceRowShape,
        &sortedExpertForSourceRowAddr,
        aclDataType::ACL_INT32,
        &sortedExperts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建Out aclTensor
    ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeComputeExpertTokens第一段接口
    ret = aclnnMoeComputeExpertTokensGetWorkspaceSize(
        sortedExperts, numExperts, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeComputeExpertTokensGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMoeComputeExpertTokens第二段接口
    ret = aclnnMoeComputeExpertTokens(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeComputeExpertTokens failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        outAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(sortedExperts);
    aclDestroyTensor(out);

    // 7. 释放Device资源，需要根据具体API的接口定义修改
    aclrtFree(sortedExpertForSourceRowAddr);
    aclrtFree(outAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

