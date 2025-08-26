声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# MoeTokenPermute

## 支持的产品型号

Atlas A2 训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

-   算子功能：MoE的permute计算，根据索引indices将tokens广播并排序返回输入tokens到permuteTokensOut的索引。
- 计算公式：

  - paddedModeOptional为`false`时
  
    $$
    sortedIndicesFirst=argSort(Indices)
    $$
  
    $$
    sortedIndicesOut=argSort(sortedIndices)
    $$
  
    $$
    permuteTokens[sortedIndices[i]]=tokens[i//topK]
    $$
  
  - paddedModeOptional为`true`时
  
    $$
    permuteTokensOut[i]=tokens[Indices[i]]
    $$
  
    $$
    sortedIndicesOut=Indices
    $$

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用 “aclnnMoeTokenPermuteGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenPermute”接口执行计算。

* `aclnnStatus aclnnMoeTokenPermuteGetWorkspaceSize(const aclTensor *tokens, const aclTensor *indices, int64_t numOutTokensOptional, bool paddedModeOptional, aclTensor *permuteTokensOut, aclTensor *sortedIndicesOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeTokenPermute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明：**
- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

## aclnnMoeTokenPermuteGetWorkspaceSize

- **参数说明：**
  
  - tokens（aclTensor\*，计算输入）：输入token，要求为一个维度为2D的Tensor，shape为（num\_tokens，hidden\_size），数据类型支持FLOAT16、BFLOAT16、FLOAT32，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)要求为ND。
  - indices （aclTensor\*，计算输入）：输入indices，要求shape为2D或1D。paddedModeOptional为false时表示每一个输入token对应的topK个处理专家索引，shape为（num\_tokens，topK）或（num\_tokens），paddedModeOptional为true时表示每个专家选中的token索引（暂不支持），数据类型支持INT32、INT64，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)要求为ND。要求元素个数小于16777215，值大于等于0小于16777215（单点支持int32或int64的最大或最小值），topK小于等于512。
  - numOutTokensOptional（int64\_t，计算输入）：有效输出token数，设置为0时，表示不会删除任何token。不为0时，会按照numOutTokensOptional进行切片丢弃按照indices排序好的token中超过numOutTokensOptional的部分，为负数时按照切片索引为负数时处理。
  - paddedModeOptional（bool，计算输入）：paddedModeOptional为true时表示indices已被填充为代表每个专家选中的token索引，此时不对indices进行排序。目前仅支持paddedModeOptional为false。
  - permuteTokensOut（aclTensor\*，计算输出）：根据indices进行扩展并排序过的tokens，要求是一个2D的Tensor，shape为（min\(num\_tokens \* topK, num_out_tokens\), hidden\_size）。数据类型同tokens，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)要求为ND。
  - sortedIndicesOut（aclTensor\*，计算输出）：permuteTokensOut和tokens的映射关系， 要求是一个1D的Tensor，Shape为（num\_tokens\*topK，），数据类型支持INT32，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)要求为ND。
  - workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR): 1. 输入和输出的Tensor是空指针。
  161002(ACLNN_ERR_PARAM_INVALID): 1. 输入和输出的数据类型不在支持的范围内。
  ```
## aclnnMoeTokenPermute

- **参数说明：**
  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenPermuteGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- indices 要求元素个数小于`16777215`，值大于等于`0`小于`16777215`(单点支持int32或int64的最大或最小值，其余值不在范围内排序结果不正确)，第二维小于`512`。
- 不支持paddedModeOptional为`True`。

## 算子原型

```c++
REG_OP(MoeTokenPermute)
    .INPUT(tokens, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT64, DT_INT64, DT_INT64, DT_INT32, DT_INT32, DT_INT32}))
    .OUTPUT(permuteTokensOut, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(sortedIndicesOut, TensorType({DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32}))
    .ATTR(numOutTokensOptional, Int, 0)
    .ATTR(paddedModeOptional, Bool, false)
    .OP_END_FACTORY_REG(MoeTokenPermute)
```
## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute.h"
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
    void* indicesDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* sortedIndicesOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* indices = nullptr;
    int64_t numTokenOut = 0;
    bool padMode = false;

    aclTensor* expandedXOut = nullptr;
    aclTensor* sortedIndicesOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> indicesHostData = {1, 2, 0, 1, 0, 2};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> sortedIndicesOutHostData = {0, 0, 0, 0, 0, 0};
    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, idxShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_BF16, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesOutHostData, idxOutShape, &sortedIndicesOutDeviceAddr, aclDataType::ACL_INT32, &sortedIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeTokenPermute第一段接口
    ret = aclnnMoeTokenPermuteGetWorkspaceSize(x, indices, numTokenOut, padMode, expandedXOut, sortedIndicesOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMoeTokenPermute第二段接口
    ret = aclnnMoeTokenPermute(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermute failed. ERROR: %d\n", ret); return ret);
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
    auto sortedIndicesSize = GetShapeSize(idxOutShape);
    std::vector<int> sortedIndicesData(sortedIndicesSize, 0);
    ret = aclrtMemcpy(sortedIndicesData.data(), sortedIndicesData.size() * sizeof(sortedIndicesData[0]), sortedIndicesOutDeviceAddr, sortedIndicesSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < sortedIndicesSize; i++) {
        LOG_PRINT("sortedIndicesData[%ld] is: %d\n", i, sortedIndicesData[i]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(indices);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(sortedIndicesOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(sortedIndicesOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

