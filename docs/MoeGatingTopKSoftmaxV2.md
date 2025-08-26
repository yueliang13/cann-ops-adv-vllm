# aclnnMoeGatingTopKSoftmaxV2

## 支持的产品型号

- 昇腾910B AI处理器

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeGatingTopKSoftmaxV2”接口执行计算。

* `aclnnStatus aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize(const aclTensor *x, const aclTensor *finishedOptional, int64_t k, int64_t renorm, bool outputSoftmaxResultFlag, aclTensor *yOut, aclTensor *expertIdxOut, aclTensor *softmaxOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeGatingTopKSoftmaxV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能说明

-   算子功能：MoE计算中，如果renorm=0，先对x的输出做Softmax计算，再取topk操作；如果renorm=1，先对x的输出做topk操作，再进行Softmax操作。其中yOut为softmax的topk结果；expertIdxOut为topk的indices结果即对应的专家序号；如果对应的行finished为True，则expert序号直接填num\_expert值（即x的最后一个轴大小）。
-   计算公式：
1. renorm = 0,
    $$
    softmaxOut=softmax(x,axis=-1)
    $$

    $$
    yOut,expertIdxOut=topK(softmaxOut,k=k)
    $$
2. renorm = 1
    $$
    topkOut,expertIdxOut=topK(x, k=k)
    $$

    $$
    yOut = softmax(topkOut,axis=-1)
    $$

## aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize

-   **参数说明：**
    -   x（aclTensor\*，计算输入）：待计算的输入，要求是一个2D/3D的Tensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND，支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   finishedOptional（aclTensor\*，可选计算输入）：要求是一个1D/2D的Tensor，数据类型支持bool，shape为x\_shape\[:-1\]，[数据格式](common/数据格式.md)要求为ND，支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   k（int64\_t，计算输入）：topk的k值，大小为0 < k <= x的-1轴大小，且k不大于1024。
    -   renorm (int64\_t, 计算输入)：renorm标记，取值0和1。0表示先计算Softmax，再计算TopK；1表示先计算TopK，再计算Softmax。
    -   outputSoftmaxResultFlag (bool, 计算输入)：表示是否输出softmax的结果，取值true和false。当renorm=0时，true表示输出Softmax的结果，false表示不输出；当renorm=1时，该参数不生效，不输出Softmax的结果。
    -   yOut（aclTensor\*，计算输出）：对x做softmax后取的topk值，要求是一个2D/3D的Tensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据类型与x需要保持一致，其非-1轴要求与x的对应轴大小一致，其-1轴要求其大小同k值，[数据格式](common/数据格式.md)要求为ND，不支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   expertIdxOut（aclTensor\*，计算输出）：对x做softmax后取topk值的索引，即专家的序号，shape要求与yOut一致，数据类型支持int32，[数据格式](common/数据格式.md)要求为ND，不支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   softmaxOut（aclTensor\*，可选输出）：计算过程中Softmax的结果（见示例），shape要求与x一致，数据类型支持FLOAT32，[数据格式](common/数据格式.md)要求为ND，不支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   workspaceSize（uint64\_t\*，出参）：Device侧的整型，返回需要在Device侧申请的workspace大小。
    -   executor（aclOpExecutor\*\*，出参）：Device侧的aclOpExecutor，返回op执行器，包含了算子计算流程。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
    ```
    第一段接口完成入参校验，出现以下场景时报错:
    161001(ACLNN_ERR_PARAM_NULLPTR): 1. 传入的x是空指针。
    161002(ACLNN_ERR_PARAM_INVALID): 1. x、yOut、expertIdxOut的数据类型不在支持的范围内。
    561002(ACLNN_ERR_INNER_TILING_ERROR): 1. x的shape维度不为2或3。
                                          2. x与finishedOptional的shape不匹配。
                                          3. k的值小于等于0或大于x-1的轴的大小。
                                          4. k的值大于1024。
                                          5. renorm的值不是0或1。
                                          6. softmaxOut的数据类型不在支持的范围内。
    ```

## aclnnMoeGatingTopKSoftmaxV2

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream, 入参）: 指定执行任务的AscendCL Stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

k的值不大于1024。
renorm的值只支持0和1。
x和finishedOptional的每一维大小应不大于int32的最大值2147483647。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k_softmax_v2.h"
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
  auto  ret = aclInit(nullptr);
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
  // 1. （固定写法）device/stream初始化, 参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> inputShape = {3, 4};
  std::vector<int64_t> outShape = {3, 2};
  std::vector<int64_t> expertIdOutShape = {3, 2};
  std::vector<int64_t> softmaxOutShape = {3, 4};

  void* inputAddr = nullptr;
  void* outAddr = nullptr;
  void* expertIdOutAddr = nullptr;
  void* softmaxOutAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* out = nullptr;
  aclTensor* expertIdOut = nullptr;
  aclTensor* softmaxOut = nullptr;

  std::vector<float> inputHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
  std::vector<float> outHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1};
  std::vector<int32_t> expertIdOutHostData = {1, 1, 1, 1, 1, 1};
  std::vector<int32_t> softmaxOutHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  // 创建expandedPermutedRows aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &inputAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建expertForSourceRow aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建expandedSrcToDstRow aclTensor
  ret = CreateAclTensor(expertIdOutHostData, expertIdOutShape, &expertIdOutAddr, aclDataType::ACL_INT32, &expertIdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建softmaxOut aclTensor
  ret = CreateAclTensor(softmaxOutHostData, softmaxOutShape, &softmaxOutAddr, aclDataType::ACL_FLOAT, &softmaxOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3.调用CANN算子库API，需要修改为具体的算子接口
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnMoeGatingTopKSoftmaxV2第一段接口
  ret = aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize(input, nullptr, 2, 0, true, out, expertIdOut, softmaxOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnMoeGatingTopKSoftmaxV2第二段接口
  ret = aclnnMoeGatingTopKSoftmaxV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKSoftmaxV2 failed. ERROR: %d\n", ret); return ret);

  // 4.（ 固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(input);
  aclDestroyTensor(out);
  aclDestroyTensor(expertIdOut);
  aclDestroyTensor(softmaxOut);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(inputAddr);
  aclrtFree(outAddr);
  aclrtFree(expertIdOutAddr);
  aclrtFree(softmaxOutAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```