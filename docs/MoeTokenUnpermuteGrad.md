声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# MoeTokenUnpermuteGrad

## 支持的产品型号

Atlas A2 训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

- **算子功能**：MoeTokenUnpermute的反向传播。
- **计算公式**：

  - probs非None：

    $$
    unpermutedTokens[i] = permutedTokens[sortedIndices[i]]
    $$

    $$
    unpermutedTokens = unpermutedTokens.reshape(-1, topK, hiddenSize)
    $$

    $$
    unpermutedTokens = unpermutedTokensGrad.unsqueeze(1) * unpermutedTokens
    $$

    $$
    probsGrad = \sum_{k=0}^{K}(unpermutedTokens_{i,j,k})
    $$

    $$
    permutedTokensGrad[sortedIndices[i]] = ((unpermutedTokensGrad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, hiddensize))[i]
    $$

  - probs为None：

    $$
    permutedTokensGrad[sortedIndices[i]] = unpermutedOutputGrad[i]
    $$

## aclnnMoeTokenUnpermuteGradGetWorkspaceSize

-   **参数说明：**
    -   permutedTokens（aclTensor\*，计算输入）：Device侧的aclTensor，输入token，要求为一个维度为2D的Tensor，shape为（tokens_num \* topK_num，hidden_size），其中topK_num <= 512，数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND。支持非连续输入。
    -   unpermutedTokensGrad（aclTensor\*，计算输入）：Device侧的aclTensor，正向输出unpermutedTokens的梯度，要求为一个维度为2D的Tensor，shape为（tokens_num，hidden_size），数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND。支持非连续输入。
    -   sortedIndices（aclTensor\*，计算输入）：Device侧的aclTensor，要求shape为一个1D的（tokens_num \* topK_num，），数据类型支持INT32，[数据格式](common/数据格式.md)要求为ND。索引取值范围[0，tokens_num \* topK_num - 1]。支持非连续输入。
    -   probsOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选输入，要求shape为一个2D的（tokens_num，topK_num），数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND。当probs传时，topK_num等于probs第2维；当probs不传时，topK_num=1。支持非连续输入。
    -   paddedMode（bool, 计算输入）：true表示开启paddedMode，false表示关闭paddedMode，paddedMode解释见restoreShape参数。目前仅支持false。
    -   restoreShape（aclIntArray\*，计算输入）：当paddedMode为true后生效，否则不会对其进行操作。当paddedMode为true以后，此为unpermutedTokens的shape。当前仅支持nullptr。
    -   permutedTokensGradOut（aclTensor\*，计算输出）：输入permutedTokens的梯度，要求是一个2D的Tensor，shape为（tokens_num \* topK_num，hidden_size）。数据类型同permutedTokens，支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND。不支持非连续输出。
    -   probsGradOut（aclTensor\*，计算输出）：可选输出，输入probs的梯度，要求是一个2D的Tensor，shape为（tokens_num，topK_num）。数据类型同probsOptional，支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND。不支持非连续输出。
    -   workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
    -   executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
    ```
    第一段接口完成入参校验，出现以下场景时报错：
    161001(ACLNN_ERR_PARAM_NULLPTR): 1. 输入和输出的Tensor是空指针。
    161002(ACLNN_ERR_PARAM_INVALID): 1. 输入和输出的数据类型不在支持的范围内。
    ```

## aclnnMoeTokenUnpermuteGrad

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenUnpermuteGradGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream,入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

## 约束说明

topK_num <= 512

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_grad.h"
#include <iostream>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> permutedTokensShape = {3, 2};
  std::vector<int64_t> unpermutedTokensGradShape = {1, 2};
  std::vector<int64_t> probsShape = {1, 3};
  std::vector<int64_t> sortedIndicesShape = {3};
  std::vector<int64_t> permutedTokensGradShape = {3, 2};
  std::vector<int64_t> probsGradShape = {1, 3};
  void* permutedTokensDeviceAddr = nullptr;
  void* unpermutedTokensGradDeviceAddr = nullptr;
  void* probsDeviceAddr = nullptr;
  void* sortedIndicesDeviceAddr = nullptr;
  void* permutedTokensGradDeviceAddr = nullptr;
  void* probsGradDeviceAddr = nullptr;

  aclTensor* permutedTokens = nullptr;
  aclTensor* unpermutedTokensGrad = nullptr;
  aclTensor* probs = nullptr;
  aclTensor* sortedIndices = nullptr;
  bool paddedMode = false;
  aclTensor *permutedTokensGrad = nullptr;
  aclTensor *probsGrad = nullptr;

  std::vector<float> permutedTokensHostData = {1, 1, 1, 1, 1, 1};
  std::vector<float> unpermutedTokensGradHostData = {1, 1};
  std::vector<float> probsHostData = {1, 1, 1};
  std::vector<int> sortedIndicesHostData = {0, 1, 2};
  std::vector<float> permutedTokensGradHostData = {0, 0, 0, 0, 0, 0};
  std::vector<float> probsGradHostData = {0, 0, 0};

  ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape,
                        &permutedTokensDeviceAddr, aclDataType::ACL_BF16,
                        &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(unpermutedTokensGradHostData, unpermutedTokensGradShape, &unpermutedTokensGradDeviceAddr,
                      aclDataType::ACL_BF16, &unpermutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr,
                      aclDataType::ACL_BF16, &probs);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sortedIndicesHostData, sortedIndicesShape, &sortedIndicesDeviceAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(permutedTokensGradHostData, permutedTokensGradShape, &permutedTokensGradDeviceAddr, aclDataType::ACL_BF16,
                        &permutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsGradHostData, probsGradShape, &probsGradDeviceAddr, aclDataType::ACL_BF16,
                        &probsGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnMoeTokenUnpermuteGrad第一段接口
  ret = aclnnMoeTokenUnpermuteGradGetWorkspaceSize(permutedTokens, unpermutedTokensGrad, sortedIndices, probs, paddedMode, nullptr,
                                               permutedTokensGrad, probsGrad, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteGradGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用aclnnMoeTokenUnpermuteGrad第二段接口
  ret = aclnnMoeTokenUnpermuteGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(permutedTokensGradShape, &permutedTokensGradDeviceAddr);
  PrintOutResult(probsGradShape, &probsGradDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(unpermutedTokensGrad);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(probs);
  aclDestroyTensor(permutedTokensGrad);
  aclDestroyTensor(probsGrad);

  // 7. 释放device资源
  aclrtFree(permutedTokensDeviceAddr);
  aclrtFree(unpermutedTokensGradDeviceAddr);
  aclrtFree(probsDeviceAddr);
  aclrtFree(sortedIndicesDeviceAddr);
  aclrtFree(permutedTokensGradDeviceAddr);
  aclrtFree(probsGradDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```