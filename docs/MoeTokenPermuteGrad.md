# MoeTokenPermuteGrad

## 支持的产品型号

- Atlas A2 训练系列产品

## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用 “aclnnMoeTokenPermuteGradGetWorkspaceSize” 接口获取入参并根据计算流程计算所需workspace大小以及包含了算子计算流程的执行器，再调用 “aclnnMoeTokenPermuteGrad” 接口执行计算。

* `aclnnStatus aclnnMoeTokenPermuteGradGetWorkspaceSize(const aclTensor *permutedOutputGrad, const aclTensor *sortedIndices, int64_t numTopk, bool paddedMode, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeTokenPermuteGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能说明

- **算子功能**：[aclnnMoeTokenPermute](./aclnnMoeTokenPermute.md)的反向传播计算。
- **计算公式**：

  $$
  inputGrad = permutedOutputGrad.indexSelect(0, sortedIndices)
  $$
  
  $$
  inputGrad = inputGrad.reshape(-1, topK, hiddenSize)
  $$
  
  $$
  inputGrad = inputGrad.sum(dim = 1)
  $$

## aclnnMoeTokenPermuteGradGetWorkspaceSize

- **参数说明：**：
  - permutedOutputGrad（aclTensor\*，计算输入）：Device侧的aclTensor，正向输出permutedTokens的梯度，要求为一个维度为2D的Tensor，shape为（tokens_num * topK_num，hidden_size），tokens_num为token数目，topK_num为numTopk的值，数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND。
  - sortedIndices （aclTensor\*，计算输入）：Device侧的aclTensor，shape为（tokens_num * topK_num），数据类型支持INT32，[数据格式](common/数据格式.md)要求为ND。
  - numTopk（int64\_t，计算输入）：被选中的专家个数，取值范围为numTopk <= 512。
  - paddedMode（bool，计算输入）：true表示开启paddedMode，false表示关闭paddedMode，目前仅支持false。
  - out（aclTensor\*，计算输出）：输入token的梯度，要求为一个维度为2D的Tensor，shape为（tokens_num，hidden_size），数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND。
  - workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. 输入和输出的Tensor是空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. 输入和输出的数据类型和数据格式不在支持的范围之内。
  ```

## aclnnMoeTokenPermuteGrad

- **参数说明：**
  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenPermuteGradGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

numTopk <= 512

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_grad.h"
#include <iostream>
#include <vector>

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

  int64_t num_topk = 2;
  std::vector<float> permuted_output_grad_Data = {1, 2, 3, 4};
  std::vector<int64_t> permuted_output_grad_Shape = {2, 2};
  void *permuted_output_grad_Addr = nullptr;
  aclTensor *permuted_output_grad = nullptr;

  ret = CreateAclTensor(permuted_output_grad_Data, permuted_output_grad_Shape,
                        &permuted_output_grad_Addr, aclDataType::ACL_BF16,
                        &permuted_output_grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> sortedIndicesData = {0, 1};
  std::vector<int64_t> sortedIndicesShape = {2};
  void *sortedIndicesAddr = nullptr;
  aclTensor *sortedIndices = nullptr;

  ret = CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> outData = {0, 0};
  std::vector<int64_t> outShape = {1, 2};
  void *outAddr = nullptr;
  aclTensor *out = nullptr;

  ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_BF16,
                        &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnMoeTokenPermuteGrad第一段接口
  ret = aclnnMoeTokenPermuteGradGetWorkspaceSize(permuted_output_grad, sortedIndices,
                                                 num_topk, false,
                                                 out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenPermuteGradGetWorkspaceSize failed. ERROR: %d\n",
                ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用aclnnMoeTokenPermuteGrad第二段接口
  ret = aclnnMoeTokenPermuteGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenPermuteGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(permuted_output_grad);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(permuted_output_grad_Addr);
  aclrtFree(sortedIndicesAddr);
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
