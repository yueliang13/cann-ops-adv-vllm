# MoeTokenUnpermute

## 支持的产品型号

- Atlas A2 训练系列产品

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnMoeTokenUnpermuteGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenUnpermute”接口执行计算。

* `aclnnStatus aclnnMoeTokenUnpermuteGetWorkspaceSize(const aclTensor *permutedTokens, const aclTensor *sortedIndices, const aclTensor *probsOptional, bool paddedMode, const aclIntArray *restoreShapeOptional, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);`

* `aclnnStatus aclnnMoeTokenUnpermute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);`

## 功能说明

- **算子功能：** 根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘；最后进行累加求和，并输出计算结果。

- **计算公式：** 

  - probs非None计算公式如下：
    
    $$
    T[k] = T[S[k]]
    $$
    
    $$
    T[k] = T[k] * P[i][j]
    $$

    $$
    O[i] = \sum_{k=i*topK}^{(i+1)*topK - 1 } T[k]
    $$
    
    其中$i \in {0,1,...,tokens-1}$；$j \in {0,1,...,topK-1}$；$k \in {0,1,...,tokens*topK-1}$；T表示permutedTokens；S表示sortedIndices；P表示probs；O表示out；topK表示topK\_num；tokens表示tokens_num。

  - probs为None时，此时topK\_num=1，计算公式如下：

    $$
    T[i] = T[S[i]]
    $$

    $$
    O[i] = T[i]
    $$

    其中 $i \in {0,1,...,tokens-1}$；T表示permutedTokens；S表示sortedIndices；O表示out；tokens表示tokens_num。

## aclnnMoeTokenUnpermuteGetWorkspaceSize

-   **参数说明：**
    -   permutedTokens（aclTensor*，计算输入）：输入数据。shape为（tokens_num*topK_num，hidden_size）。支持的数据类型BFLOAT16、FLOAT16、FLOAT32。[数据格式](common/数据格式.md)支持ND。支持非连续输入。

    -   sortedIndices（aclTensor*，计算输入）：表示需要计算的数据在permutedTokens中的位置。shape为（tokens_num * topK_num），要求值域大于等于0。支持的数据类型int32，[数据格式](common/数据格式.md)支持ND。支持非连续输入。

    -   probsOptional（aclTensor*，可选计算输入）：可选输入。当probs传时，topK_num等于probs的第二维；当probs不传时，topK_num=1。shape为（tokens_num，topK_num），支持的数据类型BFLOAT16、FLOAT16、FLOAT32。[数据格式](common/数据格式.md)支持ND。支持非连续输入。

    -   paddedMode（bool，计算输入）：true表示开启paddedMode，false表示关闭paddedMode，paddedMode解释见restoreShapeOptional参数。目前仅支持false。

    -   restoreShapeOptional（aclIntArray*，计算输入）：paddedMode=true时生效，否则不会对其进行操作。paddedMode=true时，out的shape将表征为restoreShapeOptional。目前仅支持nullptr。

    -   out（aclTensor*，计算输出）：输出结果。paddedMode=false时，shape为（tokens_num，hidden_size）。paddedMode=true时，shape与restoreShapeOptional保持一致。支持的数据类型BFLOAT16、FLOAT16、FLOAT32。[数据格式](common/数据格式.md)支持ND。不支持非连续输出。

    -   workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。

    -   executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
    ```
    第一段接口完成入参校验，出现以下场景时报错：
    161001(ACLNN_ERR_PARAM_NULLPTR): 1. 输入和输出的Tensor是空指针。
    161002(ACLNN_ERR_PARAM_INVALID): 1. 输入和输出的数据类型不在支持的范围内。
    ```

## aclnnMoeTokenUnpermute

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenUnpermuteGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

topK_num <= 512

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute.h"
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

  std::vector<float> permutedTokensData = {1, 2, 3, 4};
  std::vector<int64_t> permutedTokensShape = {2, 2};
  void *permutedTokensAddr = nullptr;
  aclTensor *permutedTokens = nullptr;

  ret = CreateAclTensor(permutedTokensData, permutedTokensShape,
                        &permutedTokensAddr, aclDataType::ACL_BF16,
                        &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> sortedIndicesData = {0,1};
  std::vector<int64_t> sortedIndicesShape = {2};
  void *sortedIndicesAddr = nullptr;
  aclTensor *sortedIndices = nullptr;

  ret =
      CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> probsOptionalData = {1, 1};
  std::vector<int64_t> probsOptionalShape = {1, 2};
  void *probsOptionalAddr = nullptr;
  aclTensor *probsOptional = nullptr;

  ret =
      CreateAclTensor(probsOptionalData, probsOptionalShape, &probsOptionalAddr,
                      aclDataType::ACL_BF16, &probsOptional);
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

  // 调用aclnnMoeTokenUnpermute第一段接口
  ret = aclnnMoeTokenUnpermuteGetWorkspaceSize(permutedTokens, sortedIndices,
                                               probsOptional, false, nullptr,
                                               out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteGetWorkspaceSize failed. ERROR: %d\n",
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

  // 调用aclnnMoeTokenUnpermute第二段接口
  ret = aclnnMoeTokenUnpermute(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermute failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(probsOptional);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(permutedTokensAddr);
  aclrtFree(sortedIndicesAddr);
  aclrtFree(probsOptionalAddr);
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

