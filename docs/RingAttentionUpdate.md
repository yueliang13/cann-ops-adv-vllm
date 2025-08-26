声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# RingAttentionUpdate

## 支持的产品型号

- 昇腾910B AI处理器。
- 昇腾910_93 AI处理器。

## 函数原型

每个算子分为[两段式接口](./common/两段式接口.md)，必须先调用“aclnnRingAttentionUpdateGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRingAttentionUpdate”接口执行计算。


- `aclnnStatus aclnnRingAttentionUpdateGetWorkspaceSize(const aclTensor *prevAttnOut, const aclTensor *prevSoftmaxMax, const aclTensor *prevSoftmaxSum, const aclTensor *curAttnOut, const aclTensor *curSoftmaxMax, const aclTensor *curSoftmaxSum, const aclTensor *actualSeqQlenOptional, char *inputLayoutOptional, const aclTensor *attnOutOut, const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnRingAttentionUpdate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能说明

- 接口功能：RingAttentionUpdate算子功能是将两次FlashAttention的输出根据其不同的softmax的max和sum更新。

- 计算公式：

$$
softmax\_max = max(prev\_softmax\_max, cur\_softmax\_max)
$$

$$
softmax\_sum = prev\_softmax\_sum * exp(prev\_softmax\_max - softmax\_max) + exp(cur\_softmax\_max - softmax\_max)
$$

$$
attn\_out = prev\_attn\_out * exp(prev\_softmax\_max - softmax\_max) / softmax\_sum + cur\_attn\_out * exp(cur\_softmax\_max - softmax\_max) / softmax\_sum
$$

## aclnnRingAttentionUpdateGetWorkspaceSize

- **参数说明：**
  - prevAttnOut（aclTensor*,计算输入）：Device侧的aclTensor，公式中的prev_attn_out，第一次FlashAttention的输出，数据类型支持FLOAT16、FLOAT、BFLOAT16，输入shape和inputLayoutOptional属性保持一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持ND。当输入数据排布inputLayoutOptional为TND时，D限制为64的倍数。
  - prevSoftmaxMax（aclTensor*,计算输入）：Device侧的aclTensor，公式中的prev_softmax_max，第一次FlashAttention的softmax的max结果，数据类型支持FLOAT，输入shape为(B,N,S,8)或(T,N,8)，最后一维8个数字相同，且需要为正数，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持ND。此处B为batch size，N为head number，S为sequence length，T为time。
  - prevSoftmaxSum（aclTensor*,计算输入）：Device侧的aclTensor，公式中的prev_softmax_sum，第一次FlashAttention的softmax的sum结果，数据类型支持FLOAT，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持ND。
  - curAttnOut（aclTensor*,计算输入）：Device侧的aclTensor，公式中的cur_attn_out，第二次FlashAttention的输出，数据类型支持FLOAT16、FLOAT、BFLOAT16，数据类型和输入shape和prevAttnOut保持一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持ND。当输入数据排布inputLayoutOptional为TND时，D限制为64的倍数。
  - curSoftmaxMax（aclTensor*,计算输入）：Device侧的aclTensor，公式中的cur_softmax_max，第二次FlashAttention的softmax的max结果，数据类型支持FLOAT，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持ND。
  - curSoftmaxSum（aclTensor*,计算输入）：Device侧的aclTensor，公式中的cur_softmax_sum，第二次FlashAttention的softmax的sum结果，数据类型支持FLOAT，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持ND。
  - actualSeqQlenOptional（aclTensor*,计算输入）：Device侧的aclTensor，从0开始的sequence length的累加，数据类型支持INT64。当数据排布inputLayoutOptional为TND时，需要传入该参数，这是一个从0开始递增至T的整数aclTensor。
  - inputLayoutOptional（char*,计算输入）：Host侧的char*常量，attn_out相关输入的数据排布。当前支持“TND”和“SBH”。
  - attnOutOut（aclTensor*,计算输出）：Device侧的aclTensor，公式中的attn_out，通过两次结果更新后的输出，数据类型支持FLOAT16、FLOAT、BFLOAT16，数据类型和输出shape和prevAttnOut保持一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持ND。
  - softmaxMaxOut（aclTensor*,计算输出）：Device侧的aclTensor，公式中的softmax_max，通过两次结果更新后的softmax的max，数据类型支持FLOAT，输出shape和prevSoftmaxMax保持一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持ND。
  - softmaxSumOut（aclTensor*,计算输出）：Device侧的aclTensor，公式中的softmax_sum，通过两次结果更新后的softmax的sum，数据类型支持FLOAT，输出shape和prevSoftmaxMax保持一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持ND。
  - workspaceSize（uint64_t*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**, 出参）：返回op执行器，包含算子计算流程。
  
- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的 prevAttnOut、prevSoftmaxMax、prevSoftmaxSum、curAttnOut、curSoftmaxMax、curSoftmaxSum、attnOutOut、softmaxMaxOut、softmaxSumOut是空指针时。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. prevAttnOut、prevSoftmaxMax、prevSoftmaxSum、curAttnOut、curSoftmaxMax、curSoftmaxSum、attnOutOut、softmaxMaxOut、softmaxSumOut数据类型不在支持的范围之内。
                                      2. prevAttnOut、prevSoftmaxMax、prevSoftmaxSum、curAttnOut、curSoftmaxMax、curSoftmaxSum、attnOutOut、softmaxMaxOut、softmaxSumOut的shape不支持。
  返回561002 (ACLNN_ERR_INNER_TILING_ERROR)：1. 当actualSeqQlenOptional有输入时，输入数据格式不在支持的范围之内。                          
  ```

## aclnnRingAttentionUpdate

- **参数说明**：
  - workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnRingAttentionUpdateGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值**：
  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明
  - 当inputLayoutOptional为“TND”时，prevAttnOut的最后一个维度需要为64的倍数。
  - 当inputLayoutOptional为“TND”时，actualSeqQlenOptional为必填。
  - 当inputLayoutOptional为“TND”时，请注意N*D的大小，限制为： (N \* D)向上对齐64的结果 \* (attention的输入数据类型的大小 \* 6 + 8) + (N \* 8)向上对齐64的结果 \* 56 <= 192 \* 1024，数据类型大小：FLOAT32为4，FLOAT16和BFLOAT16为2。若大小超过限制，会有相应拦截信息出现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](./common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ring_attention_update.h"

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
  // 1. (固定写法)device/stream初始化, 参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int64_t batchNum = 1;
  int64_t headNum = 1;
  int64_t seqSize = 2;
  int64_t headDim = 4;
  int64_t headSize = headNum * headDim;
 
  std::vector<int64_t> prevAttnOutShape = {seqSize, batchNum, headSize};
  std::vector<int64_t> prevSoftmaxMaxShape = {batchNum, headNum, seqSize, 8};
  std::vector<int64_t> prevSoftmaxSumShape = {batchNum, headNum, seqSize, 8};
  std::vector<int64_t> curAttnOutShape = {seqSize, batchNum, headSize};
  std::vector<int64_t> curSoftmaxMaxShape = {batchNum, headNum, seqSize, 8};
  std::vector<int64_t> curSoftmaxSumShape = {batchNum, headNum, seqSize, 8};
  std::vector<int64_t> actualSeqQlenOptionalShape = {batchNum, headNum};
  
  std::vector<int64_t> attnOutShape = {seqSize, batchNum, headSize};
  std::vector<int64_t> softmaxMaxShape = {batchNum, headNum, seqSize, 8};
  std::vector<int64_t> softmaxSumShape = {batchNum, headNum, seqSize, 8};

  void* prevAttnOutDeviceAddr = nullptr;
  void* prevSoftmaxMaxDeviceAddr = nullptr;
  void* prevSoftmaxSumDeviceAddr = nullptr;
  void* curAttnOutDeviceAddr = nullptr;
  void* curSoftmaxMaxDeviceAddr = nullptr;
  void* curSoftmaxSumDeviceAddr = nullptr;
  void* actualSeqQlenOptionalDeviceAddr = nullptr;

  void* attnOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;

  aclTensor* prevAttnOut = nullptr;
  aclTensor* prevSoftmaxMax = nullptr;
  aclTensor* prevSoftmaxSum = nullptr;
  aclTensor* curAttnOut = nullptr;
  aclTensor* curSoftmaxMax = nullptr;
  aclTensor* curSoftmaxSum = nullptr;
  aclTensor* actualSeqQlenOptional = nullptr;

  aclTensor* attnOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  
  std::vector<float> prevAttnOutHostData(seqSize * batchNum * headSize, 1);
  std::vector<float> prevSoftmaxMaxHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> prevSoftmaxSumHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> curAttnOutHostData(seqSize * batchNum * headSize, 1);
  std::vector<float> curSoftmaxMaxHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> curSoftmaxSumHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> actualSeqQlenOptionalHostData(batchNum * headNum, 1);

  std::vector<float> attnOutHostData(seqSize * batchNum * headSize, 1);
  std::vector<float> softmaxMaxHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> softmaxSumHostData(batchNum * headNum * seqSize * 8, 1);

  char* inputLayoutOptional = "SBH";
  // 创建prevAttnOut aclTensor
  ret = CreateAclTensor(prevAttnOutHostData, prevAttnOutShape, &prevAttnOutDeviceAddr, aclDataType::ACL_FLOAT, &prevAttnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建prevSoftmaxMax aclTensor
  ret = CreateAclTensor(prevSoftmaxMaxHostData, prevSoftmaxMaxShape, &prevSoftmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &prevSoftmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建prevSoftmaxSum aclTensor
  ret = CreateAclTensor(prevSoftmaxSumHostData, prevSoftmaxSumShape, &prevSoftmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &prevSoftmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建curAttnOut aclTensor
  ret = CreateAclTensor(curAttnOutHostData, curAttnOutShape, &curAttnOutDeviceAddr, aclDataType::ACL_FLOAT, &curAttnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建curSoftmaxMax aclTensor
  ret = CreateAclTensor(curSoftmaxMaxHostData, curSoftmaxMaxShape, &curSoftmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &curSoftmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建curSoftmaxSum aclTensor
  ret = CreateAclTensor(curSoftmaxSumHostData, curSoftmaxSumShape, &curSoftmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &curSoftmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建actualSeqQlenOptional aclTensor
  ret = CreateAclTensor(actualSeqQlenOptionalHostData, actualSeqQlenOptionalShape, &actualSeqQlenOptionalDeviceAddr, aclDataType::ACL_INT64, &actualSeqQlenOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 创建attnOut aclTensor
  ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_FLOAT, &attnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建softmaxMax aclTensor
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建softmaxSum aclTensor
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnRingAttentionUpdate第一段接口
  ret = aclnnRingAttentionUpdateGetWorkspaceSize(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, 
                                                 curAttnOut, curSoftmaxMax, curSoftmaxSum, 
                                                 actualSeqQlenOptional, inputLayoutOptional, 
                                                 attnOut, softmaxMax, softmaxSum, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnRingAttentionUpdate第二段接口
  ret = aclnnRingAttentionUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdate failed. ERROR: %d\n", ret); return ret);
  // 4. (固定写法)同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto attnOutSize = GetShapeSize(attnOutShape);
  std::vector<float> attnOutResultData(attnOutSize, 0);
  ret = aclrtMemcpy(attnOutResultData.data(), attnOutResultData.size() * sizeof(attnOutResultData[0]), attnOutDeviceAddr, attnOutSize * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < attnOutSize; i++) {
    LOG_PRINT("attnOutResultData[%ld] is: %f\n", i, attnOutResultData[i]);
  }

  auto softmaxMaxSize = GetShapeSize(softmaxMaxShape);
  std::vector<float> softmaxMaxResultData(softmaxMaxSize, 0);
  ret = aclrtMemcpy(softmaxMaxResultData.data(), softmaxMaxResultData.size() * sizeof(softmaxMaxResultData[0]), softmaxMaxDeviceAddr, softmaxMaxSize * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < softmaxMaxSize; i++) {
    LOG_PRINT("softmaxMaxResultData[%ld] is: %f\n", i, softmaxMaxResultData[i]);
  }
  
  auto softmaxSumSize = GetShapeSize(softmaxSumShape);
  std::vector<float> softmaxSumResultData(softmaxSumSize, 0);
  ret = aclrtMemcpy(softmaxSumResultData.data(), softmaxSumResultData.size() * sizeof(softmaxSumResultData[0]), softmaxSumDeviceAddr, softmaxSumSize * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < softmaxSumSize; i++) {
    LOG_PRINT("softmaxSumResultData[%ld] is: %f\n", i, softmaxSumResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(prevAttnOut);
  aclDestroyTensor(prevSoftmaxMax);
  aclDestroyTensor(prevSoftmaxSum);
  aclDestroyTensor(curAttnOut);
  aclDestroyTensor(curSoftmaxMax);
  aclDestroyTensor(curSoftmaxSum);
  aclDestroyTensor(actualSeqQlenOptional);
  aclDestroyTensor(attnOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(prevAttnOutDeviceAddr);
  aclrtFree(prevSoftmaxMaxDeviceAddr);
  aclrtFree(prevSoftmaxSumDeviceAddr);
  aclrtFree(curAttnOutDeviceAddr);
  aclrtFree(curSoftmaxMaxDeviceAddr);
  aclrtFree(curSoftmaxSumDeviceAddr);
  aclrtFree(actualSeqQlenOptionalDeviceAddr);
  aclrtFree(attnOutDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```