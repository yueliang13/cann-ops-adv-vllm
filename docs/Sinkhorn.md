声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# Sinkhorn

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 功能说明

- 算子功能：

  计算Sinkhorn距离，可以用于MoE模型中的专家路由。

- 计算公式：

  $$
  p=Sinkhorn(cost, tol)
  $$
  
  ---

  **输入**：
  cost(R, C): 2维成本矩阵
  tol: 误差

  **初始化**:
  $$
  cost = exp(cost) \\
  d0 = ones(R) \\
  d1 = ones(C) \\
  eps = 0.00000001 \\
  error = 1e9 \\
  d1\_old= d1 \\
  $$

  **重复执行**:
  $$
  d0 = \frac{1}{R * (sum(d1 * cost, 1) + eps)} \\
  d1 = \frac{1}{C * (sum(d0.unsqueeze(1) * cost, 0) + eps)} \\
  error = mean(abs(d1\_old - d1)) \\
  d1\_old = d1
  $$

  直至:
  
  $$
  error <= tol
  $$

  **输出**:
  $$
  p = d1 * cost * d0.unsqueeze(1)
  $$

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnSinkhornGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnSinkhorn”接口执行计算。

- `aclnnStatus aclnnSinkhornGetWorkspaceSize(const aclTensor *cost, const aclScalar *tol, aclTensor *p, uint64_t *workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

## aclnnSinkhornGetWorkspaceSize

- **参数说明**：

  - cost（aclTensor*，计算输入）：表示成本张量，公式中的`cost`，Device侧的aclTensor。数据类型支持BFLOAT16、FLOAT16、FLOAT。[数据格式](common/数据格式.md)支持ND，输入为二维矩阵且列数不超过4096。支持[非连续的Tensor](common/非连续的Tensor.md)。
  - tol (aclScalar*, 入参) ：表示计算Sinkhorn的误差，数据类型支持FLOAT。如果传入空指针，则tol取0.0001。
  - p（aclTensor*，计算输出）：表示最优传输张量，公式中的`p`，Device侧的aclTensor。数据类型支持BFLOAT16、FLOAT16、FLOAT。[数据格式](common/数据格式.md)支持ND。shape维度为2。不支持[非连续的Tensor](common/非连续的Tensor.md)。数据类型和shape与入参`cost`的数据类型和shape一致。
  - workspaceSize（uint64_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的cost或p是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. cost和p的数据类型不在支持的范围之内。
                                        2. cost和p无法做数据类型推导。
  ```

## aclnnSinkhorn

- **参数说明**：

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnSinkhornGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

无。

## 算子原型

```c++
REG_OP(Sinkhorn)
  .INPUT(cost, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
  .OUTPUT(p, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
  .ATTR(tol, Float, 0.0001)
  .OP_END_FACTORY_REG(Sinkhorn)
```

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sinkhorn.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，acl初始化
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
  // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> costShape = {3, 2};
  std::vector<int64_t> pShape = {3, 2};
  void* costDeviceAddr = nullptr;
  void* pDeviceAddr = nullptr;
  aclTensor* cost = nullptr;
  aclScalar* tol = nullptr;
  aclTensor* p = nullptr;
  std::vector<float> costHostData = {45, 48, 65, 68, 68, 10};
  std::vector<float> pHostData(6, 0);

  float tolValue = 0.0001;

  // 创建cost aclTensor
  ret = CreateAclTensor(costHostData, costShape, &costDeviceAddr, aclDataType::ACL_FLOAT, &cost);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建p aclTensor
  ret = CreateAclTensor(pHostData, pShape, &pDeviceAddr, aclDataType::ACL_FLOAT, &p);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建tol aclScalar
  tol = aclCreateScalar(&tolValue, aclDataType::ACL_FLOAT);
  CHECK_RET(tol != nullptr, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSinkhorn第一段接口
  ret = aclnnSinkhornGetWorkspaceSize(cost, tol, p, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSinkhornGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSinkhorn第二段接口
  ret = aclnnSinkhorn(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSinkhorn failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(pShape);
  std::vector<float> pData(size, 0);
  ret = aclrtMemcpy(pData.data(), pData.size() * sizeof(pData[0]), pDeviceAddr,
                    size * sizeof(pData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("p result[%ld] is: %e\n", i, pData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(cost);
  aclDestroyTensor(p);
  aclDestroyScalar(tol);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(costDeviceAddr);
  aclrtFree(pDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
