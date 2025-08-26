# aclnnGroupedMatmulAdd

## 支持的产品型号

- 昇腾910B AI处理器
- 昇腾910_93 AI处理器

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnGroupedMatmulAddGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnGroupedMatmulAdd”接口执行计算。

* `aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *groupList, aclTensor *yRef, bool transposeX, bool transposeWeight, int64_t groupType, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnGroupedMatmulAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能说明

- 算子功能：实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。基本功能为矩阵乘，如$y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i]+y_i[m_i,n_i], i=1...g$，其中g为分组个数，$m_i/k_i/n_i$为对应shape。输入输出数据类型均为aclTensor，K轴分组。

  - k轴分组：$k_i$各不相同，但$m_i/n_i$每组相同。
- 计算公式：

  $$
  y_i=x_i\times weight_i + y_i
  $$

## aclnnGroupedMatmulAddGetWorkspaceSize

- **参数说明：**

  - x（aclTensor\*，计算输入）：表示输入，Device侧的aclTensor类型，公式中的输入x，x必须转置，[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)，不支持空Tensor，数据类型支持FLOAT16、BFLOAT16。
  - weight（aclTensor\*，计算输入）：表示权重，Device侧的aclTensor类型，公式中的weight，weight不支持转置，[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)，不支持空Tensor，数据类型支持FLOAT16、BFLOAT16。
  - groupList（aclTensor\*，计算输入）：表示输入K轴方向的matmul大小分布的cumsum结果（累积和），必须为非负单调非递减数列，Device侧的aclTensor类型，数据类型支持INT64，[数据格式](common/数据格式.md)支持ND。
  - y（aclTensor\*，计算输入）：表示原地累加的输出矩阵，Device侧的aclTensor类型，公式中的y，[数据格式](common/数据格式.md)支持ND，数据类型支持FLOAT32。
  - yRef（aclTensor*，计算输出）：表示原地累加的输出矩阵，Device侧的aclTensorList，公式中的输出y，[数据格式](common/数据格式.md)支持ND，数据类型支持FLOAT32。
  - transposeX（bool，计算输入）：表示x矩阵是否转置，Host侧的布尔值，当前仅支持True。
  - transposeWeight（bool，计算输入）：表示weight矩阵是否转置，Host侧的布尔值，当前仅支持False。
  - groupType(int64\_t，计算输入)：表示分组类型，Host侧的整型，当前仅支持2（K轴分组）。
  - workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## aclnnGroupedMatmulAdd

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulAddGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

    ```
    第一段接口完成入参校验，出现以下场景时报错：
    返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的x、weight、groupList、y是空指针。
    返回161002 (ACLNN_ERR_PARAM_INVALID): 1. x、weight、groupList、y的数据类型或数据格式不在支持的范围之内。
                                         2. x与weight的数据类型不一致。
                                         3. x、weight、y的shape不满足矩阵乘限制要求。
    ```

## 约束说明

- x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32的最大值2147483647。
- 支持的输入类型为：
  - x为FLOAT16、weight为FLOAT16、y为FLOAT32。
  - x为BFLOAT16、weight为BFLOAT16、y为FLOAT32。

## 调用示例

- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_add.h"

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
int CreateAclTensor_New(const std::vector<int64_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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

template <typename T>
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
  std::vector<T> hostData(size, 0);
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


int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr,
                        aclDataType dataType, aclTensorList** tensor) {
  int size = shapes.size();
  aclTensor* tensors[size];
  for (int i = 0; i < size; i++) {
    int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, tensors + i);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  *tensor = aclCreateTensorList(tensors, size);
  return ACL_SUCCESS;
}


int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> xShape = {512, 256};
  std::vector<int64_t> weightShape= {512, 256};
  std::vector<int64_t> yShape = {512, 256};
  std::vector<int64_t> groupListShape = {2};
  std::vector<int64_t> groupListData = {256, 512};
  void* xDeviceAddr;
  void* weightDeviceAddr;
  void* yDeviceAddr;

  void* groupListDeviceAddr;
  aclTensor* x = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* groupedList = nullptr;
  aclTensor* y = nullptr;
  aclTensor* yRef = nullptr;

  bool transpose_x = true;
  bool transpose_weight = false;
  int group_type = 2;

  // 创建x aclTensorList
  ret = CreateAclTensor<uint16_t>(xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight aclTensorList
  ret = CreateAclTensor<uint16_t>(weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建y aclTensorList
  ret = CreateAclTensor<float>(yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建group_list aclTensor
  ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  yRef = y;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. 调用CANN算子库API
  // 调用aclnnGroupedMatmulAdd第一段接口
  ret = aclnnGroupedMatmulAddGetWorkspaceSize(x, weight, groupedList, yRef, transpose_x, transpose_weight, group_type, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnGroupedMatmulAdd第二段接口
  ret = aclnnGroupedMatmulAdd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmul failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(yShape);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t j = 0; j < size; j++) {
      LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
  }


  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(weight);
  aclDestroyTensor(y);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(yDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
