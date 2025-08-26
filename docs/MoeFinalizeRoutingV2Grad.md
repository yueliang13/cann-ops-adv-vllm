# MoeFinalizeRoutingV2Grad

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品
- Atlas A3 训练系列产品/Atlas 800I A3 推理产品

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeFinalizeRoutingV2Grad”接口执行计算。

* `aclnnStatus aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *expandedRowIdx, const aclTensor *expandedXOptional, const aclTensor *scalesOptional, const aclTensor *expertIdxOptional, const aclTensor *biasOptional, int64_t dropPadMode, int64_t activeNum, int64_t expertNum, int64_t expertCapacity, const aclTensor *gradExpandedXOut, const aclTensor *gradScalesOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeFinalizeRoutingV2Grad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能说明

- **算子功能**：aclnnMoeFinalizeRoutingV2的反向传播。
- **计算公式**：
    R: batch * sequence

    H: hidden
    
    K: topk

    gradY: (R, H)

    expandedRowIdx: (R * K)

    expandedXOptional: (R * K, H) or (activeNum, H) or (expertNum, expertCapacity, H)

    scalesOptional: (R, K)

    expertIdxOptional: (R, K)

    biasOptional：(E, H)
   
    i : 0 ~ R * K - 1

    j : 0 ~ H

    (1) scalesOptional为空指针：

    $$
    gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j]
    $$

    (2) scalesOptional不为空指针, biasOptional为空指针：

    $$
    gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j] * scalesOptional[i]
    $$

    $$
    gradScalesOut[i] = sum(expandedXOptional[expandedRowIdx[i]][j] * gradY[i / K][j])
    $$

    (3) scalesOptional不为空指针, biasOptional不为空指针：
    
    $$
    gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j] * scalesOptional[i]
    $$

    $$
    gradScalesOut[i] = sum((expandedXOptional[expandedRowIdx[i]][j] + biasOptional[expertIdxOptional[i]][j]) * gradY[i / K][j])
    $$

## aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize

-   **参数说明：**
    -   gradY（aclTensor*，计算输入）：Device侧的aclTensor，表示MoeFinalizeRoutingV2正向输出y的导数，要求是一个2D的Tensor，shape为(R, H)，数据类型支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND，支持非连续输入。
    -   expandedRowIdx（aclTensor*，计算输入）：Device侧的aclTensor，表示token按照专家序排序索引，要求是一个1D的Tensor，shape为(R * K)，当scalesOptional传入空指针的时候，K必须为1，当dropPadMode是0时，取值范围是[0, R * K - 1]，且没有重复索引；当dropPadMode是1时，取值范围是[-1, expertNum * expertCapacity - 1]，且除-1外，不允许有其它重复索引，数据类型支持INT32，[数据格式](common/数据格式.md)要求为ND，支持非连续输入。
    -   expandedXOptional（aclTensor*，可选计算输入）：Device侧的aclTensor，表示根据expertIdx进行扩展过的特征，当scalesOptional非空指针时，其也不能是空指针，当dropPadMode是0时，要求是一个2D的Tensor，当activeNum大于0且小于R * K时，shape为(activeNum, H)，否则shape为(R * K, H)；当dropPadMode是1时，要求是一个3D的Tensor，shape为(expertNum, expertCapacity, H)，数据类型同gradY，支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND，支持非连续输入。
    -   scalesOptional（aclTensor*，可选计算输入）：Device侧的aclTensor，表示对特征进行的缩放，要求是一个2D的Tensor，shape为(R, K)，数据类型同gradY，支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND，支持非连续输入。
    -   expertIdxOptional（aclTensor*，可选计算输入）：Device侧的aclTensor，表示每一个特征对应的处理专家索引，当biasOptional非空指针时，其也不能是空指针，要求是一个2D的Tensor，shape为(R, K)，取值范围是[0, E - 1], E >= 1, 允许有重复索引，数据类型同expandedRowIdx，支持INT32，[数据格式](common/数据格式.md)要求为ND，支持非连续输入。
    -   biasOptional（aclTensor*，可选计算输入）：Device侧的aclTensor，表示对特征进行的偏移，要求是一个2D的Tensor，shape为(E, H)，数据类型同gradY，支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND，支持非连续输入。
    -   dropPadMode（int64_t, 计算输入）：int64数据类型，表示使用不同的场景，取值为0和1，0代表dropless场景，不校验expertNum和expertCapacity；1代表drop场景，需要校验expertNum和expertCapacity，对于每个专家处理的超过和不足expertCapacity的值会做相应的处理。
    -   activeNum（int64_t, 计算输入）：int64数据类型，表示gradExpandedXOut最大输出行数，当dropPadMode是0时，只有当activeNum大于0且小于R * K时，该参数才生效；当dropPadMode是1时，该参数不生效。
    -   expertNum（int64_t, 计算输入）：int64数据类型，表示专家数，当dropPadMode是0时，该参数不生效；当dropPadMode是1时，当biasOptional非空指针时，expertNum必须等于E，当biasOptional是空指针时，expertNum必须大于0，否则会报错。
    -   expertCapacity（int64_t, 计算输入）：int64数据类型，表示每个专家能够处理的行数，当dropPadMode是0时，该参数不生效；当dropPadMode是1时，expertCapacity必须大于0，否则会报错。
    -   gradExpandedXOut（aclTensor*，计算输出）：Device侧的aclTensor，MoeFinalizeRoutingV2正向输入expandedX的导数，当dropPadMode是0时，要求是一个2D的Tensor，当activeNum大于0且小于R * K时，shape为(activeNum, H)，否则shape为(R * K, H)；当dropPadMode是1时，要求是一个3D的Tensor，shape为(expertNum, expertCapacity, H)，数据类型同gradY，支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND，不支持非连续输出。
    -   gradScalesOut（aclTensor*，计算输出）：Device侧的aclTensor，MoeFinalizeRoutingV2正向输入scales的导数，当scalesOptional不是空指针时，此输出才有意义，要求是一个2D的Tensor，shape为(R, K)，数据类型同gradY，支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND，不支持非连续输出。
    -   workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
    -   executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
    ```
    第一段接口完成入参校验，出现以下场景时报错:
    161001(ACLNN_ERR_PARAM_NULLPTR): 1. 必选输入和输出的Tensor是空指针。
    161002(ACLNN_ERR_PARAM_INVALID): 1. 输入和输出的数据类型和格式不在支持的范围内
    561002(ACLNN_ERR_INNER_TILING_ERROR): 1. 输入和输出的shape、取值不满足参数说明中的要求
    ```

## aclnnMoeFinalizeRoutingV2Grad

-   **参数说明：**
    -   workspace（void*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize获取。
    -   executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream,入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

## 约束说明
无

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_finalize_routing_v2_grad.h"
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
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                         size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
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
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> gradYShape = {2, 2};
  std::vector<int64_t> expandedRowIdxShape = {4};
  std::vector<int64_t> expandedXShape = {4, 2};
  std::vector<int64_t> scalesShape = {2, 2};
  std::vector<int64_t> expertIdxShape = {2, 2};
  std::vector<int64_t> biasShape = {2, 2};
  std::vector<int64_t> gradExpandedXShape = {4, 2};
  std::vector<int64_t> gradScalesShape = {2, 2};
  void* gradYDeviceAddr = nullptr;
  void* expandedRowIdxDeviceAddr = nullptr;
  void* expandedXDeviceAddr = nullptr;
  void* scalesDeviceAddr = nullptr;
  void* expertIdxDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* gradExpandedXDeviceAddr = nullptr;
  void* gradScalesDeviceAddr = nullptr;

  aclTensor* gradY = nullptr;
  aclTensor* expandedRowIdx = nullptr;
  aclTensor* expandedX = nullptr;
  aclTensor* scales = nullptr;
  aclTensor* expertIdx = nullptr;
  aclTensor* bias = nullptr;
  int64_t dropPadMode = 0;
  int64_t activeNum = 0;
  int64_t expertNum = 0;
  int64_t expertCapacity = 0;
  aclTensor* gradExpandedX = nullptr;
  aclTensor* gradScales = nullptr;

  std::vector<float> gradYHostData = {0.3816, 0.3939, 0.8474, 0.1652};
  std::vector<int> expandedRowIdxHostData = {1, 3, 0, 2};
  std::vector<float> expandedXHostData = {0.6049, 0.3315, 0.4954, 0.3284, 0.7060, 0.4359, 0.6514, 0.9476};
  std::vector<float> scalesHostData = {0.4708, 0.0656, 0.9652, 0.9512};
  std::vector<int> expertIdxHostData = {0, 1, 0, 1};
  std::vector<float> biasHostData = {0.6452, 0.1981, 0.4159, 0.9575};
  std::vector<float> gradExpandedXHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> gradScalesHostData = {0, 0, 0, 0};

  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxDeviceAddr, aclDataType::ACL_INT32,
                        &expandedRowIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(expandedXHostData, expandedXShape, &expandedXDeviceAddr, aclDataType::ACL_FLOAT, &expandedX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(expertIdxHostData, expertIdxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradExpandedXHostData, gradExpandedXShape, &gradExpandedXDeviceAddr, aclDataType::ACL_FLOAT,
                        &gradExpandedX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradScalesHostData, gradScalesShape, &gradScalesDeviceAddr, aclDataType::ACL_FLOAT, &gradScales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);  

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnMoeFinalizeRoutingV2Grad第一段接口
  ret = aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(gradY, expandedRowIdx, expandedX, scales, expertIdx, bias,
                                                      dropPadMode, activeNum, expertNum, expertCapacity, gradExpandedX,gradScales, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnMoeFinalizeRoutingV2Grad第二段接口
  ret = aclnnMoeFinalizeRoutingV2Grad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2Grad failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  LOG_PRINT("gradExpandedX result is: \n");
  PrintOutResult(gradExpandedXShape, &gradExpandedXDeviceAddr);
  LOG_PRINT("gradScales result is: \n");
  PrintOutResult(gradScalesShape, &gradScalesDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradY);
  aclDestroyTensor(expandedRowIdx);
  aclDestroyTensor(expandedX);
  aclDestroyTensor(scales);
  aclDestroyTensor(expertIdx);
  aclDestroyTensor(bias);
  aclDestroyTensor(gradExpandedX);
  aclDestroyTensor(gradScales);

  // 7. 释放device资源
  aclrtFree(gradYDeviceAddr);
  aclrtFree(expandedRowIdxDeviceAddr);
  aclrtFree(expandedXDeviceAddr);
  aclrtFree(scalesDeviceAddr);
  aclrtFree(expertIdxDeviceAddr);
  aclrtFree(biasDeviceAddr);
  aclrtFree(gradExpandedXDeviceAddr);
  aclrtFree(gradScalesDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```